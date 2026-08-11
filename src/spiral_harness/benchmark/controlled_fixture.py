"""Synthetic controlled-fault fixture for exercising the M0 evidence pipeline.

This is deliberately not a real benchmark and its scores make no claim about
model or harness quality.  The executor recognizes two fixture-owned prompts
and deterministically exposes a known exact-match normalization fault.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spiral_harness.core import ArtifactRef, canonical_sha256, sha256_bytes
from spiral_harness.verification import TrialObservation

FIXTURE_KIND = "synthetic-controlled-fault-v1"
FIXTURE_SEED = 1729
EXPLORATION_SEED = 2718
NORMALIZATION_SLICE = "synthetic-normalization-fault"
PROTECTED_CANONICAL_SLICE = "protected-canonical"

_PROMPT_HEADER = (
    "SYNTHETIC CONTROLLED-FAULT FIXTURE — NOT A REAL BENCHMARK.\n"
    "Classify whether the observed text matches the reference text.\n"
)
SEED_PROMPT = (
    _PROMPT_HEADER
    + "Comparison rule: compare exact strings; do not normalize case or surrounding whitespace.\n"
    + "Return MATCH for equal strings and DIFFERENT otherwise.\n"
)
CANDIDATE_PROMPT = (
    _PROMPT_HEADER
    + "Comparison rule: strip surrounding whitespace and casefold both strings before comparing.\n"
    + "Return MATCH for equal strings and DIFFERENT otherwise.\n"
)


class BenchmarkTask(BaseModel):
    """One frozen task in the synthetic controlled-fault suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fixture_kind: Literal["synthetic-controlled-fault-v1"] = FIXTURE_KIND
    task_id: str
    reference_text: str
    observed_text: str
    expected_label: Literal["MATCH", "DIFFERENT"]
    slice_tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("task_id")
    @classmethod
    def _non_empty_task_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task_id must not be empty")
        return value

    @field_validator("reference_text", "observed_text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value:
            raise ValueError("fixture text must not be empty")
        return value

    @field_validator("slice_tags")
    @classmethod
    def _canonical_slice_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(item.strip() for item in value))
        if any(not item for item in normalized):
            raise ValueError("slice tags must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("slice tags must be unique")
        return normalized


class DeterministicExecution(BaseModel):
    """Auditable trace emitted beside a trusted ``TrialObservation``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation: TrialObservation
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: Literal["exact", "strip-and-casefold"]
    compared_reference: str
    compared_observed: str
    predicted_label: Literal["MATCH", "DIFFERENT"]


def _task(
    task_id: str,
    reference: str,
    observed: str,
    *,
    protected: bool = False,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        reference_text=reference,
        observed_text=observed,
        expected_label="MATCH",
        slice_tags=(PROTECTED_CANONICAL_SLICE if protected else NORMALIZATION_SLICE,),
    )


# Exploration is used only to diagnose the known fault.  Its IDs and strings
# do not occur in the later gate roster.
EXPLORATION_TASKS: tuple[BenchmarkTask, ...] = (
    _task("synthetic-explore-normalize-00", "Mike", " mike "),
    _task("synthetic-explore-normalize-01", "NOVEMBER", "november"),
    _task("synthetic-explore-normalize-02", "Oscar-15", " OSCAR-15"),
    _task("synthetic-explore-normalize-03", "papa", "PAPA "),
    _task("synthetic-explore-canonical-00", "Quebec Canonical", "Quebec Canonical", protected=True),
    _task("synthetic-explore-canonical-01", "17/romeo", "17/romeo", protected=True),
)

# Twelve independent gate probes fail under exact matching and pass after the
# single prompt repair.  Four already-canonical gate probes form a protected
# non-regression slice.
GATE_TASKS: tuple[BenchmarkTask, ...] = (
    _task("synthetic-normalize-00", "Alpha", " alpha "),
    _task("synthetic-normalize-01", "BRAVO", "bravo"),
    _task("synthetic-normalize-02", "Charlie", "  Charlie"),
    _task("synthetic-normalize-03", "delta", "DELTA "),
    _task("synthetic-normalize-04", "Echo 5", " echo 5 "),
    _task("synthetic-normalize-05", "Foxtrot", "FOXTROT"),
    _task("synthetic-normalize-06", "Golf-7", "  golf-7  "),
    _task("synthetic-normalize-07", "Hotel", "hotel "),
    _task("synthetic-normalize-08", "India.Nine", " INDIA.NINE"),
    _task("synthetic-normalize-09", "Juliett", "juliett"),
    _task("synthetic-normalize-10", "Kilo_11", " KILO_11 "),
    _task("synthetic-normalize-11", "Lima", "LIMA "),
    _task("synthetic-canonical-00", "Already Correct", "Already Correct", protected=True),
    _task("synthetic-canonical-01", "42", "42", protected=True),
    _task("synthetic-canonical-02", "punctuation!?", "punctuation!?", protected=True),
    _task("synthetic-canonical-03", "路径/Value", "路径/Value", protected=True),
)

# Readable compatibility name: all controlled demo score claims use this gate
# roster, never the exploration roster.
CONTROLLED_TASKS = GATE_TASKS


class DeterministicExecutor:
    """Execute only the two sealed prompts owned by this synthetic fixture."""

    version = "synthetic-match-executor-v1"
    tokens_per_trial = 32
    latency_ms_per_trial = 1.0
    tool_calls_per_trial = 0

    @staticmethod
    def _policy(prompt: str) -> Literal["exact", "strip-and-casefold"]:
        if prompt == SEED_PROMPT:
            return "exact"
        if prompt == CANDIDATE_PROMPT:
            return "strip-and-casefold"
        raise ValueError("the controlled executor accepts only its two sealed synthetic prompts")

    def execute(
        self,
        task: BenchmarkTask,
        *,
        prompt_bytes: bytes,
        prompt_ref: ArtifactRef,
        harness_id: str,
        seed: int,
    ) -> DeterministicExecution:
        """Return a deterministic trace whose context fingerprint excludes the harness arm."""

        if len(prompt_bytes) != prompt_ref.size or sha256_bytes(prompt_bytes) != prompt_ref.sha256:
            raise ValueError("prompt bytes do not match their artifact reference")
        prompt = prompt_bytes.decode("utf-8")
        policy = self._policy(prompt)
        if policy == "strip-and-casefold":
            compared_reference = task.reference_text.strip().casefold()
            compared_observed = task.observed_text.strip().casefold()
        else:
            compared_reference = task.reference_text
            compared_observed = task.observed_text

        predicted_label = "MATCH" if compared_reference == compared_observed else "DIFFERENT"
        # This identifies the paired execution context, not the treatment arm:
        # prompt content, policy, and harness ID are intentionally absent.
        execution_fingerprint = "paired-context-sha256:" + canonical_sha256(
            {
                "executor_version": self.version,
                "fixture_kind": task.fixture_kind,
                "seed": seed,
                "task": task.model_dump(mode="json"),
                "resources": {
                    "latency_ms": self.latency_ms_per_trial,
                    "tokens": self.tokens_per_trial,
                    "tool_calls": self.tool_calls_per_trial,
                },
            }
        )
        observation = TrialObservation(
            task_id=task.task_id,
            seed=seed,
            harness_id=harness_id,
            score=1.0 if predicted_label == task.expected_label else 0.0,
            slice_tags=task.slice_tags,
            tokens=self.tokens_per_trial,
            latency_ms=self.latency_ms_per_trial,
            tool_calls=self.tool_calls_per_trial,
            execution_fingerprint=execution_fingerprint,
        )
        return DeterministicExecution(
            observation=observation,
            prompt_sha256=prompt_ref.sha256,
            policy=policy,
            compared_reference=compared_reference,
            compared_observed=compared_observed,
            predicted_label=predicted_label,
        )

    def execute_suite(
        self,
        tasks: tuple[BenchmarkTask, ...],
        *,
        prompt_bytes: bytes,
        prompt_ref: ArtifactRef,
        harness_id: str,
        seed: int,
    ) -> tuple[DeterministicExecution, ...]:
        return tuple(
            self.execute(
                task,
                prompt_bytes=prompt_bytes,
                prompt_ref=prompt_ref,
                harness_id=harness_id,
                seed=seed,
            )
            for task in tasks
        )


__all__ = [
    "CANDIDATE_PROMPT",
    "CONTROLLED_TASKS",
    "EXPLORATION_SEED",
    "EXPLORATION_TASKS",
    "FIXTURE_KIND",
    "FIXTURE_SEED",
    "GATE_TASKS",
    "NORMALIZATION_SLICE",
    "PROTECTED_CANONICAL_SLICE",
    "SEED_PROMPT",
    "BenchmarkTask",
    "DeterministicExecution",
    "DeterministicExecutor",
]
