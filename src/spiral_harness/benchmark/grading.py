"""In-process trusted grading boundary for captured candidate output.

The candidate-facing trace contains output and runner-captured telemetry, but no
score, reference answer, or grader action.  ``FixedOutputGrader`` owns the
sealed answer and is the only component that can create a ``TrialObservation``.

``TrustedCaptureService`` is a capability held by the trusted parent process.
It authenticates captured telemetry with an ephemeral HMAC key, while the
grader receives only a verification capability for that key.  Creating another
capture service cannot mint evidence accepted by an existing grader.  This is
an in-process capability boundary, not OS process isolation: code executing in
the trusted interpreter can inspect memory and must itself be trusted.  A later
worker protocol must keep the signing capability out of the candidate process.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.verification.models import (
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
    TrialStatus,
)

MECHANISM_CHECK_NAMES = (
    "mutation_applied",
    "target_component",
    "activation",
    "adherence",
    "non_placebo",
)

_CAPTURE_ATTESTATION_DOMAIN = b"spiral-harness:capture-attestation:v1\x00"
_CAPTURE_ATTESTOR_ID_DOMAIN = b"spiral-harness:capture-attestor-id:v1\x00"
_GRADER_IMPLEMENTATION_FINGERPRINT = "spiral-harness.fixed-output-grader:hmac:v2"


class TrustedGradingError(ValueError):
    """Raised before a candidate-controlled value can become a score."""


class GradingTask(ImmutableModel):
    """One sealed grading task; ``expected_output`` never enters a candidate trace."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    input_text: NonEmptyStr
    expected_output: str
    slice_tags: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]

    @field_validator("slice_tags")
    @classmethod
    def canonicalize_slice_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("slice_tags must not contain duplicates")
        return ordered

    @property
    def fingerprint(self) -> str:
        """Runner-visible identity over public input, never the low-entropy answer."""

        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "task_id": self.task_id,
                "input_text": self.input_text,
            }
        )

    @property
    def sealed_fingerprint(self) -> str:
        """Full trusted identity, including the answer and protected slice tags."""

        return canonical_sha256(self)


class SealedTaskRoster(ImmutableModel):
    """Canonical full task set owned by the verifier, including sealed fields."""

    schema_version: Literal["1"] = "1"
    tasks: Annotated[tuple[GradingTask, ...], Field(min_length=1)]

    @field_validator("tasks")
    @classmethod
    def canonicalize_and_validate_tasks(
        cls,
        tasks: tuple[GradingTask, ...],
    ) -> tuple[GradingTask, ...]:
        ordered = tuple(sorted(tasks, key=lambda task: task.task_id))
        task_ids = tuple(task.task_id for task in ordered)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("sealed task roster must contain unique task IDs")
        return ordered

    @property
    def fingerprint(self) -> str:
        """Identity over task IDs, public inputs, answers, and protected slices."""

        return canonical_sha256(self)


class CandidateOutputTrace(ImmutableModel):
    """Score-free output plus telemetry captured by the trusted runner wrapper."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    task_sha256: Sha256
    seed: Annotated[int, Field(ge=0, strict=True)]
    harness_id: NonEmptyStr
    execution_fingerprint: NonEmptyStr
    status: TrialStatus
    output: str | None
    tokens: Annotated[int, Field(ge=0, strict=True)]
    latency_ms: Annotated[float, Field(ge=0, strict=True)]
    tool_calls: Annotated[int, Field(ge=0, strict=True)]

    # These fields are capture-owned mechanism telemetry, not candidate claims.
    mutation_applied: bool
    target_component: NonEmptyStr
    activated: bool
    adhered: bool
    intervention: Literal["treatment", "placebo"]
    accessed_resources: tuple[NonEmptyStr, ...] = ()

    @field_validator("accessed_resources")
    @classmethod
    def canonicalize_accessed_resources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("accessed_resources must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def output_matches_terminal_status(self) -> CandidateOutputTrace:
        if self.status is TrialStatus.COMPLETED and self.output is None:
            raise ValueError("a completed trace must contain output")
        if self.status is not TrialStatus.COMPLETED and self.output is not None:
            raise ValueError("a non-completed trace must not claim a gradeable output")
        return self


class CapturedOutputTrace(ImmutableModel):
    """A trace plus content digest and trusted-parent HMAC attestation."""

    schema_version: Literal["1"] = "1"
    trace: CandidateOutputTrace
    captured_sha256: Sha256
    attestor_id: Sha256
    attestation_sha256: Sha256


def _capture_attestation(secret: bytes, trace: CandidateOutputTrace) -> str:
    payload = canonical_json_bytes(trace)
    return hmac.new(
        secret,
        _CAPTURE_ATTESTATION_DOMAIN + payload,
        hashlib.sha256,
    ).hexdigest()


class CaptureVerificationCapability:
    """Verify captures from one parent-held signer without exposing a sign API.

    This object still contains key material in the same Python process.  It is a
    capability-separation aid for trusted orchestration code, not protection
    from arbitrary code execution in that interpreter.
    """

    __slots__ = ("__secret", "_attestor_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("capture verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._attestor_id = hashlib.sha256(_CAPTURE_ATTESTOR_ID_DOMAIN + secret).hexdigest()

    @property
    def attestor_id(self) -> str:
        """Stable identifier for this ephemeral in-process signing capability."""

        return self._attestor_id

    def verify(self, captured: CapturedOutputTrace | object) -> CapturedOutputTrace:
        """Validate schema, content digest, signer identity, and HMAC."""

        validated = CapturedOutputTrace.model_validate(captured)
        actual_trace_sha256 = canonical_sha256(validated.trace)
        if validated.captured_sha256 != actual_trace_sha256:
            raise TrustedGradingError("captured trace digest does not match its content")
        if validated.attestor_id != self._attestor_id:
            raise TrustedGradingError("captured trace was signed by an untrusted attestor")
        expected = _capture_attestation(self.__secret, validated.trace)
        if not hmac.compare_digest(validated.attestation_sha256, expected):
            raise TrustedGradingError("captured trace HMAC attestation is invalid")
        return validated


class TrustedCaptureService:
    """Parent-process capability that signs runner-captured trace values.

    Candidate code may construct its own service, but its attestations use a
    different random key and are rejected by the grader paired with this
    instance.  Trusted orchestration must not pass this signer into candidate
    code.
    """

    __slots__ = ("__secret", "_verification_capability")

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)
        self._verification_capability = CaptureVerificationCapability(self.__secret)

    @property
    def verification_capability(self) -> CaptureVerificationCapability:
        """Return the verify-only capability to install in a trusted grader."""

        return self._verification_capability

    def capture(self, trace: CandidateOutputTrace) -> CapturedOutputTrace:
        """Authenticate telemetry constructed by the trusted runner wrapper."""

        validated = CandidateOutputTrace.model_validate(trace)
        return CapturedOutputTrace(
            trace=validated,
            captured_sha256=canonical_sha256(validated),
            attestor_id=self._verification_capability.attestor_id,
            attestation_sha256=_capture_attestation(self.__secret, validated),
        )


class FixedGraderConfig(ImmutableModel):
    """Frozen policy owned by the trusted grader, never by a candidate."""

    schema_version: Literal["1"] = "1"
    version: NonEmptyStr = "controlled-fixed-output-grader-v1"
    required_target_component: NonEmptyStr = "system_prompt"
    max_tokens: Annotated[int, Field(ge=0, strict=True)] = 64
    max_latency_ms: Annotated[float, Field(ge=0, strict=True)] = 10.0
    max_tool_calls: Annotated[int, Field(ge=0, strict=True)] = 0
    forbidden_resources: tuple[NonEmptyStr, ...] = (
        "grader_internals",
        "sealed_answers",
        "split_manifest",
    )

    @field_validator("forbidden_resources")
    @classmethod
    def canonicalize_forbidden_resources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("forbidden_resources must not contain duplicates")
        return ordered


class CaptureAttestation(ImmutableModel):
    """Replay-facing provenance retained from one authenticated capture."""

    task_id: NonEmptyStr
    seed: Annotated[int, Field(ge=0, strict=True)]
    harness_id: NonEmptyStr
    trace_sha256: Sha256
    attestor_id: Sha256
    attestation_sha256: Sha256


class GradedTrace(ImmutableModel):
    """Trusted result for one captured trace."""

    observation: TrialObservation
    trace_sha256: Sha256
    grader_fingerprint: Sha256
    capture_attestation: CaptureAttestation
    mechanism_checks: tuple[MechanismCheck, ...] = ()

    @model_validator(mode="after")
    def observation_matches_capture_provenance(self) -> GradedTrace:
        provenance = self.capture_attestation
        if self.trace_sha256 != provenance.trace_sha256:
            raise ValueError("graded trace digest does not match capture provenance")
        if (
            self.observation.task_id,
            self.observation.seed,
            self.observation.harness_id,
        ) != (provenance.task_id, provenance.seed, provenance.harness_id):
            raise ValueError("graded observation does not match capture provenance")
        return self


class GradedBatch(ImmutableModel):
    """Deterministically ordered observations and candidate mechanism evidence."""

    schema_version: Literal["1"] = "1"
    task_set_fingerprint: Sha256
    grader_fingerprint: Sha256
    observations: Annotated[tuple[TrialObservation, ...], Field(min_length=1)]
    capture_attestations: Annotated[tuple[CaptureAttestation, ...], Field(min_length=1)]
    mechanism_evidence: MechanismEvidence | None = None

    @model_validator(mode="after")
    def provenance_covers_exact_observation_roster(self) -> GradedBatch:
        if len(self.capture_attestations) != len(self.observations):
            raise ValueError("every graded observation requires one capture attestation")
        for observation, provenance in zip(
            self.observations,
            self.capture_attestations,
            strict=True,
        ):
            if (
                observation.task_id,
                observation.seed,
                observation.harness_id,
            ) != (provenance.task_id, provenance.seed, provenance.harness_id):
                raise ValueError("graded batch provenance does not match observation roster")
        return self


class FixedOutputGrader:
    """Grade captured output against sealed tasks under one immutable policy."""

    def __init__(
        self,
        config: FixedGraderConfig | None = None,
        *,
        capture_verifier: CaptureVerificationCapability,
    ) -> None:
        supplied = config or FixedGraderConfig()
        self._config = FixedGraderConfig.model_validate(
            supplied.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        if not isinstance(capture_verifier, CaptureVerificationCapability):
            raise TypeError("capture_verifier must be a CaptureVerificationCapability")
        self._capture_verifier = capture_verifier
        self._fingerprint = canonical_sha256(
            {
                "implementation": _GRADER_IMPLEMENTATION_FINGERPRINT,
                "config": self._config,
                "capture_attestor_id": capture_verifier.attestor_id,
            }
        )

    @property
    def config(self) -> FixedGraderConfig:
        return self._config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def grade(
        self,
        task: GradingTask,
        captured: CapturedOutputTrace,
        *,
        candidate: bool,
    ) -> GradedTrace:
        """Create one score only after integrity, roster, and policy checks."""

        validated_task = GradingTask.model_validate(task)
        validated_capture = self._capture_verifier.verify(captured)
        trace = validated_capture.trace
        actual_trace_sha256 = canonical_sha256(trace)
        if trace.task_id != validated_task.task_id:
            raise TrustedGradingError("captured trace belongs to a different task")
        if trace.task_sha256 != validated_task.fingerprint:
            raise TrustedGradingError("captured trace task fingerprint does not match sealed task")

        violations = self._policy_violations(trace)
        status = trace.status
        if violations:
            status = TrialStatus.POLICY_VIOLATION
        score = None
        if status is TrialStatus.COMPLETED:
            score = 1.0 if trace.output == validated_task.expected_output else 0.0

        observation = TrialObservation(
            task_id=trace.task_id,
            seed=trace.seed,
            harness_id=trace.harness_id,
            status=status,
            score=score,
            slice_tags=validated_task.slice_tags,
            tokens=trace.tokens,
            latency_ms=trace.latency_ms,
            tool_calls=trace.tool_calls,
            violations=violations,
            execution_fingerprint=trace.execution_fingerprint,
        )
        checks = self._mechanism_checks(trace, actual_trace_sha256) if candidate else ()
        return GradedTrace(
            observation=observation,
            trace_sha256=actual_trace_sha256,
            grader_fingerprint=self._fingerprint,
            capture_attestation=CaptureAttestation(
                task_id=trace.task_id,
                seed=trace.seed,
                harness_id=trace.harness_id,
                trace_sha256=actual_trace_sha256,
                attestor_id=validated_capture.attestor_id,
                attestation_sha256=validated_capture.attestation_sha256,
            ),
            mechanism_checks=checks,
        )

    def grade_batch(
        self,
        tasks: Sequence[GradingTask],
        captures: Sequence[CapturedOutputTrace | object],
        *,
        candidate: bool,
    ) -> GradedBatch:
        """Grade one exact task roster; missing, extra, or duplicate traces fail closed."""

        try:
            validated_tasks = tuple(GradingTask.model_validate(task) for task in tasks)
            validated_captures = tuple(
                CapturedOutputTrace.model_validate(capture) for capture in captures
            )
        except (TypeError, ValueError) as exc:
            raise TrustedGradingError(f"malformed captured trace: {exc}") from exc

        task_roster = SealedTaskRoster(tasks=validated_tasks)
        tasks_by_id = self._unique_tasks(task_roster.tasks)
        captures_by_id = self._unique_captures(validated_captures)
        if tasks_by_id.keys() != captures_by_id.keys():
            missing = sorted(tasks_by_id.keys() - captures_by_id.keys())
            unexpected = sorted(captures_by_id.keys() - tasks_by_id.keys())
            raise TrustedGradingError(
                f"captured roster mismatch; missing={missing!r}, unexpected={unexpected!r}"
            )

        grades = tuple(
            self.grade(tasks_by_id[task_id], captures_by_id[task_id], candidate=candidate)
            for task_id in sorted(tasks_by_id)
        )
        harness_ids = {grade.observation.harness_id for grade in grades}
        if len(harness_ids) != 1:
            raise TrustedGradingError("one grading batch must belong to exactly one harness")

        mechanism_evidence = None
        if candidate:
            checks = tuple(self._aggregate_check(name, grades) for name in MECHANISM_CHECK_NAMES)
            mechanism_evidence = MechanismEvidence(
                checks=checks,
                candidate_harness_id=next(iter(harness_ids)),
            )
        return GradedBatch(
            task_set_fingerprint=task_roster.fingerprint,
            grader_fingerprint=self._fingerprint,
            observations=tuple(grade.observation for grade in grades),
            capture_attestations=tuple(grade.capture_attestation for grade in grades),
            mechanism_evidence=mechanism_evidence,
        )

    def _policy_violations(self, trace: CandidateOutputTrace) -> tuple[str, ...]:
        violations: list[str] = []
        if trace.tokens > self._config.max_tokens:
            violations.append(f"token budget exceeded: {trace.tokens} > {self._config.max_tokens}")
        if trace.latency_ms > self._config.max_latency_ms:
            violations.append(
                f"latency budget exceeded: {trace.latency_ms} > {self._config.max_latency_ms}"
            )
        if trace.tool_calls > self._config.max_tool_calls:
            violations.append(
                f"tool-call budget exceeded: {trace.tool_calls} > {self._config.max_tool_calls}"
            )
        forbidden = sorted(
            set(trace.accessed_resources).intersection(self._config.forbidden_resources)
        )
        if forbidden:
            violations.append(f"forbidden resource access: {', '.join(forbidden)}")
        return tuple(violations)

    def _mechanism_checks(
        self,
        trace: CandidateOutputTrace,
        trace_sha256: str,
    ) -> tuple[MechanismCheck, ...]:
        values = {
            "mutation_applied": trace.mutation_applied,
            "target_component": trace.target_component == self._config.required_target_component,
            "activation": trace.activated,
            "adherence": trace.adhered,
            "non_placebo": trace.intervention == "treatment",
        }
        return tuple(
            MechanismCheck(
                name=name,
                passed=values[name],
                details=f"trusted capture observed {name}={values[name]}",
                evidence_refs=(trace_sha256,),
            )
            for name in MECHANISM_CHECK_NAMES
        )

    @staticmethod
    def _aggregate_check(name: str, grades: tuple[GradedTrace, ...]) -> MechanismCheck:
        checks = tuple(
            check for grade in grades for check in grade.mechanism_checks if check.name == name
        )
        if len(checks) != len(grades):
            raise TrustedGradingError(f"trusted grader omitted mechanism check {name!r}")
        passed = all(check.passed is True for check in checks)
        return MechanismCheck(
            name=name,
            passed=passed,
            details=f"{sum(check.passed is True for check in checks)}/{len(checks)} traces passed",
            evidence_refs=tuple(grade.trace_sha256 for grade in grades),
        )

    @staticmethod
    def _unique_tasks(values: tuple[GradingTask, ...]) -> dict[str, GradingTask]:
        indexed: dict[str, GradingTask] = {}
        for value in values:
            if value.task_id in indexed:
                raise TrustedGradingError(f"duplicate sealed task for task {value.task_id!r}")
            indexed[value.task_id] = value
        if not indexed:
            raise TrustedGradingError("sealed task roster must not be empty")
        return indexed

    @staticmethod
    def _unique_captures(
        values: tuple[CapturedOutputTrace, ...],
    ) -> dict[str, CapturedOutputTrace]:
        indexed: dict[str, CapturedOutputTrace] = {}
        for value in values:
            task_id = value.trace.task_id
            if task_id in indexed:
                raise TrustedGradingError(f"duplicate captured trace for task {task_id!r}")
            indexed[task_id] = value
        if not indexed:
            raise TrustedGradingError("captured trace roster must not be empty")
        return indexed


__all__ = [
    "MECHANISM_CHECK_NAMES",
    "CandidateOutputTrace",
    "CaptureAttestation",
    "CaptureVerificationCapability",
    "CapturedOutputTrace",
    "FixedGraderConfig",
    "FixedOutputGrader",
    "GradedBatch",
    "GradedTrace",
    "GradingTask",
    "SealedTaskRoster",
    "TrustedCaptureService",
    "TrustedGradingError",
]
