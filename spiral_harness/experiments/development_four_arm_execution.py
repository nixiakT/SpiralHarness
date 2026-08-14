"""Execution and aggregation helpers for the four-arm development runner."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from spiral_harness.benchmark.bbh import BBHLogicalDeductionSevenAdapter
from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import (
    CandidateTask,
    ModelExecutionRecord,
    ResolvedHarness,
)
from spiral_harness.execution.model import FixedModelRunner
from spiral_harness.execution.pure_contracts import PureReferenceExecutionRecord
from spiral_harness.execution.pure_model import PureModelRunner
from spiral_harness.experiments.development_four_arm_contracts import (
    AutomaticCandidateSelection,
    BenchmarkCandidateComparison,
    BenchmarkKind,
    DevelopmentArm,
    DevelopmentArmObservation,
    DevelopmentTask,
    FrozenCandidateProposal,
    FullDisclosure,
    FullFitTaskDisclosure,
    ScoreBenchmarkAggregate,
    ScoreDisclosure,
    SelectionDecision,
)

_Adapters = tuple[GSM8KBenchmarkAdapter, BBHLogicalDeductionSevenAdapter]


@dataclass(frozen=True, slots=True)
class EvaluatedDevelopmentCall:
    benchmark: BenchmarkKind
    task_id: str
    question: str
    arm: DevelopmentArm
    condition_id: str
    seed: int
    correct: bool
    output: str
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    execution: object


def evaluate_fixed(
    *,
    task: DevelopmentTask,
    arm: DevelopmentArm,
    condition_id: str,
    harness: ResolvedHarness,
    runner: FixedModelRunner,
    adapters: _Adapters,
    trace: list[dict[str, object]],
    phase: str,
) -> EvaluatedDevelopmentCall:
    adapter = _adapter(task.benchmark, adapters)
    trusted_task = adapter.load_task(task.task_id)
    record = runner.execute_record(trusted_task, harness=harness, seed=task.seed)
    observation = adapter.grade(
        trusted_task,
        record.execution,
        harness_id=harness.harness_ref.sha256,
        seed=task.seed,
        execution_fingerprint=record.execution.execution_fingerprint,
    )
    output = record.execution.output_text or _failure_output(record)
    correct = observation.score == 1.0
    trace.append(
        _trace_entry(
            sequence=len(trace) + 1,
            phase=phase,
            arm=arm,
            condition_id=condition_id,
            execution_ref=record.execution_ref,
            outcome_ref=record.outcome_ref,
            execution=record.execution,
            task_id=task.task_id,
            score=observation.score,
        )
    )
    return EvaluatedDevelopmentCall(
        benchmark=task.benchmark,
        task_id=task.task_id,
        # Disclosure is sourced from the trusted adapter, never the caller-authored plan.
        question=trusted_task.question,
        arm=arm,
        condition_id=condition_id,
        seed=task.seed,
        correct=correct,
        output=output,
        execution_ref=record.execution_ref,
        outcome_ref=record.outcome_ref,
        execution=record.execution,
    )


def evaluate_pure(
    *,
    task: DevelopmentTask,
    condition_id: str,
    runner: PureModelRunner,
    adapters: _Adapters,
    trace: list[dict[str, object]],
) -> EvaluatedDevelopmentCall:
    adapter = _adapter(task.benchmark, adapters)
    trusted_task = adapter.load_task(task.task_id)
    record = runner.execute_record(
        trusted_task,
        reference_id=condition_id,
        rollout_seed=task.seed,
    )
    observation = adapter.grade(
        trusted_task,
        record.execution,
        harness_id=condition_id,
        seed=task.seed,
        execution_fingerprint=record.execution.execution_fingerprint,
    )
    output = record.execution.output_text or _failure_output(record)
    trace.append(
        _trace_entry(
            sequence=len(trace) + 1,
            phase="development-holdout",
            arm=DevelopmentArm.PURE,
            condition_id=condition_id,
            execution_ref=record.execution_ref,
            outcome_ref=record.outcome_ref,
            execution=record.execution,
            task_id=task.task_id,
            score=observation.score,
        )
    )
    return EvaluatedDevelopmentCall(
        benchmark=task.benchmark,
        task_id=task.task_id,
        question=task.question,
        arm=DevelopmentArm.PURE,
        condition_id=condition_id,
        seed=task.seed,
        correct=observation.score == 1.0,
        output=output,
        execution_ref=record.execution_ref,
        outcome_ref=record.outcome_ref,
        execution=record.execution,
    )


def execute_proposer(
    *,
    runner: FixedModelRunner,
    harness: ResolvedHarness,
    arm: DevelopmentArm,
    prompt: str,
    adaptive_stage_fingerprint: str,
    trace: list[dict[str, object]],
) -> ModelExecutionRecord:
    task = CandidateTask(
        task_id=f"development-four-arm/proposer/{arm.value}",
        question=prompt,
    )
    seed = int(
        canonical_sha256(
            {
                "schema": "spiral-harness/development-four-arm-proposer-seed/v1",
                "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
            }
        )[:8],
        16,
    )
    record = runner.execute_record(task, harness=harness, seed=seed)
    trace.append(
        _trace_entry(
            sequence=len(trace) + 1,
            phase="proposal",
            arm=arm,
            condition_id=harness.harness_ref.sha256,
            execution_ref=record.execution_ref,
            outcome_ref=record.outcome_ref,
            execution=record.execution,
            task_id=task.task_id,
            score=None,
        )
    )
    return record


def score_disclosure(parent: tuple[EvaluatedDevelopmentCall, ...]) -> ScoreDisclosure:
    return ScoreDisclosure(
        aggregates=tuple(
            ScoreBenchmarkAggregate(
                benchmark=benchmark,
                score=sum(item.correct for item in parent if item.benchmark is benchmark) / 4,
            )
            for benchmark in (BenchmarkKind.GSM8K, BenchmarkKind.BBH)
        )
    )


def full_disclosure(parent: tuple[EvaluatedDevelopmentCall, ...]) -> FullDisclosure:
    return FullDisclosure(
        observations=tuple(
            FullFitTaskDisclosure(
                task_id=item.task_id,
                benchmark=item.benchmark,
                question=item.question,
                output=item.output,
                correct=item.correct,
            )
            for item in parent
        )
    )


def select_candidate(
    *,
    arm: DevelopmentArm,
    proposal: FrozenCandidateProposal,
    candidate_valid: bool,
    parent: tuple[EvaluatedDevelopmentCall, ...],
    candidate: tuple[EvaluatedDevelopmentCall, ...],
) -> AutomaticCandidateSelection:
    comparisons = tuple(
        BenchmarkCandidateComparison(
            benchmark=benchmark,
            parent_correct=sum(item.correct for item in parent if item.benchmark is benchmark),
            candidate_correct=sum(
                item.correct for item in candidate if item.benchmark is benchmark
            ),
        )
        for benchmark in (BenchmarkKind.GSM8K, BenchmarkKind.BBH)
    )
    promote = (
        candidate_valid
        and all(item.candidate_correct >= item.parent_correct for item in comparisons)
        and sum(item.candidate_correct for item in comparisons)
        > sum(item.parent_correct for item in comparisons)
    )
    decision = SelectionDecision.PROMOTE if promote else SelectionDecision.ROLLBACK
    return AutomaticCandidateSelection(
        arm=arm,
        candidate_id=proposal.candidate_id,
        candidate_valid=candidate_valid,
        comparisons=comparisons,
        decision=decision,
        selected_condition_id=(
            proposal.candidate_condition_id
            if decision is SelectionDecision.PROMOTE
            else proposal.parent_condition_id
        ),
    )


def holdout_metrics(
    observations: tuple[DevelopmentArmObservation, ...],
) -> dict[str, object]:
    counts = Counter(item.arm for item in observations if item.correct)
    accuracies = {arm.value: counts[arm] / 8 for arm in DevelopmentArm}
    return {
        "correct": {arm.value: counts[arm] for arm in DevelopmentArm},
        "accuracy": accuracies,
        "paired_point_deltas": {
            "full_minus_pure": accuracies["full"] - accuracies["pure"],
            "full_minus_static": accuracies["full"] - accuracies["static"],
            "full_minus_score": accuracies["full"] - accuracies["score"],
        },
    }


def provider_identity_summary(trace: list[dict[str, object]]) -> dict[str, object]:
    observations = [
        item["provider_identity_observation"]
        for item in trace
        if item["provider_identity_observation"] is not None
    ]
    fingerprints = {canonical_sha256(item) for item in observations if isinstance(item, dict)}
    complete = bool(trace) and len(observations) == len(trace)
    return {
        "trust_level": "provider-declared",
        "observed_call_count": len(observations),
        "missing_call_count": len(trace) - len(observations),
        "unique_observation_fingerprints": sorted(fingerprints),
        "all_observations_identical": complete and len(fingerprints) == 1,
        "signed_served_revision_receipt_present": False,
    }


def _adapter(benchmark: BenchmarkKind, adapters: _Adapters) -> object:
    return adapters[0] if benchmark is BenchmarkKind.GSM8K else adapters[1]


def _failure_output(record: ModelExecutionRecord | PureReferenceExecutionRecord) -> str:
    error = record.execution.error
    label = "unknown" if error is None else error.error_class.value
    return f"[execution failed closed: {label}]"


def _trace_entry(
    *,
    sequence: int,
    phase: str,
    arm: DevelopmentArm,
    condition_id: str,
    execution_ref: ArtifactRef,
    outcome_ref: ArtifactRef,
    execution: object,
    task_id: str,
    score: float | None,
) -> dict[str, object]:
    identity = execution.provider_identity_observation
    return {
        "sequence": sequence,
        "phase": phase,
        "arm": arm.value,
        "condition_id": condition_id,
        "task_id": task_id,
        "execution_ref": execution_ref.model_dump(mode="json"),
        "outcome_ref": outcome_ref.model_dump(mode="json"),
        "request_sha256": execution.request_sha256,
        "model_spec_fingerprint": execution.spec.fingerprint,
        "status": execution.status.value,
        "input_tokens": execution.usage.input_tokens,
        "output_tokens": execution.usage.output_tokens,
        "latency_ms": execution.latency_ms,
        "tool_calls": execution.tool_calls,
        "cost_usd": execution.cost_usd,
        "score": score,
        "provider_identity_observation": (
            None if identity is None else identity.model_dump(mode="json")
        ),
    }


__all__ = [
    "EvaluatedDevelopmentCall",
    "evaluate_fixed",
    "evaluate_pure",
    "execute_proposer",
    "full_disclosure",
    "holdout_metrics",
    "provider_identity_summary",
    "score_disclosure",
    "select_candidate",
]
