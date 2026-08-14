"""Fail-closed contracts for a permanently non-reportable four-arm study.

This module deliberately contains no execution code.  It describes one small
development-only exercise over sixteen public tasks and makes the limits of
that exercise part of every valid closure.  In particular, nothing representable
here is sealed, confirmatory, provider-attested, or suitable for a benchmark
claim.
"""

from __future__ import annotations

import math
from collections import Counter
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import FrozenModelSpec

DEVELOPMENT_BENCHMARKS = ("gsm8k", "bbh")
DEVELOPMENT_ARMS = ("pure", "static", "score", "full")
DEVELOPMENT_TASKS_PER_BENCHMARK = 8
DEVELOPMENT_TASKS_PER_SPLIT = 4
DEVELOPMENT_TOTAL_TASKS = 16
DEVELOPMENT_HOLDOUT_TASKS = 8
DEVELOPMENT_MAX_MODEL_CALLS = 58
DEVELOPMENT_EVIDENCE_SCOPE = "public-development-feedback-view-only"
DEVELOPMENT_ADAPTIVE_CONTRAST = "item-evidence-vs-aggregate-score-feedback-view-only"


class BenchmarkKind(StrEnum):
    """The two public benchmark kinds admitted by the development exercise."""

    GSM8K = "gsm8k"
    BBH = "bbh"


class DevelopmentSplit(StrEnum):
    """Public development partitions; neither value denotes a sealed split."""

    FIT = "fit"
    HOLDOUT = "holdout"


class DevelopmentArm(StrEnum):
    """The complete four-arm development roster."""

    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"


class SelectionDecision(StrEnum):
    """The only two outcomes of the deterministic development selector."""

    PROMOTE = "promote"
    ROLLBACK = "rollback"


_BENCHMARK_ROSTER = tuple(BenchmarkKind(value) for value in DEVELOPMENT_BENCHMARKS)
_ARM_ROSTER = tuple(DevelopmentArm(value) for value in DEVELOPMENT_ARMS)
_CANDIDATE_ARMS = (DevelopmentArm.SCORE, DevelopmentArm.FULL)


class _FingerprintedDevelopmentModel(ImmutableModel):
    """Common immutable identity for all development contract values."""

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class DevelopmentTask(_FingerprintedDevelopmentModel):
    """One public task coordinate frozen before the development exercise."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    benchmark: BenchmarkKind
    split: DevelopmentSplit
    question: NonEmptyStr
    seed: Annotated[int, Field(ge=0, strict=True)]


def _task_order(task: DevelopmentTask) -> tuple[str, str, str]:
    return task.benchmark.value, task.split.value, task.task_id


class DevelopmentFourArmPlan(_FingerprintedDevelopmentModel):
    """The sole admitted 2 x 8-task, four-arm development plan."""

    schema_version: Literal["1"] = "1"
    benchmarks: tuple[BenchmarkKind, ...] = _BENCHMARK_ROSTER
    arms: tuple[DevelopmentArm, ...] = _ARM_ROSTER
    tasks: Annotated[
        tuple[DevelopmentTask, ...],
        Field(min_length=DEVELOPMENT_TOTAL_TASKS, max_length=DEVELOPMENT_TOTAL_TASKS),
    ]
    model_spec: FrozenModelSpec
    max_model_calls: Literal[58] = DEVELOPMENT_MAX_MODEL_CALLS
    max_tokens_per_attempt: Annotated[int, Field(ge=1, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=1, strict=True)]
    sealed: Literal[False] = False
    evidence_scope: Literal["public-development-feedback-view-only"] = DEVELOPMENT_EVIDENCE_SCOPE
    adaptive_contrast: Literal["item-evidence-vs-aggregate-score-feedback-view-only"] = (
        DEVELOPMENT_ADAPTIVE_CONTRAST
    )
    score_full_shared_parent_rollouts: Literal[True] = True
    joint_policy_effect_identified: Literal[False] = False
    execution_attested: Literal[False] = False
    budget_matched: Literal[False] = False
    runtime_execution_proof: Literal[False] = False
    confirmatory_protocol: Literal[False] = False

    @field_validator("benchmarks")
    @classmethod
    def exact_benchmark_roster(cls, values: tuple[BenchmarkKind, ...]) -> tuple[BenchmarkKind, ...]:
        if len(values) != len(set(values)) or frozenset(values) != frozenset(_BENCHMARK_ROSTER):
            raise ValueError("development plan requires exactly gsm8k and bbh")
        return _BENCHMARK_ROSTER

    @field_validator("arms")
    @classmethod
    def exact_arm_roster(cls, values: tuple[DevelopmentArm, ...]) -> tuple[DevelopmentArm, ...]:
        if len(values) != len(set(values)) or frozenset(values) != frozenset(_ARM_ROSTER):
            raise ValueError("development plan requires exactly PURE, STATIC, SCORE, and FULL")
        return _ARM_ROSTER

    @field_validator("tasks")
    @classmethod
    def canonical_task_order(
        cls, values: tuple[DevelopmentTask, ...]
    ) -> tuple[DevelopmentTask, ...]:
        return tuple(sorted(values, key=_task_order))

    @model_validator(mode="after")
    def exact_task_matrix(self) -> Self:
        if self.max_tokens_per_attempt < self.model_spec.inference.max_output_tokens:
            raise ValueError("max_tokens_per_attempt must cover the model output-token ceiling")
        expected_total_tokens = self.max_model_calls * self.max_tokens_per_attempt
        if self.max_total_tokens != expected_total_tokens:
            raise ValueError("max_total_tokens must equal max_model_calls * max_tokens_per_attempt")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("all sixteen development task IDs must be globally unique")
        counts = Counter((task.benchmark, task.split) for task in self.tasks)
        expected = {
            (benchmark, split): DEVELOPMENT_TASKS_PER_SPLIT
            for benchmark in _BENCHMARK_ROSTER
            for split in DevelopmentSplit
        }
        if counts != expected:
            raise ValueError("each benchmark requires exactly four fit and four holdout tasks")
        return self


def development_adaptive_stage_fingerprint(plan: DevelopmentFourArmPlan) -> str:
    """Commit the adaptive stage without making it a function of holdout content."""

    checked = DevelopmentFourArmPlan.model_validate(plan, strict=True)
    fit_tasks = tuple(task for task in checked.tasks if task.split is DevelopmentSplit.FIT)
    return canonical_sha256(
        {
            "schema": "spiral-harness/development-four-arm-adaptive-stage/v1",
            "benchmarks": checked.benchmarks,
            "candidate_arms": _CANDIDATE_ARMS,
            "fit_tasks": fit_tasks,
            "model_spec": checked.model_spec,
            "max_model_calls": checked.max_model_calls,
            "max_tokens_per_attempt": checked.max_tokens_per_attempt,
            "max_total_tokens": checked.max_total_tokens,
            "evidence_scope": checked.evidence_scope,
            "adaptive_contrast": checked.adaptive_contrast,
            "score_full_shared_parent_rollouts": (checked.score_full_shared_parent_rollouts),
        }
    )


class ScoreBenchmarkAggregate(_FingerprintedDevelopmentModel):
    """The complete SCORE-visible view for one benchmark.

    There is intentionally no task identifier, question, output, correctness
    vector, answer, gold target, split payload, or trajectory field.
    """

    schema_version: Literal["1"] = "1"
    benchmark: BenchmarkKind
    evaluated_count: Literal[4] = DEVELOPMENT_TASKS_PER_SPLIT
    score: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]

    @model_validator(mode="after")
    def score_is_realizable_by_four_binary_tasks(self) -> Self:
        successes = self.score * self.evaluated_count
        if not math.isclose(successes, round(successes), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("aggregate score must be realizable by four binary tasks")
        return self


class ScoreDisclosure(_FingerprintedDevelopmentModel):
    """SCORE's type-level aggregate-only fit disclosure."""

    schema_version: Literal["1"] = "1"
    aggregates: Annotated[tuple[ScoreBenchmarkAggregate, ...], Field(min_length=2, max_length=2)]

    @field_validator("aggregates")
    @classmethod
    def canonical_aggregate_order(
        cls, values: tuple[ScoreBenchmarkAggregate, ...]
    ) -> tuple[ScoreBenchmarkAggregate, ...]:
        return tuple(sorted(values, key=lambda item: item.benchmark.value))

    @model_validator(mode="after")
    def exact_benchmark_coverage(self) -> Self:
        benchmarks = tuple(item.benchmark for item in self.aggregates)
        if len(benchmarks) != len(set(benchmarks)) or frozenset(benchmarks) != frozenset(
            _BENCHMARK_ROSTER
        ):
            raise ValueError("SCORE disclosure requires one aggregate per benchmark")
        return self


class FullFitTaskDisclosure(_FingerprintedDevelopmentModel):
    """One FULL-visible fit trace with no representable gold field."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    benchmark: BenchmarkKind
    split: Literal[DevelopmentSplit.FIT] = DevelopmentSplit.FIT
    question: NonEmptyStr
    output: NonEmptyStr
    correct: bool


class FullDisclosure(_FingerprintedDevelopmentModel):
    """FULL's per-task disclosure, restricted structurally to the eight fit tasks."""

    schema_version: Literal["1"] = "1"
    observations: Annotated[
        tuple[FullFitTaskDisclosure, ...],
        Field(min_length=8, max_length=8),
    ]

    @field_validator("observations")
    @classmethod
    def canonical_observation_order(
        cls, values: tuple[FullFitTaskDisclosure, ...]
    ) -> tuple[FullFitTaskDisclosure, ...]:
        return tuple(sorted(values, key=lambda item: (item.benchmark.value, item.task_id)))

    @model_validator(mode="after")
    def exact_fit_shape(self) -> Self:
        task_ids = tuple(item.task_id for item in self.observations)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("FULL disclosure must not repeat a fit task")
        counts = Counter(item.benchmark for item in self.observations)
        if counts != {benchmark: DEVELOPMENT_TASKS_PER_SPLIT for benchmark in _BENCHMARK_ROSTER}:
            raise ValueError("FULL disclosure requires four fit tasks per benchmark")
        return self


class FrozenCandidateProposal(_FingerprintedDevelopmentModel):
    """One candidate proposal included in the atomic two-proposal freeze."""

    schema_version: Literal["1"] = "1"
    arm: DevelopmentArm
    candidate_id: Sha256
    proposal_fingerprint: Sha256
    parent_condition_id: Sha256
    candidate_condition_id: Sha256

    @model_validator(mode="after")
    def candidate_arm_and_condition_are_valid(self) -> Self:
        if self.arm not in _CANDIDATE_ARMS:
            raise ValueError("only SCORE and FULL may carry candidate proposals")
        if self.parent_condition_id == self.candidate_condition_id:
            raise ValueError("candidate condition must differ from its parent condition")
        return self


def development_candidate_freeze_id(
    *,
    adaptive_stage_fingerprint: str,
    proposals: tuple[FrozenCandidateProposal, ...],
) -> str:
    """Derive the only valid identity for one joint candidate freeze."""

    ordered = tuple(sorted(proposals, key=lambda item: item.arm.value))
    return canonical_sha256(
        {
            "schema": "spiral-harness/development-four-arm-joint-freeze/v1",
            "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
            "proposals": ordered,
        }
    )


class JointCandidateFreeze(_FingerprintedDevelopmentModel):
    """One atomic freeze containing both proposals before either evaluation starts."""

    schema_version: Literal["1"] = "1"
    adaptive_stage_fingerprint: Sha256
    freeze_id: Sha256
    freeze_sequence: Annotated[int, Field(ge=0, strict=True)]
    candidate_evaluation_start_sequence: Annotated[int, Field(ge=1, strict=True)]
    proposals: Annotated[tuple[FrozenCandidateProposal, ...], Field(min_length=2, max_length=2)]
    frozen_before_candidate_evaluation: Literal[True] = True

    @field_validator("proposals")
    @classmethod
    def canonical_proposal_order(
        cls, values: tuple[FrozenCandidateProposal, ...]
    ) -> tuple[FrozenCandidateProposal, ...]:
        return tuple(sorted(values, key=lambda item: item.arm.value))

    @model_validator(mode="after")
    def both_candidates_were_frozen_first(self) -> Self:
        arms = tuple(proposal.arm for proposal in self.proposals)
        if len(arms) != len(set(arms)) or frozenset(arms) != frozenset(_CANDIDATE_ARMS):
            raise ValueError("joint freeze requires exactly the SCORE and FULL proposals")
        candidate_ids = tuple(proposal.candidate_id for proposal in self.proposals)
        proposal_ids = tuple(proposal.proposal_fingerprint for proposal in self.proposals)
        parent_condition_ids = tuple(proposal.parent_condition_id for proposal in self.proposals)
        candidate_condition_ids = tuple(
            proposal.candidate_condition_id for proposal in self.proposals
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("the two candidate proposals require distinct candidate IDs")
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("the two proposal fingerprints must be distinct")
        if len(set(parent_condition_ids)) != 1:
            raise ValueError("SCORE and FULL must reference one shared parent condition")
        if len(candidate_condition_ids) != len(set(candidate_condition_ids)):
            raise ValueError("SCORE and FULL require distinct candidate conditions")
        if self.freeze_sequence >= self.candidate_evaluation_start_sequence:
            raise ValueError("both proposals must freeze before candidate evaluation starts")
        expected_freeze_id = development_candidate_freeze_id(
            adaptive_stage_fingerprint=self.adaptive_stage_fingerprint,
            proposals=self.proposals,
        )
        if self.freeze_id != expected_freeze_id:
            raise ValueError("freeze_id must bind the fit-only adaptive stage and both proposals")
        return self


class BenchmarkCandidateComparison(_FingerprintedDevelopmentModel):
    """Fit-set parent/candidate counts used by the automatic selector."""

    schema_version: Literal["1"] = "1"
    benchmark: BenchmarkKind
    evaluated_count: Literal[4] = DEVELOPMENT_TASKS_PER_SPLIT
    parent_correct: Annotated[int, Field(ge=0, le=4, strict=True)]
    candidate_correct: Annotated[int, Field(ge=0, le=4, strict=True)]


_SELECTION_RULE = (
    "candidate-valid-and-total-correct-strictly-improves-and-each-benchmark-nondecreasing"
)


class AutomaticCandidateSelection(_FingerprintedDevelopmentModel):
    """A candidate decision whose result is completely determined by its counts."""

    schema_version: Literal["1"] = "1"
    arm: DevelopmentArm
    candidate_id: Sha256
    candidate_valid: bool
    comparisons: Annotated[
        tuple[BenchmarkCandidateComparison, ...], Field(min_length=2, max_length=2)
    ]
    decision: SelectionDecision
    selected_condition_id: Sha256
    rule: Literal[
        "candidate-valid-and-total-correct-strictly-improves-and-each-benchmark-nondecreasing"
    ] = _SELECTION_RULE
    manual_override: Literal[False] = False

    @field_validator("comparisons")
    @classmethod
    def canonical_comparison_order(
        cls, values: tuple[BenchmarkCandidateComparison, ...]
    ) -> tuple[BenchmarkCandidateComparison, ...]:
        return tuple(sorted(values, key=lambda item: item.benchmark.value))

    @model_validator(mode="after")
    def exact_automatic_decision(self) -> Self:
        if self.arm not in _CANDIDATE_ARMS:
            raise ValueError("only SCORE and FULL have candidate selections")
        benchmarks = tuple(item.benchmark for item in self.comparisons)
        if len(benchmarks) != len(set(benchmarks)) or frozenset(benchmarks) != frozenset(
            _BENCHMARK_ROSTER
        ):
            raise ValueError("selection requires one comparison per benchmark")
        total_parent = sum(item.parent_correct for item in self.comparisons)
        total_candidate = sum(item.candidate_correct for item in self.comparisons)
        no_benchmark_regressed = all(
            item.candidate_correct >= item.parent_correct for item in self.comparisons
        )
        promote = self.candidate_valid and total_candidate > total_parent and no_benchmark_regressed
        expected = SelectionDecision.PROMOTE if promote else SelectionDecision.ROLLBACK
        if self.decision is not expected:
            raise ValueError("selection decision differs from the automatic rule")
        return self


class DevelopmentArmObservation(_FingerprintedDevelopmentModel):
    """One holdout result with condition identity distinct from record identity."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    benchmark: BenchmarkKind
    split: Literal[DevelopmentSplit.HOLDOUT] = DevelopmentSplit.HOLDOUT
    arm: DevelopmentArm
    seed: Annotated[int, Field(ge=0, strict=True)]
    model_spec_fingerprint: Sha256
    condition_id: Sha256
    reference_id: Sha256
    correct: bool

    @model_validator(mode="after")
    def condition_is_not_the_observation_reference(self) -> Self:
        if self.condition_id == self.reference_id:
            raise ValueError("condition_id and reference_id must remain distinct")
        return self


def _arm_observation_order(
    observation: DevelopmentArmObservation,
) -> tuple[str, str, str]:
    return observation.benchmark.value, observation.task_id, observation.arm.value


class DevelopmentFourArmClosure(_FingerprintedDevelopmentModel):
    """Complete but permanently non-reportable closure for the development study."""

    schema_version: Literal["1"] = "1"
    plan: DevelopmentFourArmPlan
    plan_fingerprint: Sha256
    score_disclosure: ScoreDisclosure
    full_disclosure: FullDisclosure
    candidate_freeze: JointCandidateFreeze
    selections: Annotated[
        tuple[AutomaticCandidateSelection, ...], Field(min_length=2, max_length=2)
    ]
    observations: Annotated[
        tuple[DevelopmentArmObservation, ...], Field(min_length=32, max_length=32)
    ]
    evidence_scope: Literal["public-development-feedback-view-only"] = DEVELOPMENT_EVIDENCE_SCOPE
    adaptive_contrast: Literal["item-evidence-vs-aggregate-score-feedback-view-only"] = (
        DEVELOPMENT_ADAPTIVE_CONTRAST
    )
    score_full_shared_parent_projection: Literal[True] = True
    joint_policy_effect_identified: Literal[False] = False
    execution_attested: Literal[False] = False
    budget_matched: Literal[False] = False
    runtime_execution_proof: Literal[False] = False
    confirmatory_protocol: Literal[False] = False
    reportable_benchmark_result: Literal[False] = False
    sealed_evidence: Literal[False] = False
    confirmatory_inference: Literal[False] = False
    simultaneous_lcb_available: Literal[False] = False
    provider_identity_attested: Literal[False] = False

    @field_validator("selections")
    @classmethod
    def canonical_selection_order(
        cls, values: tuple[AutomaticCandidateSelection, ...]
    ) -> tuple[AutomaticCandidateSelection, ...]:
        return tuple(sorted(values, key=lambda item: item.arm.value))

    @field_validator("observations")
    @classmethod
    def canonicalize_arm_observations(
        cls, values: tuple[DevelopmentArmObservation, ...]
    ) -> tuple[DevelopmentArmObservation, ...]:
        return tuple(sorted(values, key=_arm_observation_order))

    @model_validator(mode="after")
    def close_exact_development_study(self) -> Self:
        if self.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("plan_fingerprint does not bind the embedded development plan")
        expected_adaptive_stage = development_adaptive_stage_fingerprint(self.plan)
        if self.candidate_freeze.adaptive_stage_fingerprint != expected_adaptive_stage:
            raise ValueError("candidate freeze must bind the plan's fit-only adaptive stage")

        fit_tasks = {
            task.task_id: task for task in self.plan.tasks if task.split is DevelopmentSplit.FIT
        }
        disclosed_tasks = {item.task_id: item for item in self.full_disclosure.observations}
        if frozenset(disclosed_tasks) != frozenset(fit_tasks):
            raise ValueError("FULL disclosure must cover exactly the plan's eight fit tasks")
        for task_id, disclosed in disclosed_tasks.items():
            planned = fit_tasks[task_id]
            if (disclosed.benchmark, disclosed.question) != (
                planned.benchmark,
                planned.question,
            ):
                raise ValueError("FULL disclosure differs from its frozen fit task")

        proposal_by_arm = {proposal.arm: proposal for proposal in self.candidate_freeze.proposals}
        selection_by_arm = {selection.arm: selection for selection in self.selections}
        if len(selection_by_arm) != 2 or frozenset(selection_by_arm) != frozenset(_CANDIDATE_ARMS):
            raise ValueError("closure requires exactly the SCORE and FULL selections")

        score_parent = {
            aggregate.benchmark: round(aggregate.score * aggregate.evaluated_count)
            for aggregate in self.score_disclosure.aggregates
        }
        full_parent = {
            benchmark: sum(
                item.correct
                for item in self.full_disclosure.observations
                if item.benchmark is benchmark
            )
            for benchmark in _BENCHMARK_ROSTER
        }
        if score_parent != full_parent:
            raise ValueError(
                "SCORE and FULL disclosures must be projections of the same shared parent rollouts"
            )
        expected_parent = {
            DevelopmentArm.SCORE: score_parent,
            DevelopmentArm.FULL: full_parent,
        }
        for arm in _CANDIDATE_ARMS:
            proposal = proposal_by_arm[arm]
            selection = selection_by_arm[arm]
            if selection.candidate_id != proposal.candidate_id:
                raise ValueError("selection candidate differs from its jointly frozen proposal")
            expected_condition = (
                proposal.candidate_condition_id
                if selection.decision is SelectionDecision.PROMOTE
                else proposal.parent_condition_id
            )
            if selection.selected_condition_id != expected_condition:
                raise ValueError("selected condition does not follow promote-or-rollback")
            for comparison in selection.comparisons:
                if comparison.parent_correct != expected_parent[arm][comparison.benchmark]:
                    raise ValueError(
                        "candidate comparison parent count differs from its authorized disclosure"
                    )

        holdout_tasks = {
            task.task_id: task for task in self.plan.tasks if task.split is DevelopmentSplit.HOLDOUT
        }
        observed_keys = tuple((item.task_id, item.arm) for item in self.observations)
        if len(observed_keys) != len(set(observed_keys)):
            raise ValueError("closure contains a duplicate task-arm observation")
        expected_keys = {(task_id, arm) for task_id in holdout_tasks for arm in _ARM_ROSTER}
        if set(observed_keys) != expected_keys:
            raise ValueError("closure requires all eight holdout tasks in all four arms")

        model_spec_fingerprint = self.plan.model_spec.fingerprint
        arm_conditions: dict[DevelopmentArm, str] = {}
        reference_ids: list[str] = []
        for observation in self.observations:
            task = holdout_tasks[observation.task_id]
            if (observation.benchmark, observation.seed) != (task.benchmark, task.seed):
                raise ValueError("holdout observation differs from its frozen task and seed")
            if observation.model_spec_fingerprint != model_spec_fingerprint:
                raise ValueError("every arm must use the plan's one FrozenModelSpec")
            previous_condition = arm_conditions.setdefault(
                observation.arm, observation.condition_id
            )
            if previous_condition != observation.condition_id:
                raise ValueError("one development arm cannot drift between condition IDs")
            reference_ids.append(observation.reference_id)

        if len(arm_conditions) != 4:
            raise ValueError("closure requires one stable condition for every arm")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("each arm observation requires a unique reference_id")
        if set(arm_conditions.values()).intersection(reference_ids):
            raise ValueError("condition IDs and observation reference IDs must be disjoint")
        shared_parent_condition = proposal_by_arm[DevelopmentArm.SCORE].parent_condition_id
        if arm_conditions[DevelopmentArm.STATIC] != shared_parent_condition:
            raise ValueError(
                "STATIC, SCORE, and FULL must use the jointly frozen shared parent condition"
            )
        if arm_conditions[DevelopmentArm.PURE] in {
            arm_conditions[DevelopmentArm.STATIC],
            arm_conditions[DevelopmentArm.SCORE],
            arm_conditions[DevelopmentArm.FULL],
        }:
            raise ValueError("PURE must remain distinct from every fixed-harness condition")
        for arm in _CANDIDATE_ARMS:
            if arm_conditions[arm] != selection_by_arm[arm].selected_condition_id:
                raise ValueError(
                    "adaptive arm did not evaluate its automatically selected condition"
                )
        return self


__all__ = [
    "DEVELOPMENT_ADAPTIVE_CONTRAST",
    "DEVELOPMENT_EVIDENCE_SCOPE",
    "AutomaticCandidateSelection",
    "BenchmarkCandidateComparison",
    "BenchmarkKind",
    "DevelopmentArm",
    "DevelopmentArmObservation",
    "DevelopmentFourArmClosure",
    "DevelopmentFourArmPlan",
    "DevelopmentSplit",
    "DevelopmentTask",
    "FrozenCandidateProposal",
    "FullDisclosure",
    "FullFitTaskDisclosure",
    "JointCandidateFreeze",
    "ScoreBenchmarkAggregate",
    "ScoreDisclosure",
    "SelectionDecision",
    "development_adaptive_stage_fingerprint",
    "development_candidate_freeze_id",
]
