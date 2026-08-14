"""Strict contracts for the model-output-driven HarnessFault development slice.

This is deliberately separate from the v4 deterministic middleware verifier.
It is an exploration-only execution path: a model call chooses one finite repair
rule and the unmodified outputs of later model calls are the scored behavior.
Nothing in this module upgrades public development work into confirmatory
evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.benchmark._harness_fault_cases import (
    FaultFamily,
    FaultSurface,
    PartitionEvaluationGrant,
    RepairRuleId,
    RuntimeBranch,
    ScenarioRole,
)
from spiral_harness.benchmark.harness_fault import HarnessFaultOutput
from spiral_harness.benchmark.harness_fault_compiler import (
    HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    FaultRepairAction,
    HarnessRole,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptBudget,
    ExecutionStatus,
    FrozenModelSpec,
)
from spiral_harness.experiments.confirmatory_arms import (
    FaultFactorialCell,
    OptimizerFeedbackMode,
    PromotionRule,
    make_fault_factorial_profile,
)

HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.model-driven-fault-feedback.v1+json"
)
HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.model-driven-fault-development-result.v1+json"
)


class HarnessFaultDevelopmentError(ValueError):
    """A development plan, execution closure, or provenance check failed."""


class DevelopmentInvocationMode(StrEnum):
    """How calls are routed; neither value is provider attestation."""

    FIXTURE_REPLAY = "fixture-replay"
    LIVE_PROVIDER_UNATTESTED = "live-provider-unattested"


class FeedbackEvidenceLinkage(StrEnum):
    """Projection delivered to the condition-local proposer."""

    SCORE_ONLY = "score-only"
    SOURCE_LINKED = "source-linked"
    SHUFFLED_PLACEBO = "shuffled-evidence-placebo"


class ProposerVisibleFeedbackMode(StrEnum):
    """The only labels exposed to the proposer; placebo assignment stays hidden."""

    SCORE_ONLY = "score-only"
    MECHANISM_VISIBLE = "mechanism-visible"


class PromotionDecision(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"


class HarnessFaultDevelopmentPlan(ImmutableModel):
    """One-round, exact-budget public exploration plan.

    Every plan executes one parent observation pass, one proposer call, and a
    candidate/revert/placebo pass.  The caller cannot supply a candidate.
    """

    schema_version: Literal["1"] = "1"
    partition: Literal[ProtocolPartition.EXPLORATION] = ProtocolPartition.EXPLORATION
    cell: FaultFactorialCell
    feedback_linkage: FeedbackEvidenceLinkage
    invocation_mode: DevelopmentInvocationMode
    model_spec: FrozenModelSpec
    task_count: Annotated[int, Field(ge=4, strict=True)]
    proposal_seed: Annotated[int, Field(ge=0, strict=True)]
    solver_master_seed: Annotated[int, Field(ge=0, strict=True)]
    max_tokens_per_call: Annotated[int, Field(ge=1, strict=True)]
    max_model_calls: Annotated[int, Field(ge=17, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=17, strict=True)]
    proposal_retries: Literal[0] = 0
    candidates_per_round: Literal[1] = 1
    human_candidate_selection_permitted: Literal[False] = False
    rounds: Literal[1] = 1

    @model_validator(mode="after")
    def exact_profile_and_budget(self) -> Self:
        profile = make_fault_factorial_profile(self.cell)
        if profile.optimizer_feedback is OptimizerFeedbackMode.SCORE_ONLY:
            if self.feedback_linkage is not FeedbackEvidenceLinkage.SCORE_ONLY:
                raise ValueError("SS/SM require the exact score-only proposer projection")
        elif self.feedback_linkage not in {
            FeedbackEvidenceLinkage.SOURCE_LINKED,
            FeedbackEvidenceLinkage.SHUFFLED_PLACEBO,
        }:
            raise ValueError("MS/MM require source-linked or shuffled mechanism evidence")
        expected_calls = 4 * self.task_count + 1
        if self.max_model_calls != expected_calls:
            raise ValueError("max_model_calls must equal the exact 4N+1 call topology")
        if self.max_total_tokens != expected_calls * self.max_tokens_per_call:
            raise ValueError("max_total_tokens must reserve every exact call slot")
        return self

    @property
    def optimizer_feedback(self) -> OptimizerFeedbackMode:
        return make_fault_factorial_profile(self.cell).optimizer_feedback

    @property
    def promotion_rule(self) -> PromotionRule:
        return make_fault_factorial_profile(self.cell).promotion_rule

    @property
    def attempt_budget(self) -> AttemptBudget:
        return AttemptBudget(
            max_attempts=self.max_model_calls,
            max_total_tokens=self.max_total_tokens,
            max_tokens_per_attempt=self.max_tokens_per_call,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class FaultDevelopmentObservation(ImmutableModel):
    """One scored raw model output; no middleware-created output is admitted."""

    schema_version: Literal["1"] = "1"
    harness_role: HarnessRole
    scenario_id: NonEmptyStr
    scenario_commitment: Sha256
    task_id: NonEmptyStr
    family: FaultFamily
    surface: FaultSurface
    scenario_role: ScenarioRole
    seed: Annotated[int, Field(ge=0, strict=True)]
    execution_status: ExecutionStatus
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    raw_model_output_sha256: Sha256 | None
    parsed_output: HarnessFaultOutput | None
    inferred_branch: RuntimeBranch | None
    behavior_correct: bool
    output_was_rewritten: Literal[False] = False

    @model_validator(mode="after")
    def exact_role_media_and_status(self) -> Self:
        if self.harness_role not in {
            HarnessRole.FAULTY_PARENT,
            HarnessRole.CANDIDATE,
            HarnessRole.REVERT,
            HarnessRole.PLACEBO,
        }:
            raise ValueError("development observations admit the attribution quartet only")
        if self.execution_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise ValueError("execution_ref declares the wrong media type")
        if self.outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("outcome_ref declares the wrong media type")
        completed = self.execution_status is ExecutionStatus.COMPLETED
        if completed != (self.raw_model_output_sha256 is not None):
            raise ValueError("raw output hash must exist exactly for completed executions")
        if not completed and (
            self.parsed_output is not None
            or self.inferred_branch is not None
            or self.behavior_correct
        ):
            raise ValueError("failed executions cannot expose parsed or correct behavior")
        return self


class FaultFeedbackItem(ImmutableModel):
    """One proposer-visible evidence item with explicit source linkage."""

    task_id: NonEmptyStr
    source_task_id: NonEmptyStr
    family: FaultFamily
    surface: FaultSurface
    scenario_role: ScenarioRole
    raw_model_output_sha256: Sha256 | None
    parsed_output: HarnessFaultOutput | None
    inferred_branch: RuntimeBranch | None
    behavior_correct: bool


class HarnessFaultFeedbackProjection(ImmutableModel):
    """Authority packet; treatment linkage is not directly shown to the proposer."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    linkage: FeedbackEvidenceLinkage
    parent_correct: Annotated[int, Field(ge=0, strict=True)]
    task_count: Annotated[int, Field(ge=4, strict=True)]
    items: tuple[FaultFeedbackItem, ...]
    authority_labeled_placebo: bool

    @model_validator(mode="after")
    def exact_projection_shape(self) -> Self:
        if self.parent_correct > self.task_count:
            raise ValueError("parent_correct exceeds the projection roster")
        if self.linkage is FeedbackEvidenceLinkage.SCORE_ONLY:
            if self.items or self.authority_labeled_placebo:
                raise ValueError("score-only feedback cannot contain item evidence")
            return self
        if len(self.items) != self.task_count:
            raise ValueError("mechanism feedback must cover the complete parent roster")
        if len({item.task_id for item in self.items}) != self.task_count:
            raise ValueError("mechanism feedback task IDs must be unique")
        if self.linkage is FeedbackEvidenceLinkage.SOURCE_LINKED:
            if self.authority_labeled_placebo:
                raise ValueError("source-linked evidence cannot be labeled placebo")
            if any(item.task_id != item.source_task_id for item in self.items):
                raise ValueError("source-linked evidence contains a shuffled source")
        else:
            if not self.authority_labeled_placebo:
                raise ValueError("shuffled evidence must be labeled in authority metadata")
            if all(item.task_id == item.source_task_id for item in self.items):
                raise ValueError("shuffled evidence did not change any source linkage")
        return self


class ProposerVisibleFaultFeedbackItem(ImmutableModel):
    """Item projection that hides the authority's source-assignment metadata."""

    task_id: NonEmptyStr
    family: FaultFamily
    surface: FaultSurface
    scenario_role: ScenarioRole
    raw_model_output_sha256: Sha256 | None
    parsed_output: HarnessFaultOutput | None
    inferred_branch: RuntimeBranch | None
    behavior_correct: bool


class ProposerVisibleFaultFeedback(ImmutableModel):
    """Blinded packet with identical labels for linked and shuffled mechanism arms."""

    schema_version: Literal["1"] = "1"
    feedback_mode: ProposerVisibleFeedbackMode
    parent_correct: Annotated[int, Field(ge=0, strict=True)]
    task_count: Annotated[int, Field(ge=4, strict=True)]
    items: tuple[ProposerVisibleFaultFeedbackItem, ...]

    @model_validator(mode="after")
    def exact_visible_shape(self) -> Self:
        if self.parent_correct > self.task_count:
            raise ValueError("parent_correct exceeds the visible roster")
        if self.feedback_mode is ProposerVisibleFeedbackMode.SCORE_ONLY:
            if self.items:
                raise ValueError("score-only proposer view cannot contain item evidence")
        elif len(self.items) != self.task_count:
            raise ValueError("mechanism-visible proposer view must cover the complete roster")
        return self


class ModelAuthoredFaultProposal(ImmutableModel):
    """The sole proposal output and the deterministic invalid-output fallback."""

    schema_version: Literal["1"] = "1"
    feedback_ref: ArtifactRef
    proposer_harness_ref: ArtifactRef
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    execution_status: ExecutionStatus
    raw_model_output_sha256: Sha256 | None
    parsed_action: FaultRepairAction | None
    proposal_valid: bool
    applied_rule_id: RepairRuleId
    invalid_output_fallback_rule: Literal[RepairRuleId.CONSTANT_LEGACY] = (
        RepairRuleId.CONSTANT_LEGACY
    )
    retries_used: Literal[0] = 0
    human_selection_used: Literal[False] = False

    @model_validator(mode="after")
    def exact_media_and_fallback(self) -> Self:
        if self.feedback_ref.media_type != HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE:
            raise ValueError("feedback_ref declares the wrong media type")
        if self.execution_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise ValueError("proposal execution_ref declares the wrong media type")
        if self.outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("proposal outcome_ref declares the wrong media type")
        completed = self.execution_status is ExecutionStatus.COMPLETED
        if completed != (self.raw_model_output_sha256 is not None):
            raise ValueError("proposal raw output hash differs from execution status")
        if self.proposal_valid:
            if self.parsed_action is None or self.applied_rule_id is not self.parsed_action.rule_id:
                raise ValueError("valid proposal must apply its exact parsed model action")
        elif (
            self.parsed_action is not None
            or self.applied_rule_id is not self.invalid_output_fallback_rule
        ):
            raise ValueError("invalid proposal must fail closed to the frozen parent rule")
        return self


class FaultDevelopmentGate(ImmutableModel):
    """Automatic frozen promotion decision over raw model behavior."""

    schema_version: Literal["1"] = "1"
    promotion_rule: PromotionRule
    parent_correct: Annotated[int, Field(ge=0, strict=True)]
    candidate_correct: Annotated[int, Field(ge=0, strict=True)]
    revert_correct: Annotated[int, Field(ge=0, strict=True)]
    placebo_correct: Annotated[int, Field(ge=0, strict=True)]
    protected_parent_correct: Annotated[int, Field(ge=0, strict=True)]
    protected_candidate_correct: Annotated[int, Field(ge=0, strict=True)]
    proposal_valid: bool
    resource_budget_passed: bool
    repair_activation_recall: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    null_overactivation_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    shift_route_accuracy: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    unrepairable_correct_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    parsed_output_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    performance_passed: bool
    mechanism_passed: bool
    decision: PromotionDecision
    selected_harness_ref: ArtifactRef
    automatic: Literal[True] = True

    @model_validator(mode="after")
    def exact_decision(self) -> Self:
        expected_performance = (
            self.proposal_valid
            and self.resource_budget_passed
            and self.candidate_correct > self.parent_correct
            and self.protected_candidate_correct >= self.protected_parent_correct
        )
        if self.performance_passed is not expected_performance:
            raise ValueError("performance gate differs from the frozen development rule")
        required = self.performance_passed and (
            self.promotion_rule is PromotionRule.PERFORMANCE_ONLY or self.mechanism_passed
        )
        expected = PromotionDecision.PROMOTE if required else PromotionDecision.ROLLBACK
        if self.decision is not expected:
            raise ValueError("promotion decision differs from the frozen automatic rule")
        return self


class HarnessFaultModelDrivenDevelopmentResult(ImmutableModel):
    """Complete non-reportable closure for one public development run."""

    schema_version: Literal["1"] = "1"
    kind: Literal["model-output-driven-harness-fault-public-development"] = (
        "model-output-driven-harness-fault-public-development"
    )
    plan: HarnessFaultDevelopmentPlan
    plan_fingerprint: Sha256
    partition_grant: PartitionEvaluationGrant
    runtime_producer_id: Sha256
    parent_harness_ref: ArtifactRef
    compilation_ref: ArtifactRef
    feedback_ref: ArtifactRef
    proposer_harness_ref: ArtifactRef
    proposal: ModelAuthoredFaultProposal
    observations: Annotated[tuple[FaultDevelopmentObservation, ...], Field(min_length=16)]
    gate: FaultDevelopmentGate
    ledger_id: NonEmptyStr
    attempt_budget: AttemptBudget
    ledger_tail_ref: ArtifactRef
    model_call_count: Annotated[int, Field(ge=17, strict=True)]
    model_output_drives_candidate: Literal[True] = True
    raw_model_output_drives_scored_behavior: Literal[True] = True
    deterministic_middleware_behavior_rewrite_used: Literal[False] = False
    one_model_proposal_no_manual_ranking: Literal[True] = True
    exact_nominal_model_spec_shared: Literal[True] = True
    provider_identity_attested: Literal[False] = False
    exact_served_revision_claim_allowed: Literal[False] = False
    exploration_data_only: Literal[True] = True
    sealed_evidence: Literal[False] = False
    reportable_benchmark_result: Literal[False] = False
    confirmatory_inference: Literal[False] = False
    permanently_nonreportable_development_artifact: Literal[True] = True

    @field_validator("observations")
    @classmethod
    def canonical_observations(
        cls, values: tuple[FaultDevelopmentObservation, ...]
    ) -> tuple[FaultDevelopmentObservation, ...]:
        return tuple(sorted(values, key=lambda item: (item.harness_role.value, item.task_id)))

    @model_validator(mode="after")
    def exact_closure_shape(self) -> Self:
        if self.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("plan_fingerprint differs from the embedded plan")
        if self.partition_grant.partition is not ProtocolPartition.EXPLORATION:
            raise ValueError("model-driven development admits exploration data only")
        if self.compilation_ref.media_type != HARNESS_FAULT_COMPILATION_MEDIA_TYPE:
            raise ValueError("compilation_ref declares the wrong media type")
        if self.feedback_ref != self.proposal.feedback_ref:
            raise ValueError("proposal differs from the result feedback ref")
        if self.proposer_harness_ref != self.proposal.proposer_harness_ref:
            raise ValueError("proposal differs from the result proposer harness")
        if self.attempt_budget != self.plan.attempt_budget:
            raise ValueError("attempt budget differs from the exact development plan")
        if self.model_call_count != self.plan.max_model_calls:
            raise ValueError("model call count differs from the exact development topology")
        if len(self.observations) != 4 * self.plan.task_count:
            raise ValueError("observations do not cover the complete attribution quartet")
        for role in (
            HarnessRole.FAULTY_PARENT,
            HarnessRole.CANDIDATE,
            HarnessRole.REVERT,
            HarnessRole.PLACEBO,
        ):
            selected = tuple(item for item in self.observations if item.harness_role is role)
            if len(selected) != self.plan.task_count:
                raise ValueError(f"{role.value} does not cover the complete task roster")
            if len({item.task_id for item in selected}) != self.plan.task_count:
                raise ValueError(f"{role.value} contains duplicate tasks")
        call_outcomes = {self.proposal.outcome_ref.sha256} | {
            item.outcome_ref.sha256 for item in self.observations
        }
        call_executions = {self.proposal.execution_ref.sha256} | {
            item.execution_ref.sha256 for item in self.observations
        }
        if (
            len(call_outcomes) != self.model_call_count
            or len(call_executions) != self.model_call_count
        ):
            raise ValueError("model calls must have unique execution and accounting refs")
        if self.ledger_tail_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("ledger_tail_ref declares the wrong media type")
        return self


__all__ = [
    "HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE",
    "HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE",
    "DevelopmentInvocationMode",
    "FaultDevelopmentGate",
    "FaultDevelopmentObservation",
    "FaultFeedbackItem",
    "FeedbackEvidenceLinkage",
    "HarnessFaultDevelopmentError",
    "HarnessFaultDevelopmentPlan",
    "HarnessFaultFeedbackProjection",
    "HarnessFaultModelDrivenDevelopmentResult",
    "ModelAuthoredFaultProposal",
    "PromotionDecision",
    "ProposerVisibleFaultFeedback",
    "ProposerVisibleFaultFeedbackItem",
    "ProposerVisibleFeedbackMode",
]
