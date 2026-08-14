"""Prospective, non-attested profiles for confirmatory study conditions.

This module separates two designs that both happen to contain four conditions:

* the real-task effectiveness study uses ``PURE``, ``STATIC``, ``SCORE``, and
  ``FULL``; and
* the controlled-fault factorial uses ``SS``, ``MS``, ``SM``, and ``MM``.

The models validate a frozen design only.  They deliberately cannot claim that
the design was executed, that a provider served the declared model, or that a
sealed partition remained hidden.  Runtime closures must establish those facts
from independently verifiable receipts.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import NonEmptyStr, Sha256
from spiral_harness.experiments.confirmatory_resources import (
    AdaptiveExecutionCeilings,
    AdaptiveProtocolCommitments,
    ExAnteAdaptiveTopology,
    ModelMediatedRole,
    ModelRoleCeiling,
    ProspectiveConfirmatoryModel,
)

CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-four-arm-design.v1+json"
)
FAULT_FACTORIAL_DESIGN_MEDIA_TYPE = "application/vnd.spiral-harness.fault-factorial-design.v1+json"
PURE_AT_B_PLAN_MEDIA_TYPE = "application/vnd.spiral-harness.pure-at-b-plan.v1+json"

PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class ConfirmatoryArmProfileError(ValueError):
    """A condition does not match the sole prospective profile for its design."""


class RealTaskArm(StrEnum):
    """The four real-task effectiveness conditions, not factorial cells."""

    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"


class FaultFactorialCell(StrEnum):
    """Feedback-by-promotion cells for the controlled-fault experiment."""

    SS = "SS"
    MS = "MS"
    SM = "SM"
    MM = "MM"


class HarnessMode(StrEnum):
    """Harness exposure permitted in one real-task condition."""

    PROVIDER_MINIMAL = "provider-minimal"
    FROZEN_SEED = "frozen-seed"
    MUTABLE = "mutable"


class SearchMode(StrEnum):
    """Whether a condition executes adaptive harness search."""

    NONE = "none"
    MATCHED_ADAPTIVE = "matched-adaptive"


class OptimizerFeedbackMode(StrEnum):
    """Typed projection disclosed to the condition-local optimizer."""

    NONE = "none"
    SCORE_ONLY = "score-only"
    MECHANISM_VISIBLE = "mechanism-visible"


class PromotionRule(StrEnum):
    """Independent authority rule permitted to change the champion."""

    NONE = "none"
    PERFORMANCE_ONLY = "performance-only"
    MECHANISM_GATED = "mechanism-gated"


class EvidenceComputationMode(StrEnum):
    """Evidence work performed irrespective of what an optimizer can see."""

    NONE = "none"
    FULL_ATTRIBUTION_QUARTET = "full-attribution-quartet"


class SearchBudgetScope(StrEnum):
    """The only honest search-budget statement for a condition."""

    NOT_APPLICABLE = "not-applicable"
    SCORE_FULL_EX_ANTE = "score-full-ex-ante"


class AggregationMethod(StrEnum):
    """Frozen, grader-independent PURE@B aggregation families."""

    MAJORITY_BINARY = "majority-binary"
    NORMALIZED_ARTIFACT_PLURALITY = "normalized-artifact-plurality"
    TASK_NATIVE_FIXED = "task-native-fixed"


_REAL_TASK_ARM_ORDER = (
    RealTaskArm.PURE,
    RealTaskArm.STATIC,
    RealTaskArm.SCORE,
    RealTaskArm.FULL,
)
_FAULT_FACTORIAL_ORDER = (
    FaultFactorialCell.SS,
    FaultFactorialCell.MS,
    FaultFactorialCell.SM,
    FaultFactorialCell.MM,
)
_REAL_TASK_PROFILE = MappingProxyType(
    {
        RealTaskArm.PURE: (
            HarnessMode.PROVIDER_MINIMAL,
            SearchMode.NONE,
            OptimizerFeedbackMode.NONE,
            PromotionRule.NONE,
            EvidenceComputationMode.NONE,
            SearchBudgetScope.NOT_APPLICABLE,
        ),
        RealTaskArm.STATIC: (
            HarnessMode.FROZEN_SEED,
            SearchMode.NONE,
            OptimizerFeedbackMode.NONE,
            PromotionRule.NONE,
            EvidenceComputationMode.NONE,
            SearchBudgetScope.NOT_APPLICABLE,
        ),
        RealTaskArm.SCORE: (
            HarnessMode.MUTABLE,
            SearchMode.MATCHED_ADAPTIVE,
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.PERFORMANCE_ONLY,
            EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET,
            SearchBudgetScope.SCORE_FULL_EX_ANTE,
        ),
        RealTaskArm.FULL: (
            HarnessMode.MUTABLE,
            SearchMode.MATCHED_ADAPTIVE,
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.MECHANISM_GATED,
            EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET,
            SearchBudgetScope.SCORE_FULL_EX_ANTE,
        ),
    }
)

_FAULT_FACTORIAL_PROFILE = MappingProxyType(
    {
        FaultFactorialCell.SS: (
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.MS: (
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.SM: (
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.MECHANISM_GATED,
        ),
        FaultFactorialCell.MM: (
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.MECHANISM_GATED,
        ),
    }
)


def _require_real_task_arm(arm: RealTaskArm | object) -> RealTaskArm:
    if type(arm) is not RealTaskArm:
        raise TypeError("arm must be an exact RealTaskArm")
    return arm


def _require_fault_factorial_cell(cell: FaultFactorialCell | object) -> FaultFactorialCell:
    if type(cell) is not FaultFactorialCell:
        raise TypeError("cell must be an exact FaultFactorialCell")
    return cell


class RealTaskArmProfile(ProspectiveConfirmatoryModel):
    """The exact treatment profile for one real-task effectiveness condition."""

    schema_version: Literal["1"] = "1"
    arm: RealTaskArm
    harness_mode: HarnessMode
    search_mode: SearchMode
    optimizer_feedback: OptimizerFeedbackMode
    promotion_rule: PromotionRule
    evidence_computation: EvidenceComputationMode
    search_budget_scope: SearchBudgetScope

    @model_validator(mode="after")
    def _enforce_exact_real_task_profile(self) -> Self:
        expected = _REAL_TASK_PROFILE[self.arm]
        actual = (
            self.harness_mode,
            self.search_mode,
            self.optimizer_feedback,
            self.promotion_rule,
            self.evidence_computation,
            self.search_budget_scope,
        )
        if actual != expected:
            raise ValueError(f"{self.arm.value} differs from its confirmatory arm profile")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_real_task_arm_profile(arm: RealTaskArm) -> RealTaskArmProfile:
    """Build the sole accepted profile for an exact real-task arm enum."""

    checked = _require_real_task_arm(arm)
    (
        harness_mode,
        search_mode,
        feedback,
        promotion,
        evidence,
        budget_scope,
    ) = _REAL_TASK_PROFILE[checked]
    return RealTaskArmProfile(
        arm=checked,
        harness_mode=harness_mode,
        search_mode=search_mode,
        optimizer_feedback=feedback,
        promotion_rule=promotion,
        evidence_computation=evidence,
        search_budget_scope=budget_scope,
    )


class RealTaskArmPlan(ProspectiveConfirmatoryModel):
    """One condition plus optional adaptive resources.

    PURE and STATIC intentionally have no adaptive topology or search ceiling.
    Their usage is measured, not padded to look search-budget matched.
    """

    schema_version: Literal["1"] = "1"
    profile: RealTaskArmProfile
    adaptive_topology: ExAnteAdaptiveTopology | None = None
    adaptive_ceilings: AdaptiveExecutionCeilings | None = None

    @model_validator(mode="after")
    def _resources_match_the_condition_scope(self) -> Self:
        adaptive = self.profile.search_mode is SearchMode.MATCHED_ADAPTIVE
        supplied = self.adaptive_topology is not None and self.adaptive_ceilings is not None
        partially_supplied = (self.adaptive_topology is None) != (self.adaptive_ceilings is None)
        if partially_supplied:
            raise ValueError("adaptive topology and ceilings must be supplied together")
        if adaptive and not supplied:
            raise ValueError("SCORE and FULL require adaptive topology and ceilings")
        if not adaptive and supplied:
            raise ValueError("PURE and STATIC must not claim adaptive search resources")
        if (
            adaptive
            and self.adaptive_topology is not None
            and self.adaptive_topology.condition_context_count != 2
        ):
            raise ValueError("real-task adaptive plans require exactly two condition contexts")
        return self

    @property
    def arm(self) -> RealTaskArm:
        return self.profile.arm


class ConfirmatoryFourArmDesign(ProspectiveConfirmatoryModel):
    """Structural real-task plan with a matched adaptive SCORE/FULL pair."""

    schema_version: Literal["1"] = "1"
    protocol_version: Literal["confirmatory-real-task-four-arm-v1"] = (
        "confirmatory-real-task-four-arm-v1"
    )
    arms: Annotated[tuple[RealTaskArmPlan, ...], Field(min_length=4, max_length=4)]
    primary_policy_contrast: Literal["FULL-minus-SCORE"] = "FULL-minus-SCORE"
    full_score_treatment: Literal["joint-feedback-and-promotion-policy"] = (
        "joint-feedback-and-promotion-policy"
    )
    full_score_treatment_fields: tuple[
        Literal["optimizer_feedback"],
        Literal["promotion_rule"],
    ] = ("optimizer_feedback", "promotion_rule")
    full_score_budget_match: Literal[
        "equal-ex-ante-commitments-topology-call-attempt-and-token-ceilings"
    ] = "equal-ex-ante-commitments-topology-call-attempt-and-token-ceilings"
    pure_static_budget_statement: Literal["not-search-budget-matched"] = "not-search-budget-matched"
    component_mediation_identified: Literal[False] = False
    profile_validated: Literal[True] = True
    execution_attested: Literal[False] = False
    runtime_topology_attested: Literal[False] = False
    runtime_commitments_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @field_validator("arms")
    @classmethod
    def _canonicalize_real_task_arms(
        cls,
        values: tuple[RealTaskArmPlan, ...],
    ) -> tuple[RealTaskArmPlan, ...]:
        by_arm = {value.arm: value for value in values}
        if len(by_arm) != len(values):
            raise ValueError("real-task design must not contain duplicate arms")
        if frozenset(by_arm) != frozenset(_REAL_TASK_ARM_ORDER):
            raise ValueError("real-task design requires exactly PURE, STATIC, SCORE, and FULL")
        return tuple(by_arm[arm] for arm in _REAL_TASK_ARM_ORDER)

    @model_validator(mode="after")
    def _enforce_matched_adaptive_pair(self) -> Self:
        score = self.arm(RealTaskArm.SCORE)
        full = self.arm(RealTaskArm.FULL)
        if score.adaptive_topology != full.adaptive_topology:
            raise ValueError("SCORE and FULL ex-ante topologies differ")
        if score.adaptive_ceilings != full.adaptive_ceilings:
            raise ValueError("SCORE and FULL ex-ante ceilings differ")
        if score.adaptive_topology is None:  # pragma: no cover - arm invariant
            raise ValueError("matched adaptive topology is missing")
        if score.adaptive_topology.condition_context_count != 2:
            raise ValueError("real-task adaptive topology must isolate exactly SCORE and FULL")

        score_profile = score.profile.model_dump(
            mode="python",
            exclude={"arm", "optimizer_feedback", "promotion_rule"},
            round_trip=True,
            warnings="none",
        )
        full_profile = full.profile.model_dump(
            mode="python",
            exclude={"arm", "optimizer_feedback", "promotion_rule"},
            round_trip=True,
            warnings="none",
        )
        if score_profile != full_profile:
            raise ValueError("SCORE and FULL differ outside the joint treatment fields")
        if (
            score.profile.optimizer_feedback is not OptimizerFeedbackMode.SCORE_ONLY
            or full.profile.optimizer_feedback is not OptimizerFeedbackMode.MECHANISM_VISIBLE
            or score.profile.promotion_rule is not PromotionRule.PERFORMANCE_ONLY
            or full.profile.promotion_rule is not PromotionRule.MECHANISM_GATED
        ):
            raise ValueError("FULL-minus-SCORE must preserve the joint treatment")
        return self

    def arm(self, arm: RealTaskArm) -> RealTaskArmPlan:
        checked = _require_real_task_arm(arm)
        return next(plan for plan in self.arms if plan.arm is checked)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_confirmatory_four_arm_design(
    *,
    adaptive_topology: ExAnteAdaptiveTopology,
    adaptive_ceilings: AdaptiveExecutionCeilings,
) -> ConfirmatoryFourArmDesign:
    """Construct the exact real-task profiles around one SCORE/FULL match."""

    topology = ExAnteAdaptiveTopology.model_validate(adaptive_topology, strict=True)
    ceilings = AdaptiveExecutionCeilings.model_validate(adaptive_ceilings, strict=True)
    plans = []
    for arm in _REAL_TASK_ARM_ORDER:
        profile = make_real_task_arm_profile(arm)
        adaptive = profile.search_mode is SearchMode.MATCHED_ADAPTIVE
        plans.append(
            RealTaskArmPlan(
                profile=profile,
                adaptive_topology=topology if adaptive else None,
                adaptive_ceilings=ceilings if adaptive else None,
            )
        )
    return ConfirmatoryFourArmDesign(arms=tuple(plans))


class PureAtBAggregationRule(ProspectiveConfirmatoryModel):
    """Total, nonadaptive aggregation fixed without grader or sealed feedback."""

    schema_version: Literal["1"] = "1"
    rule_id: NonEmptyStr
    method: AggregationMethod
    implementation_fingerprint: Sha256
    normalizer_fingerprint: Sha256
    output_domain_fingerprint: Sha256
    tie_breaker: Literal["first-sample-in-frozen-order"] = "first-sample-in-frozen-order"
    failed_sample_policy: Literal["abstain-and-all-abstain-select-first"] = (
        "abstain-and-all-abstain-select-first"
    )
    adaptive: Literal[False] = False
    grader_feedback_used: Literal[False] = False
    sealed_feedback_used: Literal[False] = False
    total_for_every_output_multiset: Literal[True] = True

    @field_validator("rule_id", mode="before")
    @classmethod
    def _rule_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("PURE@B rule_id must be exact and non-empty")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class PureAtBPlan(ProspectiveConfirmatoryModel):
    """Separate FULL-budget-matched bare-model sampling reference."""

    schema_version: Literal["1"] = "1"
    four_arm_design: ConfirmatoryFourArmDesign
    four_arm_design_fingerprint: Sha256
    aggregation: PureAtBAggregationRule
    model_spec_fingerprint: Sha256
    solver_config_fingerprint: Sha256
    sample_seed_schedule_fingerprint: Sha256
    pure_sample_count_ceiling: PositiveInt
    pure_model_call_ceiling: PositiveInt
    pure_token_ceiling: PositiveInt
    max_attempts_per_sample: PositiveInt
    pure_model_attempt_ceiling: PositiveInt
    pure_attempt_token_ceiling: PositiveInt
    budget_match_scope: Literal["FULL-ex-ante-logical-call-provider-attempt-and-token-ceilings"] = (
        "FULL-ex-ante-logical-call-provider-attempt-and-token-ceilings"
    )
    same_solver_configuration_required: Literal[True] = True
    independent_samples_required: Literal[True] = True
    sample_independence_definition: Literal[
        "distinct-precommitted-seeds-with-no-cross-sample-mutable-state"
    ] = "distinct-precommitted-seeds-with-no-cross-sample-mutable-state"
    hidden_grader_best_of_k_permitted: Literal[False] = False
    same_solver_configuration_attested: Literal[False] = False
    sample_independence_attested: Literal[False] = False
    aggregation_execution_attested: Literal[False] = False
    execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_full_ex_ante_budget(self) -> Self:
        if self.four_arm_design_fingerprint != self.four_arm_design.fingerprint:
            raise ValueError("PURE@B design fingerprint differs from its four-arm design")
        full = self.four_arm_design.arm(RealTaskArm.FULL)
        ceilings = full.adaptive_ceilings
        topology = full.adaptive_topology
        if ceilings is None or topology is None:  # pragma: no cover - four-arm invariant
            raise ValueError("FULL adaptive topology or ceilings are missing")
        commitments = topology.protocol_commitments
        if self.model_spec_fingerprint != commitments.model_spec_fingerprint:
            raise ValueError("PURE@B model-spec fingerprint differs from FULL")
        if self.solver_config_fingerprint != commitments.solver_config_fingerprint:
            raise ValueError("PURE@B solver configuration differs from FULL")
        if self.pure_model_call_ceiling != ceilings.max_total_model_calls:
            raise ValueError("PURE@B model-call ceiling differs from FULL")
        if self.pure_sample_count_ceiling != self.pure_model_call_ceiling:
            raise ValueError("each PURE@B sample must consume one model-call slot")
        if self.pure_token_ceiling != ceilings.max_total_tokens:
            raise ValueError("PURE@B token ceiling differs from FULL")
        if self.max_attempts_per_sample != ceilings.max_attempts_per_model_call:
            raise ValueError("PURE@B per-sample attempt ceiling differs from FULL")
        if self.pure_model_attempt_ceiling != ceilings.max_total_model_attempts:
            raise ValueError("PURE@B provider-attempt ceiling differs from FULL")
        if self.pure_attempt_token_ceiling != ceilings.max_total_attempt_tokens:
            raise ValueError("PURE@B attempt-token ceiling differs from FULL")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_pure_at_b_plan(
    *,
    four_arm_design: ConfirmatoryFourArmDesign,
    aggregation: PureAtBAggregationRule,
    sample_seed_schedule_fingerprint: Sha256,
) -> PureAtBPlan:
    """Derive PURE@B ceilings from FULL rather than accepting a second budget."""

    design = ConfirmatoryFourArmDesign.model_validate(four_arm_design, strict=True)
    rule = PureAtBAggregationRule.model_validate(aggregation, strict=True)
    full = design.arm(RealTaskArm.FULL)
    ceilings = full.adaptive_ceilings
    topology = full.adaptive_topology
    if ceilings is None or topology is None:  # pragma: no cover - four-arm invariant
        raise ConfirmatoryArmProfileError("FULL adaptive topology or ceilings are missing")
    commitments = topology.protocol_commitments
    return PureAtBPlan(
        four_arm_design=design,
        four_arm_design_fingerprint=design.fingerprint,
        aggregation=rule,
        model_spec_fingerprint=commitments.model_spec_fingerprint,
        solver_config_fingerprint=commitments.solver_config_fingerprint,
        sample_seed_schedule_fingerprint=sample_seed_schedule_fingerprint,
        pure_sample_count_ceiling=ceilings.max_total_model_calls,
        pure_model_call_ceiling=ceilings.max_total_model_calls,
        pure_token_ceiling=ceilings.max_total_tokens,
        max_attempts_per_sample=ceilings.max_attempts_per_model_call,
        pure_model_attempt_ceiling=ceilings.max_total_model_attempts,
        pure_attempt_token_ceiling=ceilings.max_total_attempt_tokens,
    )


class FaultFactorialProfile(ProspectiveConfirmatoryModel):
    """One controlled-fault cell; not a PURE/STATIC effectiveness reference."""

    schema_version: Literal["1"] = "1"
    cell: FaultFactorialCell
    optimizer_feedback: OptimizerFeedbackMode
    promotion_rule: PromotionRule
    harness_mode: Literal[HarnessMode.MUTABLE] = HarnessMode.MUTABLE
    search_mode: Literal[SearchMode.MATCHED_ADAPTIVE] = SearchMode.MATCHED_ADAPTIVE
    evidence_computation: Literal[EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET] = (
        EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET
    )

    @model_validator(mode="after")
    def _enforce_exact_factorial_profile(self) -> Self:
        if (self.optimizer_feedback, self.promotion_rule) != _FAULT_FACTORIAL_PROFILE[self.cell]:
            raise ValueError(f"{self.cell.value} differs from its factorial profile")
        return self


def make_fault_factorial_profile(cell: FaultFactorialCell) -> FaultFactorialProfile:
    """Build one exact SS/MS/SM/MM cell and reject real-task arm enums."""

    checked = _require_fault_factorial_cell(cell)
    feedback, promotion = _FAULT_FACTORIAL_PROFILE[checked]
    return FaultFactorialProfile(
        cell=checked,
        optimizer_feedback=feedback,
        promotion_rule=promotion,
    )


class FaultFactorialCellPlan(ProspectiveConfirmatoryModel):
    """One factorial treatment applied to a shared ex-ante topology and budget."""

    schema_version: Literal["1"] = "1"
    profile: FaultFactorialProfile
    adaptive_topology: ExAnteAdaptiveTopology
    adaptive_ceilings: AdaptiveExecutionCeilings

    @model_validator(mode="after")
    def _require_four_condition_contexts(self) -> Self:
        if self.adaptive_topology.condition_context_count != 4:
            raise ValueError("fault factorial topology must isolate exactly four conditions")
        return self

    @property
    def cell(self) -> FaultFactorialCell:
        return self.profile.cell


class FaultFactorialDesign(ProspectiveConfirmatoryModel):
    """Structural 2x2 design with all treatment cells ex-ante matched."""

    schema_version: Literal["1"] = "1"
    protocol_version: Literal["controlled-fault-feedback-promotion-factorial-v1"] = (
        "controlled-fault-feedback-promotion-factorial-v1"
    )
    cells: Annotated[tuple[FaultFactorialCellPlan, ...], Field(min_length=4, max_length=4)]
    conditional_guidance_effect: Literal["MS-minus-SS"] = "MS-minus-SS"
    conditional_promotion_effect: Literal["SM-minus-SS"] = "SM-minus-SS"
    interaction_estimand: Literal["(MM-minus-SM)-minus-(MS-minus-SS)"] = (
        "(MM-minus-SM)-minus-(MS-minus-SS)"
    )
    adaptive_policy_intention_to_treat: Literal[True] = True
    same_candidate_mediation_claimed: Literal[False] = False
    profile_validated: Literal[True] = True
    execution_attested: Literal[False] = False
    runtime_topology_attested: Literal[False] = False
    runtime_commitments_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @field_validator("cells")
    @classmethod
    def _canonicalize_factorial_cells(
        cls,
        values: tuple[FaultFactorialCellPlan, ...],
    ) -> tuple[FaultFactorialCellPlan, ...]:
        by_cell = {value.cell: value for value in values}
        if len(by_cell) != len(values):
            raise ValueError("factorial design must not contain duplicate cells")
        if frozenset(by_cell) != frozenset(_FAULT_FACTORIAL_ORDER):
            raise ValueError("factorial design requires exactly SS, MS, SM, and MM")
        return tuple(by_cell[cell] for cell in _FAULT_FACTORIAL_ORDER)

    @model_validator(mode="after")
    def _enforce_shared_non_treatment_coordinates(self) -> Self:
        anchor = self.cells[0]
        if anchor.adaptive_topology.condition_context_count != 4:
            raise ValueError("fault factorial topology must isolate exactly four conditions")
        for cell in self.cells[1:]:
            if cell.adaptive_topology != anchor.adaptive_topology:
                raise ValueError("factorial cells have different ex-ante topologies")
            if cell.adaptive_ceilings != anchor.adaptive_ceilings:
                raise ValueError("factorial cells have different ex-ante ceilings")
        return self

    def cell(self, cell: FaultFactorialCell) -> FaultFactorialCellPlan:
        checked = _require_fault_factorial_cell(cell)
        return next(plan for plan in self.cells if plan.cell is checked)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_fault_factorial_design(
    *,
    adaptive_topology: ExAnteAdaptiveTopology,
    adaptive_ceilings: AdaptiveExecutionCeilings,
) -> FaultFactorialDesign:
    """Construct all four controlled-fault treatment cells around one plan."""

    topology = ExAnteAdaptiveTopology.model_validate(adaptive_topology, strict=True)
    ceilings = AdaptiveExecutionCeilings.model_validate(adaptive_ceilings, strict=True)
    return FaultFactorialDesign(
        cells=tuple(
            FaultFactorialCellPlan(
                profile=make_fault_factorial_profile(cell),
                adaptive_topology=topology,
                adaptive_ceilings=ceilings,
            )
            for cell in _FAULT_FACTORIAL_ORDER
        )
    )


__all__ = [
    "CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE",
    "FAULT_FACTORIAL_DESIGN_MEDIA_TYPE",
    "PURE_AT_B_PLAN_MEDIA_TYPE",
    "AdaptiveExecutionCeilings",
    "AdaptiveProtocolCommitments",
    "AggregationMethod",
    "ConfirmatoryArmProfileError",
    "ConfirmatoryFourArmDesign",
    "EvidenceComputationMode",
    "ExAnteAdaptiveTopology",
    "FaultFactorialCell",
    "FaultFactorialCellPlan",
    "FaultFactorialDesign",
    "FaultFactorialProfile",
    "HarnessMode",
    "ModelMediatedRole",
    "ModelRoleCeiling",
    "OptimizerFeedbackMode",
    "PromotionRule",
    "PureAtBAggregationRule",
    "PureAtBPlan",
    "RealTaskArm",
    "RealTaskArmPlan",
    "RealTaskArmProfile",
    "SearchBudgetScope",
    "SearchMode",
    "make_confirmatory_four_arm_design",
    "make_fault_factorial_design",
    "make_fault_factorial_profile",
    "make_pure_at_b_plan",
    "make_real_task_arm_profile",
]
