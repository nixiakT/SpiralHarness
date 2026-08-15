"""Contracts for exact descriptive aggregation of the public BFCL campaign.

The three search seeds repeat the same eight public-development tasks.  These
contracts therefore preserve task clusters and never represent the 24
task-by-seed cells as independent inferential units.
"""

from __future__ import annotations

from math import gcd
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
    BfclV4PublicPilotCampaign,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BFCL_V4_PILOT_OUTER_SEEDS_U64,
    BfclV4PilotArm,
    BfclV4PilotOuterSeed,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    BfclV4RunClosureVerification,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import AttemptBudget, FrozenModelSpec
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BFCL_V4_HOLDOUT_TASK_IDS,
    BFCL_V4_METRIC_ARM_ORDER,
    BFCL_V4_PAIRED_CONTRASTS,
    BfclV4JointSelectionDecision,
    BfclV4PublicDescriptiveMetrics,
    BfclV4SelectionDecision,
)

BFCL_V4_CAMPAIGN_REPLICATE_COUNT = 3
BFCL_V4_CAMPAIGN_TASK_CLUSTER_COUNT = 8
BFCL_V4_CAMPAIGN_TASK_SEED_CELL_COUNT = 24
BFCL_V4_CAMPAIGN_ADAPTIVE_ARM_ORDER = (
    BfclV4PilotArm.SCORE,
    BfclV4PilotArm.FULL,
)


def _reduced_pair(numerator: int, denominator: int) -> tuple[int, int]:
    if denominator <= 0:
        raise ValueError("exact rational denominator must be positive")
    divisor = gcd(abs(numerator), denominator)
    return numerator // divisor, denominator // divisor


class BfclV4ExactRational(ImmutableModel):
    """A canonical reduced fraction; no binary floating-point value is stored."""

    numerator: Annotated[int, Field(strict=True)]
    denominator: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def _require_reduced_form(self) -> Self:
        if (self.numerator, self.denominator) != _reduced_pair(self.numerator, self.denominator):
            raise ValueError("exact rational must use canonical reduced form")
        return self

    @classmethod
    def from_ratio(cls, numerator: int, denominator: int) -> BfclV4ExactRational:
        reduced_numerator, reduced_denominator = _reduced_pair(numerator, denominator)
        return cls(numerator=reduced_numerator, denominator=reduced_denominator)


class BfclV4CampaignPerCallBudget(ImmutableModel):
    """Common per-call and complete 100-call ledger ceilings for each replicate."""

    schema_version: Literal["1"] = "1"
    provider_attempts_per_call: Literal[1] = 1
    max_output_tokens: Annotated[int, Field(ge=1, strict=True)]
    attempt_budget: AttemptBudget
    attempt_budget_fingerprint: Sha256
    automatic_retries_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_ledger_budget(self) -> Self:
        if (
            self.attempt_budget_fingerprint != self.attempt_budget.fingerprint
            or self.attempt_budget.max_attempts != 100
            or self.attempt_budget.max_total_tokens
            != 100 * self.attempt_budget.max_tokens_per_attempt
            or self.max_output_tokens > self.attempt_budget.max_tokens_per_attempt
        ):
            raise ValueError("campaign budget must bind 100 complete equal per-call ceilings")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CampaignReplicateAnalysisInput(ImmutableModel):
    """One completed replicate projected onto portable offline evidence."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=3, strict=True)]
    replicate_id: NonEmptyStr
    outer_seed_u64: BfclV4PilotOuterSeed
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    model_spec_fingerprint: Sha256
    backend_fingerprint: NonEmptyStr
    inference_fingerprint: Sha256
    attempt_budget: AttemptBudget
    attempt_budget_fingerprint: Sha256
    per_call_budget_fingerprint: Sha256
    closure_ref: ArtifactRef
    closure_verification: BfclV4RunClosureVerification
    joint_selection: BfclV4JointSelectionDecision
    joint_selection_fingerprint: Sha256
    descriptive_metrics: BfclV4PublicDescriptiveMetrics
    descriptive_metrics_fingerprint: Sha256
    provider_identity_observation_count: Annotated[int, Field(ge=0, le=100, strict=True)] = 0
    provider_declared_identity_consistent: bool = False
    provider_served_same_weights_attested: Literal[False] = False
    completed_model_calls: Literal[100] = 100
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_offline_evidence(self) -> Self:
        if self.attempt_budget_fingerprint != self.attempt_budget.fingerprint:
            raise ValueError("replicate attempt-budget fingerprint changed")
        if self.provider_declared_identity_consistent and not (
            self.provider_identity_observation_count
        ):
            raise ValueError("provider identity consistency requires an observed response")
        if self.closure_ref.media_type != BFCL_V4_RUN_CLOSURE_MEDIA_TYPE:
            raise ValueError("campaign replicate closure ref has the wrong media type")
        if (
            self.closure_ref.sha256 != self.closure_verification.closure_fingerprint
            or self.closure_verification.plan_fingerprint != self.plan_fingerprint
        ):
            raise ValueError("campaign replicate closure verification lineage changed")
        if (
            self.joint_selection.plan_fingerprint != self.plan_fingerprint
            or self.joint_selection.schedule_content_sha256 != self.schedule_content_sha256
            or self.joint_selection_fingerprint != self.joint_selection.fingerprint
        ):
            raise ValueError("campaign replicate joint-selection lineage changed")
        if (
            self.descriptive_metrics.plan_fingerprint != self.plan_fingerprint
            or self.descriptive_metrics.schedule_content_sha256 != self.schedule_content_sha256
            or self.descriptive_metrics.joint_selection_decision_fingerprint
            != self.joint_selection_fingerprint
            or self.descriptive_metrics_fingerprint != self.descriptive_metrics.fingerprint
        ):
            raise ValueError("campaign replicate descriptive-metrics lineage changed")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicCampaignAnalysisInput(ImmutableModel):
    """Exact three-seed input, including common execution and closure bindings."""

    schema_version: Literal["1"] = "1"
    campaign: BfclV4PublicPilotCampaign
    campaign_fingerprint: Literal[BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT]
    model_spec: FrozenModelSpec
    model_spec_fingerprint: Sha256
    backend: NonEmptyStr
    backend_fingerprint: NonEmptyStr
    inference_fingerprint: Sha256
    per_call_budget: BfclV4CampaignPerCallBudget
    per_call_budget_fingerprint: Sha256
    ordered_closure_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=3, max_length=3)]
    replicates: Annotated[
        tuple[BfclV4CampaignReplicateAnalysisInput, ...],
        Field(min_length=3, max_length=3),
    ]
    exact_seed_order_required: Literal[True] = True
    same_model_backend_inference_and_budget_required: Literal[True] = True
    all_three_hundred_call_closures_required: Literal[True] = True
    all_call_response_identity_coverage_complete: bool = False
    provider_identity_observation_summaries_bound: Literal[True] = True
    provider_declared_identity_coordinates_not_exposed: Literal[True] = True
    provider_declared_identity_cross_replicate_consistency_attested: Literal[False] = False
    same_provider_weights_attested: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_campaign_input(self) -> Self:
        if (
            self.campaign.fingerprint != BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT
            or self.campaign_fingerprint != self.campaign.fingerprint
        ):
            raise ValueError("analysis input differs from the frozen campaign fingerprint")
        if (
            self.model_spec_fingerprint != self.model_spec.fingerprint
            or self.backend != self.model_spec.backend
            or self.backend_fingerprint != self.model_spec.backend_fingerprint
            or self.inference_fingerprint != self.model_spec.inference_fingerprint
        ):
            raise ValueError("analysis input model, backend, or inference binding changed")
        if (
            self.per_call_budget_fingerprint != self.per_call_budget.fingerprint
            or self.per_call_budget.max_output_tokens != self.model_spec.inference.max_output_tokens
        ):
            raise ValueError("analysis input per-call budget differs from frozen inference")

        expected_coordinates = tuple(
            (
                item.ordinal,
                item.replicate_id,
                item.outer_seed_u64,
                item.call_plan.fingerprint,
                item.call_plan.schedule_content_sha256,
            )
            for item in self.campaign.replicates
        )
        observed_coordinates = tuple(
            (
                item.ordinal,
                item.replicate_id,
                item.outer_seed_u64,
                item.plan_fingerprint,
                item.schedule_content_sha256,
            )
            for item in self.replicates
        )
        if observed_coordinates != expected_coordinates:
            raise ValueError("analysis replicate seed, order, or plan binding changed")
        if tuple(item.replicate_id for item in self.replicates) != (
            BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS
        ) or tuple(item.outer_seed_u64 for item in self.replicates) != (
            BFCL_V4_PILOT_OUTER_SEEDS_U64
        ):
            raise ValueError("analysis replicate roster differs from registration")

        common = (
            self.model_spec_fingerprint,
            self.backend_fingerprint,
            self.inference_fingerprint,
            self.per_call_budget.attempt_budget_fingerprint,
            self.per_call_budget_fingerprint,
        )
        if any(
            (
                item.model_spec_fingerprint,
                item.backend_fingerprint,
                item.inference_fingerprint,
                item.attempt_budget_fingerprint,
                item.per_call_budget_fingerprint,
            )
            != common
            for item in self.replicates
        ):
            raise ValueError("analysis replicates do not share one execution identity and budget")
        if any(
            item.attempt_budget != self.per_call_budget.attempt_budget for item in self.replicates
        ):
            raise ValueError("analysis replicates do not share the complete attempt budget")
        complete_identity_coverage = all(
            item.provider_identity_observation_count == 100 for item in self.replicates
        )
        if self.all_call_response_identity_coverage_complete is not complete_identity_coverage:
            raise ValueError("provider identity coverage flag differs from replicate summaries")
        actual_closures = tuple(item.closure_ref for item in self.replicates)
        if self.ordered_closure_refs != actual_closures:
            raise ValueError("analysis closure refs differ from exact replicate order")
        if len({item.sha256 for item in actual_closures}) != 3:
            raise ValueError("analysis requires three distinct 100-call closure refs")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CampaignArmTaskCluster(ImmutableModel):
    """Three stochastic outcomes retained inside one public task cluster."""

    arm: BfclV4PilotArm
    task_id: NonEmptyStr
    replicate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    outer_seeds_u64: tuple[BfclV4PilotOuterSeed, BfclV4PilotOuterSeed, BfclV4PilotOuterSeed]
    correctness: tuple[bool, bool, bool]
    correct_replicates: Annotated[int, Field(ge=0, le=3, strict=True)]
    stochastic_replicate_count: Literal[3] = 3

    @model_validator(mode="after")
    def _close_cluster(self) -> Self:
        if self.arm not in BFCL_V4_METRIC_ARM_ORDER or self.task_id not in (
            BFCL_V4_HOLDOUT_TASK_IDS
        ):
            raise ValueError("arm task cluster is outside the frozen matrix")
        if self.replicate_ids != BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS or (
            self.outer_seeds_u64 != BFCL_V4_PILOT_OUTER_SEEDS_U64
        ):
            raise ValueError("arm task cluster replicate order changed")
        if self.correct_replicates != sum(self.correctness):
            raise ValueError("arm task cluster count differs from its correctness vector")
        return self


class BfclV4CampaignArmAggregate(ImmutableModel):
    """One arm's exact 8-cluster by 3-replicate descriptive matrix."""

    arm: BfclV4PilotArm
    task_clusters: Annotated[
        tuple[BfclV4CampaignArmTaskCluster, ...], Field(min_length=8, max_length=8)
    ]
    correct_task_seed_cells: Annotated[int, Field(ge=0, le=24, strict=True)]
    accuracy: BfclV4ExactRational
    task_cluster_count: Literal[8] = 8
    stochastic_replicates_per_task: Literal[3] = 3
    task_seed_cell_count: Literal[24] = 24

    @model_validator(mode="after")
    def _close_arm(self) -> Self:
        if self.arm not in BFCL_V4_METRIC_ARM_ORDER:
            raise ValueError("campaign arm aggregate uses an unknown arm")
        if tuple(item.task_id for item in self.task_clusters) != BFCL_V4_HOLDOUT_TASK_IDS or any(
            item.arm is not self.arm for item in self.task_clusters
        ):
            raise ValueError("campaign arm aggregate differs from the frozen task order")
        correct = sum(item.correct_replicates for item in self.task_clusters)
        if self.correct_task_seed_cells != correct or self.accuracy != (
            BfclV4ExactRational.from_ratio(correct, 24)
        ):
            raise ValueError("campaign arm accuracy differs from its exact binary matrix")
        return self


class BfclV4CampaignPromotionAggregate(ImmutableModel):
    """Promotion frequency across the three stochastic search replicates."""

    arm: BfclV4PilotArm
    replicate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    decisions: tuple[BfclV4SelectionDecision, BfclV4SelectionDecision, BfclV4SelectionDecision]
    promotion_count: Annotated[int, Field(ge=0, le=3, strict=True)]
    rollback_count: Annotated[int, Field(ge=0, le=3, strict=True)]
    stochastic_search_replicates: Literal[3] = 3
    descriptive_count_only: Literal[True] = True

    @model_validator(mode="after")
    def _close_promotions(self) -> Self:
        if self.arm not in BFCL_V4_CAMPAIGN_ADAPTIVE_ARM_ORDER:
            raise ValueError("promotion aggregate is only defined for SCORE and FULL")
        if self.replicate_ids != BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS:
            raise ValueError("promotion aggregate replicate order changed")
        promoted = sum(item is BfclV4SelectionDecision.PROMOTE for item in self.decisions)
        if (self.promotion_count, self.rollback_count) != (promoted, 3 - promoted):
            raise ValueError("promotion counts differ from exact selection decisions")
        return self


class BfclV4CampaignPairedTaskCluster(ImmutableModel):
    """Paired win/tie/loss outcomes for three repeats of one task."""

    treatment_arm: BfclV4PilotArm
    reference_arm: BfclV4PilotArm
    task_id: NonEmptyStr
    treatment_correctness: tuple[bool, bool, bool]
    reference_correctness: tuple[bool, bool, bool]
    wins: Annotated[int, Field(ge=0, le=3, strict=True)]
    ties: Annotated[int, Field(ge=0, le=3, strict=True)]
    losses: Annotated[int, Field(ge=0, le=3, strict=True)]
    exact_delta: BfclV4ExactRational
    stochastic_replicate_count: Literal[3] = 3

    @model_validator(mode="after")
    def _close_paired_cluster(self) -> Self:
        if (self.treatment_arm, self.reference_arm) not in BFCL_V4_PAIRED_CONTRASTS or (
            self.task_id not in BFCL_V4_HOLDOUT_TASK_IDS
        ):
            raise ValueError("paired task cluster is outside the frozen contrasts")
        wins = sum(
            left and not right
            for left, right in zip(
                self.treatment_correctness, self.reference_correctness, strict=True
            )
        )
        losses = sum(
            right and not left
            for left, right in zip(
                self.treatment_correctness, self.reference_correctness, strict=True
            )
        )
        expected = (wins, 3 - wins - losses, losses)
        if (self.wins, self.ties, self.losses) != expected or self.exact_delta != (
            BfclV4ExactRational.from_ratio(wins - losses, 3)
        ):
            raise ValueError("paired task-cluster summary differs from binary vectors")
        return self


class BfclV4CampaignPairedAggregate(ImmutableModel):
    """Exact descriptive paired effect, clustered by task and never inferential."""

    treatment_arm: BfclV4PilotArm
    reference_arm: BfclV4PilotArm
    task_clusters: Annotated[
        tuple[BfclV4CampaignPairedTaskCluster, ...], Field(min_length=8, max_length=8)
    ]
    wins: Annotated[int, Field(ge=0, le=24, strict=True)]
    ties: Annotated[int, Field(ge=0, le=24, strict=True)]
    losses: Annotated[int, Field(ge=0, le=24, strict=True)]
    exact_delta: BfclV4ExactRational
    descriptive_threshold: BfclV4ExactRational = Field(
        default_factory=lambda: BfclV4ExactRational(numerator=1, denominator=10)
    )
    strictly_exceeds_positive_ten_percentage_points: bool
    task_cluster_count: Literal[8] = 8
    stochastic_replicates_per_task: Literal[3] = 3
    task_seed_cell_count: Literal[24] = 24
    task_seed_cells_are_independent_inferential_units: Literal[False] = False
    descriptive_flag_only: Literal[True] = True
    confidence_interval_available: Literal[False] = False
    statistical_significance_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _close_paired_aggregate(self) -> Self:
        contrast = (self.treatment_arm, self.reference_arm)
        if contrast not in BFCL_V4_PAIRED_CONTRASTS:
            raise ValueError("campaign paired aggregate uses an unknown contrast")
        if tuple(item.task_id for item in self.task_clusters) != BFCL_V4_HOLDOUT_TASK_IDS or any(
            (item.treatment_arm, item.reference_arm) != contrast for item in self.task_clusters
        ):
            raise ValueError("campaign paired aggregate differs from task-cluster order")
        expected = (
            sum(item.wins for item in self.task_clusters),
            sum(item.ties for item in self.task_clusters),
            sum(item.losses for item in self.task_clusters),
        )
        if (self.wins, self.ties, self.losses) != expected or sum(expected) != 24:
            raise ValueError("campaign paired counts differ from exact task clusters")
        if self.exact_delta != BfclV4ExactRational.from_ratio(self.wins - self.losses, 24):
            raise ValueError("campaign exact delta differs from paired counts")
        if self.descriptive_threshold != BfclV4ExactRational(numerator=1, denominator=10):
            raise ValueError("campaign descriptive threshold must be exactly one tenth")
        exceeds = self.exact_delta.numerator * 10 > self.exact_delta.denominator
        if self.strictly_exceeds_positive_ten_percentage_points is not exceeds:
            raise ValueError("campaign >10 pp flag differs from exact rational comparison")
        return self


class BfclV4PublicCampaignDescriptiveAnalysis(ImmutableModel):
    """Portable output with provenance and hard public-development claim limits."""

    schema_version: Literal["1"] = "1"
    analysis_input_fingerprint: Sha256
    campaign_fingerprint: Literal[BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT]
    replicate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    outer_seeds_u64: tuple[BfclV4PilotOuterSeed, BfclV4PilotOuterSeed, BfclV4PilotOuterSeed]
    plan_fingerprints: tuple[Sha256, Sha256, Sha256]
    schedule_content_sha256s: tuple[Sha256, Sha256, Sha256]
    model_spec_fingerprint: Sha256
    backend_fingerprint: NonEmptyStr
    inference_fingerprint: Sha256
    attempt_budget_fingerprint: Sha256
    per_call_budget_fingerprint: Sha256
    ordered_closure_refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef]
    joint_selection_fingerprints: tuple[Sha256, Sha256, Sha256]
    descriptive_metrics_fingerprints: tuple[Sha256, Sha256, Sha256]
    provider_identity_observation_counts: Annotated[
        tuple[Annotated[int, Field(ge=0, le=100, strict=True)], ...],
        Field(min_length=3, max_length=3),
    ]
    provider_declared_identity_consistent_within_replicates: tuple[bool, bool, bool]
    arms: Annotated[tuple[BfclV4CampaignArmAggregate, ...], Field(min_length=5, max_length=5)]
    promotions: Annotated[
        tuple[BfclV4CampaignPromotionAggregate, ...], Field(min_length=2, max_length=2)
    ]
    paired_deltas: Annotated[
        tuple[BfclV4CampaignPairedAggregate, ...], Field(min_length=8, max_length=8)
    ]
    task_cluster_count: Literal[8] = 8
    stochastic_search_replicates: Literal[3] = 3
    task_seed_cell_count: Literal[24] = 24
    task_seed_cells_are_independent_inferential_units: Literal[False] = False
    exact_rational_arithmetic_only: Literal[True] = True
    greater_than_ten_pp_is_descriptive_only: Literal[True] = True
    frozen_spec_equality_does_not_attest_same_provider_weights: Literal[True] = True
    all_call_response_identity_coverage_complete: bool
    provider_declared_identity_coordinates_not_exposed: Literal[True] = True
    provider_declared_identity_cross_replicate_consistency_attested: Literal[False] = False
    same_provider_weights_attested: Literal[False] = False
    public_development_descriptive_only: Literal[True] = True
    holdout_already_development_data: Literal[True] = True
    confidence_interval_available: Literal[False] = False
    statistical_significance_claimed: Literal[False] = False
    multiplicity_adjusted_inference_available: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    sealed_evidence: Literal[False] = False
    official_full_suite: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_output_rosters(self) -> Self:
        if self.replicate_ids != BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS or (
            self.outer_seeds_u64 != BFCL_V4_PILOT_OUTER_SEEDS_U64
        ):
            raise ValueError("campaign analysis output replicate roster changed")
        if tuple(item.arm for item in self.arms) != BFCL_V4_METRIC_ARM_ORDER:
            raise ValueError("campaign analysis arm order changed")
        if tuple(item.arm for item in self.promotions) != BFCL_V4_CAMPAIGN_ADAPTIVE_ARM_ORDER:
            raise ValueError("campaign analysis promotion order changed")
        if (
            tuple((item.treatment_arm, item.reference_arm) for item in self.paired_deltas)
            != BFCL_V4_PAIRED_CONTRASTS
        ):
            raise ValueError("campaign analysis contrast order changed")
        if len({item.sha256 for item in self.ordered_closure_refs}) != 3 or any(
            item.media_type != BFCL_V4_RUN_CLOSURE_MEDIA_TYPE for item in self.ordered_closure_refs
        ):
            raise ValueError("campaign analysis output closure refs changed")
        complete_identity_coverage = all(
            count == 100 for count in self.provider_identity_observation_counts
        )
        if self.all_call_response_identity_coverage_complete is not complete_identity_coverage:
            raise ValueError("campaign output provider identity coverage flag changed")
        if any(
            consistent and count == 0
            for count, consistent in zip(
                self.provider_identity_observation_counts,
                self.provider_declared_identity_consistent_within_replicates,
                strict=True,
            )
        ):
            raise ValueError("provider identity consistency lacks observed responses")
        by_arm_task = {
            (arm.arm, cluster.task_id): cluster.correctness
            for arm in self.arms
            for cluster in arm.task_clusters
        }
        for paired in self.paired_deltas:
            for cluster in paired.task_clusters:
                if (
                    cluster.treatment_correctness
                    != by_arm_task[(paired.treatment_arm, cluster.task_id)]
                    or cluster.reference_correctness
                    != by_arm_task[(paired.reference_arm, cluster.task_id)]
                ):
                    raise ValueError("campaign paired vectors differ from arm correctness")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
