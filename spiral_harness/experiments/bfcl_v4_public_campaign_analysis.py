"""Pure offline descriptive aggregation for the three-seed BFCL campaign."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis_contracts import (
    BFCL_V4_CAMPAIGN_ADAPTIVE_ARM_ORDER,
    BfclV4CampaignArmAggregate,
    BfclV4CampaignArmTaskCluster,
    BfclV4CampaignPairedAggregate,
    BfclV4CampaignPairedTaskCluster,
    BfclV4CampaignPromotionAggregate,
    BfclV4ExactRational,
    BfclV4PublicCampaignAnalysisInput,
    BfclV4PublicCampaignDescriptiveAnalysis,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BFCL_V4_HOLDOUT_TASK_IDS,
    BFCL_V4_METRIC_ARM_ORDER,
    BFCL_V4_PAIRED_CONTRASTS,
)


class BfclV4PublicCampaignAnalysisError(ValueError):
    """Raised when campaign evidence fails strict offline aggregation checks."""


def _checked[ModelT: BaseModel](value: object, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate(value, strict=True)
    except Exception as exc:
        raise BfclV4PublicCampaignAnalysisError(f"{label} failed strict revalidation") from exc


def _arm_correctness(
    analysis_input: BfclV4PublicCampaignAnalysisInput,
    arm: BfclV4PilotArm,
) -> tuple[tuple[bool, ...], ...]:
    """Return replicate-major correctness after checking the inherited arm order."""

    vectors = []
    for replicate in analysis_input.replicates:
        metric = next(item for item in replicate.descriptive_metrics.arms if item.arm is arm)
        vectors.append(metric.correctness)
    return tuple(vectors)


def _build_arm_aggregates(
    analysis_input: BfclV4PublicCampaignAnalysisInput,
) -> tuple[BfclV4CampaignArmAggregate, ...]:
    replicate_ids = tuple(item.replicate_id for item in analysis_input.replicates)
    outer_seeds = tuple(item.outer_seed_u64 for item in analysis_input.replicates)
    aggregates = []
    for arm in BFCL_V4_METRIC_ARM_ORDER:
        replicate_vectors = _arm_correctness(analysis_input, arm)
        clusters = tuple(
            BfclV4CampaignArmTaskCluster(
                arm=arm,
                task_id=task_id,
                replicate_ids=replicate_ids,
                outer_seeds_u64=outer_seeds,
                correctness=tuple(vector[task_index] for vector in replicate_vectors),
                correct_replicates=sum(vector[task_index] for vector in replicate_vectors),
            )
            for task_index, task_id in enumerate(BFCL_V4_HOLDOUT_TASK_IDS)
        )
        correct = sum(item.correct_replicates for item in clusters)
        aggregates.append(
            BfclV4CampaignArmAggregate(
                arm=arm,
                task_clusters=clusters,
                correct_task_seed_cells=correct,
                accuracy=BfclV4ExactRational.from_ratio(correct, 24),
            )
        )
    return tuple(aggregates)


def _build_promotions(
    analysis_input: BfclV4PublicCampaignAnalysisInput,
) -> tuple[BfclV4CampaignPromotionAggregate, ...]:
    replicate_ids = tuple(item.replicate_id for item in analysis_input.replicates)
    summaries = []
    for arm in BFCL_V4_CAMPAIGN_ADAPTIVE_ARM_ORDER:
        decisions = tuple(
            (
                item.joint_selection.score.decision
                if arm is BfclV4PilotArm.SCORE
                else item.joint_selection.full.decision
            )
            for item in analysis_input.replicates
        )
        promoted = sum(item.value == "promote" for item in decisions)
        summaries.append(
            BfclV4CampaignPromotionAggregate(
                arm=arm,
                replicate_ids=replicate_ids,
                decisions=decisions,
                promotion_count=promoted,
                rollback_count=3 - promoted,
            )
        )
    return tuple(summaries)


def _paired_cluster(
    *,
    treatment: BfclV4CampaignArmAggregate,
    reference: BfclV4CampaignArmAggregate,
    task_index: int,
) -> BfclV4CampaignPairedTaskCluster:
    treatment_cluster = treatment.task_clusters[task_index]
    reference_cluster = reference.task_clusters[task_index]
    wins = sum(
        left and not right
        for left, right in zip(
            treatment_cluster.correctness, reference_cluster.correctness, strict=True
        )
    )
    losses = sum(
        right and not left
        for left, right in zip(
            treatment_cluster.correctness, reference_cluster.correctness, strict=True
        )
    )
    return BfclV4CampaignPairedTaskCluster(
        treatment_arm=treatment.arm,
        reference_arm=reference.arm,
        task_id=treatment_cluster.task_id,
        treatment_correctness=treatment_cluster.correctness,
        reference_correctness=reference_cluster.correctness,
        wins=wins,
        ties=3 - wins - losses,
        losses=losses,
        exact_delta=BfclV4ExactRational.from_ratio(wins - losses, 3),
    )


def _build_paired_aggregates(
    arms: tuple[BfclV4CampaignArmAggregate, ...],
) -> tuple[BfclV4CampaignPairedAggregate, ...]:
    by_arm = {item.arm: item for item in arms}
    aggregates = []
    for treatment_arm, reference_arm in BFCL_V4_PAIRED_CONTRASTS:
        clusters = tuple(
            _paired_cluster(
                treatment=by_arm[treatment_arm],
                reference=by_arm[reference_arm],
                task_index=task_index,
            )
            for task_index in range(8)
        )
        wins = sum(item.wins for item in clusters)
        ties = sum(item.ties for item in clusters)
        losses = sum(item.losses for item in clusters)
        exact_delta = BfclV4ExactRational.from_ratio(wins - losses, 24)
        aggregates.append(
            BfclV4CampaignPairedAggregate(
                treatment_arm=treatment_arm,
                reference_arm=reference_arm,
                task_clusters=clusters,
                wins=wins,
                ties=ties,
                losses=losses,
                exact_delta=exact_delta,
                strictly_exceeds_positive_ten_percentage_points=(
                    exact_delta.numerator * 10 > exact_delta.denominator
                ),
            )
        )
    return tuple(aggregates)


def compute_bfcl_v4_public_campaign_descriptive_analysis(
    analysis_input: BfclV4PublicCampaignAnalysisInput,
) -> BfclV4PublicCampaignDescriptiveAnalysis:
    """Aggregate exact public-development outcomes without inference or model calls."""

    checked = _checked(
        analysis_input,
        BfclV4PublicCampaignAnalysisInput,
        "BFCL public campaign analysis input",
    )
    arms = _build_arm_aggregates(checked)
    return BfclV4PublicCampaignDescriptiveAnalysis(
        analysis_input_fingerprint=checked.fingerprint,
        campaign_fingerprint=checked.campaign_fingerprint,
        replicate_ids=tuple(item.replicate_id for item in checked.replicates),
        outer_seeds_u64=tuple(item.outer_seed_u64 for item in checked.replicates),
        plan_fingerprints=tuple(item.plan_fingerprint for item in checked.replicates),
        schedule_content_sha256s=tuple(item.schedule_content_sha256 for item in checked.replicates),
        model_spec_fingerprint=checked.model_spec_fingerprint,
        backend_fingerprint=checked.backend_fingerprint,
        inference_fingerprint=checked.inference_fingerprint,
        attempt_budget_fingerprint=checked.per_call_budget.attempt_budget_fingerprint,
        per_call_budget_fingerprint=checked.per_call_budget_fingerprint,
        ordered_closure_refs=checked.ordered_closure_refs,
        joint_selection_fingerprints=tuple(
            item.joint_selection_fingerprint for item in checked.replicates
        ),
        descriptive_metrics_fingerprints=tuple(
            item.descriptive_metrics_fingerprint for item in checked.replicates
        ),
        provider_identity_observation_counts=tuple(
            item.provider_identity_observation_count for item in checked.replicates
        ),
        provider_declared_identity_consistent_within_replicates=tuple(
            item.provider_declared_identity_consistent for item in checked.replicates
        ),
        all_call_response_identity_coverage_complete=(
            checked.all_call_response_identity_coverage_complete
        ),
        arms=arms,
        promotions=_build_promotions(checked),
        paired_deltas=_build_paired_aggregates(arms),
    )


__all__ = [
    "BfclV4PublicCampaignAnalysisError",
    "compute_bfcl_v4_public_campaign_descriptive_analysis",
]
