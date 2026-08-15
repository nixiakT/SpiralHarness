"""Trusted terminal rendering for the BFCL V4 public live CLI.

The renderer only expands a canonical analysis artifact after the production
offline verifier has recomputed it.  It never accepts a callback-provided or
otherwise external performance summary.
"""

from __future__ import annotations

from spiral_harness.experiments.bfcl_v4_public_campaign_analysis_contracts import (
    BfclV4ExactRational,
    BfclV4PublicCampaignDescriptiveAnalysis,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor_contracts import (
    BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
    BfclV4CampaignExecutionStatus,
    BfclV4PublicCampaignExecutionRecord,
    BfclV4PublicCampaignExecutionVerification,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    load_canonical_model,
)
from spiral_harness.storage.protocol import ArtifactRepository


class BfclV4PublicLiveSummaryError(ValueError):
    """A sanitized failure to derive a trusted terminal summary."""


def _fraction(value: BfclV4ExactRational) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _failure_summary(record: BfclV4PublicCampaignExecutionRecord) -> dict[str, object] | None:
    failure = record.result.failure
    if failure is None:
        return None
    return {
        "stage": failure.stage.value,
        "completed_replicate_count": failure.completed_replicate_count,
        "active_replicate_ordinal": failure.active_replicate_ordinal,
        "active_replicate_id": failure.active_replicate_id,
        "active_outer_seed_u64": failure.active_outer_seed_u64,
        "unverified_run_result_ref": (
            None
            if failure.unverified_run_result_ref is None
            else failure.unverified_run_result_ref.model_dump(mode="json")
        ),
        "exception_text_persisted": failure.exception_text_persisted,
        "automatic_retry_attempted": failure.automatic_retry_attempted,
        "resumable_checkpoint_claimed": failure.resumable_checkpoint_claimed,
    }


def _analysis_summary(
    analysis: BfclV4PublicCampaignDescriptiveAnalysis,
    record: BfclV4PublicCampaignExecutionRecord,
) -> dict[str, object]:
    result = record.result
    completed = result.completed_replicates
    if result.analysis_ref is None or not (
        analysis.fingerprint == result.analysis_fingerprint == result.analysis_ref.sha256
    ):
        raise BfclV4PublicLiveSummaryError(
            "canonical campaign analysis fingerprint differs from the terminal reference"
        )
    expected_lineage = (
        result.analysis_input_fingerprint,
        result.campaign_fingerprint,
        result.model_spec_fingerprint,
        result.backend_fingerprint,
        result.inference_fingerprint,
        result.attempt_budget_fingerprint,
        tuple(item.replicate_id for item in completed),
        tuple(item.outer_seed_u64 for item in completed),
        tuple(item.plan_fingerprint for item in completed),
        tuple(item.schedule_content_sha256 for item in completed),
        tuple(item.closure_ref for item in completed),
        tuple(item.joint_selection_decision_ref.sha256 for item in completed),
        tuple(item.descriptive_metrics_ref.sha256 for item in completed),
        tuple(item.provider_identity_observation_count for item in completed),
        tuple(item.provider_declared_identity_consistent for item in completed),
    )
    observed_lineage = (
        analysis.analysis_input_fingerprint,
        analysis.campaign_fingerprint,
        analysis.model_spec_fingerprint,
        analysis.backend_fingerprint,
        analysis.inference_fingerprint,
        analysis.attempt_budget_fingerprint,
        analysis.replicate_ids,
        analysis.outer_seeds_u64,
        analysis.plan_fingerprints,
        analysis.schedule_content_sha256s,
        analysis.ordered_closure_refs,
        analysis.joint_selection_fingerprints,
        analysis.descriptive_metrics_fingerprints,
        analysis.provider_identity_observation_counts,
        analysis.provider_declared_identity_consistent_within_replicates,
    )
    if observed_lineage != expected_lineage:
        raise BfclV4PublicLiveSummaryError(
            "canonical campaign analysis differs from the verified terminal lineage"
        )

    return {
        "source": "verifier-checked-canonical-cas",
        "analysis_fingerprint": analysis.fingerprint,
        "arms": [
            {
                "arm": item.arm.value,
                "correct_task_seed_cells": item.correct_task_seed_cells,
                "task_seed_cell_count": item.task_seed_cell_count,
                "accuracy": _fraction(item.accuracy),
            }
            for item in analysis.arms
        ],
        "promotions": [
            {
                "arm": item.arm.value,
                "promotion_count": item.promotion_count,
                "rollback_count": item.rollback_count,
                "decisions": [
                    {
                        "replicate_id": replicate_id,
                        "outer_seed_u64": outer_seed,
                        "decision": decision.value,
                    }
                    for replicate_id, outer_seed, decision in zip(
                        analysis.replicate_ids,
                        analysis.outer_seeds_u64,
                        item.decisions,
                        strict=True,
                    )
                ],
            }
            for item in analysis.promotions
        ],
        "paired_contrasts": [
            {
                "treatment_arm": item.treatment_arm.value,
                "reference_arm": item.reference_arm.value,
                "wins": item.wins,
                "ties": item.ties,
                "losses": item.losses,
                "exact_delta": _fraction(item.exact_delta),
                "strictly_exceeds_positive_ten_percentage_points": (
                    item.strictly_exceeds_positive_ten_percentage_points
                ),
            }
            for item in analysis.paired_deltas
        ],
        "provider_by_seed": [
            {
                "ordinal": item.ordinal,
                "replicate_id": item.replicate_id,
                "outer_seed_u64": item.outer_seed_u64,
                "provider_attempts_succeeded": item.provider_attempts_succeeded,
                "provider_attempts_failed": item.provider_attempts_failed,
                "provider_identity_observation_count": (item.provider_identity_observation_count),
                "provider_declared_identity_consistent": (
                    item.provider_declared_identity_consistent
                ),
            }
            for item in completed
        ],
        "claim_limits": {
            "public_development_descriptive_only": (analysis.public_development_descriptive_only),
            "holdout_already_development_data": analysis.holdout_already_development_data,
            "task_seed_cells_are_independent_inferential_units": (
                analysis.task_seed_cells_are_independent_inferential_units
            ),
            "greater_than_ten_pp_is_descriptive_only": (
                analysis.greater_than_ten_pp_is_descriptive_only
            ),
            "all_call_response_identity_coverage_complete": (
                analysis.all_call_response_identity_coverage_complete
            ),
            "same_provider_weights_attested": analysis.same_provider_weights_attested,
            "confidence_interval_available": analysis.confidence_interval_available,
            "statistical_significance_claimed": analysis.statistical_significance_claimed,
            "official_full_suite": analysis.official_full_suite,
            "hidden_test_evidence": analysis.hidden_test_evidence,
            "reportable_result": analysis.reportable_result,
        },
    }


def _build_terminal_summary(
    repository: ArtifactRepository,
    record: BfclV4PublicCampaignExecutionRecord,
    verification: BfclV4PublicCampaignExecutionVerification,
) -> dict[str, object]:
    result = record.result
    expected_verification = (
        result.fingerprint,
        result.status,
        len(result.completed_replicates),
        result.verified_closed_model_calls,
        len(result.checkpoint_refs),
    )
    observed_verification = (
        verification.execution_result_fingerprint,
        verification.status,
        verification.verified_replicate_count,
        verification.verified_model_calls,
        verification.verified_checkpoint_count,
    )
    if observed_verification != expected_verification:
        raise BfclV4PublicLiveSummaryError(
            "terminal summary inputs differ from offline verification"
        )

    protocol_complete = result.status is BfclV4CampaignExecutionStatus.COMPLETE
    analysis_summary: dict[str, object] | None = None
    if protocol_complete:
        if not verification.analysis_recomputed_and_matched or result.analysis_ref is None:
            raise BfclV4PublicLiveSummaryError(
                "complete terminal lacks recomputed canonical analysis"
            )
        analysis = load_canonical_model(
            repository,
            result.analysis_ref,
            BfclV4PublicCampaignDescriptiveAnalysis,
            media_type=BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
        )
        analysis_summary = _analysis_summary(analysis, record)
    elif verification.analysis_recomputed_and_matched or result.analysis_ref is not None:
        raise BfclV4PublicLiveSummaryError("incomplete terminal cannot publish a verified analysis")

    return {
        "status": result.status.value,
        "protocol_complete": protocol_complete,
        "result_ref": record.result_ref.model_dump(mode="json"),
        "verified_closed_model_calls": verification.verified_model_calls,
        "verified_replicate_count": verification.verified_replicate_count,
        "verified_checkpoint_count": verification.verified_checkpoint_count,
        "terminal_offline_verification_passed": True,
        "analysis_input_ref": (
            None
            if result.analysis_input_ref is None
            else result.analysis_input_ref.model_dump(mode="json")
        ),
        "analysis_ref": (
            None if result.analysis_ref is None else result.analysis_ref.model_dump(mode="json")
        ),
        "analysis_recomputed_and_matched": verification.analysis_recomputed_and_matched,
        "analysis_summary": analysis_summary,
        "failure": _failure_summary(record),
        "public_development_only": result.public_development_only,
        "hidden_test_evidence": result.hidden_test_evidence,
        "reportable_result": result.reportable_result,
    }


def build_bfcl_v4_public_terminal_summary(
    repository: ArtifactRepository,
    record: BfclV4PublicCampaignExecutionRecord,
    verification: BfclV4PublicCampaignExecutionVerification,
) -> dict[str, object]:
    """Expand only canonical, verifier-matched BFCL terminal evidence."""

    derivation_failed = False
    try:
        checked_record = BfclV4PublicCampaignExecutionRecord.model_validate(
            record,
            strict=True,
        )
        checked_verification = BfclV4PublicCampaignExecutionVerification.model_validate(
            verification,
            strict=True,
        )
        terminal_summary = _build_terminal_summary(
            repository,
            checked_record,
            checked_verification,
        )
    except BfclV4PublicLiveSummaryError:
        raise
    except Exception:
        derivation_failed = True
    if derivation_failed:
        raise BfclV4PublicLiveSummaryError(
            "verified campaign terminal summary could not be derived"
        ) from None
    return terminal_summary


__all__ = [
    "BfclV4PublicLiveSummaryError",
    "build_bfcl_v4_public_terminal_summary",
]
