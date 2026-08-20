"""Portable contracts for fail-closed execution of the frozen BFCL campaign."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID,
    BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BFCL_V4_PILOT_OUTER_SEEDS_U64,
    BfclV4PilotOuterSeed,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
)

BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-registration.v1+json"
)
BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-live-config.v1+json"
)
BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-replicate-verification.v1+json"
)
BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-analysis-input.v1+json"
)
BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-analysis.v1+json"
)
BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-checkpoint.v1+json"
)
BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-campaign-execution-result.v1+json"
)


def _exact_ref(value: ImmutableModel, ref: ArtifactRef, media_type: str) -> bool:
    payload = canonical_json_bytes(value)
    return (
        ref.media_type == media_type
        and ref.size == len(payload)
        and ref.sha256 == sha256_bytes(payload)
    )


class BfclV4CampaignExecutionStatus(StrEnum):
    """Terminal status; an incomplete run is never silently resumable."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class BfclV4CampaignFailureStage(StrEnum):
    """Safe failure coordinate without persisting provider exception text."""

    REPLICATE_EXECUTION = "replicate-execution"
    REPLICATE_VERIFICATION = "replicate-verification"
    REPLICATE_CHECKPOINT = "replicate-checkpoint-publication"
    CAMPAIGN_ANALYSIS = "campaign-analysis"


class BfclV4CampaignVerifiedReplicate(ImmutableModel):
    """One result ref admitted only after its complete offline replay verifier passes."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=3, strict=True)]
    replicate_id: NonEmptyStr
    outer_seed_u64: BfclV4PilotOuterSeed
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    attempt_ledger_id: NonEmptyStr
    model_spec_fingerprint: Sha256
    backend_fingerprint: Sha256
    inference_fingerprint: Sha256
    attempt_budget_fingerprint: Sha256
    run_result_ref: ArtifactRef
    run_result_fingerprint: Sha256
    verification_ref: ArtifactRef
    verification_fingerprint: Sha256
    closure_ref: ArtifactRef
    joint_selection_decision_ref: ArtifactRef
    descriptive_metrics_ref: ArtifactRef
    provider_attempts_succeeded: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_attempts_failed: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_identity_observation_count: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_declared_identity_consistent: bool
    completed_model_calls: Literal[100] = 100
    offline_verification_passed: Literal[True] = True

    @model_validator(mode="after")
    def _bind_registered_replicate(self) -> Self:
        expected_plan = build_bfcl_v4_public_pilot_call_plan(self.outer_seed_u64)
        expected = (
            BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS[self.ordinal],
            BFCL_V4_PILOT_OUTER_SEEDS_U64[self.ordinal],
            expected_plan.fingerprint,
            expected_plan.schedule_content_sha256,
            f"{BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID}/{self.replicate_id}",
        )
        observed = (
            self.replicate_id,
            self.outer_seed_u64,
            self.plan_fingerprint,
            self.schedule_content_sha256,
            self.attempt_ledger_id,
        )
        if observed != expected:
            raise ValueError("verified replicate differs from the registered seed or call plan")
        expected_media = (
            (self.run_result_ref, BFCL_V4_RUNNER_RESULT_MEDIA_TYPE),
            (self.verification_ref, BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE),
            (self.closure_ref, BFCL_V4_RUN_CLOSURE_MEDIA_TYPE),
            (self.joint_selection_decision_ref, BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE),
            (self.descriptive_metrics_ref, BFCL_V4_RUNNER_METRICS_MEDIA_TYPE),
        )
        if any(ref.media_type != media_type for ref, media_type in expected_media):
            raise ValueError("verified replicate contains a reference with the wrong media type")
        if (
            self.run_result_ref.sha256 != self.run_result_fingerprint
            or self.verification_ref.sha256 != self.verification_fingerprint
        ):
            raise ValueError("verified replicate fingerprints differ from exact CAS refs")
        if self.provider_attempts_succeeded + self.provider_attempts_failed != 100:
            raise ValueError("verified replicate provider outcomes do not cover 100 calls")
        if self.provider_identity_observation_count > self.provider_attempts_succeeded:
            raise ValueError("verified replicate identity coverage exceeds successful calls")
        if self.provider_declared_identity_consistent and not (
            self.provider_identity_observation_count
        ):
            raise ValueError("identity consistency requires an observed provider response")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CampaignExecutionCheckpoint(ImmutableModel):
    """Append-only prefix checkpoint published after each immediate verification."""

    schema_version: Literal["1"] = "1"
    campaign_fingerprint: Literal[BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT]
    campaign_ref: ArtifactRef
    live_execution_config_ref: ArtifactRef
    live_execution_config_fingerprint: Sha256
    model_spec_fingerprint: Sha256
    backend_fingerprint: Sha256
    inference_fingerprint: Sha256
    attempt_budget_fingerprint: Sha256
    completed_replicates: Annotated[
        tuple[BfclV4CampaignVerifiedReplicate, ...], Field(min_length=1, max_length=3)
    ]
    previous_checkpoint_ref: ArtifactRef | None = None
    completed_replicate_count: Annotated[int, Field(ge=1, le=3, strict=True)]
    verified_closed_model_calls: Literal[100, 200, 300]
    next_replicate_ordinal: Annotated[int, Field(ge=1, lt=3, strict=True)] | None
    same_backend_object_required: Literal[True] = True
    explicit_resume_authorized: Literal[False] = False
    automatic_retry_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_prefix(self) -> Self:
        count = len(self.completed_replicates)
        if (
            self.campaign_ref.media_type != BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE
            or self.live_execution_config_ref.media_type != BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE
        ):
            raise ValueError("checkpoint campaign or live-config reference has the wrong type")
        if (
            self.campaign_ref.sha256 != self.campaign_fingerprint
            or self.live_execution_config_ref.sha256 != self.live_execution_config_fingerprint
        ):
            raise ValueError("checkpoint campaign or live-config CAS fingerprint changed")
        if tuple(item.ordinal for item in self.completed_replicates) != tuple(range(count)):
            raise ValueError("checkpoint replicates are not an exact registered prefix")
        common = (
            self.model_spec_fingerprint,
            self.backend_fingerprint,
            self.inference_fingerprint,
            self.attempt_budget_fingerprint,
        )
        if any(
            (
                item.model_spec_fingerprint,
                item.backend_fingerprint,
                item.inference_fingerprint,
                item.attempt_budget_fingerprint,
            )
            != common
            for item in self.completed_replicates
        ):
            raise ValueError("checkpoint replicates differ from one frozen execution identity")
        expected_next = count if count < 3 else None
        if (
            self.completed_replicate_count != count
            or self.verified_closed_model_calls != count * 100
            or self.next_replicate_ordinal != expected_next
        ):
            raise ValueError("checkpoint counts differ from its exact verified prefix")
        if count == 1 and self.previous_checkpoint_ref is not None:
            raise ValueError("first checkpoint cannot have a predecessor")
        if count > 1 and (
            self.previous_checkpoint_ref is None
            or self.previous_checkpoint_ref.media_type != BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE
        ):
            raise ValueError("later checkpoint requires a typed predecessor")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CampaignExecutionFailure(ImmutableModel):
    """Sanitized terminal failure; provider exception messages are never persisted."""

    stage: BfclV4CampaignFailureStage
    completed_replicate_count: Annotated[int, Field(ge=0, le=3, strict=True)]
    active_replicate_ordinal: Annotated[int, Field(ge=0, lt=3, strict=True)] | None = None
    active_replicate_id: NonEmptyStr | None = None
    active_outer_seed_u64: BfclV4PilotOuterSeed | None = None
    unverified_run_result_ref: ArtifactRef | None = None
    unverified_run_result_fingerprint: Sha256 | None = None
    exception_text_persisted: Literal[False] = False
    automatic_retry_attempted: Literal[False] = False
    later_seed_skipped_as_if_completed: Literal[False] = False
    resumable_checkpoint_claimed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_failure_coordinate(self) -> Self:
        active = (
            self.active_replicate_ordinal,
            self.active_replicate_id,
            self.active_outer_seed_u64,
        )
        if self.stage is BfclV4CampaignFailureStage.CAMPAIGN_ANALYSIS:
            if active != (None, None, None) or self.completed_replicate_count != 3:
                raise ValueError("analysis failure requires three completed replicates")
        else:
            ordinal = self.active_replicate_ordinal
            if ordinal is None:
                raise ValueError("replicate failure requires an active registered coordinate")
            expected = (
                ordinal,
                BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS[ordinal],
                BFCL_V4_PILOT_OUTER_SEEDS_U64[ordinal],
            )
            if active != expected:
                raise ValueError("failure coordinate differs from the registered replicate")
            expected_completed = (
                ordinal + 1
                if self.stage is BfclV4CampaignFailureStage.REPLICATE_CHECKPOINT
                else ordinal
            )
            if self.completed_replicate_count != expected_completed:
                raise ValueError("failure completed prefix differs from its active stage")
        has_unverified = self.unverified_run_result_ref is not None
        if has_unverified is not (self.unverified_run_result_fingerprint is not None):
            raise ValueError("unverified result ref and fingerprint must appear together")
        if has_unverified and (
            self.stage is not BfclV4CampaignFailureStage.REPLICATE_VERIFICATION
            or self.unverified_run_result_ref.media_type != BFCL_V4_RUNNER_RESULT_MEDIA_TYPE
            or self.unverified_run_result_ref.sha256 != self.unverified_run_result_fingerprint
        ):
            raise ValueError("unverified result is only valid at the verification stage")
        return self


class BfclV4PublicCampaignExecutionResult(ImmutableModel):
    """Terminal complete or explicit-incomplete root for one executor invocation."""

    schema_version: Literal["1"] = "1"
    status: BfclV4CampaignExecutionStatus
    campaign_fingerprint: Literal[BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT]
    campaign_ref: ArtifactRef
    live_execution_config_ref: ArtifactRef
    live_execution_config_fingerprint: Sha256
    model_spec_fingerprint: Sha256
    backend_fingerprint: Sha256
    inference_fingerprint: Sha256
    attempt_budget_fingerprint: Sha256
    completed_replicates: Annotated[
        tuple[BfclV4CampaignVerifiedReplicate, ...], Field(max_length=3)
    ] = ()
    checkpoint_refs: Annotated[tuple[ArtifactRef, ...], Field(max_length=3)] = ()
    latest_checkpoint_ref: ArtifactRef | None = None
    verified_closed_model_calls: Literal[0, 100, 200, 300]
    analysis_input_ref: ArtifactRef | None = None
    analysis_input_fingerprint: Sha256 | None = None
    analysis_ref: ArtifactRef | None = None
    analysis_fingerprint: Sha256 | None = None
    failure: BfclV4CampaignExecutionFailure | None = None
    total_registered_model_calls: Literal[300] = 300
    sequential_replicate_execution: Literal[True] = True
    same_live_config_backend_spec_and_budget_used: Literal[True] = True
    each_completed_replicate_immediately_offline_verified: Literal[True] = True
    analysis_derived_only_from_verified_result_refs: Literal[True] = True
    provider_failures_remain_score_bearing_outcomes: Literal[True] = True
    automatic_retries_used: Literal[False] = False
    seed_reordering_used: Literal[False] = False
    seed_skipping_used: Literal[False] = False
    adaptive_early_stopping_used: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_terminal_result(self) -> Self:
        count = len(self.completed_replicates)
        if (
            self.campaign_ref.media_type != BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE
            or self.live_execution_config_ref.media_type != BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE
        ):
            raise ValueError("execution result campaign or config reference has the wrong type")
        if (
            self.campaign_ref.sha256 != self.campaign_fingerprint
            or self.live_execution_config_ref.sha256 != self.live_execution_config_fingerprint
        ):
            raise ValueError("execution result campaign or config CAS fingerprint changed")
        if tuple(item.ordinal for item in self.completed_replicates) != tuple(range(count)):
            raise ValueError("execution result replicates are not a registered prefix")
        if self.verified_closed_model_calls != count * 100:
            raise ValueError("verified call count differs from completed replicate refs")
        if any(
            ref.media_type != BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE for ref in self.checkpoint_refs
        ):
            raise ValueError("execution result contains an untyped checkpoint reference")
        if len({ref.sha256 for ref in self.checkpoint_refs}) != len(self.checkpoint_refs):
            raise ValueError("execution result repeats a checkpoint reference")
        if self.latest_checkpoint_ref != (
            self.checkpoint_refs[-1] if self.checkpoint_refs else None
        ):
            raise ValueError("latest checkpoint differs from the ordered checkpoint tail")
        common = (
            self.model_spec_fingerprint,
            self.backend_fingerprint,
            self.inference_fingerprint,
            self.attempt_budget_fingerprint,
        )
        if any(
            (
                item.model_spec_fingerprint,
                item.backend_fingerprint,
                item.inference_fingerprint,
                item.attempt_budget_fingerprint,
            )
            != common
            for item in self.completed_replicates
        ):
            raise ValueError("execution result replicates differ from the frozen config")
        input_pair = (self.analysis_input_ref, self.analysis_input_fingerprint)
        analysis_pair = (self.analysis_ref, self.analysis_fingerprint)
        if (input_pair[0] is None) is not (input_pair[1] is None) or (
            (analysis_pair[0] is None) is not (analysis_pair[1] is None)
        ):
            raise ValueError("analysis refs and fingerprints must appear in complete pairs")
        if self.analysis_input_ref is not None and (
            self.analysis_input_ref.media_type != BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE
            or self.analysis_input_ref.sha256 != self.analysis_input_fingerprint
        ):
            raise ValueError("analysis-input reference has the wrong media type")
        if self.analysis_ref is not None and (
            self.analysis_ref.media_type != BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE
            or self.analysis_ref.sha256 != self.analysis_fingerprint
        ):
            raise ValueError("analysis reference has the wrong media type")
        if self.status is BfclV4CampaignExecutionStatus.COMPLETE:
            if (
                count != 3
                or len(self.checkpoint_refs) != 3
                or self.failure is not None
                or None in (*input_pair, *analysis_pair)
            ):
                raise ValueError("complete campaign lacks three closures or terminal analysis")
        else:
            if self.failure is None or self.analysis_ref is not None:
                raise ValueError("incomplete campaign requires failure and cannot publish analysis")
            if self.failure.completed_replicate_count != count:
                raise ValueError("incomplete campaign failure differs from verified prefix")
            expected_checkpoints = count
            if self.failure.stage is BfclV4CampaignFailureStage.REPLICATE_CHECKPOINT:
                expected_checkpoints -= 1
            if len(self.checkpoint_refs) != expected_checkpoints:
                raise ValueError("incomplete campaign checkpoint prefix differs from failure stage")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicCampaignExecutionRecord(ImmutableModel):
    """Return value binding exact terminal result bytes to the published root ref."""

    schema_version: Literal["1"] = "1"
    result: BfclV4PublicCampaignExecutionResult
    result_ref: ArtifactRef

    @model_validator(mode="after")
    def _bind_result_ref(self) -> Self:
        if not _exact_ref(
            self.result,
            self.result_ref,
            BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
        ):
            raise ValueError("campaign execution result ref differs from exact result bytes")
        return self


class BfclV4PublicCampaignExecutionVerification(ImmutableModel):
    """Offline audit summary reconstructed from the terminal root and all result refs."""

    schema_version: Literal["1"] = "1"
    execution_result_fingerprint: Sha256
    status: BfclV4CampaignExecutionStatus
    verified_replicate_count: Annotated[int, Field(ge=0, le=3, strict=True)]
    verified_model_calls: Literal[0, 100, 200, 300]
    verified_checkpoint_count: Annotated[int, Field(ge=0, le=3, strict=True)]
    analysis_recomputed_and_matched: bool
    exact_registered_prefix_verified: Literal[True] = True
    result_selection_metrics_lineage_verified: Literal[True] = True
    no_external_summary_trusted: Literal[True] = True
    public_development_only: Literal[True] = True
    reportable_result: Literal[False] = False


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
