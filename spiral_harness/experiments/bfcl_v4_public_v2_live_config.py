"""Frozen same-model live configuration for the 1,086-call BFCL v2 DAG."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE,
    BfclV4PublicV2SemanticDevelopmentRelease,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.execution.contracts import AttemptBudget, FrozenModelSpec
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BfclV4PublicLiveModelCatalogObservation,
)

BFCL_V4_PUBLIC_V2_LIVE_MODEL_REVISION = (
    "gateway-declared/opaque-weight-revision-unavailable@bfcl-public-v2-live-v1"
)
BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER = (
    "gateway-declared/opaque-tokenizer-identity-unavailable@bfcl-public-v2-live-v1"
)
BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER_REVISION = (
    "gateway-declared/opaque-tokenizer-revision-unavailable@bfcl-public-v2-live-v1"
)
BFCL_V4_PUBLIC_V2_LIVE_RUNTIME = (
    "gateway-declared/opaque-serving-runtime-unavailable@bfcl-public-v2-live-v1"
)
BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET = AttemptBudget(
    max_attempts=1_086,
    max_tokens_per_attempt=BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING,
    max_total_tokens=35_586_048,
)


class BfclV4PublicV2LiveExecutionConfig(ImmutableModel):
    """Credential-free activation binding all five arms to one exact route."""

    schema_version: Literal["1"] = "1"
    catalog_observation: BfclV4PublicLiveModelCatalogObservation
    catalog_observation_fingerprint: Sha256
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_ref: ArtifactRef
    semantic_release_fingerprint: Sha256
    model_spec: FrozenModelSpec
    model_spec_fingerprint: Sha256
    backend_fingerprint: Sha256
    serializer_fingerprint: Sha256
    parser_fingerprint: Sha256
    transport_fingerprint: Sha256
    inference_fingerprint: Sha256
    attempt_budget: AttemptBudget
    attempt_budget_fingerprint: Sha256
    selected_model_route: Literal["qwen36-35b-a3b"] = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
    registered_call_slots: Literal[1_086] = 1_086
    provider_attempts_per_call: Literal[1] = 1
    automatic_retries_allowed: Literal[False] = False
    failed_attempt_consumes_slot: Literal[True] = True
    route_frozen_before_semantic_review_and_campaign: Literal[True] = True
    same_exact_route_all_arms_and_replicates: Literal[True] = True
    same_inference_all_arms_and_replicates: Literal[True] = True
    paired_provider_seeds_from_plan: Literal[True] = True
    reviewer_models_excluded_from_benchmark_comparison: Literal[True] = True
    semantic_release_live_hmac_verified_before_freeze: Literal[True] = True
    gateway_revision_is_opaque_limitation: Literal[True] = True
    gateway_tokenizer_is_opaque_limitation: Literal[True] = True
    gateway_runtime_is_opaque_limitation: Literal[True] = True
    actual_weight_revision_attested: Literal[False] = False
    tokenizer_identity_attested: Literal[False] = False
    serving_runtime_attested: Literal[False] = False
    same_provider_weights_attested: Literal[False] = False
    credentials_recorded: Literal[False] = False
    score_bearing_execution_allowed: Literal[True] = True
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    confirmatory_evidence: Literal[False] = False
    provider_calls_executed: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_live_activation(self) -> Self:
        if (
            self.catalog_observation_fingerprint != self.catalog_observation.fingerprint
            or self.catalog_observation.selected_model_route
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
            or not self.catalog_observation.direct_route_advertised
            or self.catalog_observation.fallback_used
        ):
            raise ValueError("BFCL v2 live activation lacks the exact direct model route")
        if (
            self.semantic_release_ref.media_type
            != BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE
            or self.semantic_release_ref.sha256 != self.semantic_release_fingerprint
        ):
            raise ValueError("BFCL v2 live activation cites the wrong semantic release")
        expected_spec = (
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
            BFCL_V4_PUBLIC_V2_LIVE_MODEL_REVISION,
            BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER,
            BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER_REVISION,
            BFCL_V4_PUBLIC_V2_LIVE_RUNTIME,
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
        )
        observed_spec = (
            self.model_spec.model,
            self.model_spec.revision,
            self.model_spec.tokenizer,
            self.model_spec.tokenizer_revision,
            self.model_spec.runtime,
            self.model_spec.inference,
        )
        if expected_spec != observed_spec:
            raise ValueError("BFCL v2 live model or inference differs from the frozen plan")
        if (
            self.model_spec_fingerprint != self.model_spec.fingerprint
            or self.backend_fingerprint != self.model_spec.backend_fingerprint
            or self.inference_fingerprint != self.model_spec.inference_fingerprint
        ):
            raise ValueError("BFCL v2 live implementation fingerprints changed")
        if (
            self.attempt_budget != BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET
            or self.attempt_budget_fingerprint != self.attempt_budget.fingerprint
            or self.attempt_budget.max_attempts != self.registered_call_slots
        ):
            raise ValueError("BFCL v2 live attempt budget differs from all 1,086 slots")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def freeze_bfcl_v4_public_v2_live_execution_config(
    *,
    catalog_observation: BfclV4PublicLiveModelCatalogObservation,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    verified_semantic_release: BfclV4PublicV2SemanticDevelopmentRelease,
    backend_name: str,
    backend_fingerprint: str,
    serializer_fingerprint: str,
    parser_fingerprint: str,
    transport_fingerprint: str,
) -> BfclV4PublicV2LiveExecutionConfig:
    """Freeze a config only after the caller has live-verified semantic evidence."""

    checked_catalog = BfclV4PublicLiveModelCatalogObservation.model_validate(
        catalog_observation,
        strict=True,
    )
    checked_campaign = BfclV4PublicDevelopmentV2CampaignPlan.model_validate(
        campaign,
        strict=True,
    )
    checked_release = BfclV4PublicV2SemanticDevelopmentRelease.model_validate(
        verified_semantic_release,
        strict=True,
    )
    if (
        checked_campaign.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or checked_campaign.node_schedule_content_sha256
        != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
        or checked_release.campaign_plan_fingerprint != checked_campaign.fingerprint
        or not checked_release.score_bearing_execution_allowed
    ):
        raise ValueError("campaign or verified semantic release differs from the frozen v2 study")
    spec = FrozenModelSpec(
        backend=backend_name,
        backend_fingerprint=backend_fingerprint,
        model=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
        revision=BFCL_V4_PUBLIC_V2_LIVE_MODEL_REVISION,
        tokenizer=BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER,
        tokenizer_revision=BFCL_V4_PUBLIC_V2_LIVE_TOKENIZER_REVISION,
        runtime=BFCL_V4_PUBLIC_V2_LIVE_RUNTIME,
        inference=BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    )
    return BfclV4PublicV2LiveExecutionConfig(
        catalog_observation=checked_catalog,
        catalog_observation_fingerprint=checked_catalog.fingerprint,
        semantic_release_ref=checked_release.ref,
        semantic_release_fingerprint=checked_release.fingerprint,
        model_spec=spec,
        model_spec_fingerprint=spec.fingerprint,
        backend_fingerprint=backend_fingerprint,
        serializer_fingerprint=serializer_fingerprint,
        parser_fingerprint=parser_fingerprint,
        transport_fingerprint=transport_fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        attempt_budget=BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET,
        attempt_budget_fingerprint=BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET.fingerprint,
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_") or name.startswith("Bfcl") or name.startswith("freeze_bfcl")
]
