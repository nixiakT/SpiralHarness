from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE,
    BfclV4PublicV2SemanticDevelopmentRelease,
)
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET,
    BfclV4PublicV2LiveExecutionConfig,
    BfclV4PublicV2SemanticReleaseEvidenceShape,
    freeze_bfcl_v4_public_v2_live_execution_config,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(label: str, media_type: str = "application/x-test") -> ArtifactRef:
    return ArtifactRef(sha256=_digest(label), size=len(label), media_type=media_type)


def _release() -> BfclV4PublicV2SemanticDevelopmentRelease:
    return BfclV4PublicV2SemanticDevelopmentRelease(
        source_universe_fingerprint=_digest("universe"),
        reviewer_packet_ref=_ref("packet"),
        trusted_mapping_ref=_ref("mapping"),
        primary_review_refs=(_ref("review-one"), _ref("review-two")),
        primary_execution_attestation_refs=(
            _ref("attestation-one"),
            _ref("attestation-two"),
        ),
        final_partition_sha256=_digest("partition"),
        final_semantic_family_count=40,
        primary_pairwise_disagreement_count=0,
        reviewer_count=2,
        release_authority_hmac_sha256=_digest("release-hmac"),
    )


def _config():
    return freeze_bfcl_v4_public_v2_live_execution_config(
        catalog_observation=observe_bfcl_v4_public_live_model_catalog(
            (BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,),
            observed_at_utc="2026-08-15T12:00:00Z",
        ),
        campaign=build_bfcl_v4_public_development_v2_campaign_plan(),
        verified_semantic_release=_release(),
        backend_name="openai-compatible-native-function",
        backend_fingerprint=_digest("backend"),
        serializer_fingerprint=_digest("serializer"),
        parser_fingerprint=_digest("parser"),
        transport_fingerprint=_digest("transport"),
    )


def test_live_config_binds_one_model_inference_budget_plan_and_release() -> None:
    config = _config()

    assert config.campaign_plan_fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    assert config.selected_model_route == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
    assert config.model_spec.model == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
    assert config.model_spec.inference == BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE
    assert config.attempt_budget == BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET
    assert config.attempt_budget.max_attempts == 1_086
    assert config.semantic_release_ref == _release().ref
    assert (
        config.semantic_release_evidence_shape
        is BfclV4PublicV2SemanticReleaseEvidenceShape.LEGACY_SINGLE_PACKET_V1
    )
    assert (
        config.semantic_release_ref.media_type
        == BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE
    )
    assert config.same_exact_route_all_arms_and_replicates is True
    assert config.same_inference_all_arms_and_replicates is True
    assert config.score_bearing_execution_allowed is True
    assert config.public_development_only is True
    assert config.hidden_test_evidence is False
    assert config.same_provider_weights_attested is False
    assert len(config.fingerprint) == 64


def test_fallback_route_cannot_silently_replace_the_frozen_direct_route() -> None:
    fallback = observe_bfcl_v4_public_live_model_catalog(
        ("dashscope/qwen36-35b-a3b",),
        observed_at_utc="2026-08-15T12:00:00Z",
    )
    with pytest.raises(ValidationError, match="exact direct model route"):
        freeze_bfcl_v4_public_v2_live_execution_config(
            catalog_observation=fallback,
            campaign=build_bfcl_v4_public_development_v2_campaign_plan(),
            verified_semantic_release=_release(),
            backend_name="openai-compatible-native-function",
            backend_fingerprint=_digest("backend"),
            serializer_fingerprint=_digest("serializer"),
            parser_fingerprint=_digest("parser"),
            transport_fingerprint=_digest("transport"),
        )


def test_release_ref_and_attempt_budget_tampering_fail_closed() -> None:
    config = _config()
    bad_ref = _ref("wrong-release", "application/x-wrong")
    with pytest.raises(ValidationError, match="wrong semantic release"):
        BfclV4PublicV2LiveExecutionConfig(
            **{
                **config.model_dump(mode="python"),
                "semantic_release_ref": bad_ref,
                "semantic_release_fingerprint": bad_ref.sha256,
            }
        )

    smaller = BFCL_V4_PUBLIC_V2_LIVE_ATTEMPT_BUDGET.model_copy(update={"max_attempts": 1_085})
    with pytest.raises(ValidationError, match="attempt budget"):
        BfclV4PublicV2LiveExecutionConfig(
            **{
                **config.model_dump(mode="python"),
                "attempt_budget": smaller,
                "attempt_budget_fingerprint": smaller.fingerprint,
            }
        )


def test_release_shape_cannot_be_changed_independently_of_its_media_type() -> None:
    config = _config()

    with pytest.raises(ValidationError, match="wrong semantic release"):
        BfclV4PublicV2LiveExecutionConfig(
            **{
                **config.model_dump(mode="python"),
                "semantic_release_evidence_shape": (
                    BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4
                ),
            }
        )
    with pytest.raises(ValidationError):
        BfclV4PublicV2LiveExecutionConfig(
            **{
                **config.model_dump(mode="python"),
                "semantic_release_evidence_shape": "unsupported-v5",
            }
        )
