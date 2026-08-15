from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET,
    BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
    BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,
    BFCL_V4_PUBLIC_LIVE_INFERENCE,
    BFCL_V4_PUBLIC_LIVE_REVISION,
    BFCL_V4_PUBLIC_LIVE_RUNTIME,
    BFCL_V4_PUBLIC_LIVE_TOKENIZER,
    BFCL_V4_PUBLIC_LIVE_TOKENIZER_REVISION,
    freeze_bfcl_v4_public_live_execution_config,
    observe_bfcl_v4_public_live_model_catalog,
)

BACKEND = "a" * 64
SERIALIZER = "b" * 64
PARSER = "c" * 64
TRANSPORT = "d" * 64


def _observation():
    return observe_bfcl_v4_public_live_model_catalog(
        (
            "another-open-route",
            BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,
            BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
        ),
        observed_at_utc="2026-08-15T16:00:00Z",
    )


def _config():
    return freeze_bfcl_v4_public_live_execution_config(
        catalog_observation=_observation(),
        backend="litellm-openai-compatible-native",
        backend_fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
    )


def test_corrected_stochastic_live_inference_and_exact_attempt_budget_are_frozen() -> None:
    assert BFCL_V4_PUBLIC_LIVE_INFERENCE.temperature == 0.2
    assert BFCL_V4_PUBLIC_LIVE_INFERENCE.top_p == 0.95
    assert BFCL_V4_PUBLIC_LIVE_INFERENCE.max_output_tokens == 2_048
    assert BFCL_V4_PUBLIC_LIVE_INFERENCE.timeout_seconds == 120.0
    assert BFCL_V4_PUBLIC_LIVE_INFERENCE.stop_sequences == ()

    assert BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.max_attempts == 100
    assert BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.max_tokens_per_attempt == 32_768
    assert BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.max_total_tokens == 3_276_800


def test_live_config_activates_only_an_advertised_exact_route() -> None:
    observation = _observation()
    config = _config()

    assert observation.selected_model_route == BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
    assert observation.direct_route_advertised is True
    assert observation.fallback_route_advertised is True
    assert observation.fallback_used is False
    assert observation.availability_only is True
    assert observation.credentials_recorded is False
    assert config.selected_model_route == BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
    assert config.model_spec.model == BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
    assert config.model_spec.inference == BFCL_V4_PUBLIC_LIVE_INFERENCE
    assert config.attempt_budget == BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET
    assert config.provider_attempts_per_call == 1
    assert config.automatic_retries_allowed is False
    assert config.pure_at_b_requires_stochastic_multi_seed_sampling is True

    assert config.route_frozen_before_any_score_bearing_call is True
    assert config.same_exact_route_across_three_replicates_required is True
    assert config.direct_and_fallback_weight_equivalence_assumed is False

    with pytest.raises(ValidationError, match="neither frozen BFCL route"):
        observe_bfcl_v4_public_live_model_catalog(
            ("another-open-route",),
            observed_at_utc="2026-08-15T16:00:00Z",
        )


def test_catalog_observation_is_canonical_availability_metadata_not_identity_evidence() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        observe_bfcl_v4_public_live_model_catalog(
            (BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE, "another-open-route"),
            observed_at_utc="2026-08-15T16:00:00Z",
        )
    with pytest.raises(ValidationError, match="RFC3339 UTC"):
        observe_bfcl_v4_public_live_model_catalog(
            (BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,),
            observed_at_utc="2026-08-15 16:00:00",
        )


def test_dashscope_route_is_only_a_catalog_driven_fallback() -> None:
    observation = observe_bfcl_v4_public_live_model_catalog(
        (BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,),
        observed_at_utc="2026-08-15T16:00:00Z",
    )
    config = freeze_bfcl_v4_public_live_execution_config(
        catalog_observation=observation,
        backend="litellm-openai-compatible-native",
        backend_fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
    )

    assert observation.direct_route_advertised is False
    assert observation.fallback_route_advertised is True
    assert observation.fallback_used is True
    assert observation.selected_model_route == BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
    assert config.selected_model_route == BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
    assert config.model_spec.model == BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
    assert config.same_provider_weights_attested is False


def test_gateway_opaque_labels_cannot_be_mistaken_for_weight_or_runtime_pins() -> None:
    config = _config()
    spec = config.model_spec

    assert spec.revision == BFCL_V4_PUBLIC_LIVE_REVISION
    assert spec.tokenizer == BFCL_V4_PUBLIC_LIVE_TOKENIZER
    assert spec.tokenizer_revision == BFCL_V4_PUBLIC_LIVE_TOKENIZER_REVISION
    assert spec.runtime == BFCL_V4_PUBLIC_LIVE_RUNTIME
    assert "opaque" in spec.revision
    assert "unavailable" in spec.revision
    assert "opaque" in spec.tokenizer
    assert "opaque" in spec.runtime
    assert config.gateway_revision_is_opaque_limitation is True
    assert config.gateway_tokenizer_is_opaque_limitation is True
    assert config.gateway_runtime_is_opaque_limitation is True
    assert config.actual_weight_revision_attested is False
    assert config.tokenizer_identity_attested is False
    assert config.serving_runtime_attested is False
    assert config.same_provider_weights_attested is False


def test_all_four_native_implementation_identities_are_frozen_separately() -> None:
    config = _config()

    assert (
        config.backend_fingerprint,
        config.serializer_fingerprint,
        config.parser_fingerprint,
        config.transport_fingerprint,
    ) == (BACKEND, SERIALIZER, PARSER, TRANSPORT)
    assert config.model_spec.backend_fingerprint == BACKEND

    with pytest.raises(ValidationError, match="match pattern"):
        freeze_bfcl_v4_public_live_execution_config(
            catalog_observation=_observation(),
            backend="litellm-openai-compatible-native",
            backend_fingerprint=BACKEND,
            serializer_fingerprint="not-a-fingerprint",
            parser_fingerprint=PARSER,
            transport_fingerprint=TRANSPORT,
        )
