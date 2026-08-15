"""Prospective live configuration for the BFCL V4 public campaign.

This module performs no network request.  A caller may activate the frozen
route only by supplying an availability-only observation obtained from the
gateway's authenticated ``GET /v1/models`` endpoint.  Opaque gateway labels
are deliberately not represented as weight, tokenizer, or runtime attestations.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import AttemptBudget, FrozenModelSpec, InferenceConfig

BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE = "qwen36-35b-a3b"
BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE = "dashscope/qwen36-35b-a3b"
BFCL_V4_PUBLIC_LIVE_REVISION = (
    "gateway-declared/opaque-weight-revision-unavailable@bfcl-public-live-v1"
)
BFCL_V4_PUBLIC_LIVE_TOKENIZER = (
    "gateway-declared/opaque-tokenizer-identity-unavailable@bfcl-public-live-v1"
)
BFCL_V4_PUBLIC_LIVE_TOKENIZER_REVISION = (
    "gateway-declared/opaque-tokenizer-revision-unavailable@bfcl-public-live-v1"
)
BFCL_V4_PUBLIC_LIVE_RUNTIME = (
    "gateway-declared/opaque-serving-runtime-unavailable@bfcl-public-live-v1"
)

BFCL_V4_PUBLIC_LIVE_INFERENCE = InferenceConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=2_048,
    timeout_seconds=120.0,
    stop_sequences=(),
)
BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET = AttemptBudget(
    max_attempts=100,
    max_tokens_per_attempt=32_768,
    max_total_tokens=3_276_800,
)

_RFC3339_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")


class BfclV4PublicLiveModelCatalogObservation(ImmutableModel):
    """Credential-free availability observation; not a model-identity claim."""

    schema_version: Literal["1"] = "1"
    source: Literal["authenticated-direct-get-/v1/models"] = "authenticated-direct-get-/v1/models"
    observed_at_utc: NonEmptyStr
    advertised_model_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    advertised_model_ids_sha256: Sha256
    selected_model_route: Literal["qwen36-35b-a3b", "dashscope/qwen36-35b-a3b"]
    selected_route_advertised: Literal[True] = True
    direct_route_advertised: bool
    fallback_route_advertised: bool
    fallback_used: bool
    exact_direct_route_preferred: Literal[True] = True
    direct_and_fallback_weight_equivalence_assumed: Literal[False] = False
    availability_only: Literal[True] = True
    catalog_query_runtime_attested: Literal[False] = False
    catalog_response_authenticity_attested: Literal[False] = False
    credentials_recorded: Literal[False] = False
    provider_weights_identified: Literal[False] = False
    tokenizer_identified: Literal[False] = False
    serving_runtime_identified: Literal[False] = False

    @model_validator(mode="after")
    def _bind_availability_only_catalog(self) -> Self:
        if _RFC3339_UTC.fullmatch(self.observed_at_utc) is None:
            raise ValueError("model catalog observation time must be RFC3339 UTC")
        expected_ids = tuple(sorted(set(self.advertised_model_ids)))
        if self.advertised_model_ids != expected_ids:
            raise ValueError("advertised model IDs must be sorted and unique")
        direct = BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE in self.advertised_model_ids
        fallback = BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE in self.advertised_model_ids
        expected_route = (
            BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
            if direct
            else (BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE if fallback else None)
        )
        if expected_route is None:
            raise ValueError("neither frozen BFCL route is present in the /models observation")
        if (
            self.selected_model_route != expected_route
            or self.direct_route_advertised is not direct
            or self.fallback_route_advertised is not fallback
            or self.fallback_used is not (not direct and fallback)
        ):
            raise ValueError("selected BFCL route differs from direct-first catalog policy")
        if self.advertised_model_ids_sha256 != canonical_sha256(self.advertised_model_ids):
            raise ValueError("model catalog digest differs from the exact advertised IDs")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicLiveExecutionConfig(ImmutableModel):
    """Frozen execution inputs plus explicit gateway-identity limitations."""

    schema_version: Literal["1"] = "1"
    catalog_observation: BfclV4PublicLiveModelCatalogObservation
    catalog_observation_fingerprint: Sha256
    model_spec: FrozenModelSpec
    model_spec_fingerprint: Sha256
    backend_fingerprint: Sha256
    serializer_fingerprint: Sha256
    parser_fingerprint: Sha256
    transport_fingerprint: Sha256
    inference_fingerprint: Sha256
    attempt_budget: AttemptBudget
    attempt_budget_fingerprint: Sha256
    selected_model_route: Literal["qwen36-35b-a3b", "dashscope/qwen36-35b-a3b"]
    provider_attempts_per_call: Literal[1] = 1
    automatic_retries_allowed: Literal[False] = False
    model_route_activation_requires_catalog_observation: Literal[True] = True
    route_availability_runtime_attested: Literal[False] = False
    route_frozen_before_any_score_bearing_call: Literal[True] = True
    same_exact_route_across_three_replicates_required: Literal[True] = True
    direct_and_fallback_weight_equivalence_assumed: Literal[False] = False
    pure_at_b_requires_stochastic_multi_seed_sampling: Literal[True] = True
    gateway_revision_is_opaque_limitation: Literal[True] = True
    gateway_tokenizer_is_opaque_limitation: Literal[True] = True
    gateway_runtime_is_opaque_limitation: Literal[True] = True
    actual_weight_revision_attested: Literal[False] = False
    tokenizer_identity_attested: Literal[False] = False
    serving_runtime_attested: Literal[False] = False
    same_provider_weights_attested: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_live_configuration(self) -> Self:
        if self.catalog_observation_fingerprint != self.catalog_observation.fingerprint:
            raise ValueError("catalog observation fingerprint changed")
        expected_spec = (
            self.catalog_observation.selected_model_route,
            BFCL_V4_PUBLIC_LIVE_REVISION,
            BFCL_V4_PUBLIC_LIVE_TOKENIZER,
            BFCL_V4_PUBLIC_LIVE_TOKENIZER_REVISION,
            BFCL_V4_PUBLIC_LIVE_RUNTIME,
            BFCL_V4_PUBLIC_LIVE_INFERENCE,
        )
        actual_spec = (
            self.model_spec.model,
            self.model_spec.revision,
            self.model_spec.tokenizer,
            self.model_spec.tokenizer_revision,
            self.model_spec.runtime,
            self.model_spec.inference,
        )
        if actual_spec != expected_spec:
            raise ValueError("model spec differs from the frozen BFCL live route or inference")
        if self.selected_model_route != self.catalog_observation.selected_model_route:
            raise ValueError("execution route differs from its direct-first catalog selection")
        if (
            self.model_spec_fingerprint != self.model_spec.fingerprint
            or self.backend_fingerprint != self.model_spec.backend_fingerprint
            or self.inference_fingerprint != self.model_spec.inference_fingerprint
        ):
            raise ValueError("live model/backend/inference fingerprints changed")
        if (
            self.attempt_budget != BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET
            or self.attempt_budget_fingerprint != self.attempt_budget.fingerprint
        ):
            raise ValueError("attempt budget differs from the frozen 100-call campaign budget")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def observe_bfcl_v4_public_live_model_catalog(
    advertised_model_ids: tuple[str, ...],
    *,
    observed_at_utc: str,
) -> BfclV4PublicLiveModelCatalogObservation:
    """Freeze a sanitized list-models result without performing the request."""

    if not isinstance(advertised_model_ids, tuple):
        raise TypeError("advertised_model_ids must be the exact tuple returned by list_models")
    direct = BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE in advertised_model_ids
    fallback = BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE in advertised_model_ids
    selected = (
        BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
        if direct
        else BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
    )
    return BfclV4PublicLiveModelCatalogObservation(
        observed_at_utc=observed_at_utc,
        advertised_model_ids=advertised_model_ids,
        advertised_model_ids_sha256=canonical_sha256(advertised_model_ids),
        selected_model_route=selected,
        direct_route_advertised=direct,
        fallback_route_advertised=fallback,
        fallback_used=not direct and fallback,
    )


def freeze_bfcl_v4_public_live_execution_config(
    *,
    catalog_observation: BfclV4PublicLiveModelCatalogObservation,
    backend: str,
    backend_fingerprint: str,
    serializer_fingerprint: str,
    parser_fingerprint: str,
    transport_fingerprint: str,
) -> BfclV4PublicLiveExecutionConfig:
    """Bind one confirmed route to native implementation identities and budgets."""

    checked_catalog = BfclV4PublicLiveModelCatalogObservation.model_validate(
        catalog_observation,
        strict=True,
    )
    spec = FrozenModelSpec(
        backend=backend,
        backend_fingerprint=backend_fingerprint,
        model=checked_catalog.selected_model_route,
        revision=BFCL_V4_PUBLIC_LIVE_REVISION,
        tokenizer=BFCL_V4_PUBLIC_LIVE_TOKENIZER,
        tokenizer_revision=BFCL_V4_PUBLIC_LIVE_TOKENIZER_REVISION,
        runtime=BFCL_V4_PUBLIC_LIVE_RUNTIME,
        inference=BFCL_V4_PUBLIC_LIVE_INFERENCE,
    )
    return BfclV4PublicLiveExecutionConfig(
        catalog_observation=checked_catalog,
        catalog_observation_fingerprint=checked_catalog.fingerprint,
        model_spec=spec,
        model_spec_fingerprint=spec.fingerprint,
        backend_fingerprint=backend_fingerprint,
        serializer_fingerprint=serializer_fingerprint,
        parser_fingerprint=parser_fingerprint,
        transport_fingerprint=transport_fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        attempt_budget=BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET,
        attempt_budget_fingerprint=BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.fingerprint,
        selected_model_route=checked_catalog.selected_model_route,
    )


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
