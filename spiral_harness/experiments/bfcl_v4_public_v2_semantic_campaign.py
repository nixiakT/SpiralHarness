"""One-shot independent semantic-review campaign for BFCL public v2.

The campaign performs two frozen open-model reviews and invokes a third,
different open model only when the two primary partitions disagree.  It emits
credential-free evidence and a development-only semantic release; it never
starts the benchmark solver campaign.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    load_bfcl_v4_public_pilot,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    build_bfcl_v4_public_v2_semantic_reviewer_projection,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticProjection,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release import (
    issue_bfcl_v4_public_v2_semantic_development_release,
    verify_bfcl_v4_public_v2_semantic_development_release,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticDevelopmentRelease,
    BfclV4PublicV2SemanticReleaseError,
    BfclV4PublicV2SemanticReleaseFailureStage,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_review_live import (
    BfclV4PublicV2SemanticReviewRole,
    _post_raw_http,
    run_bfcl_v4_public_v2_semantic_review,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.providers.openai_native_function import (
    OpenAICompatibleNativeFunctionBackend,
)

BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS = ("DeepSeek-V3", "GLM-5.2")
BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL = "MiniMax-M2.5"
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_SEEDS_U64 = (
    2_026_081_701,
    2_026_081_702,
    2_026_081_703,
)

_CatalogLoader = Callable[[str, str, float], tuple[str, ...]]
_RawTransport = Callable[[str, str, str, dict[str, Any], float], bytes]
_SessionSink = Callable[
    [BfclV4PublicV2SemanticReviewRole, BfclV4PublicV2SemanticReviewSessionEvidence],
    None,
]


class BfclV4PublicV2SemanticCampaignError(RuntimeError):
    """The prospective semantic-review campaign could not issue authority."""


class BfclV4PublicV2SemanticCampaignEvidence(ImmutableModel):
    """Complete credential-free evidence for one review/release campaign."""

    schema_version: Literal["1"] = "1"
    projection: BfclV4PublicV2SemanticProjection
    catalog_observation_sha256: Sha256
    advertised_model_count: int = Field(ge=3, strict=True)
    primary_models: tuple[NonEmptyStr, NonEmptyStr]
    adjudicator_model: NonEmptyStr
    primary_sessions: tuple[
        BfclV4PublicV2SemanticReviewSessionEvidence,
        BfclV4PublicV2SemanticReviewSessionEvidence,
    ]
    adjudicator_session: BfclV4PublicV2SemanticReviewSessionEvidence | None = None
    release: BfclV4PublicV2SemanticDevelopmentRelease
    review_attempt_count: Literal[2, 3]
    total_review_input_tokens: int = Field(ge=0, strict=True)
    total_review_output_tokens: int = Field(ge=0, strict=True)
    total_review_tokens: int = Field(ge=0, strict=True)
    release_reverified_before_return: Literal[True] = True
    automatic_retries_used: Literal[0] = 0
    credentials_recorded: Literal[False] = False
    hmac_secret_recorded: Literal[False] = False
    solver_campaign_started: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_sessions_release_and_usage(self) -> Self:
        expected_models = (*BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS,)
        if self.primary_models != expected_models:
            raise ValueError("semantic primary model routes differ from the frozen campaign")
        if self.adjudicator_model != BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL:
            raise ValueError("semantic adjudicator route differs from the frozen campaign")
        sessions = (*self.primary_sessions,)
        if self.adjudicator_session is not None:
            sessions = (*sessions, self.adjudicator_session)
        if len(sessions) != self.review_attempt_count:
            raise ValueError("semantic session count differs from consumed review attempts")
        if tuple(item.execution_attestation.requested_model for item in sessions) != (
            *self.primary_models,
            *((self.adjudicator_model,) if self.adjudicator_session is not None else ()),
        ):
            raise ValueError("semantic sessions differ from their frozen reviewer roles")
        usage = tuple(item.execution_attestation for item in sessions)
        expected_usage = (
            sum(item.input_tokens for item in usage),
            sum(item.output_tokens for item in usage),
            sum(item.total_tokens for item in usage),
        )
        if expected_usage != (
            self.total_review_input_tokens,
            self.total_review_output_tokens,
            self.total_review_tokens,
        ):
            raise ValueError("semantic campaign usage differs from its session evidence")
        if self.total_review_tokens != (
            self.total_review_input_tokens + self.total_review_output_tokens
        ):
            raise ValueError("semantic campaign token totals are inconsistent")
        if (self.adjudicator_session is not None) is not (self.release.reviewer_count == 3):
            raise ValueError("semantic release adjudication differs from the live sessions")
        if self.release.reviewer_packet_ref != self.projection.reviewer_packet.ref:
            raise ValueError("semantic release binds another reviewer packet")
        if self.release.trusted_mapping_ref != self.projection.trusted_mapping.ref:
            raise ValueError("semantic release binds another trusted mapping")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _load_catalog(base_url: str, api_key: str, timeout_seconds: float) -> tuple[str, ...]:
    backend = OpenAICompatibleNativeFunctionBackend.from_endpoint(
        base_url=base_url,
        api_key=api_key,
    )
    return backend.list_models(timeout_seconds=timeout_seconds)


def _checked_catalog(model_ids: tuple[str, ...]) -> tuple[str, ...]:
    if type(model_ids) is not tuple or any(
        type(item) is not str or not item or item != item.strip() for item in model_ids
    ):
        raise BfclV4PublicV2SemanticCampaignError(
            "semantic review catalog observation is malformed"
        )
    normalized = tuple(sorted(set(model_ids)))
    required = {
        *BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS,
        BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,
    }
    if not required.issubset(normalized):
        raise BfclV4PublicV2SemanticCampaignError(
            "one or more frozen open semantic-review routes are unavailable"
        )
    return normalized


def _review(
    projection: BfclV4PublicV2SemanticProjection,
    *,
    runtime_hmac_secret: bytes,
    base_url: str,
    api_key: str,
    model: str,
    role: BfclV4PublicV2SemanticReviewRole,
    seed: int,
    catalog_observation_sha256: str,
    timeout_seconds: float,
    transport: _RawTransport,
    session_sink: _SessionSink | None,
) -> BfclV4PublicV2SemanticReviewSessionEvidence:
    session = run_bfcl_v4_public_v2_semantic_review(
        projection.reviewer_packet,
        projection.trusted_mapping,
        runtime_hmac_secret=runtime_hmac_secret,
        base_url=base_url,
        api_key=api_key,
        model=model,
        role=role,
        generation_seed_u64=seed,
        catalog_observation_sha256=catalog_observation_sha256,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    if session_sink is not None:
        session_sink(role, session)
    return session


def run_bfcl_v4_public_v2_semantic_campaign(
    *,
    checkout: str | Path,
    runtime_hmac_secret: bytes,
    base_url: str,
    api_key: str,
    catalog_timeout_seconds: float = 30.0,
    review_timeout_seconds: float = 300.0,
    catalog_loader: _CatalogLoader = _load_catalog,
    transport: _RawTransport,
    session_sink: _SessionSink | None = None,
) -> BfclV4PublicV2SemanticCampaignEvidence:
    """Run the frozen two-review/optional-adjudicator prerequisite exactly once."""

    try:
        v1 = load_bfcl_v4_public_pilot(checkout)
        v2 = load_bfcl_v4_public_development_v2(checkout)
        projection = build_bfcl_v4_public_v2_semantic_reviewer_projection(
            v1,
            v2,
            runtime_hmac_secret=runtime_hmac_secret,
        )
        catalog = _checked_catalog(
            catalog_loader(base_url, api_key, float(catalog_timeout_seconds))
        )
    except BfclV4PublicV2SemanticCampaignError:
        raise
    except Exception:
        raise BfclV4PublicV2SemanticCampaignError(
            "BFCL v2 semantic projection or catalog activation failed"
        ) from None
    catalog_sha256 = canonical_sha256(catalog)

    primary = (
        _review(
            projection,
            runtime_hmac_secret=runtime_hmac_secret,
            base_url=base_url,
            api_key=api_key,
            model=BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS[0],
            role=BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE,
            seed=BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_SEEDS_U64[0],
            catalog_observation_sha256=catalog_sha256,
            timeout_seconds=review_timeout_seconds,
            transport=transport,
            session_sink=session_sink,
        ),
        _review(
            projection,
            runtime_hmac_secret=runtime_hmac_secret,
            base_url=base_url,
            api_key=api_key,
            model=BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS[1],
            role=BfclV4PublicV2SemanticReviewRole.PRIMARY_TWO,
            seed=BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_SEEDS_U64[1],
            catalog_observation_sha256=catalog_sha256,
            timeout_seconds=review_timeout_seconds,
            transport=transport,
            session_sink=session_sink,
        ),
    )

    adjudicator = None
    try:
        release = issue_bfcl_v4_public_v2_semantic_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            primary,
            runtime_hmac_secret=runtime_hmac_secret,
        )
    except BfclV4PublicV2SemanticReleaseError as error:
        if (
            error.failure_stage
            is not BfclV4PublicV2SemanticReleaseFailureStage.DISAGREEMENT_ARBITRATION
        ):
            raise
        adjudicator = _review(
            projection,
            runtime_hmac_secret=runtime_hmac_secret,
            base_url=base_url,
            api_key=api_key,
            model=BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,
            role=BfclV4PublicV2SemanticReviewRole.ADJUDICATOR,
            seed=BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_SEEDS_U64[2],
            catalog_observation_sha256=catalog_sha256,
            timeout_seconds=review_timeout_seconds,
            transport=transport,
            session_sink=session_sink,
        )
        release = issue_bfcl_v4_public_v2_semantic_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            primary,
            runtime_hmac_secret=runtime_hmac_secret,
            adjudicator_session=adjudicator,
        )

    verified = verify_bfcl_v4_public_v2_semantic_development_release(
        release,
        projection.reviewer_packet,
        projection.trusted_mapping,
        primary,
        runtime_hmac_secret=runtime_hmac_secret,
        adjudicator_session=adjudicator,
    )
    sessions = (*primary, *((adjudicator,) if adjudicator is not None else ()))
    attestations = tuple(item.execution_attestation for item in sessions)
    return BfclV4PublicV2SemanticCampaignEvidence(
        projection=projection,
        catalog_observation_sha256=catalog_sha256,
        advertised_model_count=len(catalog),
        primary_models=BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS,
        adjudicator_model=BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,
        primary_sessions=primary,
        adjudicator_session=adjudicator,
        release=verified,
        review_attempt_count=len(sessions),
        total_review_input_tokens=sum(item.input_tokens for item in attestations),
        total_review_output_tokens=sum(item.output_tokens for item in attestations),
        total_review_tokens=sum(item.total_tokens for item in attestations),
    )


def run_bfcl_v4_public_v2_semantic_campaign_live(**kwargs: Any):
    """Use the direct no-redirect transport without exposing it in the main signature."""

    return run_bfcl_v4_public_v2_semantic_campaign(transport=_post_raw_http, **kwargs)


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_") or name.startswith("Bfcl") or name.startswith("run_bfcl")
]
