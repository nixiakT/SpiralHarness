"""Shared real-verification fixtures for BFCL v2 authority boundaries."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import load_bfcl_v4_public_pilot
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority import (
    BfclV4PublicV2VerifiedSemanticReleaseCapability,
    verify_bfcl_v4_public_v2_legacy_semantic_authority,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    build_bfcl_v4_public_v2_semantic_reviewer_projection,
    verify_bfcl_v4_public_v2_semantic_distribution_integrity,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticReviewerAssignment,
    BfclV4PublicV2SemanticReviewerDeclaration,
    BfclV4PublicV2SemanticReviewerKind,
    BfclV4PublicV2SemanticReviewerResponse,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release import (
    issue_bfcl_v4_public_v2_semantic_development_release,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticDevelopmentRelease,
    BfclV4PublicV2SemanticReviewExecutionAttestation,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)

_DEFAULT_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SECRET = b"shared-real-semantic-authority-test-secret" * 2
_PROMPT_SHA256 = hashlib.sha256(b"shared frozen semantic reviewer prompt").hexdigest()


@dataclass(frozen=True, slots=True)
class BfclV2VerifiedLegacyAuthorityFixture:
    capability: BfclV4PublicV2VerifiedSemanticReleaseCapability
    release: BfclV4PublicV2SemanticDevelopmentRelease
    reviewer_packet: object
    trusted_mapping: object
    primary_sessions: tuple[BfclV4PublicV2SemanticReviewSessionEvidence, ...]
    runtime_hmac_secret: bytes


def _session(projection: object, index: int) -> BfclV4PublicV2SemanticReviewSessionEvidence:
    declaration = BfclV4PublicV2SemanticReviewerDeclaration(
        declared_reviewer_kind=BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED,
        declared_principal_id=f"shared-authority-reviewer-{index}",
        model_provider="shared-authority-provider",
        model_name=f"shared-authority-model-{index}",
        model_revision=f"shared-authority-revision-{index}",
        served_model_weights_identity_sha256=hashlib.sha256(
            f"shared-authority-weights-{index}".encode()
        ).hexdigest(),
        model_prompt_sha256=_PROMPT_SHA256,
        model_generation_seed_u64=50_000 + index,
    )
    receipt = verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        declaration,
        runtime_hmac_secret=_SECRET,
    )
    response = BfclV4PublicV2SemanticReviewerResponse(
        reviewer_packet_ref=projection.reviewer_packet.ref,
        reviewer_declaration_ref=declaration.ref,
        assignments=tuple(
            BfclV4PublicV2SemanticReviewerAssignment(
                opaque_item_id=item.opaque_item_id,
                semantic_family_label=f"singleton-{position}",
                rationale="Conservative singleton decision from projected public content.",
            )
            for position, item in enumerate(projection.reviewer_packet.items)
        ),
    )
    attestation = BfclV4PublicV2SemanticReviewExecutionAttestation(
        reviewer_packet_ref=projection.reviewer_packet.ref,
        trusted_mapping_ref=projection.trusted_mapping.ref,
        reviewer_declaration_ref=declaration.ref,
        distribution_receipt_ref=receipt.ref,
        reviewer_response_ref=response.ref,
        provider_origin_sha256=hashlib.sha256(b"shared-authority-origin").hexdigest(),
        request_body_sha256=hashlib.sha256(f"shared-request-{index}".encode()).hexdigest(),
        raw_response_body_sha256=hashlib.sha256(f"shared-response-{index}".encode()).hexdigest(),
        provider_response_id=f"shared-authority-response-{index}",
        requested_model=declaration.model_name,
        provider_reported_model=declaration.model_name,
        provider_seed_u64=declaration.model_generation_seed_u64,
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
    )
    return BfclV4PublicV2SemanticReviewSessionEvidence(
        reviewer_declaration=declaration,
        distribution_receipt=receipt,
        reviewer_response=response,
        execution_attestation=attestation,
    )


@pytest.fixture(scope="session")
def bfcl_v2_verified_legacy_authority() -> BfclV2VerifiedLegacyAuthorityFixture:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for real semantic authority fixture")
    projection = build_bfcl_v4_public_v2_semantic_reviewer_projection(
        load_bfcl_v4_public_pilot(checkout),
        load_bfcl_v4_public_development_v2(checkout),
        runtime_hmac_secret=_SECRET,
    )
    primary_sessions = (_session(projection, 1), _session(projection, 2))
    release = issue_bfcl_v4_public_v2_semantic_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        primary_sessions,
        runtime_hmac_secret=_SECRET,
    )
    capability = verify_bfcl_v4_public_v2_legacy_semantic_authority(
        release,
        projection.reviewer_packet,
        projection.trusted_mapping,
        primary_sessions,
        runtime_hmac_secret=_SECRET,
    )
    return BfclV2VerifiedLegacyAuthorityFixture(
        capability=capability,
        release=release,
        reviewer_packet=projection.reviewer_packet,
        trusted_mapping=projection.trusted_mapping,
        primary_sessions=primary_sessions,
        runtime_hmac_secret=_SECRET,
    )
