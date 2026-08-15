from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release as release_subject
from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import load_bfcl_v4_public_pilot
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    build_bfcl_v4_public_v2_semantic_reviewer_projection,
    verify_bfcl_v4_public_v2_semantic_distribution_integrity,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticPrivateBoundary,
    BfclV4PublicV2SemanticReviewerAssignment,
    BfclV4PublicV2SemanticReviewerDeclaration,
    BfclV4PublicV2SemanticReviewerKind,
    BfclV4PublicV2SemanticReviewerResponse,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticReleaseError,
    BfclV4PublicV2SemanticReleaseFailureStage,
    BfclV4PublicV2SemanticReviewExecutionAttestation,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)
from spiral_harness.core.canonical import canonical_sha256

_DEFAULT_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SECRET = b"semantic-development-release-test" * 2
_PROMPT_SHA256 = hashlib.sha256(b"frozen semantic reviewer prompt").hexdigest()


@pytest.fixture(scope="module")
def projection():
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for semantic release test")
    v1_loaded = load_bfcl_v4_public_pilot(checkout)
    v2_loaded = load_bfcl_v4_public_development_v2(checkout)
    return build_bfcl_v4_public_v2_semantic_reviewer_projection(
        v1_loaded,
        v2_loaded,
        runtime_hmac_secret=_SECRET,
    )


def _declaration(index: int) -> BfclV4PublicV2SemanticReviewerDeclaration:
    return BfclV4PublicV2SemanticReviewerDeclaration(
        declared_reviewer_kind=BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED,
        declared_principal_id=f"semantic-review-model-{index}",
        model_provider="open-compatible-development-gateway",
        model_name=f"open-review-model-{index}",
        model_revision=f"gateway-route-revision-{index}",
        served_model_weights_identity_sha256=hashlib.sha256(
            f"declared-served-model-identity-{index}".encode()
        ).hexdigest(),
        model_prompt_sha256=_PROMPT_SHA256,
        model_generation_seed_u64=10_000 + index,
    )


def _session(
    projection,
    index: int,
    *,
    merged_opaque_ids: tuple[str, str] | None = None,
) -> BfclV4PublicV2SemanticReviewSessionEvidence:
    declaration = _declaration(index)
    receipt = verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        declaration,
        runtime_hmac_secret=_SECRET,
    )
    labels = {
        item.opaque_item_id: f"review-{index}-family-{position}"
        for position, item in enumerate(projection.reviewer_packet.items)
    }
    if merged_opaque_ids is not None:
        labels[merged_opaque_ids[1]] = labels[merged_opaque_ids[0]]
    response = BfclV4PublicV2SemanticReviewerResponse(
        reviewer_packet_ref=projection.reviewer_packet.ref,
        reviewer_declaration_ref=declaration.ref,
        assignments=tuple(
            BfclV4PublicV2SemanticReviewerAssignment(
                opaque_item_id=item.opaque_item_id,
                semantic_family_label=labels[item.opaque_item_id],
                rationale="Conservative comparison used only the projected question and schemas.",
            )
            for item in projection.reviewer_packet.items
        ),
    )
    attestation = BfclV4PublicV2SemanticReviewExecutionAttestation(
        reviewer_packet_ref=projection.reviewer_packet.ref,
        trusted_mapping_ref=projection.trusted_mapping.ref,
        reviewer_declaration_ref=declaration.ref,
        distribution_receipt_ref=receipt.ref,
        reviewer_response_ref=response.ref,
        provider_origin_sha256=hashlib.sha256(b"development-provider-origin").hexdigest(),
        request_body_sha256=hashlib.sha256(f"request-{index}".encode()).hexdigest(),
        raw_response_body_sha256=hashlib.sha256(f"response-{index}".encode()).hexdigest(),
        provider_response_id=f"provider-response-{index}",
        requested_model=declaration.model_name,
        provider_reported_model=declaration.model_name,
        provider_seed_u64=declaration.model_generation_seed_u64,
    )
    return BfclV4PublicV2SemanticReviewSessionEvidence(
        reviewer_declaration=declaration,
        distribution_receipt=receipt,
        reviewer_response=response,
        execution_attestation=attestation,
    )


def _same_boundary_pair(projection) -> tuple[str, str]:
    entries = tuple(
        item
        for item in projection.trusted_mapping.entries
        if item.private_boundary_label is BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT
    )
    return entries[0].opaque_item_id, entries[1].opaque_item_id


def _cross_boundary_pair(projection) -> tuple[str, str]:
    left = next(
        item
        for item in projection.trusted_mapping.entries
        if item.private_boundary_label is BfclV4PublicV2SemanticPrivateBoundary.FIT
    )
    right = next(
        item
        for item in projection.trusted_mapping.entries
        if item.private_boundary_label is BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT
    )
    return left.opaque_item_id, right.opaque_item_id


def test_two_independent_consensus_reviews_issue_hmac_bound_release(projection) -> None:
    sessions = (_session(projection, 1), _session(projection, 2))

    release = release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        sessions,
        runtime_hmac_secret=_SECRET,
    )
    checked = release_subject.verify_bfcl_v4_public_v2_semantic_development_release(
        release,
        projection.reviewer_packet,
        projection.trusted_mapping,
        sessions,
        runtime_hmac_secret=_SECRET,
    )

    assert checked == release
    assert release.score_bearing_execution_allowed is True
    assert release.public_development_only is True
    assert release.hidden_test_evidence is False
    assert release.reviewer_count == 2
    assert release.primary_pairwise_disagreement_count == 0
    assert release.final_semantic_family_count == 40
    assert release.reviewer_identity_externally_verified is False
    assert release.served_model_weights_cryptographically_verified is False
    assert release.ref.sha256 == canonical_sha256(release)


def test_disagreement_requires_distinct_third_review_and_uses_its_partition(projection) -> None:
    merged = _same_boundary_pair(projection)
    sessions = (_session(projection, 1), _session(projection, 2, merged_opaque_ids=merged))
    with pytest.raises(BfclV4PublicV2SemanticReleaseError) as captured:
        release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            sessions,
            runtime_hmac_secret=_SECRET,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2SemanticReleaseFailureStage.DISAGREEMENT_ARBITRATION
    )

    release = release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        sessions,
        runtime_hmac_secret=_SECRET,
        adjudicator_session=_session(projection, 3),
    )
    assert release.reviewer_count == 3
    assert release.primary_pairwise_disagreement_count == 1
    assert release.final_semantic_family_count == 40
    assert release.adjudicator_review_ref is not None


def test_cross_boundary_final_family_rejects_release(projection) -> None:
    merged = _cross_boundary_pair(projection)
    sessions = (
        _session(projection, 1, merged_opaque_ids=merged),
        _session(projection, 2, merged_opaque_ids=merged),
    )

    with pytest.raises(BfclV4PublicV2SemanticReleaseError) as captured:
        release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            sessions,
            runtime_hmac_secret=_SECRET,
        )
    assert captured.value.failure_stage is BfclV4PublicV2SemanticReleaseFailureStage.FAMILY_BOUNDARY


def test_same_declared_model_identity_and_tampered_release_fail_closed(projection) -> None:
    first = _session(projection, 1)
    sessions = (first, first)
    with pytest.raises(BfclV4PublicV2SemanticReleaseError) as captured:
        release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            sessions,
            runtime_hmac_secret=_SECRET,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2SemanticReleaseFailureStage.REVIEW_INDEPENDENCE
    )

    valid_sessions = (_session(projection, 1), _session(projection, 2))
    release = release_subject.issue_bfcl_v4_public_v2_semantic_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        valid_sessions,
        runtime_hmac_secret=_SECRET,
    )
    forged = release.model_copy(update={"release_authority_hmac_sha256": "0" * 64})
    with pytest.raises(BfclV4PublicV2SemanticReleaseError) as captured:
        release_subject.verify_bfcl_v4_public_v2_semantic_development_release(
            forged,
            projection.reviewer_packet,
            projection.trusted_mapping,
            valid_sessions,
            runtime_hmac_secret=_SECRET,
        )
    assert (
        captured.value.failure_stage is BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING
    )
