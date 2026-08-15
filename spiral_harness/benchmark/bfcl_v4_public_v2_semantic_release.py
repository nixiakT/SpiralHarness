"""Issue a HMAC-bound, development-only BFCL v2 semantic release."""

from __future__ import annotations

import hmac
import json
from collections import defaultdict
from hashlib import sha256
from itertools import combinations

from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_semantic_audit import (
    BFCL_V4_SEMANTIC_AUDIT_UNIVERSE,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    validate_bfcl_v4_public_v2_semantic_reviewer_response,
    verify_bfcl_v4_public_v2_semantic_distribution_receipt,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES,
    BfclV4PublicV2SemanticPrivateBoundary,
    BfclV4PublicV2SemanticReviewerKind,
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticTrustedMapping,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_RELEASE_HMAC_DOMAIN,
    BfclV4PublicV2SemanticDevelopmentRelease,
    BfclV4PublicV2SemanticMappedAssignment,
    BfclV4PublicV2SemanticMappedReview,
    BfclV4PublicV2SemanticReleaseError,
    BfclV4PublicV2SemanticReleaseFailureStage,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256


def _reject(stage: BfclV4PublicV2SemanticReleaseFailureStage) -> None:
    raise BfclV4PublicV2SemanticReleaseError(stage) from None


def _checked_secret(value: object) -> bytes:
    if type(value) is not bytes or len(value) < BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    return value


def _hmac_sha256(secret: bytes, payload: object) -> str:
    message = canonical_json_bytes(
        {
            "domain": BFCL_V4_PUBLIC_V2_SEMANTIC_RELEASE_HMAC_DOMAIN,
            "payload": payload,
        }
    )
    return hmac.new(secret, message, sha256).hexdigest()


def _source_coordinates() -> dict[str, tuple[str, str, int, str, str]]:
    """Return task -> (audit id, candidate id, row size, row hash, path)."""

    return {
        item.task_id: (
            item.audit_item_id,
            item.candidate_payload_sha256,
            item.row_size,
            item.row_sha256,
            item.question_git_path,
        )
        for item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items
    }


def _expected_private_coordinates() -> dict[
    str,
    tuple[BfclV4PublicV2SemanticPrivateBoundary, int],
]:
    coordinates = {
        task_id: (BfclV4PublicV2SemanticPrivateBoundary.V1, ordinal)
        for ordinal, task_id in enumerate(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS)
    }
    coordinates.update(
        {
            item.task_id: (
                BfclV4PublicV2SemanticPrivateBoundary(item.split.value),
                item.ordinal,
            )
            for item in BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster
        }
    )
    return coordinates


def _candidate_from_packet(question_json: str, schemas_json: str, task_id: str) -> str:
    question = json.loads(question_json)
    schemas = json.loads(schemas_json)
    return canonical_sha256({"id": task_id, "question": question, "function": schemas})


def _validate_mapping_population(
    packet: BfclV4PublicV2SemanticReviewerPacket,
    mapping: BfclV4PublicV2SemanticTrustedMapping,
) -> None:
    expected = _source_coordinates()
    expected_private = _expected_private_coordinates()
    if set(expected) != {item.public_task_id for item in mapping.entries}:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    for packet_item, entry in zip(packet.items, mapping.entries, strict=True):
        coordinate = expected.get(entry.public_task_id)
        if coordinate is None:
            _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
        _, candidate, row_size, row_sha256, path = coordinate
        expected_boundary, expected_ordinal = expected_private[entry.public_task_id]
        try:
            packet_candidate = _candidate_from_packet(
                packet_item.canonical_question_json,
                packet_item.canonical_function_schemas_json,
                entry.public_task_id,
            )
        except Exception:
            _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
        if (
            entry.candidate_payload_sha256 != candidate
            or entry.source_row_size != row_size
            or entry.source_row_sha256 != row_sha256
            or entry.question_git_path != path
            or packet_candidate != candidate
            or entry.private_boundary_label is not expected_boundary
            or entry.source_roster_ordinal != expected_ordinal
        ):
            _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)


def _strict_session(value: object) -> BfclV4PublicV2SemanticReviewSessionEvidence:
    try:
        return BfclV4PublicV2SemanticReviewSessionEvidence.model_validate(value, strict=True)
    except Exception:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)


def _map_session(
    packet: BfclV4PublicV2SemanticReviewerPacket,
    mapping: BfclV4PublicV2SemanticTrustedMapping,
    session: BfclV4PublicV2SemanticReviewSessionEvidence,
    *,
    runtime_hmac_secret: bytes,
) -> BfclV4PublicV2SemanticMappedReview:
    checked = _strict_session(session)
    declaration = checked.reviewer_declaration
    if declaration.declared_reviewer_kind is not BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.TRANSPORT_ATTESTATION)
    try:
        verify_bfcl_v4_public_v2_semantic_distribution_receipt(
            checked.distribution_receipt,
            packet,
            mapping,
            declaration,
            runtime_hmac_secret=runtime_hmac_secret,
        )
        response = validate_bfcl_v4_public_v2_semantic_reviewer_response(
            packet,
            declaration,
            checked.reviewer_response,
        )
    except Exception:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)

    attestation = checked.execution_attestation
    if (
        attestation.reviewer_packet_ref != packet.ref
        or attestation.trusted_mapping_ref != mapping.ref
        or attestation.requested_model != declaration.model_name
        or attestation.provider_reported_model != declaration.model_name
        or attestation.provider_seed_u64 != declaration.model_generation_seed_u64
    ):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.TRANSPORT_ATTESTATION)

    assert declaration.model_provider is not None
    assert declaration.model_name is not None
    assert declaration.model_revision is not None
    assert declaration.served_model_weights_identity_sha256 is not None
    assert declaration.model_prompt_sha256 is not None
    assert declaration.model_generation_seed_u64 is not None
    response_by_id = {item.opaque_item_id: item for item in response.assignments}
    mapping_by_task = {item.public_task_id: item for item in mapping.entries}
    expected = _source_coordinates()
    assignments = []
    for universe_item in BFCL_V4_SEMANTIC_AUDIT_UNIVERSE.items:
        entry = mapping_by_task[universe_item.task_id]
        response_item = response_by_id[entry.opaque_item_id]
        audit_id, candidate_id, _, _, _ = expected[entry.public_task_id]
        assignments.append(
            BfclV4PublicV2SemanticMappedAssignment(
                audit_item_id=audit_id,
                candidate_payload_sha256=candidate_id,
                public_task_id=entry.public_task_id,
                private_boundary_label=entry.private_boundary_label,
                semantic_family_label=response_item.semantic_family_label,
                rationale=response_item.rationale,
            )
        )
    return BfclV4PublicV2SemanticMappedReview(
        reviewer_packet_ref=packet.ref,
        trusted_mapping_ref=mapping.ref,
        reviewer_declaration_ref=declaration.ref,
        reviewer_response_ref=response.ref,
        distribution_receipt_ref=checked.distribution_receipt.ref,
        execution_attestation_ref=attestation.ref,
        declared_principal_id=declaration.declared_principal_id,
        declared_model_provider=declaration.model_provider,
        declared_model_name=declaration.model_name,
        declared_model_revision=declaration.model_revision,
        declared_served_weights_identity_sha256=(declaration.served_model_weights_identity_sha256),
        declared_prompt_sha256=declaration.model_prompt_sha256,
        declared_generation_seed_u64=declaration.model_generation_seed_u64,
        assignments=tuple(assignments),
    )


def _reviewer_coordinate(review: BfclV4PublicV2SemanticMappedReview) -> tuple[str, ...]:
    return (
        review.declared_principal_id,
        review.declared_model_provider,
        review.declared_model_name,
        review.declared_model_revision,
        review.declared_served_weights_identity_sha256,
    )


def _assert_independent(
    reviews: tuple[BfclV4PublicV2SemanticMappedReview, ...],
) -> None:
    if len({item.declared_principal_id for item in reviews}) != len(reviews):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.REVIEW_INDEPENDENCE)
    if len({item.declared_served_weights_identity_sha256 for item in reviews}) != len(reviews):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.REVIEW_INDEPENDENCE)
    if len({_reviewer_coordinate(item)[1:4] for item in reviews}) != len(reviews):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.REVIEW_INDEPENDENCE)
    if len({item.declared_prompt_sha256 for item in reviews}) != 1:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.REVIEW_INDEPENDENCE)


def _relations(review: BfclV4PublicV2SemanticMappedReview) -> dict[tuple[str, str], bool]:
    labels = {item.audit_item_id: item.semantic_family_label for item in review.assignments}
    item_ids = tuple(item.audit_item_id for item in review.assignments)
    return {
        (left, right): labels[left] == labels[right] for left, right in combinations(item_ids, 2)
    }


def _partition(review: BfclV4PublicV2SemanticMappedReview) -> tuple[tuple[str, ...], ...]:
    families: dict[str, list[str]] = defaultdict(list)
    for item in review.assignments:
        families[item.semantic_family_label].append(item.audit_item_id)
    return tuple(sorted(tuple(sorted(members)) for members in families.values()))


def _cross_boundary_family_count(review: BfclV4PublicV2SemanticMappedReview) -> int:
    families: dict[str, set[BfclV4PublicV2SemanticPrivateBoundary]] = defaultdict(set)
    for item in review.assignments:
        families[item.semantic_family_label].add(item.private_boundary_label)
    return sum(len(boundaries) > 1 for boundaries in families.values())


def _release_unsigned_payload(
    release: BfclV4PublicV2SemanticDevelopmentRelease,
) -> dict[str, object]:
    return release.model_dump(mode="python", exclude={"release_authority_hmac_sha256"})


def issue_bfcl_v4_public_v2_semantic_development_release(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    primary_sessions: tuple[
        BfclV4PublicV2SemanticReviewSessionEvidence,
        BfclV4PublicV2SemanticReviewSessionEvidence,
    ],
    *,
    runtime_hmac_secret: bytes,
    adjudicator_session: BfclV4PublicV2SemanticReviewSessionEvidence | None = None,
) -> BfclV4PublicV2SemanticDevelopmentRelease:
    """Verify, remap, arbitrate, and issue one public-development authority."""

    secret = _checked_secret(runtime_hmac_secret)
    try:
        packet = BfclV4PublicV2SemanticReviewerPacket.model_validate(
            reviewer_packet,
            strict=True,
        )
        mapping = BfclV4PublicV2SemanticTrustedMapping.model_validate(
            trusted_mapping,
            strict=True,
        )
    except Exception:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    if type(primary_sessions) is not tuple or len(primary_sessions) != 2:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    _validate_mapping_population(packet, mapping)
    primary = tuple(
        _map_session(packet, mapping, session, runtime_hmac_secret=secret)
        for session in primary_sessions
    )
    _assert_independent(primary)
    left_relations, right_relations = _relations(primary[0]), _relations(primary[1])
    disagreements = tuple(
        pair for pair in left_relations if left_relations[pair] != right_relations[pair]
    )

    adjudicator = None
    if disagreements:
        if adjudicator_session is None:
            _reject(BfclV4PublicV2SemanticReleaseFailureStage.DISAGREEMENT_ARBITRATION)
        adjudicator = _map_session(
            packet,
            mapping,
            adjudicator_session,
            runtime_hmac_secret=secret,
        )
        _assert_independent((*primary, adjudicator))
        final_review = adjudicator
    else:
        if adjudicator_session is not None:
            _reject(BfclV4PublicV2SemanticReleaseFailureStage.DISAGREEMENT_ARBITRATION)
        final_review = primary[0]

    if _cross_boundary_family_count(final_review):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.FAMILY_BOUNDARY)
    partition = _partition(final_review)
    draft = BfclV4PublicV2SemanticDevelopmentRelease(
        source_universe_fingerprint=mapping.source_universe_fingerprint,
        reviewer_packet_ref=packet.ref,
        trusted_mapping_ref=mapping.ref,
        primary_review_refs=(primary[0].ref, primary[1].ref),
        primary_execution_attestation_refs=(
            primary[0].execution_attestation_ref,
            primary[1].execution_attestation_ref,
        ),
        adjudicator_review_ref=None if adjudicator is None else adjudicator.ref,
        adjudicator_execution_attestation_ref=(
            None if adjudicator is None else adjudicator.execution_attestation_ref
        ),
        final_partition_sha256=canonical_sha256(
            {
                "domain": "spiral-bfcl-v4-public-v2-final-semantic-partition/v1",
                "families": partition,
            }
        ),
        final_semantic_family_count=len(partition),
        primary_pairwise_disagreement_count=len(disagreements),
        reviewer_count=3 if adjudicator is not None else 2,
        release_authority_hmac_sha256="0" * 64,
    )
    tag = _hmac_sha256(secret, _release_unsigned_payload(draft))
    return BfclV4PublicV2SemanticDevelopmentRelease(
        **{
            **draft.model_dump(mode="python"),
            "release_authority_hmac_sha256": tag,
        }
    )


def verify_bfcl_v4_public_v2_semantic_development_release(
    release: BfclV4PublicV2SemanticDevelopmentRelease,
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    primary_sessions: tuple[
        BfclV4PublicV2SemanticReviewSessionEvidence,
        BfclV4PublicV2SemanticReviewSessionEvidence,
    ],
    *,
    runtime_hmac_secret: bytes,
    adjudicator_session: BfclV4PublicV2SemanticReviewSessionEvidence | None = None,
) -> BfclV4PublicV2SemanticDevelopmentRelease:
    """Recompute the complete release and its HMAC before an execution boundary."""

    try:
        checked = BfclV4PublicV2SemanticDevelopmentRelease.model_validate(release, strict=True)
    except Exception:
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    expected = issue_bfcl_v4_public_v2_semantic_development_release(
        reviewer_packet,
        trusted_mapping,
        primary_sessions,
        runtime_hmac_secret=runtime_hmac_secret,
        adjudicator_session=adjudicator_session,
    )
    if checked != expected or not hmac.compare_digest(
        checked.release_authority_hmac_sha256,
        _hmac_sha256(_checked_secret(runtime_hmac_secret), _release_unsigned_payload(checked)),
    ):
        _reject(BfclV4PublicV2SemanticReleaseFailureStage.EVIDENCE_BINDING)
    return checked


__all__ = [
    name
    for name in globals()
    if name.startswith("Bfcl") or name.startswith("issue_bfcl") or name.startswith("verify_bfcl")
]
