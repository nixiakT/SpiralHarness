"""Issue and verify BFCL releases from honest 15-call composite reviews."""

from __future__ import annotations

import hmac
from hashlib import sha256

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4 import (
    replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_contracts import (
    BfclV4PublicV2PairwiseV4CompositeEvidence,
    BfclV4PublicV2PairwiseV4Pair,
    BfclV4PublicV2PairwiseV4PredecessorClosure,
    BfclV4PublicV2PairwiseV4ReviewerRole,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELEASE_HMAC_DOMAIN,
    BfclV4PublicV2PairwiseV4DevelopmentRelease,
    BfclV4PublicV2PairwiseV4ReleaseError,
    BfclV4PublicV2PairwiseV4ReleaseFailureStage,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    verify_bfcl_v4_public_v2_semantic_packet_mapping_integrity,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES,
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticTrustedMapping,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256


def _reject(stage: BfclV4PublicV2PairwiseV4ReleaseFailureStage) -> None:
    raise BfclV4PublicV2PairwiseV4ReleaseError(stage) from None


def _secret(value: object) -> bytes:
    if type(value) is not bytes or len(value) < BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    return value


def _tag(secret: bytes, payload: object) -> str:
    body = canonical_json_bytes(
        {
            "domain": BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELEASE_HMAC_DOMAIN,
            "payload": payload,
        }
    )
    return hmac.new(secret, body, sha256).hexdigest()


def _strict(
    model: type,
    value: object,
    stage: BfclV4PublicV2PairwiseV4ReleaseFailureStage,
):
    try:
        return model.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject(stage)


def _checked_composite(
    packet: BfclV4PublicV2SemanticReviewerPacket,
    closure: BfclV4PublicV2PairwiseV4PredecessorClosure,
    value: object,
) -> BfclV4PublicV2PairwiseV4CompositeEvidence:
    evidence = _strict(
        BfclV4PublicV2PairwiseV4CompositeEvidence,
        value,
        BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING,
    )
    try:
        replayed = replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
            packet,
            closure,
            evidence.lineage,
            evidence.plan,
            evidence.reviewer_declaration,
            evidence.call_sessions,
        )
    except Exception:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    if replayed != evidence:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    return evidence


def _reviewer_coordinate(
    evidence: BfclV4PublicV2PairwiseV4CompositeEvidence,
) -> tuple[str, ...]:
    declaration = evidence.reviewer_declaration
    return (
        declaration.declared_principal_id,
        declaration.model_provider,
        declaration.model_name,
        declaration.model_revision,
        declaration.served_model_weights_identity_sha256,
    )


def _assert_independent(
    evidence: tuple[BfclV4PublicV2PairwiseV4CompositeEvidence, ...],
) -> None:
    coordinates = tuple(_reviewer_coordinate(item) for item in evidence)
    call_sessions = tuple(session for item in evidence for session in item.call_sessions)
    if (
        len({item[0] for item in coordinates}) != len(coordinates)
        or len({item[4] for item in coordinates}) != len(coordinates)
        or len({item[1:4] for item in coordinates}) != len(coordinates)
        or len({item.reviewer_declaration.model_prompt_sha256 for item in evidence}) != 1
        or len({item.lineage.ref for item in evidence}) != len(evidence)
        or len({item.lineage.reviewer_nonce_sha256 for item in evidence}) != len(evidence)
        or len({item.lineage.seed_root_sha256 for item in evidence}) != len(evidence)
        or len(
            {session.attempt_evidence.provider_response_id for session in call_sessions}
        )
        != len(call_sessions)
    ):
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.REVIEW_INDEPENDENCE)


def _relations(
    evidence: BfclV4PublicV2PairwiseV4CompositeEvidence,
) -> dict[BfclV4PublicV2PairwiseV4Pair, bool]:
    return {edge.pair: edge.same_semantic_family for edge in evidence.relation.edges}


def _cross_boundary_count(
    evidence: BfclV4PublicV2PairwiseV4CompositeEvidence,
    mapping: BfclV4PublicV2SemanticTrustedMapping,
) -> int:
    by_opaque_id = {entry.opaque_item_id: entry for entry in mapping.entries}
    return sum(
        len({by_opaque_id[item].private_boundary_label for item in family}) > 1
        for family in evidence.relation.equivalence_classes
    )


def _trusted_partition(
    evidence: BfclV4PublicV2PairwiseV4CompositeEvidence,
    mapping: BfclV4PublicV2SemanticTrustedMapping,
) -> tuple[tuple[str, ...], ...]:
    by_opaque_id = {entry.opaque_item_id: entry for entry in mapping.entries}
    return tuple(
        sorted(
            tuple(sorted(by_opaque_id[item].candidate_payload_sha256 for item in family))
            for family in evidence.relation.equivalence_classes
        )
    )


def _unsigned(
    release: BfclV4PublicV2PairwiseV4DevelopmentRelease,
) -> dict[str, object]:
    return release.model_dump(mode="python", exclude={"release_authority_hmac_sha256"})


def issue_bfcl_v4_public_v2_pairwise_v4_development_release(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    predecessor_closure: BfclV4PublicV2PairwiseV4PredecessorClosure,
    primary_composite_evidence: tuple[
        BfclV4PublicV2PairwiseV4CompositeEvidence,
        BfclV4PublicV2PairwiseV4CompositeEvidence,
    ],
    *,
    runtime_hmac_secret: bytes,
    adjudicator_composite_evidence: BfclV4PublicV2PairwiseV4CompositeEvidence | None = None,
) -> BfclV4PublicV2PairwiseV4DevelopmentRelease:
    """Issue v4 authority only from complete independent composite relations."""

    secret = _secret(runtime_hmac_secret)
    packet = _strict(
        BfclV4PublicV2SemanticReviewerPacket,
        reviewer_packet,
        BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING,
    )
    mapping = _strict(
        BfclV4PublicV2SemanticTrustedMapping,
        trusted_mapping,
        BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING,
    )
    closure = _strict(
        BfclV4PublicV2PairwiseV4PredecessorClosure,
        predecessor_closure,
        BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING,
    )
    try:
        packet, mapping = verify_bfcl_v4_public_v2_semantic_packet_mapping_integrity(
            packet,
            mapping,
            runtime_hmac_secret=secret,
        )
    except Exception:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.MAPPING_INTEGRITY)
    if type(primary_composite_evidence) is not tuple or len(primary_composite_evidence) != 2:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    primary = tuple(
        _checked_composite(packet, closure, item) for item in primary_composite_evidence
    )
    expected_primary_roles = (
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_TWO,
    )
    if tuple(item.lineage.reviewer_role for item in primary) != expected_primary_roles:
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    _assert_independent(primary)
    left, right = _relations(primary[0]), _relations(primary[1])
    disagreements = tuple(pair for pair in left if left[pair] is not right[pair])

    adjudicator = None
    if disagreements:
        if adjudicator_composite_evidence is None:
            _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.DISAGREEMENT_ARBITRATION)
        adjudicator = _checked_composite(packet, closure, adjudicator_composite_evidence)
        if (
            adjudicator.lineage.reviewer_role
            is not BfclV4PublicV2PairwiseV4ReviewerRole.ADJUDICATOR
        ):
            _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
        _assert_independent((*primary, adjudicator))
        adjudicated_relations = _relations(adjudicator)
        if any(
            pair not in disagreements and adjudicated_relations[pair] is not left[pair]
            for pair in left
        ):
            _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.DISAGREEMENT_ARBITRATION)
        final = adjudicator
    else:
        if adjudicator_composite_evidence is not None:
            _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.DISAGREEMENT_ARBITRATION)
        final = primary[0]
    if _cross_boundary_count(final, mapping):
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.FAMILY_BOUNDARY)
    partition = _trusted_partition(final, mapping)
    reviewer_count = 3 if adjudicator is not None else 2
    draft = BfclV4PublicV2PairwiseV4DevelopmentRelease(
        source_universe_fingerprint=mapping.source_universe_fingerprint,
        reviewer_packet_ref=packet.ref,
        trusted_mapping_ref=mapping.ref,
        predecessor_closure_ref=closure.ref,
        primary_composite_evidence_refs=(primary[0].ref, primary[1].ref),
        primary_relation_refs=(primary[0].relation.ref, primary[1].relation.ref),
        adjudicator_composite_evidence_ref=None if adjudicator is None else adjudicator.ref,
        adjudicator_relation_ref=None if adjudicator is None else adjudicator.relation.ref,
        final_partition_sha256=canonical_sha256(
            {
                "domain": "spiral-bfcl-v4-public-v2-final-semantic-partition/v4",
                "families": partition,
            }
        ),
        final_semantic_family_count=len(partition),
        primary_pairwise_disagreement_count=len(disagreements),
        reviewer_count=reviewer_count,
        composite_call_count=reviewer_count * 15,
        reviewed_unordered_pair_decision_count=reviewer_count * 780,
        release_authority_hmac_sha256="0" * 64,
    )
    return BfclV4PublicV2PairwiseV4DevelopmentRelease(
        **{
            **draft.model_dump(mode="python"),
            "release_authority_hmac_sha256": _tag(secret, _unsigned(draft)),
        }
    )


def verify_bfcl_v4_public_v2_pairwise_v4_development_release(
    release: BfclV4PublicV2PairwiseV4DevelopmentRelease,
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    predecessor_closure: BfclV4PublicV2PairwiseV4PredecessorClosure,
    primary_composite_evidence: tuple[
        BfclV4PublicV2PairwiseV4CompositeEvidence,
        BfclV4PublicV2PairwiseV4CompositeEvidence,
    ],
    *,
    runtime_hmac_secret: bytes,
    adjudicator_composite_evidence: BfclV4PublicV2PairwiseV4CompositeEvidence | None = None,
) -> BfclV4PublicV2PairwiseV4DevelopmentRelease:
    """Recompute the composite release and HMAC before activation."""

    checked = _strict(
        BfclV4PublicV2PairwiseV4DevelopmentRelease,
        release,
        BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING,
    )
    expected = issue_bfcl_v4_public_v2_pairwise_v4_development_release(
        reviewer_packet,
        trusted_mapping,
        predecessor_closure,
        primary_composite_evidence,
        runtime_hmac_secret=runtime_hmac_secret,
        adjudicator_composite_evidence=adjudicator_composite_evidence,
    )
    secret = _secret(runtime_hmac_secret)
    if checked != expected or not hmac.compare_digest(
        checked.release_authority_hmac_sha256,
        _tag(secret, _unsigned(checked)),
    ):
        _reject(BfclV4PublicV2PairwiseV4ReleaseFailureStage.EVIDENCE_BINDING)
    return checked


__all__ = [
    name
    for name in globals()
    if name.startswith("Bfcl") or name.startswith("issue_bfcl") or name.startswith("verify_bfcl")
]
