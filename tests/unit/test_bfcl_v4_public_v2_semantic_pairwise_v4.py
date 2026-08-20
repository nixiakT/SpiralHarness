from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection as projection_subject
from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer import (
    materialize_bfcl_v4_public_v2_request,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2FrozenArmTreatment,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority import (
    verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4 import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT_SHA256,
    bfcl_v4_public_v2_pairwise_v4_model_visible_payload,
    build_bfcl_v4_public_v2_pairwise_v4_call_request,
    build_bfcl_v4_public_v2_pairwise_v4_lineage,
    build_bfcl_v4_public_v2_pairwise_v4_plan,
    build_bfcl_v4_public_v2_pairwise_v4_provider_request,
    build_bfcl_v4_public_v2_pairwise_v4_reviewer_declaration,
    parse_bfcl_v4_public_v2_pairwise_v4_call_session,
    replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE,
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_FAILURE_MEDIA_TYPE,
    BfclV4PublicV2PairwiseV4Error,
    BfclV4PublicV2PairwiseV4PredecessorClosure,
    BfclV4PublicV2PairwiseV4PredecessorLineage,
    BfclV4PublicV2PairwiseV4ReviewerRole,
    BfclV4PublicV2PairwiseV4TerminatedAttempt,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release import (
    issue_bfcl_v4_public_v2_pairwise_v4_development_release,
    verify_bfcl_v4_public_v2_pairwise_v4_development_release,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE,
    BfclV4PublicV2PairwiseV4ReleaseError,
    BfclV4PublicV2PairwiseV4ReleaseFailureStage,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2SemanticReleaseEvidenceShape,
    freeze_bfcl_v4_public_v2_live_execution_config,
)

SECRET = b"pairwise-v4-provider-free-unit-secret" * 2


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifact(label: str) -> ArtifactRef:
    return ArtifactRef(
        sha256=_digest(label),
        size=len(label),
        media_type=BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_FAILURE_MEDIA_TYPE,
    )


@pytest.fixture(scope="module")
def projection():
    v1_fingerprint = _digest("synthetic-v1-loaded-bundle")
    v2_fingerprint = _digest("synthetic-v2-loaded-bundle")
    source_universe = projection_subject._source_universe_fingerprint(
        v1_fingerprint,
        v2_fingerprint,
    )
    tasks = []
    for index in range(40):
        if index < 15:
            boundary = projection_subject.BfclV4PublicV2SemanticPrivateBoundary.V1
            ordinal = index
        else:
            ordinal = index - 15
            if ordinal < 5:
                boundary = projection_subject.BfclV4PublicV2SemanticPrivateBoundary.FIT
            elif ordinal < 9:
                boundary = projection_subject.BfclV4PublicV2SemanticPrivateBoundary.GATE
            else:
                boundary = projection_subject.BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT
        tasks.append(
            projection_subject._BfclV4PublicV2ProjectionSourceTask(
                source_roster_ordinal=ordinal,
                public_task_id=f"private-task-SENTINEL-{index}",
                private_boundary_label=boundary,
                question_git_path=f"private/path-SENTINEL-{index}.json",
                question_blob_size=1_000 + index,
                question_blob_sha256=_digest(f"question-blob-{index}"),
                source_row_size=2_000 + index,
                source_row_sha256=_digest(f"source-row-{index}"),
                candidate_payload_sha256=_digest(f"candidate-{index}"),
                canonical_question_json=canonical_json(
                    {"question": f"synthetic visible question {index}"}
                ),
                canonical_function_schemas_json=canonical_json(
                    [{"name": f"synthetic_visible_tool_{index}"}]
                ),
            )
        )
    source = projection_subject._BfclV4PublicV2ProjectionSourceBundle(
        v1_loaded_bundle_fingerprint=v1_fingerprint,
        v2_loaded_bundle_fingerprint=v2_fingerprint,
        source_universe_fingerprint=source_universe,
        tasks=tuple(tasks),
    )
    return projection_subject._materialize_projection(source, SECRET)


@pytest.fixture(scope="module")
def closure() -> BfclV4PublicV2PairwiseV4PredecessorClosure:
    return BfclV4PublicV2PairwiseV4PredecessorClosure(
        attempts=tuple(
            BfclV4PublicV2PairwiseV4TerminatedAttempt(
                predecessor_lineage=lineage,
                attempt_ref=_artifact(f"terminated-{lineage.value}"),
            )
            for lineage in BfclV4PublicV2PairwiseV4PredecessorLineage
        )
    )


def _reviewer(projection, closure, role, index: int):
    lineage = build_bfcl_v4_public_v2_pairwise_v4_lineage(
        closure,
        reviewer_role=role,
        reviewer_nonce_sha256=_digest(f"reviewer-nonce-{index}"),
        seed_root_sha256=_digest(f"reviewer-seed-root-{index}"),
    )
    plan = build_bfcl_v4_public_v2_pairwise_v4_plan(
        projection.reviewer_packet,
        lineage,
    )
    declaration = build_bfcl_v4_public_v2_pairwise_v4_reviewer_declaration(
        plan,
        lineage,
        declared_principal_id=f"pairwise-reviewer-{index}",
        model_provider=f"declared-provider-{index}",
        model_name=f"declared-review-model-{index}",
        model_revision=f"declared-review-revision-{index}",
        served_model_weights_identity_sha256=_digest(f"weights-{index}"),
    )
    return lineage, plan, declaration


def _raw_decisions(request, predicate: Callable[[str, str], bool]) -> bytes:
    return canonical_json(
        {
            "decisions": [
                {
                    "left_opaque_item_id": pair.left_opaque_item_id,
                    "right_opaque_item_id": pair.right_opaque_item_id,
                    "same_semantic_family": predicate(
                        pair.left_opaque_item_id,
                        pair.right_opaque_item_id,
                    ),
                    "rationale": "Compared intent, tool choice, arguments, and composition.",
                }
                for pair in request.call.ordered_pairs
            ]
        }
    ).encode()


def _sessions(projection, plan, declaration, predicate):
    sessions = []
    for index in range(15):
        request = build_bfcl_v4_public_v2_pairwise_v4_call_request(
            projection.reviewer_packet,
            plan,
            declaration,
            call_index=index,
        )
        sessions.append(
            parse_bfcl_v4_public_v2_pairwise_v4_call_session(
                request,
                declaration,
                _raw_decisions(request, predicate),
                provider_origin_sha256=_digest("provider-origin"),
                provider_response_id=(
                    f"provider-response-{declaration.reviewer_role.value}-{index}-"
                    f"{declaration.declared_principal_id}"
                ),
                provider_reported_model=declaration.model_name,
                input_tokens=1_000 + index,
                output_tokens=2_000 + index,
                operator_attests_single_provider_attempt=True,
            )
        )
    return tuple(sessions)


def _family_predicate(families: tuple[tuple[str, ...], ...]):
    groups = tuple(frozenset(family) for family in families)
    return lambda left, right: any(left in group and right in group for group in groups)


def _composite(projection, closure, role, index: int, families=()):
    lineage, plan, declaration = _reviewer(projection, closure, role, index)
    sessions = _sessions(projection, plan, declaration, _family_predicate(families))
    return replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
        projection.reviewer_packet,
        closure,
        lineage,
        plan,
        declaration,
        sessions,
    )


def test_fixed_five_by_eight_schedule_covers_every_pair_once(projection, closure) -> None:
    lineage, plan, _ = _reviewer(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        1,
    )

    assert plan.lineage_ref == lineage.ref
    assert tuple(len(block.opaque_item_ids) for block in plan.blocks) == (8,) * 5
    assert tuple(len(call.ordered_pairs) for call in plan.calls) == (28,) * 5 + (64,) * 10
    pairs = tuple(pair for call in plan.calls for pair in call.ordered_pairs)
    assert len(pairs) == len(set(pairs)) == 780
    assert plan.pair_manifest_sha256 == canonical_sha256(pairs)


def test_request_projection_contains_only_opaque_public_content(projection, closure) -> None:
    _, plan, declaration = _reviewer(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        2,
    )
    request = build_bfcl_v4_public_v2_pairwise_v4_call_request(
        projection.reviewer_packet,
        plan,
        declaration,
        call_index=5,
    )
    visible = bfcl_v4_public_v2_pairwise_v4_model_visible_payload(request)
    body = build_bfcl_v4_public_v2_pairwise_v4_provider_request(request, declaration)
    rendered = canonical_json(body)

    assert set(visible) == {"protocol", "call_id", "items", "ordered_pairs"}
    assert set(visible["items"][0]) == {
        "opaque_item_id",
        "canonical_question_json",
        "canonical_function_schemas_json",
    }
    assert len(visible["items"]) == 16
    assert len(visible["ordered_pairs"]) == 64
    assert "private-task-SENTINEL" not in rendered
    assert "private/path-SENTINEL" not in rendered
    assert "candidate_payload" not in rendered
    assert "possible_answer" not in rendered
    assert "api_key" not in rendered


def test_strict_parser_binds_one_attempt_and_exact_pair_order(projection, closure) -> None:
    _, plan, declaration = _reviewer(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        3,
    )
    request = build_bfcl_v4_public_v2_pairwise_v4_call_request(
        projection.reviewer_packet,
        plan,
        declaration,
        call_index=0,
    )
    session = parse_bfcl_v4_public_v2_pairwise_v4_call_session(
        request,
        declaration,
        _raw_decisions(request, lambda _left, _right: False),
        provider_origin_sha256=_digest("origin"),
        provider_response_id="response-3-0",
        provider_reported_model=declaration.model_name,
        input_tokens=100,
        output_tokens=200,
        operator_attests_single_provider_attempt=True,
    )

    assert len(session.response.decisions) == 28
    assert session.attempt_evidence.provider_attempts == 1
    assert session.attempt_evidence.automatic_retry_used is False
    assert session.attempt_evidence.repair_or_backfill_used is False
    assert session.attempt_evidence.provider_transport_performed_by_this_module is False

    with pytest.raises(BfclV4PublicV2PairwiseV4Error, match="single-attempt"):
        parse_bfcl_v4_public_v2_pairwise_v4_call_session(
            request,
            declaration,
            _raw_decisions(request, lambda _left, _right: False),
            provider_origin_sha256=_digest("origin"),
            provider_response_id="response-without-single-attempt-attestation",
            provider_reported_model=declaration.model_name,
            input_tokens=100,
            output_tokens=200,
            operator_attests_single_provider_attempt=False,
        )


@pytest.mark.parametrize(
    "tamper",
    ("top-level", "row-extra", "pair-order", "wrong-bool", "duplicate-key", "markdown"),
)
def test_strict_parser_rejects_malformed_completion_without_repair(
    projection,
    closure,
    tamper: str,
) -> None:
    _, plan, declaration = _reviewer(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        4,
    )
    request = build_bfcl_v4_public_v2_pairwise_v4_call_request(
        projection.reviewer_packet,
        plan,
        declaration,
        call_index=0,
    )
    decoded = {
        "decisions": [
            {
                "left_opaque_item_id": pair.left_opaque_item_id,
                "right_opaque_item_id": pair.right_opaque_item_id,
                "same_semantic_family": False,
                "rationale": "Exact pair review.",
            }
            for pair in request.call.ordered_pairs
        ]
    }
    if tamper == "top-level":
        decoded["extra"] = True
    elif tamper == "row-extra":
        decoded["decisions"][0]["extra"] = "forbidden"
    elif tamper == "pair-order":
        decoded["decisions"][0], decoded["decisions"][1] = (
            decoded["decisions"][1],
            decoded["decisions"][0],
        )
    elif tamper == "wrong-bool":
        decoded["decisions"][0]["same_semantic_family"] = "false"
    if tamper == "duplicate-key":
        raw = b'{"decisions":[],"decisions":[]}'
    elif tamper == "markdown":
        raw = b"```json\n{}\n```"
    else:
        raw = canonical_json(decoded).encode()

    with pytest.raises(BfclV4PublicV2PairwiseV4Error):
        parse_bfcl_v4_public_v2_pairwise_v4_call_session(
            request,
            declaration,
            raw,
            provider_origin_sha256=_digest("origin"),
            provider_response_id="malformed-response",
            provider_reported_model=declaration.model_name,
            input_tokens=1,
            output_tokens=1,
            operator_attests_single_provider_attempt=True,
        )


def test_replay_produces_honest_composite_equivalence_evidence(projection, closure) -> None:
    evidence = _composite(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        5,
    )

    assert len(evidence.call_sessions) == 15
    assert len(evidence.relation.edges) == 780
    assert len(evidence.relation.equivalence_classes) == 40
    assert evidence.single_call_review is False
    assert evidence.single_call_attestation_present is False
    assert evidence.composite_reviewer_evidence is True
    assert evidence.ref.media_type == BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE


def test_replay_rejects_non_transitive_relation_and_duplicate_response(projection, closure) -> None:
    lineage, plan, declaration = _reviewer(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
        6,
    )
    first, second, third = (item.opaque_item_id for item in projection.reviewer_packet.items[:3])
    true_pairs = {
        frozenset((first, second)),
        frozenset((second, third)),
    }
    sessions = _sessions(
        projection,
        plan,
        declaration,
        lambda left, right: frozenset((left, right)) in true_pairs,
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4Error, match="not an equivalence"):
        replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
            projection.reviewer_packet,
            closure,
            lineage,
            plan,
            declaration,
            sessions,
        )

    duplicate = sessions[1].model_copy(
        update={
            "attempt_evidence": sessions[1].attempt_evidence.model_copy(
                update={"provider_response_id": sessions[0].attempt_evidence.provider_response_id}
            )
        }
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4Error):
        replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
            projection.reviewer_packet,
            closure,
            lineage,
            plan,
            declaration,
            (sessions[0], duplicate, *sessions[2:]),
        )


@pytest.fixture(scope="module")
def consensus_evidence(projection, closure):
    return (
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
            10,
        ),
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_TWO,
            11,
        ),
    )


@pytest.fixture(scope="module")
def consensus_release(projection, closure, consensus_evidence):
    return issue_bfcl_v4_public_v2_pairwise_v4_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        closure,
        consensus_evidence,
        runtime_hmac_secret=SECRET,
    )


@pytest.fixture(scope="module")
def consensus_authority(projection, closure, consensus_evidence, consensus_release):
    return verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority(
        consensus_release,
        projection.reviewer_packet,
        projection.trusted_mapping,
        closure,
        consensus_evidence,
        runtime_hmac_secret=SECRET,
    )


def test_composite_release_is_hmac_bound_and_not_a_single_attestation(
    projection,
    closure,
    consensus_evidence,
    consensus_release,
    consensus_authority,
) -> None:
    checked = verify_bfcl_v4_public_v2_pairwise_v4_development_release(
        consensus_release,
        projection.reviewer_packet,
        projection.trusted_mapping,
        closure,
        consensus_evidence,
        runtime_hmac_secret=SECRET,
    )

    assert checked == consensus_release
    assert checked.reviewer_count == 2
    assert checked.composite_call_count == 30
    assert checked.reviewed_unordered_pair_decision_count == 1_560
    assert checked.single_call_review_attestation_present is False
    assert checked.ref.media_type == BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE
    assert consensus_authority.release_ref == checked.ref
    assert consensus_authority.release_fingerprint == checked.fingerprint


def test_pairwise_authority_replays_evidence_and_rejects_forgery(
    projection,
    closure,
    consensus_evidence,
    consensus_release,
) -> None:
    forged_release = consensus_release.model_copy(
        update={"release_authority_hmac_sha256": "0" * 64}
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError):
        verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority(
            forged_release,
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            consensus_evidence,
            runtime_hmac_secret=SECRET,
        )

    left, right = consensus_evidence
    sessions = list(left.call_sessions)
    sessions[1] = sessions[1].model_copy(
        update={
            "attempt_evidence": sessions[1].attempt_evidence.model_copy(
                update={"provider_response_id": (sessions[0].attempt_evidence.provider_response_id)}
            )
        }
    )
    replay_forged = left.model_copy(update={"call_sessions": tuple(sessions)})
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError):
        verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority(
            consensus_release,
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            (replay_forged, right),
            runtime_hmac_secret=SECRET,
        )


def test_release_rejects_provider_response_reuse_across_reviewers(
    projection,
    closure,
    consensus_evidence,
) -> None:
    left, right = consensus_evidence
    sessions = list(right.call_sessions)
    sessions[0] = sessions[0].model_copy(
        update={
            "attempt_evidence": sessions[0].attempt_evidence.model_copy(
                update={
                    "provider_response_id": (
                        left.call_sessions[0].attempt_evidence.provider_response_id
                    )
                }
            )
        }
    )
    replayed_right = replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
        projection.reviewer_packet,
        closure,
        right.lineage,
        right.plan,
        right.reviewer_declaration,
        tuple(sessions),
    )

    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError) as captured:
        issue_bfcl_v4_public_v2_pairwise_v4_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            (left, replayed_right),
            runtime_hmac_secret=SECRET,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2PairwiseV4ReleaseFailureStage.REVIEW_INDEPENDENCE
    )


def test_disagreement_requires_independent_composite_adjudicator(projection, closure) -> None:
    holdout_items = tuple(
        entry.opaque_item_id
        for entry in projection.trusted_mapping.entries
        if entry.private_boundary_label
        is projection_subject.BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT
    )
    same_boundary = holdout_items[:2]
    primary = (
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
            20,
        ),
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_TWO,
            21,
            (same_boundary,),
        ),
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError) as captured:
        issue_bfcl_v4_public_v2_pairwise_v4_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            primary,
            runtime_hmac_secret=SECRET,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2PairwiseV4ReleaseFailureStage.DISAGREEMENT_ARBITRATION
    )

    adjudicator = _composite(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.ADJUDICATOR,
        22,
    )
    release = issue_bfcl_v4_public_v2_pairwise_v4_development_release(
        projection.reviewer_packet,
        projection.trusted_mapping,
        closure,
        primary,
        runtime_hmac_secret=SECRET,
        adjudicator_composite_evidence=adjudicator,
    )
    assert release.reviewer_count == 3
    assert release.composite_call_count == 45
    assert release.primary_pairwise_disagreement_count == 1

    consensus_overriding_adjudicator = _composite(
        projection,
        closure,
        BfclV4PublicV2PairwiseV4ReviewerRole.ADJUDICATOR,
        23,
        (holdout_items[2:4],),
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError) as captured:
        issue_bfcl_v4_public_v2_pairwise_v4_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            primary,
            runtime_hmac_secret=SECRET,
            adjudicator_composite_evidence=consensus_overriding_adjudicator,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2PairwiseV4ReleaseFailureStage.DISAGREEMENT_ARBITRATION
    )


def test_cross_boundary_family_and_mapping_tamper_fail_closed(projection, closure) -> None:
    left = next(
        entry
        for entry in projection.trusted_mapping.entries
        if entry.private_boundary_label
        is projection_subject.BfclV4PublicV2SemanticPrivateBoundary.FIT
    )
    right = next(
        entry
        for entry in projection.trusted_mapping.entries
        if entry.private_boundary_label
        is projection_subject.BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT
    )
    merged = ((left.opaque_item_id, right.opaque_item_id),)
    primary = (
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_ONE,
            30,
            merged,
        ),
        _composite(
            projection,
            closure,
            BfclV4PublicV2PairwiseV4ReviewerRole.PRIMARY_TWO,
            31,
            merged,
        ),
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError) as captured:
        issue_bfcl_v4_public_v2_pairwise_v4_development_release(
            projection.reviewer_packet,
            projection.trusted_mapping,
            closure,
            primary,
            runtime_hmac_secret=SECRET,
        )
    assert (
        captured.value.failure_stage is BfclV4PublicV2PairwiseV4ReleaseFailureStage.FAMILY_BOUNDARY
    )

    tampered_mapping = projection.trusted_mapping.model_copy(
        update={"mapping_commitment_hmac_sha256": _digest("forged-mapping")}
    )
    with pytest.raises(BfclV4PublicV2PairwiseV4ReleaseError) as captured:
        issue_bfcl_v4_public_v2_pairwise_v4_development_release(
            projection.reviewer_packet,
            tampered_mapping,
            closure,
            primary,
            runtime_hmac_secret=SECRET,
        )
    assert (
        captured.value.failure_stage
        is BfclV4PublicV2PairwiseV4ReleaseFailureStage.MAPPING_INTEGRITY
    )


def test_live_config_consumes_typed_composite_release(
    consensus_authority,
) -> None:
    config = freeze_bfcl_v4_public_v2_live_execution_config(
        catalog_observation=observe_bfcl_v4_public_live_model_catalog(
            (BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,),
            observed_at_utc="2026-08-15T12:00:00Z",
        ),
        campaign=build_bfcl_v4_public_development_v2_campaign_plan(),
        semantic_authority=consensus_authority,
        backend_name="openai-compatible-native-function",
        backend_fingerprint=_digest("backend"),
        serializer_fingerprint=_digest("serializer"),
        parser_fingerprint=_digest("parser"),
        transport_fingerprint=_digest("transport"),
    )

    assert (
        config.semantic_release_evidence_shape
        is BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4
    )
    assert config.semantic_release_ref == consensus_authority.release_ref
    assert config.semantic_release_ref.media_type == (
        BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE
    )


def test_solver_request_materializer_consumes_composite_authority(
    consensus_authority,
) -> None:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else Path("/tmp/spiral-bfcl-upstream")
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for composite materializer test")
    loaded = load_bfcl_v4_public_development_v2(checkout)
    campaign = build_bfcl_v4_public_development_v2_campaign_plan()
    node = next(
        item
        for item in campaign.nodes
        if item.arm is BfclV4PublicDevelopmentV2Arm.PURE
        and item.kind is BfclV4PublicDevelopmentV2NodeKind.HOLDOUT
        and item.harness_variant == "bare"
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    materialized = materialize_bfcl_v4_public_v2_request(
        loaded=loaded,
        campaign=campaign,
        lineage=lineage,
        semantic_authority=consensus_authority,
        treatment=BfclV4PublicV2FrozenArmTreatment(
            arm=node.arm,
            harness_variant=node.harness_variant,
        ),
    )

    assert materialized.semantic_release_ref == consensus_authority.release_ref
    assert materialized.semantic_release_fingerprint == consensus_authority.release_fingerprint
    assert (
        materialized.semantic_release_evidence_shape
        is BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4
    )
    assert materialized.semantic_release_authority_verified is True


def test_predecessor_closure_binds_v1_v2_v3_failures_in_order(closure) -> None:
    assert tuple(item.review_role for item in closure.attempts) == ("primary-one",) * 3
    assert tuple(item.predecessor_lineage for item in closure.attempts) == tuple(
        BfclV4PublicV2PairwiseV4PredecessorLineage
    )
    with pytest.raises(ValidationError, match="v1-v3 in fixed order"):
        BfclV4PublicV2PairwiseV4PredecessorClosure(
            attempts=(closure.attempts[1], closure.attempts[0], closure.attempts[2])
        )
    wrong_type = closure.attempts[0].model_copy(
        update={
            "attempt_ref": closure.attempts[0].attempt_ref.model_copy(
                update={"media_type": "application/x-wrong-predecessor-type"}
            )
        }
    )
    with pytest.raises(ValidationError, match="wrong failure artifact type"):
        BfclV4PublicV2PairwiseV4PredecessorClosure(attempts=(wrong_type, *closure.attempts[1:]))

    assert len(BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT_SHA256) == 64
