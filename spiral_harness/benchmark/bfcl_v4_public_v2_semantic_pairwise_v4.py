"""Build, strictly parse, and replay chunked BFCL semantic reviews without I/O."""

from __future__ import annotations

import json
from base64 import b64decode, b64encode
from itertools import combinations
from typing import Any

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL,
    BfclV4PublicV2PairwiseV4AttemptEvidence,
    BfclV4PublicV2PairwiseV4Block,
    BfclV4PublicV2PairwiseV4CallKind,
    BfclV4PublicV2PairwiseV4CallRequest,
    BfclV4PublicV2PairwiseV4CallResponse,
    BfclV4PublicV2PairwiseV4CallSession,
    BfclV4PublicV2PairwiseV4CallSpec,
    BfclV4PublicV2PairwiseV4CompositeEvidence,
    BfclV4PublicV2PairwiseV4Decision,
    BfclV4PublicV2PairwiseV4Error,
    BfclV4PublicV2PairwiseV4Lineage,
    BfclV4PublicV2PairwiseV4Pair,
    BfclV4PublicV2PairwiseV4Plan,
    BfclV4PublicV2PairwiseV4PredecessorClosure,
    BfclV4PublicV2PairwiseV4Relation,
    BfclV4PublicV2PairwiseV4RelationEdge,
    BfclV4PublicV2PairwiseV4ReviewerDeclaration,
    BfclV4PublicV2PairwiseV4ReviewerRole,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticReviewerPacket,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256, sha256_bytes
from spiral_harness.core.models import NonEmptyStr, Sha256

BFCL_V4_PUBLIC_V2_PAIRWISE_V4_MAX_OUTPUT_TOKENS = 24_000
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_MAX_COMPLETION_BYTES = 4_194_304
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT = (
    "You are an independent semantic-equivalence reviewer for public function-calling "
    "items. You receive one fixed chunk from a secret-permuted 40-item packet. Each item "
    "contains only an opaque ID, a canonical user question, and canonical function schemas. "
    "For every ordered pair supplied, decide whether the two items require materially the "
    "same user intent, tool-selection logic, argument structure, multiplicity/order behavior, "
    "and composition pattern after abstracting incidental names, entities, constants, and "
    "wording. Broad domain or API overlap alone is not equivalence. Return one JSON object and "
    "no markdown or other prose. It must have exactly one key, 'decisions'. The value must be "
    "an array in the exact supplied pair order. Every row must have exactly four keys: "
    "'left_opaque_item_id', 'right_opaque_item_id', 'same_semantic_family', and 'rationale'. "
    "Copy both opaque IDs exactly; same_semantic_family must be a JSON boolean and rationale "
    "must briefly state the decisive semantic and tool-structure evidence."
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT_SHA256 = sha256_bytes(
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT.encode("utf-8")
)


def _reject(message: str) -> None:
    raise BfclV4PublicV2PairwiseV4Error(message) from None


def _strict(model: type[Any], value: object, label: str) -> Any:
    try:
        return model.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject(f"invalid {label}")


def _seed(seed_root_sha256: str, call_id: str) -> int:
    digest = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-pairwise-v4-provider-seed/v1",
            "seed_root_sha256": seed_root_sha256,
            "call_id": call_id,
        }
    )
    return int(digest[:16], 16)


def build_bfcl_v4_public_v2_pairwise_v4_lineage(
    predecessor_closure: BfclV4PublicV2PairwiseV4PredecessorClosure,
    *,
    reviewer_role: BfclV4PublicV2PairwiseV4ReviewerRole,
    reviewer_nonce_sha256: Sha256,
    seed_root_sha256: Sha256,
) -> BfclV4PublicV2PairwiseV4Lineage:
    """Open a new v4 lineage without reusing or retrying a v1 attempt."""

    closure = _strict(
        BfclV4PublicV2PairwiseV4PredecessorClosure,
        predecessor_closure,
        "predecessor closure",
    )
    try:
        role = BfclV4PublicV2PairwiseV4ReviewerRole(reviewer_role)
        payload = {
            "domain": "spiral-bfcl-v4-public-v2-pairwise-v4-lineage/v1",
            "reviewer_role": role,
            "predecessor_closure_ref": closure.ref,
            "reviewer_nonce_sha256": reviewer_nonce_sha256,
            "seed_root_sha256": seed_root_sha256,
        }
        return BfclV4PublicV2PairwiseV4Lineage(
            reviewer_role=role,
            predecessor_closure_ref=closure.ref,
            reviewer_nonce_sha256=reviewer_nonce_sha256,
            seed_root_sha256=seed_root_sha256,
            lineage_sha256=canonical_sha256(payload),
        )
    except (TypeError, ValidationError, ValueError):
        _reject("invalid fresh pairwise lineage coordinates")


def _call_coordinates() -> tuple[tuple[int, ...], ...]:
    return tuple((index,) for index in range(5)) + tuple(combinations(range(5), 2))


def _call_id(coordinate: tuple[int, ...]) -> str:
    if len(coordinate) == 1:
        return f"pairwise-v4-within-{coordinate[0]}"
    return f"pairwise-v4-cross-{coordinate[0]}-{coordinate[1]}"


def _call_pairs(
    blocks: tuple[BfclV4PublicV2PairwiseV4Block, ...],
    coordinate: tuple[int, ...],
) -> tuple[BfclV4PublicV2PairwiseV4Pair, ...]:
    if len(coordinate) == 1:
        pairs = combinations(blocks[coordinate[0]].opaque_item_ids, 2)
    else:
        pairs = (
            (left, right)
            for left in blocks[coordinate[0]].opaque_item_ids
            for right in blocks[coordinate[1]].opaque_item_ids
        )
    return tuple(
        BfclV4PublicV2PairwiseV4Pair(
            left_opaque_item_id=left,
            right_opaque_item_id=right,
        )
        for left, right in pairs
    )


def build_bfcl_v4_public_v2_pairwise_v4_plan(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    lineage: BfclV4PublicV2PairwiseV4Lineage,
) -> BfclV4PublicV2PairwiseV4Plan:
    """Derive the only admissible 5x8 and 15-call schedule from packet order."""

    packet = _strict(BfclV4PublicV2SemanticReviewerPacket, reviewer_packet, "reviewer packet")
    checked_lineage = _strict(BfclV4PublicV2PairwiseV4Lineage, lineage, "pairwise lineage")
    blocks = tuple(
        BfclV4PublicV2PairwiseV4Block(
            block_index=index,
            opaque_item_ids=tuple(
                item.opaque_item_id for item in packet.items[index * 8 : (index + 1) * 8]
            ),
        )
        for index in range(5)
    )
    calls = []
    flattened_pairs = []
    for call_index, coordinate in enumerate(_call_coordinates()):
        call_id = _call_id(coordinate)
        pairs = _call_pairs(blocks, coordinate)
        flattened_pairs.extend(pairs)
        calls.append(
            BfclV4PublicV2PairwiseV4CallSpec(
                call_index=call_index,
                call_id=call_id,
                call_kind=(
                    BfclV4PublicV2PairwiseV4CallKind.WITHIN
                    if len(coordinate) == 1
                    else BfclV4PublicV2PairwiseV4CallKind.CROSS
                ),
                block_indices=coordinate,
                opaque_item_ids=tuple(
                    item for index in coordinate for item in blocks[index].opaque_item_ids
                ),
                ordered_pairs=pairs,
                provider_seed_u64=_seed(checked_lineage.seed_root_sha256, call_id),
            )
        )
    return BfclV4PublicV2PairwiseV4Plan(
        reviewer_packet_ref=packet.ref,
        lineage_ref=checked_lineage.ref,
        seed_root_sha256=checked_lineage.seed_root_sha256,
        blocks=blocks,
        calls=tuple(calls),
        pair_manifest_sha256=canonical_sha256(tuple(flattened_pairs)),
    )


def build_bfcl_v4_public_v2_pairwise_v4_reviewer_declaration(
    plan: BfclV4PublicV2PairwiseV4Plan,
    lineage: BfclV4PublicV2PairwiseV4Lineage,
    *,
    declared_principal_id: NonEmptyStr,
    model_provider: NonEmptyStr,
    model_name: NonEmptyStr,
    model_revision: NonEmptyStr,
    served_model_weights_identity_sha256: Sha256,
) -> BfclV4PublicV2PairwiseV4ReviewerDeclaration:
    """Bind one declared reviewer identity across all 15 planned calls."""

    checked_plan = _strict(BfclV4PublicV2PairwiseV4Plan, plan, "pairwise plan")
    checked_lineage = _strict(BfclV4PublicV2PairwiseV4Lineage, lineage, "pairwise lineage")
    if checked_plan.lineage_ref != checked_lineage.ref:
        _reject("pairwise declaration plan and lineage differ")
    try:
        return BfclV4PublicV2PairwiseV4ReviewerDeclaration(
            plan_ref=checked_plan.ref,
            lineage_ref=checked_lineage.ref,
            reviewer_role=checked_lineage.reviewer_role,
            declared_principal_id=declared_principal_id,
            model_provider=model_provider,
            model_name=model_name,
            model_revision=model_revision,
            served_model_weights_identity_sha256=served_model_weights_identity_sha256,
            model_prompt_sha256=BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT_SHA256,
        )
    except (TypeError, ValidationError, ValueError):
        _reject("invalid pairwise reviewer declaration")


def build_bfcl_v4_public_v2_pairwise_v4_call_request(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    plan: BfclV4PublicV2PairwiseV4Plan,
    declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration,
    *,
    call_index: int,
) -> BfclV4PublicV2PairwiseV4CallRequest:
    """Build one trusted wrapper whose model-visible projection is minimal."""

    packet = _strict(BfclV4PublicV2SemanticReviewerPacket, reviewer_packet, "reviewer packet")
    checked_plan = _strict(BfclV4PublicV2PairwiseV4Plan, plan, "pairwise plan")
    checked_declaration = _strict(
        BfclV4PublicV2PairwiseV4ReviewerDeclaration,
        declaration,
        "reviewer declaration",
    )
    if (
        type(call_index) is not int
        or not 0 <= call_index < 15
        or checked_plan.reviewer_packet_ref != packet.ref
        or checked_declaration.plan_ref != checked_plan.ref
    ):
        _reject("pairwise request roots or call index are invalid")
    call = checked_plan.calls[call_index]
    by_id = {item.opaque_item_id: item for item in packet.items}
    return BfclV4PublicV2PairwiseV4CallRequest(
        plan_ref=checked_plan.ref,
        reviewer_packet_ref=packet.ref,
        reviewer_declaration_ref=checked_declaration.ref,
        call=call,
        items=tuple(by_id[item_id] for item_id in call.opaque_item_ids),
    )


def bfcl_v4_public_v2_pairwise_v4_model_visible_payload(
    request: BfclV4PublicV2PairwiseV4CallRequest,
) -> dict[str, Any]:
    """Return only opaque IDs, questions, schemas, and exact opaque pair order."""

    checked = _strict(BfclV4PublicV2PairwiseV4CallRequest, request, "pairwise request")
    return {
        "protocol": BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL,
        "call_id": checked.call.call_id,
        "items": [
            {
                "opaque_item_id": item.opaque_item_id,
                "canonical_question_json": item.canonical_question_json,
                "canonical_function_schemas_json": item.canonical_function_schemas_json,
            }
            for item in checked.items
        ],
        "ordered_pairs": [pair.model_dump(mode="json") for pair in checked.call.ordered_pairs],
    }


def build_bfcl_v4_public_v2_pairwise_v4_provider_request(
    request: BfclV4PublicV2PairwiseV4CallRequest,
    declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration,
) -> dict[str, Any]:
    """Build a credential-free OpenAI-compatible body; never send it."""

    checked_request = _strict(
        BfclV4PublicV2PairwiseV4CallRequest,
        request,
        "pairwise request",
    )
    checked_declaration = _strict(
        BfclV4PublicV2PairwiseV4ReviewerDeclaration,
        declaration,
        "reviewer declaration",
    )
    if checked_request.reviewer_declaration_ref != checked_declaration.ref:
        _reject("pairwise request cites another reviewer declaration")
    return {
        "model": checked_declaration.model_name,
        "messages": [
            {"role": "system", "content": BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT},
            {
                "role": "user",
                "content": canonical_json(
                    bfcl_v4_public_v2_pairwise_v4_model_visible_payload(checked_request)
                ),
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": BFCL_V4_PUBLIC_V2_PAIRWISE_V4_MAX_OUTPUT_TOKENS,
        "seed": checked_request.call.provider_seed_u64,
        "response_format": {"type": "json_object"},
    }


def _strict_json_bytes(raw: bytes) -> object:
    if type(raw) is not bytes or len(raw) > BFCL_V4_PUBLIC_V2_PAIRWISE_V4_MAX_COMPLETION_BYTES:
        _reject("pairwise completion is not bounded bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        del value
        raise ValueError("non-finite number")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _reject("pairwise completion is not strict JSON")


def _parse_decisions(
    raw_completion: bytes,
    call: BfclV4PublicV2PairwiseV4CallSpec,
) -> tuple[BfclV4PublicV2PairwiseV4Decision, ...]:
    decoded = _strict_json_bytes(raw_completion)
    if type(decoded) is not dict or set(decoded) != {"decisions"}:
        _reject("pairwise completion has the wrong top-level schema")
    rows = decoded["decisions"]
    if type(rows) is not list or len(rows) != len(call.ordered_pairs):
        _reject("pairwise completion lacks exact scheduled pair coverage")
    decisions = []
    try:
        for row in rows:
            if type(row) is not dict or set(row) != {
                "left_opaque_item_id",
                "right_opaque_item_id",
                "same_semantic_family",
                "rationale",
            }:
                raise ValueError("row schema")
            decisions.append(BfclV4PublicV2PairwiseV4Decision(**row))
    except (TypeError, ValidationError, ValueError):
        _reject("pairwise completion contains an invalid decision")
    if tuple(item.pair for item in decisions) != call.ordered_pairs:
        _reject("pairwise completion changed ordered pairs")
    return tuple(decisions)


def parse_bfcl_v4_public_v2_pairwise_v4_call_session(
    request: BfclV4PublicV2PairwiseV4CallRequest,
    declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration,
    raw_completion: bytes,
    *,
    provider_origin_sha256: Sha256,
    provider_response_id: NonEmptyStr,
    provider_reported_model: NonEmptyStr,
    input_tokens: int,
    output_tokens: int,
    operator_attests_single_provider_attempt: bool,
) -> BfclV4PublicV2PairwiseV4CallSession:
    """Strictly parse one consumed attempt; malformed content is never repaired."""

    checked_request = _strict(
        BfclV4PublicV2PairwiseV4CallRequest,
        request,
        "pairwise request",
    )
    checked_declaration = _strict(
        BfclV4PublicV2PairwiseV4ReviewerDeclaration,
        declaration,
        "reviewer declaration",
    )
    if checked_request.reviewer_declaration_ref != checked_declaration.ref:
        _reject("pairwise parser received another reviewer declaration")
    if operator_attests_single_provider_attempt is not True:
        _reject("pairwise parser lacks an explicit single-attempt operator attestation")
    decisions = _parse_decisions(raw_completion, checked_request.call)
    response = BfclV4PublicV2PairwiseV4CallResponse(
        request_ref=checked_request.ref,
        plan_ref=checked_request.plan_ref,
        reviewer_declaration_ref=checked_declaration.ref,
        call_id=checked_request.call.call_id,
        decisions=decisions,
    )
    provider_request = build_bfcl_v4_public_v2_pairwise_v4_provider_request(
        checked_request,
        checked_declaration,
    )
    try:
        attempt = BfclV4PublicV2PairwiseV4AttemptEvidence(
            request_ref=checked_request.ref,
            response_ref=response.ref,
            provider_origin_sha256=provider_origin_sha256,
            model_visible_request_sha256=canonical_sha256(provider_request),
            raw_completion_content_sha256=sha256_bytes(raw_completion),
            provider_response_id=provider_response_id,
            requested_model=checked_declaration.model_name,
            provider_reported_model=provider_reported_model,
            provider_seed_u64=checked_request.call.provider_seed_u64,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return BfclV4PublicV2PairwiseV4CallSession(
            request=checked_request,
            response=response,
            attempt_evidence=attempt,
            raw_completion_content_base64=b64encode(raw_completion).decode("ascii"),
        )
    except (TypeError, ValidationError, ValueError):
        _reject("pairwise attempt metadata is invalid")


def _validate_replay_session(
    expected_request: BfclV4PublicV2PairwiseV4CallRequest,
    declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration,
    value: object,
) -> BfclV4PublicV2PairwiseV4CallSession:
    session = _strict(BfclV4PublicV2PairwiseV4CallSession, value, "pairwise call session")
    if session.request != expected_request:
        _reject("pairwise replay request differs from its fixed plan")
    reparsed = _parse_decisions(
        b64decode(session.raw_completion_content_base64, validate=True),
        expected_request.call,
    )
    if (
        session.response.decisions != reparsed
        or tuple(item.pair for item in reparsed) != expected_request.call.ordered_pairs
    ):
        _reject("pairwise replay response lacks exact ordered pairs")
    expected_request_sha256 = canonical_sha256(
        build_bfcl_v4_public_v2_pairwise_v4_provider_request(expected_request, declaration)
    )
    if (
        session.attempt_evidence.model_visible_request_sha256 != expected_request_sha256
        or session.attempt_evidence.requested_model != declaration.model_name
        or session.attempt_evidence.provider_reported_model != declaration.model_name
    ):
        _reject("pairwise replay attempt differs from its request or reviewer")
    return session


def replay_bfcl_v4_public_v2_pairwise_v4_composite_evidence(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    predecessor_closure: BfclV4PublicV2PairwiseV4PredecessorClosure,
    lineage: BfclV4PublicV2PairwiseV4Lineage,
    plan: BfclV4PublicV2PairwiseV4Plan,
    declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration,
    call_sessions: tuple[BfclV4PublicV2PairwiseV4CallSession, ...],
) -> BfclV4PublicV2PairwiseV4CompositeEvidence:
    """Rebuild all 780 decisions and reject any non-equivalence relation."""

    packet = _strict(BfclV4PublicV2SemanticReviewerPacket, reviewer_packet, "reviewer packet")
    closure = _strict(
        BfclV4PublicV2PairwiseV4PredecessorClosure,
        predecessor_closure,
        "predecessor closure",
    )
    checked_lineage = _strict(BfclV4PublicV2PairwiseV4Lineage, lineage, "pairwise lineage")
    checked_plan = _strict(BfclV4PublicV2PairwiseV4Plan, plan, "pairwise plan")
    checked_declaration = _strict(
        BfclV4PublicV2PairwiseV4ReviewerDeclaration,
        declaration,
        "reviewer declaration",
    )
    expected_plan = build_bfcl_v4_public_v2_pairwise_v4_plan(packet, checked_lineage)
    if (
        checked_lineage.predecessor_closure_ref != closure.ref
        or checked_plan != expected_plan
        or checked_declaration.plan_ref != checked_plan.ref
        or checked_declaration.lineage_ref != checked_lineage.ref
        or checked_declaration.reviewer_role is not checked_lineage.reviewer_role
        or checked_declaration.model_prompt_sha256 != BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROMPT_SHA256
        or type(call_sessions) is not tuple
        or len(call_sessions) != 15
    ):
        _reject("pairwise replay roots or exact call population are invalid")
    sessions = tuple(
        _validate_replay_session(
            build_bfcl_v4_public_v2_pairwise_v4_call_request(
                packet,
                checked_plan,
                checked_declaration,
                call_index=index,
            ),
            checked_declaration,
            session,
        )
        for index, session in enumerate(call_sessions)
    )
    if len({item.attempt_evidence.provider_response_id for item in sessions}) != 15:
        _reject("pairwise replay repeats a provider response identity")

    by_pair: dict[
        BfclV4PublicV2PairwiseV4Pair,
        tuple[bool, str, Any],
    ] = {}
    for session in sessions:
        for decision in session.response.decisions:
            if decision.pair in by_pair:
                _reject("pairwise replay repeats an unordered pair")
            by_pair[decision.pair] = (
                decision.same_semantic_family,
                session.request.call.call_id,
                session.response.ref,
            )
    item_ids = tuple(item.opaque_item_id for item in packet.items)
    canonical_pairs = tuple(
        BfclV4PublicV2PairwiseV4Pair(
            left_opaque_item_id=left,
            right_opaque_item_id=right,
        )
        for left, right in combinations(item_ids, 2)
    )
    if len(by_pair) != 780 or set(by_pair) != set(canonical_pairs):
        _reject("pairwise replay does not cover all 780 unordered pairs exactly once")

    parent = {item: item for item in item_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair, (same, _, _) in by_pair.items():
        if same:
            union(pair.left_opaque_item_id, pair.right_opaque_item_id)
    for pair, (same, _, _) in by_pair.items():
        implied = find(pair.left_opaque_item_id) == find(pair.right_opaque_item_id)
        if same is not implied:
            _reject("pairwise replay decisions are not an equivalence relation")
    groups: dict[str, list[str]] = {}
    for item in item_ids:
        groups.setdefault(find(item), []).append(item)
    positions = {item: index for index, item in enumerate(item_ids)}
    classes = tuple(
        sorted(
            (tuple(members) for members in groups.values()),
            key=lambda members: positions[members[0]],
        )
    )
    response_refs = tuple(item.response.ref for item in sessions)
    try:
        relation = BfclV4PublicV2PairwiseV4Relation(
            reviewer_packet_ref=packet.ref,
            reviewer_declaration_ref=checked_declaration.ref,
            ordered_opaque_item_ids=item_ids,
            source_response_refs=response_refs,
            edges=tuple(
                BfclV4PublicV2PairwiseV4RelationEdge(
                    pair=pair,
                    same_semantic_family=by_pair[pair][0],
                    source_call_id=by_pair[pair][1],
                    source_response_ref=by_pair[pair][2],
                )
                for pair in canonical_pairs
            ),
            equivalence_classes=classes,
        )
        return BfclV4PublicV2PairwiseV4CompositeEvidence(
            predecessor_closure_ref=closure.ref,
            lineage=checked_lineage,
            reviewer_declaration=checked_declaration,
            plan=checked_plan,
            call_sessions=sessions,
            relation=relation,
        )
    except (TypeError, ValidationError, ValueError):
        _reject("pairwise replay could not bind composite evidence")


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("bfcl_v4")
    or name.startswith("build_bfcl")
    or name.startswith("parse_bfcl")
    or name.startswith("replay_bfcl")
]
