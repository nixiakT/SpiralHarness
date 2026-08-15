"""Provider-free construction and verification of the 15+25 semantic packet."""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4LoadedPublicDevelopmentV2,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4LoadedPublicPilot,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BFCL_V4_PUBLIC_V2_MAPPING_HMAC_DOMAIN,
    BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES,
    BFCL_V4_PUBLIC_V2_OPAQUE_ID_HMAC_DOMAIN,
    BFCL_V4_PUBLIC_V2_ORDER_HMAC_DOMAIN,
    BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL,
    BfclV4PublicV2SemanticDistributionReceipt,
    BfclV4PublicV2SemanticPrivateBoundary,
    BfclV4PublicV2SemanticProjection,
    BfclV4PublicV2SemanticProjectionError,
    BfclV4PublicV2SemanticReviewerAssignment,
    BfclV4PublicV2SemanticReviewerDeclaration,
    BfclV4PublicV2SemanticReviewerKind,
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticReviewerPacketItem,
    BfclV4PublicV2SemanticReviewerResponse,
    BfclV4PublicV2SemanticScoreExecutionError,
    BfclV4PublicV2SemanticTrustedMapping,
    BfclV4PublicV2SemanticTrustedMappingEntry,
    OpaqueItemId,
    PacketPosition,
    _reviewer_content_sha256,
    _source_universe_fingerprint,
    _validate_canonical_json,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256


def _checked_secret(secret: object) -> bytes:
    if type(secret) is not bytes or len(secret) < BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES:
        raise BfclV4PublicV2SemanticProjectionError(
            "semantic projection requires a process-local bytes secret of at least 32 bytes"
        )
    return secret


def _hmac_sha256(secret: bytes, domain: str, payload: object) -> str:
    message = canonical_json_bytes({"domain": domain, "payload": payload})
    return hmac.new(secret, message, sha256).hexdigest()


def _opaque_item_id(secret: bytes, universe_fingerprint: str, public_task_id: str) -> str:
    tag = _hmac_sha256(
        secret,
        BFCL_V4_PUBLIC_V2_OPAQUE_ID_HMAC_DOMAIN,
        {
            "source_universe_fingerprint": universe_fingerprint,
            "public_task_id": public_task_id,
        },
    )
    return f"bfclv2-review-{tag}"


def _order_hmac(secret: bytes, universe_fingerprint: str, public_task_id: str) -> str:
    return _hmac_sha256(
        secret,
        BFCL_V4_PUBLIC_V2_ORDER_HMAC_DOMAIN,
        {
            "source_universe_fingerprint": universe_fingerprint,
            "public_task_id": public_task_id,
        },
    )


class _BfclV4PublicV2ProjectionSourceTask(ImmutableModel):
    source_roster_ordinal: PacketPosition
    public_task_id: NonEmptyStr
    private_boundary_label: BfclV4PublicV2SemanticPrivateBoundary
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    source_row_size: Annotated[int, Field(gt=0, strict=True)]
    source_row_sha256: Sha256
    candidate_payload_sha256: Sha256
    canonical_question_json: NonEmptyStr
    canonical_function_schemas_json: NonEmptyStr

    @model_validator(mode="after")
    def _canonical_public_content(self) -> Self:
        _validate_canonical_json(self.canonical_question_json, label="projection-source question")
        _validate_canonical_json(
            self.canonical_function_schemas_json,
            label="projection-source function schemas",
            require_list=True,
        )
        return self


class _BfclV4PublicV2ProjectionSourceBundle(ImmutableModel):
    v1_loaded_bundle_fingerprint: Sha256
    v2_loaded_bundle_fingerprint: Sha256
    source_universe_fingerprint: Sha256
    tasks: Annotated[
        tuple[_BfclV4PublicV2ProjectionSourceTask, ...],
        Field(min_length=40, max_length=40),
    ]
    both_loaded_bundle_contracts_strictly_revalidated: Literal[True] = True
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    runs_or_cas_read: Literal[False] = False
    model_invoked: Literal[False] = False
    network_used: Literal[False] = False

    @model_validator(mode="after")
    def _close_source_population(self) -> Self:
        if len({item.public_task_id for item in self.tasks}) != 40:
            raise ValueError("projection source repeats a public task ID")
        if len({item.candidate_payload_sha256 for item in self.tasks}) != 40:
            raise ValueError("projection source repeats a candidate payload identity")
        if self.source_universe_fingerprint != _source_universe_fingerprint(
            self.v1_loaded_bundle_fingerprint,
            self.v2_loaded_bundle_fingerprint,
        ):
            raise ValueError("projection source universe does not bind both loaded bundles")
        v1_tasks = tuple(
            item
            for item in self.tasks
            if item.private_boundary_label is BfclV4PublicV2SemanticPrivateBoundary.V1
        )
        v2_tasks = tuple(
            item
            for item in self.tasks
            if item.private_boundary_label is not BfclV4PublicV2SemanticPrivateBoundary.V1
        )
        if (
            len(v1_tasks) != 15
            or tuple(item.source_roster_ordinal for item in v1_tasks) != tuple(range(15))
            or len(v2_tasks) != 25
            or tuple(item.source_roster_ordinal for item in v2_tasks) != tuple(range(25))
        ):
            raise ValueError("projection source lacks exact ordered 15+25 roster coverage")
        counts = {
            boundary: sum(item.private_boundary_label is boundary for item in self.tasks)
            for boundary in BfclV4PublicV2SemanticPrivateBoundary
        }
        if counts != {
            BfclV4PublicV2SemanticPrivateBoundary.V1: 15,
            BfclV4PublicV2SemanticPrivateBoundary.FIT: 5,
            BfclV4PublicV2SemanticPrivateBoundary.GATE: 4,
            BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT: 16,
        }:
            raise ValueError("projection source has wrong private boundary population")
        return self


def _mapping_unsigned_content(
    *,
    v1_loaded_bundle_fingerprint: str,
    v2_loaded_bundle_fingerprint: str,
    source_universe_fingerprint: str,
    reviewer_packet_ref: ArtifactRef,
    entries: tuple[BfclV4PublicV2SemanticTrustedMappingEntry, ...],
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL,
        "v1_loaded_bundle_fingerprint": v1_loaded_bundle_fingerprint,
        "v2_loaded_bundle_fingerprint": v2_loaded_bundle_fingerprint,
        "source_universe_fingerprint": source_universe_fingerprint,
        "reviewer_packet_ref": reviewer_packet_ref,
        "entries": entries,
        "trusted_control_plane_only": True,
        "reviewer_facing": False,
        "public_task_ids_and_partitions_present": True,
        "source_coordinates_present": True,
        "hmac_secret_present": False,
        "exact_public_content_may_be_reidentified": True,
        "partition_inference_cryptographically_prevented": False,
        "possible_answers_present": False,
        "scores_present": False,
        "model_or_harness_results_present": False,
        "score_bearing_execution_allowed": False,
    }


def _mapping_commitment(
    secret: bytes,
    *,
    v1_loaded_bundle_fingerprint: str,
    v2_loaded_bundle_fingerprint: str,
    source_universe_fingerprint: str,
    reviewer_packet_ref: ArtifactRef,
    entries: tuple[BfclV4PublicV2SemanticTrustedMappingEntry, ...],
) -> str:
    return _hmac_sha256(
        secret,
        BFCL_V4_PUBLIC_V2_MAPPING_HMAC_DOMAIN,
        _mapping_unsigned_content(
            v1_loaded_bundle_fingerprint=v1_loaded_bundle_fingerprint,
            v2_loaded_bundle_fingerprint=v2_loaded_bundle_fingerprint,
            source_universe_fingerprint=source_universe_fingerprint,
            reviewer_packet_ref=reviewer_packet_ref,
            entries=entries,
        ),
    )


def _strict_loaded_bundle(value: object, expected_type: object, *, label: str) -> object:
    if type(value) is not expected_type:
        raise BfclV4PublicV2SemanticProjectionError(
            f"semantic projection requires the exact loaded BFCL {label} bundle type"
        )
    try:
        return expected_type.model_validate(value, strict=True)  # type: ignore[attr-defined]
    except (TypeError, ValueError) as error:
        raise BfclV4PublicV2SemanticProjectionError(
            f"the loaded BFCL {label} bundle failed strict revalidation"
        ) from error


def _projection_source_from_loaded_bundles(
    v1_loaded_public_pilot: object,
    v2_loaded_public_development: object,
) -> _BfclV4PublicV2ProjectionSourceBundle:
    v1_checked = _strict_loaded_bundle(
        v1_loaded_public_pilot,
        BfclV4LoadedPublicPilot,
        label="v1 public-pilot",
    )
    v2_checked = _strict_loaded_bundle(
        v2_loaded_public_development,
        BfclV4LoadedPublicDevelopmentV2,
        label="v2 public-development",
    )
    try:
        v1_fingerprint = canonical_sha256(v1_checked)
        v2_fingerprint = canonical_sha256(v2_checked)
        v1_tasks = tuple(
            _BfclV4PublicV2ProjectionSourceTask(
                source_roster_ordinal=roster.ordinal,
                public_task_id=roster.task_id,
                private_boundary_label=BfclV4PublicV2SemanticPrivateBoundary.V1,
                question_git_path=roster.question_git_path,
                question_blob_size=roster.question_blob_size,
                question_blob_sha256=roster.question_blob_sha256,
                source_row_size=roster.row_size,
                source_row_sha256=roster.row_sha256,
                candidate_payload_sha256=roster.candidate_payload_sha256,
                canonical_question_json=task.question_json,
                canonical_function_schemas_json=task.function_schemas_json,
            )
            for roster, task in zip(
                v1_checked.manifest.roster,  # type: ignore[attr-defined]
                v1_checked.tasks,  # type: ignore[attr-defined]
                strict=True,
            )
        )
        v2_tasks = tuple(
            _BfclV4PublicV2ProjectionSourceTask(
                source_roster_ordinal=roster.ordinal,
                public_task_id=roster.task_id,
                private_boundary_label=BfclV4PublicV2SemanticPrivateBoundary(roster.split.value),
                question_git_path=roster.question_git_path,
                question_blob_size=roster.question_blob_size,
                question_blob_sha256=roster.question_blob_sha256,
                source_row_size=roster.row_size,
                source_row_sha256=roster.row_sha256,
                candidate_payload_sha256=roster.candidate_payload_sha256,
                canonical_question_json=task.question_json,
                canonical_function_schemas_json=task.function_schemas_json,
            )
            for roster, task in zip(
                v2_checked.manifest.roster,  # type: ignore[attr-defined]
                v2_checked.tasks,  # type: ignore[attr-defined]
                strict=True,
            )
        )
        return _BfclV4PublicV2ProjectionSourceBundle(
            v1_loaded_bundle_fingerprint=v1_fingerprint,
            v2_loaded_bundle_fingerprint=v2_fingerprint,
            source_universe_fingerprint=_source_universe_fingerprint(
                v1_fingerprint,
                v2_fingerprint,
            ),
            tasks=v1_tasks + v2_tasks,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise BfclV4PublicV2SemanticProjectionError(
            "the loaded BFCL 15+25 universe failed strict projection adaptation"
        ) from error


def _materialize_projection(
    source: _BfclV4PublicV2ProjectionSourceBundle,
    secret: bytes,
) -> BfclV4PublicV2SemanticProjection:
    checked_source = _BfclV4PublicV2ProjectionSourceBundle.model_validate(source, strict=True)
    checked_secret = _checked_secret(secret)
    decorated = tuple(
        (
            _order_hmac(
                checked_secret,
                checked_source.source_universe_fingerprint,
                task.public_task_id,
            ),
            _opaque_item_id(
                checked_secret,
                checked_source.source_universe_fingerprint,
                task.public_task_id,
            ),
            task,
        )
        for task in checked_source.tasks
    )
    if len({opaque_id for _, opaque_id, _ in decorated}) != 40:
        raise BfclV4PublicV2SemanticProjectionError("opaque reviewer item ID collision")
    ordered = tuple(sorted(decorated, key=lambda item: (item[0], item[1])))
    packet = BfclV4PublicV2SemanticReviewerPacket(
        items=tuple(
            BfclV4PublicV2SemanticReviewerPacketItem(
                opaque_item_id=opaque_id,
                canonical_question_json=task.canonical_question_json,
                canonical_function_schemas_json=task.canonical_function_schemas_json,
            )
            for _, opaque_id, task in ordered
        )
    )
    entries = tuple(
        BfclV4PublicV2SemanticTrustedMappingEntry(
            packet_position=position,
            opaque_item_id=opaque_id,
            order_hmac_sha256=order_token,
            public_task_id=task.public_task_id,
            source_roster_ordinal=task.source_roster_ordinal,
            private_boundary_label=task.private_boundary_label,
            question_git_path=task.question_git_path,
            question_blob_size=task.question_blob_size,
            question_blob_sha256=task.question_blob_sha256,
            source_row_size=task.source_row_size,
            source_row_sha256=task.source_row_sha256,
            candidate_payload_sha256=task.candidate_payload_sha256,
            reviewer_content_sha256=_reviewer_content_sha256(packet.items[position]),
        )
        for position, (order_token, opaque_id, task) in enumerate(ordered)
    )
    mapping = BfclV4PublicV2SemanticTrustedMapping(
        v1_loaded_bundle_fingerprint=checked_source.v1_loaded_bundle_fingerprint,
        v2_loaded_bundle_fingerprint=checked_source.v2_loaded_bundle_fingerprint,
        source_universe_fingerprint=checked_source.source_universe_fingerprint,
        reviewer_packet_ref=packet.ref,
        entries=entries,
        mapping_commitment_hmac_sha256=_mapping_commitment(
            checked_secret,
            v1_loaded_bundle_fingerprint=checked_source.v1_loaded_bundle_fingerprint,
            v2_loaded_bundle_fingerprint=checked_source.v2_loaded_bundle_fingerprint,
            source_universe_fingerprint=checked_source.source_universe_fingerprint,
            reviewer_packet_ref=packet.ref,
            entries=entries,
        ),
    )
    projection = BfclV4PublicV2SemanticProjection(
        reviewer_packet=packet,
        trusted_mapping=mapping,
    )
    _verify_packet_mapping(packet, mapping, checked_secret)
    return projection


def build_bfcl_v4_public_v2_semantic_reviewer_projection(
    v1_loaded_public_pilot: object,
    v2_loaded_public_development: object,
    *,
    runtime_hmac_secret: bytes,
) -> BfclV4PublicV2SemanticProjection:
    """Project the strict 15-item v1 plus 25-item v2 public semantic universe."""

    source = _projection_source_from_loaded_bundles(
        v1_loaded_public_pilot,
        v2_loaded_public_development,
    )
    return _materialize_projection(source, runtime_hmac_secret)


def _verify_packet_mapping(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    runtime_hmac_secret: bytes,
) -> tuple[
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticTrustedMapping,
]:
    packet = BfclV4PublicV2SemanticReviewerPacket.model_validate(
        reviewer_packet,
        strict=True,
    )
    mapping = BfclV4PublicV2SemanticTrustedMapping.model_validate(
        trusted_mapping,
        strict=True,
    )
    secret = _checked_secret(runtime_hmac_secret)
    if mapping.reviewer_packet_ref != packet.ref:
        raise BfclV4PublicV2SemanticProjectionError(
            "reviewer packet and trusted mapping have different populations"
        )
    for packet_item, mapping_entry in zip(packet.items, mapping.entries, strict=True):
        expected_opaque_id = _opaque_item_id(
            secret,
            mapping.source_universe_fingerprint,
            mapping_entry.public_task_id,
        )
        expected_order = _order_hmac(
            secret,
            mapping.source_universe_fingerprint,
            mapping_entry.public_task_id,
        )
        if (
            not hmac.compare_digest(mapping_entry.opaque_item_id, expected_opaque_id)
            or not hmac.compare_digest(mapping_entry.order_hmac_sha256, expected_order)
            or packet_item.opaque_item_id != mapping_entry.opaque_item_id
            or _reviewer_content_sha256(packet_item) != mapping_entry.reviewer_content_sha256
        ):
            raise BfclV4PublicV2SemanticProjectionError(
                "reviewer packet differs from its HMAC-derived trusted mapping"
            )
    expected_ordered_ids = tuple(
        item.opaque_item_id
        for item in sorted(
            mapping.entries,
            key=lambda item: (item.order_hmac_sha256, item.opaque_item_id),
        )
    )
    if tuple(item.opaque_item_id for item in packet.items) != expected_ordered_ids:
        raise BfclV4PublicV2SemanticProjectionError(
            "reviewer packet order differs from its HMAC-derived permutation"
        )
    expected_commitment = _mapping_commitment(
        secret,
        v1_loaded_bundle_fingerprint=mapping.v1_loaded_bundle_fingerprint,
        v2_loaded_bundle_fingerprint=mapping.v2_loaded_bundle_fingerprint,
        source_universe_fingerprint=mapping.source_universe_fingerprint,
        reviewer_packet_ref=mapping.reviewer_packet_ref,
        entries=mapping.entries,
    )
    if not hmac.compare_digest(mapping.mapping_commitment_hmac_sha256, expected_commitment):
        raise BfclV4PublicV2SemanticProjectionError("trusted mapping HMAC commitment is invalid")
    return packet, mapping


def verify_bfcl_v4_public_v2_semantic_distribution_integrity(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    reviewer_declaration: BfclV4PublicV2SemanticReviewerDeclaration,
    *,
    runtime_hmac_secret: bytes,
) -> BfclV4PublicV2SemanticDistributionReceipt:
    packet, mapping = _verify_packet_mapping(
        reviewer_packet,
        trusted_mapping,
        runtime_hmac_secret,
    )
    declaration = BfclV4PublicV2SemanticReviewerDeclaration.model_validate(
        reviewer_declaration,
        strict=True,
    )
    return BfclV4PublicV2SemanticDistributionReceipt(
        reviewer_packet_ref=packet.ref,
        trusted_mapping_ref=mapping.ref,
        reviewer_declaration_ref=declaration.ref,
        mapping_commitment_hmac_sha256=mapping.mapping_commitment_hmac_sha256,
    )


def verify_bfcl_v4_public_v2_semantic_distribution_receipt(
    receipt: BfclV4PublicV2SemanticDistributionReceipt,
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    reviewer_declaration: BfclV4PublicV2SemanticReviewerDeclaration,
    *,
    runtime_hmac_secret: bytes,
) -> BfclV4PublicV2SemanticDistributionReceipt:
    checked = BfclV4PublicV2SemanticDistributionReceipt.model_validate(receipt, strict=True)
    expected = verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        reviewer_packet,
        trusted_mapping,
        reviewer_declaration,
        runtime_hmac_secret=runtime_hmac_secret,
    )
    if checked != expected:
        raise BfclV4PublicV2SemanticProjectionError(
            "distribution receipt differs from live packet/mapping verification"
        )
    return checked


def validate_bfcl_v4_public_v2_semantic_reviewer_response(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    reviewer_declaration: BfclV4PublicV2SemanticReviewerDeclaration,
    response: BfclV4PublicV2SemanticReviewerResponse,
) -> BfclV4PublicV2SemanticReviewerResponse:
    packet = BfclV4PublicV2SemanticReviewerPacket.model_validate(
        reviewer_packet,
        strict=True,
    )
    declaration = BfclV4PublicV2SemanticReviewerDeclaration.model_validate(
        reviewer_declaration,
        strict=True,
    )
    checked = BfclV4PublicV2SemanticReviewerResponse.model_validate(response, strict=True)
    if (
        checked.reviewer_packet_ref != packet.ref
        or checked.reviewer_declaration_ref != declaration.ref
        or tuple(item.opaque_item_id for item in checked.assignments)
        != tuple(item.opaque_item_id for item in packet.items)
    ):
        raise BfclV4PublicV2SemanticProjectionError(
            "reviewer response lacks exact ordered opaque-ID packet coverage"
        )
    return checked


def assert_bfcl_v4_public_v2_semantic_projection_score_execution_allowed(
    distribution_receipt: BfclV4PublicV2SemanticDistributionReceipt,
) -> None:
    BfclV4PublicV2SemanticDistributionReceipt.model_validate(
        distribution_receipt,
        strict=True,
    )
    raise BfclV4PublicV2SemanticScoreExecutionError(
        "reviewer projection and distribution integrity cannot authorize BFCL v2 scoring"
    )


__all__ = [
    "BFCL_V4_PUBLIC_V2_MAPPING_HMAC_DOMAIN",
    "BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES",
    "BFCL_V4_PUBLIC_V2_OPAQUE_ID_HMAC_DOMAIN",
    "BFCL_V4_PUBLIC_V2_ORDER_HMAC_DOMAIN",
    "BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL",
    "BfclV4PublicV2SemanticDistributionReceipt",
    "BfclV4PublicV2SemanticPrivateBoundary",
    "BfclV4PublicV2SemanticProjection",
    "BfclV4PublicV2SemanticProjectionError",
    "BfclV4PublicV2SemanticReviewerAssignment",
    "BfclV4PublicV2SemanticReviewerDeclaration",
    "BfclV4PublicV2SemanticReviewerKind",
    "BfclV4PublicV2SemanticReviewerPacket",
    "BfclV4PublicV2SemanticReviewerPacketItem",
    "BfclV4PublicV2SemanticReviewerResponse",
    "BfclV4PublicV2SemanticScoreExecutionError",
    "BfclV4PublicV2SemanticTrustedMapping",
    "BfclV4PublicV2SemanticTrustedMappingEntry",
    "OpaqueItemId",
    "assert_bfcl_v4_public_v2_semantic_projection_score_execution_allowed",
    "build_bfcl_v4_public_v2_semantic_reviewer_projection",
    "validate_bfcl_v4_public_v2_semantic_reviewer_response",
    "verify_bfcl_v4_public_v2_semantic_distribution_integrity",
    "verify_bfcl_v4_public_v2_semantic_distribution_receipt",
]
