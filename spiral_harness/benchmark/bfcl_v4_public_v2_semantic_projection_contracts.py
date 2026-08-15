"""Typed contracts for the provider-free BFCL semantic reviewer projection."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_json, canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL = (
    "spiral-bfcl-v4-public-v2-semantic-reviewer-projection/v1"
)
BFCL_V4_PUBLIC_V2_REVIEWER_PACKET_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-reviewer-packet.v1+json"
)
BFCL_V4_PUBLIC_V2_TRUSTED_MAPPING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-trusted-mapping.v1+json"
)
BFCL_V4_PUBLIC_V2_REVIEWER_DECLARATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-reviewer-declaration.v1+json"
)
BFCL_V4_PUBLIC_V2_REVIEWER_RESPONSE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-reviewer-response.v1+json"
)
BFCL_V4_PUBLIC_V2_DISTRIBUTION_RECEIPT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-distribution-receipt.v1+json"
)
BFCL_V4_PUBLIC_V2_OPAQUE_ID_HMAC_DOMAIN = "spiral-bfcl-v4-public-v2-semantic-reviewer-opaque-id/v1"
BFCL_V4_PUBLIC_V2_ORDER_HMAC_DOMAIN = "spiral-bfcl-v4-public-v2-semantic-reviewer-order/v1"
BFCL_V4_PUBLIC_V2_MAPPING_HMAC_DOMAIN = "spiral-bfcl-v4-public-v2-semantic-trusted-mapping/v1"
BFCL_V4_PUBLIC_V2_MINIMUM_HMAC_SECRET_BYTES = 32

OpaqueItemId = Annotated[str, Field(pattern=r"^bfclv2-review-[0-9a-f]{64}$")]
PacketPosition = Annotated[int, Field(ge=0, lt=1_000, strict=True)]


class BfclV4PublicV2SemanticProjectionError(RuntimeError):
    """Fail-closed projection, mapping, response, or distribution rejection."""


class BfclV4PublicV2SemanticScoreExecutionError(RuntimeError):
    """Reviewer projection artifacts cannot authorize benchmark scoring."""


def _artifact_ref(value: object, media_type: str) -> ArtifactRef:
    content = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=canonical_sha256(value),
        size=len(content),
        media_type=media_type,
    )


def _validate_canonical_json(value: str, *, label: str, require_list: bool = False) -> None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not JSON") from error
    if canonical_json(parsed) != value:
        raise ValueError(f"{label} is not canonical JSON")
    if require_list and not isinstance(parsed, list):
        raise ValueError(f"{label} must contain a JSON list")


class BfclV4PublicV2SemanticPrivateBoundary(StrEnum):
    """Trusted-only population boundary used for cross-generation auditing."""

    V1 = "v1"
    FIT = "fit"
    GATE = "gate"
    HOLDOUT = "holdout"


def _source_universe_fingerprint(v1_fingerprint: str, v2_fingerprint: str) -> str:
    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-semantic-source-universe/v1",
            "v1_loaded_bundle_fingerprint": v1_fingerprint,
            "v2_loaded_bundle_fingerprint": v2_fingerprint,
            "population": "15-v1-public-pilot-plus-25-v2-public-development",
        }
    )


class BfclV4PublicV2SemanticReviewerPacketItem(ImmutableModel):
    """The complete reviewer-facing fields for one semantic grouping item."""

    opaque_item_id: OpaqueItemId
    canonical_question_json: NonEmptyStr
    canonical_function_schemas_json: NonEmptyStr

    @model_validator(mode="after")
    def _canonical_reviewer_content(self) -> Self:
        _validate_canonical_json(
            self.canonical_question_json,
            label="reviewer question",
        )
        _validate_canonical_json(
            self.canonical_function_schemas_json,
            label="reviewer function schemas",
            require_list=True,
        )
        return self


class BfclV4PublicV2SemanticReviewerPacket(ImmutableModel):
    """Forty reviewer-facing items without direct roster or partition coordinates."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL
    )
    items: Annotated[
        tuple[BfclV4PublicV2SemanticReviewerPacketItem, ...],
        Field(min_length=40, max_length=40),
    ]
    opaque_ids_hmac_sha256_derived: Literal[True] = True
    packet_order_hmac_sha256_derived: Literal[True] = True
    direct_public_task_ids_present: Literal[False] = False
    source_paths_present: Literal[False] = False
    partition_labels_present: Literal[False] = False
    source_or_candidate_hashes_present: Literal[False] = False
    source_roster_order_present: Literal[False] = False
    hmac_secret_present: Literal[False] = False
    exact_public_content_may_be_reidentified: Literal[True] = True
    partition_inference_cryptographically_prevented: Literal[False] = False
    possible_answers_present: Literal[False] = False
    scores_present: Literal[False] = False
    model_or_harness_results_present: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _unique_opaque_population(self) -> Self:
        opaque_ids = tuple(item.opaque_item_id for item in self.items)
        if len(set(opaque_ids)) != len(opaque_ids):
            raise ValueError("reviewer packet repeats an opaque item ID")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_REVIEWER_PACKET_MEDIA_TYPE)


class BfclV4PublicV2SemanticTrustedMappingEntry(ImmutableModel):
    """One private link from an opaque packet item to its trusted source row."""

    packet_position: PacketPosition
    opaque_item_id: OpaqueItemId
    order_hmac_sha256: Sha256
    public_task_id: NonEmptyStr
    source_roster_ordinal: PacketPosition
    private_boundary_label: BfclV4PublicV2SemanticPrivateBoundary
    question_git_path: NonEmptyStr
    question_blob_size: Annotated[int, Field(gt=0, strict=True)]
    question_blob_sha256: Sha256
    source_row_size: Annotated[int, Field(gt=0, strict=True)]
    source_row_sha256: Sha256
    candidate_payload_sha256: Sha256
    reviewer_content_sha256: Sha256


class BfclV4PublicV2SemanticTrustedMapping(ImmutableModel):
    """HMAC-committed private control-plane mapping; never distribute it."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL
    )
    v1_loaded_bundle_fingerprint: Sha256
    v2_loaded_bundle_fingerprint: Sha256
    source_universe_fingerprint: Sha256
    reviewer_packet_ref: ArtifactRef
    entries: Annotated[
        tuple[BfclV4PublicV2SemanticTrustedMappingEntry, ...],
        Field(min_length=40, max_length=40),
    ]
    mapping_commitment_hmac_sha256: Sha256
    trusted_control_plane_only: Literal[True] = True
    reviewer_facing: Literal[False] = False
    public_task_ids_and_partitions_present: Literal[True] = True
    source_coordinates_present: Literal[True] = True
    hmac_secret_present: Literal[False] = False
    exact_public_content_may_be_reidentified: Literal[True] = True
    partition_inference_cryptographically_prevented: Literal[False] = False
    possible_answers_present: Literal[False] = False
    scores_present: Literal[False] = False
    model_or_harness_results_present: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_mapping_shape(self) -> Self:
        if tuple(item.packet_position for item in self.entries) != tuple(range(40)):
            raise ValueError("trusted mapping does not cover exact packet order")
        if len({item.opaque_item_id for item in self.entries}) != 40:
            raise ValueError("trusted mapping repeats an opaque item ID")
        if len({item.public_task_id for item in self.entries}) != 40:
            raise ValueError("trusted mapping repeats a public task ID")
        coordinates = {
            (
                "v1"
                if item.private_boundary_label is BfclV4PublicV2SemanticPrivateBoundary.V1
                else "v2",
                item.source_roster_ordinal,
            )
            for item in self.entries
        }
        if len(coordinates) != 40:
            raise ValueError("trusted mapping repeats a source roster coordinate")
        if self.source_universe_fingerprint != _source_universe_fingerprint(
            self.v1_loaded_bundle_fingerprint,
            self.v2_loaded_bundle_fingerprint,
        ):
            raise ValueError("trusted mapping does not bind both loaded bundles")
        counts = {
            boundary: sum(item.private_boundary_label is boundary for item in self.entries)
            for boundary in BfclV4PublicV2SemanticPrivateBoundary
        }
        if counts != {
            BfclV4PublicV2SemanticPrivateBoundary.V1: 15,
            BfclV4PublicV2SemanticPrivateBoundary.FIT: 5,
            BfclV4PublicV2SemanticPrivateBoundary.GATE: 4,
            BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT: 16,
        }:
            raise ValueError("trusted mapping has wrong private boundary population")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_TRUSTED_MAPPING_MEDIA_TYPE)


def _reviewer_content_sha256(item: BfclV4PublicV2SemanticReviewerPacketItem) -> str:
    return canonical_sha256(
        {
            "canonical_question_json": item.canonical_question_json,
            "canonical_function_schemas_json": item.canonical_function_schemas_json,
        }
    )


class BfclV4PublicV2SemanticProjection(ImmutableModel):
    """Trusted materialization containing both public packet and private map."""

    schema_version: Literal["1"] = "1"
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping
    both_loaded_bundle_contracts_strictly_revalidated: Literal[True] = True
    packet_mapping_hmac_verification_required: Literal[True] = True
    trusted_control_plane_only: Literal[True] = True
    distribute_only_reviewer_packet: Literal[True] = True
    hmac_secret_present: Literal[False] = False
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    runs_or_cas_read: Literal[False] = False
    model_invoked: Literal[False] = False
    network_used: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_packet_to_private_mapping(self) -> Self:
        if self.trusted_mapping.reviewer_packet_ref != self.reviewer_packet.ref:
            raise ValueError("trusted mapping cites a different reviewer packet")
        for item, entry in zip(
            self.reviewer_packet.items,
            self.trusted_mapping.entries,
            strict=True,
        ):
            if (
                item.opaque_item_id != entry.opaque_item_id
                or _reviewer_content_sha256(item) != entry.reviewer_content_sha256
            ):
                raise ValueError("trusted mapping differs from reviewer packet content")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2SemanticReviewerKind(StrEnum):
    """Unauthenticated reviewer-kind declaration."""

    HUMAN = "human"
    MODEL_ASSISTED = "model-assisted"


class BfclV4PublicV2SemanticReviewerDeclaration(ImmutableModel):
    """Required reviewer provenance whose declarations remain unauthenticated."""

    schema_version: Literal["1"] = "1"
    declared_reviewer_kind: BfclV4PublicV2SemanticReviewerKind
    declared_principal_id: NonEmptyStr
    human_identity_commitment_sha256: Sha256 | None = None
    model_provider: NonEmptyStr | None = None
    model_name: NonEmptyStr | None = None
    model_revision: NonEmptyStr | None = None
    served_model_weights_identity_sha256: Sha256 | None = None
    model_prompt_sha256: Sha256 | None = None
    model_generation_seed_u64: (
        Annotated[
            int,
            Field(ge=0, le=2**64 - 1, strict=True),
        ]
        | None
    ) = None
    declared_independent_generation: Literal[True] = True
    declared_no_partition_roster_lookup: Literal[True] = True
    declared_no_answers_or_scores_used: Literal[True] = True
    external_reviewer_identity_verified: Literal[False] = False
    declaration_authenticated: Literal[False] = False

    @model_validator(mode="after")
    def _bind_declared_reviewer_provenance(self) -> Self:
        model_coordinates = (
            self.model_provider,
            self.model_name,
            self.model_revision,
            self.served_model_weights_identity_sha256,
            self.model_prompt_sha256,
            self.model_generation_seed_u64,
        )
        if self.declared_reviewer_kind is BfclV4PublicV2SemanticReviewerKind.HUMAN:
            if self.human_identity_commitment_sha256 is None or any(
                item is not None for item in model_coordinates
            ):
                raise ValueError("human reviewer requires only an identity commitment")
        elif self.human_identity_commitment_sha256 is not None or any(
            item is None for item in model_coordinates
        ):
            raise ValueError("model-assisted reviewer requires complete model provenance")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_REVIEWER_DECLARATION_MEDIA_TYPE)


class BfclV4PublicV2SemanticReviewerAssignment(ImmutableModel):
    """One response row keyed only by the reviewer packet's opaque ID."""

    opaque_item_id: OpaqueItemId
    semantic_family_label: Annotated[str, Field(min_length=1, max_length=200)]
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]
    abstained: Literal[False] = False


class BfclV4PublicV2SemanticReviewerResponse(ImmutableModel):
    """Opaque-keyed response requiring trusted remapping before any later audit."""

    schema_version: Literal["1"] = "1"
    reviewer_packet_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    assignments: Annotated[
        tuple[BfclV4PublicV2SemanticReviewerAssignment, ...],
        Field(min_length=40, max_length=40),
    ]
    opaque_item_ids_only: Literal[True] = True
    explicit_partition_labels_received: Literal[False] = False
    public_task_ids_received: Literal[False] = False
    partition_roster_lookup_nonuse_externally_verified: Literal[False] = False
    reviewer_identity_externally_verified: Literal[False] = False
    review_declaration_authenticated: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _unique_assignment_ids(self) -> Self:
        if len({item.opaque_item_id for item in self.assignments}) != 40:
            raise ValueError("reviewer response repeats an opaque item ID")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_REVIEWER_RESPONSE_MEDIA_TYPE)


class BfclV4PublicV2SemanticDistributionReceipt(ImmutableModel):
    """Integrity receipt that remains blocked on every external authority claim."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_SEMANTIC_PROJECTION_PROTOCOL
    )
    reviewer_packet_ref: ArtifactRef
    trusted_mapping_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    mapping_commitment_hmac_sha256: Sha256
    status: Literal["integrity-verified-external-authentication-blocked"] = (
        "integrity-verified-external-authentication-blocked"
    )
    reviewer_packet_integrity_verified: Literal[True] = True
    trusted_mapping_integrity_verified: Literal[True] = True
    mapping_commitment_hmac_verified: Literal[True] = True
    opaque_id_hmac_derivation_verified: Literal[True] = True
    packet_order_hmac_derivation_verified: Literal[True] = True
    receipt_self_authenticating: Literal[False] = False
    live_secret_reverification_required: Literal[True] = True
    external_reviewer_identity_verified: Literal[False] = False
    review_declaration_authenticated: Literal[False] = False
    distribution_channel_authenticated: Literal[False] = False
    packet_delivery_externally_verified: Literal[False] = False
    reviewer_response_admissible: Literal[False] = False
    semantic_audit_authority_issued: Literal[False] = False
    exact_public_content_may_be_reidentified: Literal[True] = True
    partition_inference_cryptographically_prevented: Literal[False] = False
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    model_invoked: Literal[False] = False
    network_used: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_DISTRIBUTION_RECEIPT_MEDIA_TYPE)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
