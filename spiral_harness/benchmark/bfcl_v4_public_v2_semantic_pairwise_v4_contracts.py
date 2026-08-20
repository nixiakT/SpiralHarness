"""Contracts for provider-free chunked pairwise BFCL semantic review v4."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from enum import StrEnum
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticReviewerPacketItem,
    OpaqueItemId,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL = (
    "spiral-bfcl-v4-public-v2-chunked-pairwise-semantic-review/v4"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_PROTOCOL = "spiral-bfcl-v4-public-v2-semantic-review/v1"
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-predecessor-closure.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_FAILURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-predecessor-failure.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_LINEAGE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-lineage.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PLAN_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-plan.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DECLARATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-declaration.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_REQUEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-request.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RESPONSE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-response.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_ATTEMPT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-attempt.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-relation.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-pairwise-composite-evidence.v4+json"
)

BlockIndex = Annotated[int, Field(ge=0, lt=5, strict=True)]
CallIndex = Annotated[int, Field(ge=0, lt=15, strict=True)]
PairwiseCallId = Annotated[str, Field(pattern=r"^pairwise-v4-[a-z0-9-]+$")]


def _ref(value: object, media_type: str) -> ArtifactRef:
    content = canonical_json_bytes(value)
    return ArtifactRef(sha256=canonical_sha256(value), size=len(content), media_type=media_type)


class BfclV4PublicV2PairwiseV4Error(RuntimeError):
    """Fail-closed planning, parsing, or replay error."""


class BfclV4PublicV2PairwiseV4ReviewerRole(StrEnum):
    """Independent reviewer role in the v4 release protocol."""

    PRIMARY_ONE = "primary-one"
    PRIMARY_TWO = "primary-two"
    ADJUDICATOR = "adjudicator"


class BfclV4PublicV2PairwiseV4PredecessorLineage(StrEnum):
    """The three exhausted full-packet lineages superseded by v4."""

    SEMANTIC_V1 = "semantic-v1"
    SEMANTIC_V2 = "semantic-v2"
    SEMANTIC_V3 = "semantic-v3"


class BfclV4PublicV2PairwiseV4CallKind(StrEnum):
    """The two fixed pair-coverage call shapes."""

    WITHIN = "within"
    CROSS = "cross"


class BfclV4PublicV2PairwiseV4TerminatedAttempt(ImmutableModel):
    """One exhausted full-packet predecessor call, never reusable by v4."""

    predecessor_lineage: BfclV4PublicV2PairwiseV4PredecessorLineage
    review_role: Literal["primary-one"] = "primary-one"
    attempt_ref: ArtifactRef
    provider_attempts: Literal[1] = 1
    valid_40_item_response_produced: Literal[False] = False
    automatic_retry_used: Literal[False] = False
    retry_or_repair_allowed: Literal[False] = False
    admitted_to_v4_lineage: Literal[False] = False


class BfclV4PublicV2PairwiseV4PredecessorClosure(ImmutableModel):
    """Evidence that all three v1 full-packet calls ended without retry."""

    schema_version: Literal["4"] = "4"
    predecessor_protocol: Literal[BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_PROTOCOL
    )
    attempts: Annotated[
        tuple[BfclV4PublicV2PairwiseV4TerminatedAttempt, ...],
        Field(min_length=3, max_length=3),
    ]
    predecessor_call_count: Literal[3] = 3
    predecessor_lineage_closed: Literal[True] = True
    predecessor_outputs_reusable: Literal[False] = False
    predecessor_retry_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _exact_predecessor_lineages(self) -> Self:
        expected = tuple(BfclV4PublicV2PairwiseV4PredecessorLineage)
        if tuple(item.predecessor_lineage for item in self.attempts) != expected:
            raise ValueError("predecessor closure must cover v1-v3 in fixed order")
        if len({item.attempt_ref for item in self.attempts}) != 3:
            raise ValueError("predecessor closure repeats an attempt artifact")
        if any(
            item.attempt_ref.media_type
            != BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_FAILURE_MEDIA_TYPE
            for item in self.attempts
        ):
            raise ValueError("predecessor closure cites the wrong failure artifact type")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_CLOSURE_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4Lineage(ImmutableModel):
    """Fresh reviewer lineage cryptographically separated from the three v1 calls."""

    schema_version: Literal["4"] = "4"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL
    )
    reviewer_role: BfclV4PublicV2PairwiseV4ReviewerRole
    predecessor_closure_ref: ArtifactRef
    reviewer_nonce_sha256: Sha256
    seed_root_sha256: Sha256
    lineage_sha256: Sha256
    fresh_lineage: Literal[True] = True
    predecessor_request_or_output_reused: Literal[False] = False
    predecessor_attempt_retried: Literal[False] = False

    @model_validator(mode="after")
    def _bind_fresh_lineage(self) -> Self:
        if (
            self.predecessor_closure_ref.media_type
            != BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PREDECESSOR_CLOSURE_MEDIA_TYPE
        ):
            raise ValueError("pairwise lineage cites the wrong predecessor closure type")
        expected = canonical_sha256(
            {
                "domain": "spiral-bfcl-v4-public-v2-pairwise-v4-lineage/v1",
                "reviewer_role": self.reviewer_role,
                "predecessor_closure_ref": self.predecessor_closure_ref,
                "reviewer_nonce_sha256": self.reviewer_nonce_sha256,
                "seed_root_sha256": self.seed_root_sha256,
            }
        )
        if self.lineage_sha256 != expected:
            raise ValueError("pairwise lineage identity is not derived from its closed inputs")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_LINEAGE_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4Pair(ImmutableModel):
    """One canonically oriented opaque unordered pair."""

    left_opaque_item_id: OpaqueItemId
    right_opaque_item_id: OpaqueItemId

    @model_validator(mode="after")
    def _different_endpoints(self) -> Self:
        if self.left_opaque_item_id == self.right_opaque_item_id:
            raise ValueError("pair endpoints must differ")
        return self


class BfclV4PublicV2PairwiseV4Block(ImmutableModel):
    """One fixed contiguous eight-item block from packet order."""

    block_index: BlockIndex
    opaque_item_ids: Annotated[tuple[OpaqueItemId, ...], Field(min_length=8, max_length=8)]

    @model_validator(mode="after")
    def _unique_items(self) -> Self:
        if len(set(self.opaque_item_ids)) != 8:
            raise ValueError("pairwise block repeats an opaque item")
        return self


class BfclV4PublicV2PairwiseV4CallSpec(ImmutableModel):
    """One immutable within-block or cross-block review call."""

    call_index: CallIndex
    call_id: PairwiseCallId
    call_kind: BfclV4PublicV2PairwiseV4CallKind
    block_indices: Annotated[tuple[BlockIndex, ...], Field(min_length=1, max_length=2)]
    opaque_item_ids: Annotated[tuple[OpaqueItemId, ...], Field(min_length=8, max_length=16)]
    ordered_pairs: Annotated[
        tuple[BfclV4PublicV2PairwiseV4Pair, ...], Field(min_length=28, max_length=64)
    ]
    provider_seed_u64: Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)]
    provider_attempt_budget: Literal[1] = 1
    automatic_retry_allowed: Literal[False] = False
    repair_or_backfill_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_call_shape(self) -> Self:
        if self.call_kind is BfclV4PublicV2PairwiseV4CallKind.WITHIN:
            if len(self.block_indices) != 1 or len(self.opaque_item_ids) != 8:
                raise ValueError("within call must contain one complete eight-item block")
            expected_count = 28
        else:
            if (
                len(self.block_indices) != 2
                or self.block_indices[0] >= self.block_indices[1]
                or len(self.opaque_item_ids) != 16
            ):
                raise ValueError("cross call must contain two ordered complete blocks")
            expected_count = 64
        if len(self.ordered_pairs) != expected_count:
            raise ValueError("pairwise call has the wrong pair count")
        if len(set(self.opaque_item_ids)) != len(self.opaque_item_ids):
            raise ValueError("pairwise call repeats an opaque item")
        return self


def _expected_call_coordinates() -> tuple[tuple[int, ...], ...]:
    within = tuple((index,) for index in range(5))
    cross = tuple(combinations(range(5), 2))
    return within + cross


def _pairs_for_blocks(
    blocks: tuple[BfclV4PublicV2PairwiseV4Block, ...],
    coordinate: tuple[int, ...],
) -> tuple[BfclV4PublicV2PairwiseV4Pair, ...]:
    if len(coordinate) == 1:
        return tuple(
            BfclV4PublicV2PairwiseV4Pair(
                left_opaque_item_id=left,
                right_opaque_item_id=right,
            )
            for left, right in combinations(blocks[coordinate[0]].opaque_item_ids, 2)
        )
    return tuple(
        BfclV4PublicV2PairwiseV4Pair(
            left_opaque_item_id=left,
            right_opaque_item_id=right,
        )
        for left in blocks[coordinate[0]].opaque_item_ids
        for right in blocks[coordinate[1]].opaque_item_ids
    )


def _call_id(coordinate: tuple[int, ...]) -> str:
    if len(coordinate) == 1:
        return f"pairwise-v4-within-{coordinate[0]}"
    return f"pairwise-v4-cross-{coordinate[0]}-{coordinate[1]}"


class BfclV4PublicV2PairwiseV4Plan(ImmutableModel):
    """Exact 5x8, 15-call, 780-pair plan for one reviewer lineage."""

    schema_version: Literal["4"] = "4"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PROTOCOL
    )
    reviewer_packet_ref: ArtifactRef
    lineage_ref: ArtifactRef
    seed_root_sha256: Sha256
    blocks: Annotated[tuple[BfclV4PublicV2PairwiseV4Block, ...], Field(min_length=5, max_length=5)]
    calls: Annotated[
        tuple[BfclV4PublicV2PairwiseV4CallSpec, ...], Field(min_length=15, max_length=15)
    ]
    pair_manifest_sha256: Sha256
    block_count: Literal[5] = 5
    items_per_block: Literal[8] = 8
    call_count: Literal[15] = 15
    unordered_pair_count: Literal[780] = 780
    exact_pair_coverage_required: Literal[True] = True

    @model_validator(mode="after")
    def _exact_schedule(self) -> Self:
        if tuple(block.block_index for block in self.blocks) != tuple(range(5)):
            raise ValueError("pairwise plan blocks are not in fixed order")
        item_ids = tuple(item for block in self.blocks for item in block.opaque_item_ids)
        if len(set(item_ids)) != 40:
            raise ValueError("pairwise plan does not contain 40 unique opaque items")
        coordinates = _expected_call_coordinates()
        expected_pairs = tuple(_pairs_for_blocks(self.blocks, item) for item in coordinates)
        for index, (call, coordinate, pairs) in enumerate(
            zip(self.calls, coordinates, expected_pairs, strict=True)
        ):
            expected_kind = (
                BfclV4PublicV2PairwiseV4CallKind.WITHIN
                if len(coordinate) == 1
                else BfclV4PublicV2PairwiseV4CallKind.CROSS
            )
            expected_items = tuple(
                item
                for block_index in coordinate
                for item in self.blocks[block_index].opaque_item_ids
            )
            if (
                call.call_index != index
                or call.call_id != _call_id(coordinate)
                or call.call_kind is not expected_kind
                or call.block_indices != coordinate
                or call.opaque_item_ids != expected_items
                or call.ordered_pairs != pairs
                or call.provider_seed_u64
                != int(
                    canonical_sha256(
                        {
                            "domain": "spiral-bfcl-v4-public-v2-pairwise-v4-provider-seed/v1",
                            "seed_root_sha256": self.seed_root_sha256,
                            "call_id": call.call_id,
                        }
                    )[:16],
                    16,
                )
            ):
                raise ValueError("pairwise plan call differs from the fixed schedule")
        flattened = tuple(pair for group in expected_pairs for pair in group)
        if len(flattened) != 780 or len(set(flattened)) != 780:
            raise ValueError("pairwise plan does not cover 780 unordered pairs exactly once")
        if set(flattened) != {
            BfclV4PublicV2PairwiseV4Pair(
                left_opaque_item_id=left,
                right_opaque_item_id=right,
            )
            for left, right in combinations(item_ids, 2)
        }:
            raise ValueError("pairwise plan pair manifest differs from the 40-item universe")
        if self.pair_manifest_sha256 != canonical_sha256(flattened):
            raise ValueError("pairwise plan pair manifest fingerprint is invalid")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_PLAN_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4ReviewerDeclaration(ImmutableModel):
    """Unauthenticated reviewer coordinates shared by all 15 calls."""

    schema_version: Literal["4"] = "4"
    plan_ref: ArtifactRef
    lineage_ref: ArtifactRef
    reviewer_role: BfclV4PublicV2PairwiseV4ReviewerRole
    declared_principal_id: NonEmptyStr
    model_provider: NonEmptyStr
    model_name: NonEmptyStr
    model_revision: NonEmptyStr
    served_model_weights_identity_sha256: Sha256
    model_prompt_sha256: Sha256
    declared_independent_generation: Literal[True] = True
    declaration_authenticated: Literal[False] = False
    served_weights_cryptographically_verified: Literal[False] = False
    declared_no_partition_roster_lookup: Literal[True] = True
    declared_no_answers_scores_or_prior_outputs_used: Literal[True] = True

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DECLARATION_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4CallRequest(ImmutableModel):
    """Trusted wrapper for one minimal model-visible pairwise call."""

    schema_version: Literal["4"] = "4"
    plan_ref: ArtifactRef
    reviewer_packet_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    call: BfclV4PublicV2PairwiseV4CallSpec
    items: Annotated[
        tuple[BfclV4PublicV2SemanticReviewerPacketItem, ...], Field(min_length=8, max_length=16)
    ]
    private_mapping_present: Literal[False] = False
    task_ids_present: Literal[False] = False
    partition_labels_present: Literal[False] = False
    possible_answers_present: Literal[False] = False
    scores_or_campaign_outputs_present: Literal[False] = False
    predecessor_outputs_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_visible_items(self) -> Self:
        if tuple(item.opaque_item_id for item in self.items) != self.call.opaque_item_ids:
            raise ValueError("pairwise request items differ from its fixed call")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_REQUEST_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4Decision(ImmutableModel):
    """One strict reviewer decision for a scheduled pair."""

    left_opaque_item_id: OpaqueItemId
    right_opaque_item_id: OpaqueItemId
    same_semantic_family: bool = Field(strict=True)
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]

    @property
    def pair(self) -> BfclV4PublicV2PairwiseV4Pair:
        return BfclV4PublicV2PairwiseV4Pair(
            left_opaque_item_id=self.left_opaque_item_id,
            right_opaque_item_id=self.right_opaque_item_id,
        )


class BfclV4PublicV2PairwiseV4CallResponse(ImmutableModel):
    """Strictly parsed exact ordered pair coverage for one call."""

    schema_version: Literal["4"] = "4"
    request_ref: ArtifactRef
    plan_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    call_id: PairwiseCallId
    decisions: Annotated[
        tuple[BfclV4PublicV2PairwiseV4Decision, ...], Field(min_length=28, max_length=64)
    ]
    strict_json_verified: Literal[True] = True
    exact_ordered_pair_coverage_verified: Literal[True] = True
    repair_or_backfill_used: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RESPONSE_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4AttemptEvidence(ImmutableModel):
    """Operator-recorded evidence for one call, not a provider signature."""

    schema_version: Literal["4"] = "4"
    request_ref: ArtifactRef
    response_ref: ArtifactRef
    provider_origin_sha256: Sha256
    model_visible_request_sha256: Sha256
    raw_completion_content_sha256: Sha256
    provider_response_id: NonEmptyStr
    requested_model: NonEmptyStr
    provider_reported_model: NonEmptyStr
    provider_seed_u64: Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)]
    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_tokens: Annotated[int, Field(ge=0, strict=True)]
    provider_attempts: Literal[1] = 1
    operator_attested_single_provider_attempt: Literal[True] = True
    automatic_retry_used: Literal[False] = False
    repair_or_backfill_used: Literal[False] = False
    provider_response_signature_verified: Literal[False] = False
    served_weights_cryptographically_verified: Literal[False] = False
    provider_transport_performed_by_this_module: Literal[False] = False
    credentials_present: Literal[False] = False

    @model_validator(mode="after")
    def _usage_and_model(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("pairwise call token usage is inconsistent")
        if self.requested_model != self.provider_reported_model:
            raise ValueError("pairwise call provider reported another model")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_ATTEMPT_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4CallSession(ImmutableModel):
    """One request, response, and single-attempt evidence bundle."""

    request: BfclV4PublicV2PairwiseV4CallRequest
    response: BfclV4PublicV2PairwiseV4CallResponse
    attempt_evidence: BfclV4PublicV2PairwiseV4AttemptEvidence
    raw_completion_content_base64: NonEmptyStr

    @model_validator(mode="after")
    def _bind_session(self) -> Self:
        if (
            self.response.request_ref != self.request.ref
            or self.response.plan_ref != self.request.plan_ref
            or self.response.reviewer_declaration_ref != self.request.reviewer_declaration_ref
            or self.response.call_id != self.request.call.call_id
            or self.attempt_evidence.request_ref != self.request.ref
            or self.attempt_evidence.response_ref != self.response.ref
            or self.attempt_evidence.provider_seed_u64 != self.request.call.provider_seed_u64
        ):
            raise ValueError("pairwise call session has inconsistent nested references")
        try:
            raw_completion = b64decode(self.raw_completion_content_base64, validate=True)
        except (BinasciiError, ValueError):
            raise ValueError("pairwise call session has invalid completion bytes") from None
        if sha256_bytes(raw_completion) != self.attempt_evidence.raw_completion_content_sha256:
            raise ValueError("pairwise call session completion hash differs from its bytes")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2PairwiseV4RelationEdge(ImmutableModel):
    """One complete-relation edge with its source call."""

    pair: BfclV4PublicV2PairwiseV4Pair
    same_semantic_family: bool = Field(strict=True)
    source_call_id: PairwiseCallId
    source_response_ref: ArtifactRef


class BfclV4PublicV2PairwiseV4Relation(ImmutableModel):
    """A complete 40-item equivalence relation reconstructed from 15 calls."""

    schema_version: Literal["4"] = "4"
    reviewer_packet_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    ordered_opaque_item_ids: Annotated[
        tuple[OpaqueItemId, ...], Field(min_length=40, max_length=40)
    ]
    source_response_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=15, max_length=15)]
    edges: Annotated[
        tuple[BfclV4PublicV2PairwiseV4RelationEdge, ...], Field(min_length=780, max_length=780)
    ]
    equivalence_classes: Annotated[
        tuple[tuple[OpaqueItemId, ...], ...], Field(min_length=1, max_length=40)
    ]
    unordered_pair_count: Literal[780] = 780
    reflexivity_implied_for_all_items: Literal[True] = True
    symmetry_implied_by_canonical_unordered_pairs: Literal[True] = True
    transitivity_verified: Literal[True] = True
    complete_equivalence_relation_verified: Literal[True] = True

    @model_validator(mode="after")
    def _validate_equivalence_relation(self) -> Self:
        item_ids = self.ordered_opaque_item_ids
        if len(set(item_ids)) != 40 or len(set(self.source_response_refs)) != 15:
            raise ValueError("pairwise relation lacks unique items or response sources")
        expected_pairs = tuple(
            BfclV4PublicV2PairwiseV4Pair(
                left_opaque_item_id=left,
                right_opaque_item_id=right,
            )
            for left, right in combinations(item_ids, 2)
        )
        if tuple(edge.pair for edge in self.edges) != expected_pairs:
            raise ValueError("pairwise relation lacks exact canonical 780-pair order")
        if any(edge.source_response_ref not in self.source_response_refs for edge in self.edges):
            raise ValueError("pairwise relation edge cites an unknown response")

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

        for edge in self.edges:
            if edge.same_semantic_family:
                union(edge.pair.left_opaque_item_id, edge.pair.right_opaque_item_id)
        for edge in self.edges:
            implied = find(edge.pair.left_opaque_item_id) == find(edge.pair.right_opaque_item_id)
            if edge.same_semantic_family is not implied:
                raise ValueError("pairwise decisions do not define an equivalence relation")
        groups: dict[str, list[str]] = {}
        for item in item_ids:
            groups.setdefault(find(item), []).append(item)
        position = {item: index for index, item in enumerate(item_ids)}
        expected_classes = tuple(
            sorted(
                (tuple(members) for members in groups.values()),
                key=lambda members: position[members[0]],
            )
        )
        if self.equivalence_classes != expected_classes:
            raise ValueError("pairwise equivalence classes differ from the complete relation")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELATION_MEDIA_TYPE)


class BfclV4PublicV2PairwiseV4CompositeEvidence(ImmutableModel):
    """Honest 15-call reviewer evidence; never represented as one attestation."""

    schema_version: Literal["4"] = "4"
    predecessor_closure_ref: ArtifactRef
    lineage: BfclV4PublicV2PairwiseV4Lineage
    reviewer_declaration: BfclV4PublicV2PairwiseV4ReviewerDeclaration
    plan: BfclV4PublicV2PairwiseV4Plan
    call_sessions: Annotated[
        tuple[BfclV4PublicV2PairwiseV4CallSession, ...], Field(min_length=15, max_length=15)
    ]
    relation: BfclV4PublicV2PairwiseV4Relation
    composite_call_count: Literal[15] = 15
    composite_unordered_pair_count: Literal[780] = 780
    all_calls_single_attempt: Literal[True] = True
    any_retry_repair_or_backfill_used: Literal[False] = False
    single_call_review: Literal[False] = False
    single_call_attestation_present: Literal[False] = False
    composite_reviewer_evidence: Literal[True] = True
    provider_response_signatures_verified: Literal[False] = False
    served_weights_cryptographically_verified: Literal[False] = False

    @model_validator(mode="after")
    def _bind_composite(self) -> Self:
        if (
            self.predecessor_closure_ref != self.lineage.predecessor_closure_ref
            or self.plan.lineage_ref != self.lineage.ref
            or self.reviewer_declaration.lineage_ref != self.lineage.ref
            or self.reviewer_declaration.plan_ref != self.plan.ref
            or self.reviewer_declaration.reviewer_role is not self.lineage.reviewer_role
            or self.relation.reviewer_packet_ref != self.plan.reviewer_packet_ref
            or self.relation.reviewer_declaration_ref != self.reviewer_declaration.ref
        ):
            raise ValueError("pairwise composite evidence has inconsistent roots")
        if tuple(item.request.call.call_index for item in self.call_sessions) != tuple(range(15)):
            raise ValueError("pairwise composite sessions are not in exact plan order")
        if tuple(item.request.plan_ref for item in self.call_sessions) != (self.plan.ref,) * 15:
            raise ValueError("pairwise composite session cites another plan")
        if (
            tuple(item.request.reviewer_declaration_ref for item in self.call_sessions)
            != (self.reviewer_declaration.ref,) * 15
        ):
            raise ValueError("pairwise composite session cites another reviewer")
        if (
            tuple(item.response.ref for item in self.call_sessions)
            != self.relation.source_response_refs
        ):
            raise ValueError("pairwise relation cites another response set")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
