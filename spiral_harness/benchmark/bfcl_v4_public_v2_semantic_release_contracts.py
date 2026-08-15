"""Trusted contracts for a development-only BFCL v2 semantic release."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticDistributionReceipt,
    BfclV4PublicV2SemanticPrivateBoundary,
    BfclV4PublicV2SemanticReviewerDeclaration,
    BfclV4PublicV2SemanticReviewerResponse,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_ATTESTATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-review-attestation.v1+json"
)
BFCL_V4_PUBLIC_V2_SEMANTIC_MAPPED_REVIEW_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-mapped-review.v1+json"
)
BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-development-release.v1+json"
)
BFCL_V4_PUBLIC_V2_SEMANTIC_RELEASE_HMAC_DOMAIN = (
    "spiral-bfcl-v4-public-v2-semantic-development-release/v1"
)


def _artifact_ref(value: object, media_type: str) -> ArtifactRef:
    content = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=canonical_sha256(value),
        size=len(content),
        media_type=media_type,
    )


class BfclV4PublicV2SemanticReleaseFailureStage(StrEnum):
    """Credential-free failure stage for a fail-closed development release."""

    EVIDENCE_BINDING = "evidence-binding"
    TRANSPORT_ATTESTATION = "transport-attestation"
    REVIEW_INDEPENDENCE = "review-independence"
    DISAGREEMENT_ARBITRATION = "disagreement-arbitration"
    FAMILY_BOUNDARY = "family-boundary"


class BfclV4PublicV2SemanticReleaseError(RuntimeError):
    """Typed semantic-release rejection without provider or annotation content."""

    def __init__(self, failure_stage: BfclV4PublicV2SemanticReleaseFailureStage) -> None:
        self.failure_stage = failure_stage
        super().__init__(f"BFCL v2 semantic release rejected at {failure_stage.value}")


class BfclV4PublicV2SemanticReviewExecutionAttestation(ImmutableModel):
    """Operator attestation for one direct review call; not a provider signature."""

    schema_version: Literal["1"] = "1"
    reviewer_packet_ref: ArtifactRef
    trusted_mapping_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    distribution_receipt_ref: ArtifactRef
    reviewer_response_ref: ArtifactRef
    provider_origin_sha256: Sha256
    request_body_sha256: Sha256
    raw_response_body_sha256: Sha256
    provider_response_id: NonEmptyStr
    requested_model: NonEmptyStr
    provider_reported_model: NonEmptyStr
    provider_system_fingerprint: NonEmptyStr | None = None
    provider_seed_u64: Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)]
    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_tokens: Annotated[int, Field(ge=0, strict=True)]
    provider_attempts: Literal[1] = 1
    http_status: Literal[200] = 200
    direct_openai_compatible_session_operator_attested: Literal[True] = True
    automatic_retry_used: Literal[False] = False
    reviewer_received_only_packet_and_protocol_prompt: Literal[True] = True
    provider_response_signature_verified: Literal[False] = False
    served_weights_cryptographically_verified: Literal[False] = False
    external_reviewer_identity_verified: Literal[False] = False
    credentials_present: Literal[False] = False
    partition_labels_present: Literal[False] = False
    possible_answers_present: Literal[False] = False
    scores_present: Literal[False] = False
    model_campaign_started: Literal[False] = False

    @model_validator(mode="after")
    def _bind_usage(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("semantic review token usage is inconsistent")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_ATTESTATION_MEDIA_TYPE)


class BfclV4PublicV2SemanticReviewSessionEvidence(ImmutableModel):
    """Complete persisted evidence for one independently generated review."""

    schema_version: Literal["1"] = "1"
    reviewer_declaration: BfclV4PublicV2SemanticReviewerDeclaration
    distribution_receipt: BfclV4PublicV2SemanticDistributionReceipt
    reviewer_response: BfclV4PublicV2SemanticReviewerResponse
    execution_attestation: BfclV4PublicV2SemanticReviewExecutionAttestation

    @model_validator(mode="after")
    def _bind_nested_references(self) -> Self:
        if (
            self.distribution_receipt.reviewer_declaration_ref != self.reviewer_declaration.ref
            or self.reviewer_response.reviewer_declaration_ref != self.reviewer_declaration.ref
            or self.execution_attestation.reviewer_declaration_ref != self.reviewer_declaration.ref
            or self.execution_attestation.distribution_receipt_ref != self.distribution_receipt.ref
            or self.execution_attestation.reviewer_response_ref != self.reviewer_response.ref
        ):
            raise ValueError("semantic review session has inconsistent nested references")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2SemanticMappedAssignment(ImmutableModel):
    """Trusted remap of one opaque review row to its frozen source coordinate."""

    audit_item_id: Sha256
    candidate_payload_sha256: Sha256
    public_task_id: NonEmptyStr
    private_boundary_label: BfclV4PublicV2SemanticPrivateBoundary
    semantic_family_label: Annotated[str, Field(min_length=1, max_length=200)]
    rationale: Annotated[str, Field(min_length=1, max_length=2_000)]


class BfclV4PublicV2SemanticMappedReview(ImmutableModel):
    """One exact opaque response after trusted, HMAC-verified remapping."""

    schema_version: Literal["1"] = "1"
    reviewer_packet_ref: ArtifactRef
    trusted_mapping_ref: ArtifactRef
    reviewer_declaration_ref: ArtifactRef
    reviewer_response_ref: ArtifactRef
    distribution_receipt_ref: ArtifactRef
    execution_attestation_ref: ArtifactRef
    declared_principal_id: NonEmptyStr
    declared_model_provider: NonEmptyStr
    declared_model_name: NonEmptyStr
    declared_model_revision: NonEmptyStr
    declared_served_weights_identity_sha256: Sha256
    declared_prompt_sha256: Sha256
    declared_generation_seed_u64: Annotated[int, Field(ge=0, le=2**64 - 1, strict=True)]
    assignments: Annotated[
        tuple[BfclV4PublicV2SemanticMappedAssignment, ...],
        Field(min_length=40, max_length=40),
    ]
    exact_source_universe_coverage: Literal[True] = True
    live_hmac_mapping_verified: Literal[True] = True
    response_packet_coverage_verified: Literal[True] = True
    direct_provider_session_operator_attested: Literal[True] = True
    external_reviewer_identity_verified: Literal[False] = False
    declaration_authenticated_by_provider: Literal[False] = False
    served_weights_cryptographically_verified: Literal[False] = False
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    model_campaign_started: Literal[False] = False

    @model_validator(mode="after")
    def _close_assignment_population(self) -> Self:
        if len({item.audit_item_id for item in self.assignments}) != 40:
            raise ValueError("mapped semantic review repeats an audit item")
        if len({item.public_task_id for item in self.assignments}) != 40:
            raise ValueError("mapped semantic review repeats a public task")
        if len({item.candidate_payload_sha256 for item in self.assignments}) != 40:
            raise ValueError("mapped semantic review repeats a candidate identity")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_SEMANTIC_MAPPED_REVIEW_MEDIA_TYPE)


class BfclV4PublicV2SemanticDevelopmentRelease(ImmutableModel):
    """Development-only authority issued from independent opaque semantic reviews."""

    schema_version: Literal["1"] = "1"
    manifest_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    source_universe_fingerprint: Sha256
    reviewer_packet_ref: ArtifactRef
    trusted_mapping_ref: ArtifactRef
    primary_review_refs: Annotated[
        tuple[ArtifactRef, ArtifactRef], Field(min_length=2, max_length=2)
    ]
    primary_execution_attestation_refs: Annotated[
        tuple[ArtifactRef, ArtifactRef],
        Field(min_length=2, max_length=2),
    ]
    adjudicator_review_ref: ArtifactRef | None = None
    adjudicator_execution_attestation_ref: ArtifactRef | None = None
    final_partition_sha256: Sha256
    final_semantic_family_count: Annotated[int, Field(ge=1, le=40, strict=True)]
    primary_pairwise_disagreement_count: Annotated[int, Field(ge=0, le=780, strict=True)]
    reviewer_count: Literal[2, 3]
    cross_boundary_family_count: Literal[0] = 0
    release_authority_hmac_sha256: Sha256
    exact_40_item_population_verified: Literal[True] = True
    live_hmac_projection_integrity_verified: Literal[True] = True
    independent_primary_reviewer_coordinates_verified: Literal[True] = True
    every_primary_disagreement_resolved: Literal[True] = True
    final_cross_boundary_semantic_isolation_verified: Literal[True] = True
    operator_attested_direct_provider_sessions: Literal[True] = True
    release_hmac_present: Literal[True] = True
    live_secret_reverification_required: Literal[True] = True
    reviewer_identity_externally_verified: Literal[False] = False
    review_declarations_provider_authenticated: Literal[False] = False
    provider_response_signatures_verified: Literal[False] = False
    served_model_weights_cryptographically_verified: Literal[False] = False
    status: Literal["development-semantic-isolation-released"] = (
        "development-semantic-isolation-released"
    )
    independent_semantic_audit_release_bound: Literal[True] = True
    score_bearing_execution_allowed: Literal[True] = True
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    confirmatory_evidence: Literal[False] = False
    descriptive_public_development_result_allowed: Literal[True] = True
    possible_answers_read: Literal[False] = False
    scores_read: Literal[False] = False
    model_campaign_started: Literal[False] = False

    @model_validator(mode="after")
    def _bind_adjudication_shape(self) -> Self:
        adjudicated = self.adjudicator_review_ref is not None
        if adjudicated != (self.adjudicator_execution_attestation_ref is not None):
            raise ValueError("semantic release has incomplete adjudicator evidence")
        if adjudicated != (self.primary_pairwise_disagreement_count > 0):
            raise ValueError(
                "semantic release adjudication differs from primary disagreement count"
            )
        if self.reviewer_count != (3 if adjudicated else 2):
            raise ValueError("semantic release reviewer count differs from adjudication shape")
        if self.primary_review_refs[0] == self.primary_review_refs[1]:
            raise ValueError("semantic release repeats a primary review")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _artifact_ref(self, BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
