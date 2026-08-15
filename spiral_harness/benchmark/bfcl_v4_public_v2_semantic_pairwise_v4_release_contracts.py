"""Release contracts for composite chunked-pairwise BFCL semantic evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE,
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELATION_MEDIA_TYPE,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256

BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-semantic-composite-release.v4+json"
)
BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELEASE_HMAC_DOMAIN = (
    "spiral-bfcl-v4-public-v2-semantic-composite-release/v4"
)


def _ref(value: object, media_type: str) -> ArtifactRef:
    content = canonical_json_bytes(value)
    return ArtifactRef(sha256=canonical_sha256(value), size=len(content), media_type=media_type)


class BfclV4PublicV2PairwiseV4ReleaseFailureStage(StrEnum):
    """Sanitized fail-closed stage for composite release issuance."""

    EVIDENCE_BINDING = "evidence-binding"
    MAPPING_INTEGRITY = "mapping-integrity"
    REVIEW_INDEPENDENCE = "review-independence"
    DISAGREEMENT_ARBITRATION = "disagreement-arbitration"
    FAMILY_BOUNDARY = "family-boundary"


class BfclV4PublicV2PairwiseV4ReleaseError(RuntimeError):
    """Typed composite-release rejection without annotation content."""

    def __init__(self, failure_stage: BfclV4PublicV2PairwiseV4ReleaseFailureStage) -> None:
        self.failure_stage = failure_stage
        super().__init__(f"BFCL v2 pairwise v4 release rejected at {failure_stage.value}")


class BfclV4PublicV2PairwiseV4DevelopmentRelease(ImmutableModel):
    """Development authority backed by two or three 15-call composite reviews."""

    schema_version: Literal["4"] = "4"
    manifest_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    source_universe_fingerprint: Sha256
    reviewer_packet_ref: ArtifactRef
    trusted_mapping_ref: ArtifactRef
    predecessor_closure_ref: ArtifactRef
    primary_composite_evidence_refs: Annotated[
        tuple[ArtifactRef, ArtifactRef], Field(min_length=2, max_length=2)
    ]
    primary_relation_refs: Annotated[
        tuple[ArtifactRef, ArtifactRef], Field(min_length=2, max_length=2)
    ]
    adjudicator_composite_evidence_ref: ArtifactRef | None = None
    adjudicator_relation_ref: ArtifactRef | None = None
    final_partition_sha256: Sha256
    final_semantic_family_count: Annotated[int, Field(ge=1, le=40, strict=True)]
    primary_pairwise_disagreement_count: Annotated[int, Field(ge=0, le=780, strict=True)]
    reviewer_count: Literal[2, 3]
    composite_call_count: Literal[30, 45]
    reviewed_unordered_pair_decision_count: Literal[1_560, 2_340]
    cross_boundary_family_count: Literal[0] = 0
    release_authority_hmac_sha256: Sha256
    exact_40_item_population_verified: Literal[True] = True
    exact_780_pairs_per_reviewer_verified: Literal[True] = True
    every_composite_relation_is_equivalence_relation: Literal[True] = True
    every_call_single_attempt_no_retry_verified: Literal[True] = True
    every_primary_disagreement_resolved: Literal[True] = True
    final_cross_boundary_semantic_isolation_verified: Literal[True] = True
    live_hmac_projection_integrity_verified: Literal[True] = True
    independent_primary_reviewer_coordinates_verified: Literal[True] = True
    evidence_shape: Literal["chunked-pairwise-composite-v4"] = "chunked-pairwise-composite-v4"
    single_call_review_attestation_present: Literal[False] = False
    composite_reviewer_evidence_bound: Literal[True] = True
    release_hmac_present: Literal[True] = True
    live_secret_reverification_required: Literal[True] = True
    reviewer_identity_externally_verified: Literal[False] = False
    provider_response_signatures_verified: Literal[False] = False
    served_model_weights_cryptographically_verified: Literal[False] = False
    status: Literal["development-semantic-isolation-composite-v4-released"] = (
        "development-semantic-isolation-composite-v4-released"
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
    def _bind_evidence_shape(self) -> Self:
        adjudicated = self.adjudicator_composite_evidence_ref is not None
        if adjudicated != (self.adjudicator_relation_ref is not None):
            raise ValueError("composite release has incomplete adjudicator evidence")
        if adjudicated != (self.primary_pairwise_disagreement_count > 0):
            raise ValueError("composite release adjudication differs from disagreement count")
        expected_reviewers = 3 if adjudicated else 2
        if (
            self.reviewer_count != expected_reviewers
            or self.composite_call_count != expected_reviewers * 15
            or self.reviewed_unordered_pair_decision_count != expected_reviewers * 780
        ):
            raise ValueError("composite release counts differ from reviewer evidence shape")
        if len(set(self.primary_composite_evidence_refs)) != 2:
            raise ValueError("composite release repeats primary evidence")
        evidence_refs = self.primary_composite_evidence_refs + (
            ()
            if self.adjudicator_composite_evidence_ref is None
            else (self.adjudicator_composite_evidence_ref,)
        )
        relation_refs = self.primary_relation_refs + (
            () if self.adjudicator_relation_ref is None else (self.adjudicator_relation_ref,)
        )
        if any(
            ref.media_type != BFCL_V4_PUBLIC_V2_PAIRWISE_V4_COMPOSITE_EVIDENCE_MEDIA_TYPE
            for ref in evidence_refs
        ) or any(
            ref.media_type != BFCL_V4_PUBLIC_V2_PAIRWISE_V4_RELATION_MEDIA_TYPE
            for ref in relation_refs
        ):
            raise ValueError("composite release cites the wrong evidence media type")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        return _ref(self, BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
