"""Mint non-serializable semantic authority only after complete live verification."""

from __future__ import annotations

import hmac
import os
import secrets
import weakref
from hashlib import sha256
from typing import NoReturn

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority_contracts import (
    BfclV4PublicV2SemanticReleaseEvidenceShape,
    bfcl_v4_public_v2_semantic_release_media_type,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_contracts import (
    BfclV4PublicV2PairwiseV4CompositeEvidence,
    BfclV4PublicV2PairwiseV4PredecessorClosure,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release import (
    verify_bfcl_v4_public_v2_pairwise_v4_development_release,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release_contracts import (
    BfclV4PublicV2PairwiseV4DevelopmentRelease,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticTrustedMapping,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release import (
    verify_bfcl_v4_public_v2_semantic_development_release,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticDevelopmentRelease,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef

_PROCESS_CAPABILITY_KEY = secrets.token_bytes(32)
_ISSUED_CAPABILITIES: weakref.WeakSet[BfclV4PublicV2VerifiedSemanticReleaseCapability]


class BfclV4PublicV2SemanticAuthorityError(ValueError):
    """A value is not a live capability minted by a complete verification path."""


class BfclV4PublicV2VerifiedSemanticReleaseCapability:
    """Opaque process-local proof that one complete semantic release was verified."""

    __slots__ = (
        "__weakref__",
        "_evidence_shape",
        "_process_id",
        "_process_tag",
        "_release_fingerprint",
        "_release_ref",
        "_verification_input_fingerprint",
    )

    def __new__(cls, *_args: object, **_kwargs: object) -> NoReturn:
        del cls, _args, _kwargs
        raise BfclV4PublicV2SemanticAuthorityError(
            "verified semantic authority can only be minted by complete verification"
        )

    def __setattr__(self, _name: str, _value: object) -> NoReturn:
        raise BfclV4PublicV2SemanticAuthorityError("verified semantic authority is immutable")

    def __delattr__(self, _name: str) -> NoReturn:
        raise BfclV4PublicV2SemanticAuthorityError("verified semantic authority is immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("verified semantic authority cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("verified semantic authority cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified semantic authority cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("verified semantic authority cannot be serialized")

    @property
    def evidence_shape(self) -> BfclV4PublicV2SemanticReleaseEvidenceShape:
        return self._evidence_shape

    @property
    def release_ref(self) -> ArtifactRef:
        return self._release_ref

    @property
    def release_fingerprint(self) -> str:
        return self._release_fingerprint

    @property
    def verification_input_fingerprint(self) -> str:
        return self._verification_input_fingerprint

    def __repr__(self) -> str:
        return (
            "BfclV4PublicV2VerifiedSemanticReleaseCapability("
            f"shape={self.evidence_shape.value!r}, release={self.release_fingerprint!r})"
        )


_ISSUED_CAPABILITIES = weakref.WeakSet()


def _binding_payload(
    shape: BfclV4PublicV2SemanticReleaseEvidenceShape,
    release_ref: ArtifactRef,
    release_fingerprint: str,
    verification_input_fingerprint: str,
    process_id: int,
) -> dict[str, object]:
    return {
        "domain": "spiral-bfcl-v4-public-v2-process-local-semantic-authority/v1",
        "evidence_shape": shape,
        "release_ref": release_ref,
        "release_fingerprint": release_fingerprint,
        "verification_input_fingerprint": verification_input_fingerprint,
        "process_id": process_id,
    }


def _process_tag(payload: object) -> str:
    return hmac.new(_PROCESS_CAPABILITY_KEY, canonical_json_bytes(payload), sha256).hexdigest()


def _mint(
    *,
    shape: BfclV4PublicV2SemanticReleaseEvidenceShape,
    release_ref: ArtifactRef,
    release_fingerprint: str,
    verification_input_fingerprint: str,
) -> BfclV4PublicV2VerifiedSemanticReleaseCapability:
    if (
        release_ref.media_type != bfcl_v4_public_v2_semantic_release_media_type(shape)
        or release_ref.sha256 != release_fingerprint
        or release_ref.size <= 0
    ):
        raise BfclV4PublicV2SemanticAuthorityError("verified release binding is inconsistent")
    capability = object.__new__(BfclV4PublicV2VerifiedSemanticReleaseCapability)
    object.__setattr__(capability, "_evidence_shape", shape)
    object.__setattr__(capability, "_release_ref", release_ref)
    object.__setattr__(capability, "_release_fingerprint", release_fingerprint)
    object.__setattr__(capability, "_process_id", os.getpid())
    object.__setattr__(
        capability,
        "_verification_input_fingerprint",
        verification_input_fingerprint,
    )
    object.__setattr__(
        capability,
        "_process_tag",
        _process_tag(
            _binding_payload(
                shape,
                release_ref,
                release_fingerprint,
                verification_input_fingerprint,
                capability._process_id,
            )
        ),
    )
    _ISSUED_CAPABILITIES.add(capability)
    return capability


def validate_bfcl_v4_public_v2_verified_semantic_release_capability(
    value: object,
) -> BfclV4PublicV2VerifiedSemanticReleaseCapability:
    """Reject copied, reconstructed, cross-process, or internally altered capabilities."""

    if type(value) is not BfclV4PublicV2VerifiedSemanticReleaseCapability:
        raise BfclV4PublicV2SemanticAuthorityError(
            "semantic authority must be an exact process-local verified capability"
        )
    capability = value
    try:
        payload = _binding_payload(
            capability.evidence_shape,
            capability.release_ref,
            capability.release_fingerprint,
            capability.verification_input_fingerprint,
            capability._process_id,
        )
        expected_tag = _process_tag(payload)
        valid = (
            capability in _ISSUED_CAPABILITIES
            and capability._process_id == os.getpid()
            and type(capability.evidence_shape) is BfclV4PublicV2SemanticReleaseEvidenceShape
            and capability.release_ref.media_type
            == bfcl_v4_public_v2_semantic_release_media_type(capability.evidence_shape)
            and capability.release_ref.sha256 == capability.release_fingerprint
            and capability.release_ref.size > 0
            and hmac.compare_digest(capability._process_tag, expected_tag)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise BfclV4PublicV2SemanticAuthorityError(
            "semantic authority was not minted by this process verification path"
        )
    return capability


def verify_bfcl_v4_public_v2_legacy_semantic_authority(
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
) -> BfclV4PublicV2VerifiedSemanticReleaseCapability:
    """Reverify complete legacy-v1 evidence, then mint one process-local capability."""

    checked = verify_bfcl_v4_public_v2_semantic_development_release(
        release,
        reviewer_packet,
        trusted_mapping,
        primary_sessions,
        runtime_hmac_secret=runtime_hmac_secret,
        adjudicator_session=adjudicator_session,
    )
    input_fingerprint = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-legacy-authority-input/v1",
            "release": checked,
            "reviewer_packet": reviewer_packet,
            "trusted_mapping": trusted_mapping,
            "primary_sessions": primary_sessions,
            "adjudicator_session": adjudicator_session,
        }
    )
    return _mint(
        shape=BfclV4PublicV2SemanticReleaseEvidenceShape.LEGACY_SINGLE_PACKET_V1,
        release_ref=checked.ref,
        release_fingerprint=checked.fingerprint,
        verification_input_fingerprint=input_fingerprint,
    )


def verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority(
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
) -> BfclV4PublicV2VerifiedSemanticReleaseCapability:
    """Reverify and replay complete composite-v4 evidence before minting authority."""

    checked = verify_bfcl_v4_public_v2_pairwise_v4_development_release(
        release,
        reviewer_packet,
        trusted_mapping,
        predecessor_closure,
        primary_composite_evidence,
        runtime_hmac_secret=runtime_hmac_secret,
        adjudicator_composite_evidence=adjudicator_composite_evidence,
    )
    input_fingerprint = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-pairwise-v4-authority-input/v1",
            "release": checked,
            "reviewer_packet": reviewer_packet,
            "trusted_mapping": trusted_mapping,
            "predecessor_closure": predecessor_closure,
            "primary_composite_evidence": primary_composite_evidence,
            "adjudicator_composite_evidence": adjudicator_composite_evidence,
        }
    )
    return _mint(
        shape=BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4,
        release_ref=checked.ref,
        release_fingerprint=checked.fingerprint,
        verification_input_fingerprint=input_fingerprint,
    )


__all__ = [
    "BfclV4PublicV2SemanticAuthorityError",
    "BfclV4PublicV2VerifiedSemanticReleaseCapability",
    "validate_bfcl_v4_public_v2_verified_semantic_release_capability",
    "verify_bfcl_v4_public_v2_legacy_semantic_authority",
    "verify_bfcl_v4_public_v2_pairwise_v4_semantic_authority",
]
