from __future__ import annotations

import copy
import json
import pickle

import pytest

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority import (
    BfclV4PublicV2SemanticAuthorityError,
    BfclV4PublicV2VerifiedSemanticReleaseCapability,
    validate_bfcl_v4_public_v2_verified_semantic_release_capability,
    verify_bfcl_v4_public_v2_legacy_semantic_authority,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority_contracts import (
    BfclV4PublicV2SemanticReleaseEvidenceShape,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticReleaseError,
)


def _verify(bundle):
    return verify_bfcl_v4_public_v2_legacy_semantic_authority(
        bundle.release,
        bundle.reviewer_packet,
        bundle.trusted_mapping,
        bundle.primary_sessions,
        runtime_hmac_secret=bundle.runtime_hmac_secret,
    )


def test_real_legacy_hmac_verification_mints_one_process_local_capability(
    bfcl_v2_verified_legacy_authority,
) -> None:
    bundle = bfcl_v2_verified_legacy_authority
    capability = bundle.capability

    assert validate_bfcl_v4_public_v2_verified_semantic_release_capability(capability) is capability
    assert (
        capability.evidence_shape
        is BfclV4PublicV2SemanticReleaseEvidenceShape.LEGACY_SINGLE_PACKET_V1
    )
    assert capability.release_ref == bundle.release.ref
    assert capability.release_fingerprint == bundle.release.fingerprint
    assert len(capability.verification_input_fingerprint) == 64
    assert not hasattr(capability, "runtime_hmac_secret")
    with pytest.raises(TypeError):
        vars(capability)


def test_capability_cannot_be_constructed_copied_or_serialized(
    bfcl_v2_verified_legacy_authority,
) -> None:
    capability = bfcl_v2_verified_legacy_authority.capability

    with pytest.raises(BfclV4PublicV2SemanticAuthorityError, match="only be minted"):
        BfclV4PublicV2VerifiedSemanticReleaseCapability()
    for operation in (
        lambda: copy.copy(capability),
        lambda: copy.deepcopy(capability),
        lambda: pickle.dumps(capability),
        lambda: json.dumps(capability),
    ):
        with pytest.raises(TypeError):
            operation()

    reconstructed = object.__new__(BfclV4PublicV2VerifiedSemanticReleaseCapability)
    with pytest.raises(BfclV4PublicV2SemanticAuthorityError, match="not minted"):
        validate_bfcl_v4_public_v2_verified_semantic_release_capability(reconstructed)


@pytest.mark.parametrize("field", ("release_authority_hmac_sha256", "final_partition_sha256"))
def test_forged_or_model_copied_legacy_release_never_mints_authority(
    bfcl_v2_verified_legacy_authority,
    field: str,
) -> None:
    bundle = bfcl_v2_verified_legacy_authority
    forged = bundle.release.model_copy(update={field: "0" * 64})

    with pytest.raises(BfclV4PublicV2SemanticReleaseError):
        verify_bfcl_v4_public_v2_legacy_semantic_authority(
            forged,
            bundle.reviewer_packet,
            bundle.trusted_mapping,
            bundle.primary_sessions,
            runtime_hmac_secret=bundle.runtime_hmac_secret,
        )


def test_wrong_secret_and_post_mint_mutation_fail_closed(
    bfcl_v2_verified_legacy_authority,
) -> None:
    bundle = bfcl_v2_verified_legacy_authority
    with pytest.raises(BfclV4PublicV2SemanticReleaseError):
        verify_bfcl_v4_public_v2_legacy_semantic_authority(
            bundle.release,
            bundle.reviewer_packet,
            bundle.trusted_mapping,
            bundle.primary_sessions,
            runtime_hmac_secret=b"wrong-secret-that-is-long-enough" * 2,
        )

    capability = _verify(bundle)
    object.__setattr__(capability, "_release_fingerprint", "f" * 64)
    with pytest.raises(BfclV4PublicV2SemanticAuthorityError, match="not minted"):
        validate_bfcl_v4_public_v2_verified_semantic_release_capability(capability)


def test_post_mint_shape_or_process_mutation_fail_closed(
    bfcl_v2_verified_legacy_authority,
    monkeypatch,
) -> None:
    capability = _verify(bfcl_v2_verified_legacy_authority)
    object.__setattr__(capability, "_evidence_shape", "unsupported-v5")
    with pytest.raises(BfclV4PublicV2SemanticAuthorityError, match="not minted"):
        validate_bfcl_v4_public_v2_verified_semantic_release_capability(capability)

    capability = _verify(bfcl_v2_verified_legacy_authority)
    current_pid = capability._process_id
    monkeypatch.setattr(
        "spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority.os.getpid",
        lambda: current_pid + 1,
    )
    with pytest.raises(BfclV4PublicV2SemanticAuthorityError, match="not minted"):
        validate_bfcl_v4_public_v2_verified_semantic_release_capability(capability)
