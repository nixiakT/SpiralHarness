"""Canonical digest binding for caller-declared power-pilot metadata.

This module verifies only manifest syntax, canonical encoding, and equality of
declared digests.  It does not load referenced artifacts or establish their
existence, authenticity, pilot observation, disjointness, closure, or blinding.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.power_joint import PRIMARY_ESTIMATOR_ORDER


class PilotEndpointAssumptionRef(ImmutableModel):
    hypothesis: NonEmptyStr
    artifact_ref: ArtifactRef


class DeclaredPowerPilotManifest(ImmutableModel):
    """Caller-declared pilot metadata; none of its empirical claims are verified."""

    schema_version: Literal["1"] = "1"
    study_id: NonEmptyStr
    source_kind: Literal["declared-power-pilot-metadata"]
    declared_synthetic: Literal[False]
    declared_partition: Literal["disjoint-pilot"]
    partition_ref: ArtifactRef
    closure_ref: ArtifactRef
    declared_closure_status: Literal["closed-before-sensitivity-run"]
    code_revision_sha256: Sha256
    endpoint_assumption_refs: Annotated[
        tuple[PilotEndpointAssumptionRef, ...], Field(min_length=3, max_length=3)
    ]
    joint_dependence_ref: ArtifactRef
    declared_main_search_outcomes_visible: Literal[False]

    @model_validator(mode="after")
    def validate_distinct_closure_and_primary_order(self) -> DeclaredPowerPilotManifest:
        lowered_study_id = self.study_id.casefold()
        if any(marker in lowered_study_id for marker in ("synthetic", "fixture", "unit-test")):
            raise ValueError("declared pilot manifest study_id contains a synthetic marker")
        if self.partition_ref.sha256 == self.closure_ref.sha256:
            raise ValueError("pilot partition_ref and closure_ref must be distinct")
        order = tuple(item.hypothesis for item in self.endpoint_assumption_refs)
        if order != PRIMARY_ESTIMATOR_ORDER:
            raise ValueError("pilot endpoint assumption order does not match the primary family")
        hashes = [item.artifact_ref.sha256 for item in self.endpoint_assumption_refs]
        if len(hashes) != len(set(hashes)):
            raise ValueError("pilot endpoint assumption artifacts must be distinct")
        return self


class DigestBoundPowerPilotManifest(ImmutableModel):
    """Canonical declaration digest, without evidence-authenticity verification."""

    manifest_sha256: Sha256
    partition_ref: ArtifactRef
    closure_ref: ArtifactRef
    code_revision_sha256: Sha256
    binding: Literal["canonical-declared-manifest-digest-bound"]
    referenced_artifacts_loaded: Literal[False]
    artifact_existence_verified: Literal[False]
    artifact_content_digests_verified: Literal[False]
    manifest_authenticity_verified: Literal[False]
    observed_pilot_status_verified: Literal[False]
    pilot_disjointness_verified: Literal[False]
    pilot_closure_verified: Literal[False]
    main_search_blinding_verified: Literal[False]


class DeclaredPowerPilotManifestExpectation(ImmutableModel):
    """Config-side commitment to a declared manifest and reference digests."""

    manifest_sha256: Sha256
    partition_ref_sha256: Sha256
    closure_ref_sha256: Sha256
    code_revision_sha256: Sha256


def validate_digest_bound_report_claim(
    *,
    embedded_config_sha256: str,
    actual_config_sha256: str,
    fixture: bool,
    expectation: DeclaredPowerPilotManifestExpectation | None,
    binding: DigestBoundPowerPilotManifest | None,
    digest_bound_claimed: bool,
) -> None:
    """Reject contradictory digest-binding metadata in a serialized proxy report."""

    if embedded_config_sha256 != actual_config_sha256:
        raise ValueError("config_sha256 does not match the embedded sensitivity config")
    has_binding = binding is not None
    if digest_bound_claimed != has_binding:
        raise ValueError("declared pilot digest-bound flag does not match the binding")
    if (expectation is not None) != has_binding:
        raise ValueError("declared pilot expectation and digest binding must appear together")
    if fixture and has_binding:
        raise ValueError("a deterministic test fixture cannot contain a pilot digest binding")
    if binding is None or expectation is None:
        return
    comparisons = (
        ("manifest", binding.manifest_sha256, expectation.manifest_sha256),
        ("partition", binding.partition_ref.sha256, expectation.partition_ref_sha256),
        ("closure", binding.closure_ref.sha256, expectation.closure_ref_sha256),
        ("code revision", binding.code_revision_sha256, expectation.code_revision_sha256),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"pilot {label} digest binding does not match the config")


def bind_declared_power_pilot_manifest(
    payload: bytes,
    *,
    expectation: DeclaredPowerPilotManifestExpectation,
    expected_study_id: str,
    expected_endpoint_assumption_hashes: tuple[str, str, str],
    expected_joint_assumption_hash: str,
) -> tuple[DeclaredPowerPilotManifest, DigestBoundPowerPilotManifest]:
    """Bind canonical caller declarations without authenticating referenced evidence."""

    manifest = DeclaredPowerPilotManifest.model_validate_json(payload, strict=True)
    canonical = canonical_json_bytes(manifest)
    if payload != canonical:
        raise ValueError("declared pilot manifest must use canonical JSON encoding")
    digest = sha256_bytes(canonical)
    if digest != expectation.manifest_sha256:
        raise ValueError("declared pilot manifest digest does not match the sensitivity config")
    if manifest.study_id != expected_study_id:
        raise ValueError("declared pilot manifest study_id does not match the sensitivity config")
    actual_endpoint_hashes = tuple(
        item.artifact_ref.sha256 for item in manifest.endpoint_assumption_refs
    )
    if actual_endpoint_hashes != expected_endpoint_assumption_hashes:
        raise ValueError("declared pilot endpoint assumption refs do not match the config")
    if manifest.joint_dependence_ref.sha256 != expected_joint_assumption_hash:
        raise ValueError("declared pilot joint dependence ref does not match the config")
    if manifest.partition_ref.sha256 != expectation.partition_ref_sha256:
        raise ValueError("declared pilot partition_ref does not match the sensitivity config")
    if manifest.closure_ref.sha256 != expectation.closure_ref_sha256:
        raise ValueError("declared pilot closure_ref does not match the sensitivity config")
    if manifest.code_revision_sha256 != expectation.code_revision_sha256:
        raise ValueError("declared pilot code revision does not match the sensitivity config")
    binding = DigestBoundPowerPilotManifest(
        manifest_sha256=digest,
        partition_ref=manifest.partition_ref,
        closure_ref=manifest.closure_ref,
        code_revision_sha256=manifest.code_revision_sha256,
        binding="canonical-declared-manifest-digest-bound",
        referenced_artifacts_loaded=False,
        artifact_existence_verified=False,
        artifact_content_digests_verified=False,
        manifest_authenticity_verified=False,
        observed_pilot_status_verified=False,
        pilot_disjointness_verified=False,
        pilot_closure_verified=False,
        main_search_blinding_verified=False,
    )
    return manifest, binding


def bind_optional_declared_power_pilot_manifest(
    payload: bytes | None,
    *,
    expectation: DeclaredPowerPilotManifestExpectation | None,
    fixture: bool,
    expected_study_id: str,
    expected_endpoint_assumption_hashes: tuple[str, str, str],
    expected_joint_assumption_hash: str,
) -> DigestBoundPowerPilotManifest | None:
    """Require raw manifest bytes exactly when the config declares an expectation."""

    if expectation is None and payload is not None:
        raise ValueError("declared pilot manifest bytes require a config-side expectation")
    if expectation is not None and payload is None:
        raise ValueError("the config-bound declared pilot manifest bytes are required")
    if payload is None:
        return None
    if fixture:
        raise ValueError("a deterministic test fixture cannot bind declared pilot metadata")
    assert expectation is not None
    return bind_declared_power_pilot_manifest(
        payload,
        expectation=expectation,
        expected_study_id=expected_study_id,
        expected_endpoint_assumption_hashes=expected_endpoint_assumption_hashes,
        expected_joint_assumption_hash=expected_joint_assumption_hash,
    )[1]


__all__ = [
    "DeclaredPowerPilotManifest",
    "DeclaredPowerPilotManifestExpectation",
    "DigestBoundPowerPilotManifest",
    "PilotEndpointAssumptionRef",
    "bind_declared_power_pilot_manifest",
    "bind_optional_declared_power_pilot_manifest",
    "validate_digest_bound_report_claim",
]
