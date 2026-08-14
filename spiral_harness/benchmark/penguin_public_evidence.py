"""Typed provenance closure for Penguin's public recursive demonstration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field

from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessManifest,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.storage.protocol import ArtifactRepository

PENGUIN_PUBLIC_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.penguin-public-self-evolution.v1+json"
)
PENGUIN_PROTOCOL_BINDING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.penguin-public-protocol-binding.v1+json"
)
PENGUIN_TRUSTED_PLANE_VERSION = "spiral-harness/penguin-public-custom-executor/v2"
PENGUIN_RUNNER_IMPLEMENTATION = "spiral-harness/penguin-public-runner/v2"


class PenguinProtocolBinding(ImmutableModel):
    """Exact protocol and local implementation bound into one harness manifest."""

    schema_version: Literal["1"] = "1"
    kind: Literal["penguin-public-harness-protocol-binding"] = (
        "penguin-public-harness-protocol-binding"
    )
    role: Literal["worker", "reflection"]
    spec_fingerprint: Sha256
    protocol_sha256: Sha256
    runs_per_generation: Annotated[int, Field(ge=1, le=5, strict=True)]
    upstream_revision: NonEmptyStr
    upstream_source_sha256: Sha256
    local_implementation_source_sha256: Sha256
    runner_implementation: Literal["spiral-harness/penguin-public-runner/v2"] = (
        PENGUIN_RUNNER_IMPLEMENTATION
    )


class PenguinEvidenceVerificationError(ValueError):
    """Raised when an offline Penguin result cannot close its provenance graph."""


def verify_penguin_public_result(
    repository: ArtifactRepository,
    result_ref: ArtifactRef,
) -> dict[str, object]:
    """Verify result-to-call-to-manifest-to-component closure without live state."""

    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    checked_ref = ArtifactRef.model_validate(result_ref, strict=True)
    if checked_ref.media_type != PENGUIN_PUBLIC_RESULT_MEDIA_TYPE:
        raise PenguinEvidenceVerificationError("result declares the wrong media type")
    payload = _mapping(repository.get_json(checked_ref), "result")

    runs = payload.get("runs_per_generation")
    if type(runs) is not int or not 1 <= runs <= 5:
        raise PenguinEvidenceVerificationError("runs_per_generation must be from 1 through 5")
    canonical = runs == 5
    expected_class = "canonical" if canonical else "noncanonical_exploratory"
    if payload.get("protocol_class") != expected_class:
        raise PenguinEvidenceVerificationError("protocol class contradicts the run count")
    if payload.get("reportable_as_canonical") is not canonical:
        raise PenguinEvidenceVerificationError("canonical reportability contradicts the run count")
    if payload.get("official_15_40_suite_public") is not False:
        raise PenguinEvidenceVerificationError("public-suite availability must remain false")

    protocol_sha256 = _sha(payload.get("protocol_sha256"), "protocol_sha256")
    spec_fingerprint = _sha(payload.get("spec_fingerprint"), "spec_fingerprint")
    model_fingerprint = _sha(payload.get("model_fingerprint"), "model_fingerprint")
    runtime_fingerprint = _sha(payload.get("runtime_fingerprint"), "runtime_fingerprint")
    local_source_sha256 = _sha(
        payload.get("local_implementation_source_sha256"),
        "local_implementation_source_sha256",
    )

    role_refs = {
        "worker": _artifact_ref(payload.get("worker_harness_ref"), "worker_harness_ref"),
        "reflection": _artifact_ref(
            payload.get("reflection_harness_ref"),
            "reflection_harness_ref",
        ),
    }
    if role_refs["worker"] == role_refs["reflection"]:
        raise PenguinEvidenceVerificationError("worker and reflection harnesses must differ")

    resolved_prompt_hashes: dict[str, str] = {}
    for role, harness_ref in role_refs.items():
        if harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise PenguinEvidenceVerificationError(f"{role} harness is not a manifest v2")
        manifest = repository.get_json(harness_ref, HarnessManifest)
        if manifest.model_fingerprint != model_fingerprint:
            raise PenguinEvidenceVerificationError(f"{role} manifest model fingerprint drifted")
        if manifest.runtime_fingerprint != runtime_fingerprint:
            raise PenguinEvidenceVerificationError(f"{role} manifest runtime fingerprint drifted")
        if manifest.trusted_plane_version != PENGUIN_TRUSTED_PLANE_VERSION:
            raise PenguinEvidenceVerificationError(f"{role} manifest trusted plane drifted")

        components = {component.name: component for component in manifest.components}
        if set(components) != {"penguin_protocol", "system_prompt"}:
            raise PenguinEvidenceVerificationError(f"{role} manifest component set is invalid")
        prompt_component = components["system_prompt"]
        protocol_component = components["penguin_protocol"]
        if prompt_component.kind is not ComponentKind.PROMPT:
            raise PenguinEvidenceVerificationError(f"{role} prompt component has the wrong kind")
        if protocol_component.kind is not ComponentKind.CONTROL_FLOW:
            raise PenguinEvidenceVerificationError(f"{role} protocol component has the wrong kind")
        prompt = repository.get_bytes(prompt_component.artifact)
        resolved_prompt_hashes[harness_ref.sha256] = sha256_bytes(prompt)

        if protocol_component.artifact.media_type != PENGUIN_PROTOCOL_BINDING_MEDIA_TYPE:
            raise PenguinEvidenceVerificationError(f"{role} protocol binding has the wrong type")
        binding = repository.get_json(protocol_component.artifact, PenguinProtocolBinding)
        if binding.role != role:
            raise PenguinEvidenceVerificationError(f"{role} protocol binding role drifted")
        if (
            binding.spec_fingerprint != spec_fingerprint
            or binding.protocol_sha256 != protocol_sha256
            or binding.runs_per_generation != runs
            or binding.local_implementation_source_sha256 != local_source_sha256
        ):
            raise PenguinEvidenceVerificationError(f"{role} protocol binding drifted")

    calls = payload.get("calls")
    if not isinstance(calls, list) or len(calls) != 3 * runs + 2:
        raise PenguinEvidenceVerificationError("call schedule is incomplete")
    allowed_refs = {ref.sha256: ref for ref in role_refs.values()}
    for index, raw_call in enumerate(calls):
        call = _mapping(raw_call, f"calls[{index}]")
        call_ref = _artifact_ref(call.get("harness_ref"), f"calls[{index}].harness_ref")
        if allowed_refs.get(call_ref.sha256) != call_ref:
            raise PenguinEvidenceVerificationError(f"calls[{index}] cites an unknown harness")
        if call.get("resolved_prompt_sha256") != resolved_prompt_hashes[call_ref.sha256]:
            raise PenguinEvidenceVerificationError(
                f"calls[{index}] prompt hash is not manifest-bound"
            )
        task_id = call.get("task_id")
        if not isinstance(task_id, str):
            raise PenguinEvidenceVerificationError(f"calls[{index}] has no task id")
        expected_ref = role_refs["reflection"] if "/reflect/" in task_id else role_refs["worker"]
        if call_ref != expected_ref:
            raise PenguinEvidenceVerificationError(f"calls[{index}] uses the wrong harness role")
    return dict(payload)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PenguinEvidenceVerificationError(f"{label} must be an object")
    return value


def _artifact_ref(value: object, label: str) -> ArtifactRef:
    try:
        return ArtifactRef.model_validate(value, strict=True)
    except Exception as exc:
        raise PenguinEvidenceVerificationError(f"{label} is malformed") from exc


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PenguinEvidenceVerificationError(f"{label} is malformed")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PenguinEvidenceVerificationError(f"{label} is malformed") from exc
    if value != value.lower():
        raise PenguinEvidenceVerificationError(f"{label} is malformed")
    return value


__all__ = [
    "PENGUIN_PROTOCOL_BINDING_MEDIA_TYPE",
    "PENGUIN_PUBLIC_RESULT_MEDIA_TYPE",
    "PENGUIN_RUNNER_IMPLEMENTATION",
    "PENGUIN_TRUSTED_PLANE_VERSION",
    "PenguinEvidenceVerificationError",
    "PenguinProtocolBinding",
    "verify_penguin_public_result",
]
