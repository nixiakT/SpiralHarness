from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import MutationPolicy
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.harness.registry import HarnessRegistry, HarnessRegistryError


def artifact_for(payload: bytes, *, media_type: str = "text/plain") -> ArtifactRef:
    return ArtifactRef(
        sha256=sha256_bytes(payload),
        size=len(payload),
        media_type=media_type,
    )


def component(
    name: str,
    kind: ComponentKind,
    payload: bytes,
    *,
    media_type: str = "text/plain",
) -> HarnessComponentRef:
    return HarnessComponentRef(
        name=name,
        kind=kind,
        artifact=artifact_for(payload, media_type=media_type),
    )


def hypothesis() -> MutationHypothesis:
    return MutationHypothesis(
        evidence_refs=(artifact_for(b"evidence", media_type="application/json"),),
        where="system prompt",
        why="verification instructions are absent",
        expected_activation="the candidate prompt is injected",
        expected_adherence="the verification step executes",
        expected_behavior="the final response changes after a failed check",
        expected_benefit="paired benchmark score improves",
        protected_slices=("already-correct",),
        falsifier="activation occurs without the expected behavior",
        negative_control="an equal-sized inert prompt edit",
        risks=("token cost",),
    )


def mutation(
    before: HarnessComponentRef,
    after: HarnessComponentRef,
) -> CandidateMutation:
    return CandidateMutation(
        target_component=before.name,
        before=before,
        after=after,
        hypothesis=hypothesis(),
    )


def parent_manifest() -> HarnessManifest:
    return HarnessManifest(
        model_fingerprint="model-fixed",
        runtime_fingerprint="runtime-fixed",
        trusted_plane_version="trusted-fixed",
        components=(
            component("system", ComponentKind.PROMPT, b"seed prompt"),
            component("retrieval", ComponentKind.SKILL, b"seed skill"),
        ),
        budget=BudgetPolicy(max_tokens=2_000, max_evaluations=16),
    )


def manifest_ref(manifest: HarnessManifest) -> ArtifactRef:
    payload = canonical_json_bytes(manifest)
    return artifact_for(
        payload,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )


def by_name(manifest: HarnessManifest) -> dict[str, HarnessComponentRef]:
    return {item.name: item for item in manifest.components}


def test_registry_applies_one_verified_mutation_without_touching_frozen_fields() -> None:
    parent = parent_manifest()
    parent_components = by_name(parent)
    payload = b"candidate prompt"
    candidate = mutation(
        parent_components["system"],
        component("system", ComponentKind.PROMPT, payload),
    )
    registry = HarnessRegistry(
        MutationPolicy(
            allowed_component_names=("system",),
            allowed_media_types=("text/plain",),
            max_artifact_size_bytes=64,
        )
    )

    child = registry.apply_mutation(
        parent=parent,
        parent_ref=manifest_ref(parent),
        mutation=candidate,
        artifact_bytes=payload,
        artifact_media_type="text/plain",
    )

    child_components = by_name(child)
    assert child is not parent
    assert child.parent == manifest_ref(parent)
    assert child_components["system"] == candidate.after
    assert child_components["retrieval"] == parent_components["retrieval"]
    assert child_components["retrieval"] is parent_components["retrieval"]
    assert by_name(parent)["system"] == candidate.before
    assert child.model_fingerprint == parent.model_fingerprint
    assert child.runtime_fingerprint == parent.runtime_fingerprint
    assert child.trusted_plane_version == parent.trusted_plane_version
    assert child.budget == parent.budget


def test_default_policy_accepts_plain_text_with_declared_charset() -> None:
    parent = parent_manifest()
    before = by_name(parent)["system"]
    payload = b"candidate prompt"
    candidate = mutation(
        before,
        component(
            "system",
            ComponentKind.PROMPT,
            payload,
            media_type="text/plain; charset=utf-8",
        ),
    )

    child = HarnessRegistry().apply_mutation(
        parent=parent,
        parent_ref=manifest_ref(parent),
        mutation=candidate,
        artifact_bytes=payload,
        artifact_media_type="text/plain; charset=utf-8",
    )

    assert by_name(child)["system"] == candidate.after


def test_registry_requires_exact_actual_before_and_existing_target() -> None:
    parent = parent_manifest()
    payload = b"candidate prompt"
    wrong_before = component("system", ComponentKind.PROMPT, b"not the actual parent")
    wrong_candidate = mutation(
        wrong_before,
        component("system", ComponentKind.PROMPT, payload),
    )
    registry = HarnessRegistry()

    with pytest.raises(HarnessRegistryError, match="exactly match"):
        registry.apply_mutation(
            parent=parent,
            parent_ref=manifest_ref(parent),
            mutation=wrong_candidate,
            artifact_bytes=payload,
            artifact_media_type="text/plain",
        )

    missing_before = component("missing", ComponentKind.PROMPT, b"old")
    missing_candidate = mutation(
        missing_before,
        component("missing", ComponentKind.PROMPT, payload),
    )
    with pytest.raises(HarnessRegistryError, match="absent from parent"):
        registry.apply_mutation(
            parent=parent,
            parent_ref=manifest_ref(parent),
            mutation=missing_candidate,
            artifact_bytes=payload,
            artifact_media_type="text/plain",
        )


def test_registry_enforces_kind_and_name_allowlists() -> None:
    parent = parent_manifest()
    components = by_name(parent)
    skill_payload = b"candidate skill"
    skill_mutation = mutation(
        components["retrieval"],
        component("retrieval", ComponentKind.SKILL, skill_payload),
    )

    with pytest.raises(HarnessRegistryError, match="component kind"):
        HarnessRegistry().apply_mutation(
            parent=parent,
            parent_ref=manifest_ref(parent),
            mutation=skill_mutation,
            artifact_bytes=skill_payload,
            artifact_media_type="text/plain",
        )

    prompt_payload = b"candidate prompt"
    prompt_mutation = mutation(
        components["system"],
        component("system", ComponentKind.PROMPT, prompt_payload),
    )
    with pytest.raises(HarnessRegistryError, match="component name"):
        HarnessRegistry(MutationPolicy(allowed_component_names=("other",))).apply_mutation(
            parent=parent,
            parent_ref=manifest_ref(parent),
            mutation=prompt_mutation,
            artifact_bytes=prompt_payload,
            artifact_media_type="text/plain",
        )


def test_registry_verifies_actual_artifact_bytes_media_type_and_size_policy() -> None:
    parent = parent_manifest()
    before = by_name(parent)["system"]
    payload = b"candidate"
    candidate = mutation(
        before,
        component("system", ComponentKind.PROMPT, payload),
    )

    cases = [
        (
            HarnessRegistry(),
            b"tampered!",
            "text/plain",
            "hash mismatch",
        ),
        (
            HarnessRegistry(),
            b"short",
            "text/plain",
            "size mismatch",
        ),
        (
            HarnessRegistry(),
            payload,
            "text/markdown",
            "media type does not match",
        ),
        (
            HarnessRegistry(MutationPolicy(allowed_media_types=("application/json",))),
            payload,
            "text/plain",
            "not allowed by mutation policy",
        ),
        (
            HarnessRegistry(MutationPolicy(max_artifact_size_bytes=len(payload) - 1)),
            payload,
            "text/plain",
            "size ceiling",
        ),
    ]
    for registry, actual_payload, media_type, error in cases:
        with pytest.raises(HarnessRegistryError, match=error):
            registry.apply_mutation(
                parent=parent,
                parent_ref=manifest_ref(parent),
                mutation=candidate,
                artifact_bytes=actual_payload,
                artifact_media_type=media_type,
            )


def test_registry_verifies_parent_reference_content_and_exact_media_type() -> None:
    parent = parent_manifest()
    payload = b"candidate prompt"
    candidate = mutation(
        by_name(parent)["system"],
        component("system", ComponentKind.PROMPT, payload),
    )
    valid_ref = manifest_ref(parent)
    invalid_refs = [
        (
            ArtifactRef(
                sha256="f" * 64,
                size=valid_ref.size,
                media_type=valid_ref.media_type,
            ),
            "hash mismatch",
        ),
        (
            ArtifactRef(
                sha256=valid_ref.sha256,
                size=valid_ref.size + 1,
                media_type=valid_ref.media_type,
            ),
            "size mismatch",
        ),
        (
            ArtifactRef(
                sha256=valid_ref.sha256,
                size=valid_ref.size,
                media_type="application/json",
            ),
            "exact harness manifest v2 media type",
        ),
    ]

    for invalid_ref, error in invalid_refs:
        with pytest.raises(HarnessRegistryError, match=error):
            HarnessRegistry().apply_mutation(
                parent=parent,
                parent_ref=invalid_ref,
                mutation=candidate,
                artifact_bytes=payload,
                artifact_media_type="text/plain",
            )


def test_registry_revalidates_bypassed_model_instances() -> None:
    parent = parent_manifest()
    invalid_parent = parent.model_copy(update={"model_fingerprint": ""})
    payload = b"candidate prompt"
    candidate = mutation(
        by_name(parent)["system"],
        component("system", ComponentKind.PROMPT, payload),
    )

    with pytest.raises(ValidationError, match="at least 1 character"):
        HarnessRegistry().apply_mutation(
            parent=invalid_parent,
            parent_ref=manifest_ref(parent),
            mutation=candidate,
            artifact_bytes=payload,
            artifact_media_type="text/plain",
        )


@pytest.mark.parametrize(
    ("artifact_bytes", "media_type", "error"),
    [
        ("not bytes", "text/plain", "bytes-like"),
        (b"candidate prompt", "", "non-empty string"),
        (b"candidate prompt", 1, "non-empty string"),
    ],
)
def test_registry_rejects_invalid_actual_artifact_inputs(
    artifact_bytes: object,
    media_type: object,
    error: str,
) -> None:
    parent = parent_manifest()
    payload = b"candidate prompt"
    candidate = mutation(
        by_name(parent)["system"],
        component("system", ComponentKind.PROMPT, payload),
    )

    with pytest.raises(TypeError, match=error):
        HarnessRegistry().apply_mutation(
            parent=parent,
            parent_ref=manifest_ref(parent),
            mutation=candidate,
            artifact_bytes=artifact_bytes,  # type: ignore[arg-type]
            artifact_media_type=media_type,  # type: ignore[arg-type]
        )
