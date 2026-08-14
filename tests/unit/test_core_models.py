from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import (
    callable_source_sha256,
    canonical_json_bytes,
    canonical_sha256,
    module_source_sha256,
    sha256_bytes,
)
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


def artifact(digit: str, *, size: int = 1, media_type: str = "text/plain") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=size, media_type=media_type)


def component(name: str, kind: ComponentKind, digit: str) -> HarnessComponentRef:
    return HarnessComponentRef(name=name, kind=kind, artifact=artifact(digit))


def source_fingerprint_fixture(value: int) -> int:
    return value + 1


def hypothesis() -> MutationHypothesis:
    return MutationHypothesis(
        evidence_refs=(artifact("e"),),
        where="the planning prompt",
        why="it omits an explicit verification step",
        expected_activation="the prompt is injected on every targeted task",
        expected_adherence="the agent performs the requested check",
        expected_behavior="unsupported answers are revised before submission",
        expected_benefit="paired correctness improves",
        protected_slices=("already-correct", "low-latency"),
        falsifier="activation rises but verification behavior does not",
        negative_control="inject an inert comment of equal length",
        risks=("extra tokens",),
    )


def test_component_kind_covers_the_full_harness_surface() -> None:
    assert {kind.value for kind in ComponentKind} == {
        "prompt",
        "skill",
        "memory",
        "tool",
        "middleware",
        "control_flow",
    }


def test_artifact_ref_is_strict_validated_and_frozen() -> None:
    ref = artifact("a")

    with pytest.raises(ValidationError):
        ArtifactRef(sha256="A" * 64, size=1, media_type="text/plain")
    with pytest.raises(ValidationError):
        ArtifactRef(sha256="a" * 64, size=-1, media_type="text/plain")
    with pytest.raises(ValidationError):
        ArtifactRef(sha256="a" * 64, size="1", media_type="text/plain")
    with pytest.raises((ValidationError, FrozenInstanceError)):
        ref.size = 2


def test_immutable_models_reject_python_coercion_but_accept_json_forms() -> None:
    prompt = component("system", ComponentKind.PROMPT, "a")
    manifest_values = {
        "model_fingerprint": "model",
        "runtime_fingerprint": "runtime",
        "trusted_plane_version": "trusted-plane-v1",
    }

    with pytest.raises(ValidationError):
        ArtifactRef(sha256="a" * 64, size=1, media_type=b"text/plain")
    with pytest.raises(ValidationError):
        HarnessComponentRef(name="system", kind="prompt", artifact=artifact("a"))
    with pytest.raises(ValidationError):
        HarnessManifest(**manifest_values, components=[prompt])

    manifest = HarnessManifest(**manifest_values, components=(prompt,))
    assert HarnessManifest.model_validate_json(canonical_json_bytes(manifest)) == manifest


def test_budget_requires_a_finite_nonnegative_ceiling() -> None:
    assert BudgetPolicy(max_tokens=0).max_tokens == 0
    with pytest.raises(ValidationError, match="at least one"):
        BudgetPolicy()
    with pytest.raises(ValidationError):
        BudgetPolicy(max_tokens=-1)
    with pytest.raises(ValidationError):
        BudgetPolicy(max_cost_usd=float("nan"))


def test_manifest_rejects_duplicate_component_names() -> None:
    first = component("system", ComponentKind.PROMPT, "a")
    duplicate = component("system", ComponentKind.SKILL, "b")

    with pytest.raises(ValidationError, match="component names must be unique"):
        HarnessManifest(
            model_fingerprint="model@sha256:123",
            runtime_fingerprint="container@sha256:456",
            trusted_plane_version="gate-v1",
            components=(first, duplicate),
        )


def test_manifest_records_lineage_and_is_deeply_immutable() -> None:
    parent = artifact("f", media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    manifest = HarnessManifest(
        model_fingerprint="fixed-model/settings/seed",
        runtime_fingerprint="runtime-image-digest",
        trusted_plane_version="verification-v1",
        parent_ref=parent,
        components=(component("system", ComponentKind.PROMPT, "a"),),
        budget=BudgetPolicy(max_tokens=10_000, max_evaluations=32),
    )

    # Nested immutable values are revalidated rather than trusted by identity.
    assert manifest.parent == parent
    assert manifest.parent_ref == parent
    assert isinstance(manifest.components, tuple)
    with pytest.raises(ValidationError):
        manifest.components[0].name = "changed"


@pytest.mark.parametrize(
    "media_type",
    [
        "application/json",
        "application/vnd.spiral-harness.manifest+json",
        f"{HARNESS_MANIFEST_MEDIA_TYPE}; charset=utf-8",
    ],
)
def test_manifest_parent_requires_the_exact_v1_media_type(media_type: str) -> None:
    with pytest.raises(ValidationError, match="exact harness manifest v2 media type"):
        HarnessManifest(
            model_fingerprint="fixed-model/settings/seed",
            runtime_fingerprint="runtime-image-digest",
            trusted_plane_version="verification-v1",
            parent=artifact("f", media_type=media_type),
            components=(component("system", ComponentKind.PROMPT, "a"),),
        )


def test_candidate_mutation_is_atomic_and_non_noop() -> None:
    before = component("system", ComponentKind.PROMPT, "a")
    after = component("system", ComponentKind.PROMPT, "b")
    candidate = CandidateMutation(
        target_component="system",
        before=before,
        after=after,
        hypothesis=hypothesis(),
    )

    assert candidate.target == "system"
    with pytest.raises(ValidationError, match="no-op"):
        CandidateMutation(
            target_component="system",
            before=before,
            after=before,
            hypothesis=hypothesis(),
        )
    with pytest.raises(ValidationError, match="same component kind"):
        CandidateMutation(
            target_component="system",
            before=before,
            after=component("system", ComponentKind.SKILL, "b"),
            hypothesis=hypothesis(),
        )
    with pytest.raises(ValidationError, match="target_component"):
        CandidateMutation(
            target_component="other",
            before=before,
            after=after,
            hypothesis=hypothesis(),
        )


def test_hypothesis_requires_evidence_and_rejects_duplicate_evidence() -> None:
    values = hypothesis().model_dump()
    values["evidence_refs"] = ()
    with pytest.raises(ValidationError):
        MutationHypothesis.model_validate(values)

    values["evidence_refs"] = (artifact("e"), artifact("e"))
    with pytest.raises(ValidationError, match="duplicate"):
        MutationHypothesis.model_validate(values)

    for field_name in ("protected_slices", "risks"):
        values = hypothesis().model_dump()
        values[field_name] = ("duplicate", "duplicate")
        with pytest.raises(ValidationError, match="duplicate"):
            MutationHypothesis.model_validate(values)


def test_set_like_manifest_and_hypothesis_fields_have_canonical_order() -> None:
    system = component("system", ComponentKind.PROMPT, "a")
    retrieval = component("retrieval", ComponentKind.SKILL, "b")
    manifest_values = {
        "model_fingerprint": "model",
        "runtime_fingerprint": "runtime",
        "trusted_plane_version": "gate-v1",
    }
    left_manifest = HarnessManifest(**manifest_values, components=(system, retrieval))
    right_manifest = HarnessManifest(**manifest_values, components=(retrieval, system))

    assert [item.name for item in left_manifest.components] == ["retrieval", "system"]
    assert left_manifest == right_manifest
    assert canonical_sha256(left_manifest) == canonical_sha256(right_manifest)

    left_values = hypothesis().model_dump()
    left_values.update(
        evidence_refs=(artifact("e"), artifact("d")),
        protected_slices=("latency", "accuracy"),
        risks=("cost", "latency"),
    )
    right_values = {
        **left_values,
        "evidence_refs": tuple(reversed(left_values["evidence_refs"])),
        "protected_slices": tuple(reversed(left_values["protected_slices"])),
        "risks": tuple(reversed(left_values["risks"])),
    }
    left_hypothesis = MutationHypothesis.model_validate(left_values)
    right_hypothesis = MutationHypothesis.model_validate(right_values)

    assert left_hypothesis == right_hypothesis
    assert canonical_sha256(left_hypothesis) == canonical_sha256(right_hypothesis)


def test_canonical_json_and_hash_ignore_mapping_key_order() -> None:
    left = {"nested": {"z": 1, "a": 2}, "name": "测试"}
    right = {"name": "测试", "nested": {"a": 2, "z": 1}}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_json_bytes(left) == (
        b'{"name":"\xe6\xb5\x8b\xe8\xaf\x95","nested":{"a":2,"z":1}}'
    )


def test_canonical_json_rejects_non_string_keys_and_normalizes_negative_zero() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes({"nested": [{1: "silently coerced by json.dumps"}]})

    assert canonical_json_bytes({"zero": -0.0}) == b'{"zero":0.0}'


def test_canonical_json_revalidates_bypassed_model_instances() -> None:
    valid = artifact("a")
    invalid_copy = valid.model_copy(update={"size": -1})
    invalid_constructed = ArtifactRef.model_construct(
        sha256="not-a-digest",
        size=-1,
        media_type="text/plain",
    )

    for invalid in (invalid_copy, invalid_constructed):
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate(invalid)
        with pytest.raises(ValidationError):
            canonical_json_bytes(invalid)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": value})


def test_callable_source_fingerprint_hashes_source_not_cpython_bytecode() -> None:
    expected_source = b"def source_fingerprint_fixture(value: int) -> int:\n    return value + 1\n"

    assert callable_source_sha256(source_fingerprint_fixture) == sha256_bytes(expected_source)
    assert callable_source_sha256(source_fingerprint_fixture) != sha256_bytes(
        source_fingerprint_fixture.__code__.co_code
    )
    with pytest.raises(TypeError, match="Python function"):
        callable_source_sha256(len)


def test_module_source_fingerprint_includes_normalized_module_bundle() -> None:
    module = sys.modules[__name__]
    source = Path(__file__).read_text(encoding="utf-8")
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"

    assert module_source_sha256(module) == sha256_bytes(normalized.encode("utf-8"))
    with pytest.raises(TypeError, match="Python module"):
        module_source_sha256(source_fingerprint_fixture)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source is unavailable"):
        module_source_sha256(sys)
