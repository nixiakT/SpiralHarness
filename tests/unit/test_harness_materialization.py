from __future__ import annotations

from pathlib import Path

import pytest

from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.execution.materialization import (
    HarnessMaterializationError,
    HarnessMaterializer,
)
from spiral_harness.skills.loading import SkillDisclosureLevel
from spiral_harness.skills.package import (
    SKILL_CONTEXT_END_DELIMITER,
    SKILL_CONTEXT_START_DELIMITER,
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillExample,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BACKEND_FINGERPRINT = "materialization-replay@sha256:fixed-v1"


def fixed_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=BACKEND_FINGERPRINT,
        model="hosted/materialization-model",
        revision="snapshot-2026-08-12",
        tokenizer="provider/materialization-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="materialization-worker@sha256:fixed-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=16,
            timeout_seconds=5.0,
        ),
    )


def skill_license(store: ArtifactStore) -> SkillLicense:
    return SkillLicense(
        spdx_expression="Apache-2.0",
        source_kind=SkillSourceKind.FIRST_PARTY,
        provenance_refs=(store.put_bytes(b"source", media_type="text/plain"),),
        compliance_review_ref=store.put_json(
            {"approved": True},
            media_type="application/vnd.spiral-harness.compliance-review.v1+json",
        ),
    )


def skill_package(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    *,
    skill_id: str = "verify-arithmetic",
    model_fingerprints: tuple[str, ...] | None = None,
    runtime_fingerprints: tuple[str, ...] | None = None,
) -> SkillPackage:
    return SkillPackage(
        skill_id=skill_id,
        revision=0,
        name="Verify arithmetic",
        summary="Checks arithmetic before returning a final answer.",
        activation_guidance="Use for multi-step arithmetic questions.",
        applicability_tags=("arithmetic", "verification"),
        rules=(
            SkillRule(
                rule_id="solve-first",
                instruction="Solve independently before checking the result.",
            ),
            SkillRule(
                rule_id="recheck-answer",
                instruction="Recompute the final arithmetic using another method.",
            ),
        ),
        procedure="FULL_ONLY_PROCEDURE: solve, recheck, then answer.",
        examples=(
            SkillExample(
                input="FULL_ONLY_EXAMPLE_INPUT",
                output="FULL_ONLY_EXAMPLE_OUTPUT",
                explanation="This example is disclosed only at the full tier.",
            ),
        ),
        compatible_model_fingerprints=(
            (spec.model_fingerprint,) if model_fingerprints is None else model_fingerprints
        ),
        runtime_fingerprints=(
            (spec.runtime_fingerprint,) if runtime_fingerprints is None else runtime_fingerprints
        ),
        license=skill_license(store),
    )


def put_skill(store: ArtifactStore, package: SkillPackage) -> ArtifactRef:
    ref = store.put_json(package, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    assert ref == package.artifact_ref
    return ref


def prompt_component(
    store: ArtifactStore,
    *,
    name: str = "system-prompt",
    prompt: bytes = b"Solve carefully.",
    media_type: str = "text/plain; charset=utf-8",
) -> HarnessComponentRef:
    return HarnessComponentRef(
        name=name,
        kind=ComponentKind.PROMPT,
        artifact=store.put_bytes(prompt, media_type=media_type),
    )


def skill_component(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    *,
    name: str = "verify-arithmetic",
    package: SkillPackage | None = None,
) -> HarnessComponentRef:
    value = skill_package(store, spec, skill_id=name) if package is None else package
    return HarnessComponentRef(
        name=name,
        kind=ComponentKind.SKILL,
        artifact=put_skill(store, value),
    )


def put_harness(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    components: tuple[HarnessComponentRef, ...],
    *,
    model_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
) -> ArtifactRef:
    manifest = HarnessManifest(
        model_fingerprint=(
            spec.model_fingerprint if model_fingerprint is None else model_fingerprint
        ),
        runtime_fingerprint=(
            spec.runtime_fingerprint if runtime_fingerprint is None else runtime_fingerprint
        ),
        trusted_plane_version="trusted-materialization-v1",
        components=components,
    )
    return store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)


def test_prompt_only_materialization_retains_exact_base_and_resolved_hashes(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    spec = fixed_spec()
    prompt = "  先思考。\nReturn π.  "
    harness_ref = put_harness(
        store,
        spec,
        (prompt_component(store, prompt=prompt.encode("utf-8")),),
    )

    resolved = HarnessMaterializer(store, spec=spec).materialize(harness_ref)

    assert resolved.harness_id == harness_ref.sha256
    assert resolved.base_system_prompt == prompt
    assert resolved.system_prompt == prompt
    assert resolved.base_system_prompt_sha256 == resolved.resolved_prompt_sha256
    assert resolved.skill_disclosure is None


def test_default_skill_materialization_exposes_rules_but_not_full_only_content(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    spec = fixed_spec()
    base = "Solve carefully."
    package = skill_package(store, spec)
    skill = skill_component(store, spec, package=package)
    harness_ref = put_harness(
        store,
        spec,
        (prompt_component(store, prompt=base.encode()), skill),
    )

    resolved = HarnessMaterializer(store, spec=spec).materialize(harness_ref)

    disclosure = resolved.skill_disclosure
    assert disclosure is not None
    assert disclosure.package_ref == skill.artifact
    assert disclosure.level is SkillDisclosureLevel.RULES
    assert disclosure.exposed_sections == ("metadata", "rules")
    assert "solve-first" in disclosure.context
    assert "FULL_ONLY_PROCEDURE" not in disclosure.context
    assert "FULL_ONLY_EXAMPLE_OUTPUT" not in disclosure.context
    assert disclosure.context.count(SKILL_CONTEXT_START_DELIMITER) == 1
    assert disclosure.context.count(SKILL_CONTEXT_END_DELIMITER) == 1
    assert resolved.system_prompt == base + "\n\n" + disclosure.context
    assert resolved.resolved_prompt_sha256 != resolved.base_system_prompt_sha256


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-prompt", "exactly one prompt"),
        ("extra-prompt", "exactly one prompt"),
        ("extra-skill", "at most one skill"),
        ("unsupported", "unsupported execution component"),
    ],
)
def test_component_roster_fails_closed(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path / case)
    spec = fixed_spec()
    prompt = prompt_component(store)
    skill = skill_component(store, spec)
    if case == "missing-prompt":
        components = (skill,)
    elif case == "extra-prompt":
        components = (prompt, prompt_component(store, name="second-prompt"))
    elif case == "extra-skill":
        components = (
            prompt,
            skill,
            skill_component(store, spec, name="second-skill"),
        )
    else:
        components = (
            prompt,
            HarnessComponentRef(
                name="memory",
                kind=ComponentKind.MEMORY,
                artifact=store.put_bytes(b"memory", media_type="text/plain"),
            ),
        )
    harness_ref = put_harness(store, spec, components)

    with pytest.raises(HarnessMaterializationError, match=message):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("model", "model fingerprint"),
        ("runtime", "runtime fingerprint"),
    ],
)
def test_harness_execution_context_must_match_the_frozen_spec(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path / field_name)
    spec = fixed_spec()
    updates = {f"{field_name}_fingerprint": f"foreign-{field_name}"}
    harness_ref = put_harness(store, spec, (prompt_component(store),), **updates)

    with pytest.raises(HarnessMaterializationError, match=message):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


@pytest.mark.parametrize(
    "media_type",
    [
        "application/json",
        "application/vnd.spiral-harness.manifest+json",
        f"{HARNESS_MANIFEST_MEDIA_TYPE}; charset=utf-8",
    ],
)
def test_materializer_rejects_manifest_bytes_under_any_media_alias(
    tmp_path: Path,
    media_type: str,
) -> None:
    store = ArtifactStore(tmp_path / "wrong-media")
    spec = fixed_spec()
    harness_ref = put_harness(store, spec, (prompt_component(store),))
    aliased_ref = ArtifactRef(
        sha256=harness_ref.sha256,
        size=harness_ref.size,
        media_type=media_type,
    )

    with pytest.raises(HarnessMaterializationError, match="exact harness manifest v2"):
        HarnessMaterializer(store, spec=spec).materialize(aliased_ref)


@pytest.mark.parametrize("kind", ["model", "runtime"])
def test_skill_compatibility_is_checked_during_materialization(
    tmp_path: Path,
    kind: str,
) -> None:
    store = ArtifactStore(tmp_path / kind)
    spec = fixed_spec()
    updates = (
        {"model_fingerprints": ("foreign-model",)}
        if kind == "model"
        else {"runtime_fingerprints": ("foreign-runtime",)}
    )
    package = skill_package(store, spec, **updates)
    harness_ref = put_harness(
        store,
        spec,
        (
            prompt_component(store),
            skill_component(store, spec, package=package),
        ),
    )

    with pytest.raises(HarnessMaterializationError, match=f"frozen {kind}"):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


def test_skill_package_identity_must_match_component_name(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    spec = fixed_spec()
    other = skill_package(store, spec, skill_id="other-skill")
    harness_ref = put_harness(
        store,
        spec,
        (
            prompt_component(store),
            skill_component(
                store,
                spec,
                name="verify-arithmetic",
                package=other,
            ),
        ),
    )

    with pytest.raises(HarnessMaterializationError, match="identity"):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


@pytest.mark.parametrize(
    ("payload", "media_type", "message"),
    [
        (b"prompt", "application/json", "text/plain"),
        (b"\xff", "text/plain", "UTF-8"),
        (b"", "text/plain", "must not be empty"),
    ],
)
def test_prompt_payload_must_be_nonempty_exact_utf8_text(
    tmp_path: Path,
    payload: bytes,
    media_type: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path / message.replace("/", "-"))
    spec = fixed_spec()
    harness_ref = put_harness(
        store,
        spec,
        (prompt_component(store, prompt=payload, media_type=media_type),),
    )

    with pytest.raises(HarnessMaterializationError, match=message):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


@pytest.mark.parametrize("with_skill", [False, True])
@pytest.mark.parametrize(
    "delimiter",
    [
        pytest.param(SKILL_CONTEXT_START_DELIMITER, id="start"),
        pytest.param(SKILL_CONTEXT_END_DELIMITER, id="end"),
    ],
)
def test_base_prompt_cannot_forge_skill_context_boundaries(
    tmp_path: Path,
    with_skill: bool,
    delimiter: str,
) -> None:
    store = ArtifactStore(tmp_path)
    spec = fixed_spec()
    components = [prompt_component(store, prompt=f"base {delimiter}".encode())]
    if with_skill:
        components.append(skill_component(store, spec))
    harness_ref = put_harness(store, spec, tuple(components))

    with pytest.raises(HarnessMaterializationError, match="reserved skill context"):
        HarnessMaterializer(store, spec=spec).materialize(harness_ref)


def test_manifest_or_prompt_tampering_is_rejected(tmp_path: Path) -> None:
    spec = fixed_spec()

    manifest_store = ArtifactStore(tmp_path / "manifest")
    manifest_ref = put_harness(
        manifest_store,
        spec,
        (prompt_component(manifest_store),),
    )
    manifest_store.path_for(manifest_ref).write_bytes(b"x" * manifest_ref.size)
    with pytest.raises(HarnessMaterializationError, match="could not be verified"):
        HarnessMaterializer(manifest_store, spec=spec).materialize(manifest_ref)

    prompt_store = ArtifactStore(tmp_path / "prompt")
    prompt = prompt_component(prompt_store)
    prompt_ref = prompt.artifact
    harness_ref = put_harness(prompt_store, spec, (prompt,))
    prompt_store.path_for(prompt_ref).write_bytes(b"x" * prompt_ref.size)
    with pytest.raises(HarnessMaterializationError, match="could not be verified"):
        HarnessMaterializer(prompt_store, spec=spec).materialize(harness_ref)
