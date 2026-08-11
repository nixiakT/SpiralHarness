from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.skills.loading import (
    SKILL_DISCLOSURE_RENDERER_FINGERPRINT,
    SkillDisclosure,
    SkillDisclosureLevel,
    SkillPackageError,
    SkillPackageLoader,
)
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

MODEL_FINGERPRINT = "model-fixed"
RUNTIME_FINGERPRINT = "runtime-fixed"


def missing_ref(
    digit: str,
    *,
    media_type: str = "application/json",
) -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def make_license(
    store: ArtifactStore,
    **updates: Any,
) -> SkillLicense:
    values: dict[str, Any] = {
        "spdx_expression": "Apache-2.0",
        "source_kind": SkillSourceKind.THIRD_PARTY,
        "provenance_refs": (store.put_bytes(b"pinned source", media_type="text/plain"),),
        "compliance_review_ref": store.put_json(
            {"approved": True},
            media_type="application/vnd.spiral-harness.compliance-review.v1+json",
        ),
        "notice_ref": store.put_bytes(b"third-party notice", media_type="text/plain"),
    }
    values.update(updates)
    return SkillLicense(**values)


def make_package(store: ArtifactStore, **updates: Any) -> SkillPackage:
    values: dict[str, Any] = {
        "skill_id": "verify-arithmetic",
        "revision": 0,
        "name": "Verify arithmetic",
        "summary": "Checks arithmetic before returning a final answer.",
        "activation_guidance": "Use for multi-step arithmetic questions.",
        "applicability_tags": ("arithmetic", "verification"),
        "rules": (
            SkillRule(
                rule_id="solve-first",
                instruction="Solve the problem independently before checking the result.",
            ),
            SkillRule(
                rule_id="recheck-answer",
                instruction="Recompute the final arithmetic using a second method.",
            ),
        ),
        "procedure": "Solve, recheck, and emit only the verified final answer.",
        "examples": (
            SkillExample(
                input="What is 17 + 25?",
                output="42",
                explanation="The independent sum and recheck agree.",
            ),
        ),
        "compatible_model_fingerprints": (MODEL_FINGERPRINT,),
        "runtime_fingerprints": (RUNTIME_FINGERPRINT,),
        "license": make_license(store),
    }
    values.update(updates)
    return SkillPackage(**values)


def replace_package(package: SkillPackage, **updates: Any) -> SkillPackage:
    values = package.model_dump(
        mode="python",
        by_alias=False,
        exclude_none=False,
        round_trip=True,
        warnings="none",
    )
    values.update(updates)
    return SkillPackage.model_validate(values, strict=True)


def put_package(store: ArtifactStore, package: SkillPackage) -> ArtifactRef:
    ref = store.put_json(package, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    assert ref == package.artifact_ref
    return ref


def load(
    loader: SkillPackageLoader,
    ref: ArtifactRef,
    *,
    model_fingerprint: str = MODEL_FINGERPRINT,
    runtime_fingerprint: str = RUNTIME_FINGERPRINT,
) -> SkillPackage:
    return loader.load(
        ref,
        model_fingerprint=model_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    )


def context_body(disclosure: SkillDisclosure) -> dict[str, Any]:
    prefix = f"{SKILL_CONTEXT_START_DELIMITER}\n"
    suffix = f"\n{SKILL_CONTEXT_END_DELIMITER}"
    assert disclosure.context.startswith(prefix)
    assert disclosure.context.endswith(suffix)
    return json.loads(disclosure.context[len(prefix) : -len(suffix)])


def rerendered_disclosure_values(
    disclosure: SkillDisclosure,
    envelope: dict[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    body = canonical_json(envelope)
    context = f"{SKILL_CONTEXT_START_DELIMITER}\n{body}\n{SKILL_CONTEXT_END_DELIMITER}"
    payload = context.encode("utf-8")
    values = disclosure.model_dump(mode="python", round_trip=True, warnings="none")
    values.update(
        context=context,
        context_sha256=sha256_bytes(payload),
        context_size_bytes=len(payload),
    )
    values.update(updates)
    return values


def test_loader_reads_one_exact_canonical_package_and_compliance_closure(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    package = make_package(store)
    ref = put_package(store, package)

    loaded = load(SkillPackageLoader(store), ref)

    assert loaded == package
    assert loaded.artifact_ref == ref


def test_loader_rejects_tampered_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    store.path_for(ref).write_bytes(b"x" * ref.size)

    with pytest.raises(SkillPackageError, match="could not be verified"):
        load(SkillPackageLoader(store), ref)


def test_loader_requires_exact_skill_media_type(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    wrong_type = ArtifactRef(
        sha256=ref.sha256,
        size=ref.size,
        media_type="application/json",
    )

    with pytest.raises(SkillPackageError, match="exact skill package media type"):
        load(SkillPackageLoader(store), wrong_type)


def test_loader_rejects_noncanonical_typed_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    package = make_package(store)
    noncanonical = json.dumps(
        package.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    ref = store.put_bytes(noncanonical, media_type=SKILL_PACKAGE_MEDIA_TYPE)

    with pytest.raises(SkillPackageError, match="could not be verified"):
        load(SkillPackageLoader(store), ref)


def test_loader_rejects_package_without_required_license(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    content = make_package(store).model_dump(mode="json")
    del content["license"]
    ref = store.put_json(content, media_type=SKILL_PACKAGE_MEDIA_TYPE)

    with pytest.raises(SkillPackageError, match="could not be verified"):
        load(SkillPackageLoader(store), ref)


@pytest.mark.parametrize("missing", ["provenance", "review", "notice"])
def test_loader_requires_every_compliance_artifact(tmp_path: Path, missing: str) -> None:
    store = ArtifactStore(tmp_path)
    valid = make_license(store)
    values = valid.model_dump(
        mode="python",
        by_alias=False,
        exclude_none=False,
        round_trip=True,
        warnings="none",
    )
    if missing == "provenance":
        values["provenance_refs"] = (missing_ref("a", media_type="text/plain"),)
    elif missing == "review":
        values["compliance_review_ref"] = missing_ref(
            "b",
            media_type="application/vnd.spiral-harness.compliance-review.v1+json",
        )
    else:
        values["notice_ref"] = missing_ref("c", media_type="text/plain")
    package = make_package(store, license=SkillLicense.model_validate(values, strict=True))
    ref = put_package(store, package)

    with pytest.raises(SkillPackageError, match="could not be verified"):
        load(SkillPackageLoader(store), ref)


@pytest.mark.parametrize(
    ("model_fingerprint", "runtime_fingerprint", "message"),
    [
        ("another-model", RUNTIME_FINGERPRINT, "frozen model"),
        (MODEL_FINGERPRINT, "another-runtime", "frozen runtime"),
    ],
)
def test_loader_requires_exact_model_and_runtime_compatibility(
    tmp_path: Path,
    model_fingerprint: str,
    runtime_fingerprint: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))

    with pytest.raises(SkillPackageError, match=message):
        load(
            SkillPackageLoader(store),
            ref,
            model_fingerprint=model_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )


def test_progressive_disclosure_is_deterministic_and_never_exposes_a_higher_tier(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    loader = SkillPackageLoader(store)

    metadata = loader.disclose(
        ref,
        level=SkillDisclosureLevel.METADATA,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    rules = loader.disclose(
        ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    full = loader.disclose(
        ref,
        level=SkillDisclosureLevel.FULL,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )

    assert metadata.exposed_sections == ("metadata",)
    assert rules.exposed_sections == ("metadata", "rules")
    assert full.exposed_sections == ("metadata", "rules", "procedure", "examples")
    assert set(context_body(metadata)["sections"]) == {"metadata"}
    assert set(context_body(rules)["sections"]) == {"metadata", "rules"}
    assert set(context_body(full)["sections"]) == {
        "metadata",
        "rules",
        "procedure",
        "examples",
    }
    assert metadata.context_sha256 == sha256_bytes(metadata.context.encode("utf-8"))
    assert metadata.context_size_bytes == len(metadata.context.encode("utf-8"))
    assert metadata.renderer_fingerprint == SKILL_DISCLOSURE_RENDERER_FINGERPRINT
    assert (
        loader.disclose(
            ref,
            level=SkillDisclosureLevel.METADATA,
            model_fingerprint=MODEL_FINGERPRINT,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
        )
        == metadata
    )


def test_disclosure_accepts_the_package_text_length_boundaries(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    package = make_package(
        store,
        skill_id="s" * 128,
        name="n" * 256,
        summary="s" * 2_048,
        activation_guidance="a" * 2_048,
        applicability_tags=("t" * 256,),
        procedure="p" * 65_536,
    )
    disclosure = SkillPackageLoader(store).disclose(
        put_package(store, package),
        level=SkillDisclosureLevel.FULL,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )

    metadata = context_body(disclosure)["sections"]["metadata"]
    assert metadata["skill_id"] == package.skill_id
    assert metadata["name"] == package.name
    assert metadata["summary"] == package.summary
    assert metadata["activation_guidance"] == package.activation_guidance
    assert metadata["applicability_tags"] == list(package.applicability_tags)
    assert context_body(disclosure)["sections"]["procedure"] == package.procedure


def test_disclosure_rejects_boolean_context_revision_equal_to_zero(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    disclosure = SkillPackageLoader(store).disclose(
        ref,
        level=SkillDisclosureLevel.METADATA,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    envelope = context_body(disclosure)
    envelope["sections"]["metadata"]["revision"] = False

    with pytest.raises(ValidationError, match="revision"):
        SkillDisclosure(**rerendered_disclosure_values(disclosure, envelope))


@pytest.mark.parametrize("skill_id", ["Invalid_Slug", "s" * 129])
def test_disclosure_rejects_skill_id_outside_package_slug_schema(
    tmp_path: Path,
    skill_id: str,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    disclosure = SkillPackageLoader(store).disclose(
        ref,
        level=SkillDisclosureLevel.METADATA,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    envelope = context_body(disclosure)
    envelope["sections"]["metadata"]["skill_id"] = skill_id

    with pytest.raises(ValidationError):
        SkillDisclosure(
            **rerendered_disclosure_values(
                disclosure,
                envelope,
                skill_id=skill_id,
            )
        )


@pytest.mark.parametrize(
    ("section", "field_name", "value", "message"),
    [
        ("metadata", "name", "n" * 257, "metadata name"),
        ("metadata", "summary", "s" * 2_049, "metadata summary"),
        (
            "metadata",
            "activation_guidance",
            "a" * 2_049,
            "metadata activation_guidance",
        ),
        ("metadata", "applicability_tags", ["t" * 257], "applicability tag"),
        ("procedure", None, "p" * 65_537, "procedure"),
    ],
    ids=[
        "metadata-name",
        "metadata-summary",
        "metadata-activation-guidance",
        "metadata-applicability-tag",
        "procedure",
    ],
)
def test_disclosure_rejects_text_beyond_package_schema_boundaries(
    tmp_path: Path,
    section: str,
    field_name: str | None,
    value: object,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    disclosure = SkillPackageLoader(store).disclose(
        ref,
        level=SkillDisclosureLevel.FULL,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    envelope = context_body(disclosure)
    sections = envelope["sections"]
    if field_name is None:
        sections[section] = value
    else:
        sections[section][field_name] = value

    with pytest.raises(ValidationError, match=message):
        SkillDisclosure(**rerendered_disclosure_values(disclosure, envelope))


def test_disclosure_model_rejects_tampered_hash_size_renderer_sections_and_framing(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    disclosure = SkillPackageLoader(store).disclose(
        ref,
        level=SkillDisclosureLevel.FULL,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    values = disclosure.model_dump(mode="python", round_trip=True, warnings="none")

    invalid_updates = (
        {"context_sha256": "f" * 64},
        {"context_size_bytes": disclosure.context_size_bytes + 1},
        {"renderer_fingerprint": "e" * 64},
        {"exposed_sections": ("metadata",)},
        {"context": disclosure.context.removeprefix(SKILL_CONTEXT_START_DELIMITER)},
    )
    for update in invalid_updates:
        with pytest.raises(ValidationError):
            SkillDisclosure(**{**values, **update})


def test_verify_disclosure_replays_package_instead_of_trusting_self_consistency(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = put_package(store, make_package(store))
    loader = SkillPackageLoader(store)
    disclosure = loader.disclose(
        ref,
        level=SkillDisclosureLevel.METADATA,
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )
    envelope = context_body(disclosure)
    envelope["sections"]["metadata"]["summary"] = "Caller-authored replacement summary."
    body = canonical_json(envelope)
    forged_context = f"{SKILL_CONTEXT_START_DELIMITER}\n{body}\n{SKILL_CONTEXT_END_DELIMITER}"
    forged_payload = forged_context.encode("utf-8")
    forged = SkillDisclosure(
        **disclosure.model_dump(
            mode="python",
            exclude={"context", "context_sha256", "context_size_bytes"},
            round_trip=True,
            warnings="none",
        ),
        context=forged_context,
        context_sha256=sha256_bytes(forged_payload),
        context_size_bytes=len(forged_payload),
    )

    with pytest.raises(SkillPackageError, match="trusted package projection"):
        loader.verify_disclosure(
            forged,
            model_fingerprint=MODEL_FINGERPRINT,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
        )
    assert (
        loader.verify_disclosure(
            disclosure,
            model_fingerprint=MODEL_FINGERPRINT,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
        )
        == disclosure
    )


def changed_rules() -> tuple[SkillRule, ...]:
    return (
        SkillRule(
            rule_id="solve-first",
            instruction="Solve independently and record every intermediate result.",
        ),
        SkillRule(
            rule_id="recheck-answer",
            instruction="Recompute the final arithmetic using a second method.",
        ),
    )


def evolvable_package(store: ArtifactStore) -> SkillPackage:
    return make_package(
        store,
        license=make_license(
            store,
            source_kind=SkillSourceKind.GENERATED,
            notice_ref=None,
        ),
    )


def test_verify_revision_accepts_only_the_next_rules_revision(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    before = evolvable_package(store)
    before_ref = put_package(store, before)
    after = replace_package(
        before,
        revision=1,
        parent_package_ref=before_ref,
        rules=changed_rules(),
    )
    after_ref = put_package(store, after)

    verified = SkillPackageLoader(store).verify_revision(
        before_ref=before_ref,
        after_ref=after_ref,
        expected_component_name="verify-arithmetic",
        model_fingerprint=MODEL_FINGERPRINT,
        runtime_fingerprint=RUNTIME_FINGERPRINT,
    )

    assert verified == (before, after)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong-parent", "parent_package_ref"),
        ("revision-skip", "increment"),
        ("no-op", "must change rules"),
        ("frozen-field", "frozen fields"),
        ("wrong-component", "expected component"),
    ],
)
def test_verify_revision_rejects_invalid_lineage_or_scope(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path)
    before = evolvable_package(store)
    before_ref = put_package(store, before)
    updates: dict[str, Any] = {
        "revision": 1,
        "parent_package_ref": before_ref,
        "rules": changed_rules(),
    }
    expected_name = "verify-arithmetic"
    if case == "wrong-parent":
        updates["parent_package_ref"] = missing_ref(
            "d",
            media_type=SKILL_PACKAGE_MEDIA_TYPE,
        )
    elif case == "revision-skip":
        updates["revision"] = 2
    elif case == "no-op":
        updates["rules"] = before.rules
    elif case == "frozen-field":
        updates["summary"] = "A drifted summary that was not authorized by this mutation."
    else:
        expected_name = "another-component"
    after_ref = put_package(store, replace_package(before, **updates))

    with pytest.raises(SkillPackageError, match=message):
        SkillPackageLoader(store).verify_revision(
            before_ref=before_ref,
            after_ref=after_ref,
            expected_component_name=expected_name,
            model_fingerprint=MODEL_FINGERPRINT,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "source_kind",
    [SkillSourceKind.FIRST_PARTY, SkillSourceKind.THIRD_PARTY],
)
def test_verify_revision_rejects_packages_without_generated_provenance(
    tmp_path: Path,
    source_kind: SkillSourceKind,
) -> None:
    store = ArtifactStore(tmp_path)
    license_updates: dict[str, Any] = {"source_kind": source_kind}
    if source_kind is SkillSourceKind.FIRST_PARTY:
        license_updates["notice_ref"] = None
    before = make_package(store, license=make_license(store, **license_updates))
    before_ref = put_package(store, before)
    after_ref = put_package(
        store,
        replace_package(
            before,
            revision=1,
            parent_package_ref=before_ref,
            rules=changed_rules(),
        ),
    )

    with pytest.raises(SkillPackageError, match="declaring source_kind=generated"):
        SkillPackageLoader(store).verify_revision(
            before_ref=before_ref,
            after_ref=after_ref,
            expected_component_name="verify-arithmetic",
            model_fingerprint=MODEL_FINGERPRINT,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
        )
