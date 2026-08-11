from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.skills.package import (
    RESERVED_SKILL_CONTEXT_DELIMITERS,
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillExample,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)


def artifact(
    digit: str,
    *,
    media_type: str = "application/json",
    size: int = 1,
) -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=size, media_type=media_type)


def license_value(**updates: Any) -> SkillLicense:
    values: dict[str, Any] = {
        "spdx_expression": "Apache-2.0",
        "source_kind": SkillSourceKind.FIRST_PARTY,
        "provenance_refs": (artifact("b"), artifact("a", media_type="text/plain")),
        "compliance_review_ref": artifact(
            "c",
            media_type="application/vnd.spiral-harness.compliance-review.v1+json",
        ),
    }
    values.update(updates)
    return SkillLicense(**values)


def package_values(**updates: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "skill_id": "verify-arithmetic",
        "revision": 0,
        "name": "Verify arithmetic",
        "summary": "Checks arithmetic before returning a final answer.",
        "activation_guidance": "Use for multi-step arithmetic questions.",
        "applicability_tags": ("verification", "arithmetic"),
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
                explanation="The independent sum and the recheck agree.",
            ),
        ),
        "compatible_model_fingerprints": ("model-z", "model-a"),
        "runtime_fingerprints": ("runtime-z", "runtime-a"),
        "license": license_value(),
    }
    values.update(updates)
    return values


def package(**updates: Any) -> SkillPackage:
    return SkillPackage(**package_values(**updates))


def test_package_is_one_canonical_vendor_json_artifact() -> None:
    value = package()
    payload = canonical_json_bytes(value)

    assert SKILL_PACKAGE_MEDIA_TYPE == ("application/vnd.spiral-harness.skill-package.v1+json")
    assert value.schema_version == "1"
    assert value.format == "declarative_text"
    assert value.loader_contract_version == "v1"
    assert value.permissions == "none"
    assert value.artifact_ref == ArtifactRef(
        sha256=sha256_bytes(payload),
        size=len(payload),
        media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    assert SkillPackage.model_validate_json(payload, strict=True) == value


def test_set_like_fields_are_sorted_and_rules_remain_in_semantic_order() -> None:
    value = package()

    assert value.applicability_tags == ("arithmetic", "verification")
    assert value.compatible_model_fingerprints == ("model-a", "model-z")
    assert value.runtime_fingerprints == ("runtime-a", "runtime-z")
    assert tuple(rule.rule_id for rule in value.rules) == (
        "solve-first",
        "recheck-answer",
    )
    assert tuple(ref.sha256 for ref in value.license.provenance_refs) == (
        "a" * 64,
        "b" * 64,
    )


def test_set_input_order_does_not_change_identity_but_rule_order_does() -> None:
    first = package()
    reordered_sets = package(
        applicability_tags=("arithmetic", "verification"),
        compatible_model_fingerprints=("model-a", "model-z"),
        runtime_fingerprints=("runtime-a", "runtime-z"),
        license=license_value(
            provenance_refs=(artifact("a", media_type="text/plain"), artifact("b"))
        ),
    )
    reversed_rules = package(rules=tuple(reversed(first.rules)))

    assert reordered_sets == first
    assert reordered_sets.artifact_ref == first.artifact_ref
    assert reversed_rules.artifact_ref != first.artifact_ref


@pytest.mark.parametrize(
    "field_name",
    [
        "applicability_tags",
        "compatible_model_fingerprints",
        "runtime_fingerprints",
    ],
)
def test_nonempty_string_sets_reject_empty_and_duplicate_values(field_name: str) -> None:
    with pytest.raises(ValidationError):
        package(**{field_name: ()})

    with pytest.raises(ValidationError, match="duplicate"):
        package(**{field_name: ("same", "same")})


def test_license_requires_nonempty_unique_sorted_provenance() -> None:
    with pytest.raises(ValidationError):
        license_value(provenance_refs=())

    with pytest.raises(ValidationError, match="duplicate"):
        license_value(
            provenance_refs=(
                artifact("a", media_type="application/json"),
                artifact("a", media_type="text/plain", size=2),
            )
        )


@pytest.mark.parametrize(
    "expression",
    [
        "NOASSERTION",
        "none",
        "Unknown",
        "MIT OR NOASSERTION",
        "LicenseRef-UNKNOWN",
    ],
)
def test_license_rejects_unknown_or_absent_spdx_claims(expression: str) -> None:
    with pytest.raises(ValidationError, match="NOASSERTION"):
        license_value(spdx_expression=expression)


def test_license_parses_and_canonicalizes_spdx_expressions() -> None:
    assert license_value(spdx_expression="mit or apache-2.0").spdx_expression == "MIT OR Apache-2.0"
    assert (
        license_value(spdx_expression="gpl-2.0-only with classpath-exception-2.0").spdx_expression
        == "GPL-2.0-only WITH Classpath-exception-2.0"
    )
    assert (
        license_value(spdx_expression="LicenseRef-Project-Internal").spdx_expression
        == "LicenseRef-Project-Internal"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "Definitely-Not-A-License",
        "MIT nonsense",
        "MIT OR",
        "(MIT",
    ],
)
def test_license_rejects_invalid_or_unknown_spdx_expressions(expression: str) -> None:
    with pytest.raises(ValidationError, match="valid SPDX license expression"):
        license_value(spdx_expression=expression)


def test_compliance_review_must_be_json() -> None:
    with pytest.raises(ValidationError, match="JSON media type"):
        license_value(compliance_review_ref=artifact("c", media_type="text/plain"))

    assert license_value(
        compliance_review_ref=artifact("c", media_type="application/json; charset=utf-8")
    ).compliance_review_ref.media_type.startswith("application/json")


def test_third_party_license_requires_notice_but_other_sources_do_not() -> None:
    with pytest.raises(ValidationError, match="third_party skills require notice_ref"):
        license_value(source_kind=SkillSourceKind.THIRD_PARTY)

    notice = artifact("d", media_type="text/plain")
    third_party = license_value(
        source_kind=SkillSourceKind.THIRD_PARTY,
        notice_ref=notice,
    )

    assert third_party.notice_ref == notice
    assert license_value(source_kind=SkillSourceKind.GENERATED).notice_ref is None


@pytest.mark.parametrize(
    ("revision", "parent", "message"),
    [
        (0, artifact("d", media_type=SKILL_PACKAGE_MEDIA_TYPE), "revision 0"),
        (1, None, "later revisions"),
        (1, artifact("d", media_type="application/json"), "exact skill package"),
    ],
)
def test_revision_and_parent_reference_form_exact_lineage(
    revision: int,
    parent: ArtifactRef | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        package(revision=revision, parent_package_ref=parent)

    child = package(
        revision=1,
        parent_package_ref=artifact("d", media_type=SKILL_PACKAGE_MEDIA_TYPE),
    )
    assert child.revision == 1


@pytest.mark.parametrize(
    "invalid_slug",
    ["Uppercase", "under_score", "leading-", "-trailing", "double--dash", "has space"],
)
def test_skill_and_rule_identifiers_are_lowercase_slugs(invalid_slug: str) -> None:
    with pytest.raises(ValidationError):
        package(skill_id=invalid_slug)

    with pytest.raises(ValidationError):
        SkillRule(rule_id=invalid_slug, instruction="Valid instruction")


def test_rules_are_nonempty_unique_and_preserve_authored_order() -> None:
    with pytest.raises(ValidationError):
        package(rules=())

    duplicate_rules = (
        SkillRule(rule_id="same", instruction="First instruction"),
        SkillRule(rule_id="same", instruction="Second instruction"),
    )
    with pytest.raises(ValidationError, match="unique rule_id"):
        package(rules=duplicate_rules)


def test_procedure_is_nonempty_and_examples_may_be_empty() -> None:
    with pytest.raises(ValidationError):
        package(procedure="")

    assert package(examples=()).examples == ()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SkillRule(rule_id="bounded", instruction="x" * 8_193),
        lambda: SkillExample(input="x" * 8_193, output="ok", explanation="why"),
        lambda: SkillExample(input="ok", output="x" * 8_193, explanation="why"),
        lambda: SkillExample(input="ok", output="ok", explanation="x" * 4_097),
    ],
)
def test_rule_and_example_text_is_bounded(factory: Any) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize("delimiter", RESERVED_SKILL_CONTEXT_DELIMITERS)
def test_reserved_context_delimiters_are_rejected_from_leaf_text(delimiter: str) -> None:
    with pytest.raises(ValidationError, match="reserved context delimiter"):
        SkillRule(rule_id="rule", instruction=f"before {delimiter} after")
    with pytest.raises(ValidationError, match="reserved context delimiter"):
        SkillExample(input="input", output=delimiter, explanation="explanation")
    with pytest.raises(ValidationError, match="reserved context delimiter"):
        license_value(spdx_expression=f"MIT {delimiter}")


@pytest.mark.parametrize("delimiter", RESERVED_SKILL_CONTEXT_DELIMITERS)
@pytest.mark.parametrize(
    "field_name",
    [
        "name",
        "summary",
        "activation_guidance",
        "applicability_tags",
        "procedure",
        "compatible_model_fingerprints",
        "runtime_fingerprints",
    ],
)
def test_reserved_context_delimiters_are_rejected_from_package_text(
    field_name: str,
    delimiter: str,
) -> None:
    value: str | tuple[str, ...] = delimiter
    if field_name in {
        "applicability_tags",
        "compatible_model_fingerprints",
        "runtime_fingerprints",
    }:
        value = (delimiter,)
    with pytest.raises(ValidationError, match="reserved context delimiter"):
        package(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", "2"),
        ("format", "python"),
        ("loader_contract_version", "v2"),
        ("permissions", "tool"),
    ],
)
def test_closed_contract_literals_cannot_be_relaxed(field_name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        package(**{field_name: value})


def test_models_are_frozen_strict_and_forbid_executable_extensions() -> None:
    value = package()
    with pytest.raises(ValidationError):
        value.revision = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        package(revision=True)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SkillPackage(**package_values(), executable_validator="validator.py")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SkillPackage(**package_values(), entrypoint="skill:run")


def test_artifact_identity_revalidates_unsafe_model_copies() -> None:
    unchecked = package().model_copy(update={"permissions": "network"})

    with pytest.raises(ValidationError):
        _ = unchecked.artifact_ref
