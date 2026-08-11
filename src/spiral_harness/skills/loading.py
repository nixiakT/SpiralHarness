"""Trusted loading, disclosure, and revision checks for declarative skills."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json, canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.skills.package import (
    SKILL_CONTEXT_END_DELIMITER,
    SKILL_CONTEXT_START_DELIMITER,
    SKILL_PACKAGE_MEDIA_TYPE,
    ProcedureText,
    ShortText,
    SkillExample,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
    Slug,
    SummaryText,
)
from spiral_harness.storage.protocol import ArtifactRepository

_RENDERER_ID = "spiral-harness/skill-disclosure-json/v1"
SKILL_DISCLOSURE_RENDERER_FINGERPRINT = sha256_bytes(_RENDERER_ID.encode("utf-8"))

DisclosureSection = Literal["metadata", "rules", "procedure", "examples"]


class SkillPackageError(ValueError):
    """Raised when a package cannot cross the trusted skill boundary."""


class SkillDisclosureLevel(StrEnum):
    """The three fixed progressive-disclosure projections."""

    METADATA = "metadata"
    RULES = "rules"
    FULL = "full"


_EXPOSED_SECTIONS: dict[SkillDisclosureLevel, tuple[DisclosureSection, ...]] = {
    SkillDisclosureLevel.METADATA: ("metadata",),
    SkillDisclosureLevel.RULES: ("metadata", "rules"),
    SkillDisclosureLevel.FULL: ("metadata", "rules", "procedure", "examples"),
}

_SLUG_ADAPTER = TypeAdapter(Slug)
_SHORT_TEXT_ADAPTER = TypeAdapter(ShortText)
_SUMMARY_TEXT_ADAPTER = TypeAdapter(SummaryText)
_PROCEDURE_TEXT_ADAPTER = TypeAdapter(ProcedureText)


def _validate_package_text(
    value: object,
    *,
    field_name: str,
    adapter: TypeAdapter[str],
) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"context {field_name} is not exact non-empty text")
    try:
        adapter.validate_python(value, strict=True)
    except ValueError as exc:
        raise ValueError(f"context {field_name} does not satisfy its package schema") from exc


def _decode_canonical_context(context: str) -> dict[str, Any]:
    def reject_non_finite(token: str) -> None:
        raise ValueError(f"non-finite disclosure value: {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate disclosure context key: {key!r}")
            result[key] = value
        return result

    prefix = f"{SKILL_CONTEXT_START_DELIMITER}\n"
    suffix = f"\n{SKILL_CONTEXT_END_DELIMITER}"
    if not context.startswith(prefix) or not context.endswith(suffix):
        raise ValueError("context must use the trusted skill boundary delimiters")
    inner = context[len(prefix) : -len(suffix)]
    if SKILL_CONTEXT_START_DELIMITER in inner or SKILL_CONTEXT_END_DELIMITER in inner:
        raise ValueError("context body contains a reserved skill boundary delimiter")
    try:
        value = json.loads(
            inner,
            parse_constant=reject_non_finite,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("context must be canonical JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != inner:
        raise ValueError("context body must be one canonical JSON object")
    return value


def _validate_disclosed_sections(
    sections: dict[str, Any],
    *,
    skill_id: str,
    revision: int,
) -> None:
    metadata = sections.get("metadata")
    expected_metadata_fields = {
        "activation_guidance",
        "applicability_tags",
        "name",
        "revision",
        "skill_id",
        "summary",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_fields:
        raise ValueError("context metadata has an unexpected shape")
    _validate_package_text(
        metadata["skill_id"],
        field_name="metadata skill_id",
        adapter=_SLUG_ADAPTER,
    )
    disclosed_revision = metadata["revision"]
    if type(disclosed_revision) is not int or disclosed_revision < 0:
        raise ValueError("context metadata revision must be a non-negative exact integer")
    if (metadata["skill_id"], disclosed_revision) != (skill_id, revision):
        raise ValueError("context metadata does not identify the disclosed package")
    for field_name, adapter in (
        ("activation_guidance", _SUMMARY_TEXT_ADAPTER),
        ("name", _SHORT_TEXT_ADAPTER),
        ("summary", _SUMMARY_TEXT_ADAPTER),
    ):
        _validate_package_text(
            metadata[field_name],
            field_name=f"metadata {field_name}",
            adapter=adapter,
        )
    tags = metadata["applicability_tags"]
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) for tag in tags)
        or tags != sorted(tags)
        or len(tags) != len(set(tags))
    ):
        raise ValueError("context applicability_tags are not the canonical package set")
    for tag in tags:
        _validate_package_text(
            tag,
            field_name="applicability tag",
            adapter=_SHORT_TEXT_ADAPTER,
        )

    if "rules" in sections:
        rules = sections["rules"]
        if not isinstance(rules, list) or not rules:
            raise ValueError("context rules must be a non-empty JSON array")
        checked_rules = tuple(SkillRule.model_validate(rule, strict=True) for rule in rules)
        if [rule.model_dump(mode="json") for rule in checked_rules] != rules:
            raise ValueError("context rules differ from their canonical typed form")
    if "procedure" in sections:
        procedure = sections["procedure"]
        _validate_package_text(
            procedure,
            field_name="procedure",
            adapter=_PROCEDURE_TEXT_ADAPTER,
        )
    if "examples" in sections:
        examples = sections["examples"]
        if not isinstance(examples, list):
            raise ValueError("context examples must be a JSON array")
        checked_examples = tuple(
            SkillExample.model_validate(example, strict=True) for example in examples
        )
        if [example.model_dump(mode="json") for example in checked_examples] != examples:
            raise ValueError("context examples differ from their canonical typed form")


class SkillDisclosure(ImmutableModel):
    """Exact, self-checking text disclosed from one immutable package."""

    schema_version: Literal["1"] = "1"
    package_ref: ArtifactRef
    skill_id: Slug
    revision: Annotated[int, Field(ge=0, strict=True)]
    level: SkillDisclosureLevel
    renderer_fingerprint: Sha256 = SKILL_DISCLOSURE_RENDERER_FINGERPRINT
    exposed_sections: tuple[DisclosureSection, ...]
    context: NonEmptyStr
    context_sha256: Sha256
    context_size_bytes: Annotated[int, Field(ge=1, strict=True)]

    @field_validator("context", mode="before")
    @classmethod
    def context_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and value != value.strip():
            raise ValueError("context must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def disclosure_is_self_consistent(self) -> Self:
        if self.package_ref.media_type != SKILL_PACKAGE_MEDIA_TYPE:
            raise ValueError("package_ref must declare the exact skill package media type")
        if self.renderer_fingerprint != SKILL_DISCLOSURE_RENDERER_FINGERPRINT:
            raise ValueError("renderer_fingerprint is not the trusted disclosure renderer")
        expected_sections = _EXPOSED_SECTIONS[self.level]
        if self.exposed_sections != expected_sections:
            raise ValueError("exposed_sections do not match the disclosure level")

        payload = self.context.encode("utf-8")
        if self.context_sha256 != sha256_bytes(payload):
            raise ValueError("context_sha256 does not match the exact context bytes")
        if self.context_size_bytes != len(payload):
            raise ValueError("context_size_bytes does not match the exact context byte length")

        envelope = _decode_canonical_context(self.context)
        if set(envelope) != {
            "level",
            "renderer_fingerprint",
            "schema_version",
            "sections",
        }:
            raise ValueError("context has an unexpected renderer envelope")
        if (
            envelope["schema_version"] != "1"
            or envelope["renderer_fingerprint"] != self.renderer_fingerprint
            or envelope["level"] != self.level.value
        ):
            raise ValueError("context renderer coordinates do not match the disclosure")
        sections = envelope["sections"]
        if not isinstance(sections, dict) or set(sections) != set(expected_sections):
            raise ValueError("context sections do not match exposed_sections")
        _validate_disclosed_sections(
            sections,
            skill_id=self.skill_id,
            revision=self.revision,
        )
        rendered_body = canonical_json(envelope)
        expected_context = (
            f"{SKILL_CONTEXT_START_DELIMITER}\n{rendered_body}\n{SKILL_CONTEXT_END_DELIMITER}"
        )
        if self.context != expected_context:
            raise ValueError("context does not equal the deterministic disclosure rendering")
        return self


class SkillPackageLoader:
    """No-list repository capability for exact package reads and projections."""

    def __init__(self, repository: ArtifactRepository) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise SkillPackageError("repository must implement ArtifactRepository")
        self.__repository = repository

    def load(
        self,
        package_ref: ArtifactRef,
        *,
        model_fingerprint: str,
        runtime_fingerprint: str,
    ) -> SkillPackage:
        """Load a canonical package and require every declared reference to exist.

        Reference resolution proves content-addressed existence only.  It does
        not authenticate the producer or interpret a compliance record as an
        approval.
        """

        try:
            checked_ref = ArtifactRef.model_validate(package_ref, strict=True)
            if checked_ref.media_type != SKILL_PACKAGE_MEDIA_TYPE:
                raise SkillPackageError(
                    "package reference must declare the exact skill package media type"
                )
            checked_model = self._exact_fingerprint(model_fingerprint, "model_fingerprint")
            checked_runtime = self._exact_fingerprint(
                runtime_fingerprint,
                "runtime_fingerprint",
            )
            payload = self.__repository.get_bytes(checked_ref)
            loaded = self.__repository.get_json(checked_ref, SkillPackage)
            if not isinstance(loaded, SkillPackage):
                raise SkillPackageError("repository did not return a typed SkillPackage")
            package = SkillPackage.model_validate(
                loaded.model_dump(
                    mode="python",
                    by_alias=False,
                    exclude_none=False,
                    round_trip=True,
                    warnings="none",
                ),
                strict=True,
            )
            if payload != canonical_json_bytes(package) or checked_ref != package.artifact_ref:
                raise SkillPackageError("skill package is not its canonical typed artifact")
            if checked_model not in package.compatible_model_fingerprints:
                raise SkillPackageError("skill package is incompatible with the frozen model")
            if checked_runtime not in package.runtime_fingerprints:
                raise SkillPackageError("skill package is incompatible with the frozen runtime")

            related_refs: tuple[tuple[str, ArtifactRef], ...] = (
                *(
                    (("parent_package_ref", package.parent_package_ref),)
                    if package.parent_package_ref is not None
                    else ()
                ),
                *(("provenance_ref", ref) for ref in package.license.provenance_refs),
                ("compliance_review_ref", package.license.compliance_review_ref),
                *(
                    (("notice_ref", package.license.notice_ref),)
                    if package.license.notice_ref is not None
                    else ()
                ),
            )
            for field_name, ref in related_refs:
                try:
                    self.__repository.get_bytes(ref)
                except Exception as exc:
                    raise SkillPackageError(f"{field_name} could not be verified") from exc
            return package
        except SkillPackageError:
            raise
        except Exception as exc:
            raise SkillPackageError(f"skill package could not be verified: {exc}") from exc

    def disclose(
        self,
        package_ref: ArtifactRef,
        *,
        level: SkillDisclosureLevel,
        model_fingerprint: str,
        runtime_fingerprint: str,
    ) -> SkillDisclosure:
        """Return one deterministic projection without exposing higher tiers."""

        try:
            if not isinstance(level, SkillDisclosureLevel):
                raise SkillPackageError("level must be a SkillDisclosureLevel")
            package = self.load(
                package_ref,
                model_fingerprint=model_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
            sections: dict[str, object] = {
                "metadata": {
                    "activation_guidance": package.activation_guidance,
                    "applicability_tags": package.applicability_tags,
                    "name": package.name,
                    "revision": package.revision,
                    "skill_id": package.skill_id,
                    "summary": package.summary,
                }
            }
            if level in {SkillDisclosureLevel.RULES, SkillDisclosureLevel.FULL}:
                sections["rules"] = package.rules
            if level is SkillDisclosureLevel.FULL:
                sections["procedure"] = package.procedure
                sections["examples"] = package.examples
            context_body = canonical_json(
                {
                    "schema_version": "1",
                    "renderer_fingerprint": SKILL_DISCLOSURE_RENDERER_FINGERPRINT,
                    "level": level.value,
                    "sections": sections,
                }
            )
            context = (
                f"{SKILL_CONTEXT_START_DELIMITER}\n{context_body}\n{SKILL_CONTEXT_END_DELIMITER}"
            )
            payload = context.encode("utf-8")
            return SkillDisclosure(
                package_ref=package_ref,
                skill_id=package.skill_id,
                revision=package.revision,
                level=level,
                exposed_sections=_EXPOSED_SECTIONS[level],
                context=context,
                context_sha256=sha256_bytes(payload),
                context_size_bytes=len(payload),
            )
        except SkillPackageError:
            raise
        except Exception as exc:
            raise SkillPackageError(f"skill package could not be disclosed: {exc}") from exc

    def verify_disclosure(
        self,
        disclosure: SkillDisclosure,
        *,
        model_fingerprint: str,
        runtime_fingerprint: str,
    ) -> SkillDisclosure:
        """Replay a disclosure from its package and require exact equality."""

        try:
            if not isinstance(disclosure, SkillDisclosure):
                raise SkillPackageError("disclosure must be a SkillDisclosure")
            checked = SkillDisclosure.model_validate(
                disclosure.model_dump(
                    mode="python",
                    by_alias=False,
                    exclude_none=False,
                    round_trip=True,
                    warnings="none",
                ),
                strict=True,
            )
            expected = self.disclose(
                checked.package_ref,
                level=checked.level,
                model_fingerprint=model_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
            if checked != expected:
                raise SkillPackageError(
                    "skill disclosure does not equal the trusted package projection"
                )
            return checked
        except SkillPackageError:
            raise
        except Exception as exc:
            raise SkillPackageError(f"skill disclosure could not be verified: {exc}") from exc

    def verify_revision(
        self,
        *,
        before_ref: ArtifactRef,
        after_ref: ArtifactRef,
        expected_component_name: str,
        model_fingerprint: str,
        runtime_fingerprint: str,
    ) -> tuple[SkillPackage, SkillPackage]:
        """Verify the only first-version mutation: one rules-only next revision."""

        try:
            expected_name = self._exact_fingerprint(
                expected_component_name,
                "expected_component_name",
            )
            before = self.load(
                before_ref,
                model_fingerprint=model_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
            after = self.load(
                after_ref,
                model_fingerprint=model_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
            if before.skill_id != expected_name or after.skill_id != expected_name:
                raise SkillPackageError("skill_id does not match the expected component name")
            if before.license.source_kind is not SkillSourceKind.GENERATED:
                raise SkillPackageError(
                    "only packages declaring source_kind=generated may use this revision path"
                )
            if after.parent_package_ref != before_ref:
                raise SkillPackageError("after parent_package_ref does not exactly match before")
            if after.revision != before.revision + 1:
                raise SkillPackageError("after revision must increment before revision by one")
            if after.rules == before.rules:
                raise SkillPackageError("skill revision must change rules")

            mutable_fields = {"parent_package_ref", "revision", "rules"}
            before_content = before.model_dump(
                mode="python",
                by_alias=False,
                exclude_none=False,
                round_trip=True,
                warnings="none",
            )
            after_content = after.model_dump(
                mode="python",
                by_alias=False,
                exclude_none=False,
                round_trip=True,
                warnings="none",
            )
            drifted = tuple(
                field_name
                for field_name in before_content
                if field_name not in mutable_fields
                and before_content[field_name] != after_content[field_name]
            )
            if drifted:
                raise SkillPackageError(
                    "skill revision changed frozen fields: " + ", ".join(drifted)
                )
            return before, after
        except SkillPackageError:
            raise
        except Exception as exc:
            raise SkillPackageError(f"skill revision could not be verified: {exc}") from exc

    @staticmethod
    def _exact_fingerprint(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise SkillPackageError(f"{field_name} must be an exact non-empty string")
        return value


__all__ = [
    "SKILL_DISCLOSURE_RENDERER_FINGERPRINT",
    "SkillDisclosure",
    "SkillDisclosureLevel",
    "SkillPackageError",
    "SkillPackageLoader",
]
