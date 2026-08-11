"""Canonical, declarative skill package contracts.

The complete skill is one immutable JSON artifact.  It deliberately contains
no executable entry point, validator, path, or runtime permission grant.  A
trusted loader may project metadata, rules, or the full package without
changing the package's content-addressed identity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel

SKILL_PACKAGE_MEDIA_TYPE = "application/vnd.spiral-harness.skill-package.v1+json"

# The trusted loader owns these framing tokens.  Package-authored text may not
# contain them, otherwise a skill could escape or forge its context boundary.
SKILL_CONTEXT_START_DELIMITER = "<|spiral-harness:skill-context:start|>"
SKILL_CONTEXT_END_DELIMITER = "<|spiral-harness:skill-context:end|>"
RESERVED_SKILL_CONTEXT_DELIMITERS = (
    SKILL_CONTEXT_START_DELIMITER,
    SKILL_CONTEXT_END_DELIMITER,
)

_SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_UNSAFE_LICENSE_TOKEN = re.compile(
    r"(?<![A-Z0-9])(?:NOASSERTION|NONE|UNKNOWN)(?![A-Z0-9])",
    flags=re.IGNORECASE,
)

Slug = Annotated[str, Field(min_length=1, max_length=128, pattern=_SLUG_PATTERN)]
ShortText = Annotated[str, Field(min_length=1, max_length=256)]
SummaryText = Annotated[str, Field(min_length=1, max_length=2_048)]
InstructionText = Annotated[str, Field(min_length=1, max_length=8_192)]
ProcedureText = Annotated[str, Field(min_length=1, max_length=65_536)]
ExampleText = Annotated[str, Field(min_length=1, max_length=8_192)]
ExplanationText = Annotated[str, Field(min_length=1, max_length=4_096)]
Fingerprint = Annotated[str, Field(min_length=1, max_length=1_024)]


def _iter_text(value: object) -> tuple[str, ...]:
    """Return every string nested in a validated model payload."""

    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(text for item in value.values() for text in _iter_text(item))
    if isinstance(value, list | tuple):
        return tuple(text for item in value for text in _iter_text(item))
    return ()


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _artifact_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return ref.sha256, ref.size, ref.media_type


class _SkillModel(ImmutableModel):
    """Shared fail-closed validation for skill-owned text."""

    @model_validator(mode="after")
    def reject_reserved_context_delimiters(self) -> Self:
        content = self.model_dump(
            mode="python",
            by_alias=False,
            exclude_none=False,
            round_trip=True,
            warnings="none",
        )
        for text in _iter_text(content):
            for delimiter in RESERVED_SKILL_CONTEXT_DELIMITERS:
                if delimiter in text:
                    raise ValueError("skill text contains a reserved context delimiter")
        return self


class SkillSourceKind(StrEnum):
    """Package-declared provenance class; this value is not an attestation."""

    FIRST_PARTY = "first_party"
    GENERATED = "generated"
    THIRD_PARTY = "third_party"


class SkillLicense(_SkillModel):
    """A normalized SPDX declaration plus content-addressed compliance references.

    The references make their exact bytes part of the package closure.  Their
    presence does not authenticate the declared source class or a reviewer,
    prove approval, or establish a legal conclusion.
    """

    spdx_expression: Annotated[str, Field(min_length=1, max_length=256)]
    source_kind: SkillSourceKind
    provenance_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    compliance_review_ref: ArtifactRef
    notice_ref: ArtifactRef | None = None

    @field_validator("spdx_expression")
    @classmethod
    def normalize_spdx_expression(cls, value: str) -> str:
        if any(delimiter in value for delimiter in RESERVED_SKILL_CONTEXT_DELIMITERS):
            raise ValueError("skill text contains a reserved context delimiter")
        try:
            normalized = canonicalize_license_expression(value)
        except InvalidLicenseExpression as exc:
            raise ValueError(
                "spdx_expression must be a valid SPDX license expression and must not "
                "contain NOASSERTION, NONE, or UNKNOWN"
            ) from exc
        if _UNSAFE_LICENSE_TOKEN.search(normalized):
            raise ValueError("spdx_expression must not contain NOASSERTION, NONE, or UNKNOWN")
        return normalized

    @field_validator("provenance_refs")
    @classmethod
    def canonicalize_provenance_refs(
        cls,
        refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(refs, key=_artifact_key))
        digests = tuple(ref.sha256 for ref in ordered)
        if len(digests) != len(set(digests)):
            raise ValueError("provenance_refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def validate_compliance_refs(self) -> Self:
        _require_json_ref(
            self.compliance_review_ref,
            field_name="compliance_review_ref",
        )
        if self.source_kind is SkillSourceKind.THIRD_PARTY and self.notice_ref is None:
            raise ValueError("third_party skills require notice_ref")
        return self


class SkillRule(_SkillModel):
    """One ordered, declarative instruction in a skill procedure."""

    rule_id: Slug
    instruction: InstructionText


class SkillExample(_SkillModel):
    """One declarative input/output example with a bounded explanation."""

    input: ExampleText
    output: ExampleText
    explanation: ExplanationText


class SkillPackage(_SkillModel):
    """One complete immutable skill revision encoded as canonical JSON."""

    schema_version: Literal["1"] = "1"
    format: Literal["declarative_text"] = "declarative_text"
    skill_id: Slug
    revision: Annotated[int, Field(ge=0, strict=True)]
    parent_package_ref: ArtifactRef | None = None
    name: ShortText
    summary: SummaryText
    activation_guidance: SummaryText
    applicability_tags: Annotated[tuple[ShortText, ...], Field(min_length=1)]
    rules: Annotated[tuple[SkillRule, ...], Field(min_length=1)]
    procedure: ProcedureText
    examples: tuple[SkillExample, ...] = ()
    compatible_model_fingerprints: Annotated[
        tuple[Fingerprint, ...],
        Field(min_length=1),
    ]
    runtime_fingerprints: Annotated[tuple[Fingerprint, ...], Field(min_length=1)]
    license: SkillLicense
    loader_contract_version: Literal["v1"] = "v1"
    permissions: Literal["none"] = "none"

    @field_validator(
        "applicability_tags",
        "compatible_model_fingerprints",
        "runtime_fingerprints",
    )
    @classmethod
    def canonicalize_string_sets(
        cls,
        values: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            field_name = getattr(info, "field_name", "set-like field")
            raise ValueError(f"{field_name} must not contain duplicate entries")
        return ordered

    @field_validator("rules")
    @classmethod
    def require_unique_rule_ids(
        cls,
        rules: tuple[SkillRule, ...],
    ) -> tuple[SkillRule, ...]:
        rule_ids = tuple(rule.rule_id for rule in rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rules must have unique rule_id values")
        # Rule order is semantic and must remain exactly as authored.
        return rules

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        is_root = self.revision == 0
        has_parent = self.parent_package_ref is not None
        if is_root == has_parent:
            raise ValueError("revision 0 must have no parent; later revisions require one")
        if self.parent_package_ref is not None and (
            self.parent_package_ref.media_type != SKILL_PACKAGE_MEDIA_TYPE
        ):
            raise ValueError("parent_package_ref must declare the exact skill package media type")
        return self

    @property
    def artifact_ref(self) -> ArtifactRef:
        """Return the exact reference for this package's canonical JSON bytes."""

        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=sha256_bytes(payload),
            size=len(payload),
            media_type=SKILL_PACKAGE_MEDIA_TYPE,
        )


__all__ = [
    "RESERVED_SKILL_CONTEXT_DELIMITERS",
    "SKILL_CONTEXT_END_DELIMITER",
    "SKILL_CONTEXT_START_DELIMITER",
    "SKILL_PACKAGE_MEDIA_TYPE",
    "SkillExample",
    "SkillLicense",
    "SkillPackage",
    "SkillRule",
    "SkillSourceKind",
]
