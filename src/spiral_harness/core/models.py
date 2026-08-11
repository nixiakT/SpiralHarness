"""Immutable M0 domain models for versioned harness experiments."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ImmutableModel(BaseModel):
    """Base contract for content-addressable domain values."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class ComponentKind(StrEnum):
    """Explicitly supported harness mutation surfaces."""

    PROMPT = "prompt"
    SKILL = "skill"
    MEMORY = "memory"
    TOOL = "tool"
    MIDDLEWARE = "middleware"
    CONTROL_FLOW = "control_flow"


class ArtifactRef(ImmutableModel):
    """A verified pointer to immutable content."""

    sha256: Sha256
    size: Annotated[int, Field(ge=0, strict=True)]
    media_type: NonEmptyStr


class HarnessComponentRef(ImmutableModel):
    """The content-addressed value of one named harness component."""

    name: NonEmptyStr
    kind: ComponentKind
    artifact: ArtifactRef


class BudgetPolicy(ImmutableModel):
    """Hard resource ceilings kept separate from the task score.

    A zero ceiling is meaningful (for example, a no-tools probe), while a
    missing field means that this policy does not constrain that resource.
    At least one ceiling must be declared so an empty object cannot masquerade
    as an active budget policy.
    """

    max_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None
    max_tool_calls: Annotated[int, Field(ge=0, strict=True)] | None = None
    max_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)] | None = None
    max_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None
    max_evaluations: Annotated[int, Field(ge=0, strict=True)] | None = None

    @model_validator(mode="after")
    def require_a_limit(self) -> BudgetPolicy:
        if all(
            value is None
            for value in (
                self.max_tokens,
                self.max_tool_calls,
                self.max_wall_time_seconds,
                self.max_cost_usd,
                self.max_evaluations,
            )
        ):
            raise ValueError("a budget policy must declare at least one resource ceiling")
        return self


class HarnessManifest(ImmutableModel):
    """An immutable snapshot of the complete mutable harness surface.

    The model, runtime, and trusted verification plane are fingerprints rather
    than mutable nested configuration.  A change to any of them creates a new
    experiment protocol rather than an attributed component mutation.
    """

    schema_version: Literal["1"] = "1"
    model_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    trusted_plane_version: NonEmptyStr
    parent: ArtifactRef | None = Field(
        default=None,
        validation_alias=AliasChoices("parent", "parent_ref"),
    )
    components: Annotated[tuple[HarnessComponentRef, ...], Field(min_length=1)]
    budget: BudgetPolicy | None = None

    @field_validator("components")
    @classmethod
    def canonicalize_component_order(
        cls,
        components: tuple[HarnessComponentRef, ...],
    ) -> tuple[HarnessComponentRef, ...]:
        """Make a name-to-component snapshot independent of input order."""

        return tuple(sorted(components, key=lambda component: component.name))

    @model_validator(mode="after")
    def component_names_are_unique(self) -> HarnessManifest:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for component in self.components:
            if component.name in seen:
                duplicates.add(component.name)
            seen.add(component.name)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"component names must be unique; duplicates: {joined}")
        return self

    @property
    def parent_ref(self) -> ArtifactRef | None:
        """Readable synonym for callers that use explicit ``*_ref`` names."""

        return self.parent


class MutationHypothesis(ImmutableModel):
    """A pre-registered, falsifiable explanation for a candidate patch."""

    evidence_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)] = Field(
        validation_alias=AliasChoices("evidence_refs", "evidence")
    )
    where: NonEmptyStr
    why: NonEmptyStr
    expected_activation: NonEmptyStr
    expected_adherence: NonEmptyStr
    expected_behavior: NonEmptyStr
    expected_benefit: NonEmptyStr
    protected_slices: tuple[NonEmptyStr, ...]
    falsifier: NonEmptyStr
    negative_control: NonEmptyStr
    risks: tuple[NonEmptyStr, ...]

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_evidence_order(
        cls,
        evidence_refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        """Treat cited evidence as a set while retaining duplicate auditing."""

        return tuple(
            sorted(
                evidence_refs,
                key=lambda ref: (ref.sha256, ref.size, ref.media_type),
            )
        )

    @field_validator("protected_slices", "risks")
    @classmethod
    def canonicalize_label_order(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """Give set-like hypothesis labels a stable serialized order."""

        return tuple(sorted(labels))

    @model_validator(mode="after")
    def reject_duplicate_evidence_or_labels(self) -> MutationHypothesis:
        evidence_hashes = [item.sha256 for item in self.evidence_refs]
        if len(evidence_hashes) != len(set(evidence_hashes)):
            raise ValueError("evidence_refs must not contain duplicate artifacts")
        for field_name in ("protected_slices", "risks"):
            labels = getattr(self, field_name)
            if len(labels) != len(set(labels)):
                raise ValueError(f"{field_name} must not contain duplicate entries")
        return self

    @property
    def evidence(self) -> tuple[ArtifactRef, ...]:
        """Short synonym retained for ergonomic diagnosis code."""

        return self.evidence_refs


class CandidateMutation(ImmutableModel):
    """One atomic before/after patch with its pre-registered hypothesis."""

    schema_version: Literal["1"] = "1"
    target_component: NonEmptyStr = Field(
        validation_alias=AliasChoices("target_component", "target")
    )
    before: HarnessComponentRef
    after: HarnessComponentRef
    hypothesis: MutationHypothesis

    @model_validator(mode="after")
    def is_one_non_noop_component_patch(self) -> CandidateMutation:
        if self.before.name != self.after.name:
            raise ValueError("before and after must refer to the same component name")
        if self.before.kind != self.after.kind:
            raise ValueError("before and after must refer to the same component kind")
        if self.target_component != self.before.name:
            raise ValueError("target_component must match the before/after component name")
        if self.before.artifact.sha256 == self.after.artifact.sha256:
            raise ValueError("before and after content must differ; no-op mutations are invalid")
        return self

    @property
    def target(self) -> str:
        """Short synonym for ``target_component``."""

        return self.target_component


def model_content(model: ImmutableModel) -> dict[str, Any]:
    """Return a model's JSON-safe content for explicit hashing/storage code."""

    validated = type(model).model_validate(
        model.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    return validated.model_dump(mode="json", by_alias=False, exclude_none=False)


__all__ = [
    "ArtifactRef",
    "BudgetPolicy",
    "CandidateMutation",
    "ComponentKind",
    "HarnessComponentRef",
    "HarnessManifest",
    "ImmutableModel",
    "MutationHypothesis",
    "model_content",
]
