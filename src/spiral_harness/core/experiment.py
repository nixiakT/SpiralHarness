"""Frozen protocol and experiment contracts for reproducible harness search."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator

from spiral_harness.core.models import (
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)

_MIME_TOKEN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_MEDIA_TYPE_RE = re.compile(rf"^{_MIME_TOKEN}/{_MIME_TOKEN}$")

# The M0.2 trusted terminal-decision service implements exactly this gate.
# Protocol artifacts persist the value even when a Python caller accepts the
# schema default, so an older artifact that omits it is not canonical.
PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT = (
    "spiral-harness.verification.promotion-gate:paired-bootstrap:v1"
)
PROTOCOL_MANIFEST_MEDIA_TYPE = "application/vnd.spiral-harness.protocol-manifest.v2+json"
EXPERIMENT_MANIFEST_MEDIA_TYPE = "application/vnd.spiral-harness.experiment-manifest.v1+json"


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


class ProtocolPartition(StrEnum):
    """Information-flow partitions owned by the trusted verification plane."""

    EXPLORATION = "exploration"
    GATE = "gate"
    SEALED = "sealed"


class ProtocolSplit(ImmutableModel):
    """Bind one protocol partition to its immutable split manifest."""

    partition: ProtocolPartition
    manifest_ref: ArtifactRef = Field(
        validation_alias=AliasChoices("manifest_ref", "manifest", "split_ref", "ref")
    )

    @model_validator(mode="after")
    def split_manifest_is_json(self) -> ProtocolSplit:
        _require_json_ref(self.manifest_ref, field_name="manifest_ref")
        return self


class MutationPolicy(ImmutableModel):
    """Allowlist the component artifacts an optimizer may replace.

    M1 starts with prompt-only mutation.  ``None`` for an optional allowlist
    means unrestricted within the allowed component kinds; an empty tuple is
    rejected because it would describe a policy that cannot authorize any
    candidate.
    """

    allowed_kinds: Annotated[tuple[ComponentKind, ...], Field(min_length=1)] = Field(
        default=(ComponentKind.PROMPT,),
        validation_alias=AliasChoices("allowed_kinds", "component_kinds", "kinds"),
    )
    allowed_component_names: tuple[NonEmptyStr, ...] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "allowed_component_names",
            "component_names",
        ),
    )
    allowed_media_types: tuple[NonEmptyStr, ...] | None = Field(
        default=("text/plain",),
        validation_alias=AliasChoices("allowed_media_types", "media_types"),
    )
    max_artifact_size_bytes: Annotated[int, Field(ge=0, strict=True)] | None = Field(
        default=65_536,
        validation_alias=AliasChoices(
            "max_artifact_size_bytes",
            "max_artifact_bytes",
            "max_size_bytes",
        ),
    )

    @field_validator("allowed_kinds")
    @classmethod
    def canonicalize_kinds(
        cls,
        kinds: tuple[ComponentKind, ...],
    ) -> tuple[ComponentKind, ...]:
        return tuple(sorted(kinds, key=lambda kind: kind.value))

    @field_validator("allowed_component_names")
    @classmethod
    def canonicalize_optional_allowlist(
        cls,
        values: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("an explicit mutation allowlist must not be empty")
        return tuple(sorted(values))

    @field_validator("allowed_media_types")
    @classmethod
    def canonicalize_media_type_allowlist(
        cls,
        values: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("an explicit mutation allowlist must not be empty")
        normalized: list[str] = []
        for value in values:
            base_type = value.partition(";")[0].strip().lower()
            if _MEDIA_TYPE_RE.fullmatch(base_type) is None:
                raise ValueError(f"invalid media type in mutation policy: {value!r}")
            normalized.append(base_type)
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def reject_duplicate_allowlist_entries(self) -> MutationPolicy:
        fields = (
            ("allowed_kinds", self.allowed_kinds),
            ("allowed_component_names", self.allowed_component_names),
            ("allowed_media_types", self.allowed_media_types),
        )
        for field_name, values in fields:
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicate entries")
        return self


class ProtocolManifest(ImmutableModel):
    """Freeze every trusted input that can affect an evaluation result."""

    schema_version: Literal["2"] = "2"
    benchmark_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("benchmark_fingerprint", "benchmark")
    )
    splits: Annotated[tuple[ProtocolSplit, ...], Field(min_length=2, max_length=3)] = Field(
        validation_alias=AliasChoices("splits", "split_manifests")
    )
    model_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("model_fingerprint", "model")
    )
    inference_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("inference_fingerprint", "inference")
    )
    runtime_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("runtime_fingerprint", "runtime")
    )
    sandbox_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("sandbox_fingerprint", "sandbox")
    )
    capability_policy_ref: ArtifactRef = Field(
        validation_alias=AliasChoices("capability_policy_ref", "capability_policy")
    )
    grader_fingerprint: NonEmptyStr = Field(
        validation_alias=AliasChoices("grader_fingerprint", "grader")
    )
    gate_batch_attestor_id: Sha256 = Field(
        validation_alias=AliasChoices(
            "gate_batch_attestor_id",
            "gate_batch_attestor_fingerprint",
        )
    )
    mechanism_evidence_attestor_id: Sha256 = Field(
        validation_alias=AliasChoices(
            "mechanism_evidence_attestor_id",
            "mechanism_evidence_attestor_fingerprint",
        )
    )
    gate_config_ref: ArtifactRef = Field(
        validation_alias=AliasChoices("gate_config_ref", "gate_config")
    )
    gate_implementation_fingerprint: NonEmptyStr = PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT
    trusted_plane_version: NonEmptyStr = Field(
        validation_alias=AliasChoices(
            "trusted_plane_version",
            "trusted_plane_fingerprint",
            "trusted_plane",
        )
    )
    budget: BudgetPolicy

    @field_validator("splits")
    @classmethod
    def canonicalize_split_order(
        cls,
        splits: tuple[ProtocolSplit, ...],
    ) -> tuple[ProtocolSplit, ...]:
        return tuple(sorted(splits, key=lambda split: split.partition.value))

    @model_validator(mode="after")
    def require_disjoint_protocol_partitions(self) -> ProtocolManifest:
        partitions = [split.partition for split in self.splits]
        if len(partitions) != len(set(partitions)):
            raise ValueError("split partitions must be unique")

        required = {ProtocolPartition.EXPLORATION, ProtocolPartition.GATE}
        missing = required.difference(partitions)
        if missing:
            joined = ", ".join(sorted(partition.value for partition in missing))
            raise ValueError(f"protocol is missing required split partitions: {joined}")

        split_hashes = [split.manifest_ref.sha256 for split in self.splits]
        if len(split_hashes) != len(set(split_hashes)):
            raise ValueError("split partitions must refer to distinct manifests")
        _require_json_ref(self.gate_config_ref, field_name="gate_config_ref")
        _require_json_ref(self.capability_policy_ref, field_name="capability_policy_ref")
        if self.budget.max_evaluations is None:
            raise ValueError("protocol budget must cap max_evaluations")
        return self


class ExperimentManifest(ImmutableModel):
    """Freeze one search plan before either harness arm is executed."""

    schema_version: Literal["1"] = "1"
    protocol_ref: ArtifactRef = Field(validation_alias=AliasChoices("protocol_ref", "protocol"))
    seed_harness_ref: ArtifactRef = Field(
        validation_alias=AliasChoices("seed_harness_ref", "seed_harness")
    )
    mutation_policy: MutationPolicy = Field(default_factory=MutationPolicy)
    objective: NonEmptyStr
    baselines: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    stopping: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)] = Field(
        validation_alias=AliasChoices("stopping", "stopping_criteria")
    )
    search_budget: BudgetPolicy

    @field_validator("baselines", "stopping")
    @classmethod
    def canonicalize_set_like_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(values))

    @model_validator(mode="after")
    def reject_duplicate_set_like_labels(self) -> ExperimentManifest:
        for field_name in ("baselines", "stopping"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicate entries")
        if self.search_budget.max_evaluations is None:
            raise ValueError("search_budget must cap max_evaluations")
        if self.protocol_ref.media_type != PROTOCOL_MANIFEST_MEDIA_TYPE:
            raise ValueError("protocol_ref must declare the exact protocol manifest v2 media type")
        _require_json_ref(self.seed_harness_ref, field_name="seed_harness_ref")
        return self


class CandidateManifest(ImmutableModel):
    """Freeze one nominated atomic child and every input to its evaluation."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    child_harness_ref: ArtifactRef
    mutation_ref: ArtifactRef
    evidence_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    evaluation_plan_ref: ArtifactRef

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_evidence_refs(
        cls,
        refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(refs, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("evidence_refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def parent_and_child_are_distinct(self) -> CandidateManifest:
        for field_name in (
            "experiment_ref",
            "parent_harness_ref",
            "child_harness_ref",
            "mutation_ref",
            "evaluation_plan_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)
        if self.parent_harness_ref.sha256 == self.child_harness_ref.sha256:
            raise ValueError("candidate child_harness_ref must differ from its parent")
        return self


__all__ = [
    "EXPERIMENT_MANIFEST_MEDIA_TYPE",
    "PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT",
    "PROTOCOL_MANIFEST_MEDIA_TYPE",
    "CandidateManifest",
    "ExperimentManifest",
    "MutationPolicy",
    "ProtocolManifest",
    "ProtocolPartition",
    "ProtocolSplit",
]
