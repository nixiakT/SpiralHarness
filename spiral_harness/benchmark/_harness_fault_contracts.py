"""Typed v4 contracts and frozen multi-surface catalog for HarnessFaultBench."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256

GENERATOR_VERSION = "spiral-harness.harness-fault-generator:v4-multi-surface"
AUTHORITY_VERSION = "spiral-harness.harness-fault-authority:v4"
PARTITION_OPENING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-partition-opening.v4+json"
)
PARTITION_ROSTER_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-partition-roster.v4+json"
)
_SALT_HEX_RE = re.compile(r"^[0-9a-f]{64,}$")


class HarnessFaultAuthorityError(ValueError):
    """An authority opening, commitment, roster, or design invariant failed."""


class FaultSurface(StrEnum):
    PROMPT = "prompt"
    MEMORY = "memory-retrieval"
    TOOL = "tool"
    MIDDLEWARE = "middleware"
    CONTROL_FLOW = "control-flow"
    SKILL = "skill"


class FaultFamily(StrEnum):
    PROMPT_INSTRUCTION_PRECEDENCE = "prompt-instruction-precedence"
    MEMORY_RETRIEVAL_ISOLATION = "memory-retrieval-isolation"
    TOOL_SCHEMA_VALIDATION = "tool-schema-validation"
    TOOL_ROUTING_AUTHORIZATION = "tool-routing-authorization"
    MIDDLEWARE_CANONICALIZATION = "middleware-canonicalization"
    CONTROL_FLOW_GUARD = "control-flow-guard"
    SKILL_SCOPE_ACTIVATION = "skill-scope-activation"


_SURFACE_BY_FAMILY = {
    FaultFamily.PROMPT_INSTRUCTION_PRECEDENCE: FaultSurface.PROMPT,
    FaultFamily.MEMORY_RETRIEVAL_ISOLATION: FaultSurface.MEMORY,
    FaultFamily.TOOL_SCHEMA_VALIDATION: FaultSurface.TOOL,
    FaultFamily.TOOL_ROUTING_AUTHORIZATION: FaultSurface.TOOL,
    FaultFamily.MIDDLEWARE_CANONICALIZATION: FaultSurface.MIDDLEWARE,
    FaultFamily.CONTROL_FLOW_GUARD: FaultSurface.CONTROL_FLOW,
    FaultFamily.SKILL_SCOPE_ACTIVATION: FaultSurface.SKILL,
}


def surface_for_family(family: FaultFamily) -> FaultSurface:
    return _SURFACE_BY_FAMILY[FaultFamily(family)]


class ScenarioRole(StrEnum):
    REPAIRABLE_TARGET = "repairable-target"
    NULL_CONTROL = "null-control"
    UNREPAIRABLE_CONTROL = "unrepairable-control"
    DISTRACTOR_SHIFT_HARD_NEGATIVE = "distractor-shift-hard-negative"


class RouteContext(StrEnum):
    CONTEXT_0 = "context-0"
    CONTEXT_1 = "context-1"
    CONTEXT_X = "context-x"


class RuntimeBranch(StrEnum):
    SAFE = "branch-0"
    LEGACY = "branch-1"


class RepairRuleId(StrEnum):
    """Opaque proposal IDs; descriptive names are trusted-code-only aliases."""

    CONSTANT_LEGACY = "r-13f0a9c2"
    CONSTANT_SAFE = "r-7bd91e40"
    ROUTED_POLICY = "r-c4a82f16"
    PROMPT_MEMORY_PATCH = "r-291f60bd"
    TOOL_PATCH = "r-893e14ac"
    RUNTIME_PATCH = "r-b07c45e1"
    CONTROL_NEUTRAL = "c-50d8b731"

    RULE_00 = "r-13f0a9c2"
    RULE_01 = "r-7bd91e40"
    RULE_02 = "r-c4a82f16"
    CONTROL_00 = "c-50d8b731"


CANDIDATE_RULE_CATALOG = (
    RepairRuleId.CONSTANT_LEGACY,
    RepairRuleId.CONSTANT_SAFE,
    RepairRuleId.ROUTED_POLICY,
    RepairRuleId.PROMPT_MEMORY_PATCH,
    RepairRuleId.TOOL_PATCH,
    RepairRuleId.RUNTIME_PATCH,
)
_CANDIDATE_RULE_SET = frozenset(CANDIDATE_RULE_CATALOG)
PATCH_FAMILIES = {
    RepairRuleId.PROMPT_MEMORY_PATCH: frozenset(
        {
            FaultFamily.PROMPT_INSTRUCTION_PRECEDENCE,
            FaultFamily.MEMORY_RETRIEVAL_ISOLATION,
        }
    ),
    RepairRuleId.TOOL_PATCH: frozenset(
        {
            FaultFamily.TOOL_SCHEMA_VALIDATION,
            FaultFamily.TOOL_ROUTING_AUTHORIZATION,
        }
    ),
    RepairRuleId.RUNTIME_PATCH: frozenset(
        {
            FaultFamily.MIDDLEWARE_CANONICALIZATION,
            FaultFamily.CONTROL_FLOW_GUARD,
            FaultFamily.SKILL_SCOPE_ACTIVATION,
        }
    ),
}


class HarnessFaultSplitConfig(ImmutableModel):
    schema_version: Literal["4"] = "4"
    groups_per_family: Annotated[int, Field(ge=1, strict=True)] = 1
    scenarios_per_group: Literal[4] = 4

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


DEFAULT_HARNESS_FAULT_SPLIT_CONFIG = HarnessFaultSplitConfig()


class HarnessFaultTask(ImmutableModel):
    schema_version: Literal["4"] = "4"
    task_id: NonEmptyStr
    question: NonEmptyStr


class PublicTaskInput(ImmutableModel):
    family: FaultFamily
    context: RouteContext
    primary: NonEmptyStr
    secondary: NonEmptyStr

    model_config = ConfigDict(**{**ImmutableModel.model_config, "str_strip_whitespace": False})


class HiddenScenarioSpec(ImmutableModel):
    schema_version: Literal["4"] = "4"
    scenario_id: NonEmptyStr
    scenario_commitment: Sha256
    task: HarnessFaultTask
    partition: ProtocolPartition
    family: FaultFamily
    surface: FaultSurface
    template_id: NonEmptyStr
    source_id: NonEmptyStr
    group_id: NonEmptyStr
    role: ScenarioRole
    context: RouteContext
    primary: NonEmptyStr
    secondary: NonEmptyStr
    oracle_branch: RuntimeBranch | None
    expected_answer: Literal["MATCH", "DIFFERENT"]
    expected_observable: NonEmptyStr

    model_config = ConfigDict(**{**ImmutableModel.model_config, "str_strip_whitespace": False})

    @model_validator(mode="after")
    def exact_surface_and_repairability(self) -> Self:
        if self.surface is not surface_for_family(self.family):
            raise ValueError("scenario surface differs from its frozen family")
        if (self.oracle_branch is None) is not (self.role is ScenarioRole.UNREPAIRABLE_CONTROL):
            raise ValueError("only unrepairable controls may omit an oracle branch")
        return self


class PartitionRoster(ImmutableModel):
    schema_version: Literal["4"] = "4"
    authority_id: Sha256
    partition: ProtocolPartition
    tasks: Annotated[tuple[HarnessFaultTask, ...], Field(min_length=4)]

    @field_validator("tasks")
    @classmethod
    def canonical_tasks(cls, values: tuple[HarnessFaultTask, ...]) -> tuple[HarnessFaultTask, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.task_id))
        if len({item.task_id for item in ordered}) != len(ordered):
            raise ValueError("partition roster task IDs must be unique")
        return ordered

    @property
    def root(self) -> str:
        return canonical_sha256(
            {"authority_id": self.authority_id, "partition": self.partition, "tasks": self.tasks}
        )


class PartitionCommitment(ImmutableModel):
    authority_id: Sha256
    partition: ProtocolPartition
    config_fingerprint: Sha256
    template_id: NonEmptyStr
    family_count: Annotated[int, Field(ge=2, strict=True)]
    surface_count: Annotated[int, Field(ge=2, strict=True)]
    group_count: Annotated[int, Field(ge=1, strict=True)]
    scenario_count: Annotated[int, Field(ge=4, strict=True)]
    salt_commitment: Sha256
    scenario_root: Sha256
    roster_root: Sha256


class HarnessFaultPublicCommitment(ImmutableModel):
    schema_version: Literal["4"] = "4"
    authority_version: Literal["spiral-harness.harness-fault-authority:v4"] = AUTHORITY_VERSION
    generator_version: Literal["spiral-harness.harness-fault-generator:v4-multi-surface"] = (
        GENERATOR_VERSION
    )
    authority_id: Sha256
    config: HarnessFaultSplitConfig
    partitions: Annotated[tuple[PartitionCommitment, ...], Field(min_length=3, max_length=3)]

    @field_validator("partitions")
    @classmethod
    def canonical_partitions(
        cls, values: tuple[PartitionCommitment, ...]
    ) -> tuple[PartitionCommitment, ...]:
        return tuple(sorted(values, key=lambda item: item.partition.value))

    @model_validator(mode="after")
    def exact_partition_set(self) -> Self:
        if {item.partition for item in self.partitions} != set(ProtocolPartition):
            raise ValueError("public commitment requires all protocol partitions")
        if any(item.authority_id != self.authority_id for item in self.partitions):
            raise ValueError("partition commitment belongs to another authority")
        if any(item.config_fingerprint != self.config.fingerprint for item in self.partitions):
            raise ValueError("partition commitment uses another split config")
        return self

    def partition(self, partition: ProtocolPartition) -> PartitionCommitment:
        for item in self.partitions:
            if item.partition is partition:
                return item
        raise KeyError(partition)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class PublicSearchTaskView(ImmutableModel):
    schema_version: Literal["4"] = "4"
    public_commitment: HarnessFaultPublicCommitment
    exploration_tasks: Annotated[tuple[HarnessFaultTask, ...], Field(min_length=4)]

    @field_validator("exploration_tasks")
    @classmethod
    def canonical_tasks(cls, values: tuple[HarnessFaultTask, ...]) -> tuple[HarnessFaultTask, ...]:
        return tuple(sorted(values, key=lambda item: item.task_id))


class PartitionOpening(ImmutableModel):
    schema_version: Literal["4"] = "4"
    authority_id: Sha256
    partition: ProtocolPartition
    config: HarnessFaultSplitConfig
    salt_hex: NonEmptyStr
    scenario_commitments: Annotated[tuple[Sha256, ...], Field(min_length=4)]
    scenario_root: Sha256
    roster_root: Sha256

    @field_validator("salt_hex")
    @classmethod
    def valid_salt_hex(cls, value: str) -> str:
        if _SALT_HEX_RE.fullmatch(value) is None or len(value) % 2:
            raise ValueError("salt_hex must encode at least 32 bytes")
        return value


class PartitionEvaluationGrant(ImmutableModel):
    schema_version: Literal["4"] = "4"
    public_commitment: HarnessFaultPublicCommitment
    partition: ProtocolPartition
    opening_ref: ArtifactRef
    roster_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_ref_media(self) -> Self:
        if self.opening_ref.media_type != PARTITION_OPENING_MEDIA_TYPE:
            raise ValueError("opening_ref declares the wrong media type")
        if self.roster_ref.media_type != PARTITION_ROSTER_MEDIA_TYPE:
            raise ValueError("roster_ref declares the wrong media type")
        return self


class VerifiedPartitionOpening(ImmutableModel):
    public_commitment_fingerprint: Sha256
    opening_ref: ArtifactRef
    roster_ref: ArtifactRef
    roster: PartitionRoster
    scenarios: Annotated[tuple[HiddenScenarioSpec, ...], Field(min_length=4)]


class HarnessFaultAntiCheatSummary(ImmutableModel):
    schema_version: Literal["4"] = "4"
    scenario_count: Annotated[int, Field(ge=4, strict=True)]
    oracle_behavior_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_constant_branch_selector_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_constant_output_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    best_single_family_patch_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    oracle_minus_best_constant_branch: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    oracle_minus_best_single_family_patch: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]


def is_candidate_rule(rule_id: RepairRuleId) -> bool:
    return rule_id in _CANDIDATE_RULE_SET


__all__ = [
    "AUTHORITY_VERSION",
    "CANDIDATE_RULE_CATALOG",
    "DEFAULT_HARNESS_FAULT_SPLIT_CONFIG",
    "GENERATOR_VERSION",
    "PARTITION_OPENING_MEDIA_TYPE",
    "PARTITION_ROSTER_MEDIA_TYPE",
    "PATCH_FAMILIES",
    "FaultFamily",
    "FaultSurface",
    "HarnessFaultAntiCheatSummary",
    "HarnessFaultAuthorityError",
    "HarnessFaultPublicCommitment",
    "HarnessFaultSplitConfig",
    "HarnessFaultTask",
    "HiddenScenarioSpec",
    "PartitionCommitment",
    "PartitionEvaluationGrant",
    "PartitionOpening",
    "PartitionRoster",
    "PublicSearchTaskView",
    "PublicTaskInput",
    "RepairRuleId",
    "RouteContext",
    "RuntimeBranch",
    "ScenarioRole",
    "VerifiedPartitionOpening",
    "is_candidate_rule",
    "surface_for_family",
]
