"""Trust-closed authority and one-family scenarios for HarnessFaultBench v3.

This module is intentionally a *single-family vertical slice*, not a complete
benchmark.  Public search state contains exploration tasks only.  Gate and
sealed tasks can be reconstructed only from an authority-issued opening ref.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.storage.artifact_store import ArtifactStore

GENERATOR_VERSION = "spiral-harness.harness-fault-generator:v3-one-family"
AUTHORITY_VERSION = "spiral-harness.harness-fault-authority:v3"
PARTITION_OPENING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-partition-opening.v3+json"
)
PARTITION_ROSTER_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-partition-roster.v3+json"
)
_COMMITMENT_DOMAIN = b"spiral-harness:harness-fault-v3-commitment\x00"
_SALT_HEX_RE = re.compile(r"^[0-9a-f]{64,}$")


class HarnessFaultAuthorityError(ValueError):
    """An authority opening, commitment, or roster failed exact replay."""


class FaultFamily(StrEnum):
    """Only the first executable v3 family; do not claim broader coverage."""

    CONDITIONAL_TRIM_CASEFOLD = "conditional-trim-casefold"


class ScenarioRole(StrEnum):
    ACTIVATION_TARGET = "activation-target"
    PROTECTED_HARD_NEGATIVE = "protected-hard-negative"


class ComparisonPolicy(StrEnum):
    CANONICAL = "canonical"
    LITERAL = "literal"


class RuntimeBranch(StrEnum):
    CANONICAL = "branch-0"
    LITERAL = "branch-1"


class RepairRuleId(StrEnum):
    """Opaque public rule identifiers; semantics remain inside trusted code."""

    RULE_00 = "r-13f0a9c2"
    RULE_01 = "r-7bd91e40"
    RULE_02 = "r-c4a82f16"
    CONTROL_00 = "c-50d8b731"


CANDIDATE_RULE_CATALOG = (
    RepairRuleId.RULE_00,
    RepairRuleId.RULE_01,
    RepairRuleId.RULE_02,
)
_CANDIDATE_RULE_SET = frozenset(CANDIDATE_RULE_CATALOG)


class HarnessFaultSplitConfig(ImmutableModel):
    """Public generator shape, frozen into every partition commitment."""

    schema_version: Literal["3"] = "3"
    groups_per_partition: Annotated[int, Field(ge=2, strict=True)] = 2
    scenarios_per_group: Literal[2] = 2

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


DEFAULT_HARNESS_FAULT_SPLIT_CONFIG = HarnessFaultSplitConfig()


class HarnessFaultTask(ImmutableModel):
    """Complete worker-visible task; no family, split, label, or gold field."""

    schema_version: Literal["3"] = "3"
    task_id: NonEmptyStr
    question: NonEmptyStr


class PublicTaskInput(ImmutableModel):
    policy: ComparisonPolicy
    left: NonEmptyStr
    right: NonEmptyStr

    model_config = ConfigDict(**{**ImmutableModel.model_config, "str_strip_whitespace": False})


class HiddenScenarioSpec(ImmutableModel):
    """Private scenario reconstructed only inside a partition evaluator."""

    schema_version: Literal["3"] = "3"
    scenario_id: NonEmptyStr
    scenario_commitment: Sha256
    task: HarnessFaultTask
    partition: ProtocolPartition
    family: Literal[FaultFamily.CONDITIONAL_TRIM_CASEFOLD] = FaultFamily.CONDITIONAL_TRIM_CASEFOLD
    template_id: NonEmptyStr
    source_id: NonEmptyStr
    group_id: NonEmptyStr
    role: ScenarioRole
    policy: ComparisonPolicy
    left: NonEmptyStr
    right: NonEmptyStr
    expected_answer: Literal["MATCH", "DIFFERENT"]
    expected_observable: NonEmptyStr

    model_config = ConfigDict(**{**ImmutableModel.model_config, "str_strip_whitespace": False})


class PartitionRoster(ImmutableModel):
    """Answer-free exact task roster persisted outside public search payloads."""

    schema_version: Literal["3"] = "3"
    authority_id: Sha256
    partition: ProtocolPartition
    tasks: Annotated[tuple[HarnessFaultTask, ...], Field(min_length=2)]

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
            {
                "authority_id": self.authority_id,
                "partition": self.partition,
                "tasks": self.tasks,
            }
        )


class PartitionCommitment(ImmutableModel):
    """Public commitment without task content, salt, labels, or gold."""

    authority_id: Sha256
    partition: ProtocolPartition
    config_fingerprint: Sha256
    template_id: NonEmptyStr
    group_count: Annotated[int, Field(ge=1, strict=True)]
    scenario_count: Annotated[int, Field(ge=2, strict=True)]
    salt_commitment: Sha256
    scenario_root: Sha256
    roster_root: Sha256


class HarnessFaultPublicCommitment(ImmutableModel):
    schema_version: Literal["3"] = "3"
    authority_version: Literal["spiral-harness.harness-fault-authority:v3"] = AUTHORITY_VERSION
    generator_version: Literal["spiral-harness.harness-fault-generator:v3-one-family"] = (
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
    """The only authority payload that may enter the search process."""

    schema_version: Literal["3"] = "3"
    public_commitment: HarnessFaultPublicCommitment
    exploration_tasks: Annotated[tuple[HarnessFaultTask, ...], Field(min_length=2)]

    @field_validator("exploration_tasks")
    @classmethod
    def canonical_tasks(cls, values: tuple[HarnessFaultTask, ...]) -> tuple[HarnessFaultTask, ...]:
        return tuple(sorted(values, key=lambda item: item.task_id))


class PartitionOpening(ImmutableModel):
    """Withheld salt/config opening persisted for exactly one evaluator."""

    schema_version: Literal["3"] = "3"
    authority_id: Sha256
    partition: ProtocolPartition
    config: HarnessFaultSplitConfig
    salt_hex: NonEmptyStr
    scenario_commitments: Annotated[tuple[Sha256, ...], Field(min_length=2)]
    scenario_root: Sha256
    roster_root: Sha256

    @field_validator("salt_hex")
    @classmethod
    def valid_salt_hex(cls, value: str) -> str:
        if _SALT_HEX_RE.fullmatch(value) is None or len(value) % 2:
            raise ValueError("salt_hex must encode at least 32 bytes")
        return value


class PartitionEvaluationGrant(ImmutableModel):
    """Evaluator-only refs; grants are never included in PublicSearchTaskView."""

    schema_version: Literal["3"] = "3"
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
    """Trusted verifier result. Never pass this value to model/search code."""

    public_commitment_fingerprint: Sha256
    opening_ref: ArtifactRef
    roster_ref: ArtifactRef
    roster: PartitionRoster
    scenarios: Annotated[tuple[HiddenScenarioSpec, ...], Field(min_length=2)]


_TEMPLATE_BY_PARTITION = {
    ProtocolPartition.EXPLORATION: "record-reconciliation-template-v1",
    ProtocolPartition.GATE: "identifier-audit-template-v1",
    ProtocolPartition.SEALED: "registry-merge-template-v1",
}
_OPENING_BY_TEMPLATE = {
    "record-reconciliation-template-v1": "Reconcile the following two record keys.",
    "identifier-audit-template-v1": (
        "Audit whether the following two identifiers denote the same value."
    ),
    "registry-merge-template-v1": ("Decide whether these two registry entries should be merged."),
}
_TASK_INSTRUCTION = (
    "For canonical policy, remove leading and trailing Unicode whitespace and then apply "
    "Unicode casefold to both values before comparison. For literal policy, compare the "
    "values exactly as written. Return exactly one JSON object with exactly two string "
    'fields: "answer" (MATCH or DIFFERENT) and "observable". observable must be '
    "left=<JSON string>;right=<JSON string> for the values actually compared. Return no "
    "Markdown or additional prose."
)


def candidate_rule_ids() -> tuple[RepairRuleId, ...]:
    """Return an opaque catalog with no semantic labels."""

    return CANDIDATE_RULE_CATALOG


def is_candidate_rule(rule_id: RepairRuleId) -> bool:
    return rule_id in _CANDIDATE_RULE_SET


def branch_for_rule(rule_id: RepairRuleId, policy: ComparisonPolicy) -> RuntimeBranch:
    """Trusted finite grammar; opaque IDs deliberately carry no semantic names."""

    rule = RepairRuleId(rule_id)
    checked_policy = ComparisonPolicy(policy)
    if rule is RepairRuleId.RULE_01:
        return RuntimeBranch.CANONICAL
    if rule is RepairRuleId.RULE_02 and checked_policy is ComparisonPolicy.CANONICAL:
        return RuntimeBranch.CANONICAL
    return RuntimeBranch.LITERAL


def _observable(left: str, right: str) -> str:
    return (
        "left="
        + json.dumps(left, ensure_ascii=False, separators=(",", ":"))
        + ";right="
        + json.dumps(right, ensure_ascii=False, separators=(",", ":"))
    )


def evaluate_branch(branch: RuntimeBranch, *, left: str, right: str) -> tuple[str, str]:
    checked = RuntimeBranch(branch)
    if checked is RuntimeBranch.CANONICAL:
        compared_left, compared_right = left.strip().casefold(), right.strip().casefold()
    else:
        compared_left, compared_right = left, right
    answer = "MATCH" if compared_left == compared_right else "DIFFERENT"
    return answer, _observable(compared_left, compared_right)


def parse_public_task_input(question: str) -> PublicTaskInput:
    """Strictly parse the fixed public task format used by trusted middleware."""

    if type(question) is not str:
        raise HarnessFaultAuthorityError("question must be a string")
    lines = question.splitlines()
    if len(lines) != 6 or lines[0] not in set(_OPENING_BY_TEMPLATE.values()):
        raise HarnessFaultAuthorityError("question does not use a frozen v3 template")
    if lines[4] or lines[5] != _TASK_INSTRUCTION:
        raise HarnessFaultAuthorityError("question instruction differs from the frozen contract")
    if not lines[1].startswith("COMPARISON_POLICY="):
        raise HarnessFaultAuthorityError("question lacks an exact comparison policy")
    try:
        policy = ComparisonPolicy(lines[1].removeprefix("COMPARISON_POLICY="))
    except ValueError as exc:
        raise HarnessFaultAuthorityError("question comparison policy is invalid") from exc

    def string_value(line: str, prefix: str) -> str:
        if not line.startswith(prefix):
            raise HarnessFaultAuthorityError(f"question lacks exact {prefix[:-1]}")
        try:
            value = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError as exc:
            raise HarnessFaultAuthorityError("question contains invalid string JSON") from exc
        if type(value) is not str or not value:
            raise HarnessFaultAuthorityError("question values must be non-empty strings")
        return value

    return PublicTaskInput(
        policy=policy,
        left=string_value(lines[2], "LEFT="),
        right=string_value(lines[3], "RIGHT="),
    )


def _secret_digest(salt: bytes, value: object) -> str:
    return hashlib.sha256(
        _COMMITMENT_DOMAIN + salt + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


def _question(template_id: str, policy: ComparisonPolicy, left: str, right: str) -> str:
    return (
        f"{_OPENING_BY_TEMPLATE[template_id]}\n"
        f"COMPARISON_POLICY={policy.value}\n"
        f"LEFT={json.dumps(left, ensure_ascii=False)}\n"
        f"RIGHT={json.dumps(right, ensure_ascii=False)}\n\n"
        f"{_TASK_INSTRUCTION}"
    )


def _generate_partition(
    config: HarnessFaultSplitConfig,
    partition: ProtocolPartition,
    salt: bytes,
) -> tuple[HiddenScenarioSpec, ...]:
    template_id = _TEMPLATE_BY_PARTITION[partition]
    words = ("Albatross", "Birch", "Cobalt", "Delta", "Ember", "Fjord", "Garnet")
    scenarios: list[HiddenScenarioSpec] = []
    for index in range(config.groups_per_partition):
        entropy = int(
            _secret_digest(salt, {"partition": partition, "index": index, "axis": "word"})[:16],
            16,
        )
        stem = words[entropy % len(words)] + f"-{index}"
        left, right = f"  {stem.swapcase()}  ", stem.casefold()
        source_id = (
            "source-"
            + _secret_digest(salt, {"partition": partition, "index": index, "axis": "source"})[:24]
        )
        group_id = (
            "group-"
            + _secret_digest(salt, {"partition": partition, "source": source_id, "axis": "group"})[
                :24
            ]
        )
        for role, policy in (
            (ScenarioRole.ACTIVATION_TARGET, ComparisonPolicy.CANONICAL),
            (ScenarioRole.PROTECTED_HARD_NEGATIVE, ComparisonPolicy.LITERAL),
        ):
            question = _question(template_id, policy, left, right)
            task_id = "hfb3-" + _secret_digest(
                salt, {"group": group_id, "role": role, "question": question}
            )
            payload = {
                "generator_version": GENERATOR_VERSION,
                "partition": partition,
                "family": FaultFamily.CONDITIONAL_TRIM_CASEFOLD,
                "template_id": template_id,
                "source_id": source_id,
                "group_id": group_id,
                "role": role,
                "policy": policy,
                "left": left,
                "right": right,
                "task_id": task_id,
                "question": question,
            }
            scenario_id = "scenario-" + _secret_digest(
                salt, {"payload": payload, "axis": "scenario"}
            )
            scenario_commitment = _secret_digest(
                salt, {"scenario_id": scenario_id, "payload": payload}
            )
            oracle_branch = branch_for_rule(RepairRuleId.RULE_02, policy)
            answer, observable = evaluate_branch(oracle_branch, left=left, right=right)
            scenarios.append(
                HiddenScenarioSpec(
                    scenario_id=scenario_id,
                    scenario_commitment=scenario_commitment,
                    task=HarnessFaultTask(task_id=task_id, question=question),
                    partition=partition,
                    template_id=template_id,
                    source_id=source_id,
                    group_id=group_id,
                    role=role,
                    policy=policy,
                    left=left,
                    right=right,
                    expected_answer=answer,
                    expected_observable=observable,
                )
            )
    return tuple(sorted(scenarios, key=lambda item: item.task.task_id))


def _scenario_root(
    authority_id: str,
    partition: ProtocolPartition,
    scenarios: tuple[HiddenScenarioSpec, ...],
) -> str:
    return canonical_sha256(
        {
            "authority_id": authority_id,
            "partition": partition,
            "scenario_commitments": tuple(item.scenario_commitment for item in scenarios),
        }
    )


def _validate_partition(scenarios: tuple[HiddenScenarioSpec, ...]) -> None:
    if not scenarios or len({item.partition for item in scenarios}) != 1:
        raise HarnessFaultAuthorityError("scenario set must cover exactly one partition")
    groups: dict[str, list[HiddenScenarioSpec]] = {}
    for item in scenarios:
        groups.setdefault(item.group_id, []).append(item)
    for group in groups.values():
        if len(group) != 2 or {item.role for item in group} != set(ScenarioRole):
            raise HarnessFaultAuthorityError("every group requires target and hard negative")
    answers = tuple(item.expected_answer for item in scenarios)
    if answers.count("MATCH") != answers.count("DIFFERENT"):
        raise HarnessFaultAuthorityError("partition labels are not balanced")


def verify_partition_opening(
    store: ArtifactStore,
    grant: PartitionEvaluationGrant,
) -> VerifiedPartitionOpening:
    """Recompute salt/config/scenarios/root/roster and reject any cross-authority mix."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked = PartitionEvaluationGrant.model_validate(grant, strict=True)
    try:
        opening = store.get_json(checked.opening_ref, PartitionOpening)
        roster = store.get_json(checked.roster_ref, PartitionRoster)
    except Exception as exc:
        raise HarnessFaultAuthorityError("partition opening or roster cannot be verified") from exc
    commitment = checked.public_commitment.partition(checked.partition)
    if (
        opening.authority_id != checked.public_commitment.authority_id
        or roster.authority_id != checked.public_commitment.authority_id
        or opening.partition is not checked.partition
        or roster.partition is not checked.partition
        or opening.config != checked.public_commitment.config
    ):
        raise HarnessFaultAuthorityError("opening/roster belongs to another authority or partition")
    salt = bytes.fromhex(opening.salt_hex)
    if hashlib.sha256(salt).hexdigest() != commitment.salt_commitment:
        raise HarnessFaultAuthorityError("opening salt does not match public commitment")
    scenarios = _generate_partition(opening.config, checked.partition, salt)
    _validate_partition(scenarios)
    expected_roster = PartitionRoster(
        authority_id=opening.authority_id,
        partition=checked.partition,
        tasks=tuple(item.task for item in scenarios),
    )
    root = _scenario_root(opening.authority_id, checked.partition, scenarios)
    if (
        opening.scenario_commitments != tuple(item.scenario_commitment for item in scenarios)
        or opening.scenario_root != root
        or commitment.scenario_root != root
        or opening.roster_root != expected_roster.root
        or commitment.roster_root != expected_roster.root
        or roster != expected_roster
        or commitment.scenario_count != len(scenarios)
        or commitment.group_count != opening.config.groups_per_partition
        or commitment.config_fingerprint != opening.config.fingerprint
    ):
        raise HarnessFaultAuthorityError("partition opening/root/roster replay mismatch")
    return VerifiedPartitionOpening(
        public_commitment_fingerprint=checked.public_commitment.fingerprint,
        opening_ref=checked.opening_ref,
        roster_ref=checked.roster_ref,
        roster=roster,
        scenarios=scenarios,
    )


__all__ = [
    "AUTHORITY_VERSION",
    "CANDIDATE_RULE_CATALOG",
    "DEFAULT_HARNESS_FAULT_SPLIT_CONFIG",
    "GENERATOR_VERSION",
    "PARTITION_OPENING_MEDIA_TYPE",
    "PARTITION_ROSTER_MEDIA_TYPE",
    "ComparisonPolicy",
    "FaultFamily",
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
    "RuntimeBranch",
    "ScenarioRole",
    "VerifiedPartitionOpening",
    "branch_for_rule",
    "candidate_rule_ids",
    "evaluate_branch",
    "is_candidate_rule",
    "parse_public_task_input",
    "verify_partition_opening",
]
