"""Trusted GSM8K adapter with deterministic, leakage-resistant partitions."""

from __future__ import annotations

import io
import json
import re
import unicodedata
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import Field, ValidationError, field_validator, model_validator

from spiral_harness.benchmark.datasets import (
    GSM8K_PROVENANCE,
    DatasetIntegrityError,
    DatasetSnapshot,
    read_dataset_snapshot,
)
from spiral_harness.core import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.verification.models import TrialObservation, TrialStatus

GSM8K_ANSWER_PATTERN = r"#### (\-?[0-9\.\,]+)"
GSM8K_ANSWER_RE = re.compile(GSM8K_ANSWER_PATTERN)
GSM8K_ADAPTER_VERSION = "spiral-harness.gsm8k-adapter:v1"
GSM8K_TASK_ID_VERSION = "spiral-harness.gsm8k-task-id:v1"
GSM8K_TEMPLATE_VERSION = "spiral-harness.gsm8k-template-family:v1"
GSM8K_SPLIT_ALGORITHM = "template-family-balanced-sha256-v1"
GSM8K_PINNED_DEFAULT_PARTITION_COUNTS = (
    (ProtocolPartition.EXPLORATION, 5_978),
    (ProtocolPartition.GATE, 1_495),
    (ProtocolPartition.SEALED, 1_319),
)
GSM8K_PINNED_DEFAULT_SPLIT_FINGERPRINT = (
    "2b5678c280095a8435f50f25f9c8d4e43af9530bac9663694b23ede899c52acc"
)
GSM8K_PINNED_DEFAULT_ADAPTER_FINGERPRINT = (
    "7fc22b487fde502c0730265d5844962bacd1a5c98cc6dfab4d8beb962e29a1c2"
)

_NUMBER_RE = re.compile(r"(?<!\w)[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")


class GSM8KDataError(ValueError):
    """A GSM8K source or split violated the trusted data contract."""


class GSM8KGradingError(ValueError):
    """An execution could not be safely bound to its trusted GSM8K task."""


class GSM8KExample(ImmutableModel):
    """Trusted source record; this model must never be sent to a worker."""

    question: NonEmptyStr
    answer: NonEmptyStr

    @field_validator("question", "answer", mode="before")
    @classmethod
    def source_strings_are_nonempty_and_trimmed(cls, value: Any) -> Any:
        if type(value) is not str:
            raise ValueError("GSM8K question and answer must be strings")
        if not value.strip():
            raise ValueError("GSM8K question and answer must not be empty")
        if value != value.strip():
            raise ValueError("GSM8K question and answer must not have outer whitespace")
        return value


class GSM8KTask(ImmutableModel):
    """The complete candidate-facing task: opaque identity plus public question."""

    task_id: NonEmptyStr
    question: NonEmptyStr


class GSM8KSplitConfig(ImmutableModel):
    """Frozen group-aware train split configuration."""

    schema_version: Literal["1"] = "1"
    algorithm: Literal["template-family-balanced-sha256-v1"] = GSM8K_SPLIT_ALGORITHM
    seed: Annotated[int, Field(ge=0, strict=True)] = 8_172_021
    gate_fraction: Annotated[float, Field(gt=0.0, lt=1.0, strict=True)] = 0.2
    normalization: Literal["nfkc-casefold-whitespace-number-placeholder-v1"] = (
        "nfkc-casefold-whitespace-number-placeholder-v1"
    )


DEFAULT_GSM8K_SPLIT_CONFIG = GSM8KSplitConfig()


class GSM8KSourceIdentity(ImmutableModel):
    """Byte identity used to replay a split without storing dataset contents."""

    role: Literal["train", "test"]
    size: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256


class GSM8KTaskManifestEntry(ImmutableModel):
    """Trusted split membership without a source index or answer."""

    task_id: NonEmptyStr
    template_family_sha256: Sha256
    source_role: Literal["train", "test"]


class GSM8KPartitionManifest(ImmutableModel):
    """One deterministic, answer-free task roster in the split manifest."""

    partition: ProtocolPartition
    tasks: Annotated[tuple[GSM8KTaskManifestEntry, ...], Field(min_length=1)]

    @field_validator("tasks")
    @classmethod
    def canonicalize_tasks(
        cls,
        tasks: tuple[GSM8KTaskManifestEntry, ...],
    ) -> tuple[GSM8KTaskManifestEntry, ...]:
        return tuple(sorted(tasks, key=lambda task: task.task_id))

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> GSM8KPartitionManifest:
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("partition task IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class GSM8KSplitManifest(ImmutableModel):
    """Replayable mapping from pinned source bytes to three disjoint rosters."""

    schema_version: Literal["1"] = "1"
    benchmark_version: Literal["gsm8k-main-v1"] = "gsm8k-main-v1"
    sources: Annotated[tuple[GSM8KSourceIdentity, ...], Field(min_length=2, max_length=2)]
    config: GSM8KSplitConfig
    partitions: Annotated[
        tuple[GSM8KPartitionManifest, ...],
        Field(min_length=3, max_length=3),
    ]

    @field_validator("sources")
    @classmethod
    def canonicalize_sources(
        cls,
        sources: tuple[GSM8KSourceIdentity, ...],
    ) -> tuple[GSM8KSourceIdentity, ...]:
        return tuple(sorted(sources, key=lambda source: source.role))

    @field_validator("partitions")
    @classmethod
    def canonicalize_partitions(
        cls,
        partitions: tuple[GSM8KPartitionManifest, ...],
    ) -> tuple[GSM8KPartitionManifest, ...]:
        return tuple(sorted(partitions, key=lambda partition: partition.partition.value))

    @model_validator(mode="after")
    def require_exact_disjoint_partition_contract(self) -> GSM8KSplitManifest:
        source_roles = {source.role for source in self.sources}
        if source_roles != {"train", "test"}:
            raise ValueError("split manifest requires exactly train and test source identities")

        expected = {
            ProtocolPartition.EXPLORATION,
            ProtocolPartition.GATE,
            ProtocolPartition.SEALED,
        }
        actual = {partition.partition for partition in self.partitions}
        if actual != expected:
            raise ValueError("split manifest requires exploration, gate, and sealed partitions")

        all_task_ids: list[str] = []
        families_by_partition: dict[ProtocolPartition, set[str]] = {}
        for partition in self.partitions:
            all_task_ids.extend(task.task_id for task in partition.tasks)
            families_by_partition[partition.partition] = {
                task.template_family_sha256 for task in partition.tasks
            }
            expected_source = "test" if partition.partition is ProtocolPartition.SEALED else "train"
            if any(task.source_role != expected_source for task in partition.tasks):
                raise ValueError(
                    "exploration/gate must map only train and sealed must map only test"
                )
        if len(all_task_ids) != len(set(all_task_ids)):
            raise ValueError("GSM8K partition rosters must be disjoint")
        partitions = tuple(ProtocolPartition)
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                if families_by_partition[left].intersection(families_by_partition[right]):
                    raise ValueError(
                        "a GSM8K template family must not cross protocol partitions: "
                        f"{left.value!r} and {right.value!r}"
                    )
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    def partition(self, partition: ProtocolPartition) -> GSM8KPartitionManifest:
        for manifest in self.partitions:
            if manifest.partition is partition:
                return manifest
        raise KeyError(partition)


@runtime_checkable
class GSM8KExecution(Protocol):
    """Score-free structural result expected from a fixed model runner."""

    @property
    def task_id(self) -> str: ...

    @property
    def seed(self) -> int: ...

    @property
    def harness_id(self) -> str: ...

    @property
    def output_text(self) -> str | None: ...

    @property
    def status(self) -> object: ...

    @property
    def tokens(self) -> int: ...

    @property
    def latency_ms(self) -> float: ...

    @property
    def tool_calls(self) -> int: ...

    @property
    def cost_usd(self) -> float | None: ...

    @property
    def execution_fingerprint(self) -> str: ...


class _ValidatedExecution(ImmutableModel):
    task_id: NonEmptyStr
    seed: Annotated[int, Field(ge=0, strict=True)]
    harness_id: NonEmptyStr
    output_text: str | None
    status: TrialStatus
    tokens: Annotated[int, Field(ge=0, strict=True)]
    latency_ms: Annotated[float, Field(ge=0.0, strict=True)]
    tool_calls: Annotated[int, Field(ge=0, strict=True)]
    cost_usd: Annotated[float, Field(ge=0.0, strict=True)] | None
    execution_fingerprint: Sha256

    @model_validator(mode="after")
    def completed_execution_has_output(self) -> _ValidatedExecution:
        if self.status is TrialStatus.COMPLETED and self.output_text is None:
            raise ValueError("completed GSM8K execution must contain output text")
        return self


class _TrustedTaskRecord(ImmutableModel):
    task: GSM8KTask
    gold_answer: NonEmptyStr
    template_family_sha256: Sha256
    source_role: Literal["train", "test"]


def extract_gsm8k_answer(text: str) -> str | None:
    r"""Apply the upstream-compatible first-marker extraction semantics.

    This intentionally does not parse a number.  It uses exactly
    ``r'#### (\-?[0-9\.\,]+)'``, takes the first match, and removes commas, as
    the official evaluator does.  Therefore spellings such as ``1.0`` and
    ``1`` remain distinct.
    """

    if type(text) is not str:
        raise TypeError("GSM8K answer extraction requires a string")
    match = GSM8K_ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GSM8KDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_gsm8k_snapshot(
    snapshot: DatasetSnapshot,
    *,
    source_label: str,
) -> tuple[GSM8KExample, ...]:
    """Parse only the immutable bytes whose identity was already captured."""

    snapshot = DatasetSnapshot.model_validate(snapshot, strict=True)
    examples: list[GSM8KExample] = []
    seen_questions: dict[str, int] = {}
    try:
        decoded = snapshot.content.decode("utf-8", errors="strict")
        with io.StringIO(decoded, newline="") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise GSM8KDataError(f"blank JSONL record at {source_label}:{line_number}")
                try:
                    value = json.loads(line, object_pairs_hook=_reject_duplicate_object_keys)
                except (json.JSONDecodeError, GSM8KDataError) as error:
                    raise GSM8KDataError(
                        f"malformed JSONL record at {source_label}:{line_number}: {error}"
                    ) from error
                if type(value) is not dict or set(value) != {"question", "answer"}:
                    raise GSM8KDataError(
                        f"record at {source_label}:{line_number} "
                        "must contain exactly question and answer"
                    )
                try:
                    example = GSM8KExample.model_validate(value, strict=True)
                except ValidationError as error:
                    raise GSM8KDataError(
                        f"invalid GSM8K schema at {source_label}:{line_number}: {error}"
                    ) from error
                previous_line = seen_questions.get(example.question)
                if previous_line is not None:
                    raise GSM8KDataError(
                        f"duplicate GSM8K question at {source_label}:{line_number}; "
                        f"first seen at line {previous_line}"
                    )
                if extract_gsm8k_answer(example.answer) is None:
                    raise GSM8KDataError(
                        f"gold answer at {source_label}:{line_number} has no official GSM8K marker"
                    )
                seen_questions[example.question] = line_number
                examples.append(example)
    except UnicodeDecodeError as error:
        raise GSM8KDataError(f"GSM8K source is not strict UTF-8: {source_label}") from error
    if not examples:
        raise GSM8KDataError(f"GSM8K source contains no records: {source_label}")
    return tuple(examples)


def load_gsm8k_jsonl(path: Path) -> tuple[GSM8KExample, ...]:
    """Read once and strictly parse a UTF-8 GSM8K JSONL file."""

    path = Path(path)
    snapshot = read_dataset_snapshot(path)
    return _load_gsm8k_snapshot(snapshot, source_label=str(path))


def normalize_gsm8k_template(question: str) -> str:
    """Normalize a question and replace numeric literals with one placeholder."""

    if type(question) is not str or not question.strip():
        raise ValueError("GSM8K template normalization requires a non-empty string")
    normalized = unicodedata.normalize("NFKC", question).casefold()
    normalized = " ".join(normalized.split())
    return _NUMBER_RE.sub("<number>", normalized)


def gsm8k_template_family_sha256(question: str) -> str:
    """Return the stable identity of a normalized template family."""

    return canonical_sha256(
        {
            "version": GSM8K_TEMPLATE_VERSION,
            "template": normalize_gsm8k_template(question),
        }
    )


def _source_identity(
    snapshot: DatasetSnapshot,
    role: Literal["train", "test"],
) -> GSM8KSourceIdentity:
    if snapshot.size == 0:
        raise GSM8KDataError(f"GSM8K {role} source is empty")
    return GSM8KSourceIdentity(role=role, size=snapshot.size, sha256=snapshot.sha256)


def _task_record(
    example: GSM8KExample,
    *,
    role: Literal["train", "test"],
    source_sha256: str,
) -> _TrustedTaskRecord:
    task_id = "gsm8k-" + canonical_sha256(
        {
            "version": GSM8K_TASK_ID_VERSION,
            "role": role,
            "source_sha256": source_sha256,
            "question": example.question,
        }
    )
    gold = extract_gsm8k_answer(example.answer)
    if gold is None:
        raise GSM8KDataError("trusted GSM8K record has no official gold marker")
    return _TrustedTaskRecord(
        task=GSM8KTask(task_id=task_id, question=example.question),
        gold_answer=gold,
        template_family_sha256=gsm8k_template_family_sha256(example.question),
        source_role=role,
    )


def _family_partition(
    train_records: Sequence[_TrustedTaskRecord],
    config: GSM8KSplitConfig,
) -> tuple[tuple[_TrustedTaskRecord, ...], tuple[_TrustedTaskRecord, ...]]:
    families: dict[str, list[_TrustedTaskRecord]] = {}
    for record in train_records:
        families.setdefault(record.template_family_sha256, []).append(record)
    if len(families) < 2:
        raise GSM8KDataError(
            "group-aware GSM8K splitting requires at least two distinct template families"
        )

    ordered_families = sorted(
        families,
        key=lambda family: canonical_sha256(
            {
                "algorithm": config.algorithm,
                "seed": config.seed,
                "template_family_sha256": family,
            }
        ),
    )
    target_gate_records = len(train_records) * config.gate_fraction
    prefix_sizes: list[int] = []
    running_size = 0
    for family in ordered_families[:-1]:
        running_size += len(families[family])
        prefix_sizes.append(running_size)
    gate_family_count = min(
        range(1, len(ordered_families)),
        key=lambda count: (abs(prefix_sizes[count - 1] - target_gate_records), count),
    )
    gate_families = set(ordered_families[:gate_family_count])

    exploration: list[_TrustedTaskRecord] = []
    gate: list[_TrustedTaskRecord] = []
    for family, records in families.items():
        target = gate if family in gate_families else exploration
        target.extend(records)
    return (
        tuple(sorted(exploration, key=lambda record: record.task.task_id)),
        tuple(sorted(gate, key=lambda record: record.task.task_id)),
    )


def _partition_manifest(
    partition: ProtocolPartition,
    records: Sequence[_TrustedTaskRecord],
) -> GSM8KPartitionManifest:
    return GSM8KPartitionManifest(
        partition=partition,
        tasks=tuple(
            GSM8KTaskManifestEntry(
                task_id=record.task.task_id,
                template_family_sha256=record.template_family_sha256,
                source_role=record.source_role,
            )
            for record in records
        ),
    )


def _normalize_status(value: object) -> TrialStatus:
    if isinstance(value, TrialStatus):
        return value
    raw_value = value.value if isinstance(value, Enum) else value
    if raw_value == "failed":
        return TrialStatus.INFRA_ERROR
    if type(raw_value) is str:
        try:
            return TrialStatus(raw_value)
        except ValueError as error:
            raise GSM8KGradingError(f"unsupported runner status: {raw_value!r}") from error
    raise GSM8KGradingError("runner status must be a recognized string enum")


def _validated_execution(execution: GSM8KExecution | object) -> _ValidatedExecution:
    field_names = (
        "task_id",
        "seed",
        "harness_id",
        "output_text",
        "status",
        "tokens",
        "latency_ms",
        "tool_calls",
        "cost_usd",
        "execution_fingerprint",
    )
    try:
        values = {field_name: getattr(execution, field_name) for field_name in field_names}
    except (AttributeError, RuntimeError) as error:
        raise GSM8KGradingError(
            "runner execution is missing required score-free telemetry"
        ) from error
    values["status"] = _normalize_status(values["status"])
    try:
        return _ValidatedExecution.model_validate(values, strict=True)
    except ValidationError as error:
        raise GSM8KGradingError(f"invalid runner execution telemetry: {error}") from error


class GSM8KBenchmarkAdapter:
    """Trusted GSM8K task roster, split owner, and exact-match grader.

    The adapter must stay in the trusted process.  Workers receive only a
    :class:`GSM8KTask`, whose two fields contain neither the gold answer nor a
    source index or protocol partition.
    """

    __slots__ = (
        "_fingerprint",
        "_partition_records",
        "_records_by_id",
        "_split_manifest",
    )

    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        *,
        split_config: GSM8KSplitConfig = DEFAULT_GSM8K_SPLIT_CONFIG,
        verify_pinned: bool = True,
    ) -> None:
        if type(verify_pinned) is not bool:
            raise TypeError("verify_pinned must be a bool")
        split_config = GSM8KSplitConfig.model_validate(split_config, strict=True)
        train_path = Path(train_path)
        test_path = Path(test_path)
        train_artifact = GSM8K_PROVENANCE.artifact("train") if verify_pinned else None
        test_artifact = GSM8K_PROVENANCE.artifact("test") if verify_pinned else None
        try:
            train_snapshot = read_dataset_snapshot(train_path, artifact=train_artifact)
            test_snapshot = read_dataset_snapshot(test_path, artifact=test_artifact)
        except DatasetIntegrityError as error:
            qualifier = "pinned " if verify_pinned else ""
            raise GSM8KDataError(f"{qualifier}GSM8K source verification failed: {error}") from error

        train_source = _source_identity(train_snapshot, "train")
        test_source = _source_identity(test_snapshot, "test")
        if verify_pinned:
            if train_artifact is None or test_artifact is None:
                raise GSM8KDataError("pinned GSM8K registry artifacts are missing")
            if (
                train_source.size,
                train_source.sha256,
                test_source.size,
                test_source.sha256,
            ) != (
                train_artifact.size,
                train_artifact.sha256,
                test_artifact.size,
                test_artifact.sha256,
            ):
                raise GSM8KDataError("pinned GSM8K snapshot identity changed after verification")
        train_examples = _load_gsm8k_snapshot(
            train_snapshot,
            source_label=str(train_path),
        )
        test_examples = _load_gsm8k_snapshot(
            test_snapshot,
            source_label=str(test_path),
        )
        train_records = tuple(
            _task_record(example, role="train", source_sha256=train_source.sha256)
            for example in train_examples
        )
        sealed_records = tuple(
            sorted(
                (
                    _task_record(example, role="test", source_sha256=test_source.sha256)
                    for example in test_examples
                ),
                key=lambda record: record.task.task_id,
            )
        )
        exploration_records, gate_records = _family_partition(train_records, split_config)

        self._partition_records = {
            ProtocolPartition.EXPLORATION: exploration_records,
            ProtocolPartition.GATE: gate_records,
            ProtocolPartition.SEALED: sealed_records,
        }
        all_records = exploration_records + gate_records + sealed_records
        self._records_by_id = {record.task.task_id: record for record in all_records}
        if len(self._records_by_id) != len(all_records):
            raise GSM8KDataError("GSM8K generated duplicate opaque task IDs")

        self._split_manifest = GSM8KSplitManifest(
            sources=(train_source, test_source),
            config=split_config,
            partitions=tuple(
                _partition_manifest(partition, self._partition_records[partition])
                for partition in ProtocolPartition
            ),
        )
        if verify_pinned and split_config == DEFAULT_GSM8K_SPLIT_CONFIG:
            expected_counts = dict(GSM8K_PINNED_DEFAULT_PARTITION_COUNTS)
            actual_counts = {
                partition: len(self._partition_records[partition])
                for partition in ProtocolPartition
            }
            if actual_counts != expected_counts:
                raise GSM8KDataError(
                    "pinned GSM8K default partition counts drifted: "
                    f"expected={expected_counts!r}, actual={actual_counts!r}"
                )
            if self._split_manifest.fingerprint != GSM8K_PINNED_DEFAULT_SPLIT_FINGERPRINT:
                raise GSM8KDataError(
                    "pinned GSM8K default split fingerprint drifted; "
                    "change the versioned split contract before changing membership"
                )
        self._fingerprint = canonical_sha256(
            {
                "adapter_version": GSM8K_ADAPTER_VERSION,
                "official_answer_pattern": GSM8K_ANSWER_PATTERN,
                "registered_provenance": GSM8K_PROVENANCE.fingerprint,
                "split_manifest": self._split_manifest,
                "sources_are_registered_snapshot": verify_pinned,
            }
        )
        if (
            verify_pinned
            and split_config == DEFAULT_GSM8K_SPLIT_CONFIG
            and self._fingerprint != GSM8K_PINNED_DEFAULT_ADAPTER_FINGERPRINT
        ):
            raise GSM8KDataError(
                "pinned GSM8K default adapter fingerprint drifted; "
                "change the adapter version before changing trusted semantics"
            )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def split_manifest(self) -> GSM8KSplitManifest:
        """Trusted replay artifact; do not expose it to candidate workers."""

        return self._split_manifest

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        if not isinstance(partition, ProtocolPartition):
            raise GSM8KDataError(f"unknown GSM8K protocol partition: {partition!r}")
        try:
            records = self._partition_records[partition]
        except (KeyError, TypeError) as error:
            raise GSM8KDataError(f"unknown GSM8K protocol partition: {partition!r}") from error
        return tuple(record.task.task_id for record in records)

    def load_task(self, task_ref: str) -> GSM8KTask:
        if type(task_ref) is not str or not task_ref:
            raise GSM8KDataError("GSM8K task reference must be a non-empty opaque ID")
        try:
            return self._records_by_id[task_ref].task
        except KeyError as error:
            raise GSM8KDataError(f"unknown GSM8K task reference: {task_ref!r}") from error

    def grade(
        self,
        task: GSM8KTask,
        execution: GSM8KExecution | object,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation:
        """Bind score-free telemetry to a task, then grade inside the trusted plane."""

        try:
            task = GSM8KTask.model_validate(task, strict=True)
        except ValidationError as error:
            raise GSM8KGradingError(f"invalid candidate-facing GSM8K task: {error}") from error
        try:
            record = self._records_by_id[task.task_id]
        except KeyError as error:
            raise GSM8KGradingError(
                "cannot grade a task outside the frozen GSM8K roster"
            ) from error
        if task != record.task:
            raise GSM8KGradingError(
                "candidate-facing GSM8K task content does not match its task ID"
            )

        captured = _validated_execution(execution)
        expected_provenance = (task.task_id, seed, harness_id, execution_fingerprint)
        actual_provenance = (
            captured.task_id,
            captured.seed,
            captured.harness_id,
            captured.execution_fingerprint,
        )
        if actual_provenance != expected_provenance:
            raise GSM8KGradingError("runner execution provenance does not match grading context")

        score: float | None = None
        if captured.status is TrialStatus.COMPLETED:
            output_text = captured.output_text
            if output_text is None:
                raise GSM8KGradingError("completed GSM8K execution lost its output text")
            predicted_answer = extract_gsm8k_answer(output_text)
            score = float(predicted_answer is not None and predicted_answer == record.gold_answer)

        return TrialObservation(
            task_id=task.task_id,
            seed=seed,
            harness_id=harness_id,
            status=captured.status,
            score=score,
            slice_tags=("benchmark:gsm8k",),
            tokens=captured.tokens,
            latency_ms=captured.latency_ms,
            tool_calls=captured.tool_calls,
            cost_usd=captured.cost_usd,
            execution_fingerprint=execution_fingerprint,
        )


__all__ = [
    "DEFAULT_GSM8K_SPLIT_CONFIG",
    "GSM8K_ADAPTER_VERSION",
    "GSM8K_ANSWER_PATTERN",
    "GSM8K_ANSWER_RE",
    "GSM8K_PINNED_DEFAULT_ADAPTER_FINGERPRINT",
    "GSM8K_PINNED_DEFAULT_PARTITION_COUNTS",
    "GSM8K_PINNED_DEFAULT_SPLIT_FINGERPRINT",
    "GSM8K_SPLIT_ALGORITHM",
    "GSM8KBenchmarkAdapter",
    "GSM8KDataError",
    "GSM8KExample",
    "GSM8KExecution",
    "GSM8KGradingError",
    "GSM8KPartitionManifest",
    "GSM8KSourceIdentity",
    "GSM8KSplitConfig",
    "GSM8KSplitManifest",
    "GSM8KTask",
    "GSM8KTaskManifestEntry",
    "extract_gsm8k_answer",
    "gsm8k_template_family_sha256",
    "load_gsm8k_jsonl",
    "normalize_gsm8k_template",
]
