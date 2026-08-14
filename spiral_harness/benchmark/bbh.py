"""Trusted development adapter for one pinned BIG-Bench Hard task."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import Field, ValidationError, field_validator

from spiral_harness.benchmark.datasets import (
    BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE,
    DatasetIntegrityError,
    read_dataset_snapshot,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ImmutableModel, NonEmptyStr
from spiral_harness.verification.models import TrialObservation, TrialStatus

BBH_TASK_NAME = "logical_deduction_seven_objects"
BBH_ADAPTER_VERSION = "spiral-harness.bbh-logical-deduction-seven:v1"
_ANSWER_RE = re.compile(r"(?im)^\s*(?:the answer is\s*)?(\([A-G]\))\s*[.!]?\s*$")


class BBHDataError(ValueError):
    """Pinned BBH data violated the development adapter contract."""


class BBHGradingError(ValueError):
    """Execution evidence could not be joined to its trusted task."""


class BBHExample(ImmutableModel):
    input: NonEmptyStr
    target: Annotated[str, Field(pattern=r"^\([A-G]\)$")]

    @field_validator("input", "target", mode="before")
    @classmethod
    def strings_are_trimmed(cls, value: Any) -> Any:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("BBH strings must be non-empty and trimmed")
        return value


class BBHTask(ImmutableModel):
    task_id: NonEmptyStr
    question: NonEmptyStr


class BBHExecution(Protocol):
    task_id: str
    seed: int
    harness_id: str
    output_text: str | None
    status: object
    tokens: int
    latency_ms: float
    tool_calls: int
    cost_usd: float | None
    execution_fingerprint: str


class _Record(ImmutableModel):
    task: BBHTask
    target: str


def extract_bbh_answer(text: str) -> str | None:
    """Extract only a standalone final multiple-choice marker."""

    if type(text) is not str:
        raise TypeError("BBH answer extraction requires a string")
    matches = _ANSWER_RE.findall(text)
    return None if not matches else matches[-1].upper()


class BBHLogicalDeductionSevenAdapter:
    """Pinned public BBH task adapter; development-only, with no sealed split."""

    def __init__(self, path: Path, *, verify_pinned: bool = True) -> None:
        artifact = (
            BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE.artifact("development")
            if verify_pinned
            else None
        )
        try:
            snapshot = read_dataset_snapshot(Path(path), artifact=artifact)
        except DatasetIntegrityError as error:
            raise BBHDataError(f"BBH source verification failed: {error}") from error
        try:
            value = json.loads(snapshot.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BBHDataError("BBH source is not strict UTF-8 JSON") from error
        if type(value) is not dict or set(value) != {"canary", "examples"}:
            raise BBHDataError("BBH source must contain exactly canary and examples")
        if type(value["canary"]) is not str or type(value["examples"]) is not list:
            raise BBHDataError("BBH source has invalid top-level fields")
        records: list[_Record] = []
        for index, raw in enumerate(value["examples"]):
            try:
                example = BBHExample.model_validate(raw, strict=True)
            except ValidationError as error:
                raise BBHDataError(f"invalid BBH example {index}: {error}") from error
            task_id = "bbh-logical7-" + canonical_sha256(
                {"version": BBH_ADAPTER_VERSION, "source": snapshot.sha256, "input": example.input}
            )
            records.append(
                _Record(
                    task=BBHTask(task_id=task_id, question=example.input), target=example.target
                )
            )
        if not records or len({item.task.task_id for item in records}) != len(records):
            raise BBHDataError("BBH source is empty or generated duplicate task IDs")
        self._records = tuple(sorted(records, key=lambda item: item.task.task_id))
        self._by_id = {item.task.task_id: item for item in self._records}
        self.fingerprint = canonical_sha256(
            {
                "version": BBH_ADAPTER_VERSION,
                "source_sha256": snapshot.sha256,
                "count": len(records),
            }
        )

    def task_roster(
        self,
        partition: ProtocolPartition = ProtocolPartition.EXPLORATION,
    ) -> tuple[str, ...]:
        """Return the public development roster only as exploration data.

        BBH publishes this task as one public development file.  Treating it as
        a gate or sealed partition would fabricate an information-flow boundary
        that the upstream data does not provide, so every other value fails
        closed.  The default preserves the explicitly non-reportable smoke path;
        protocol-driven callers should always pass the enum value.
        """

        if partition is not ProtocolPartition.EXPLORATION:
            raise BBHDataError(
                "BBH logical-deduction-seven exposes only the exploration partition; "
                f"gate, sealed, and unknown partitions are unavailable: {partition!r}"
            )
        return tuple(item.task.task_id for item in self._records)

    def load_task(self, task_id: str) -> BBHTask:
        try:
            return self._by_id[task_id].task
        except KeyError as error:
            raise KeyError(f"unknown BBH task ID: {task_id!r}") from error

    def grade(
        self,
        task: BBHTask,
        execution: BBHExecution,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation:
        checked_task = BBHTask.model_validate(task, strict=True)
        record = self._by_id.get(checked_task.task_id)
        if record is None or record.task != checked_task:
            raise BBHGradingError("task does not match the trusted BBH roster")
        if (
            execution.task_id != checked_task.task_id
            or execution.harness_id != harness_id
            or execution.seed != seed
        ):
            raise BBHGradingError("execution coordinates differ from the grading request")
        if execution.execution_fingerprint != execution_fingerprint:
            raise BBHGradingError("execution fingerprint differs from the grading request")
        raw_status = (
            execution.status.value if isinstance(execution.status, Enum) else execution.status
        )
        status = TrialStatus.INFRA_ERROR if raw_status == "failed" else TrialStatus(raw_status)
        answer = (
            extract_bbh_answer(execution.output_text or "")
            if status is TrialStatus.COMPLETED
            else None
        )
        return TrialObservation(
            task_id=checked_task.task_id,
            seed=seed,
            harness_id=harness_id,
            status=status,
            score=(1.0 if answer == record.target else 0.0)
            if status is TrialStatus.COMPLETED
            else None,
            slice_tags=("benchmark:bbh", "task:logical_deduction_seven_objects"),
            tokens=execution.tokens,
            latency_ms=execution.latency_ms,
            tool_calls=execution.tool_calls,
            cost_usd=execution.cost_usd,
            execution_fingerprint=execution_fingerprint,
        )


__all__ = ["BBHLogicalDeductionSevenAdapter", "BBHTask", "extract_bbh_answer"]
