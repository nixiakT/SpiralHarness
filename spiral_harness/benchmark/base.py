"""Trusted benchmark adapter contract."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.verification.models import TrialObservation

TaskRefT = TypeVar("TaskRefT")
TaskT = TypeVar("TaskT")
ExecutionT = TypeVar("ExecutionT")


@runtime_checkable
class BenchmarkAdapter(Protocol[TaskRefT, TaskT, ExecutionT]):
    """Load a frozen roster and grade untrusted execution output.

    The execution worker never receives this adapter. In particular, ``grade``
    belongs to the trusted plane and must not be callable by candidate code.
    """

    @property
    def fingerprint(self) -> str: ...

    def task_roster(self, partition: ProtocolPartition) -> tuple[TaskRefT, ...]: ...

    def load_task(self, task_ref: TaskRefT) -> TaskT: ...

    def grade(
        self,
        task: TaskT,
        execution: ExecutionT,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation: ...


__all__ = ["BenchmarkAdapter"]
