"""Protocol for an untrusted harness execution boundary."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

TaskT_contra = TypeVar("TaskT_contra", contravariant=True)
HarnessT_contra = TypeVar("HarnessT_contra", contravariant=True)
ExecutionT_co = TypeVar("ExecutionT_co", covariant=True)


@runtime_checkable
class HarnessExecutor(Protocol[TaskT_contra, HarnessT_contra, ExecutionT_co]):
    """Execute one materialized harness/task pair without grading authority."""

    @property
    def fingerprint(self) -> str: ...

    def execute(
        self,
        task: TaskT_contra,
        *,
        harness: HarnessT_contra,
        seed: int,
    ) -> ExecutionT_co: ...


__all__ = ["HarnessExecutor"]
