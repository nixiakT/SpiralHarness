from __future__ import annotations

from dataclasses import dataclass

from spiral_harness.benchmark.base import BenchmarkAdapter
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.evolution.interfaces import Diagnoser, Proposer
from spiral_harness.execution import HarnessExecutor
from spiral_harness.storage import ArtifactStore
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification import PromotionGate, TrialObservation
from spiral_harness.verification.protocol import CandidateVerifier


@dataclass(frozen=True)
class FakeExecution:
    output: str


class FakeBenchmark:
    fingerprint = "fake-benchmark-v1"

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        return (f"{partition.value}-task",)

    def load_task(self, task_ref: str) -> str:
        return task_ref

    def grade(
        self,
        task: str,
        execution: FakeExecution,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation:
        return TrialObservation(
            task_id=task,
            seed=seed,
            harness_id=harness_id,
            score=float(execution.output == task),
            execution_fingerprint=execution_fingerprint,
        )


class FakeExecutor:
    fingerprint = "fake-executor-v1"

    def execute(self, task: str, *, harness: str, seed: int) -> FakeExecution:
        del harness, seed
        return FakeExecution(output=task)


class FakeDiagnoser:
    def diagnose(self, evidence: str) -> tuple[str, ...]:
        return (f"diagnosis:{evidence}",)


class FakeProposer:
    def propose(self, diagnosis: str, *, parent: str) -> tuple[str, ...]:
        return (f"{parent}:{diagnosis}",)


def test_runtime_protocols_accept_structural_implementations(tmp_path) -> None:
    assert isinstance(ArtifactStore(tmp_path / "objects"), ArtifactRepository)
    assert isinstance(FakeBenchmark(), BenchmarkAdapter)
    assert isinstance(FakeExecutor(), HarnessExecutor)
    assert isinstance(FakeDiagnoser(), Diagnoser)
    assert isinstance(FakeProposer(), Proposer)
    assert isinstance(PromotionGate(), CandidateVerifier)


def test_benchmark_grading_is_separate_from_execution() -> None:
    benchmark = FakeBenchmark()
    executor = FakeExecutor()
    task_ref = benchmark.task_roster(ProtocolPartition.EXPLORATION)[0]
    task = benchmark.load_task(task_ref)
    execution = executor.execute(task, harness="candidate", seed=7)

    observation = benchmark.grade(
        task,
        execution,
        harness_id="candidate",
        seed=7,
        execution_fingerprint="paired-context",
    )

    assert not hasattr(execution, "score")
    assert observation.score == 1.0
