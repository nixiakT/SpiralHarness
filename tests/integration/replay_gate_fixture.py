from __future__ import annotations

from dataclasses import dataclass

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.evolution.replay_setup import REPLAY_BENCHMARK_FINGERPRINT
from spiral_harness.evolution.replay_study import ReplayStudyExecution
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import BackendResponse, BackendTokenUsage, FrozenModelSpec
from spiral_harness.execution.model import FixedModelRunner, ReplayBackend
from spiral_harness.execution.schedule import EvaluationBatchSchedule
from spiral_harness.experiments.baseline_gate_runner import (
    TrustedBaselineGateRunner,
    baseline_gate_attempt_budget,
)
from spiral_harness.experiments.baselines import BaselineKind, BaselineStudyPlan
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.mechanism import ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE
from spiral_harness.verification.models import (
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
    TrialStatus,
)


@dataclass(frozen=True, slots=True)
class ReplayGateTask:
    task_id: str
    question: str


class ReplayGateAdapter:
    @property
    def fingerprint(self) -> str:
        return REPLAY_BENCHMARK_FINGERPRINT

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        if partition is ProtocolPartition.GATE:
            return ("gate-1",)
        if partition is ProtocolPartition.EXPLORATION:
            return ("exploration-1",)
        return ("sealed-1",)

    def load_task(self, task_ref: str) -> ReplayGateTask:
        return ReplayGateTask(task_id=task_ref, question=f"Answer fixture task {task_ref}.")

    def grade(
        self,
        task: ReplayGateTask,
        execution: object,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation:
        status = execution.status
        completed = getattr(status, "value", status) == "completed"
        return TrialObservation(
            task_id=task.task_id,
            seed=seed,
            harness_id=harness_id,
            status=TrialStatus.COMPLETED if completed else TrialStatus.INFRA_ERROR,
            score=1.0 if completed else 0.0,
            tokens=execution.tokens,
            latency_ms=execution.latency_ms,
            tool_calls=execution.tool_calls,
            cost_usd=execution.cost_usd,
            execution_fingerprint=execution_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class GateBoundary:
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    gate_split_ref: ArtifactRef


def gate_boundary(execution: ReplayStudyExecution) -> GateBoundary:
    return GateBoundary(
        protocol=execution.fixture.protocol,
        protocol_ref=execution.fixture.protocol_ref,
        gate_split_ref=next(
            split.manifest_ref
            for split in execution.fixture.protocol.splits
            if split.partition is ProtocolPartition.GATE
        ),
    )


def foreign_gate_boundary(execution: ReplayStudyExecution) -> GateBoundary:
    store = execution.fixture.store
    foreign_gate_ref = store.put_json(
        {"fixture": "foreign-baseline-gate", "partition": "gate", "task_ids": ["gate-1"]}
    )
    foreign_splits = tuple(
        ProtocolSplit(partition=split.partition, manifest_ref=foreign_gate_ref)
        if split.partition is ProtocolPartition.GATE
        else split
        for split in execution.fixture.protocol.splits
    )
    foreign_protocol = execution.fixture.protocol.model_copy(update={"splits": foreign_splits})
    foreign_protocol_ref = store.put_json(
        foreign_protocol,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
    )
    return GateBoundary(
        protocol=foreign_protocol,
        protocol_ref=foreign_protocol_ref,
        gate_split_ref=foreign_gate_ref,
    )


def replay_gate_closure(
    execution: ReplayStudyExecution,
    boundary: GateBoundary | None = None,
) -> ArtifactRef:
    store = execution.fixture.store
    selected_boundary = gate_boundary(execution) if boundary is None else boundary
    plan = execution.fixture.store.get_json(
        execution.fixture.baseline_study_plan_ref,
        BaselineStudyPlan,
    )
    candidate_refs = {
        kind: _candidate_ref(store, baseline=kind, harness_ref=execution.fixture.seed_harness_ref)
        for kind in BaselineKind
    }
    candidate_harness_refs = {
        kind: _candidate_harness_ref(execution, baseline=kind) for kind in BaselineKind
    }
    mechanism_refs = {
        kind: _mechanism_ref(
            execution,
            baseline=kind,
            candidate_ref=candidate_refs[kind],
            harness_ref=candidate_harness_refs[kind],
        )
        for kind in BaselineKind
    }
    return (
        TrustedBaselineGateRunner(
            store,
            adapter=ReplayGateAdapter(),
            gate_batch_service=execution.fixture.gate_batch_service,
        )
        .execute_study(
            plan,
            query=0,
            protocol_ref=selected_boundary.protocol_ref,
            protocol=selected_boundary.protocol,
            candidate_refs=candidate_refs,
            parent_harness_ref=execution.fixture.seed_harness_ref,
            candidate_harness_refs=candidate_harness_refs,
            gate_split_ref=selected_boundary.gate_split_ref,
            mechanism_evidence_refs=mechanism_refs,
            task_ids=("gate-1",),
            token_ceiling_per_attempt=32,
            runner_factories={kind: _runner_factory(execution, kind=kind) for kind in BaselineKind},
        )
        .closure_ref
    )


def _candidate_ref(
    repository: ArtifactRepository,
    *,
    baseline: BaselineKind,
    harness_ref: ArtifactRef,
) -> ArtifactRef:
    return repository.put_json(
        {"baseline": baseline.value, "harness": harness_ref.sha256},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )


def _mechanism_ref(
    execution: ReplayStudyExecution,
    *,
    baseline: BaselineKind,
    candidate_ref: ArtifactRef,
    harness_ref: ArtifactRef,
) -> ArtifactRef:
    source_ref = execution.fixture.store.put_json(
        {"mechanism": "baseline gate acceptance fixture", "baseline": baseline.value}
    )
    evidence = execution.fixture.mechanism_evidence_service.create(
        protocol_ref=execution.fixture.protocol_ref,
        protocol=execution.fixture.protocol,
        candidate_ref=candidate_ref,
        candidate_harness_ref=harness_ref,
        source_refs=(source_ref,),
        evidence=MechanismEvidence(
            candidate_harness_id=harness_ref.sha256,
            checks=(
                MechanismCheck(
                    name="baseline-gate-acceptance-fixture",
                    passed=True,
                    evidence_refs=(source_ref.sha256,),
                ),
            ),
        ),
    )
    return execution.fixture.store.put_json(
        evidence,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )


def _candidate_harness_ref(
    execution: ReplayStudyExecution,
    *,
    baseline: BaselineKind,
) -> ArtifactRef:
    prompt_ref = execution.fixture.store.put_bytes(
        f"Answer the task directly.\nReplay GATE candidate: {baseline.value}.\n".encode(),
        media_type="text/plain; charset=utf-8",
    )
    harness = HarnessManifest(
        model_fingerprint=execution.fixture.model_spec.model_fingerprint,
        runtime_fingerprint=execution.fixture.model_spec.runtime_fingerprint,
        trusted_plane_version="fixture-trusted-plane-v1",
        parent=execution.fixture.seed_harness_ref,
        components=(
            HarnessComponentRef(
                name="system_prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
    )
    return execution.fixture.store.put_json(
        harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )


def _runner_factory(
    execution: ReplayStudyExecution,
    *,
    kind: BaselineKind,
):
    def factory(schedule: EvaluationBatchSchedule) -> tuple[FixedModelRunner, AttemptLedger]:
        backend = ReplayBackend(
            fingerprint=execution.fixture.model_spec.backend_fingerprint,
            default_response=BackendResponse(
                output="ok",
                usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
            ),
        )
        ledger = AttemptLedger(
            execution.fixture.store,
            ledger_id=f"accepted-baseline-gate-{kind.value}-{schedule.search_runs[0]}",
            budget=baseline_gate_attempt_budget(schedule),
        )
        runner = FixedModelRunner(
            spec=FrozenModelSpec.model_validate(execution.fixture.model_spec, strict=True),
            backend=backend,
            attempt_ledger=ledger,
        )
        return runner, ledger

    return factory
