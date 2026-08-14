"""Execute the non-reportable replay study through accepted baseline GATE evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.evolution.replay_gate_result import (
    ReplayStudyGateResult,
    publish_replay_study_gate_result,
)
from spiral_harness.evolution.replay_setup import (
    PROMPT_COMPONENT_NAME,
    PROMPT_MEDIA_TYPE,
    REPLAY_BENCHMARK_FINGERPRINT,
)
from spiral_harness.evolution.replay_study import (
    ReplayStudyExecution,
    run_non_reportable_replay_study,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import BackendResponse, BackendTokenUsage, FrozenModelSpec
from spiral_harness.execution.model import FixedModelRunner, ReplayBackend
from spiral_harness.execution.schedule import EvaluationBatchSchedule
from spiral_harness.experiments.baseline_gate_acceptance import (
    BaselineGateStudyAcceptance,
    publish_baseline_gate_study_acceptance,
)
from spiral_harness.experiments.baseline_gate_runner import (
    BaselineGateStudyExecution,
    TrustedBaselineGateRunner,
    baseline_gate_attempt_budget,
)
from spiral_harness.experiments.baselines import (
    LEGACY_BASELINE_KINDS,
    BaselineKind,
    BaselineStudyPlan,
)
from spiral_harness.verification.mechanism import ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE
from spiral_harness.verification.models import (
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
    TrialStatus,
)


@dataclass(frozen=True, slots=True)
class ReplayGateTask:
    """One score-free task exposed by the deterministic replay GATE adapter."""

    task_id: str
    question: str


class ReplayGateAdapter:
    """Trusted adapter for the non-reportable replay GATE slice."""

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
class ReplayGateExecution:
    """Live replay execution plus every published post-barrier GATE artifact."""

    study: ReplayStudyExecution
    gate: BaselineGateStudyExecution
    baseline_gate_acceptance_ref: ArtifactRef
    baseline_gate_acceptance: BaselineGateStudyAcceptance
    result_ref: ArtifactRef
    result: ReplayStudyGateResult
    non_reportable_fixture: Literal[True] = True


@dataclass(frozen=True, slots=True)
class ReplayGateBoundary:
    """Protocol coordinates consumed by one replay baseline GATE closure."""

    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    gate_split_ref: ArtifactRef


def run_non_reportable_replay_gate_result(root: str | Path) -> ReplayGateExecution:
    """Run the complete deterministic replay GATE acceptance path."""

    study = run_non_reportable_replay_study(root)
    gate = execute_replay_gate(study)
    acceptance_ref = publish_baseline_gate_study_acceptance(
        study.fixture.store,
        study_controller_manifest_ref=study.result.study_controller_manifest_ref,
        sealed_authorized_tail_ref=study.result.sealed_authorized_tail_ref,
        event_verifier=study.study_controller.event_service.verification_capability,
        baseline_gate_closure_ref=gate.closure_ref,
    )
    result_ref = publish_replay_study_gate_result(
        study.fixture.store,
        study_result_ref=study.result_ref,
        baseline_gate_acceptance_ref=acceptance_ref,
        event_verifier=study.study_controller.event_service.verification_capability,
    )
    return ReplayGateExecution(
        study=study,
        gate=gate,
        baseline_gate_acceptance_ref=acceptance_ref,
        baseline_gate_acceptance=study.fixture.store.get_json(
            acceptance_ref,
            BaselineGateStudyAcceptance,
        ),
        result_ref=result_ref,
        result=study.fixture.store.get_json(result_ref, ReplayStudyGateResult),
    )


def execute_replay_gate(
    study: ReplayStudyExecution,
    *,
    boundary: ReplayGateBoundary | None = None,
) -> BaselineGateStudyExecution:
    """Execute the fixture's trusted baseline GATE runner without publishing a final score."""

    selected_boundary = _gate_boundary(study) if boundary is None else boundary
    plan = study.fixture.store.get_json(
        study.fixture.baseline_study_plan_ref,
        BaselineStudyPlan,
    )
    candidate_refs = {
        kind: _candidate_ref(study, baseline=kind, harness_ref=study.fixture.seed_harness_ref)
        for kind in LEGACY_BASELINE_KINDS
    }
    candidate_harness_refs = {
        kind: _candidate_harness_ref(study, baseline=kind) for kind in LEGACY_BASELINE_KINDS
    }
    mechanism_refs = {
        kind: _mechanism_ref(
            study,
            baseline=kind,
            candidate_ref=candidate_refs[kind],
            harness_ref=candidate_harness_refs[kind],
        )
        for kind in LEGACY_BASELINE_KINDS
    }
    return TrustedBaselineGateRunner(
        study.fixture.store,
        adapter=ReplayGateAdapter(),
        gate_batch_service=study.fixture.gate_batch_service,
    ).execute_study(
        plan,
        query=0,
        protocol_ref=selected_boundary.protocol_ref,
        protocol=selected_boundary.protocol,
        candidate_refs=candidate_refs,
        parent_harness_ref=study.fixture.seed_harness_ref,
        candidate_harness_refs=candidate_harness_refs,
        gate_split_ref=selected_boundary.gate_split_ref,
        mechanism_evidence_refs=mechanism_refs,
        task_ids=("gate-1",),
        token_ceiling_per_attempt=32,
        runner_factories={
            kind: _runner_factory(study, kind=kind) for kind in LEGACY_BASELINE_KINDS
        },
    )


def _gate_boundary(study: ReplayStudyExecution) -> ReplayGateBoundary:
    return ReplayGateBoundary(
        protocol=study.fixture.protocol,
        protocol_ref=study.fixture.protocol_ref,
        gate_split_ref=next(
            split.manifest_ref
            for split in study.fixture.protocol.splits
            if split.partition is ProtocolPartition.GATE
        ),
    )


def _candidate_ref(
    study: ReplayStudyExecution,
    *,
    baseline: BaselineKind,
    harness_ref: ArtifactRef,
) -> ArtifactRef:
    return study.fixture.store.put_json(
        {"baseline": baseline.value, "harness": harness_ref.sha256},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )


def _mechanism_ref(
    study: ReplayStudyExecution,
    *,
    baseline: BaselineKind,
    candidate_ref: ArtifactRef,
    harness_ref: ArtifactRef,
) -> ArtifactRef:
    source_ref = study.fixture.store.put_json(
        {"mechanism": "baseline gate acceptance fixture", "baseline": baseline.value}
    )
    evidence = study.fixture.mechanism_evidence_service.create(
        protocol_ref=study.fixture.protocol_ref,
        protocol=study.fixture.protocol,
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
    return study.fixture.store.put_json(
        evidence,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )


def _candidate_harness_ref(
    study: ReplayStudyExecution,
    *,
    baseline: BaselineKind,
) -> ArtifactRef:
    prompt_ref = study.fixture.store.put_bytes(
        f"Answer the task directly.\nReplay GATE candidate: {baseline.value}.\n".encode(),
        media_type=PROMPT_MEDIA_TYPE,
    )
    harness = HarnessManifest(
        model_fingerprint=study.fixture.model_spec.model_fingerprint,
        runtime_fingerprint=study.fixture.model_spec.runtime_fingerprint,
        trusted_plane_version="fixture-trusted-plane-v1",
        parent=study.fixture.seed_harness_ref,
        components=(
            HarnessComponentRef(
                name=PROMPT_COMPONENT_NAME,
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
        budget=study.fixture.experiment.search_budget,
    )
    return study.fixture.store.put_json(
        harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )


def _runner_factory(
    study: ReplayStudyExecution,
    *,
    kind: BaselineKind,
):
    def factory(schedule: EvaluationBatchSchedule) -> tuple[FixedModelRunner, AttemptLedger]:
        backend = ReplayBackend(
            fingerprint=study.fixture.model_spec.backend_fingerprint,
            default_response=BackendResponse(
                output="ok",
                usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
            ),
        )
        ledger = AttemptLedger(
            study.fixture.store,
            ledger_id=f"accepted-baseline-gate-{kind.value}-{schedule.search_runs[0]}",
            budget=baseline_gate_attempt_budget(schedule),
        )
        runner = FixedModelRunner(
            spec=FrozenModelSpec.model_validate(study.fixture.model_spec, strict=True),
            backend=backend,
            attempt_ledger=ledger,
        )
        return runner, ledger

    return factory


__all__ = [
    "ReplayGateAdapter",
    "ReplayGateBoundary",
    "ReplayGateExecution",
    "ReplayGateTask",
    "execute_replay_gate",
    "run_non_reportable_replay_gate_result",
]
