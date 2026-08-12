from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

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
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.model import FixedModelRunner, ReplayBackend
from spiral_harness.execution.schedule import EvaluationBatchSchedule
from spiral_harness.experiments.baseline_gate_closure import (
    BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
    BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE,
    BaselineGateClosureError,
    BaselineGateStudyClosure,
    verify_baseline_gate_study_closure,
)
from spiral_harness.experiments.baseline_gate_runner import (
    BaselineGateRunnerError,
    TrustedBaselineGateRunner,
    baseline_gate_attempt_budget,
)
from spiral_harness.experiments.baselines import (
    BaselineKind,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    BaselineUsageReport,
    FeedbackType,
    FrozenMutationPolicy,
    FrozenRunContext,
    PairedEvaluationPlan,
    ResourceCeilings,
    ResourceUsage,
    plan_four_baselines,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    TrustedGateBatchService,
)
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import (
    GateConfig,
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
    TrialStatus,
)

_BACKEND = "baseline-gate-replay@sha256:fixture-v1"


@dataclass(frozen=True, slots=True)
class _ToyTask:
    task_id: str
    question: str


class _ToyBenchmarkAdapter:
    def __init__(self) -> None:
        self._tasks = {
            "toy-add": _ToyTask(task_id="toy-add", question="What is 1+1?"),
            "toy-mul": _ToyTask(task_id="toy-mul", question="What is 2*3?"),
        }

    @property
    def fingerprint(self) -> str:
        return "toy-benchmark-adapter@sha256:fixture-v1"

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        if partition is ProtocolPartition.GATE:
            return tuple(self._tasks)
        if partition is ProtocolPartition.EXPLORATION:
            return ("toy-explore",)
        return ()

    def load_task(self, task_ref: str) -> _ToyTask:
        return self._tasks[task_ref]

    def grade(
        self,
        task: _ToyTask,
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
            score=1.0 if completed and execution.output_text == "ok" else 0.0,
            tokens=execution.tokens,
            latency_ms=execution.latency_ms,
            tool_calls=execution.tool_calls,
            cost_usd=execution.cost_usd,
            execution_fingerprint=execution_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class _BaselineGateFixture:
    store: ArtifactStore
    adapter: _ToyBenchmarkAdapter
    spec: FrozenModelSpec
    gate_batch_service: TrustedGateBatchService
    mechanism_service: TrustedMechanismEvidenceService
    plan: BaselineStudyPlan
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    parent_ref: ArtifactRef
    candidate_refs: dict[BaselineKind, ArtifactRef]
    candidate_harness_refs: dict[BaselineKind, ArtifactRef]
    gate_split_ref: ArtifactRef
    mechanism_evidence_refs: dict[BaselineKind, ArtifactRef]
    task_ids: tuple[str, ...]
    backends: list[ReplayBackend]


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=_BACKEND,
        model="fixture/baseline-gate-model",
        revision="snapshot-2026-08-12",
        tokenizer="fixture/baseline-gate-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="python-3.12/replay-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=16,
            timeout_seconds=5.0,
        ),
    )


def _put_harness(store: ArtifactStore, spec: FrozenModelSpec, prompt: str) -> ArtifactRef:
    prompt_ref = store.put_bytes(prompt.encode("utf-8"), media_type="text/plain")
    manifest = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version="trusted-baseline-gate-fixture-v1",
        components=(
            HarnessComponentRef(
                name="system-prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
    )
    return store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)


def _protocol(
    store: ArtifactStore,
    adapter: _ToyBenchmarkAdapter,
    spec: FrozenModelSpec,
    gate_batch_service: TrustedGateBatchService,
    mechanism_service: TrustedMechanismEvidenceService,
) -> tuple[ProtocolManifest, ArtifactRef, ArtifactRef]:
    exploration_split_ref = store.put_json(
        {"partition": "exploration", "tasks": adapter.task_roster(ProtocolPartition.EXPLORATION)},
        media_type="application/vnd.spiral-harness.toy-partition.v1+json",
    )
    gate_split_ref = store.put_json(
        {"partition": "gate", "tasks": adapter.task_roster(ProtocolPartition.GATE)},
        media_type="application/vnd.spiral-harness.toy-partition.v1+json",
    )
    capability_policy_ref = store.put_json(
        {"policy": "no-tools"},
        media_type="application/vnd.spiral-harness.capability-policy.v1+json",
    )
    gate_config_ref = store.put_json(
        GateConfig(version="toy-gate-v1", min_tasks=2, bootstrap_seed=101),
        media_type="application/json",
    )
    protocol = ProtocolManifest(
        benchmark_fingerprint=adapter.fingerprint,
        splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=exploration_split_ref,
            ),
            ProtocolSplit(partition=ProtocolPartition.GATE, manifest_ref=gate_split_ref),
        ),
        model_fingerprint=spec.model_fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        model_spec_fingerprint=spec.fingerprint,
        sandbox_fingerprint="local-replay-sandbox-v1",
        capability_policy_ref=capability_policy_ref,
        grader_fingerprint=adapter.fingerprint,
        gate_batch_attestor_id=gate_batch_service.attestor_id,
        mechanism_evidence_attestor_id=mechanism_service.attestor_id,
        gate_config_ref=gate_config_ref,
        trusted_plane_version="trusted-baseline-gate-fixture-v1",
        budget=BudgetPolicy(max_evaluations=128, max_tokens=10_000),
    )
    protocol_ref = store.put_json(protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)
    return protocol, protocol_ref, gate_split_ref


def _study_plan(
    store: ArtifactStore,
    spec: FrozenModelSpec,
    parent_ref: ArtifactRef,
) -> BaselineStudyPlan:
    benchmark_ref = store.put_json(
        {"benchmark": "toy", "split": "gate"},
        media_type="application/vnd.spiral-harness.search-benchmark-binding.v1+json",
    )
    return plan_four_baselines(
        context=FrozenRunContext(
            benchmark_ref=benchmark_ref,
            model_fingerprint=spec.model_fingerprint,
            inference_fingerprint=spec.inference_fingerprint,
            runtime_fingerprint=spec.runtime_fingerprint,
            seed_harness_ref=parent_ref,
            mutation_policy=FrozenMutationPolicy(
                grammar_version="atomic-prompt-replace-v1",
                allowed_component_kinds=(ComponentKind.PROMPT,),
                max_artifact_size_bytes=65_536,
            ),
            proposal_random_seed=20260812,
        ),
        evaluation=PairedEvaluationPlan(
            search_run_seeds=(17, 19),
            repeat_seeds=(3, 5),
        ),
        ceilings=ResourceCeilings(
            max_evaluations=64,
            max_feedback_queries=4,
            max_proposals=4,
            max_optimizer_model_calls=4,
            max_tokens=5_000,
            max_wall_time_seconds=600.0,
            max_cost_usd=0.0,
        ),
    )


def _mechanism_evidence(
    store: ArtifactStore,
    *,
    protocol_ref: ArtifactRef,
    protocol: ProtocolManifest,
    mechanism_service: TrustedMechanismEvidenceService,
    candidate_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
) -> ArtifactRef:
    source_ref = store.put_json(
        {"mechanism": "baseline gate fixture"},
        media_type="application/vnd.spiral-harness.mechanism-source.v1+json",
    )
    evidence = mechanism_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        candidate_harness_ref=candidate_harness_ref,
        source_refs=(source_ref,),
        evidence=MechanismEvidence(
            candidate_harness_id=candidate_harness_ref.sha256,
            checks=(
                MechanismCheck(
                    name="baseline-runner-fixture",
                    passed=True,
                    evidence_refs=(source_ref.sha256,),
                ),
            ),
        ),
    )
    return store.put_json(evidence, media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE)


def _fixture(tmp_path: Path) -> _BaselineGateFixture:
    store = ArtifactStore(tmp_path / "artifacts")
    adapter = _ToyBenchmarkAdapter()
    spec = _spec()
    gate_batch_service = TrustedGateBatchService()
    mechanism_service = TrustedMechanismEvidenceService()
    parent_ref = _put_harness(store, spec, "Solve the toy task.")
    plan = _study_plan(store, spec, parent_ref)
    protocol, protocol_ref, gate_split_ref = _protocol(
        store,
        adapter,
        spec,
        gate_batch_service,
        mechanism_service,
    )
    candidate_harness_refs = {
        BaselineKind.STATIC: parent_ref,
        BaselineKind.RANDOM_VALID: _put_harness(store, spec, "Solve the toy task randomly."),
        BaselineKind.PROMPT_ONLY: _put_harness(store, spec, "Solve the toy task carefully."),
        BaselineKind.EVIDENCE_TARGETED: _put_harness(
            store,
            spec,
            "Solve the toy task and verify arithmetic.",
        ),
    }
    candidate_refs = {
        kind: store.put_json(
            {"baseline": kind.value, "harness": harness_ref.sha256},
            media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
        )
        for kind, harness_ref in candidate_harness_refs.items()
    }
    mechanism_evidence_refs = {
        kind: _mechanism_evidence(
            store,
            protocol_ref=protocol_ref,
            protocol=protocol,
            mechanism_service=mechanism_service,
            candidate_ref=candidate_refs[kind],
            candidate_harness_ref=harness_ref,
        )
        for kind, harness_ref in candidate_harness_refs.items()
    }
    return _BaselineGateFixture(
        store=store,
        adapter=adapter,
        spec=spec,
        gate_batch_service=gate_batch_service,
        mechanism_service=mechanism_service,
        plan=plan,
        protocol=protocol,
        protocol_ref=protocol_ref,
        parent_ref=parent_ref,
        candidate_refs=candidate_refs,
        candidate_harness_refs=candidate_harness_refs,
        gate_split_ref=gate_split_ref,
        mechanism_evidence_refs=mechanism_evidence_refs,
        task_ids=adapter.task_roster(ProtocolPartition.GATE),
        backends=[],
    )


def _runner_factory(
    fixture: _BaselineGateFixture,
    *,
    kind: BaselineKind,
):
    def factory(schedule: EvaluationBatchSchedule) -> tuple[FixedModelRunner, AttemptLedger]:
        backend = ReplayBackend(
            fingerprint=_BACKEND,
            default_response=BackendResponse(
                output="ok",
                usage=BackendTokenUsage(input_tokens=4, output_tokens=3),
            ),
        )
        ledger = AttemptLedger(
            fixture.store,
            ledger_id=f"{kind.value}-{schedule.search_runs[0]}",
            budget=baseline_gate_attempt_budget(schedule),
        )
        fixture.backends.append(backend)
        return FixedModelRunner(spec=fixture.spec, backend=backend, attempt_ledger=ledger), ledger

    return factory


def test_baseline_gate_runner_executes_condition_and_publishes_usage_report(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    execution = TrustedBaselineGateRunner(
        fixture.store,
        adapter=fixture.adapter,
        gate_batch_service=fixture.gate_batch_service,
    ).execute_condition(
        fixture.plan,
        kind=BaselineKind.PROMPT_ONLY,
        query=4,
        protocol_ref=fixture.protocol_ref,
        protocol=fixture.protocol,
        candidate_ref=fixture.candidate_refs[BaselineKind.PROMPT_ONLY],
        parent_harness_ref=fixture.parent_ref,
        candidate_harness_ref=fixture.candidate_harness_refs[BaselineKind.PROMPT_ONLY],
        gate_split_ref=fixture.gate_split_ref,
        mechanism_evidence_ref=fixture.mechanism_evidence_refs[BaselineKind.PROMPT_ONLY],
        task_ids=fixture.task_ids,
        token_ceiling_per_attempt=32,
        runner_factory=_runner_factory(fixture, kind=BaselineKind.PROMPT_ONLY),
        additional_usage=ResourceUsage(
            feedback_queries=1,
            proposals=1,
            optimizer_model_calls=1,
            tokens=5,
        ),
        feedback_used=(
            FeedbackType.BENCHMARK_METADATA,
            FeedbackType.EXPLORATION_INPUTS,
        ),
        mutated_component_kinds=(ComponentKind.PROMPT,),
    )

    assert execution.report_ref.media_type == BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE
    assert fixture.store.get_json(execution.report_ref, BaselineUsageReport) == execution.report
    loaded_report = fixture.store.get_json(execution.report_ref)
    assert loaded_report["kind"] == BaselineKind.PROMPT_ONLY.value
    assert tuple(schedule.search_runs[0] for schedule in execution.schedules) == (17, 19)
    assert len(execution.batches) == 2
    assert [len(backend.calls) for backend in fixture.backends] == [8, 8]
    assert execution.report.used.evaluations == 16
    assert execution.report.used.tokens == 117
    assert execution.report.used.feedback_queries == 1
    assert execution.report.used.proposals == 1
    assert execution.report.used.optimizer_model_calls == 1
    assert execution.report.mutated_component_kinds == (ComponentKind.PROMPT,)
    assert {batch.parent_batch_ref.media_type for batch in execution.batches} == {
        GATE_TRIAL_BATCH_MEDIA_TYPE
    }
    assert {batch.candidate_batch_ref.media_type for batch in execution.batches} == {
        GATE_TRIAL_BATCH_MEDIA_TYPE
    }


def test_baseline_gate_runner_executes_all_four_conditions_and_validates_usage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    factories = {kind: _runner_factory(fixture, kind=kind) for kind in BaselineKind}

    study = TrustedBaselineGateRunner(
        fixture.store,
        adapter=fixture.adapter,
        gate_batch_service=fixture.gate_batch_service,
    ).execute_study(
        fixture.plan,
        query=0,
        protocol_ref=fixture.protocol_ref,
        protocol=fixture.protocol,
        candidate_refs=fixture.candidate_refs,
        parent_harness_ref=fixture.parent_ref,
        candidate_harness_refs=fixture.candidate_harness_refs,
        gate_split_ref=fixture.gate_split_ref,
        mechanism_evidence_refs=fixture.mechanism_evidence_refs,
        task_ids=fixture.task_ids,
        token_ceiling_per_attempt=32,
        runner_factories=factories,
    )

    assert study.consistency == BaselineProtocolValidator.validate_usage(
        fixture.plan,
        tuple(execution.report for execution in study.executions),
    )
    assert study.consistency.execution_attested is False
    assert tuple(execution.kind for execution in study.executions) == tuple(BaselineKind)
    assert study.closure_ref.media_type == BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE
    assert fixture.store.get_json(study.closure_ref, BaselineGateStudyClosure) == study.closure
    assert (
        verify_baseline_gate_study_closure(fixture.store, study.closure_ref, plan=fixture.plan)
        == study.closure
    )
    assert study.closure.reportable_benchmark_result is False
    assert frozenset(condition.kind for condition in study.closure.conditions) == frozenset(
        BaselineKind
    )
    execution_refs = {execution.kind: execution.report_ref for execution in study.executions}
    closure_refs = {condition.kind: condition.report_ref for condition in study.closure.conditions}
    assert closure_refs == execution_refs
    assert len(fixture.backends) == 8
    assert sum(len(backend.calls) for backend in fixture.backends) == 64
    assert {execution.report.used.evaluations for execution in study.executions} == {16}


def test_baseline_gate_closure_rejects_tampered_embedded_usage_report(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    factories = {kind: _runner_factory(fixture, kind=kind) for kind in BaselineKind}
    study = TrustedBaselineGateRunner(
        fixture.store,
        adapter=fixture.adapter,
        gate_batch_service=fixture.gate_batch_service,
    ).execute_study(
        fixture.plan,
        query=0,
        protocol_ref=fixture.protocol_ref,
        protocol=fixture.protocol,
        candidate_refs=fixture.candidate_refs,
        parent_harness_ref=fixture.parent_ref,
        candidate_harness_refs=fixture.candidate_harness_refs,
        gate_split_ref=fixture.gate_split_ref,
        mechanism_evidence_refs=fixture.mechanism_evidence_refs,
        task_ids=fixture.task_ids,
        token_ceiling_per_attempt=32,
        runner_factories=factories,
    )

    condition = study.closure.conditions[0]
    forged_report = condition.report.model_copy(
        update={"used": condition.report.used.model_copy(update={"tokens": 0})}
    )
    forged_condition = condition.model_copy(update={"report": forged_report})
    forged_closure = study.closure.model_copy(
        update={"conditions": (forged_condition, *study.closure.conditions[1:])}
    )
    forged_ref = fixture.store.put_json(
        forged_closure,
        media_type=BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
    )

    with pytest.raises(BaselineGateClosureError, match="embedded usage report differs"):
        verify_baseline_gate_study_closure(fixture.store, forged_ref, plan=fixture.plan)


def test_baseline_gate_runner_rejects_non_fresh_or_wrong_budget_ledger_before_backend(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    backend = ReplayBackend(
        fingerprint=_BACKEND,
        default_response=BackendResponse(
            output="ok",
            usage=BackendTokenUsage(input_tokens=4, output_tokens=3),
        ),
    )

    def factory(schedule: EvaluationBatchSchedule) -> tuple[FixedModelRunner, AttemptLedger]:
        exact_budget = baseline_gate_attempt_budget(schedule)
        ledger = AttemptLedger(
            fixture.store,
            ledger_id="wrong-budget",
            budget=exact_budget.model_copy(
                update={"max_total_tokens": exact_budget.max_total_tokens + 1}
            ),
        )
        return FixedModelRunner(spec=fixture.spec, backend=backend, attempt_ledger=ledger), ledger

    with pytest.raises(BaselineGateRunnerError, match="exact worst-case budget"):
        TrustedBaselineGateRunner(
            fixture.store,
            adapter=fixture.adapter,
            gate_batch_service=fixture.gate_batch_service,
        ).execute_condition(
            fixture.plan,
            kind=BaselineKind.STATIC,
            query=0,
            protocol_ref=fixture.protocol_ref,
            protocol=fixture.protocol,
            candidate_ref=fixture.candidate_refs[BaselineKind.STATIC],
            parent_harness_ref=fixture.parent_ref,
            candidate_harness_ref=fixture.candidate_harness_refs[BaselineKind.STATIC],
            gate_split_ref=fixture.gate_split_ref,
            mechanism_evidence_ref=fixture.mechanism_evidence_refs[BaselineKind.STATIC],
            task_ids=fixture.task_ids,
            token_ceiling_per_attempt=32,
            runner_factory=factory,
        )
    assert backend.calls == ()
