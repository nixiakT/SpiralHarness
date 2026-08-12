from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.benchmark.runner import (
    BenchmarkBatchRunnerError,
    TrustedBenchmarkBatchRunner,
)
from spiral_harness.core.experiment import (
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
    AttemptBudget,
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
    materialize_request,
    replay_key,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
)
from spiral_harness.experiments.decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    GateEvaluationManifest,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    TrustedGateBatchService,
)
from spiral_harness.verification.gate import Decision, PromotionGate
from spiral_harness.verification.mechanism import TrustedMechanismEvidenceService
from spiral_harness.verification.models import GateConfig, MechanismCheck, MechanismEvidence

_BACKEND = "benchmark-batch-replay@sha256:fixture-v1"
_CANDIDATE_MEDIA_TYPE = "application/vnd.spiral-harness.candidate-manifest.v2+json"


def _write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _adapter(tmp_path: Path) -> GSM8KBenchmarkAdapter:
    train_rows = (
        {
            "question": "Lina has 2 shells and finds 3 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 5",
        },
        {
            "question": "Lina has 20 shells and finds 30 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 50",
        },
        {
            "question": "Four boxes hold 6 pens each. How many pens are there?",
            "answer": "Multiply.\n#### 24",
        },
        {
            "question": "A class has 11 girls and 9 boys. How many students are there?",
            "answer": "Add the students.\n#### 20",
        },
    )
    test_rows = (
        {
            "question": "A shelf has 7 red and 8 blue books. How many books are there?",
            "answer": "Add the books.\n#### 15",
        },
    )
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)
    return GSM8KBenchmarkAdapter(train_path, test_path, verify_pinned=False)


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=_BACKEND,
        model="fixture/gsm8k-model",
        revision="snapshot-2026-08-12",
        tokenizer="fixture/gsm8k-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="python-3.12/replay-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=32,
            timeout_seconds=10.0,
        ),
    )


def _put_harness(store: ArtifactStore, spec: FrozenModelSpec, prompt: str) -> ArtifactRef:
    prompt_ref = store.put_bytes(prompt.encode("utf-8"), media_type="text/plain")
    manifest = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version="trusted-benchmark-runner-v1",
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
    adapter: GSM8KBenchmarkAdapter,
    spec: FrozenModelSpec,
    gate_batch_service: TrustedGateBatchService,
) -> tuple[ProtocolManifest, ArtifactRef, ArtifactRef, ArtifactRef]:
    gate_split_ref = store.put_json(
        adapter.split_manifest.partition(ProtocolPartition.GATE),
        media_type="application/vnd.spiral-harness.gsm8k-partition.v1+json",
    )
    exploration_split_ref = store.put_json(
        adapter.split_manifest.partition(ProtocolPartition.EXPLORATION),
        media_type="application/vnd.spiral-harness.gsm8k-partition.v1+json",
    )
    gate_config = GateConfig(
        version="gsm8k-fixture-gate-v1",
        min_tasks=1,
        min_effect=0.0,
        bootstrap_samples=1_000,
        bootstrap_seed=17,
        expected_task_ids=adapter.task_roster(ProtocolPartition.GATE)[:1],
        required_mechanism_checks=("runner",),
    )
    gate_config_ref = store.put_json(gate_config, media_type="application/json")
    mechanism_service = TrustedMechanismEvidenceService()
    capability_policy_ref = store.put_json(
        {"policy": "no-tools"},
        media_type="application/vnd.spiral-harness.capability-policy.v1+json",
    )
    protocol = ProtocolManifest(
        benchmark_fingerprint=adapter.fingerprint,
        splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=exploration_split_ref,
            ),
            ProtocolSplit(
                partition=ProtocolPartition.GATE,
                manifest_ref=gate_split_ref,
            ),
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
        trusted_plane_version="trusted-benchmark-runner-v1",
        budget=BudgetPolicy(max_evaluations=20, max_tokens=1_000),
    )
    protocol_ref = store.put_json(protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)
    mechanism_source_ref = store.put_json(
        {"mechanism": "trusted benchmark runner fixture"},
        media_type="application/vnd.spiral-harness.mechanism-source.v1+json",
    )
    evidence = mechanism_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=ArtifactRef(
            sha256="c" * 64,
            size=1,
            media_type="application/vnd.spiral-harness.candidate-manifest.v2+json",
        ),
        candidate_harness_ref=ArtifactRef(
            sha256="d" * 64,
            size=1,
            media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        ),
        source_refs=(mechanism_source_ref,),
        evidence=MechanismEvidence(
            candidate_harness_id="d" * 64,
            checks=(
                MechanismCheck(
                    name="runner",
                    passed=True,
                    evidence_refs=(mechanism_source_ref.sha256,),
                ),
            ),
        ),
    )
    evidence_ref = store.put_json(
        evidence,
        media_type="application/vnd.spiral-harness.attested-mechanism-evidence.v1+json",
    )
    return protocol, protocol_ref, gate_split_ref, evidence_ref


@dataclass(frozen=True, slots=True)
class _UnstartedBatch:
    store: ArtifactStore
    adapter: GSM8KBenchmarkAdapter
    spec: FrozenModelSpec
    gate_batch_service: TrustedGateBatchService
    parent_ref: ArtifactRef
    candidate_ref: ArtifactRef
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef
    candidate_manifest_ref: ArtifactRef
    schedule: EvaluationBatchSchedule
    ledger: AttemptLedger
    runner: FixedModelRunner
    backend: ReplayBackend


def _unstarted_batch(tmp_path: Path) -> _UnstartedBatch:
    store = ArtifactStore(tmp_path / "artifacts")
    adapter = _adapter(tmp_path)
    spec = _spec()
    gate_batch_service = TrustedGateBatchService()
    parent_ref = _put_harness(store, spec, "Solve.")
    candidate_ref = _put_harness(store, spec, "Solve better.")
    protocol, protocol_ref, gate_split_ref, mechanism_evidence_ref = _protocol(
        store,
        adapter,
        spec,
        gate_batch_service,
    )
    schedule = EvaluationBatchSchedule(
        study="gsm8k-fixture",
        kind="prompt-only",
        phase=EvaluationPhase.GATE,
        query=0,
        master_seed=91,
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=candidate_ref.sha256,
        task_ids=(adapter.task_roster(ProtocolPartition.GATE)[0],),
        search_runs=(0,),
        repeat_seeds=(11,),
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=64,
    )
    backend = ReplayBackend(
        fingerprint=_BACKEND,
        default_response=BackendResponse(
            output="#### 0",
            usage=BackendTokenUsage(input_tokens=1, output_tokens=1),
        ),
    )
    ledger = AttemptLedger(
        store,
        ledger_id="gsm8k-fixture-batch",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=schedule.required_tokens,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        ),
    )
    return _UnstartedBatch(
        store=store,
        adapter=adapter,
        spec=spec,
        gate_batch_service=gate_batch_service,
        parent_ref=parent_ref,
        candidate_ref=candidate_ref,
        protocol=protocol,
        protocol_ref=protocol_ref,
        gate_split_ref=gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        candidate_manifest_ref=ArtifactRef(
            sha256="c" * 64,
            size=1,
            media_type=_CANDIDATE_MEDIA_TYPE,
        ),
        schedule=schedule,
        ledger=ledger,
        runner=FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger),
        backend=backend,
    )


def _execute_unstarted_batch(
    batch: _UnstartedBatch,
    *,
    protocol: ProtocolManifest | None = None,
    schedule: EvaluationBatchSchedule | None = None,
    gate_split_ref: ArtifactRef | None = None,
    parent_harness_ref: ArtifactRef | None = None,
    candidate_harness_ref: ArtifactRef | None = None,
) -> None:
    TrustedBenchmarkBatchRunner(
        batch.store,
        adapter=batch.adapter,
        gate_batch_service=batch.gate_batch_service,
    ).execute_paired_batch(
        protocol_ref=batch.protocol_ref,
        protocol=batch.protocol if protocol is None else protocol,
        candidate_ref=batch.candidate_manifest_ref,
        schedule=batch.schedule if schedule is None else schedule,
        parent_harness_ref=batch.parent_ref if parent_harness_ref is None else parent_harness_ref,
        candidate_harness_ref=(
            batch.candidate_ref if candidate_harness_ref is None else candidate_harness_ref
        ),
        gate_split_ref=batch.gate_split_ref if gate_split_ref is None else gate_split_ref,
        mechanism_evidence_ref=batch.mechanism_evidence_ref,
        runner=batch.runner,
        attempt_ledger=batch.ledger,
    )


def test_trusted_benchmark_batch_runner_executes_grades_and_signs_paired_gsm8k(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    adapter = _adapter(tmp_path)
    spec = _spec()
    gate_batch_service = TrustedGateBatchService()
    parent_ref = _put_harness(store, spec, "Solve and end with #### <number>.")
    candidate_ref = _put_harness(store, spec, "Solve, verify, and end with #### <number>.")
    protocol, protocol_ref, gate_split_ref, mechanism_evidence_ref = _protocol(
        store,
        adapter,
        spec,
        gate_batch_service,
    )
    candidate_manifest_ref = ArtifactRef(
        sha256="c" * 64,
        size=1,
        media_type=_CANDIDATE_MEDIA_TYPE,
    )
    task_id = adapter.task_roster(ProtocolPartition.GATE)[0]
    schedule = EvaluationBatchSchedule(
        study="gsm8k-fixture",
        kind="prompt-only",
        phase=EvaluationPhase.GATE,
        query=0,
        master_seed=91,
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=candidate_ref.sha256,
        task_ids=(task_id,),
        search_runs=(0,),
        repeat_seeds=(11,),
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=64,
    )
    materializer = HarnessMaterializer(store, spec=spec)
    task = adapter.load_task(task_id)
    responses = {}
    for cell in schedule.iter_cells():
        harness_ref = parent_ref if cell.side.value == "parent" else candidate_ref
        request = materialize_request(
            task,
            materializer.materialize(harness_ref),
            seed=schedule.seed_for(cell, attempt_index=0),
        )
        responses[replay_key(spec, request)] = BackendResponse(
            output="The answer is correct.\n#### 20",
            usage=BackendTokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )
    backend = ReplayBackend(fingerprint=_BACKEND, responses=responses)
    ledger = AttemptLedger(
        store,
        ledger_id="gsm8k-fixture-batch",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=schedule.required_tokens,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        ),
    )
    runner = FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)

    execution = TrustedBenchmarkBatchRunner(
        store,
        adapter=adapter,
        gate_batch_service=gate_batch_service,
    ).execute_paired_batch(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_manifest_ref,
        schedule=schedule,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=candidate_ref,
        gate_split_ref=gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        runner=runner,
        attempt_ledger=ledger,
    )

    assert len(backend.calls) == schedule.cell_count
    assert execution.usage.cell_count == schedule.cell_count
    assert execution.usage.settled_attempts == schedule.cell_count
    assert execution.usage.charged_tokens == 30
    assert len(execution.receipt_refs) == schedule.cell_count
    assert len(execution.execution_refs) == schedule.cell_count
    assert execution.parent_batch_ref.media_type == GATE_TRIAL_BATCH_MEDIA_TYPE
    assert execution.candidate_batch_ref.media_type == GATE_TRIAL_BATCH_MEDIA_TYPE
    parent_batch = gate_batch_service.verification_capability.verify(execution.parent_batch)
    candidate_batch = gate_batch_service.verification_capability.verify(execution.candidate_batch)
    assert parent_batch.observations[0].score == 1.0
    assert candidate_batch.observations[0].score == 1.0
    assert {ref.sha256 for ref in parent_batch.source_refs} == {
        ref.sha256 for ref in execution.receipt_refs
    }
    assert {ref.sha256 for ref in candidate_batch.source_refs} == {
        ref.sha256 for ref in execution.receipt_refs
    }
    assert parent_batch.observations[0].execution_fingerprint == (
        candidate_batch.observations[0].execution_fingerprint
    )

    evaluation = GateEvaluationManifest(
        candidate_ref=candidate_manifest_ref,
        admission_report_ref=store.put_json(
            {"admission": "fixture"},
            media_type="application/vnd.spiral-harness.admission-report.v1+json",
        ),
        gate_config_ref=protocol.gate_config_ref,
        gate_split_ref=gate_split_ref,
        parent_batch_ref=execution.parent_batch_ref,
        candidate_batch_ref=execution.candidate_batch_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
    )
    evaluation_ref = store.put_json(
        evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    assert evaluation_ref.media_type == GATE_EVALUATION_MANIFEST_MEDIA_TYPE
    assert (
        PromotionGate(store.get_json(protocol.gate_config_ref, GateConfig))
        .evaluate(
            parent_batch.observations,
            candidate_batch.observations,
            mechanism_evidence=MechanismEvidence(
                candidate_harness_id=candidate_ref.sha256,
                checks=(MechanismCheck(name="runner", passed=True, evidence_refs=("fixture",)),),
            ),
        )
        .decision
        is Decision.INCONCLUSIVE
    )


def test_benchmark_batch_runner_rejects_harness_that_does_not_match_schedule(
    tmp_path: Path,
) -> None:
    batch = _unstarted_batch(tmp_path)

    with pytest.raises(BenchmarkBatchRunnerError, match="parent_harness_ref"):
        _execute_unstarted_batch(batch, parent_harness_ref=batch.candidate_ref)
    assert batch.backend.calls == ()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("benchmark_fingerprint", "wrong-benchmark-fingerprint"),
        ("grader_fingerprint", "wrong-grader-fingerprint"),
        ("model_fingerprint", "wrong-model-fingerprint"),
        ("inference_fingerprint", "wrong-inference-fingerprint"),
        ("runtime_fingerprint", "wrong-runtime-fingerprint"),
        ("model_spec_fingerprint", "f" * 64),
    ),
)
def test_benchmark_batch_runner_rejects_protocol_identity_drift_before_backend(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
) -> None:
    batch = _unstarted_batch(tmp_path)
    protocol = batch.protocol.model_copy(update={field_name: bad_value})

    with pytest.raises(BenchmarkBatchRunnerError, match=field_name):
        _execute_unstarted_batch(batch, protocol=protocol)
    assert batch.backend.calls == ()


def test_benchmark_batch_runner_rejects_non_gate_schedule_before_backend(
    tmp_path: Path,
) -> None:
    batch = _unstarted_batch(tmp_path)
    schedule = batch.schedule.model_copy(update={"phase": EvaluationPhase.PROBE})

    with pytest.raises(BenchmarkBatchRunnerError, match="GATE schedule"):
        _execute_unstarted_batch(batch, schedule=schedule)
    assert batch.backend.calls == ()


def test_benchmark_batch_runner_rejects_unknown_gate_roster_task_before_backend(
    tmp_path: Path,
) -> None:
    batch = _unstarted_batch(tmp_path)
    schedule = batch.schedule.model_copy(update={"task_ids": ("not-in-the-gate-roster",)})

    with pytest.raises(BenchmarkBatchRunnerError, match="outside the trusted GATE roster"):
        _execute_unstarted_batch(batch, schedule=schedule)
    assert batch.backend.calls == ()


def test_benchmark_batch_runner_rejects_gate_split_drift_before_backend(
    tmp_path: Path,
) -> None:
    batch = _unstarted_batch(tmp_path)
    gate_split_ref = batch.store.put_json(
        {"not": "the protocol gate split"},
        media_type="application/vnd.spiral-harness.gsm8k-partition.v1+json",
    )

    with pytest.raises(BenchmarkBatchRunnerError, match="protocol GATE split"):
        _execute_unstarted_batch(batch, gate_split_ref=gate_split_ref)
    assert batch.backend.calls == ()


def test_benchmark_batch_runner_rejects_attestor_drift_before_backend(
    tmp_path: Path,
) -> None:
    batch = _unstarted_batch(tmp_path)
    protocol = batch.protocol.model_copy(update={"gate_batch_attestor_id": "a" * 64})

    with pytest.raises(BenchmarkBatchRunnerError, match="gate batch attestor"):
        _execute_unstarted_batch(batch, protocol=protocol)
    assert batch.backend.calls == ()
