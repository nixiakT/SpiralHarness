from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.pure_reference import (
    PURE_REFERENCE_BATCH_MEDIA_TYPE,
    PureReferenceCell,
    PureReferencePlan,
    PureReferenceRunnerError,
    TrustedPureReferenceRunner,
    verify_pure_reference_batch,
)
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.execution.accounted_execution import load_accounted_execution
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    BackendResponse,
    BackendTokenUsage,
    CandidateTask,
    ExecutionErrorClass,
    ExecutionStatus,
    FrozenModelSpec,
    InferenceConfig,
    ProviderIdentityObservation,
)
from spiral_harness.execution.model import paired_execution_fingerprint
from spiral_harness.execution.pure_contracts import (
    PureReferenceCapabilities,
    PureReferenceExecution,
    PureReferenceRequest,
    materialize_pure_request,
)
from spiral_harness.execution.pure_model import PureModelRunner
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.models import TrialObservation, TrialStatus

_BACKEND = "pure-fixture-backend@sha256:snapshot-v1"


def _spec(*, model: str = "fixture/exact-model") -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="pure-fixture",
        backend_fingerprint=_BACKEND,
        model=model,
        revision="snapshot-2026-08-14",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-14",
        runtime="python-3.12/pure-fixture-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=32,
            timeout_seconds=5.0,
        ),
    )


def _budget(*, attempts: int = 1, per_attempt: int = 64) -> AttemptBudget:
    return AttemptBudget(
        max_attempts=attempts,
        max_total_tokens=attempts * per_attempt,
        max_tokens_per_attempt=per_attempt,
    )


class _CapturingPureBackend:
    def __init__(self, output: str = "correct") -> None:
        self._output = output
        self.requests: list[PureReferenceRequest] = []
        self.specs: list[FrozenModelSpec] = []

    @property
    def fingerprint(self) -> str:
        return _BACKEND

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse:
        self.specs.append(spec)
        self.requests.append(request)
        return BackendResponse(
            output=self._output,
            usage=BackendTokenUsage(input_tokens=5, output_tokens=2),
        )


class _TrustedFixtureAdapter:
    fingerprint = "trusted-fixture-adapter-v1"

    def __init__(self, task: CandidateTask) -> None:
        self.task = task
        self.grade_calls = 0

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        assert partition is ProtocolPartition.GATE
        return (self.task.task_id,)

    def load_task(self, task_ref: str) -> CandidateTask:
        assert task_ref == self.task.task_id
        return self.task

    def grade(
        self,
        task: CandidateTask,
        execution: object,
        *,
        harness_id: str,
        seed: int,
        execution_fingerprint: str,
    ) -> TrialObservation:
        self.grade_calls += 1
        output = execution.output_text
        return TrialObservation(
            task_id=task.task_id,
            seed=seed,
            harness_id=harness_id,
            status=TrialStatus.COMPLETED,
            score=float(output == "correct"),
            tokens=execution.tokens,
            latency_ms=execution.latency_ms,
            tool_calls=execution.tool_calls,
            execution_fingerprint=execution_fingerprint,
        )


def _plan(task: CandidateTask, spec: FrozenModelSpec | None = None) -> PureReferencePlan:
    return PureReferencePlan(
        adapter_fingerprint=_TrustedFixtureAdapter.fingerprint,
        partition=ProtocolPartition.GATE,
        spec=_spec() if spec is None else spec,
        cells=(PureReferenceCell.from_task(task, rollout_seed=17),),
        ledger_id="pure-reference-fixture",
        attempt_budget=_budget(),
    )


def _runner_context(
    tmp_path: Path,
    *,
    spec: FrozenModelSpec | None = None,
) -> tuple[ArtifactStore, AttemptLedger, _CapturingPureBackend, PureModelRunner]:
    checked_spec = _spec() if spec is None else spec
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = AttemptLedger(store, ledger_id="pure-reference-fixture", budget=_budget())
    backend = _CapturingPureBackend()
    runner = PureModelRunner(spec=checked_spec, backend=backend, attempt_ledger=ledger)
    return store, ledger, backend, runner


def test_pure_request_preserves_exact_task_bytes_and_has_no_harness_surface() -> None:
    task = CandidateTask(task_id="whitespace-task", question="  keep this\nexactly  ")
    request = materialize_pure_request(task, reference_id="a" * 64, rollout_seed=9)

    assert request.task_payload == task.question
    assert request.messages[0].model_dump() == {"role": "user", "content": task.question}
    assert request.capabilities == PureReferenceCapabilities()
    assert request.capabilities.search is False
    assert request.capabilities.evolution is False
    dumped = request.model_dump(mode="json")
    assert not {"harness_ref", "system_prompt", "skill", "memory", "tools"}.intersection(dumped)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [
            {"role": "user", "content": "payload"},
            {"role": "user", "content": "payload"},
        ],
        [{"role": "system", "content": "payload"}],
    ],
)
def test_pure_request_rejects_zero_multiple_or_non_user_messages(
    messages: list[dict[str, str]],
) -> None:
    request = materialize_pure_request(
        CandidateTask(task_id="task", question="payload"),
        reference_id="a" * 64,
        rollout_seed=0,
    )
    payload = request.model_dump(mode="python")
    payload["messages"] = messages

    with pytest.raises(ValidationError):
        PureReferenceRequest.model_validate(payload, strict=True)


def test_pure_contract_cannot_enable_search_evolution_or_a_harness() -> None:
    payload = PureReferenceCapabilities().model_dump(mode="python")

    for field_name in ("harness", "search", "evolution", "tools", "control_flow"):
        with pytest.raises(ValidationError):
            PureReferenceCapabilities.model_validate(
                {**payload, field_name: True},
                strict=True,
            )


def test_pure_runner_uses_same_paired_identity_and_settles_shared_ledger(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    store, ledger, backend, runner = _runner_context(tmp_path)

    record = runner.execute_record(task, reference_id="b" * 64, rollout_seed=17)

    assert backend.specs == [_spec()]
    assert len(backend.requests) == 1
    assert backend.requests[0].messages[0].content == "payload"
    assert record.execution.execution_fingerprint == paired_execution_fingerprint(
        _spec(),
        task,
        seed=17,
        backend_fingerprint=_BACKEND,
    )
    assert record.execution.capabilities.harness is False
    assert record.execution.capabilities.search is False
    assert record.execution.capabilities.evolution is False
    assert ledger.state().completed_attempts == 1
    assert ledger.state().charged_tokens == 7
    assert store.get_json(record.execution_ref, type(record.execution)) == record.execution
    legacy_payload = record.execution.model_dump(mode="python")
    legacy_payload.pop("provider_identity_observation")
    assert (
        PureReferenceExecution.model_validate(
            legacy_payload,
            strict=True,
        ).provider_identity_observation
        is None
    )


def test_reference_id_is_derived_from_the_complete_frozen_plan() -> None:
    task = CandidateTask(task_id="task", question="payload")
    first = _plan(task)
    identical = _plan(task)
    changed_model = _plan(task, _spec(model="fixture/different-exact-model"))
    changed_seed = first.model_copy(
        update={"cells": (PureReferenceCell.from_task(task, rollout_seed=18),)}
    )

    assert first.reference_id == identical.reference_id
    assert first.reference_id != changed_model.reference_id
    assert first.reference_id != changed_seed.reference_id
    assert "reference_id" not in first.model_dump(mode="json")
    assert "harness" not in first.model_dump(mode="json")


def test_trusted_runner_scores_after_pure_execution_and_publishes_flags(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    adapter = _TrustedFixtureAdapter(task)
    store, ledger, backend, runner = _runner_context(tmp_path)
    plan = _plan(task)

    result = TrustedPureReferenceRunner(store, adapter=adapter).execute(
        plan=plan,
        runner=runner,
        attempt_ledger=ledger,
    )

    # One grading pass creates the observation and a second durable replay pass
    # independently recomputes it before the batch is returned.
    assert adapter.grade_calls == 2
    assert len(backend.requests) == 1
    assert result.batch_ref.media_type == PURE_REFERENCE_BATCH_MEDIA_TYPE
    assert result.batch.reference_id == plan.reference_id
    assert result.batch.observations[0].score == 1.0
    assert result.batch.observations[0].harness_id == plan.reference_id
    assert result.batch.capabilities.harness is False
    assert result.batch.capabilities.search is False
    assert result.batch.capabilities.evolution is False
    assert result.batch.all_executions_completed_successfully is True
    assert result.batch.terminal_ledger_poisoned is False
    assert result.batch.reportable_benchmark_result is False
    assert result.batch.sealed_evidence is False
    assert result.batch.paired_cross_arm_receipt_closure is False
    assert result.batch.provider_identity_attested is False
    assert result.batch.standalone_verification is False
    assert result.executions[0].request.reference_id == plan.reference_id


def test_durable_verifier_replays_chain_and_regrades_observation(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    adapter = _TrustedFixtureAdapter(task)
    store, ledger, _, runner = _runner_context(tmp_path)
    plan = _plan(task)
    result = TrustedPureReferenceRunner(store, adapter=adapter).execute(
        plan=plan,
        runner=runner,
        attempt_ledger=ledger,
    )

    verified = verify_pure_reference_batch(
        store,
        result.batch_ref,
        adapter=adapter,
        expected_plan=plan,
    )

    assert verified == result.batch
    assert adapter.grade_calls == 3


def test_durable_verifier_rejects_a_caller_authored_score(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    adapter = _TrustedFixtureAdapter(task)
    store, ledger, _, runner = _runner_context(tmp_path)
    result = TrustedPureReferenceRunner(store, adapter=adapter).execute(
        plan=_plan(task),
        runner=runner,
        attempt_ledger=ledger,
    )
    forged_observation = result.batch.observations[0].model_copy(update={"score": 0.0})
    forged_batch = result.batch.model_copy(update={"observations": (forged_observation,)})
    forged_ref = store.put_json(forged_batch, media_type=PURE_REFERENCE_BATCH_MEDIA_TYPE)

    with pytest.raises(PureReferenceRunnerError, match="trusted regrading"):
        verify_pure_reference_batch(store, forged_ref, adapter=adapter)


def test_durable_verifier_rejects_a_missing_execution_artifact(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    adapter = _TrustedFixtureAdapter(task)
    store, ledger, _, runner = _runner_context(tmp_path)
    result = TrustedPureReferenceRunner(store, adapter=adapter).execute(
        plan=_plan(task),
        runner=runner,
        attempt_ledger=ledger,
    )
    missing_ref = result.batch.execution_refs[0].model_copy(update={"sha256": "f" * 64, "size": 1})
    forged_batch = result.batch.model_copy(update={"execution_refs": (missing_ref,)})
    forged_ref = store.put_json(forged_batch, media_type=PURE_REFERENCE_BATCH_MEDIA_TYPE)

    with pytest.raises(PureReferenceRunnerError, match="execution artifact"):
        verify_pure_reference_batch(store, forged_ref, adapter=adapter)


class _FingerprintFailureAfterResponseBackend:
    def __init__(self, *, total_tokens: int, raise_on_final_read: bool) -> None:
        self._reads = 0
        self._total_tokens = total_tokens
        self._raise_on_final_read = raise_on_final_read

    @property
    def fingerprint(self) -> str:
        self._reads += 1
        if self._reads >= 3:
            if self._raise_on_final_read:
                raise RuntimeError("fingerprint unavailable after response")
            return "drifted-backend"
        return _BACKEND

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse:
        del spec, request
        return BackendResponse(
            output="untrusted output",
            usage=BackendTokenUsage(input_tokens=self._total_tokens - 1, output_tokens=1),
        )


class _TerminalPureBackend:
    def __init__(self, result: object) -> None:
        self._result = result

    @property
    def fingerprint(self) -> str:
        return _BACKEND

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse:
        del spec, request
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("backend_result", "expected_error"),
    [
        (TimeoutError("provider timeout"), ExecutionErrorClass.BACKEND_TIMEOUT),
        ({"malformed": True}, ExecutionErrorClass.INVALID_BACKEND_RESPONSE),
    ],
)
def test_pure_failures_burn_the_full_unknown_usage_reservation(
    tmp_path: Path,
    backend_result: object,
    expected_error: ExecutionErrorClass,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    budget = _budget(per_attempt=16)
    ledger = AttemptLedger(store, ledger_id="pure-reference-fixture", budget=budget)
    runner = PureModelRunner(
        spec=_spec(),
        backend=_TerminalPureBackend(backend_result),
        attempt_ledger=ledger,
    )

    record = runner.execute_record(
        CandidateTask(task_id="task", question="payload"),
        reference_id="b" * 64,
        rollout_seed=17,
        reservation_token_ceiling=16,
    )
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert record.execution.error is not None
    assert record.execution.error.error_class is expected_error
    assert record.execution.usage.total_tokens == 0
    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.reported_tokens == 0
    assert outcome.charged_tokens == 16


def test_pure_known_usage_overrun_is_persisted_and_poisons_ledger(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    budget = _budget(per_attempt=16)
    ledger = AttemptLedger(store, ledger_id="pure-reference-fixture", budget=budget)
    observed_identity = ProviderIdentityObservation(
        requested_model="fixture/exact-model",
        response_model="fixture/exact-model-served-revision",
        system_fingerprint="fp_pure_fixture_snapshot",
        backend_fingerprint=_BACKEND,
    )
    runner = PureModelRunner(
        spec=_spec(),
        backend=_TerminalPureBackend(
            BackendResponse(
                output="too expensive",
                usage=BackendTokenUsage(input_tokens=18, output_tokens=1),
                provider_identity_observation=observed_identity,
            )
        ),
        attempt_ledger=ledger,
    )

    record = runner.execute_record(
        CandidateTask(task_id="task", question="payload"),
        reference_id="b" * 64,
        rollout_seed=17,
        reservation_token_ceiling=16,
    )
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.USAGE_EXCEEDED
    assert record.execution.usage.total_tokens == 19
    assert record.execution.provider_identity_observation == observed_identity
    assert (
        store.get_json(record.execution_ref, PureReferenceExecution).provider_identity_observation
        == observed_identity
    )
    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == 19
    assert outcome.charged_tokens == 19
    assert ledger.state().poisoned is True


@pytest.mark.parametrize("raise_on_final_read", [False, True])
def test_post_response_fingerprint_failure_preserves_usage_and_classification(
    tmp_path: Path,
    raise_on_final_read: bool,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    budget = _budget(per_attempt=16)
    ledger = AttemptLedger(store, ledger_id="pure-reference-fixture", budget=budget)
    runner = PureModelRunner(
        spec=_spec(),
        backend=_FingerprintFailureAfterResponseBackend(
            total_tokens=200,
            raise_on_final_read=raise_on_final_read,
        ),
        attempt_ledger=ledger,
    )

    record = runner.execute_record(
        CandidateTask(task_id="task", question="payload"),
        reference_id="b" * 64,
        rollout_seed=17,
        reservation_token_ceiling=16,
    )
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert record.execution.status is ExecutionStatus.FAILED
    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH
    assert record.execution.usage.total_tokens == 200
    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == 200
    assert outcome.charged_tokens == 200
    assert ledger.state().poisoned is True


def test_trusted_runner_rejects_task_byte_drift_before_backend(tmp_path: Path) -> None:
    frozen_task = CandidateTask(task_id="task", question="payload")
    drifted_task = CandidateTask(task_id="task", question="payload ")
    adapter = _TrustedFixtureAdapter(drifted_task)
    store, ledger, backend, runner = _runner_context(tmp_path)

    with pytest.raises(PureReferenceRunnerError, match="task bytes"):
        TrustedPureReferenceRunner(store, adapter=adapter).execute(
            plan=_plan(frozen_task),
            runner=runner,
            attempt_ledger=ledger,
        )
    assert backend.requests == []
    assert ledger.state().attempts_used == 0


def test_trusted_runner_rejects_model_or_ledger_drift_before_backend(tmp_path: Path) -> None:
    task = CandidateTask(task_id="task", question="payload")
    adapter = _TrustedFixtureAdapter(task)
    other_spec = _spec(model="fixture/different-exact-model")
    store, ledger, backend, runner = _runner_context(tmp_path, spec=other_spec)

    with pytest.raises(PureReferenceRunnerError, match="exact frozen model spec"):
        TrustedPureReferenceRunner(store, adapter=adapter).execute(
            plan=_plan(task),
            runner=runner,
            attempt_ledger=ledger,
        )
    assert backend.requests == []
    assert ledger.state().attempts_used == 0


def test_accounted_execution_loader_rejects_unknown_media_itself(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    unknown_ref = store.put_json(
        {"not": "an execution"},
        media_type="application/vnd.example.unknown-execution+json",
    )

    with pytest.raises(ValueError, match="unsupported execution artifact media type"):
        load_accounted_execution(store, unknown_ref)
