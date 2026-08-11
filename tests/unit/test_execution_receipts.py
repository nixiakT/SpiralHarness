from __future__ import annotations

from dataclasses import dataclass

import pytest

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    BackendResponse,
    BackendTokenUsage,
    CandidateTask,
    ExecutionError,
    ExecutionErrorClass,
    ExecutionStatus,
    FrozenModelSpec,
    InferenceConfig,
    ModelExecution,
    ModelRequest,
    ModelUsage,
    PromptHarness,
)
from spiral_harness.execution.model import (
    FixedModelRunner,
    ReplayBackend,
)
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    ExecutionReceiptIntegrityError,
    ScheduledExecutionRecord,
    execute_scheduled_attempt,
    publish_execution_receipt,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationCellKey,
    EvaluationPhase,
    EvaluationSide,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.storage.artifact_store import ArtifactStore

SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
RUNNER_BACKEND_FINGERPRINT = "scheduled-replay@sha256:fixed-v1"


def batch(**updates: object) -> EvaluationBatchSchedule:
    values: dict[str, object] = {
        "study": "receipt-study",
        "kind": "evidence-targeted",
        "phase": EvaluationPhase.GATE,
        "query": 2,
        "master_seed": 991,
        "parent_harness_id": "parent-v3",
        "candidate_harness_id": "candidate-v7",
        "task_ids": ("task-1",),
        "search_runs": (101,),
        "repeat_seeds": (11,),
        "max_attempts_per_cell": 2,
        "token_ceiling_per_attempt": 10,
    }
    values.update(updates)
    return EvaluationBatchSchedule(**values)


def ledger_for(
    store: ArtifactStore,
    schedule: EvaluationBatchSchedule,
    *,
    extra_attempts: int = 0,
) -> AttemptLedger:
    attempts = schedule.required_attempts + extra_attempts
    return AttemptLedger(
        store,
        ledger_id="receipt-ledger",
        budget=AttemptBudget(
            max_attempts=attempts,
            max_total_tokens=attempts * schedule.token_ceiling_per_attempt,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        ),
    )


def runner_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint=RUNNER_BACKEND_FINGERPRINT,
        model="hosted/test-model",
        revision="snapshot-2026-08-11",
        tokenizer="provider/test-tokenizer",
        tokenizer_revision="snapshot-2026-08-11",
        runtime="test-runtime@sha256:fixed-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
        ),
    )


def prompt_harness(schedule: EvaluationBatchSchedule, side: EvaluationSide) -> PromptHarness:
    return PromptHarness.from_prompt(
        harness_id=schedule.harness_id_for(side),
        system_prompt=f"Prompt for {side.value}",
    )


def persist_preflight(
    store: ArtifactStore,
    schedule: EvaluationBatchSchedule,
    ledger: AttemptLedger,
) -> ArtifactRef:
    certificate = preflight_attempt_budget(schedule, ledger.state())
    return publish_schedule_preflight(store, certificate)


def execution_for(
    schedule: EvaluationBatchSchedule,
    cell: EvaluationCellKey,
    *,
    attempt_index: int,
    status: ExecutionStatus,
    input_tokens: int,
    output_tokens: int,
) -> ModelExecution:
    question = f"Question for {cell.task_id}?"
    task = CandidateTask(task_id=cell.task_id, question=question)
    request = ModelRequest(
        task_id=cell.task_id,
        harness_id=schedule.harness_id_for(cell.side),
        system_prompt=f"System prompt for {cell.side.value}",
        user_prompt=question,
        seed=schedule.seed_for(cell, attempt_index=attempt_index),
    )
    failed = status is ExecutionStatus.FAILED
    execution_fingerprint = canonical_sha256(
        {
            "schema": "test/receipt-execution/v1",
            "cell": cell,
            "attempt_index": attempt_index,
            "request": request,
            "status": status,
            "tokens": (input_tokens, output_tokens),
        }
    )
    return ModelExecution(
        task=task,
        request=request,
        output=None if failed else "answer",
        status=status,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1.0,
            cost_usd=None,
        ),
        backend_fingerprint="replay@fixed",
        model_fingerprint=SHA_B,
        inference_fingerprint=SHA_C,
        runtime_fingerprint=SHA_D,
        spec_fingerprint=SHA_E,
        execution_fingerprint=execution_fingerprint,
        request_sha256=request.fingerprint,
        error=(
            ExecutionError(
                error_class=ExecutionErrorClass.BACKEND_TIMEOUT,
                detail="builtins.TimeoutError",
            )
            if failed
            else None
        ),
    )


@dataclass(frozen=True)
class RecordedAttempt:
    cell: EvaluationCellKey
    attempt_index: int
    reservation_ref: ArtifactRef
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    receipt_ref: ArtifactRef


def record_attempt(
    store: ArtifactStore,
    ledger: AttemptLedger,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    cell: EvaluationCellKey,
    *,
    attempt_index: int,
    status: ExecutionStatus,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> RecordedAttempt:
    execution = execution_for(
        schedule,
        cell,
        attempt_index=attempt_index,
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    reservation_ref = ledger.reserve(
        task_fingerprint=execution.task.fingerprint,
        execution_fingerprint=execution.execution_fingerprint,
        request_sha256=execution.request_sha256,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    execution_ref = store.put_json(execution, media_type=MODEL_EXECUTION_MEDIA_TYPE)
    if status is ExecutionStatus.COMPLETED:
        outcome_ref = ledger.settle(reservation_ref, execution_ref=execution_ref)
    else:
        outcome_ref = ledger.burn(
            reservation_ref,
            error_class="backend_timeout",
            execution_ref=execution_ref,
        )
    receipt_ref = publish_execution_receipt(
        store,
        schedule=schedule,
        cell=cell,
        attempt_index=attempt_index,
        preflight_ref=preflight_ref,
        reservation_ref=reservation_ref,
        outcome_ref=outcome_ref,
        ledger_tail_ref=outcome_ref,
    )
    return RecordedAttempt(
        cell=cell,
        attempt_index=attempt_index,
        reservation_ref=reservation_ref,
        execution_ref=execution_ref,
        outcome_ref=outcome_ref,
        receipt_ref=receipt_ref,
    )


def complete_paired_retry_run(
    store: ArtifactStore,
) -> tuple[
    EvaluationBatchSchedule,
    AttemptLedger,
    ArtifactRef,
    tuple[RecordedAttempt, ...],
]:
    schedule = batch()
    ledger = ledger_for(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)
    cells = tuple(schedule.iter_cells())
    parent = next(cell for cell in cells if cell.side is EvaluationSide.PARENT)
    candidate = next(cell for cell in cells if cell.side is EvaluationSide.CANDIDATE)
    records = (
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            parent,
            attempt_index=0,
            status=ExecutionStatus.FAILED,
        ),
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            candidate,
            attempt_index=0,
            status=ExecutionStatus.FAILED,
        ),
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            parent,
            attempt_index=1,
            status=ExecutionStatus.COMPLETED,
            input_tokens=3,
            output_tokens=1,
        ),
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            candidate,
            attempt_index=1,
            status=ExecutionStatus.COMPLETED,
            input_tokens=3,
            output_tokens=2,
        ),
    )
    return schedule, ledger, preflight_ref, records


def test_receipt_replay_uses_charged_outcomes_including_timeout_burn(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule, ledger, preflight_ref, records = complete_paired_retry_run(store)

    usage = replay_trusted_usage(
        store,
        schedule=schedule,
        preflight_ref=preflight_ref,
        attempt_ledger=ledger,
        receipt_refs=(record.receipt_ref for record in reversed(records)),
    )

    assert usage.cell_count == 2
    assert usage.attempt_count == 4
    assert usage.settled_attempts == 2
    assert usage.burned_attempts == 2
    assert usage.poisoned_attempts == 0
    assert usage.reported_tokens == 9
    assert usage.charged_tokens == usage.tokens == 29
    assert usage.ledger_tail_refs == (ledger.tail_ref,)
    assert all(ref.media_type == EXECUTION_RECEIPT_MEDIA_TYPE for ref in usage.receipt_refs)

    first = store.get_json(records[0].receipt_ref, ExecutionReceipt)
    assert first.reported_tokens == 0
    assert first.charged_tokens == schedule.token_ceiling_per_attempt
    assert first.preflight_ref == preflight_ref
    assert first.cell_fingerprint == first.cell.fingerprint


def test_missing_duplicate_and_asymmetric_retry_receipts_fail_closed(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule, ledger, preflight_ref, records = complete_paired_retry_run(store)
    refs = tuple(record.receipt_ref for record in records)

    with pytest.raises(ExecutionReceiptIntegrityError, match="missing"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=(records[0].receipt_ref, records[2].receipt_ref),
        )
    with pytest.raises(ExecutionReceiptIntegrityError, match="duplicate"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=(*refs, refs[0]),
        )
    with pytest.raises(ExecutionReceiptIntegrityError, match="exactly paired"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=refs[:-1],
        )


def test_receipt_token_claim_tampering_is_rejected_against_outcome(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule, ledger, preflight_ref, records = complete_paired_retry_run(store)
    original = store.get_json(records[0].receipt_ref, ExecutionReceipt)
    tampered = original.model_copy(update={"charged_tokens": 0})
    tampered_ref = store.put_json(tampered, media_type=EXECUTION_RECEIPT_MEDIA_TYPE)
    refs = (tampered_ref, *(record.receipt_ref for record in records[1:]))

    with pytest.raises(ExecutionReceiptIntegrityError, match="charged tokens"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=refs,
        )


def test_side_swap_and_cross_task_relabeling_are_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule, _, preflight_ref, records = complete_paired_retry_run(store)
    parent_record = records[0]
    candidate_cell = next(
        cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.CANDIDATE
    )

    with pytest.raises(ExecutionReceiptIntegrityError, match="frozen receipt side"):
        publish_execution_receipt(
            store,
            schedule=schedule,
            cell=candidate_cell,
            attempt_index=0,
            preflight_ref=preflight_ref,
            reservation_ref=parent_record.reservation_ref,
            outcome_ref=parent_record.outcome_ref,
            ledger_tail_ref=parent_record.outcome_ref,
        )

    two_task_schedule = batch(
        task_ids=("task-1", "task-2"),
        max_attempts_per_cell=1,
    )
    two_task_ledger = ledger_for(store, two_task_schedule)
    two_task_preflight = persist_preflight(store, two_task_schedule, two_task_ledger)
    task_one_parent = next(
        cell
        for cell in two_task_schedule.iter_cells()
        if cell.task_id == "task-1" and cell.side is EvaluationSide.PARENT
    )
    task_two_parent = next(
        cell
        for cell in two_task_schedule.iter_cells()
        if cell.task_id == "task-2" and cell.side is EvaluationSide.PARENT
    )
    task_one = record_attempt(
        store,
        two_task_ledger,
        two_task_schedule,
        two_task_preflight,
        task_one_parent,
        attempt_index=0,
        status=ExecutionStatus.COMPLETED,
        input_tokens=2,
        output_tokens=1,
    )
    with pytest.raises(ExecutionReceiptIntegrityError, match="execution task"):
        publish_execution_receipt(
            store,
            schedule=two_task_schedule,
            cell=task_two_parent,
            attempt_index=0,
            preflight_ref=two_task_preflight,
            reservation_ref=task_one.reservation_ref,
            outcome_ref=task_one.outcome_ref,
            ledger_tail_ref=task_one.outcome_ref,
        )


def test_unreceipted_burn_after_preflight_is_not_silently_omitted(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule, extra_attempts=1)
    preflight_ref = persist_preflight(store, schedule, ledger)

    # This conservative failure is in the exact post-preflight ledger suffix,
    # but deliberately receives no cell receipt.
    foreign_task = CandidateTask(task_id="foreign", question="not in schedule")
    foreign_request = ModelRequest(
        task_id="foreign",
        harness_id="foreign",
        system_prompt="foreign",
        user_prompt=foreign_task.question,
        seed=1,
    )
    unreceipted = ledger.reserve(
        task_fingerprint=foreign_task.fingerprint,
        execution_fingerprint="f" * 64,
        request_sha256=foreign_request.fingerprint,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    ledger.burn(unreceipted, error_class="timeout-before-persistence")

    records = tuple(
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            cell,
            attempt_index=0,
            status=ExecutionStatus.COMPLETED,
            input_tokens=2,
            output_tokens=1,
        )
        for cell in schedule.iter_cells()
    )
    with pytest.raises(ExecutionReceiptIntegrityError, match="exactly cover every outcome"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=tuple(record.receipt_ref for record in records),
        )


def test_unreceipted_trailing_burn_cannot_be_hidden_with_a_stale_receipt_tail(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule, extra_attempts=1)
    preflight_ref = persist_preflight(store, schedule, ledger)
    records = tuple(
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            cell,
            attempt_index=0,
            status=ExecutionStatus.COMPLETED,
            input_tokens=2,
            output_tokens=1,
        )
        for cell in schedule.iter_cells()
    )

    foreign_task = CandidateTask(task_id="trailing", question="trailing")
    foreign_request = ModelRequest(
        task_id="trailing",
        harness_id="trailing",
        system_prompt="trailing",
        user_prompt="trailing",
        seed=2,
    )
    trailing = ledger.reserve(
        task_fingerprint=foreign_task.fingerprint,
        execution_fingerprint="9" * 64,
        request_sha256=foreign_request.fingerprint,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    ledger.burn(trailing, error_class="trailing-persistence-failure")

    with pytest.raises(ExecutionReceiptIntegrityError, match="exactly cover every outcome"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=tuple(record.receipt_ref for record in records),
        )


def test_preflight_start_tail_excludes_verified_prior_ledger_history(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule, extra_attempts=1)

    prior_task = CandidateTask(task_id="prior", question="prior")
    prior_request = ModelRequest(
        task_id="prior",
        harness_id="prior",
        system_prompt="prior",
        user_prompt="prior",
        seed=0,
    )
    prior = ledger.reserve(
        task_fingerprint=prior_task.fingerprint,
        execution_fingerprint="a" * 64,
        request_sha256=prior_request.fingerprint,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    prior_tail = ledger.burn(prior, error_class="prior-persistence-failure")
    preflight_ref = persist_preflight(store, schedule, ledger)

    records = tuple(
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            cell,
            attempt_index=0,
            status=ExecutionStatus.COMPLETED,
            input_tokens=2,
            output_tokens=1,
        )
        for cell in schedule.iter_cells()
    )
    usage = replay_trusted_usage(
        store,
        schedule=schedule,
        preflight_ref=preflight_ref,
        attempt_ledger=ledger,
        receipt_refs=tuple(record.receipt_ref for record in records),
    )

    assert usage.attempt_count == 2
    assert usage.charged_tokens == 6
    preflight = store.get_json(preflight_ref)
    assert preflight["ledger_tail_ref"]["sha256"] == prior_tail.sha256


def test_scheduled_attempts_bind_each_atomic_record_to_serial_receipt_progress(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)

    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="answer",
            usage=BackendTokenUsage(input_tokens=3, output_tokens=1),
        ),
    )
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    cells = tuple(schedule.iter_cells())
    candidate_task = CandidateTask(task_id="task-1", question="What is 2 + 2?")
    records: list[ScheduledExecutionRecord] = []
    expected_tail: ArtifactRef | None = None
    previous_receipt_ref: ArtifactRef | None = None
    for cell in cells:
        record = execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=expected_tail,
            previous_receipt_ref=previous_receipt_ref,
            cell=cell,
            attempt_index=0,
            task=candidate_task,
            harness=prompt_harness(schedule, cell.side),
        )
        records.append(record)
        expected_tail = record.outcome_ref
        previous_receipt_ref = record.receipt_ref

    assert len(backend.calls) == 2
    assert len({record.execution_ref.sha256 for record in records}) == 2
    assert len({record.outcome_ref.sha256 for record in records}) == 2
    for record in records:
        outcome = store.get_json(record.outcome_ref)
        assert outcome["execution_ref"]["sha256"] == record.execution_ref.sha256
        assert record.receipt.execution_ref == record.execution_ref
        assert record.receipt.outcome_ref == record.outcome_ref
        assert record.receipt.reservation_ref == record.reservation_ref

    assert (runner.last_execution_ref, runner.last_outcome_ref) in {
        (record.execution_ref, record.outcome_ref) for record in records
    }
    usage = replay_trusted_usage(
        store,
        schedule=schedule,
        preflight_ref=preflight_ref,
        attempt_ledger=ledger,
        receipt_refs=tuple(record.receipt_ref for record in records),
    )
    assert usage.attempt_count == 2
    assert usage.charged_tokens == 8


def test_scheduled_timeout_receipt_charges_full_reservation(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = AttemptLedger(
        store,
        ledger_id="wide-ledger",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=200,
            max_tokens_per_attempt=100,
        ),
    )
    preflight_ref = persist_preflight(store, schedule, ledger)

    class TimeoutBackend:
        fingerprint = RUNNER_BACKEND_FINGERPRINT

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, *, spec: FrozenModelSpec, request: ModelRequest) -> BackendResponse:
            del spec, request
            self.calls += 1
            raise TimeoutError("controlled timeout")

    backend = TimeoutBackend()
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)
    result = execute_scheduled_attempt(
        runner=runner,
        schedule=schedule,
        preflight_ref=preflight_ref,
        expected_previous_ledger_tail_ref=None,
        cell=parent,
        attempt_index=0,
        task=CandidateTask(task_id="task-1", question="What is 2 + 2?"),
        harness=prompt_harness(schedule, EvaluationSide.PARENT),
    )

    assert backend.calls == 1
    assert result.execution.status is ExecutionStatus.FAILED
    assert result.execution.usage.total_tokens == 0
    assert result.receipt.reported_tokens == 0
    assert result.receipt.charged_tokens == schedule.token_ceiling_per_attempt
    reservation = store.get_json(result.reservation_ref)
    assert reservation["reserved_tokens"] == schedule.token_ceiling_per_attempt


def test_schedule_ceiling_overrun_poisons_wider_ledger_and_stops_before_retry(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = AttemptLedger(
        store,
        ledger_id="wide-overrun-ledger",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=200,
            max_tokens_per_attempt=100,
        ),
    )
    preflight_ref = persist_preflight(store, schedule, ledger)
    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="over schedule ceiling",
            usage=BackendTokenUsage(input_tokens=3, output_tokens=8),
        ),
    )
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    cells = tuple(schedule.iter_cells())
    parent = next(cell for cell in cells if cell.side is EvaluationSide.PARENT)
    candidate = next(cell for cell in cells if cell.side is EvaluationSide.CANDIDATE)
    item = CandidateTask(task_id="task-1", question="What is 2 + 2?")

    first = execute_scheduled_attempt(
        runner=runner,
        schedule=schedule,
        preflight_ref=preflight_ref,
        expected_previous_ledger_tail_ref=None,
        cell=parent,
        attempt_index=0,
        task=item,
        harness=prompt_harness(schedule, EvaluationSide.PARENT),
    )

    reservation = store.get_json(first.reservation_ref, AttemptReservation)
    outcome = store.get_json(first.outcome_ref, AttemptOutcome)
    assert reservation.reserved_tokens == schedule.token_ceiling_per_attempt == 10
    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == outcome.charged_tokens == 11
    assert ledger.state().poisoned

    with pytest.raises(ExecutionReceiptIntegrityError, match="poisoned"):
        execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=first.outcome_ref,
            previous_receipt_ref=first.receipt_ref,
            cell=candidate,
            attempt_index=0,
            task=item,
            harness=prompt_harness(schedule, EvaluationSide.CANDIDATE),
        )
    assert len(backend.calls) == 1


def test_stale_preflight_is_rejected_before_scheduled_backend_call(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule, extra_attempts=1)
    preflight_ref = persist_preflight(store, schedule, ledger)

    foreign_task = CandidateTask(task_id="foreign", question="foreign")
    foreign_request = ModelRequest(
        task_id="foreign",
        harness_id="foreign",
        system_prompt="foreign",
        user_prompt="foreign",
        seed=1,
    )
    foreign_reservation = ledger.reserve(
        task_fingerprint=foreign_task.fingerprint,
        execution_fingerprint="f" * 64,
        request_sha256=foreign_request.fingerprint,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    ledger.burn(foreign_reservation, error_class="foreign-attempt")

    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="must not run",
            usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
        ),
    )
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)

    with pytest.raises(ExecutionReceiptIntegrityError, match="advanced"):
        execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=None,
            cell=parent,
            attempt_index=0,
            task=CandidateTask(task_id="task-1", question="What is 2 + 2?"),
            harness=prompt_harness(schedule, EvaluationSide.PARENT),
        )
    assert backend.calls == ()


def test_forged_previous_receipt_cannot_authorize_a_foreign_tail_backend_call(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule, extra_attempts=1)
    preflight_ref = persist_preflight(store, schedule, ledger)

    foreign_task = CandidateTask(task_id="foreign", question="foreign")
    foreign_request = ModelRequest(
        task_id="foreign",
        harness_id="foreign",
        system_prompt="foreign",
        user_prompt="foreign",
        seed=1,
    )
    foreign_reservation_ref = ledger.reserve(
        task_fingerprint=foreign_task.fingerprint,
        execution_fingerprint="f" * 64,
        request_sha256=foreign_request.fingerprint,
        token_ceiling=schedule.token_ceiling_per_attempt,
    )
    foreign_tail = ledger.burn(
        foreign_reservation_ref,
        error_class="foreign-attempt",
    )
    foreign_outcome = store.get_json(foreign_tail, AttemptOutcome)

    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)
    decoy_execution = execution_for(
        schedule,
        parent,
        attempt_index=0,
        status=ExecutionStatus.COMPLETED,
        input_tokens=1,
        output_tokens=1,
    )
    decoy_execution_ref = store.put_json(
        decoy_execution,
        media_type=MODEL_EXECUTION_MEDIA_TYPE,
    )
    forged_receipt = ExecutionReceipt(
        schedule_fingerprint=schedule.fingerprint,
        preflight_ref=preflight_ref,
        preflight_fingerprint=preflight_ref.sha256,
        cell=parent,
        cell_fingerprint=parent.fingerprint,
        attempt_index=0,
        reservation_ref=foreign_reservation_ref,
        execution_ref=decoy_execution_ref,
        outcome_ref=foreign_tail,
        ledger_tail_ref=foreign_tail,
        reported_tokens=foreign_outcome.reported_tokens,
        charged_tokens=foreign_outcome.charged_tokens,
    )
    forged_receipt_ref = store.put_json(
        forged_receipt,
        media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
    )
    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="must not run",
            usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
        ),
    )
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )

    with pytest.raises(ExecutionReceiptIntegrityError, match="persisted ModelExecution"):
        execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=foreign_tail,
            previous_receipt_ref=forged_receipt_ref,
            cell=parent,
            attempt_index=0,
            task=CandidateTask(task_id="task-1", question="What is 2 + 2?"),
            harness=prompt_harness(schedule, EvaluationSide.PARENT),
        )
    assert backend.calls == ()
    assert ledger.tail_ref == foreign_tail


def test_receipt_publish_and_replay_reject_reservation_above_preflight_ceiling(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = AttemptLedger(
        store,
        ledger_id="forged-wide-ledger",
        budget=AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=200,
            max_tokens_per_attempt=100,
        ),
    )
    preflight_ref = persist_preflight(store, schedule, ledger)
    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)
    execution = execution_for(
        schedule,
        parent,
        attempt_index=0,
        status=ExecutionStatus.COMPLETED,
        input_tokens=2,
        output_tokens=1,
    )
    reservation_ref = ledger.reserve(
        task_fingerprint=execution.task.fingerprint,
        execution_fingerprint=execution.execution_fingerprint,
        request_sha256=execution.request_sha256,
        token_ceiling=100,
    )
    execution_ref = store.put_json(execution, media_type=MODEL_EXECUTION_MEDIA_TYPE)
    outcome_ref = ledger.settle(reservation_ref, execution_ref=execution_ref)

    with pytest.raises(ExecutionReceiptIntegrityError, match="preflight token ceiling"):
        publish_execution_receipt(
            store,
            schedule=schedule,
            cell=parent,
            attempt_index=0,
            preflight_ref=preflight_ref,
            reservation_ref=reservation_ref,
            outcome_ref=outcome_ref,
            ledger_tail_ref=outcome_ref,
        )

    outcome = store.get_json(outcome_ref, AttemptOutcome)
    forged_receipt = ExecutionReceipt(
        schedule_fingerprint=schedule.fingerprint,
        preflight_ref=preflight_ref,
        preflight_fingerprint=preflight_ref.sha256,
        cell=parent,
        cell_fingerprint=parent.fingerprint,
        attempt_index=0,
        reservation_ref=reservation_ref,
        execution_ref=execution_ref,
        outcome_ref=outcome_ref,
        ledger_tail_ref=outcome_ref,
        reported_tokens=outcome.reported_tokens,
        charged_tokens=outcome.charged_tokens,
    )
    forged_receipt_ref = store.put_json(
        forged_receipt,
        media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
    )
    with pytest.raises(ExecutionReceiptIntegrityError, match="preflight token ceiling"):
        replay_trusted_usage(
            store,
            schedule=schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=(forged_receipt_ref,),
        )


def test_scheduled_wrapper_rejects_fixed_runner_subclasses_before_backend_call(
    tmp_path,
) -> None:
    class ForgedRunner(FixedModelRunner):
        pass

    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)
    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="must not run",
            usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
        ),
    )
    runner = ForgedRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)

    with pytest.raises(TypeError, match="exact FixedModelRunner"):
        execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=None,
            cell=parent,
            attempt_index=0,
            task=CandidateTask(task_id="task-1", question="What is 2 + 2?"),
            harness=prompt_harness(schedule, EvaluationSide.PARENT),
        )
    assert backend.calls == ()


def test_scheduled_side_harness_mismatch_is_rejected_before_backend_call(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    schedule = batch(max_attempts_per_cell=1)
    ledger = ledger_for(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)
    backend = ReplayBackend(
        fingerprint=RUNNER_BACKEND_FINGERPRINT,
        default_response=BackendResponse(
            output="answer",
            usage=BackendTokenUsage(input_tokens=2, output_tokens=1),
        ),
    )
    runner = FixedModelRunner(
        spec=runner_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    parent = next(cell for cell in schedule.iter_cells() if cell.side is EvaluationSide.PARENT)
    wrong_harness = PromptHarness.from_prompt(
        harness_id=schedule.candidate_harness_id,
        system_prompt="wrong side",
    )

    with pytest.raises(ExecutionReceiptIntegrityError, match="scheduled side"):
        execute_scheduled_attempt(
            runner=runner,
            schedule=schedule,
            preflight_ref=preflight_ref,
            expected_previous_ledger_tail_ref=None,
            cell=parent,
            attempt_index=0,
            task=CandidateTask(task_id="task-1", question="What is 2 + 2?"),
            harness=wrong_harness,
        )

    assert backend.calls == ()
    assert ledger.state().attempts_used == 0
