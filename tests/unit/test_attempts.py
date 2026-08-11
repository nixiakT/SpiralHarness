from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import (
    AttemptBudgetExceeded,
    AttemptLedger,
    AttemptLedgerIntegrityError,
    AttemptReservationError,
)
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    CandidateTask,
    ExecutionError,
    ExecutionErrorClass,
    ExecutionStatus,
    ModelExecution,
    ModelRequest,
    ModelUsage,
)
from spiral_harness.storage.artifact_store import ArtifactStore

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def budget(
    *,
    attempts: int = 3,
    total_tokens: int = 30,
    per_attempt: int = 10,
) -> AttemptBudget:
    return AttemptBudget(
        max_attempts=attempts,
        max_total_tokens=total_tokens,
        max_tokens_per_attempt=per_attempt,
    )


def execution(
    *,
    task_id: str = "task-1",
    question: str = "What is 2 + 2?",
    harness_id: str = "harness-1",
    system_prompt: str = "Solve carefully.",
    seed: int = 7,
    input_tokens: int = 3,
    output_tokens: int = 1,
    execution_fingerprint: str = SHA_A,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
) -> ModelExecution:
    candidate_task = CandidateTask(task_id=task_id, question=question)
    request = ModelRequest(
        task_id=task_id,
        harness_id=harness_id,
        system_prompt=system_prompt,
        user_prompt=question,
        seed=seed,
    )
    failed = status is ExecutionStatus.FAILED
    return ModelExecution(
        task=candidate_task,
        request=request,
        output=None if failed else "4",
        status=status,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1.0,
            cost_usd=None,
        ),
        backend_fingerprint="backend@fixed",
        model_fingerprint=SHA_B,
        inference_fingerprint=SHA_C,
        runtime_fingerprint=SHA_D,
        spec_fingerprint="e" * 64,
        execution_fingerprint=execution_fingerprint,
        request_sha256=request.fingerprint,
        error=(
            ExecutionError(
                error_class=ExecutionErrorClass.BACKEND_EXCEPTION,
                detail="builtins.RuntimeError",
            )
            if failed
            else None
        ),
    )


def store_execution(store: ArtifactStore, value: ModelExecution) -> ArtifactRef:
    return store.put_json(value, media_type=MODEL_EXECUTION_MEDIA_TYPE)


def reserve(ledger: AttemptLedger, value: ModelExecution) -> ArtifactRef:
    return ledger.reserve(
        task_fingerprint=value.task.fingerprint,
        execution_fingerprint=value.execution_fingerprint,
        request_sha256=value.request_sha256,
    )


def test_reservation_is_persisted_before_settlement_and_usage_is_derived(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run-1", budget=budget())
    completed = execution(input_tokens=3, output_tokens=1)

    reservation_ref = reserve(ledger, completed)
    pending = ledger.state()

    assert reservation_ref.media_type == ATTEMPT_RESERVATION_MEDIA_TYPE
    assert pending.pending_reservation_ref == reservation_ref
    assert pending.poisoned is False
    assert pending.attempts_used == 1
    assert pending.completed_attempts == 0
    assert pending.charged_tokens == 0
    assert pending.encumbered_tokens == 10
    reservation = store.get_json(reservation_ref, AttemptReservation)
    assert reservation.request_sha256 == completed.request_sha256
    assert reservation.budget_fingerprint == ledger.budget.fingerprint

    outcome_ref = ledger.settle(
        reservation_ref,
        execution_ref=store_execution(store, completed),
    )
    state = ledger.state()
    outcome = store.get_json(outcome_ref, AttemptOutcome)

    assert outcome_ref.media_type == ATTEMPT_OUTCOME_MEDIA_TYPE
    assert outcome.disposition is AttemptDisposition.SETTLED
    assert outcome.reported_tokens == outcome.charged_tokens == 4
    assert state.pending_reservation_ref is None
    assert state.completed_attempts == 1
    assert state.charged_tokens == state.encumbered_tokens == 4
    assert state.remaining_tokens == 26

    reopened = AttemptLedger(
        store,
        ledger_id="run-1",
        budget=budget(),
        tail_ref=outcome_ref,
    )
    assert reopened.state() == state


def test_burn_without_execution_conservatively_charges_full_reservation(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    failed = execution(status=ExecutionStatus.FAILED)
    reservation_ref = reserve(ledger, failed)

    outcome_ref = ledger.burn(
        reservation_ref,
        error_class="execution_persistence_failure",
    )
    outcome = store.get_json(outcome_ref, AttemptOutcome)

    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.reported_tokens == 0
    assert outcome.charged_tokens == 10
    assert outcome.execution_ref is None
    assert ledger.state().remaining_tokens == 20


def test_burn_derives_usage_from_failed_execution(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    failed = execution(
        status=ExecutionStatus.FAILED,
        input_tokens=3,
        output_tokens=2,
    )
    reservation_ref = reserve(ledger, failed)

    outcome_ref = ledger.burn(
        reservation_ref,
        error_class="backend_exception",
        execution_ref=store_execution(store, failed),
    )
    outcome = store.get_json(outcome_ref, AttemptOutcome)

    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.reported_tokens == 5
    assert outcome.charged_tokens == 10


def test_caller_cannot_override_verified_execution_usage(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    failed = execution(status=ExecutionStatus.FAILED, input_tokens=4, output_tokens=2)
    reservation_ref = reserve(ledger, failed)

    with pytest.raises(AttemptReservationError, match="does not match verified"):
        ledger.burn(
            reservation_ref,
            error_class="backend_exception",
            reported_tokens=0,
            execution_ref=store_execution(store, failed),
        )

    assert ledger.state().pending_reservation_ref == reservation_ref


def test_budget_is_rejected_before_a_new_reservation_can_be_published(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(
        store,
        ledger_id="run",
        budget=budget(attempts=2, total_tokens=15, per_attempt=10),
    )
    completed = execution(input_tokens=5, output_tokens=1)
    first = reserve(ledger, completed)
    ledger.settle(first, execution_ref=store_execution(store, completed))

    with pytest.raises(AttemptBudgetExceeded, match="total token budget"):
        reserve(ledger, execution(task_id="task-2"))

    assert ledger.state().attempts_used == 1


def test_hard_attempt_count_is_enforced_after_success(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(
        store,
        ledger_id="run",
        budget=budget(attempts=1, total_tokens=10, per_attempt=10),
    )
    completed = execution()
    first = reserve(ledger, completed)
    ledger.settle(first, execution_ref=store_execution(store, completed))

    with pytest.raises(AttemptBudgetExceeded, match="attempt budget"):
        reserve(ledger, execution(task_id="task-2"))


def test_open_double_stale_and_forged_reservations_fail_closed(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    completed = execution()
    reservation_ref = reserve(ledger, completed)
    result_ref = store_execution(store, completed)

    with pytest.raises(AttemptReservationError, match="open reservation"):
        reserve(ledger, completed)

    forged = reservation_ref.model_copy(update={"size": reservation_ref.size + 1})
    with pytest.raises(AttemptReservationError, match="stale, foreign, forged"):
        ledger.settle(forged, execution_ref=result_ref)

    ledger.settle(reservation_ref, execution_ref=result_ref)
    with pytest.raises(AttemptReservationError, match="already consumed"):
        ledger.settle(reservation_ref, execution_ref=result_ref)
    with pytest.raises(AttemptReservationError, match="already consumed"):
        ledger.burn(reservation_ref, error_class="reuse", execution_ref=result_ref)


def test_completed_usage_overrun_poisons_ledger_and_consumes_actual_tokens(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    overrun = execution(input_tokens=8, output_tokens=3)
    reservation_ref = reserve(ledger, overrun)

    outcome_ref = ledger.settle(
        reservation_ref,
        execution_ref=store_execution(store, overrun),
    )
    outcome = store.get_json(outcome_ref, AttemptOutcome)
    state = ledger.state()

    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == outcome.charged_tokens == 11
    assert state.poisoned is True
    assert state.charged_tokens == 11
    assert state.remaining_tokens == 0
    assert state.remaining_attempts == 0
    with pytest.raises(AttemptBudgetExceeded, match="poisoned"):
        reserve(ledger, execution(task_id="task-2"))

    reopened = AttemptLedger(
        store,
        ledger_id="run",
        budget=budget(),
        tail_ref=outcome_ref,
    )
    assert reopened.state().poisoned is True
    with pytest.raises(AttemptBudgetExceeded, match="poisoned"):
        reserve(reopened, execution(task_id="task-2"))


def test_failed_usage_overrun_burn_also_poisons_ledger(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    overrun = execution(
        status=ExecutionStatus.FAILED,
        input_tokens=9,
        output_tokens=4,
    )
    reservation_ref = reserve(ledger, overrun)

    outcome_ref = ledger.burn(
        reservation_ref,
        error_class="usage_exceeded",
        execution_ref=store_execution(store, overrun),
    )
    outcome = store.get_json(outcome_ref, AttemptOutcome)

    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == outcome.charged_tokens == 13
    assert ledger.state().poisoned is True


def test_same_media_type_arbitrary_json_cannot_settle(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    completed = execution()
    reservation_ref = reserve(ledger, completed)
    forged_ref = store.put_json(
        {"not": "a ModelExecution", "claimed_usage": 999_999},
        media_type=MODEL_EXECUTION_MEDIA_TYPE,
    )

    with pytest.raises(AttemptReservationError, match="canonical ModelExecution"):
        ledger.settle(reservation_ref, execution_ref=forged_ref)

    assert ledger.state().pending_reservation_ref == reservation_ref
    assert ledger.state().charged_tokens == 0


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        (execution(task_id="other-task"), "task does not match"),
        (execution(execution_fingerprint="f" * 64), "fingerprint does not match"),
        (execution(harness_id="other-harness"), "request does not match"),
    ],
)
def test_settlement_rejects_execution_from_another_context(
    tmp_path,
    replacement: ModelExecution,
    error: str,
) -> None:
    store = ArtifactStore(tmp_path / replacement.task_id)
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    expected = execution()
    reservation_ref = reserve(ledger, expected)

    with pytest.raises(AttemptReservationError, match=error):
        ledger.settle(
            reservation_ref,
            execution_ref=store_execution(store, replacement),
        )

    assert ledger.state().pending_reservation_ref == reservation_ref


def test_failed_execution_cannot_be_settled(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    failed = execution(status=ExecutionStatus.FAILED)
    reservation_ref = reserve(ledger, failed)

    with pytest.raises(AttemptReservationError, match="completed execution"):
        ledger.settle(
            reservation_ref,
            execution_ref=store_execution(store, failed),
        )

    assert ledger.state().pending_reservation_ref == reservation_ref


@pytest.mark.parametrize(
    ("disposition", "status", "tokens", "charged_tokens", "error_class"),
    [
        (AttemptDisposition.SETTLED, ExecutionStatus.COMPLETED, 4, 4, None),
        (AttemptDisposition.BURNED, ExecutionStatus.FAILED, 4, 10, "backend_exception"),
        (AttemptDisposition.POISONED, ExecutionStatus.FAILED, 13, 13, "usage_exceeded"),
    ],
)
def test_replay_checks_execution_binding_for_every_disposition(
    tmp_path,
    disposition: AttemptDisposition,
    status: ExecutionStatus,
    tokens: int,
    charged_tokens: int,
    error_class: str | None,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    configured_budget = budget()
    ledger = AttemptLedger(store, ledger_id="run", budget=configured_budget)
    expected = execution()
    reservation_ref = reserve(ledger, expected)
    replacement = execution(
        task_id="other-task",
        status=status,
        input_tokens=tokens - 1,
        output_tokens=1,
    )
    replacement_ref = store_execution(store, replacement)
    forged_outcome = AttemptOutcome(
        ledger_id="run",
        budget_fingerprint=configured_budget.fingerprint,
        sequence=0,
        reservation_ref=reservation_ref,
        disposition=disposition,
        reported_tokens=tokens,
        charged_tokens=charged_tokens,
        execution_ref=replacement_ref,
        error_class=error_class,
    )
    forged_tail = store.put_json(
        forged_outcome,
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )

    with pytest.raises(AttemptLedgerIntegrityError, match="cannot be verified"):
        AttemptLedger(
            store,
            ledger_id="run",
            budget=configured_budget,
            tail_ref=forged_tail,
        )


@pytest.mark.parametrize(
    ("disposition", "status", "reported_tokens", "charged_tokens", "error_class"),
    [
        (AttemptDisposition.SETTLED, ExecutionStatus.COMPLETED, 0, 0, None),
        (AttemptDisposition.BURNED, ExecutionStatus.FAILED, 0, 10, "backend_exception"),
        (AttemptDisposition.POISONED, ExecutionStatus.FAILED, 11, 11, "usage_exceeded"),
    ],
)
def test_replay_checks_bound_execution_usage_for_every_disposition(
    tmp_path,
    disposition: AttemptDisposition,
    status: ExecutionStatus,
    reported_tokens: int,
    charged_tokens: int,
    error_class: str | None,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    configured_budget = budget()
    ledger = AttemptLedger(store, ledger_id="run", budget=configured_budget)
    actual = execution(
        status=status,
        input_tokens=8 if disposition is AttemptDisposition.POISONED else 3,
        output_tokens=4 if disposition is AttemptDisposition.POISONED else 2,
    )
    reservation_ref = reserve(ledger, actual)
    execution_ref = store_execution(store, actual)
    forged_outcome = AttemptOutcome(
        ledger_id="run",
        budget_fingerprint=configured_budget.fingerprint,
        sequence=0,
        reservation_ref=reservation_ref,
        disposition=disposition,
        reported_tokens=reported_tokens,
        charged_tokens=charged_tokens,
        execution_ref=execution_ref,
        error_class=error_class,
    )
    forged_tail = store.put_json(
        forged_outcome,
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )

    with pytest.raises(AttemptLedgerIntegrityError, match="does not match its execution"):
        AttemptLedger(
            store,
            ledger_id="run",
            budget=configured_budget,
            tail_ref=forged_tail,
        )


def test_foreign_execution_media_type_is_not_accepted(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(store, ledger_id="run", budget=budget())
    completed = execution()
    reservation_ref = reserve(ledger, completed)
    wrong_ref = store.put_json(completed)

    with pytest.raises(AttemptReservationError, match="wrong media type"):
        ledger.settle(reservation_ref, execution_ref=wrong_ref)

    assert ledger.state().pending_reservation_ref == reservation_ref


def test_reopen_rejects_foreign_ledger_or_changed_budget(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    original_budget = budget()
    ledger = AttemptLedger(store, ledger_id="run", budget=original_budget)
    completed = execution(input_tokens=1, output_tokens=1)
    reservation_ref = reserve(ledger, completed)
    outcome_ref = ledger.settle(
        reservation_ref,
        execution_ref=store_execution(store, completed),
    )

    with pytest.raises(AttemptLedgerIntegrityError, match="another ledger"):
        AttemptLedger(
            store,
            ledger_id="other-run",
            budget=original_budget,
            tail_ref=outcome_ref,
        )
    with pytest.raises(AttemptLedgerIntegrityError, match="another budget"):
        AttemptLedger(
            store,
            ledger_id="run",
            budget=budget(attempts=4, total_tokens=40, per_attempt=10),
            tail_ref=outcome_ref,
        )


def test_budget_and_artifact_models_are_strict_and_immutable() -> None:
    configured = budget()
    with pytest.raises(ValidationError):
        AttemptBudget(
            max_attempts="3",
            max_total_tokens=30,
            max_tokens_per_attempt=10,
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        budget(total_tokens=5, per_attempt=10)
    with pytest.raises(ValidationError):
        configured.max_attempts = 99
