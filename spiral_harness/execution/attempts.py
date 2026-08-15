"""Conservative attempt accounting for one process-local writer and audit views."""

from __future__ import annotations

import secrets
from threading import RLock

import spiral_harness.execution.accounted_execution as _execution
import spiral_harness.execution.attempt_checkpoint as _checkpoint
import spiral_harness.execution.contracts as _contracts
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository


class AttemptAccountingError(RuntimeError):
    """Base error for fail-closed attempt accounting."""


class AttemptBudgetExceeded(AttemptAccountingError):
    """Raised before backend execution when no complete reservation fits."""


class AttemptLedgerIntegrityError(AttemptAccountingError):
    """Raised when an immutable accounting chain cannot be verified."""


class AttemptReservationError(AttemptAccountingError):
    """Raised for a stale, foreign, forged, or already-consumed reservation."""


class AttemptLedger:
    """Append-only reservation/outcome accounting for one process-local writer.

    Reservations precede backend calls; failures burn their reservation and an
    overrun poisons the stream. Reopened tails are audit-only. Cross-process
    writers still require durable coordination around this trusted capability.

    A live writer advances one in-memory checkpoint derived from the initial
    full replay and each subsequently verified append. This relies on the
    repository's immutable-reference contract and avoids replaying the complete
    history before every write. Public state reads still replay from storage and
    compare the result with that checkpoint, so audit boundaries remain fresh.
    """

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        ledger_id: str,
        budget: _contracts.AttemptBudget,
        tail_ref: ArtifactRef | None = None,
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        if not isinstance(ledger_id, str):
            raise TypeError("ledger_id must be a string")
        normalized_id = ledger_id.strip()
        if not normalized_id:
            raise ValueError("ledger_id must not be empty")
        self._repository = repository
        self._ledger_id = normalized_id
        self._budget = _contracts.AttemptBudget.model_validate(budget, strict=True)
        self.__writer_epoch_id = secrets.token_hex(32)
        self._lock = RLock()
        initial_tail_ref = (
            None if tail_ref is None else ArtifactRef.model_validate(tail_ref, strict=True)
        )
        self._checkpoint = _checkpoint.AttemptWriterCheckpoint(
            self._replay(initial_tail_ref),
            integrity_error_type=AttemptLedgerIntegrityError,
        )

    @property
    def repository(self) -> ArtifactRepository:
        return self._repository

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def budget(self) -> _contracts.AttemptBudget:
        return self._budget

    @property
    def tail_ref(self) -> ArtifactRef | None:
        with self._lock:
            return self._checkpoint.tail_ref

    def state(self) -> _contracts.AttemptLedgerState:
        with self._lock:
            return self._checkpoint.fresh_current(self._replay)

    def state_at(self, tail_ref: ArtifactRef | None) -> _contracts.AttemptLedgerState:
        with self._lock:
            return self._checkpoint.fresh_at(self._replay, tail_ref)

    def reserve(
        self,
        *,
        task_fingerprint: str,
        execution_fingerprint: str,
        request_sha256: str,
        token_ceiling: int | None = None,
    ) -> ArtifactRef:
        """Persist a single-use reservation before any backend call is allowed."""
        with self._lock:
            state = self._checkpoint.current()
            if state.pending_reservation_ref is not None:
                raise AttemptReservationError("the ledger already has an open reservation")
            if state.tail_ref is not None:
                current_outcome = self._checkpoint.verify_current_outcome(
                    state.tail_ref,
                    load_outcome=self._load_outcome,
                    load_reservation=self._load_reservation,
                    verify_pair=self._verify_pair,
                )
                if current_outcome.writer_epoch_id != self.__writer_epoch_id:
                    raise AttemptReservationError("a reopened ledger is an audit-only view")
            if state.poisoned:
                raise AttemptBudgetExceeded("attempt ledger is poisoned by a token overrun")
            if state.attempts_used >= self._budget.max_attempts:
                raise AttemptBudgetExceeded("attempt budget is exhausted")
            ceiling = (
                self._budget.max_tokens_per_attempt
                if token_ceiling is None
                else self._strict_non_negative_int(token_ceiling, "token_ceiling")
            )
            if ceiling < 1:
                raise ValueError("token_ceiling must be at least one")
            if ceiling > self._budget.max_tokens_per_attempt:
                raise AttemptBudgetExceeded("requested tokens exceed the per-attempt ceiling")
            if state.charged_tokens + ceiling > self._budget.max_total_tokens:
                raise AttemptBudgetExceeded("total token budget cannot fit a full reservation")

            reservation = _contracts.AttemptReservation(
                ledger_id=self._ledger_id,
                writer_epoch_id=self.__writer_epoch_id,
                budget_fingerprint=self._budget.fingerprint,
                sequence=state.attempts_used,
                task_fingerprint=task_fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                reserved_tokens=ceiling,
                previous_outcome_ref=state.tail_ref,
            )
            reservation_ref = self._repository.put_json(
                reservation,
                media_type=_contracts.ATTEMPT_RESERVATION_MEDIA_TYPE,
            )
            checked_ref = self._verify_published(
                reservation_ref,
                expected_media_type=_contracts.ATTEMPT_RESERVATION_MEDIA_TYPE,
                model_type=_contracts.AttemptReservation,
                expected_value=reservation,
            )
            self._checkpoint.append_reservation(
                checked_ref,
                reserved_tokens=ceiling,
            )
            return checked_ref

    def reserve_at_tail(
        self,
        *,
        expected_previous_outcome_ref: ArtifactRef | None,
        task_fingerprint: str,
        execution_fingerprint: str,
        request_sha256: str,
        token_ceiling: int | None = None,
    ) -> ArtifactRef:
        with self._lock:
            checked_expected = (
                None
                if expected_previous_outcome_ref is None
                else ArtifactRef.model_validate(
                    expected_previous_outcome_ref,
                    strict=True,
                )
            )
            if (
                checked_expected is not None
                and checked_expected.media_type != _contracts.ATTEMPT_OUTCOME_MEDIA_TYPE
            ):
                raise AttemptReservationError(
                    "expected previous outcome declares the wrong media type"
                )
            if self._checkpoint.tail_ref != checked_expected:
                raise AttemptReservationError(
                    "attempt ledger advanced beyond the expected scheduled boundary"
                )
            return self.reserve(
                task_fingerprint=task_fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                token_ceiling=token_ceiling,
            )

    def settle(
        self,
        reservation_ref: ArtifactRef,
        *,
        execution_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Consume the current reservation using only verified execution usage."""
        with self._lock:
            checked_reservation_ref, reservation = self._require_current_reservation(
                reservation_ref
            )
            checked_execution_ref, execution = self._load_execution(execution_ref)
            self._verify_execution_binding(reservation, execution)
            if execution.status.value != "completed":
                raise AttemptReservationError("only a completed execution may be settled")
            tokens = execution.usage.total_tokens
            if tokens > reservation.reserved_tokens:
                return self._publish_outcome(
                    reservation_ref=checked_reservation_ref,
                    reservation=reservation,
                    disposition=_contracts.AttemptDisposition.POISONED,
                    reported_tokens=tokens,
                    charged_tokens=tokens,
                    execution_ref=checked_execution_ref,
                    error_class="usage_exceeded",
                )
            return self._publish_outcome(
                reservation_ref=checked_reservation_ref,
                reservation=reservation,
                disposition=_contracts.AttemptDisposition.SETTLED,
                reported_tokens=tokens,
                charged_tokens=tokens,
                execution_ref=checked_execution_ref,
                error_class=None,
            )

    def burn(
        self,
        reservation_ref: ArtifactRef,
        *,
        error_class: str,
        reported_tokens: int | None = None,
        execution_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Burn a failed attempt, poisoning the stream on a known overrun."""
        with self._lock:
            checked_reservation_ref, reservation = self._require_current_reservation(
                reservation_ref
            )
            if not isinstance(error_class, str):
                raise TypeError("error_class must be a string")
            normalized_error = error_class.strip()
            if not normalized_error:
                raise ValueError("error_class must not be empty")
            if reported_tokens is None:
                supplied_tokens = None
            else:
                supplied_tokens = self._strict_non_negative_int(
                    reported_tokens,
                    "reported_tokens",
                )
            if execution_ref is None:
                checked_execution_ref = None
                tokens = 0 if supplied_tokens is None else supplied_tokens
            else:
                checked_execution_ref, execution = self._load_execution(execution_ref)
                self._verify_execution_binding(reservation, execution)
                tokens = execution.usage.total_tokens
                if supplied_tokens is not None and supplied_tokens != tokens:
                    raise AttemptReservationError(
                        "reported_tokens does not match verified execution usage"
                    )
            poisoned = tokens > reservation.reserved_tokens
            return self._publish_outcome(
                reservation_ref=checked_reservation_ref,
                reservation=reservation,
                disposition=(
                    _contracts.AttemptDisposition.POISONED
                    if poisoned
                    else _contracts.AttemptDisposition.BURNED
                ),
                reported_tokens=tokens,
                charged_tokens=tokens if poisoned else reservation.reserved_tokens,
                execution_ref=checked_execution_ref,
                error_class=normalized_error,
            )

    def _publish_outcome(
        self,
        *,
        reservation_ref: ArtifactRef,
        reservation: _contracts.AttemptReservation,
        disposition: _contracts.AttemptDisposition,
        reported_tokens: int,
        charged_tokens: int,
        execution_ref: ArtifactRef | None,
        error_class: str | None,
    ) -> ArtifactRef:
        outcome = _contracts.AttemptOutcome(
            ledger_id=self._ledger_id,
            writer_epoch_id=self.__writer_epoch_id,
            budget_fingerprint=self._budget.fingerprint,
            sequence=reservation.sequence,
            reservation_ref=reservation_ref,
            disposition=disposition,
            reported_tokens=reported_tokens,
            charged_tokens=charged_tokens,
            execution_ref=execution_ref,
            error_class=error_class,
        )
        outcome_ref = self._repository.put_json(
            outcome,
            media_type=_contracts.ATTEMPT_OUTCOME_MEDIA_TYPE,
        )
        checked_ref = self._verify_published(
            outcome_ref,
            expected_media_type=_contracts.ATTEMPT_OUTCOME_MEDIA_TYPE,
            model_type=_contracts.AttemptOutcome,
            expected_value=outcome,
        )
        self._checkpoint.append_outcome(reservation_ref, checked_ref, outcome)
        return checked_ref

    def _require_current_reservation(
        self,
        reservation_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, _contracts.AttemptReservation]:
        state = self._checkpoint.current()
        try:
            checked_ref = ArtifactRef.model_validate(reservation_ref, strict=True)
        except Exception as exc:
            raise AttemptReservationError("reservation reference is malformed") from exc
        if checked_ref.media_type != _contracts.ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise AttemptReservationError("reservation reference declares the wrong media type")
        if state.pending_reservation_ref is None or checked_ref != state.pending_reservation_ref:
            raise AttemptReservationError(
                "reservation is stale, foreign, forged, or already consumed"
            )
        reservation = self._checkpoint.load_current(self._load_reservation, checked_ref)
        if reservation.writer_epoch_id != self.__writer_epoch_id:
            raise AttemptReservationError("a reopened ledger is an audit-only view")
        return checked_ref, reservation

    def _load_execution(
        self,
        execution_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, _execution.AccountedExecution]:
        try:
            checked_ref = ArtifactRef.model_validate(execution_ref, strict=True)
        except Exception as exc:
            raise AttemptReservationError("execution reference is malformed") from exc
        if checked_ref.media_type not in _contracts.ATTEMPT_EXECUTION_MEDIA_TYPES:
            raise AttemptReservationError("execution reference declares the wrong media type")
        try:
            execution = _execution.load_accounted_execution(self._repository, checked_ref)
        except _execution.AccountedExecutionLoadError as exc:
            raise AttemptReservationError(str(exc)) from exc
        return checked_ref, execution

    @staticmethod
    def _verify_execution_binding(
        reservation: _contracts.AttemptReservation,
        execution: _execution.AccountedExecution,
    ) -> None:
        if execution.task.fingerprint != reservation.task_fingerprint:
            raise AttemptReservationError("execution task does not match its reservation")
        if execution.execution_fingerprint != reservation.execution_fingerprint:
            raise AttemptReservationError("execution fingerprint does not match its reservation")
        if execution.request_sha256 != reservation.request_sha256:
            raise AttemptReservationError("execution request does not match its reservation")

    def _verify_published(
        self,
        ref: ArtifactRef,
        *,
        expected_media_type: str,
        model_type: type[_contracts.AttemptReservation] | type[_contracts.AttemptOutcome],
        expected_value: _contracts.AttemptReservation | _contracts.AttemptOutcome,
    ) -> ArtifactRef:
        try:
            checked_ref = ArtifactRef.model_validate(ref, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError("repository returned a malformed reference") from exc
        if checked_ref.media_type != expected_media_type:
            raise AttemptLedgerIntegrityError("repository returned the wrong artifact media type")
        try:
            loaded = self._repository.get_json(checked_ref, model_type)
            checked_value = model_type.model_validate(loaded, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError(
                "published accounting artifact is unreadable"
            ) from exc
        if checked_value != expected_value:
            raise AttemptLedgerIntegrityError("published accounting artifact changed content")
        return checked_ref

    def _replay(self, tail_ref: ArtifactRef | None) -> _contracts.AttemptLedgerState:
        return _checkpoint.replay_attempt_ledger(
            ledger_id=self._ledger_id,
            writer_epoch_id=self.__writer_epoch_id,
            budget=self._budget,
            tail_ref=tail_ref,
            integrity_error_type=AttemptLedgerIntegrityError,
            load_reservation=self._load_reservation,
            load_outcome=self._load_outcome,
            verify_pair=self._verify_pair,
        )

    def _verify_pair(
        self,
        reservation: _contracts.AttemptReservation,
        outcome: _contracts.AttemptOutcome,
    ) -> None:
        if reservation.ledger_id != self._ledger_id or outcome.ledger_id != self._ledger_id:
            raise AttemptLedgerIntegrityError("attempt artifact belongs to another ledger")
        if reservation.writer_epoch_id != outcome.writer_epoch_id:
            raise AttemptLedgerIntegrityError("reservation and outcome writer epochs differ")
        if (
            reservation.budget_fingerprint != self._budget.fingerprint
            or outcome.budget_fingerprint != self._budget.fingerprint
        ):
            raise AttemptLedgerIntegrityError("attempt artifact belongs to another budget")
        if reservation.sequence != outcome.sequence:
            raise AttemptLedgerIntegrityError("reservation and outcome sequences differ")
        if reservation.reserved_tokens > self._budget.max_tokens_per_attempt:
            raise AttemptLedgerIntegrityError("reservation exceeds the per-attempt ceiling")
        if outcome.disposition is _contracts.AttemptDisposition.SETTLED:
            if outcome.reported_tokens > reservation.reserved_tokens:
                raise AttemptLedgerIntegrityError("settled usage exceeds its reservation")
            if outcome.charged_tokens != outcome.reported_tokens:
                raise AttemptLedgerIntegrityError("settled charge does not match actual usage")
        elif outcome.disposition is _contracts.AttemptDisposition.BURNED:
            if outcome.reported_tokens > reservation.reserved_tokens:
                raise AttemptLedgerIntegrityError(
                    "an overrun must poison rather than burn the reservation"
                )
            if outcome.charged_tokens != reservation.reserved_tokens:
                raise AttemptLedgerIntegrityError(
                    "burned attempt did not charge its full reservation"
                )
        else:
            if outcome.reported_tokens <= reservation.reserved_tokens:
                raise AttemptLedgerIntegrityError(
                    "poisoned outcome does not contain a token overrun"
                )
            if outcome.charged_tokens != outcome.reported_tokens:
                raise AttemptLedgerIntegrityError("poisoned charge does not match reported usage")
        if outcome.execution_ref is not None:
            try:
                _, execution = self._load_execution(outcome.execution_ref)
                self._verify_execution_binding(reservation, execution)
            except Exception as exc:
                raise AttemptLedgerIntegrityError(
                    "attempt outcome execution artifact cannot be verified"
                ) from exc
            if execution.usage.total_tokens != outcome.reported_tokens:
                raise AttemptLedgerIntegrityError(
                    "attempt outcome token usage does not match its execution"
                )
            if (
                outcome.disposition is _contracts.AttemptDisposition.SETTLED
                and execution.status.value != "completed"
            ):
                raise AttemptLedgerIntegrityError(
                    "settled outcome does not reference a completed execution"
                )

    def _load_reservation(self, ref: ArtifactRef) -> _contracts.AttemptReservation:
        if ref.media_type != _contracts.ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise AttemptLedgerIntegrityError("reservation ref declares the wrong media type")
        try:
            loaded = self._repository.get_json(ref, _contracts.AttemptReservation)
            reservation = _contracts.AttemptReservation.model_validate(loaded, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError("attempt reservation cannot be verified") from exc
        if reservation.ledger_id != self._ledger_id:
            raise AttemptLedgerIntegrityError("reservation belongs to another ledger")
        if reservation.budget_fingerprint != self._budget.fingerprint:
            raise AttemptLedgerIntegrityError("reservation belongs to another budget")
        if reservation.reserved_tokens > self._budget.max_tokens_per_attempt:
            raise AttemptLedgerIntegrityError("reservation exceeds the per-attempt ceiling")
        return reservation

    def _load_outcome(self, ref: ArtifactRef) -> _contracts.AttemptOutcome:
        if ref.media_type != _contracts.ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise AttemptLedgerIntegrityError("outcome ref declares the wrong media type")
        try:
            loaded = self._repository.get_json(ref, _contracts.AttemptOutcome)
            outcome = _contracts.AttemptOutcome.model_validate(loaded, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError("attempt outcome cannot be verified") from exc
        if outcome.ledger_id != self._ledger_id:
            raise AttemptLedgerIntegrityError("outcome belongs to another ledger")
        if outcome.budget_fingerprint != self._budget.fingerprint:
            raise AttemptLedgerIntegrityError("outcome belongs to another budget")
        return outcome

    @staticmethod
    def _strict_non_negative_int(value: int, field_name: str) -> int:
        if type(value) is not int:
            raise TypeError(f"{field_name} must be an integer")
        if value < 0:
            raise ValueError(f"{field_name} must not be negative")
        return value


__all__ = [
    "AttemptAccountingError",
    "AttemptBudgetExceeded",
    "AttemptLedger",
    "AttemptLedgerIntegrityError",
    "AttemptReservationError",
]
