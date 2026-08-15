"""Replay and live-writer checkpoint state machine for attempt accounting."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import spiral_harness.execution.contracts as _contracts
from spiral_harness.core.models import ArtifactRef

_Loaded = TypeVar("_Loaded")


class AttemptWriterCheckpoint:
    """Advance one verified writer while keeping public replays fresh.

    The checkpoint is valid only under ``ArtifactRepository``'s immutable and
    non-equivocating reference contract. It never replaces an authority replay:
    current state reads re-run the complete chain, compare it with this state,
    and permanently latch any current-lineage integrity failure.
    """

    def __init__(
        self,
        state: _contracts.AttemptLedgerState,
        *,
        integrity_error_type: type[RuntimeError],
    ) -> None:
        if not isinstance(integrity_error_type, type) or not issubclass(
            integrity_error_type, RuntimeError
        ):
            raise TypeError("integrity_error_type must be a RuntimeError type")
        checked = _contracts.AttemptLedgerState.model_validate(state, strict=True)
        self._state = checked
        self._ledger_id = checked.ledger_id
        self._writer_epoch_id = checked.writer_epoch_id
        self._budget = checked.budget
        self._error_type = integrity_error_type
        self._integrity_failure: RuntimeError | None = None

    @property
    def tail_ref(self) -> ArtifactRef | None:
        return self._state.tail_ref

    def current(self) -> _contracts.AttemptLedgerState:
        """Return the operational state only after checking its writer binding."""

        self._raise_if_failed()
        state = self._state
        if (
            state.ledger_id != self._ledger_id
            or state.writer_epoch_id != self._writer_epoch_id
            or state.budget != self._budget
        ):
            error = self._error("live attempt checkpoint diverged from its writer")
            self.latch(error)
            raise error
        return state

    def fresh_current(
        self,
        replay: Callable[[ArtifactRef | None], _contracts.AttemptLedgerState],
    ) -> _contracts.AttemptLedgerState:
        """Replay and compare the current chain, latching every failure."""

        self._raise_if_failed()
        try:
            replayed = replay(self.tail_ref)
        except self._error_type as exc:
            self.latch(exc)
            raise
        return self._verify_replay(replayed)

    def fresh_at(
        self,
        replay: Callable[[ArtifactRef | None], _contracts.AttemptLedgerState],
        tail_ref: ArtifactRef | None,
    ) -> _contracts.AttemptLedgerState:
        """Replay one boundary without letting a foreign bad ref poison the writer.

        A failed non-current replay triggers an exceptional-path replay of the
        current tail. If that also fails, the requested ref exposed corruption
        on the live lineage and the writer is permanently latched. A malformed
        foreign branch cannot cause that denial of service while the current
        chain remains valid.
        """

        if tail_ref == self.tail_ref:
            return self.fresh_current(replay)
        try:
            return replay(tail_ref)
        except self._error_type as requested_error:
            try:
                replayed_current = replay(self.tail_ref)
            except self._error_type as current_error:
                self.latch(current_error)
            else:
                self._verify_replay(replayed_current)
            raise requested_error

    def load_current(
        self,
        loader: Callable[[ArtifactRef], _Loaded],
        ref: ArtifactRef,
    ) -> _Loaded:
        """Load a current-lineage artifact and latch typed integrity failures."""

        self._raise_if_failed()
        try:
            return loader(ref)
        except self._error_type as exc:
            self.latch(exc)
            raise

    def verify_current_outcome(
        self,
        outcome_ref: ArtifactRef,
        *,
        load_outcome: Callable[[ArtifactRef], _contracts.AttemptOutcome],
        load_reservation: Callable[[ArtifactRef], _contracts.AttemptReservation],
        verify_pair: Callable[[_contracts.AttemptReservation, _contracts.AttemptOutcome], None],
    ) -> _contracts.AttemptOutcome:
        """Verify the closed current tip, including its reservation and execution."""

        state = self.current()
        if state.tail_ref != outcome_ref:
            error = self._error("current outcome ref differs from its checkpoint")
            self.latch(error)
            raise error
        outcome = self.load_current(load_outcome, outcome_ref)
        reservation = self.load_current(load_reservation, outcome.reservation_ref)
        try:
            verify_pair(reservation, outcome)
        except self._error_type as exc:
            self.latch(exc)
            raise
        if (
            state.pending_reservation_ref is not None
            or state.completed_attempts < 1
            or state.attempts_used != state.completed_attempts
            or state.encumbered_tokens != state.charged_tokens
            or outcome.sequence != state.completed_attempts - 1
            or reservation.sequence != outcome.sequence
            or state.poisoned != (outcome.disposition is _contracts.AttemptDisposition.POISONED)
        ):
            error = self._error("current attempt outcome differs from its checkpoint")
            self.latch(error)
            raise error
        return outcome

    def append_reservation(
        self,
        reservation_ref: ArtifactRef,
        *,
        reserved_tokens: int,
    ) -> None:
        """Advance the checkpoint after an exact reservation publication readback."""

        state = self.current()
        if (
            state.pending_reservation_ref is not None
            or state.poisoned
            or state.completed_attempts != state.attempts_used
            or state.encumbered_tokens != state.charged_tokens
        ):
            error = self._error("live attempt checkpoint cannot admit a reservation")
            self.latch(error)
            raise error
        attempts_used = state.attempts_used + 1
        encumbered_tokens = state.charged_tokens + reserved_tokens
        self._state = _contracts.AttemptLedgerState(
            ledger_id=self._ledger_id,
            writer_epoch_id=self._writer_epoch_id,
            budget=self._budget,
            tail_ref=reservation_ref,
            pending_reservation_ref=reservation_ref,
            poisoned=False,
            attempts_used=attempts_used,
            completed_attempts=state.completed_attempts,
            charged_tokens=state.charged_tokens,
            encumbered_tokens=encumbered_tokens,
            remaining_attempts=self._budget.max_attempts - attempts_used,
            remaining_tokens=self._budget.max_total_tokens - encumbered_tokens,
        )

    def append_outcome(
        self,
        reservation_ref: ArtifactRef,
        outcome_ref: ArtifactRef,
        outcome: _contracts.AttemptOutcome,
    ) -> None:
        """Advance the checkpoint after an exact outcome publication readback."""

        state = self.current()
        if (
            state.tail_ref != reservation_ref
            or state.pending_reservation_ref != reservation_ref
            or state.poisoned
            or state.attempts_used != state.completed_attempts + 1
            or outcome.reservation_ref != reservation_ref
            or outcome.sequence != state.completed_attempts
        ):
            error = self._error("live attempt checkpoint cannot admit an outcome")
            self.latch(error)
            raise error
        charged_tokens = state.charged_tokens + outcome.charged_tokens
        poisoned = outcome.disposition is _contracts.AttemptDisposition.POISONED
        if not poisoned and charged_tokens > self._budget.max_total_tokens:
            error = self._error("live attempt checkpoint exceeds its hard token budget")
            self.latch(error)
            raise error
        self._state = _contracts.AttemptLedgerState(
            ledger_id=self._ledger_id,
            writer_epoch_id=self._writer_epoch_id,
            budget=self._budget,
            tail_ref=outcome_ref,
            pending_reservation_ref=None,
            poisoned=poisoned,
            attempts_used=state.attempts_used,
            completed_attempts=state.completed_attempts + 1,
            charged_tokens=charged_tokens,
            encumbered_tokens=charged_tokens,
            remaining_attempts=(0 if poisoned else self._budget.max_attempts - state.attempts_used),
            remaining_tokens=(0 if poisoned else self._budget.max_total_tokens - charged_tokens),
        )

    def latch(self, error: RuntimeError) -> None:
        """Permanently poison live-writer authority after an integrity failure."""

        if self._integrity_failure is None:
            self._integrity_failure = error

    def _verify_replay(
        self,
        replayed: _contracts.AttemptLedgerState,
    ) -> _contracts.AttemptLedgerState:
        if replayed != self._state:
            error = self._error("live attempt checkpoint differs from full replay")
            self.latch(error)
            raise error
        return replayed

    def _raise_if_failed(self) -> None:
        if self._integrity_failure is not None:
            raise self._error("attempt ledger is locked after an integrity failure") from (
                self._integrity_failure
            )

    def _error(self, message: str) -> RuntimeError:
        return self._error_type(message)


def replay_attempt_ledger(
    *,
    ledger_id: str,
    writer_epoch_id: str,
    budget: _contracts.AttemptBudget,
    tail_ref: ArtifactRef | None,
    integrity_error_type: type[RuntimeError],
    load_reservation: Callable[[ArtifactRef], _contracts.AttemptReservation],
    load_outcome: Callable[[ArtifactRef], _contracts.AttemptOutcome],
    verify_pair: Callable[[_contracts.AttemptReservation, _contracts.AttemptOutcome], None],
) -> _contracts.AttemptLedgerState:
    """Fully replay one attempt chain without retaining repository capabilities."""

    if tail_ref is None:
        return _contracts.AttemptLedgerState(
            ledger_id=ledger_id,
            writer_epoch_id=writer_epoch_id,
            budget=budget,
            tail_ref=None,
            pending_reservation_ref=None,
            poisoned=False,
            attempts_used=0,
            completed_attempts=0,
            charged_tokens=0,
            encumbered_tokens=0,
            remaining_attempts=budget.max_attempts,
            remaining_tokens=budget.max_total_tokens,
        )

    try:
        checked_tail = ArtifactRef.model_validate(tail_ref, strict=True)
    except Exception as exc:
        raise integrity_error_type("attempt ledger tail is malformed") from exc

    cursor: ArtifactRef | None = checked_tail
    pending: tuple[ArtifactRef, _contracts.AttemptReservation] | None = None
    backwards: list[
        tuple[
            ArtifactRef,
            _contracts.AttemptReservation,
            ArtifactRef,
            _contracts.AttemptOutcome,
        ]
    ] = []
    seen: set[str] = set()

    if cursor.media_type == _contracts.ATTEMPT_RESERVATION_MEDIA_TYPE:
        reservation = load_reservation(cursor)
        pending = (cursor, reservation)
        seen.add(cursor.sha256)
        cursor = reservation.previous_outcome_ref
    elif cursor.media_type != _contracts.ATTEMPT_OUTCOME_MEDIA_TYPE:
        raise integrity_error_type("attempt ledger tail declares the wrong media type")

    while cursor is not None:
        if cursor.media_type != _contracts.ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise integrity_error_type("attempt outcome link declares the wrong media type")
        if cursor.sha256 in seen:
            raise integrity_error_type("attempt ledger contains a cycle")
        seen.add(cursor.sha256)
        outcome_ref = cursor
        outcome = load_outcome(outcome_ref)
        reservation_ref = outcome.reservation_ref
        if reservation_ref.sha256 in seen:
            raise integrity_error_type("attempt ledger contains a cycle")
        seen.add(reservation_ref.sha256)
        reservation = load_reservation(reservation_ref)
        verify_pair(reservation, outcome)
        backwards.append((reservation_ref, reservation, outcome_ref, outcome))
        cursor = reservation.previous_outcome_ref

    pairs = tuple(reversed(backwards))
    previous_outcome_ref: ArtifactRef | None = None
    chain_writer_epoch_id: str | None = None
    charged_tokens = 0
    poisoned = False
    for sequence, (reservation_ref, reservation, outcome_ref, outcome) in enumerate(pairs):
        del reservation_ref
        if poisoned:
            raise integrity_error_type("attempt entries must not follow a poisoned outcome")
        if reservation.sequence != sequence or outcome.sequence != sequence:
            raise integrity_error_type("attempt sequences are not contiguous")
        if reservation.previous_outcome_ref != previous_outcome_ref:
            raise integrity_error_type("attempt ledger link does not match prior outcome")
        if chain_writer_epoch_id is None:
            chain_writer_epoch_id = reservation.writer_epoch_id
        elif reservation.writer_epoch_id != chain_writer_epoch_id:
            raise integrity_error_type("attempt chain crosses writer epochs")
        charged_tokens += outcome.charged_tokens
        poisoned = outcome.disposition is _contracts.AttemptDisposition.POISONED
        previous_outcome_ref = outcome_ref

    pending_tokens = 0
    if pending is not None:
        if poisoned:
            raise integrity_error_type("a poisoned ledger must not contain a pending reservation")
        _, reservation = pending
        if reservation.sequence != len(pairs):
            raise integrity_error_type("pending reservation sequence is not contiguous")
        if reservation.previous_outcome_ref != previous_outcome_ref:
            raise integrity_error_type("pending reservation link is inconsistent")
        if (
            chain_writer_epoch_id is not None
            and reservation.writer_epoch_id != chain_writer_epoch_id
        ):
            raise integrity_error_type("attempt chain crosses writer epochs")
        pending_tokens = reservation.reserved_tokens

    attempts_used = len(pairs) + int(pending is not None)
    encumbered_tokens = charged_tokens + pending_tokens
    if attempts_used > budget.max_attempts:
        raise integrity_error_type("attempt chain exceeds its hard attempt budget")
    if not poisoned and encumbered_tokens > budget.max_total_tokens:
        raise integrity_error_type("attempt chain exceeds its hard token budget")

    return _contracts.AttemptLedgerState(
        ledger_id=ledger_id,
        writer_epoch_id=writer_epoch_id,
        budget=budget,
        tail_ref=checked_tail,
        pending_reservation_ref=None if pending is None else pending[0],
        poisoned=poisoned,
        attempts_used=attempts_used,
        completed_attempts=len(pairs),
        charged_tokens=charged_tokens,
        encumbered_tokens=encumbered_tokens,
        remaining_attempts=(0 if poisoned else budget.max_attempts - attempts_used),
        remaining_tokens=(0 if poisoned else budget.max_total_tokens - encumbered_tokens),
    )


__all__ = ["AttemptWriterCheckpoint", "replay_attempt_ledger"]
