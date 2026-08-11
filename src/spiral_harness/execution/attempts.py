"""Immutable, conservative accounting for single model attempts.

The ledger deliberately has no mutable, repository-wide head.  One
``AttemptLedger`` instance is the single writer for its in-memory tail and a
process-local lock serializes its methods.  A caller may persist and later
reopen an explicit tail, but concurrent processes need a durable lease/CAS
service above this module; this class does not claim to provide one.
"""

from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.storage.protocol import ArtifactRepository

if TYPE_CHECKING:
    from spiral_harness.execution.model import ModelExecution

ATTEMPT_RESERVATION_MEDIA_TYPE = "application/vnd.spiral-harness.attempt-reservation.v1+json"
ATTEMPT_OUTCOME_MEDIA_TYPE = "application/vnd.spiral-harness.attempt-outcome.v1+json"
MODEL_EXECUTION_MEDIA_TYPE = "application/vnd.spiral-harness.model-execution.v1+json"


class AttemptAccountingError(RuntimeError):
    """Base error for fail-closed attempt accounting."""


class AttemptBudgetExceeded(AttemptAccountingError):
    """Raised before backend execution when no complete reservation fits."""


class AttemptLedgerIntegrityError(AttemptAccountingError):
    """Raised when an immutable accounting chain cannot be verified."""


class AttemptReservationError(AttemptAccountingError):
    """Raised for a stale, foreign, forged, or already-consumed reservation."""


class AttemptDisposition(StrEnum):
    """Terminal accounting treatment for one reservation."""

    SETTLED = "settled"
    BURNED = "burned"
    POISONED = "poisoned"


class AttemptBudget(ImmutableModel):
    """Hard ceilings enforced before a backend is invoked."""

    schema_version: Literal["1"] = "1"
    max_attempts: Annotated[int, Field(ge=1, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=1, strict=True)]
    max_tokens_per_attempt: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def per_attempt_ceiling_fits_total(self) -> Self:
        if self.max_tokens_per_attempt > self.max_total_tokens:
            raise ValueError("max_tokens_per_attempt must not exceed max_total_tokens")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AttemptReservation(ImmutableModel):
    """A persisted, single-use authorization created before model execution."""

    schema_version: Literal["1"] = "1"
    ledger_id: NonEmptyStr
    budget_fingerprint: Sha256
    sequence: Annotated[int, Field(ge=0, strict=True)]
    task_fingerprint: Sha256
    execution_fingerprint: Sha256
    request_sha256: Sha256
    reserved_tokens: Annotated[int, Field(ge=1, strict=True)]
    previous_outcome_ref: ArtifactRef | None

    @model_validator(mode="after")
    def previous_ref_has_exact_media_type(self) -> Self:
        if (
            self.previous_outcome_ref is not None
            and self.previous_outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE
        ):
            raise ValueError("previous_outcome_ref declares the wrong media type")
        if self.sequence == 0 and self.previous_outcome_ref is not None:
            raise ValueError("reservation sequence 0 must not have a previous outcome")
        if self.sequence > 0 and self.previous_outcome_ref is None:
            raise ValueError("reservation sequence greater than 0 requires a previous outcome")
        return self


class AttemptOutcome(ImmutableModel):
    """The immutable terminal record consuming exactly one reservation."""

    schema_version: Literal["1"] = "1"
    ledger_id: NonEmptyStr
    budget_fingerprint: Sha256
    sequence: Annotated[int, Field(ge=0, strict=True)]
    reservation_ref: ArtifactRef
    disposition: AttemptDisposition
    reported_tokens: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]
    execution_ref: ArtifactRef | None
    error_class: NonEmptyStr | None

    @model_validator(mode="after")
    def disposition_shape_is_valid(self) -> Self:
        if self.reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise ValueError("reservation_ref declares the wrong media type")
        if (
            self.execution_ref is not None
            and self.execution_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE
        ):
            raise ValueError("execution_ref declares the wrong media type")
        if self.disposition is AttemptDisposition.SETTLED:
            if self.execution_ref is None:
                raise ValueError("a settled outcome requires an execution_ref")
            if self.error_class is not None:
                raise ValueError("a settled outcome must not declare an error_class")
            if self.charged_tokens != self.reported_tokens:
                raise ValueError("a settled outcome charges exactly its reported tokens")
        elif self.error_class is None:
            raise ValueError("a burned or poisoned outcome requires an error_class")
        if (
            self.disposition is AttemptDisposition.POISONED
            and self.charged_tokens != self.reported_tokens
        ):
            raise ValueError("a poisoned outcome charges its reported token overrun")
        return self


class AttemptLedgerState(ImmutableModel):
    """Replay-derived accounting state for one explicit immutable tail."""

    ledger_id: NonEmptyStr
    budget: AttemptBudget
    tail_ref: ArtifactRef | None
    pending_reservation_ref: ArtifactRef | None
    poisoned: bool
    attempts_used: Annotated[int, Field(ge=0, strict=True)]
    completed_attempts: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]
    encumbered_tokens: Annotated[int, Field(ge=0, strict=True)]
    remaining_attempts: Annotated[int, Field(ge=0, strict=True)]
    remaining_tokens: Annotated[int, Field(ge=0, strict=True)]


class AttemptLedger:
    """Append-only reservation/outcome accounting for one process-local writer.

    Reservations are persisted before a caller may invoke a backend.  Success
    settles usage derived from a strictly parsed execution; every uncertain or
    failed attempt burns the complete reservation.  A reported overrun charges
    its actual usage and permanently poisons the stream, so later work cannot
    compound a backend's violation of its reservation.

    The internal ``RLock`` protects threads sharing this exact object only.  It
    is not an inter-process lock, durable lease, or distributed head compare-and-
    swap.  Such a coordinator must pass this ledger a uniquely-owned stream.

    Strict ``ModelExecution`` replay establishes semantic integrity between the
    trusted runner and this ledger; content hashes are not producer signatures.
    Code holding the ledger/reservation capability remains trusted in this M1
    boundary and must not be exposed to candidate workers or plugins.
    """

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        ledger_id: str,
        budget: AttemptBudget,
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
        self._budget = AttemptBudget.model_validate(budget, strict=True)
        self._lock = RLock()
        self._tail_ref = (
            None if tail_ref is None else ArtifactRef.model_validate(tail_ref, strict=True)
        )
        # Refuse a foreign, malformed, or over-budget supplied tail eagerly.
        self._replay(self._tail_ref)

    @property
    def repository(self) -> ArtifactRepository:
        """Repository used for both accounting and execution artifacts."""

        return self._repository

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def budget(self) -> AttemptBudget:
        return self._budget

    @property
    def tail_ref(self) -> ArtifactRef | None:
        return self._tail_ref

    def state(self) -> AttemptLedgerState:
        """Re-verify and derive state from the current explicit tail."""

        with self._lock:
            return self._replay(self._tail_ref)

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
            state = self._replay(self._tail_ref)
            if state.pending_reservation_ref is not None:
                raise AttemptReservationError("the ledger already has an open reservation")
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

            reservation = AttemptReservation(
                ledger_id=self._ledger_id,
                budget_fingerprint=self._budget.fingerprint,
                sequence=state.attempts_used,
                task_fingerprint=task_fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                reserved_tokens=ceiling,
                previous_outcome_ref=self._tail_ref,
            )
            reservation_ref = self._repository.put_json(
                reservation,
                media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
            )
            checked_ref = self._verify_published(
                reservation_ref,
                expected_media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
                model_type=AttemptReservation,
                expected_value=reservation,
            )
            self._tail_ref = checked_ref
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
        """Reserve only if the process-local ledger is still at an exact outcome head."""

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
                and checked_expected.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE
            ):
                raise AttemptReservationError(
                    "expected previous outcome declares the wrong media type"
                )
            if self._tail_ref != checked_expected:
                raise AttemptReservationError(
                    "attempt ledger advanced beyond the expected scheduled boundary"
                )
            # ``RLock`` keeps the comparison and the nested reservation atomic
            # for every writer sharing this exact process-local ledger object.
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
                    disposition=AttemptDisposition.POISONED,
                    reported_tokens=tokens,
                    charged_tokens=tokens,
                    execution_ref=checked_execution_ref,
                    error_class="usage_exceeded",
                )
            return self._publish_outcome(
                reservation_ref=checked_reservation_ref,
                reservation=reservation,
                disposition=AttemptDisposition.SETTLED,
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
                    AttemptDisposition.POISONED if poisoned else AttemptDisposition.BURNED
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
        reservation: AttemptReservation,
        disposition: AttemptDisposition,
        reported_tokens: int,
        charged_tokens: int,
        execution_ref: ArtifactRef | None,
        error_class: str | None,
    ) -> ArtifactRef:
        outcome = AttemptOutcome(
            ledger_id=self._ledger_id,
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
            media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
        )
        checked_ref = self._verify_published(
            outcome_ref,
            expected_media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
            model_type=AttemptOutcome,
            expected_value=outcome,
        )
        self._tail_ref = checked_ref
        return checked_ref

    def _require_current_reservation(
        self,
        reservation_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, AttemptReservation]:
        try:
            checked_ref = ArtifactRef.model_validate(reservation_ref, strict=True)
        except Exception as exc:
            raise AttemptReservationError("reservation reference is malformed") from exc
        if checked_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise AttemptReservationError("reservation reference declares the wrong media type")
        state = self._replay(self._tail_ref)
        if state.pending_reservation_ref is None or checked_ref != state.pending_reservation_ref:
            raise AttemptReservationError(
                "reservation is stale, foreign, forged, or already consumed"
            )
        return checked_ref, self._load_reservation(checked_ref)

    def _load_execution(
        self,
        execution_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, ModelExecution]:
        # Local import avoids a module cycle: the fixed runner depends on this
        # ledger, while settlement must still validate the runner's exact type.
        from spiral_harness.execution.model import ModelExecution

        try:
            checked_ref = ArtifactRef.model_validate(execution_ref, strict=True)
        except Exception as exc:
            raise AttemptReservationError("execution reference is malformed") from exc
        if checked_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise AttemptReservationError("execution reference declares the wrong media type")
        try:
            loaded = self._repository.get_json(checked_ref, ModelExecution)
            execution = ModelExecution.model_validate(loaded, strict=True)
        except Exception as exc:
            raise AttemptReservationError(
                "execution artifact is not a canonical ModelExecution"
            ) from exc
        return checked_ref, execution

    @staticmethod
    def _verify_execution_binding(
        reservation: AttemptReservation,
        execution: ModelExecution,
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
        model_type: type[AttemptReservation] | type[AttemptOutcome],
        expected_value: AttemptReservation | AttemptOutcome,
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

    def _replay(self, tail_ref: ArtifactRef | None) -> AttemptLedgerState:
        if tail_ref is None:
            return AttemptLedgerState(
                ledger_id=self._ledger_id,
                budget=self._budget,
                tail_ref=None,
                pending_reservation_ref=None,
                poisoned=False,
                attempts_used=0,
                completed_attempts=0,
                charged_tokens=0,
                encumbered_tokens=0,
                remaining_attempts=self._budget.max_attempts,
                remaining_tokens=self._budget.max_total_tokens,
            )

        try:
            checked_tail = ArtifactRef.model_validate(tail_ref, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError("attempt ledger tail is malformed") from exc

        cursor: ArtifactRef | None = checked_tail
        pending: tuple[ArtifactRef, AttemptReservation] | None = None
        backwards: list[tuple[ArtifactRef, AttemptReservation, ArtifactRef, AttemptOutcome]] = []
        seen: set[str] = set()

        if cursor.media_type == ATTEMPT_RESERVATION_MEDIA_TYPE:
            reservation = self._load_reservation(cursor)
            pending = (cursor, reservation)
            seen.add(cursor.sha256)
            cursor = reservation.previous_outcome_ref
        elif cursor.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise AttemptLedgerIntegrityError("attempt ledger tail declares the wrong media type")

        while cursor is not None:
            if cursor.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
                raise AttemptLedgerIntegrityError(
                    "attempt outcome link declares the wrong media type"
                )
            if cursor.sha256 in seen:
                raise AttemptLedgerIntegrityError("attempt ledger contains a cycle")
            seen.add(cursor.sha256)
            outcome_ref = cursor
            outcome = self._load_outcome(outcome_ref)
            reservation_ref = outcome.reservation_ref
            if reservation_ref.sha256 in seen:
                raise AttemptLedgerIntegrityError("attempt ledger contains a cycle")
            seen.add(reservation_ref.sha256)
            reservation = self._load_reservation(reservation_ref)
            self._verify_pair(reservation, outcome)
            backwards.append((reservation_ref, reservation, outcome_ref, outcome))
            cursor = reservation.previous_outcome_ref

        pairs = tuple(reversed(backwards))
        previous_outcome_ref: ArtifactRef | None = None
        charged_tokens = 0
        poisoned = False
        for sequence, (reservation_ref, reservation, outcome_ref, outcome) in enumerate(pairs):
            del reservation_ref
            if poisoned:
                raise AttemptLedgerIntegrityError(
                    "attempt entries must not follow a poisoned outcome"
                )
            if reservation.sequence != sequence or outcome.sequence != sequence:
                raise AttemptLedgerIntegrityError("attempt sequences are not contiguous")
            if reservation.previous_outcome_ref != previous_outcome_ref:
                raise AttemptLedgerIntegrityError(
                    "attempt ledger link does not match prior outcome"
                )
            charged_tokens += outcome.charged_tokens
            poisoned = outcome.disposition is AttemptDisposition.POISONED
            previous_outcome_ref = outcome_ref

        pending_tokens = 0
        if pending is not None:
            if poisoned:
                raise AttemptLedgerIntegrityError(
                    "a poisoned ledger must not contain a pending reservation"
                )
            _, reservation = pending
            if reservation.sequence != len(pairs):
                raise AttemptLedgerIntegrityError("pending reservation sequence is not contiguous")
            if reservation.previous_outcome_ref != previous_outcome_ref:
                raise AttemptLedgerIntegrityError("pending reservation link is inconsistent")
            pending_tokens = reservation.reserved_tokens

        attempts_used = len(pairs) + int(pending is not None)
        encumbered_tokens = charged_tokens + pending_tokens
        if attempts_used > self._budget.max_attempts:
            raise AttemptLedgerIntegrityError("attempt chain exceeds its hard attempt budget")
        if not poisoned and encumbered_tokens > self._budget.max_total_tokens:
            raise AttemptLedgerIntegrityError("attempt chain exceeds its hard token budget")

        return AttemptLedgerState(
            ledger_id=self._ledger_id,
            budget=self._budget,
            tail_ref=checked_tail,
            pending_reservation_ref=None if pending is None else pending[0],
            poisoned=poisoned,
            attempts_used=attempts_used,
            completed_attempts=len(pairs),
            charged_tokens=charged_tokens,
            encumbered_tokens=encumbered_tokens,
            remaining_attempts=(0 if poisoned else self._budget.max_attempts - attempts_used),
            remaining_tokens=(0 if poisoned else self._budget.max_total_tokens - encumbered_tokens),
        )

    def _verify_pair(
        self,
        reservation: AttemptReservation,
        outcome: AttemptOutcome,
    ) -> None:
        if reservation.ledger_id != self._ledger_id or outcome.ledger_id != self._ledger_id:
            raise AttemptLedgerIntegrityError("attempt artifact belongs to another ledger")
        if (
            reservation.budget_fingerprint != self._budget.fingerprint
            or outcome.budget_fingerprint != self._budget.fingerprint
        ):
            raise AttemptLedgerIntegrityError("attempt artifact belongs to another budget")
        if reservation.sequence != outcome.sequence:
            raise AttemptLedgerIntegrityError("reservation and outcome sequences differ")
        if reservation.reserved_tokens > self._budget.max_tokens_per_attempt:
            raise AttemptLedgerIntegrityError("reservation exceeds the per-attempt ceiling")
        if outcome.disposition is AttemptDisposition.SETTLED:
            if outcome.reported_tokens > reservation.reserved_tokens:
                raise AttemptLedgerIntegrityError("settled usage exceeds its reservation")
            if outcome.charged_tokens != outcome.reported_tokens:
                raise AttemptLedgerIntegrityError("settled charge does not match actual usage")
        elif outcome.disposition is AttemptDisposition.BURNED:
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
                outcome.disposition is AttemptDisposition.SETTLED
                and execution.status.value != "completed"
            ):
                raise AttemptLedgerIntegrityError(
                    "settled outcome does not reference a completed execution"
                )

    def _load_reservation(self, ref: ArtifactRef) -> AttemptReservation:
        if ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise AttemptLedgerIntegrityError("reservation ref declares the wrong media type")
        try:
            loaded = self._repository.get_json(ref, AttemptReservation)
            reservation = AttemptReservation.model_validate(loaded, strict=True)
        except Exception as exc:
            raise AttemptLedgerIntegrityError("attempt reservation cannot be verified") from exc
        if reservation.ledger_id != self._ledger_id:
            raise AttemptLedgerIntegrityError("reservation belongs to another ledger")
        if reservation.budget_fingerprint != self._budget.fingerprint:
            raise AttemptLedgerIntegrityError("reservation belongs to another budget")
        if reservation.reserved_tokens > self._budget.max_tokens_per_attempt:
            raise AttemptLedgerIntegrityError("reservation exceeds the per-attempt ceiling")
        return reservation

    def _load_outcome(self, ref: ArtifactRef) -> AttemptOutcome:
        if ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise AttemptLedgerIntegrityError("outcome ref declares the wrong media type")
        try:
            loaded = self._repository.get_json(ref, AttemptOutcome)
            outcome = AttemptOutcome.model_validate(loaded, strict=True)
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
    "ATTEMPT_OUTCOME_MEDIA_TYPE",
    "ATTEMPT_RESERVATION_MEDIA_TYPE",
    "MODEL_EXECUTION_MEDIA_TYPE",
    "AttemptAccountingError",
    "AttemptBudget",
    "AttemptBudgetExceeded",
    "AttemptDisposition",
    "AttemptLedger",
    "AttemptLedgerIntegrityError",
    "AttemptLedgerState",
    "AttemptOutcome",
    "AttemptReservation",
    "AttemptReservationError",
]
