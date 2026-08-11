"""Immutable execution receipts with outcome-derived trusted usage replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

import spiral_harness.execution.materialization as _materialization
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    CandidateTask,
    ExecutionStatus,
    ModelExecution,
    ModelExecutionRecord,
)
from spiral_harness.execution.model import FixedModelRunner
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationCellKey,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.protocol import ArtifactRepository

EXECUTION_RECEIPT_MEDIA_TYPE = "application/vnd.spiral-harness.execution-receipt.v3+json"


class ExecutionReceiptError(RuntimeError):
    """Base error for receipt publication and replay."""


class ExecutionReceiptIntegrityError(ExecutionReceiptError):
    """Raised when receipts or their referenced accounting chain are incomplete."""


class ExecutionReceiptKey(ImmutableModel):
    schema_version: Literal["1"] = "1"
    cell_fingerprint: Sha256
    attempt_index: Annotated[int, Field(ge=0, strict=True)]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ExecutionReceipt(ImmutableModel):
    """Content-addressed binding from a scheduled cell to one terminal attempt."""

    schema_version: Literal["3"] = "3"
    schedule_fingerprint: Sha256
    preflight_ref: ArtifactRef
    preflight_fingerprint: Sha256
    cell: EvaluationCellKey
    cell_fingerprint: Sha256
    attempt_index: Annotated[int, Field(ge=0, strict=True)]
    reservation_ref: ArtifactRef
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    ledger_tail_ref: ArtifactRef
    reported_tokens: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def declared_bindings_have_exact_shapes(self) -> Self:
        if self.cell_fingerprint != self.cell.fingerprint:
            raise ValueError("cell_fingerprint does not match the complete evaluation cell")
        expected_media_types = (
            ("preflight_ref", SCHEDULE_PREFLIGHT_MEDIA_TYPE),
            ("reservation_ref", ATTEMPT_RESERVATION_MEDIA_TYPE),
            ("execution_ref", MODEL_EXECUTION_MEDIA_TYPE),
            ("outcome_ref", ATTEMPT_OUTCOME_MEDIA_TYPE),
            ("ledger_tail_ref", ATTEMPT_OUTCOME_MEDIA_TYPE),
        )
        for field_name, expected in expected_media_types:
            if getattr(self, field_name).media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        if self.ledger_tail_ref != self.outcome_ref:
            raise ValueError("ledger_tail_ref must be the terminal outcome of this attempt")
        return self

    @property
    def key(self) -> ExecutionReceiptKey:
        return ExecutionReceiptKey(
            cell_fingerprint=self.cell_fingerprint,
            attempt_index=self.attempt_index,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ScheduledExecutionRecord(ImmutableModel):
    schema_version: Literal["3"] = "3"
    schedule_fingerprint: Sha256
    preflight_ref: ArtifactRef
    cell: EvaluationCellKey
    attempt_index: Annotated[int, Field(ge=0, strict=True)]
    execution: ModelExecution
    reservation_ref: ArtifactRef
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef
    ledger_tail_ref: ArtifactRef
    receipt: ExecutionReceipt
    receipt_ref: ArtifactRef

    @model_validator(mode="after")
    def receipt_and_runner_records_are_consistent(self) -> Self:
        expected_media_types = (
            ("preflight_ref", SCHEDULE_PREFLIGHT_MEDIA_TYPE),
            ("reservation_ref", ATTEMPT_RESERVATION_MEDIA_TYPE),
            ("execution_ref", MODEL_EXECUTION_MEDIA_TYPE),
            ("outcome_ref", ATTEMPT_OUTCOME_MEDIA_TYPE),
            ("ledger_tail_ref", ATTEMPT_OUTCOME_MEDIA_TYPE),
            ("receipt_ref", EXECUTION_RECEIPT_MEDIA_TYPE),
        )
        for field_name, expected in expected_media_types:
            if getattr(self, field_name).media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        if self.ledger_tail_ref != self.outcome_ref:
            raise ValueError("ledger_tail_ref must equal outcome_ref")
        expected_receipt_values = (
            (self.receipt.schedule_fingerprint, self.schedule_fingerprint),
            (self.receipt.preflight_ref, self.preflight_ref),
            (self.receipt.cell, self.cell),
            (self.receipt.attempt_index, self.attempt_index),
            (self.receipt.reservation_ref, self.reservation_ref),
            (self.receipt.execution_ref, self.execution_ref),
            (self.receipt.outcome_ref, self.outcome_ref),
            (self.receipt.ledger_tail_ref, self.ledger_tail_ref),
        )
        if any(actual != expected for actual, expected in expected_receipt_values):
            raise ValueError("receipt does not match its scheduled execution record")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class TrustedExecutionUsage(ImmutableModel):
    schema_version: Literal["3"] = "3"
    schedule_fingerprint: Sha256
    receipt_refs: tuple[ArtifactRef, ...]
    ledger_tail_refs: tuple[ArtifactRef, ...]
    cell_count: Annotated[int, Field(ge=1, strict=True)]
    attempt_count: Annotated[int, Field(ge=1, strict=True)]
    settled_attempts: Annotated[int, Field(ge=0, strict=True)]
    burned_attempts: Annotated[int, Field(ge=0, strict=True)]
    poisoned_attempts: Annotated[int, Field(ge=0, strict=True)]
    reported_tokens: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def counts_and_refs_are_consistent(self) -> Self:
        if len(self.receipt_refs) != self.attempt_count:
            raise ValueError("receipt_refs does not match attempt_count")
        if self.attempt_count != (
            self.settled_attempts + self.burned_attempts + self.poisoned_attempts
        ):
            raise ValueError("attempt disposition counts do not match attempt_count")
        if len(set(ref.sha256 for ref in self.receipt_refs)) != len(self.receipt_refs):
            raise ValueError("receipt_refs must not contain duplicates")
        if not self.ledger_tail_refs:
            raise ValueError("trusted usage requires at least one ledger tail")
        if len(set(ref.sha256 for ref in self.ledger_tail_refs)) != len(self.ledger_tail_refs):
            raise ValueError("ledger_tail_refs must not contain duplicates")
        for ref in self.receipt_refs:
            if ref.media_type != EXECUTION_RECEIPT_MEDIA_TYPE:
                raise ValueError("receipt_refs contains the wrong media type")
        for ref in self.ledger_tail_refs:
            if ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
                raise ValueError("ledger_tail_refs contains the wrong media type")
        if self.charged_tokens < self.reported_tokens:
            raise ValueError("charged token total cannot be less than reported token total")
        return self

    @property
    def tokens(self) -> int:
        """Research and hard-budget usage is the conservative charged amount."""
        return self.charged_tokens

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def publish_execution_receipt(
    repository: ArtifactRepository,
    *,
    schedule: EvaluationBatchSchedule,
    cell: EvaluationCellKey,
    attempt_index: int,
    preflight_ref: ArtifactRef,
    reservation_ref: ArtifactRef,
    outcome_ref: ArtifactRef,
    ledger_tail_ref: ArtifactRef,
) -> ArtifactRef:
    """Verify one terminal attempt and publish its immutable cell receipt."""
    checked_repository = _require_repository(repository)
    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    checked_cell = EvaluationCellKey.model_validate(cell, strict=True)
    if not checked_schedule.contains(checked_cell):
        raise ExecutionReceiptIntegrityError("receipt cell is outside the frozen schedule")
    _require_attempt_index(checked_schedule, attempt_index)
    checked_preflight_ref, preflight = _load_preflight(
        checked_repository, preflight_ref, checked_schedule
    )
    checked_reservation_ref = _validate_ref(
        reservation_ref,
        ATTEMPT_RESERVATION_MEDIA_TYPE,
        "reservation",
    )
    checked_outcome_ref = _validate_ref(
        outcome_ref,
        ATTEMPT_OUTCOME_MEDIA_TYPE,
        "outcome",
    )
    checked_tail_ref = _validate_ref(
        ledger_tail_ref,
        ATTEMPT_OUTCOME_MEDIA_TYPE,
        "ledger tail",
    )
    if checked_tail_ref != checked_outcome_ref:
        raise ExecutionReceiptIntegrityError(
            "receipt may only bind the outcome that was the terminal ledger tail"
        )

    reservation, outcome, execution = _load_linked_attempt(
        checked_repository,
        reservation_ref=checked_reservation_ref,
        outcome_ref=checked_outcome_ref,
    )
    _verify_cell_execution_binding(
        checked_schedule,
        checked_cell,
        attempt_index,
        execution,
    )
    _verify_preflight_attempt_binding(preflight, reservation, outcome, execution)

    receipt = ExecutionReceipt(
        schedule_fingerprint=checked_schedule.fingerprint,
        preflight_ref=checked_preflight_ref,
        preflight_fingerprint=preflight.fingerprint,
        cell=checked_cell,
        cell_fingerprint=checked_cell.fingerprint,
        attempt_index=attempt_index,
        reservation_ref=checked_reservation_ref,
        execution_ref=outcome.execution_ref,
        outcome_ref=checked_outcome_ref,
        ledger_tail_ref=checked_tail_ref,
        reported_tokens=outcome.reported_tokens,
        charged_tokens=outcome.charged_tokens,
    )
    raw_ref = checked_repository.put_json(receipt, media_type=EXECUTION_RECEIPT_MEDIA_TYPE)
    checked_ref = _validate_ref(raw_ref, EXECUTION_RECEIPT_MEDIA_TYPE, "published receipt")
    try:
        loaded = checked_repository.get_json(checked_ref, ExecutionReceipt)
        checked_receipt = ExecutionReceipt.model_validate(loaded, strict=True)
    except Exception as exc:
        raise ExecutionReceiptIntegrityError("published receipt cannot be verified") from exc
    if checked_receipt != receipt:
        raise ExecutionReceiptIntegrityError("published receipt changed content")
    return checked_ref


def execute_scheduled_attempt(
    *,
    runner: FixedModelRunner,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    expected_previous_ledger_tail_ref: ArtifactRef | None,
    previous_receipt_ref: ArtifactRef | None = None,
    cell: EvaluationCellKey,
    attempt_index: int,
    task: object,
    harness_ref: ArtifactRef,
) -> ScheduledExecutionRecord:
    """Execute one serialized scheduled attempt and bind its accounting refs."""
    if type(runner) is not FixedModelRunner:
        raise TypeError("runner must be an exact FixedModelRunner")
    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    checked_cell = EvaluationCellKey.model_validate(cell, strict=True)
    if not checked_schedule.contains(checked_cell):
        raise ExecutionReceiptIntegrityError("scheduled cell is outside the frozen schedule")
    _require_attempt_index(checked_schedule, attempt_index)
    repository = runner.repository
    checked_preflight_ref, preflight = _load_preflight(repository, preflight_ref, checked_schedule)
    if runner.spec != preflight.model_spec:
        raise ExecutionReceiptIntegrityError("runner model spec differs from preflight")
    checked_harness = _materialization.HarnessMaterializer(
        repository,
        spec=runner.spec,
    ).materialize(harness_ref)
    expected_harness_id = checked_schedule.harness_id_for(checked_cell.side)
    if checked_harness.harness_id != expected_harness_id:
        raise ExecutionReceiptIntegrityError(
            "harness_id does not match the frozen identity for the scheduled side"
        )
    try:
        checked_task = CandidateTask.from_task_view(task)
    except (TypeError, ValueError) as exc:
        raise ExecutionReceiptIntegrityError(
            "task must be one candidate-facing task view without adapter, roster, or gold access"
        ) from exc
    if checked_task.task_id != checked_cell.task_id:
        raise ExecutionReceiptIntegrityError(
            "candidate-facing task id does not match the scheduled cell"
        )

    expected_tail = (
        None
        if expected_previous_ledger_tail_ref is None
        else _validate_ref(
            expected_previous_ledger_tail_ref,
            ATTEMPT_OUTCOME_MEDIA_TYPE,
            "expected previous ledger tail",
        )
    )
    if preflight.ledger_id is None:
        raise ExecutionReceiptIntegrityError(
            "scheduled execution requires a ledger-bound preflight certificate"
        )
    try:
        boundary_state = runner.attempt_state_at(preflight.ledger_tail_ref)
    except Exception as exc:
        raise ExecutionReceiptIntegrityError(
            "preflight ledger boundary cannot be replayed"
        ) from exc
    if boundary_state.writer_epoch_id != preflight.writer_epoch_id:
        raise ExecutionReceiptIntegrityError("runner ledger writer epoch differs from preflight")
    if not preflight.binds_ledger_state(boundary_state):
        raise ExecutionReceiptIntegrityError("preflight capacity differs from its ledger boundary")
    if expected_tail == preflight.ledger_tail_ref:
        if previous_receipt_ref is not None:
            raise ExecutionReceiptIntegrityError(
                "the first scheduled attempt must not cite a previous batch receipt"
            )
    else:
        if expected_tail is None or previous_receipt_ref is None:
            raise ExecutionReceiptIntegrityError(
                "a later scheduled attempt requires the immediately previous receipt"
            )
        checked_previous_receipt_ref = _validate_ref(
            previous_receipt_ref,
            EXECUTION_RECEIPT_MEDIA_TYPE,
            "previous receipt",
        )
        previous_receipt = _load_model(
            repository,
            checked_previous_receipt_ref,
            ExecutionReceipt,
            "previous receipt",
        )
        if (
            previous_receipt.schedule_fingerprint != checked_schedule.fingerprint
            or previous_receipt.preflight_ref != checked_preflight_ref
            or previous_receipt.preflight_fingerprint != preflight.fingerprint
            or previous_receipt.outcome_ref != expected_tail
            or previous_receipt.ledger_tail_ref != expected_tail
        ):
            raise ExecutionReceiptIntegrityError(
                "previous receipt does not authorize the current batch progress"
            )
        expected_previous_cell = checked_schedule.contains(previous_receipt.cell)
        if not expected_previous_cell:
            raise ExecutionReceiptIntegrityError(
                "previous receipt belongs to a foreign evaluation cell"
            )
        _require_attempt_index(checked_schedule, previous_receipt.attempt_index)
        previous_reservation, previous_outcome, previous_execution = _load_linked_attempt(
            repository,
            reservation_ref=previous_receipt.reservation_ref,
            outcome_ref=previous_receipt.outcome_ref,
        )
        if previous_receipt.execution_ref != previous_outcome.execution_ref:
            raise ExecutionReceiptIntegrityError(
                "previous receipt execution does not match its outcome"
            )
        if (
            previous_receipt.reported_tokens != previous_outcome.reported_tokens
            or previous_receipt.charged_tokens != previous_outcome.charged_tokens
        ):
            raise ExecutionReceiptIntegrityError(
                "previous receipt token claims do not match its outcome"
            )
        _verify_cell_execution_binding(
            checked_schedule,
            previous_receipt.cell,
            previous_receipt.attempt_index,
            previous_execution,
        )
        _verify_preflight_attempt_binding(
            preflight,
            previous_reservation,
            previous_outcome,
            previous_execution,
        )
    state = runner.attempt_state()
    if state.ledger_id != preflight.ledger_id:
        raise ExecutionReceiptIntegrityError("runner ledger differs from preflight ledger")
    if state.writer_epoch_id != preflight.writer_epoch_id:
        raise ExecutionReceiptIntegrityError("runner ledger writer epoch differs from preflight")
    if state.budget.fingerprint != preflight.budget_fingerprint:
        raise ExecutionReceiptIntegrityError("runner budget differs from preflight budget")
    if state.pending_reservation_ref is not None:
        raise ExecutionReceiptIntegrityError("runner ledger has an open reservation")
    if state.poisoned:
        raise ExecutionReceiptIntegrityError("runner ledger is poisoned")
    if state.tail_ref != expected_tail:
        raise ExecutionReceiptIntegrityError(
            "runner ledger advanced beyond the expected scheduled boundary"
        )
    seed = checked_schedule.seed_for(checked_cell, attempt_index=attempt_index)
    raw_record = runner.execute_record(
        checked_task,
        harness=checked_harness,
        seed=seed,
        reservation_token_ceiling=preflight.token_ceiling_per_attempt,
        expected_previous_ledger_tail_ref=expected_tail,
    )
    record = ModelExecutionRecord.model_validate(raw_record, strict=True)
    persisted_execution = _load_model(
        repository,
        record.execution_ref,
        ModelExecution,
        "execution",
    )
    if persisted_execution != record.execution:
        raise ExecutionReceiptIntegrityError("runner execution record changed after persistence")
    outcome = _load_model(repository, record.outcome_ref, AttemptOutcome, "outcome")
    if outcome.execution_ref != record.execution_ref:
        raise ExecutionReceiptIntegrityError(
            "runner outcome does not reference its atomic execution record"
        )
    reservation_ref = outcome.reservation_ref
    reservation, checked_outcome, checked_execution = _load_linked_attempt(
        repository,
        reservation_ref=reservation_ref,
        outcome_ref=record.outcome_ref,
    )
    if checked_outcome != outcome or checked_execution != record.execution:
        raise ExecutionReceiptIntegrityError("runner record does not match verified accounting")
    del reservation

    receipt_ref = publish_execution_receipt(
        repository,
        schedule=checked_schedule,
        cell=checked_cell,
        attempt_index=attempt_index,
        preflight_ref=checked_preflight_ref,
        reservation_ref=reservation_ref,
        outcome_ref=record.outcome_ref,
        ledger_tail_ref=record.outcome_ref,
    )
    receipt = _load_model(
        repository,
        receipt_ref,
        ExecutionReceipt,
        "receipt",
    )
    return ScheduledExecutionRecord(
        schedule_fingerprint=checked_schedule.fingerprint,
        preflight_ref=checked_preflight_ref,
        cell=checked_cell,
        attempt_index=attempt_index,
        execution=record.execution,
        reservation_ref=reservation_ref,
        execution_ref=record.execution_ref,
        outcome_ref=record.outcome_ref,
        ledger_tail_ref=record.outcome_ref,
        receipt=receipt,
        receipt_ref=receipt_ref,
    )


def replay_trusted_usage(
    repository: ArtifactRepository,
    *,
    schedule: EvaluationBatchSchedule,
    preflight_ref: ArtifactRef,
    attempt_ledger: AttemptLedger,
    receipt_refs: Iterable[ArtifactRef],
) -> TrustedExecutionUsage:
    """Replay complete paired usage against an exact ledger writer."""
    checked_repository = _require_repository(repository)
    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    checked_preflight_ref, preflight = _load_preflight(
        checked_repository, preflight_ref, checked_schedule
    )
    if type(attempt_ledger) is not AttemptLedger:
        raise TypeError("attempt_ledger must be an exact AttemptLedger")
    if attempt_ledger.repository is not checked_repository:
        raise ExecutionReceiptIntegrityError(
            "live attempt ledger uses a different artifact repository"
        )
    try:
        live_state = attempt_ledger.state()
    except Exception as exc:
        raise ExecutionReceiptIntegrityError("live attempt ledger cannot be verified") from exc
    if preflight.ledger_id is None:
        raise ExecutionReceiptIntegrityError(
            "trusted usage replay requires a ledger-bound preflight certificate"
        )
    try:
        boundary_state = attempt_ledger.state_at(preflight.ledger_tail_ref)
    except Exception as exc:
        raise ExecutionReceiptIntegrityError(
            "preflight ledger boundary cannot be replayed"
        ) from exc
    if boundary_state.writer_epoch_id != preflight.writer_epoch_id:
        raise ExecutionReceiptIntegrityError("live ledger writer epoch differs from preflight")
    if not preflight.binds_ledger_state(boundary_state):
        raise ExecutionReceiptIntegrityError("preflight capacity differs from its ledger boundary")
    if live_state.ledger_id != preflight.ledger_id:
        raise ExecutionReceiptIntegrityError("live ledger differs from preflight ledger")
    if live_state.writer_epoch_id != preflight.writer_epoch_id:
        raise ExecutionReceiptIntegrityError("live ledger writer epoch differs from preflight")
    if live_state.budget.fingerprint != preflight.budget_fingerprint:
        raise ExecutionReceiptIntegrityError("live ledger budget differs from preflight budget")
    if live_state.pending_reservation_ref is not None:
        raise ExecutionReceiptIntegrityError("live ledger has an open reservation")
    if live_state.tail_ref is None:
        raise ExecutionReceiptIntegrityError("live ledger has no terminal outcome tail")
    final_tail = _validate_ref(
        live_state.tail_ref,
        ATTEMPT_OUTCOME_MEDIA_TYPE,
        "live final ledger tail",
    )
    if isinstance(receipt_refs, str | bytes | bytearray):
        raise TypeError("receipt_refs must be an iterable of ArtifactRef values")
    try:
        supplied_refs = tuple(receipt_refs)
    except TypeError as exc:
        raise TypeError("receipt_refs must be iterable") from exc
    if not supplied_refs:
        raise ExecutionReceiptIntegrityError("receipt replay cannot be empty")

    expected_cells = {cell.fingerprint: cell for cell in checked_schedule.iter_cells()}
    receipts_by_slot: dict[tuple[str, int], tuple[ArtifactRef, ExecutionReceipt]] = {}
    attempts_by_cell: dict[
        str,
        list[tuple[int, AttemptDisposition, ArtifactRef, ExecutionReceipt]],
    ] = defaultdict(list)
    linked_by_receipt: dict[str, tuple[AttemptReservation, AttemptOutcome, ModelExecution]] = {}
    seen_reservations: set[str] = set()
    seen_outcomes: set[str] = set()

    for raw_ref in supplied_refs:
        receipt_ref = _validate_ref(raw_ref, EXECUTION_RECEIPT_MEDIA_TYPE, "receipt")
        try:
            loaded = checked_repository.get_json(receipt_ref, ExecutionReceipt)
            receipt = ExecutionReceipt.model_validate(loaded, strict=True)
        except Exception as exc:
            raise ExecutionReceiptIntegrityError("receipt artifact cannot be verified") from exc
        if receipt.schedule_fingerprint != checked_schedule.fingerprint:
            raise ExecutionReceiptIntegrityError("receipt belongs to another schedule")
        if (
            receipt.preflight_ref != checked_preflight_ref
            or receipt.preflight_fingerprint != preflight.fingerprint
        ):
            raise ExecutionReceiptIntegrityError("receipt belongs to another preflight boundary")
        expected_cell = expected_cells.get(receipt.cell_fingerprint)
        if expected_cell is None or receipt.cell != expected_cell:
            raise ExecutionReceiptIntegrityError("receipt belongs to a foreign evaluation cell")
        _require_attempt_index(checked_schedule, receipt.attempt_index)
        slot = (receipt.cell_fingerprint, receipt.attempt_index)
        if slot in receipts_by_slot:
            raise ExecutionReceiptIntegrityError("duplicate receipt attempt slot")
        if receipt.reservation_ref.sha256 in seen_reservations:
            raise ExecutionReceiptIntegrityError("reservation is reused by multiple receipts")
        if receipt.outcome_ref.sha256 in seen_outcomes:
            raise ExecutionReceiptIntegrityError("outcome is reused by multiple receipts")

        reservation, outcome, execution = _load_linked_attempt(
            checked_repository,
            reservation_ref=receipt.reservation_ref,
            outcome_ref=receipt.outcome_ref,
        )
        if receipt.execution_ref != outcome.execution_ref:
            raise ExecutionReceiptIntegrityError("receipt execution does not match its outcome")
        if receipt.reported_tokens != outcome.reported_tokens:
            raise ExecutionReceiptIntegrityError(
                "receipt reported tokens do not match its AttemptOutcome"
            )
        if receipt.charged_tokens != outcome.charged_tokens:
            raise ExecutionReceiptIntegrityError(
                "receipt charged tokens do not match its AttemptOutcome"
            )
        _verify_cell_execution_binding(
            checked_schedule,
            receipt.cell,
            receipt.attempt_index,
            execution,
        )
        _verify_preflight_attempt_binding(preflight, reservation, outcome, execution)

        receipts_by_slot[slot] = (receipt_ref, receipt)
        attempts_by_cell[receipt.cell_fingerprint].append(
            (receipt.attempt_index, outcome.disposition, receipt_ref, receipt)
        )
        linked_by_receipt[receipt_ref.sha256] = (reservation, outcome, execution)
        seen_reservations.add(receipt.reservation_ref.sha256)
        seen_outcomes.add(receipt.outcome_ref.sha256)

    missing_cells = set(expected_cells).difference(attempts_by_cell)
    if missing_cells:
        raise ExecutionReceiptIntegrityError(
            f"receipt replay is missing {len(missing_cells)} scheduled evaluation cell(s)"
        )
    if len(attempts_by_cell) != checked_schedule.cell_count:
        raise ExecutionReceiptIntegrityError("receipt replay does not cover the exact schedule")

    for attempts in attempts_by_cell.values():
        attempts.sort(key=lambda item: item[0])
        indexes = tuple(item[0] for item in attempts)
        if indexes != tuple(range(len(indexes))):
            raise ExecutionReceiptIntegrityError(
                "receipt retry indexes must be contiguous and begin at zero"
            )
        sequences = tuple(linked_by_receipt[item[2].sha256][1].sequence for item in attempts)
        if any(later <= earlier for earlier, later in pairwise(sequences)):
            raise ExecutionReceiptIntegrityError(
                "receipt retry indexes contradict physical ledger order"
            )
        for _, disposition, _, _ in attempts[:-1]:
            if disposition is not AttemptDisposition.BURNED:
                raise ExecutionReceiptIntegrityError(
                    "a settled or poisoned attempt cannot be followed by a retry"
                )

    paired_attempts: dict[str, dict[str, tuple[tuple[int, str], ...]]] = defaultdict(dict)
    for cell_fingerprint, attempts in attempts_by_cell.items():
        cell = expected_cells[cell_fingerprint]
        paired_attempts[cell.pairing_fingerprint][cell.side.value] = tuple(
            (item[0], linked_by_receipt[item[2].sha256][2].execution_fingerprint)
            for item in attempts
        )
    for sides in paired_attempts.values():
        if set(sides) != {"parent", "candidate"}:
            raise ExecutionReceiptIntegrityError("paired receipt schedule is missing a side")
        if sides["parent"] != sides["candidate"]:
            raise ExecutionReceiptIntegrityError(
                "parent and candidate retry executions must be exactly paired"
            )

    if any(
        outcome.disposition is AttemptDisposition.POISONED
        for _, outcome, _ in linked_by_receipt.values()
    ):
        raise ExecutionReceiptIntegrityError(
            "a poisoned attempt invalidates the complete paired evaluation batch"
        )

    ledger_ids = {
        linked_by_receipt[receipt_ref.sha256][1].ledger_id
        for receipt_ref, _ in receipts_by_slot.values()
    }
    if len(ledger_ids) != 1:
        raise ExecutionReceiptIntegrityError(
            "one preflight boundary cannot authorize multiple attempt ledgers"
        )
    ledger_id = next(iter(ledger_ids))
    if preflight.ledger_id is not None and preflight.ledger_id != ledger_id:
        raise ExecutionReceiptIntegrityError("receipt ledger differs from preflight ledger")

    receipt_entries = sorted(
        (
            (receipt_ref, receipt, linked_by_receipt[receipt_ref.sha256][1])
            for receipt_ref, receipt in receipts_by_slot.values()
        ),
        key=lambda item: item[2].sequence,
    )
    highest_sequence = receipt_entries[-1][2].sequence
    highest = [item for item in receipt_entries if item[2].sequence == highest_sequence]
    if len(highest) != 1:
        raise ExecutionReceiptIntegrityError("ledger receipts contain divergent tails")
    chain, chain_order = _replay_attempt_chain(checked_repository, final_tail)

    if preflight.ledger_tail_ref is None:
        batch_chain = chain_order
    else:
        boundary_positions = tuple(
            index for index, ref in enumerate(chain_order) if ref == preflight.ledger_tail_ref
        )
        if len(boundary_positions) != 1:
            raise ExecutionReceiptIntegrityError(
                "preflight start tail is not an ancestor of the final ledger tail"
            )
        batch_chain = chain_order[boundary_positions[0] + 1 :]

    receipt_outcome_digests = {receipt.outcome_ref.sha256 for _, receipt, _ in receipt_entries}
    batch_chain_digests = {ref.sha256 for ref in batch_chain}
    if len(batch_chain) != len(receipt_entries) or batch_chain_digests != receipt_outcome_digests:
        raise ExecutionReceiptIntegrityError(
            "receipt set does not exactly cover every outcome after the preflight boundary"
        )
    for _, receipt, outcome in receipt_entries:
        chained = chain.get(receipt.outcome_ref.sha256)
        if chained is None or chained[1] != outcome:
            raise ExecutionReceiptIntegrityError(
                "receipt outcome is not an ancestor of its ledger tail"
            )

    ordered_attempts = sorted(
        (
            (receipt.cell_fingerprint, receipt.attempt_index, receipt_ref, receipt)
            for receipt_ref, receipt in receipts_by_slot.values()
        ),
        key=lambda item: (item[0], item[1]),
    )
    ordered_refs = tuple(item[2] for item in ordered_attempts)
    ordered_outcomes = tuple(linked_by_receipt[ref.sha256][1] for ref in ordered_refs)
    settled = sum(outcome.disposition is AttemptDisposition.SETTLED for outcome in ordered_outcomes)
    burned = sum(outcome.disposition is AttemptDisposition.BURNED for outcome in ordered_outcomes)
    poisoned = sum(
        outcome.disposition is AttemptDisposition.POISONED for outcome in ordered_outcomes
    )
    reported_tokens = sum(outcome.reported_tokens for outcome in ordered_outcomes)
    charged_tokens = sum(outcome.charged_tokens for outcome in ordered_outcomes)
    try:
        closing_state = attempt_ledger.state()
    except Exception as exc:
        raise ExecutionReceiptIntegrityError("live attempt ledger cannot be reverified") from exc
    if (
        closing_state.tail_ref != final_tail
        or closing_state.pending_reservation_ref is not None
        or closing_state.ledger_id != preflight.ledger_id
        or closing_state.writer_epoch_id != preflight.writer_epoch_id
        or closing_state.budget.fingerprint != preflight.budget_fingerprint
    ):
        raise ExecutionReceiptIntegrityError(
            "live attempt ledger advanced during trusted usage replay"
        )
    return TrustedExecutionUsage(
        schedule_fingerprint=checked_schedule.fingerprint,
        receipt_refs=ordered_refs,
        ledger_tail_refs=(final_tail,),
        cell_count=checked_schedule.cell_count,
        attempt_count=len(ordered_outcomes),
        settled_attempts=settled,
        burned_attempts=burned,
        poisoned_attempts=poisoned,
        reported_tokens=reported_tokens,
        charged_tokens=charged_tokens,
    )


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def _require_attempt_index(schedule: EvaluationBatchSchedule, attempt_index: int) -> None:
    if type(attempt_index) is not int:
        raise TypeError("attempt_index must be an integer")
    if attempt_index < 0:
        raise ValueError("attempt_index must not be negative")
    if attempt_index >= schedule.max_attempts_per_cell:
        raise ExecutionReceiptIntegrityError("receipt exceeds the frozen retry ceiling")


def _validate_ref(
    ref: ArtifactRef,
    expected_media_type: str,
    label: str,
) -> ArtifactRef:
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise ExecutionReceiptIntegrityError(f"{label} reference is malformed") from exc
    if checked.media_type != expected_media_type:
        raise ExecutionReceiptIntegrityError(f"{label} reference declares the wrong media type")
    return checked


def _load_model[ModelT: ImmutableModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    try:
        loaded = repository.get_json(ref, model_type)
        return model_type.model_validate(loaded, strict=True)
    except Exception as exc:
        raise ExecutionReceiptIntegrityError(f"{label} artifact cannot be verified") from exc


def _load_preflight(
    repository: ArtifactRepository,
    preflight_ref: ArtifactRef,
    schedule: EvaluationBatchSchedule,
) -> tuple[ArtifactRef, SchedulePreflightCertificate]:
    checked_ref = _validate_ref(
        preflight_ref,
        SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        "preflight",
    )
    preflight = _load_model(
        repository,
        checked_ref,
        SchedulePreflightCertificate,
        "preflight",
    )
    if preflight.fingerprint != checked_ref.sha256:
        raise ExecutionReceiptIntegrityError(
            "preflight reference does not match its canonical certificate fingerprint"
        )
    if not preflight.binds_schedule(schedule):
        raise ExecutionReceiptIntegrityError("preflight shape differs from the frozen schedule")
    return checked_ref, preflight


def _verify_preflight_attempt_binding(
    preflight: SchedulePreflightCertificate,
    reservation: AttemptReservation,
    outcome: AttemptOutcome,
    execution: ModelExecution,
) -> None:
    if not preflight.binds_execution(execution):
        raise ExecutionReceiptIntegrityError("execution model boundary differs from preflight")
    if (
        reservation.budget_fingerprint != preflight.budget_fingerprint
        or outcome.budget_fingerprint != preflight.budget_fingerprint
    ):
        raise ExecutionReceiptIntegrityError("attempt budget differs from preflight budget")
    if preflight.ledger_id is not None and (
        reservation.ledger_id != preflight.ledger_id or outcome.ledger_id != preflight.ledger_id
    ):
        raise ExecutionReceiptIntegrityError("attempt ledger differs from preflight ledger")
    if preflight.writer_epoch_id is not None and (
        reservation.writer_epoch_id != preflight.writer_epoch_id
        or outcome.writer_epoch_id != preflight.writer_epoch_id
    ):
        raise ExecutionReceiptIntegrityError("attempt writer epoch differs from preflight")
    if reservation.reserved_tokens != preflight.token_ceiling_per_attempt:
        raise ExecutionReceiptIntegrityError(
            "attempt reservation differs from the preflight token ceiling"
        )


def _load_linked_attempt(
    repository: ArtifactRepository,
    *,
    reservation_ref: ArtifactRef,
    outcome_ref: ArtifactRef,
) -> tuple[AttemptReservation, AttemptOutcome, ModelExecution]:
    checked_reservation_ref = _validate_ref(
        reservation_ref,
        ATTEMPT_RESERVATION_MEDIA_TYPE,
        "reservation",
    )
    checked_outcome_ref = _validate_ref(outcome_ref, ATTEMPT_OUTCOME_MEDIA_TYPE, "outcome")
    reservation = _load_model(
        repository,
        checked_reservation_ref,
        AttemptReservation,
        "reservation",
    )
    outcome = _load_model(repository, checked_outcome_ref, AttemptOutcome, "outcome")
    _verify_reservation_outcome_pair(checked_reservation_ref, reservation, outcome)
    if outcome.execution_ref is None:
        raise ExecutionReceiptIntegrityError(
            "trusted execution receipt requires a persisted ModelExecution"
        )
    execution = _load_model(
        repository,
        outcome.execution_ref,
        ModelExecution,
        "execution",
    )
    _verify_execution_binding(repository, reservation, outcome, execution)
    return reservation, outcome, execution


def _verify_reservation_outcome_pair(
    reservation_ref: ArtifactRef,
    reservation: AttemptReservation,
    outcome: AttemptOutcome,
) -> None:
    if outcome.reservation_ref != reservation_ref:
        raise ExecutionReceiptIntegrityError("outcome does not consume the receipt reservation")
    if reservation.ledger_id != outcome.ledger_id:
        raise ExecutionReceiptIntegrityError("reservation and outcome ledger ids differ")
    if reservation.budget_fingerprint != outcome.budget_fingerprint:
        raise ExecutionReceiptIntegrityError("reservation and outcome budgets differ")
    if reservation.sequence != outcome.sequence:
        raise ExecutionReceiptIntegrityError("reservation and outcome sequences differ")
    if outcome.disposition is AttemptDisposition.SETTLED:
        if outcome.reported_tokens > reservation.reserved_tokens:
            raise ExecutionReceiptIntegrityError("settled usage exceeds its reservation")
        if outcome.charged_tokens != outcome.reported_tokens:
            raise ExecutionReceiptIntegrityError("settled charge differs from reported usage")
    elif outcome.disposition is AttemptDisposition.BURNED:
        if outcome.reported_tokens > reservation.reserved_tokens:
            raise ExecutionReceiptIntegrityError("an overrun cannot be recorded as burned")
        if outcome.charged_tokens != reservation.reserved_tokens:
            raise ExecutionReceiptIntegrityError("burned attempt did not charge its reservation")
    else:
        if outcome.reported_tokens <= reservation.reserved_tokens:
            raise ExecutionReceiptIntegrityError("poisoned outcome does not contain an overrun")
        if outcome.charged_tokens != outcome.reported_tokens:
            raise ExecutionReceiptIntegrityError("poisoned charge differs from reported usage")


def _verify_execution_binding(
    repository: ArtifactRepository,
    reservation: AttemptReservation,
    outcome: AttemptOutcome,
    execution: ModelExecution,
) -> None:
    if execution.task.fingerprint != reservation.task_fingerprint:
        raise ExecutionReceiptIntegrityError("execution task differs from its reservation")
    if execution.execution_fingerprint != reservation.execution_fingerprint:
        raise ExecutionReceiptIntegrityError("execution fingerprint differs from its reservation")
    if execution.request_sha256 != reservation.request_sha256:
        raise ExecutionReceiptIntegrityError("execution request differs from its reservation")
    try:
        materializer = _materialization.HarnessMaterializer(repository, spec=execution.spec)
        materializer.verify_execution_request(execution.request.harness_ref, execution)
    except _materialization.HarnessMaterializationError as exc:
        raise ExecutionReceiptIntegrityError(f"harness request replay failed: {exc}") from exc
    if execution.usage.total_tokens != outcome.reported_tokens:
        raise ExecutionReceiptIntegrityError("execution tokens differ from its AttemptOutcome")
    if (
        outcome.disposition is AttemptDisposition.SETTLED
        and execution.status is not ExecutionStatus.COMPLETED
    ):
        raise ExecutionReceiptIntegrityError(
            "settled outcome does not reference a completed execution"
        )


def _verify_cell_execution_binding(
    schedule: EvaluationBatchSchedule,
    cell: EvaluationCellKey,
    attempt_index: int,
    execution: ModelExecution,
) -> None:
    if execution.task_id != cell.task_id:
        raise ExecutionReceiptIntegrityError("execution task does not match its receipt cell")
    expected_harness_id = schedule.harness_id_for(cell.side)
    if execution.harness_id != expected_harness_id:
        raise ExecutionReceiptIntegrityError(
            "execution harness does not match the frozen receipt side"
        )
    expected_seed = schedule.seed_for(cell, attempt_index=attempt_index)
    if execution.seed != expected_seed:
        raise ExecutionReceiptIntegrityError(
            "execution seed does not bind the receipt's run/phase/query coordinates"
        )


def _replay_attempt_chain(
    repository: ArtifactRepository,
    tail_ref: ArtifactRef,
) -> tuple[
    dict[str, tuple[AttemptReservation, AttemptOutcome, ModelExecution | None]],
    tuple[ArtifactRef, ...],
]:
    cursor: ArtifactRef | None = _validate_ref(
        tail_ref,
        ATTEMPT_OUTCOME_MEDIA_TYPE,
        "ledger tail",
    )
    backwards: list[
        tuple[ArtifactRef, AttemptReservation, AttemptOutcome, ModelExecution | None]
    ] = []
    seen: set[str] = set()
    ledger_id: str | None = None
    budget_fingerprint: str | None = None

    while cursor is not None:
        if cursor.sha256 in seen:
            raise ExecutionReceiptIntegrityError("attempt ledger contains a cycle")
        seen.add(cursor.sha256)
        outcome_ref = cursor
        outcome = _load_model(repository, outcome_ref, AttemptOutcome, "outcome")
        reservation_ref = outcome.reservation_ref
        if reservation_ref.sha256 in seen:
            raise ExecutionReceiptIntegrityError("attempt ledger contains a cycle")
        seen.add(reservation_ref.sha256)
        reservation = _load_model(
            repository,
            reservation_ref,
            AttemptReservation,
            "reservation",
        )
        _verify_reservation_outcome_pair(reservation_ref, reservation, outcome)
        execution: ModelExecution | None = None
        if outcome.execution_ref is not None:
            execution = _load_model(
                repository,
                outcome.execution_ref,
                ModelExecution,
                "execution",
            )
            _verify_execution_binding(repository, reservation, outcome, execution)
        if ledger_id is None:
            ledger_id = outcome.ledger_id
            budget_fingerprint = outcome.budget_fingerprint
        elif outcome.ledger_id != ledger_id or outcome.budget_fingerprint != budget_fingerprint:
            raise ExecutionReceiptIntegrityError("attempt chain crosses a ledger or budget")
        backwards.append((outcome_ref, reservation, outcome, execution))
        cursor = reservation.previous_outcome_ref

    entries = tuple(reversed(backwards))
    previous_outcome_ref: ArtifactRef | None = None
    poisoned = False
    replayed: dict[
        str,
        tuple[AttemptReservation, AttemptOutcome, ModelExecution | None],
    ] = {}
    ordered_refs: list[ArtifactRef] = []
    for expected_sequence, (outcome_ref, reservation, outcome, execution) in enumerate(entries):
        if poisoned:
            raise ExecutionReceiptIntegrityError("attempt entries follow a poisoned outcome")
        if reservation.sequence != expected_sequence or outcome.sequence != expected_sequence:
            raise ExecutionReceiptIntegrityError("attempt sequences are not contiguous")
        if reservation.previous_outcome_ref != previous_outcome_ref:
            raise ExecutionReceiptIntegrityError("attempt chain link skips its prior outcome")
        replayed[outcome_ref.sha256] = (reservation, outcome, execution)
        ordered_refs.append(outcome_ref)
        poisoned = outcome.disposition is AttemptDisposition.POISONED
        previous_outcome_ref = outcome_ref
    return replayed, tuple(ordered_refs)


__all__ = [
    "EXECUTION_RECEIPT_MEDIA_TYPE",
    "ExecutionReceipt",
    "ExecutionReceiptError",
    "ExecutionReceiptIntegrityError",
    "ExecutionReceiptKey",
    "ScheduledExecutionRecord",
    "TrustedExecutionUsage",
    "execute_scheduled_attempt",
    "publish_execution_receipt",
    "replay_trusted_usage",
]
