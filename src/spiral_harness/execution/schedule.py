"""Canonical evaluation coordinates, seed derivation, and budget preflight.

The scheduler models one immutable evaluation batch.  A batch is deliberately
smaller than a complete study: it has one baseline kind, phase, and query
number, then expands the frozen roster across independent search runs, paired
repeat seeds, and both harness sides.  This boundary lets a controller reject
an entire next batch before it invokes either a model backend or an optimizer.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.attempts import AttemptBudgetExceeded
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    AttemptBudget,
    AttemptLedgerState,
    FrozenModelSpec,
    ModelExecution,
)
from spiral_harness.execution.model import paired_execution_fingerprint
from spiral_harness.storage.protocol import ArtifactRepository

_SEED_MASK = (1 << 63) - 1
_SEED_SCHEMA = "spiral-harness/deterministic-seed/v2"
_CELL_SCHEMA = "spiral-harness/evaluation-cell/v2"
_PAIRED_CELL_SCHEMA = "spiral-harness/paired-evaluation-cell/v2"
SCHEDULE_PREFLIGHT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.schedule-preflight-certificate.v2+json"
)


class EvaluationSide(StrEnum):
    """The two members of a complete paired evaluation cell."""

    PARENT = "parent"
    CANDIDATE = "candidate"


class EvaluationPhase(StrEnum):
    """Trusted benchmark-access phases supported by the first study."""

    EXPLORATION = "exploration"
    PROBE = "probe"
    GATE = "gate"
    SEALED = "sealed"


class PairedEvaluationCellKey(ImmutableModel):
    """Pairing coordinates shared by parent and candidate executions.

    ``search_run``, ``phase``, and ``query`` intentionally remain in this key.
    Removing any of them would make distinct adaptive evaluations collide.
    Harness identity and side intentionally do not appear, so the two arms can
    share randomness inside an otherwise exact matched pair.
    """

    schema_version: Literal["2"] = "2"
    study: NonEmptyStr
    kind: NonEmptyStr
    search_run: Annotated[int, Field(ge=0, strict=True)]
    phase: EvaluationPhase
    query: Annotated[int, Field(ge=0, strict=True)]
    task_id: NonEmptyStr
    repeat_seed: Annotated[int, Field(ge=0, strict=True)]

    @field_validator("study", "kind", "task_id", mode="before")
    @classmethod
    def coordinate_text_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("evaluation coordinate text must be exact and non-empty")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"schema": _PAIRED_CELL_SCHEMA, "key": self})


class EvaluationCellKey(ImmutableModel):
    """The complete logical coordinate of one harness execution."""

    schema_version: Literal["2"] = "2"
    study: NonEmptyStr
    kind: NonEmptyStr
    search_run: Annotated[int, Field(ge=0, strict=True)]
    phase: EvaluationPhase
    query: Annotated[int, Field(ge=0, strict=True)]
    task_id: NonEmptyStr
    repeat_seed: Annotated[int, Field(ge=0, strict=True)]
    side: EvaluationSide

    @field_validator("study", "kind", "task_id", mode="before")
    @classmethod
    def coordinate_text_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("evaluation coordinate text must be exact and non-empty")
        return value

    @property
    def paired_key(self) -> PairedEvaluationCellKey:
        return PairedEvaluationCellKey(
            study=self.study,
            kind=self.kind,
            search_run=self.search_run,
            phase=self.phase,
            query=self.query,
            task_id=self.task_id,
            repeat_seed=self.repeat_seed,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"schema": _CELL_SCHEMA, "key": self})

    @property
    def pairing_fingerprint(self) -> str:
        return self.paired_key.fingerprint


def derive_seed_v2(
    *,
    master_seed: int,
    domain: str,
    coordinates: EvaluationCellKey | PairedEvaluationCellKey,
    nonce: int = 0,
) -> int:
    """Derive a portable non-negative 63-bit seed from typed coordinates.

    Callers choose whether side belongs in the derivation by passing a full or
    paired cell key.  The explicit domain and schema label prevent a digest
    used for model rollouts from being silently reused for proposals or other
    future stochastic decisions.  ``nonce`` is normally the retry index.
    """

    if type(master_seed) is not int:
        raise TypeError("master_seed must be an integer")
    if master_seed < 0:
        raise ValueError("master_seed must not be negative")
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    if not domain or domain != domain.strip():
        raise ValueError("domain must be exact and non-empty")
    if type(nonce) is not int:
        raise TypeError("nonce must be an integer")
    if nonce < 0:
        raise ValueError("nonce must not be negative")
    if isinstance(coordinates, EvaluationCellKey):
        checked_coordinates: EvaluationCellKey | PairedEvaluationCellKey = (
            EvaluationCellKey.model_validate(coordinates, strict=True)
        )
        coordinate_kind = "full"
    elif isinstance(coordinates, PairedEvaluationCellKey):
        checked_coordinates = PairedEvaluationCellKey.model_validate(coordinates, strict=True)
        coordinate_kind = "paired"
    else:
        raise TypeError("coordinates must be an evaluation cell key")

    digest = canonical_sha256(
        {
            "schema": _SEED_SCHEMA,
            "domain": domain,
            "master_seed": master_seed,
            "coordinate_kind": coordinate_kind,
            "coordinates": checked_coordinates,
            "nonce": nonce,
        }
    )
    # Providers commonly accept signed 64-bit seeds.  Masking instead of using
    # Python's arbitrary-width integer keeps the value portable across runtimes.
    return int(digest[:16], 16) & _SEED_MASK


class EvaluationBatchSchedule(ImmutableModel):
    """Frozen Cartesian schedule for one study/kind/phase/query batch."""

    schema_version: Literal["2"] = "2"
    study: NonEmptyStr
    kind: NonEmptyStr
    phase: EvaluationPhase
    query: Annotated[int, Field(ge=0, strict=True)]
    master_seed: Annotated[int, Field(ge=0, strict=True)]
    parent_harness_id: NonEmptyStr
    candidate_harness_id: NonEmptyStr
    task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    search_runs: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=1),
    ]
    repeat_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=1),
    ]
    sides: tuple[EvaluationSide, ...] = (
        EvaluationSide.PARENT,
        EvaluationSide.CANDIDATE,
    )
    max_attempts_per_cell: Annotated[int, Field(ge=1, strict=True)]
    token_ceiling_per_attempt: Annotated[int, Field(ge=1, strict=True)]

    @field_validator(
        "study",
        "kind",
        "parent_harness_id",
        "candidate_harness_id",
        mode="before",
    )
    @classmethod
    def batch_text_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("schedule coordinate text must be exact and non-empty")
        return value

    @field_validator("task_ids")
    @classmethod
    def canonicalize_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or value != value.strip():
                raise ValueError("task_ids must contain exact non-empty identifiers")
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("task_ids must not contain duplicates")
        return ordered

    @field_validator("search_runs", "repeat_seeds")
    @classmethod
    def canonicalize_integer_axes(
        cls,
        values: tuple[int, ...],
        info: object,
    ) -> tuple[int, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            field_name = getattr(info, "field_name", "schedule axis")
            raise ValueError(f"{field_name} must not contain duplicates")
        return ordered

    @field_validator("sides")
    @classmethod
    def canonicalize_complete_pair(
        cls,
        values: tuple[EvaluationSide, ...],
    ) -> tuple[EvaluationSide, ...]:
        required = {EvaluationSide.PARENT, EvaluationSide.CANDIDATE}
        if len(values) != 2 or set(values) != required:
            raise ValueError("sides must contain exactly parent and candidate")
        return (EvaluationSide.PARENT, EvaluationSide.CANDIDATE)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def cell_count(self) -> int:
        return len(self.task_ids) * len(self.search_runs) * len(self.repeat_seeds) * len(self.sides)

    @property
    def required_attempts(self) -> int:
        """Worst-case reservations needed for an all-or-nothing batch."""

        return self.cell_count * self.max_attempts_per_cell

    @property
    def required_tokens(self) -> int:
        """Worst-case charged tokens if every reservation is burned."""

        return self.required_attempts * self.token_ceiling_per_attempt

    def iter_cells(self) -> Iterator[EvaluationCellKey]:
        """Yield cells lazily in canonical order instead of materializing a matrix."""

        for search_run in self.search_runs:
            for task_id in self.task_ids:
                for repeat_seed in self.repeat_seeds:
                    for side in self.sides:
                        yield EvaluationCellKey(
                            study=self.study,
                            kind=self.kind,
                            search_run=search_run,
                            phase=self.phase,
                            query=self.query,
                            task_id=task_id,
                            repeat_seed=repeat_seed,
                            side=side,
                        )

    def contains(self, cell: EvaluationCellKey) -> bool:
        """Return whether *cell* belongs to this exact frozen batch."""

        try:
            checked = EvaluationCellKey.model_validate(cell, strict=True)
        except Exception:
            return False
        return (
            checked.study == self.study
            and checked.kind == self.kind
            and checked.phase is self.phase
            and checked.query == self.query
            and checked.search_run in self.search_runs
            and checked.task_id in self.task_ids
            and checked.repeat_seed in self.repeat_seeds
            and checked.side in self.sides
        )

    def harness_id_for(self, side: EvaluationSide) -> str:
        """Return the exact frozen harness identity for one paired side."""

        try:
            checked_side = EvaluationSide(side)
        except (TypeError, ValueError) as exc:
            raise TypeError("side must be an EvaluationSide") from exc
        if checked_side is EvaluationSide.PARENT:
            return self.parent_harness_id
        return self.candidate_harness_id

    def seed_for(self, cell: EvaluationCellKey, *, attempt_index: int = 0) -> int:
        """Return paired model randomness for one bounded attempt slot."""

        checked = EvaluationCellKey.model_validate(cell, strict=True)
        if not self.contains(checked):
            raise ValueError("cell does not belong to this evaluation schedule")
        if type(attempt_index) is not int:
            raise TypeError("attempt_index must be an integer")
        if attempt_index < 0:
            raise ValueError("attempt_index must not be negative")
        if attempt_index >= self.max_attempts_per_cell:
            raise ValueError("attempt_index exceeds the frozen retry ceiling")
        return derive_seed_v2(
            master_seed=self.master_seed,
            domain="model-rollout",
            coordinates=checked.paired_key,
            nonce=attempt_index,
        )


class ScheduleBudgetExceeded(AttemptBudgetExceeded):
    """Raised before a batch when its complete worst-case reservation cannot fit."""


class SchedulePreflightCertificate(ImmutableModel):
    """Immutable proof of the exact capacity checked before batch execution."""

    schema_version: Literal["2"] = "2"
    schedule_fingerprint: Sha256
    model_spec: FrozenModelSpec
    budget_fingerprint: Sha256
    ledger_id: NonEmptyStr | None
    ledger_tail_ref: ArtifactRef | None
    cell_count: Annotated[int, Field(ge=1, strict=True)]
    max_attempts_per_cell: Annotated[int, Field(ge=1, strict=True)]
    token_ceiling_per_attempt: Annotated[int, Field(ge=1, strict=True)]
    required_attempts: Annotated[int, Field(ge=1, strict=True)]
    required_tokens: Annotated[int, Field(ge=1, strict=True)]
    available_attempts: Annotated[int, Field(ge=0, strict=True)]
    available_tokens: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def arithmetic_and_capacity_are_consistent(self) -> Self:
        if (
            self.ledger_tail_ref is not None
            and self.ledger_tail_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE
        ):
            raise ValueError("ledger_tail_ref declares the wrong media type")
        if self.ledger_tail_ref is not None and self.ledger_id is None:
            raise ValueError("a non-empty ledger boundary requires ledger_id")
        if self.required_attempts != self.cell_count * self.max_attempts_per_cell:
            raise ValueError("required_attempts does not match the frozen schedule shape")
        if self.required_tokens != self.required_attempts * self.token_ceiling_per_attempt:
            raise ValueError("required_tokens does not match the reservation ceiling")
        if self.required_attempts > self.available_attempts:
            raise ValueError("certificate claims insufficient attempt capacity")
        if self.required_tokens > self.available_tokens:
            raise ValueError("certificate claims insufficient token capacity")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    def binds_execution(self, execution: ModelExecution) -> bool:
        """Return whether an execution exactly satisfies this frozen model boundary."""

        checked = ModelExecution.model_validate(execution, strict=True)
        return checked.spec == self.model_spec and checked.execution_fingerprint == (
            paired_execution_fingerprint(
                self.model_spec,
                checked.task,
                seed=checked.seed,
                backend_fingerprint=self.model_spec.backend_fingerprint,
            )
        )


def preflight_attempt_budget(
    schedule: EvaluationBatchSchedule,
    capacity: AttemptBudget | AttemptLedgerState,
    model_spec: FrozenModelSpec,
) -> SchedulePreflightCertificate:
    """Atomically decide whether the complete frozen batch can be started.

    This function is pure and performs no repository, backend, or optimizer
    call.  Controllers must obtain the certificate before the first such call
    and must discard it if the bound ledger tail changes.
    """

    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    checked_model_spec = FrozenModelSpec.model_validate(model_spec, strict=True)
    if isinstance(capacity, AttemptBudget):
        budget = AttemptBudget.model_validate(capacity, strict=True)
        ledger_id = None
        ledger_tail_ref = None
        available_attempts = budget.max_attempts
        available_tokens = budget.max_total_tokens
    elif isinstance(capacity, AttemptLedgerState):
        state = AttemptLedgerState.model_validate(capacity, strict=True)
        budget = state.budget
        if state.poisoned:
            raise ScheduleBudgetExceeded("attempt ledger is poisoned")
        if state.pending_reservation_ref is not None:
            raise ScheduleBudgetExceeded(
                "all-or-nothing preflight requires a ledger without an open reservation"
            )
        ledger_id = state.ledger_id
        ledger_tail_ref = state.tail_ref
        available_attempts = state.remaining_attempts
        available_tokens = state.remaining_tokens
    else:
        raise TypeError("capacity must be an AttemptBudget or AttemptLedgerState")

    if checked_schedule.token_ceiling_per_attempt > budget.max_tokens_per_attempt:
        raise ScheduleBudgetExceeded(
            "schedule token ceiling exceeds the attempt budget per-attempt ceiling"
        )
    if checked_schedule.required_attempts > available_attempts:
        raise ScheduleBudgetExceeded(
            "remaining attempt budget cannot fit the complete evaluation batch"
        )
    if checked_schedule.required_tokens > available_tokens:
        raise ScheduleBudgetExceeded(
            "remaining token budget cannot fit the complete evaluation batch"
        )

    return SchedulePreflightCertificate(
        schedule_fingerprint=checked_schedule.fingerprint,
        model_spec=checked_model_spec,
        budget_fingerprint=budget.fingerprint,
        ledger_id=ledger_id,
        ledger_tail_ref=ledger_tail_ref,
        cell_count=checked_schedule.cell_count,
        max_attempts_per_cell=checked_schedule.max_attempts_per_cell,
        token_ceiling_per_attempt=checked_schedule.token_ceiling_per_attempt,
        required_attempts=checked_schedule.required_attempts,
        required_tokens=checked_schedule.required_tokens,
        available_attempts=available_attempts,
        available_tokens=available_tokens,
    )


def publish_schedule_preflight(
    repository: ArtifactRepository,
    certificate: SchedulePreflightCertificate,
) -> ArtifactRef:
    """Persist and verify the exact boundary that later receipts must bind."""

    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    checked = SchedulePreflightCertificate.model_validate(certificate, strict=True)
    raw_ref = repository.put_json(checked, media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE)
    try:
        ref = ArtifactRef.model_validate(raw_ref, strict=True)
    except Exception as exc:
        raise RuntimeError("repository returned a malformed preflight reference") from exc
    if ref.media_type != SCHEDULE_PREFLIGHT_MEDIA_TYPE:
        raise RuntimeError("repository returned the wrong preflight media type")
    try:
        loaded = repository.get_json(ref, SchedulePreflightCertificate)
        published = SchedulePreflightCertificate.model_validate(loaded, strict=True)
    except Exception as exc:
        raise RuntimeError("published preflight certificate cannot be verified") from exc
    if published != checked:
        raise RuntimeError("published preflight certificate changed content")
    return ref


__all__ = [
    "SCHEDULE_PREFLIGHT_MEDIA_TYPE",
    "EvaluationBatchSchedule",
    "EvaluationCellKey",
    "EvaluationPhase",
    "EvaluationSide",
    "PairedEvaluationCellKey",
    "ScheduleBudgetExceeded",
    "SchedulePreflightCertificate",
    "derive_seed_v2",
    "preflight_attempt_budget",
    "publish_schedule_preflight",
]
