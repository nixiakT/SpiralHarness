"""Prepare complete matched skill-probe arms before executing either backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    CandidateTask,
    FrozenModelSpec,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.execution.receipts import (
    execute_scheduled_attempt,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationCellKey,
    EvaluationSide,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_inclusion import (
    publish_settled_skill_request_inclusion,
    verify_settled_skill_request_inclusion,
)

from .skill_probe_closure import SkillProbeArmClosure


class _SkillProbeArmError(RuntimeError):
    """Raised when one arm cannot be prepared or closed exactly."""


@dataclass(frozen=True, slots=True)
class _PreparedSkillProbeArm:
    """All live authority and immutable inputs required to execute one arm."""

    control: Literal["revert", "placebo"]
    repository: ArtifactRepository
    schedule: EvaluationBatchSchedule
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    tasks: Mapping[str, CandidateTask]
    pairs: tuple[tuple[EvaluationCellKey, EvaluationCellKey], ...]
    runner: FixedModelRunner
    ledger: AttemptLedger
    opening_ledger_tail_ref: ArtifactRef | None
    preflight_ref: ArtifactRef

    @property
    def terminal_tail_ref(self) -> ArtifactRef | None:
        """Return the exact live tail reached so far, including a pending reservation."""

        return self.ledger.tail_ref


def _paired_cells(
    schedule: EvaluationBatchSchedule,
) -> tuple[tuple[EvaluationCellKey, EvaluationCellKey], ...]:
    cells = tuple(schedule.iter_cells())
    if len(cells) % 2:
        raise _SkillProbeArmError("probe schedule has an incomplete pair")
    pairs: list[tuple[EvaluationCellKey, EvaluationCellKey]] = []
    for index in range(0, len(cells), 2):
        parent, candidate = cells[index : index + 2]
        if (
            parent.side is not EvaluationSide.PARENT
            or candidate.side is not EvaluationSide.CANDIDATE
            or parent.paired_key != candidate.paired_key
        ):
            raise _SkillProbeArmError("probe schedule is not in canonical paired order")
        pairs.append((parent, candidate))
    return tuple(pairs)


def _prepare_skill_probe_arm(
    repository: ArtifactRepository,
    *,
    control: Literal["revert", "placebo"],
    schedule: EvaluationBatchSchedule,
    parent_harness_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    tasks: Mapping[str, CandidateTask],
    spec: FrozenModelSpec,
    backend: ModelBackend,
    execution_nonce: str,
) -> _PreparedSkillProbeArm:
    """Validate and publish one arm's preflight without invoking its backend."""

    try:
        if control not in {"revert", "placebo"}:
            raise ValueError("control must be revert or placebo")
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
        checked_parent_ref = ArtifactRef.model_validate(parent_harness_ref, strict=True)
        checked_candidate_ref = ArtifactRef.model_validate(candidate_harness_ref, strict=True)
        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        if not isinstance(backend, ModelBackend):
            raise TypeError("backend must implement ModelBackend")
        if backend.fingerprint != checked_spec.backend_fingerprint:
            raise ValueError("backend fingerprint differs from the frozen model spec")
        checked_tasks = {
            task_id: CandidateTask.from_task_view(task) for task_id, task in tasks.items()
        }
        pairs = _paired_cells(checked_schedule)
        scheduled_ids = {cell.task_id for pair in pairs for cell in pair}
        if not scheduled_ids.issubset(checked_tasks):
            raise ValueError("scheduled cell is absent from the frozen task roster")

        materializer = HarnessMaterializer(repository, spec=checked_spec)
        materializer.materialize(checked_parent_ref)
        materializer.materialize(checked_candidate_ref)
        budget = AttemptBudget(
            max_attempts=checked_schedule.required_attempts,
            max_total_tokens=checked_schedule.required_tokens,
            max_tokens_per_attempt=checked_schedule.token_ceiling_per_attempt,
        )
        ledger = AttemptLedger(
            repository,
            ledger_id=f"skill-probe/{execution_nonce}/{control}",
            budget=budget,
        )
        opening = ledger.state()
        preflight_ref = publish_schedule_preflight(
            repository,
            preflight_attempt_budget(checked_schedule, ledger, checked_spec),
        )
        runner = FixedModelRunner(
            spec=checked_spec,
            backend=backend,
            attempt_ledger=ledger,
        )
        return _PreparedSkillProbeArm(
            control=control,
            repository=repository,
            schedule=checked_schedule,
            parent_harness_ref=checked_parent_ref,
            candidate_harness_ref=checked_candidate_ref,
            tasks=MappingProxyType(checked_tasks),
            pairs=pairs,
            runner=runner,
            ledger=ledger,
            opening_ledger_tail_ref=opening.tail_ref,
            preflight_ref=preflight_ref,
        )
    except _SkillProbeArmError:
        raise
    except Exception as exc:
        raise _SkillProbeArmError(f"{control} arm preparation failed: {exc}") from exc


def _harness_ref(
    prepared: _PreparedSkillProbeArm,
    cell: EvaluationCellKey,
) -> ArtifactRef:
    if cell.side is EvaluationSide.PARENT:
        return prepared.parent_harness_ref
    return prepared.candidate_harness_ref


def _load_outcome(
    prepared: _PreparedSkillProbeArm,
    ref: ArtifactRef,
) -> AttemptOutcome:
    checked_ref = ArtifactRef.model_validate(ref, strict=True)
    if checked_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
        raise _SkillProbeArmError("skill probe attempt outcome declares the wrong media type")
    try:
        loaded = prepared.repository.get_json(checked_ref, AttemptOutcome)
        return AttemptOutcome.model_validate(loaded, strict=True)
    except Exception as exc:
        raise _SkillProbeArmError("skill probe attempt outcome cannot be loaded exactly") from exc


def _execute_prepared_skill_probe_arm(
    prepared: _PreparedSkillProbeArm,
) -> SkillProbeArmClosure:
    """Execute and close one arm whose complete preflight already succeeded."""

    expected_tail = prepared.opening_ledger_tail_ref
    previous_receipt_ref: ArtifactRef | None = None
    receipt_refs: list[ArtifactRef] = []
    for parent_cell, candidate_cell in prepared.pairs:
        dispositions: list[AttemptDisposition] = []
        for cell in (parent_cell, candidate_cell):
            record = execute_scheduled_attempt(
                runner=prepared.runner,
                schedule=prepared.schedule,
                preflight_ref=prepared.preflight_ref,
                expected_previous_ledger_tail_ref=expected_tail,
                previous_receipt_ref=previous_receipt_ref,
                cell=cell,
                attempt_index=0,
                task=prepared.tasks[cell.task_id],
                harness_ref=_harness_ref(prepared, cell),
            )
            outcome = _load_outcome(prepared, record.outcome_ref)
            receipt_refs.append(record.receipt_ref)
            expected_tail = record.outcome_ref
            previous_receipt_ref = record.receipt_ref
            dispositions.append(outcome.disposition)
            # A poisoned outcome has already charged the provider-reported
            # overrun and terminally poisons this arm's ledger.  Do not issue
            # the paired sibling request (or enter the other arm) after that
            # fact; global settlement below will preserve the exact tail.
            if outcome.disposition is AttemptDisposition.POISONED:
                raise _SkillProbeArmError(
                    "skill probe attempt exceeded its preregistered token ceiling"
                )
        if dispositions != [AttemptDisposition.SETTLED, AttemptDisposition.SETTLED]:
            raise _SkillProbeArmError(
                "matched probe pair did not settle on its preregistered first attempt"
            )
    usage = replay_trusted_usage(
        prepared.repository,
        schedule=prepared.schedule,
        preflight_ref=prepared.preflight_ref,
        attempt_ledger=prepared.ledger,
        receipt_refs=receipt_refs,
    )
    if usage.burned_attempts or usage.poisoned_attempts:
        raise _SkillProbeArmError("skill probe closure requires first-attempt settlement")
    inclusion_ref = publish_settled_skill_request_inclusion(
        prepared.repository,
        schedule=prepared.schedule,
        preflight_ref=prepared.preflight_ref,
        attempt_ledger=prepared.ledger,
        receipt_refs=usage.receipt_refs,
        candidate_harness_ref=prepared.candidate_harness_ref,
    )
    inclusion = verify_settled_skill_request_inclusion(
        prepared.repository,
        evidence_ref=inclusion_ref,
        schedule=prepared.schedule,
        preflight_ref=prepared.preflight_ref,
        attempt_ledger=prepared.ledger,
        candidate_harness_ref=prepared.candidate_harness_ref,
    )
    state = prepared.ledger.state()
    if state.tail_ref is None:
        raise _SkillProbeArmError("skill probe ledger has no closing tail")
    return SkillProbeArmClosure(
        control=prepared.control,
        schedule_fingerprint=prepared.schedule.fingerprint,
        preflight_ref=prepared.preflight_ref,
        opening_ledger_tail_ref=prepared.opening_ledger_tail_ref,
        closing_ledger_tail_ref=state.tail_ref,
        ledger_id=state.ledger_id,
        writer_epoch_id=state.writer_epoch_id,
        budget_fingerprint=state.budget.fingerprint,
        receipt_refs=usage.receipt_refs,
        usage=inclusion.usage,
        request_inclusion_ref=inclusion_ref,
    )


__all__: list[str] = []
