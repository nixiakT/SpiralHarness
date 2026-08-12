"""Independent replay of matched-probe attempt boundaries for global accounting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_plan import (
    SKILL_PROBE_ROSTER_MEDIA_TYPE,
    SkillMechanismPlan,
    SkillProbeRoster,
)

from .controller_artifacts import SkillProbeSettlementKind, SkillProbeUsageArmSettlement
from .skill_probe_closure import (
    MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
    MatchedSkillProbeClosure,
)


class SkillProbeUsageReplayError(RuntimeError):
    """Raised when a local probe ledger cannot authorize a global settlement."""


def _load_exact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    media_type: str,
    label: str,
) -> ModelT:
    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise SkillProbeUsageReplayError(f"{label} reference is malformed") from exc
    if checked_ref.media_type != media_type:
        raise SkillProbeUsageReplayError(f"{label} declares the wrong media type")
    try:
        payload = repository.get_bytes(checked_ref)
        loaded = repository.get_json(checked_ref, model_type)
        checked = model_type.model_validate(loaded, strict=True)
    except Exception as exc:
        raise SkillProbeUsageReplayError(f"{label} cannot be loaded exactly") from exc
    canonical = canonical_json_bytes(checked)
    if (
        payload != canonical
        or len(payload) != checked_ref.size
        or sha256_bytes(payload) != checked_ref.sha256
    ):
        raise SkillProbeUsageReplayError(f"{label} is not canonical under its reference")
    return checked


def replay_skill_probe_arm(
    repository: ArtifactRepository,
    *,
    control: Literal["revert", "placebo"],
    execution_nonce: str,
    schedule: EvaluationBatchSchedule,
    plan: SkillMechanismPlan,
    preflight_ref: ArtifactRef,
    terminal_tail_ref: ArtifactRef | None,
    terminal_kind: SkillProbeSettlementKind,
) -> SkillProbeUsageArmSettlement:
    """Recompute one arm's conservative charge from its immutable attempt chain."""

    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    preflight = _load_exact(
        repository,
        preflight_ref,
        SchedulePreflightCertificate,
        media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        label=f"{control} schedule preflight",
    )
    budget = AttemptBudget(
        max_attempts=checked_schedule.required_attempts,
        max_total_tokens=checked_schedule.required_tokens,
        max_tokens_per_attempt=checked_schedule.token_ceiling_per_attempt,
    )
    expected_ledger_id = f"skill-probe/{execution_nonce}/{control}"
    if not preflight.binds_schedule(checked_schedule):
        raise SkillProbeUsageReplayError(f"{control} preflight differs from the frozen schedule")
    if preflight.model_spec.fingerprint != plan.model_spec_fingerprint:
        raise SkillProbeUsageReplayError(f"{control} preflight uses another frozen model")
    if preflight.model_spec.runtime_fingerprint != plan.runtime_fingerprint:
        raise SkillProbeUsageReplayError(f"{control} preflight uses another frozen runtime")
    expected_preflight = (
        expected_ledger_id,
        budget.fingerprint,
        None,
        checked_schedule.required_attempts,
        checked_schedule.required_tokens,
    )
    actual_preflight = (
        preflight.ledger_id,
        preflight.budget_fingerprint,
        preflight.ledger_tail_ref,
        preflight.available_attempts,
        preflight.available_tokens,
    )
    if actual_preflight != expected_preflight or preflight.writer_epoch_id is None:
        raise SkillProbeUsageReplayError(
            f"{control} preflight is not the fresh full-capacity probe boundary"
        )

    checked_tail = (
        None
        if terminal_tail_ref is None
        else ArtifactRef.model_validate(terminal_tail_ref, strict=True)
    )
    try:
        audit_ledger = AttemptLedger(
            repository,
            ledger_id=expected_ledger_id,
            budget=budget,
            tail_ref=checked_tail,
        )
        state = audit_ledger.state()
    except Exception as exc:
        raise SkillProbeUsageReplayError(
            f"{control} terminal attempt boundary cannot be replayed"
        ) from exc

    if checked_tail is not None:
        if checked_tail.media_type == ATTEMPT_RESERVATION_MEDIA_TYPE:
            terminal = _load_exact(
                repository,
                checked_tail,
                AttemptReservation,
                media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
                label=f"{control} terminal reservation",
            )
        elif checked_tail.media_type == ATTEMPT_OUTCOME_MEDIA_TYPE:
            terminal = _load_exact(
                repository,
                checked_tail,
                AttemptOutcome,
                media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
                label=f"{control} terminal outcome",
            )
        else:
            raise SkillProbeUsageReplayError(f"{control} terminal tail has the wrong media type")
        if terminal.writer_epoch_id != preflight.writer_epoch_id:
            raise SkillProbeUsageReplayError(
                f"{control} terminal tail crosses the preflight writer epoch"
            )
    if state.attempts_used > checked_schedule.cell_count:
        raise SkillProbeUsageReplayError(f"{control} ledger exceeds first-attempt execution")
    if terminal_kind is SkillProbeSettlementKind.COMPLETED and (
        state.pending_reservation_ref is not None
        or state.poisoned
        or state.completed_attempts != checked_schedule.cell_count
    ):
        raise SkillProbeUsageReplayError(f"{control} completed settlement is not fully settled")
    if terminal_kind is SkillProbeSettlementKind.COMPLETED:
        cursor = checked_tail
        while cursor is not None:
            outcome = _load_exact(
                repository,
                cursor,
                AttemptOutcome,
                media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
                label=f"{control} completed outcome",
            )
            if outcome.disposition is not AttemptDisposition.SETTLED:
                raise SkillProbeUsageReplayError(
                    f"{control} completed settlement contains a non-settled attempt"
                )
            reservation = _load_exact(
                repository,
                outcome.reservation_ref,
                AttemptReservation,
                media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
                label=f"{control} completed reservation",
            )
            cursor = reservation.previous_outcome_ref
    return SkillProbeUsageArmSettlement(
        control=control,
        preflight_ref=preflight_ref,
        terminal_tail_ref=checked_tail,
        encumbered_tokens=state.encumbered_tokens,
        poisoned=state.poisoned,
    )


def replay_skill_probe_settlement(
    repository: ArtifactRepository,
    *,
    authorization_ref: ArtifactRef,
    execution_nonce: str,
    running_probes_tail_ref: ArtifactRef,
    plan: SkillMechanismPlan,
    revert_preflight_ref: ArtifactRef,
    revert_terminal_tail_ref: ArtifactRef | None,
    placebo_preflight_ref: ArtifactRef,
    placebo_terminal_tail_ref: ArtifactRef | None,
    terminal_kind: SkillProbeSettlementKind,
    closure_ref: ArtifactRef | None,
) -> tuple[SkillProbeUsageArmSettlement, SkillProbeUsageArmSettlement]:
    """Replay both independent arms under their preregistered schedules."""

    checked_plan = SkillMechanismPlan.model_validate(plan, strict=True)
    checked_kind = SkillProbeSettlementKind(terminal_kind)
    revert = replay_skill_probe_arm(
        repository,
        control="revert",
        execution_nonce=execution_nonce,
        schedule=checked_plan.revert_schedule,
        plan=checked_plan,
        preflight_ref=revert_preflight_ref,
        terminal_tail_ref=revert_terminal_tail_ref,
        terminal_kind=checked_kind,
    )
    placebo = replay_skill_probe_arm(
        repository,
        control="placebo",
        execution_nonce=execution_nonce,
        schedule=checked_plan.placebo_schedule,
        plan=checked_plan,
        preflight_ref=placebo_preflight_ref,
        terminal_tail_ref=placebo_terminal_tail_ref,
        terminal_kind=checked_kind,
    )
    if revert.preflight_ref == placebo.preflight_ref:
        raise SkillProbeUsageReplayError("matched probe arms reuse one preflight")
    if closure_ref is not None:
        _verify_settlement_closure(
            repository,
            closure_ref=closure_ref,
            authorization_ref=authorization_ref,
            execution_nonce=execution_nonce,
            running_probes_tail_ref=running_probes_tail_ref,
            plan=checked_plan,
            revert=revert,
            placebo=placebo,
        )
    elif checked_kind is SkillProbeSettlementKind.COMPLETED:
        raise SkillProbeUsageReplayError("completed settlement requires its exact closure")
    return revert, placebo


def _verify_settlement_closure(
    repository: ArtifactRepository,
    *,
    closure_ref: ArtifactRef,
    authorization_ref: ArtifactRef,
    execution_nonce: str,
    running_probes_tail_ref: ArtifactRef,
    plan: SkillMechanismPlan,
    revert: SkillProbeUsageArmSettlement,
    placebo: SkillProbeUsageArmSettlement,
) -> None:
    closure = _load_exact(
        repository,
        closure_ref,
        MatchedSkillProbeClosure,
        media_type=MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
        label="matched skill probe closure",
    )
    roster = _load_exact(
        repository,
        plan.probe_roster_ref,
        SkillProbeRoster,
        media_type=SKILL_PROBE_ROSTER_MEDIA_TYPE,
        label="skill probe roster",
    )
    expected_context = (
        authorization_ref,
        execution_nonce,
        plan.experiment_ref,
        plan.protocol_ref,
        plan.candidate_ref,
        plan.artifact_ref,
        running_probes_tail_ref,
        plan.candidate_harness_ref,
        plan.probe_roster_ref,
        roster.task_refs,
        plan.model_spec_fingerprint,
        plan.runtime_fingerprint,
        plan.reset_fingerprint,
        plan.execution_order_fingerprint,
    )
    actual_context = (
        closure.authorization_ref,
        closure.execution_nonce,
        closure.experiment_ref,
        closure.protocol_ref,
        closure.candidate_ref,
        closure.plan_ref,
        closure.running_probes_tail_ref,
        closure.candidate_harness_ref,
        closure.probe_roster_ref,
        closure.task_refs,
        closure.model_spec_fingerprint,
        closure.runtime_fingerprint,
        closure.reset_fingerprint,
        closure.execution_order_fingerprint,
    )
    if actual_context != expected_context:
        raise SkillProbeUsageReplayError(
            "matched skill probe closure differs from the reservation context"
        )
    for control, claimed, settled, schedule in (
        ("revert", closure.revert, revert, plan.revert_schedule),
        ("placebo", closure.placebo, placebo, plan.placebo_schedule),
    ):
        if settled.terminal_tail_ref is None:
            raise SkillProbeUsageReplayError(
                f"{control} closure cannot bind an empty terminal boundary"
            )
        preflight = _load_exact(
            repository,
            settled.preflight_ref,
            SchedulePreflightCertificate,
            media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
            label=f"{control} closure preflight",
        )
        terminal = _load_exact(
            repository,
            settled.terminal_tail_ref,
            AttemptOutcome,
            media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
            label=f"{control} closure terminal outcome",
        )
        budget = AttemptBudget(
            max_attempts=schedule.required_attempts,
            max_total_tokens=schedule.required_tokens,
            max_tokens_per_attempt=schedule.token_ceiling_per_attempt,
        )
        expected_arm = (
            control,
            schedule.fingerprint,
            settled.preflight_ref,
            None,
            settled.terminal_tail_ref,
            f"skill-probe/{execution_nonce}/{control}",
            preflight.writer_epoch_id,
            budget.fingerprint,
            settled.encumbered_tokens,
        )
        actual_arm = (
            claimed.control,
            claimed.schedule_fingerprint,
            claimed.preflight_ref,
            claimed.opening_ledger_tail_ref,
            claimed.closing_ledger_tail_ref,
            claimed.ledger_id,
            claimed.writer_epoch_id,
            claimed.budget_fingerprint,
            claimed.usage.charged_tokens,
        )
        if actual_arm != expected_arm or terminal.writer_epoch_id != preflight.writer_epoch_id:
            raise SkillProbeUsageReplayError(
                f"{control} closure differs from its replayed attempt boundary"
            )


__all__ = [
    "SkillProbeUsageReplayError",
    "replay_skill_probe_arm",
    "replay_skill_probe_settlement",
]
