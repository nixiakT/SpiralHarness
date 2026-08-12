"""Bind four-condition baseline plans to trusted benchmark execution batches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE, ArtifactRef, ComponentKind
from spiral_harness.execution.receipts import TrustedExecutionUsage
from spiral_harness.execution.schedule import EvaluationBatchSchedule, EvaluationPhase
from spiral_harness.experiments.baselines import (
    BaselineArmPlan,
    BaselineKind,
    BaselineProtocolError,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    BaselineUsageReport,
    FeedbackType,
    ResourceUsage,
)

_MASTER_SEED_SCHEMA = "spiral-harness/baseline-gate-schedule-master-seed/v1"
_SEED_MASK = (1 << 63) - 1


class BaselineExecutionBindingError(BaselineProtocolError):
    """Raised when a baseline plan cannot bind to trusted batch execution."""


@dataclass(frozen=True, slots=True)
class BaselineGateBatchUsage:
    """Receipt-backed usage for one plan-derived GATE schedule."""

    schedule: EvaluationBatchSchedule
    usage: TrustedExecutionUsage


def baseline_gate_study_label(plan: BaselineStudyPlan | object) -> str:
    """Return the canonical schedule study coordinate for a baseline plan."""

    checked_plan = BaselineProtocolValidator.validate_plan(plan)
    return f"baseline-gate:{checked_plan.fingerprint}"


def derive_baseline_gate_schedule(
    plan: BaselineStudyPlan | object,
    *,
    kind: BaselineKind,
    search_run_seed: int,
    query: int,
    parent_harness_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    task_ids: tuple[str, ...],
    token_ceiling_per_attempt: int,
) -> EvaluationBatchSchedule:
    """Derive the only GATE schedule shape allowed for one baseline run."""

    checked_plan = BaselineProtocolValidator.validate_plan(plan)
    checked_kind = _require_baseline_kind(kind)
    arm = checked_plan.arm(checked_kind)
    _require_search_run_seed(arm, search_run_seed)
    parent_ref = _require_harness_ref(parent_harness_ref, field_name="parent_harness_ref")
    candidate_ref = _require_harness_ref(candidate_harness_ref, field_name="candidate_harness_ref")

    schedule = EvaluationBatchSchedule(
        study=baseline_gate_study_label(checked_plan),
        kind=checked_kind.value,
        phase=EvaluationPhase.GATE,
        query=query,
        master_seed=_derive_master_seed(
            checked_plan,
            kind=checked_kind,
            search_run_seed=search_run_seed,
            query=query,
        ),
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=candidate_ref.sha256,
        task_ids=task_ids,
        search_runs=(search_run_seed,),
        repeat_seeds=arm.evaluation.repeat_seeds,
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=token_ceiling_per_attempt,
    )
    _require_plan_can_cover_full_gate_axis(
        arm,
        schedule,
        all_search_run_count=len(arm.evaluation.search_run_seeds),
    )
    return schedule


def derive_baseline_gate_schedules(
    plan: BaselineStudyPlan | object,
    *,
    kind: BaselineKind,
    query: int,
    parent_harness_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    task_ids: tuple[str, ...],
    token_ceiling_per_attempt: int,
) -> tuple[EvaluationBatchSchedule, ...]:
    """Derive one complete schedule per frozen independent search-run seed."""

    checked_plan = BaselineProtocolValidator.validate_plan(plan)
    checked_kind = _require_baseline_kind(kind)
    arm = checked_plan.arm(checked_kind)
    return tuple(
        derive_baseline_gate_schedule(
            checked_plan,
            kind=checked_kind,
            search_run_seed=search_run_seed,
            query=query,
            parent_harness_ref=parent_harness_ref,
            candidate_harness_ref=candidate_harness_ref,
            task_ids=task_ids,
            token_ceiling_per_attempt=token_ceiling_per_attempt,
        )
        for search_run_seed in arm.evaluation.search_run_seeds
    )


def summarize_baseline_gate_usage(
    plan: BaselineStudyPlan | object,
    *,
    kind: BaselineKind,
    batches: tuple[BaselineGateBatchUsage, ...],
    additional_usage: ResourceUsage | None = None,
    feedback_used: tuple[FeedbackType, ...] = (FeedbackType.BENCHMARK_METADATA,),
    mutated_component_kinds: tuple[ComponentKind, ...] = (),
) -> BaselineUsageReport:
    """Build a baseline usage report from complete receipt-backed GATE batches."""

    checked_plan = BaselineProtocolValidator.validate_plan(plan)
    checked_kind = _require_baseline_kind(kind)
    arm = checked_plan.arm(checked_kind)
    checked_batches = _require_complete_gate_batches(checked_plan, arm, batches)
    checked_additional_usage = (
        ResourceUsage()
        if additional_usage is None
        else ResourceUsage.model_validate(additional_usage, strict=True)
    )
    execution_evaluations = sum(batch.usage.cell_count for batch in checked_batches)
    execution_tokens = sum(batch.usage.tokens for batch in checked_batches)
    used = _combine_usage(
        checked_additional_usage,
        evaluations=execution_evaluations,
        tokens=execution_tokens,
    )
    report = BaselineUsageReport(
        plan_fingerprint=checked_plan.fingerprint,
        kind=checked_kind,
        available=arm.ceilings,
        used=used,
        executed_search_run_seeds=arm.evaluation.search_run_seeds,
        executed_repeat_seeds=arm.evaluation.repeat_seeds,
        feedback_used=feedback_used,
        mutated_component_kinds=mutated_component_kinds,
    )
    _require_report_within_arm(arm, report)
    return report


def _derive_master_seed(
    plan: BaselineStudyPlan,
    *,
    kind: BaselineKind,
    search_run_seed: int,
    query: int,
) -> int:
    digest = canonical_sha256(
        {
            "schema": _MASTER_SEED_SCHEMA,
            "plan_fingerprint": plan.fingerprint,
            "baseline_kind": kind.value,
            "search_run_seed": search_run_seed,
            "query": query,
        }
    )
    return int(digest[:16], 16) & _SEED_MASK


def _require_baseline_kind(kind: BaselineKind) -> BaselineKind:
    if not isinstance(kind, BaselineKind):
        raise TypeError("kind must be a BaselineKind")
    return kind


def _require_search_run_seed(arm: BaselineArmPlan, search_run_seed: int) -> None:
    if type(search_run_seed) is not int:
        raise TypeError("search_run_seed must be an integer")
    if search_run_seed not in arm.evaluation.search_run_seeds:
        raise BaselineExecutionBindingError(
            f"{arm.kind.value} search_run_seed is not frozen into the baseline plan"
        )


def _require_harness_ref(ref: ArtifactRef, *, field_name: str) -> ArtifactRef:
    checked = ArtifactRef.model_validate(ref, strict=True)
    if checked.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
        raise BaselineExecutionBindingError(f"{field_name} must be a harness manifest")
    return checked


def _require_plan_can_cover_full_gate_axis(
    arm: BaselineArmPlan,
    schedule: EvaluationBatchSchedule,
    *,
    all_search_run_count: int,
) -> None:
    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    required_evaluations = checked_schedule.cell_count * all_search_run_count
    required_tokens = required_evaluations * checked_schedule.token_ceiling_per_attempt
    if required_evaluations > arm.ceilings.max_evaluations:
        raise BaselineExecutionBindingError(
            f"{arm.kind.value} gate schedule exceeds max_evaluations: "
            f"required={required_evaluations}, available={arm.ceilings.max_evaluations}"
        )
    if required_tokens > arm.ceilings.max_tokens:
        raise BaselineExecutionBindingError(
            f"{arm.kind.value} gate schedule exceeds max_tokens: "
            f"required={required_tokens}, available={arm.ceilings.max_tokens}"
        )


def _require_complete_gate_batches(
    plan: BaselineStudyPlan,
    arm: BaselineArmPlan,
    batches: tuple[BaselineGateBatchUsage, ...],
) -> tuple[BaselineGateBatchUsage, ...]:
    if not batches:
        raise BaselineExecutionBindingError("baseline gate usage requires at least one batch")
    checked: list[BaselineGateBatchUsage] = []
    for batch in batches:
        if type(batch) is not BaselineGateBatchUsage:
            raise TypeError("batches must contain BaselineGateBatchUsage values")
        schedule = _require_gate_schedule(plan, arm, batch.schedule)
        usage = TrustedExecutionUsage.model_validate(batch.usage, strict=True)
        if usage.schedule_fingerprint != schedule.fingerprint:
            raise BaselineExecutionBindingError("gate usage belongs to another schedule")
        if usage.cell_count != schedule.cell_count or usage.attempt_count != schedule.cell_count:
            raise BaselineExecutionBindingError("gate usage does not cover its schedule cells")
        if usage.settled_attempts != schedule.cell_count or usage.burned_attempts:
            raise BaselineExecutionBindingError("gate usage must contain only settled attempts")
        if usage.poisoned_attempts:
            raise BaselineExecutionBindingError("poisoned gate usage cannot enter a baseline")
        checked.append(BaselineGateBatchUsage(schedule=schedule, usage=usage))

    ordered = tuple(sorted(checked, key=lambda item: item.schedule.search_runs[0]))
    covered = tuple(item.schedule.search_runs[0] for item in ordered)
    if covered != arm.evaluation.search_run_seeds:
        raise BaselineExecutionBindingError("gate batches must cover every frozen search run once")
    _require_consistent_schedule_axes(ordered)
    return ordered


def _require_gate_schedule(
    plan: BaselineStudyPlan,
    arm: BaselineArmPlan,
    schedule: EvaluationBatchSchedule,
) -> EvaluationBatchSchedule:
    checked = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    if checked.study != baseline_gate_study_label(plan):
        raise BaselineExecutionBindingError("gate schedule belongs to another baseline plan")
    if checked.kind != arm.kind.value:
        raise BaselineExecutionBindingError("gate schedule belongs to another baseline kind")
    if checked.phase is not EvaluationPhase.GATE:
        raise BaselineExecutionBindingError("baseline execution requires GATE schedules")
    if (
        len(checked.search_runs) != 1
        or checked.search_runs[0] not in arm.evaluation.search_run_seeds
    ):
        raise BaselineExecutionBindingError("gate schedule search_run_seed is not frozen")
    if checked.repeat_seeds != arm.evaluation.repeat_seeds:
        raise BaselineExecutionBindingError("gate schedule repeat seeds differ from the plan")
    if checked.max_attempts_per_cell != 1:
        raise BaselineExecutionBindingError("trusted fixed benchmark batches allow one attempt")
    _require_plan_can_cover_full_gate_axis(
        arm,
        checked,
        all_search_run_count=len(arm.evaluation.search_run_seeds),
    )
    return checked


def _require_consistent_schedule_axes(batches: Iterable[BaselineGateBatchUsage]) -> None:
    iterator = iter(batches)
    anchor = next(iterator).schedule
    for batch in iterator:
        schedule = batch.schedule
        if (
            schedule.query != anchor.query
            or schedule.task_ids != anchor.task_ids
            or schedule.parent_harness_id != anchor.parent_harness_id
            or schedule.candidate_harness_id != anchor.candidate_harness_id
            or schedule.token_ceiling_per_attempt != anchor.token_ceiling_per_attempt
        ):
            raise BaselineExecutionBindingError(
                "gate batches must share query, task, harness, and token axes"
            )


def _combine_usage(
    additional: ResourceUsage,
    *,
    evaluations: int,
    tokens: int,
) -> ResourceUsage:
    return ResourceUsage(
        evaluations=additional.evaluations + evaluations,
        feedback_queries=additional.feedback_queries,
        proposals=additional.proposals,
        optimizer_model_calls=additional.optimizer_model_calls,
        tokens=additional.tokens + tokens,
        wall_time_seconds=additional.wall_time_seconds,
        cost_usd=additional.cost_usd,
    )


def _require_report_within_arm(
    arm: BaselineArmPlan,
    report: BaselineUsageReport,
) -> None:
    checks = (
        ("evaluations", report.used.evaluations, arm.ceilings.max_evaluations),
        ("feedback_queries", report.used.feedback_queries, arm.ceilings.max_feedback_queries),
        ("proposals", report.used.proposals, arm.ceilings.max_proposals),
        (
            "optimizer_model_calls",
            report.used.optimizer_model_calls,
            arm.ceilings.max_optimizer_model_calls,
        ),
        ("tokens", report.used.tokens, arm.ceilings.max_tokens),
        ("wall_time_seconds", report.used.wall_time_seconds, arm.ceilings.max_wall_time_seconds),
        ("cost_usd", report.used.cost_usd, arm.ceilings.max_cost_usd),
    )
    for field_name, used, available in checks:
        if used > available:
            raise BaselineExecutionBindingError(
                f"{arm.kind.value} gate usage exceeds {field_name}: "
                f"used={used}, available={available}"
            )
    if not frozenset(report.feedback_used).issubset(arm.available_feedback):
        raise BaselineExecutionBindingError("gate usage reports forbidden feedback")
    if not frozenset(report.mutated_component_kinds).issubset(arm.mutation.mutable_component_kinds):
        raise BaselineExecutionBindingError("gate usage reports forbidden mutations")
    if report.used.proposals == 0 and report.mutated_component_kinds:
        raise BaselineExecutionBindingError("gate usage reports mutations without proposals")
    if report.used.proposals > 0 and not report.mutated_component_kinds:
        raise BaselineExecutionBindingError("gate usage reports proposals without mutations")
    if not arm.mutation.may_call_optimizer_model and report.used.optimizer_model_calls:
        raise BaselineExecutionBindingError("gate usage reports forbidden optimizer calls")
    if arm.kind is BaselineKind.STATIC:
        search_use = (
            report.used.feedback_queries,
            report.used.proposals,
            report.used.optimizer_model_calls,
        )
        if any(search_use) or report.mutated_component_kinds:
            raise BaselineExecutionBindingError("static gate usage cannot claim search work")


__all__ = [
    "BaselineExecutionBindingError",
    "BaselineGateBatchUsage",
    "baseline_gate_study_label",
    "derive_baseline_gate_schedule",
    "derive_baseline_gate_schedules",
    "summarize_baseline_gate_usage",
]
