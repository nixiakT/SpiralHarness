"""Execute plan-derived baseline GATE batches with trusted receipts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from spiral_harness.benchmark.base import BenchmarkAdapter
from spiral_harness.benchmark.runner import BenchmarkBatchExecution, TrustedBenchmarkBatchRunner
from spiral_harness.core.experiment import ProtocolManifest
from spiral_harness.core.models import ArtifactRef, ComponentKind
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import AttemptBudget
from spiral_harness.execution.model import FixedModelRunner
from spiral_harness.execution.schedule import EvaluationBatchSchedule
from spiral_harness.experiments.baseline_execution import (
    BaselineGateBatchUsage,
    derive_baseline_gate_schedules,
    summarize_baseline_gate_usage,
)
from spiral_harness.experiments.baseline_gate_closure import (
    BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE,
    BaselineGateClosureError,
    BaselineGateStudyClosure,
    build_baseline_gate_study_closure,
    publish_baseline_gate_study_closure,
)
from spiral_harness.experiments.baselines import (
    LEGACY_BASELINE_KINDS,
    BaselineKind,
    BaselineProtocolConsistencyReport,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    BaselineUsageReport,
    FeedbackType,
    ResourceUsage,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import TrustedGateBatchService


class BaselineGateRunnerError(RuntimeError):
    """Raised when a trusted baseline GATE run cannot be closed exactly."""


@dataclass(frozen=True, slots=True)
class BaselineGateExecution:
    """One condition's complete plan-derived GATE execution."""

    kind: BaselineKind
    report: BaselineUsageReport
    report_ref: ArtifactRef
    schedules: tuple[EvaluationBatchSchedule, ...]
    batches: tuple[BenchmarkBatchExecution, ...]


@dataclass(frozen=True, slots=True)
class BaselineGateStudyExecution:
    """All four baseline GATE executions plus structural validation."""

    executions: tuple[BaselineGateExecution, ...]
    consistency: BaselineProtocolConsistencyReport
    closure: BaselineGateStudyClosure
    closure_ref: ArtifactRef


RunnerFactory = Callable[[EvaluationBatchSchedule], tuple[FixedModelRunner, AttemptLedger]]


class TrustedBaselineGateRunner[TaskT]:
    """Run the receipt-backed GATE slice for frozen baseline conditions."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        adapter: BenchmarkAdapter[str, TaskT, object],
        gate_batch_service: TrustedGateBatchService,
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        if not isinstance(adapter, BenchmarkAdapter):
            raise TypeError("adapter must implement BenchmarkAdapter")
        if type(gate_batch_service) is not TrustedGateBatchService:
            raise TypeError("gate_batch_service must be an exact TrustedGateBatchService")
        self._repository = repository
        self._batch_runner = TrustedBenchmarkBatchRunner(
            repository,
            adapter=adapter,
            gate_batch_service=gate_batch_service,
        )

    def execute_condition(
        self,
        plan: BaselineStudyPlan | object,
        *,
        kind: BaselineKind,
        query: int,
        protocol_ref: ArtifactRef,
        protocol: ProtocolManifest,
        candidate_ref: ArtifactRef,
        parent_harness_ref: ArtifactRef,
        candidate_harness_ref: ArtifactRef,
        gate_split_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
        task_ids: tuple[str, ...],
        token_ceiling_per_attempt: int,
        runner_factory: RunnerFactory,
        additional_usage: ResourceUsage | None = None,
        feedback_used: tuple[FeedbackType, ...] = (FeedbackType.BENCHMARK_METADATA,),
        mutated_component_kinds: tuple[ComponentKind, ...] = (),
        source_refs: tuple[ArtifactRef, ...] = (),
    ) -> BaselineGateExecution:
        """Execute every frozen search-run seed for one baseline condition."""

        checked_plan = BaselineProtocolValidator.validate_plan(plan)
        checked_kind = _require_kind(kind)
        schedules = derive_baseline_gate_schedules(
            checked_plan,
            kind=checked_kind,
            query=query,
            parent_harness_ref=parent_harness_ref,
            candidate_harness_ref=candidate_harness_ref,
            task_ids=task_ids,
            token_ceiling_per_attempt=token_ceiling_per_attempt,
        )
        batches = tuple(
            self._execute_schedule(
                protocol_ref=protocol_ref,
                protocol=protocol,
                candidate_ref=candidate_ref,
                schedule=schedule,
                parent_harness_ref=parent_harness_ref,
                candidate_harness_ref=candidate_harness_ref,
                gate_split_ref=gate_split_ref,
                mechanism_evidence_ref=mechanism_evidence_ref,
                runner_factory=runner_factory,
                source_refs=source_refs,
            )
            for schedule in schedules
        )
        report = summarize_baseline_gate_usage(
            checked_plan,
            kind=checked_kind,
            batches=tuple(
                BaselineGateBatchUsage(schedule=schedule, usage=batch.usage)
                for schedule, batch in zip(schedules, batches, strict=True)
            ),
            additional_usage=additional_usage,
            feedback_used=feedback_used,
            mutated_component_kinds=mutated_component_kinds,
        )
        report_ref = _publish_usage_report(self._repository, report)
        return BaselineGateExecution(
            kind=checked_kind,
            report=report,
            report_ref=report_ref,
            schedules=schedules,
            batches=batches,
        )

    def execute_study(
        self,
        plan: BaselineStudyPlan | object,
        *,
        query: int,
        protocol_ref: ArtifactRef,
        protocol: ProtocolManifest,
        candidate_refs: dict[BaselineKind, ArtifactRef],
        parent_harness_ref: ArtifactRef,
        candidate_harness_refs: dict[BaselineKind, ArtifactRef],
        gate_split_ref: ArtifactRef,
        mechanism_evidence_refs: dict[BaselineKind, ArtifactRef],
        task_ids: tuple[str, ...],
        token_ceiling_per_attempt: int,
        runner_factories: dict[BaselineKind, RunnerFactory],
        additional_usage: dict[BaselineKind, ResourceUsage] | None = None,
        feedback_used: dict[BaselineKind, tuple[FeedbackType, ...]] | None = None,
        mutated_component_kinds: dict[BaselineKind, tuple[ComponentKind, ...]] | None = None,
        source_refs: dict[BaselineKind, tuple[ArtifactRef, ...]] | None = None,
    ) -> BaselineGateStudyExecution:
        """Execute all four conditions and validate the resulting reports together."""

        checked_plan = BaselineProtocolValidator.validate_plan(plan)
        executions = tuple(
            self.execute_condition(
                checked_plan,
                kind=kind,
                query=query,
                protocol_ref=protocol_ref,
                protocol=protocol,
                candidate_ref=_require_mapping_value(candidate_refs, kind, "candidate_refs"),
                parent_harness_ref=parent_harness_ref,
                candidate_harness_ref=_require_mapping_value(
                    candidate_harness_refs,
                    kind,
                    "candidate_harness_refs",
                ),
                gate_split_ref=gate_split_ref,
                mechanism_evidence_ref=_require_mapping_value(
                    mechanism_evidence_refs,
                    kind,
                    "mechanism_evidence_refs",
                ),
                task_ids=task_ids,
                token_ceiling_per_attempt=token_ceiling_per_attempt,
                runner_factory=_require_runner_factory(runner_factories, kind),
                additional_usage=_optional_mapping_value(additional_usage, kind),
                feedback_used=_optional_mapping_value(
                    feedback_used,
                    kind,
                    default=(FeedbackType.BENCHMARK_METADATA,),
                ),
                mutated_component_kinds=_optional_mapping_value(
                    mutated_component_kinds,
                    kind,
                    default=(),
                ),
                source_refs=_optional_mapping_value(source_refs, kind, default=()),
            )
            for kind in LEGACY_BASELINE_KINDS
        )
        consistency = BaselineProtocolValidator.validate_usage(
            checked_plan,
            tuple(execution.report for execution in executions),
        )
        try:
            closure = build_baseline_gate_study_closure(
                plan=checked_plan,
                query=query,
                protocol_ref=protocol_ref,
                gate_split_ref=gate_split_ref,
                parent_harness_ref=parent_harness_ref,
                task_ids=task_ids,
                token_ceiling_per_attempt=token_ceiling_per_attempt,
                executions=executions,
                consistency=consistency,
            )
            closure_ref = publish_baseline_gate_study_closure(
                self._repository,
                closure,
                plan=checked_plan,
            )
        except BaselineGateClosureError as exc:
            raise BaselineGateRunnerError(str(exc)) from exc
        return BaselineGateStudyExecution(
            executions=executions,
            consistency=consistency,
            closure=closure,
            closure_ref=closure_ref,
        )

    def _execute_schedule(
        self,
        *,
        protocol_ref: ArtifactRef,
        protocol: ProtocolManifest,
        candidate_ref: ArtifactRef,
        schedule: EvaluationBatchSchedule,
        parent_harness_ref: ArtifactRef,
        candidate_harness_ref: ArtifactRef,
        gate_split_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
        runner_factory: RunnerFactory,
        source_refs: tuple[ArtifactRef, ...],
    ) -> BenchmarkBatchExecution:
        if not callable(runner_factory):
            raise TypeError("runner_factory must be callable")
        runner, attempt_ledger = runner_factory(schedule)
        if type(runner) is not FixedModelRunner:
            raise TypeError("runner_factory must return a FixedModelRunner")
        if type(attempt_ledger) is not AttemptLedger:
            raise TypeError("runner_factory must return an AttemptLedger")
        if runner.repository is not self._repository:
            raise BaselineGateRunnerError("runner_factory returned a foreign runner")
        if attempt_ledger.repository is not self._repository:
            raise BaselineGateRunnerError("runner_factory returned a foreign attempt ledger")
        if runner.attempt_state().ledger_id != attempt_ledger.ledger_id:
            raise BaselineGateRunnerError("runner_factory returned mismatched runner and ledger")
        _require_fresh_ledger(schedule, attempt_ledger)
        return self._batch_runner.execute_paired_batch(
            protocol_ref=protocol_ref,
            protocol=protocol,
            candidate_ref=candidate_ref,
            schedule=schedule,
            parent_harness_ref=parent_harness_ref,
            candidate_harness_ref=candidate_harness_ref,
            gate_split_ref=gate_split_ref,
            mechanism_evidence_ref=mechanism_evidence_ref,
            runner=runner,
            attempt_ledger=attempt_ledger,
            source_refs=source_refs,
        )


def baseline_gate_attempt_budget(schedule: EvaluationBatchSchedule) -> AttemptBudget:
    """Return the minimal exact ledger budget for one GATE batch."""

    checked = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    return AttemptBudget(
        max_attempts=checked.required_attempts,
        max_total_tokens=checked.required_tokens,
        max_tokens_per_attempt=checked.token_ceiling_per_attempt,
    )


def _publish_usage_report(
    repository: ArtifactRepository,
    report: BaselineUsageReport,
) -> ArtifactRef:
    checked = BaselineUsageReport.model_validate(report, strict=True)
    ref = repository.put_json(checked, media_type=BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE)
    loaded = repository.get_json(ref, BaselineUsageReport)
    if BaselineUsageReport.model_validate(loaded, strict=True) != checked:
        raise BaselineGateRunnerError("published baseline usage report changed content")
    return ArtifactRef.model_validate(ref, strict=True)


def _require_kind(kind: BaselineKind) -> BaselineKind:
    if type(kind) is not BaselineKind:
        raise TypeError("kind must be an exact BaselineKind")
    if kind not in LEGACY_BASELINE_KINDS:
        raise BaselineGateRunnerError("legacy gate runner cannot execute protocol-v2 SCORE")
    return kind


def _require_fresh_ledger(
    schedule: EvaluationBatchSchedule,
    attempt_ledger: AttemptLedger,
) -> None:
    checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
    state = attempt_ledger.state()
    if state.tail_ref is not None or state.pending_reservation_ref is not None:
        raise BaselineGateRunnerError("baseline GATE batches require a fresh attempt ledger")
    if state.attempts_used or state.completed_attempts or state.charged_tokens:
        raise BaselineGateRunnerError("baseline GATE batches require an unused attempt ledger")
    budget = state.budget
    expected = baseline_gate_attempt_budget(checked_schedule)
    if budget != expected:
        raise BaselineGateRunnerError(
            "baseline GATE ledger budget must equal the schedule's exact worst-case budget"
        )


def _require_mapping_value[ValueT](
    values: dict[BaselineKind, ValueT],
    kind: BaselineKind,
    field_name: str,
) -> ValueT:
    checked_kind = _require_kind(kind)
    _require_exact_legacy_mapping(values, field_name=field_name)
    return values[checked_kind]


def _require_exact_legacy_mapping[ValueT](
    values: dict[BaselineKind, ValueT],
    *,
    field_name: str,
) -> None:
    if type(values) is not dict:
        raise TypeError(f"{field_name} must be a dict keyed by BaselineKind")
    if any(type(key) is not BaselineKind for key in values):
        raise TypeError(f"{field_name} keys must be exact BaselineKind values")
    supplied = frozenset(values)
    expected = frozenset(LEGACY_BASELINE_KINDS)
    if supplied != expected or len(values) != len(LEGACY_BASELINE_KINDS):
        missing = sorted(kind.value for kind in expected.difference(supplied))
        unexpected = sorted(kind.value for kind in supplied.difference(expected))
        raise BaselineGateRunnerError(
            f"{field_name} must contain exactly the legacy baseline keys; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _optional_mapping_value[ValueT](
    values: dict[BaselineKind, ValueT] | None,
    kind: BaselineKind,
    default: ValueT | None = None,
) -> ValueT | None:
    if values is None:
        return default
    return _require_mapping_value(values, kind, "baseline option mapping")


def _require_runner_factory(
    values: dict[BaselineKind, RunnerFactory],
    kind: BaselineKind,
) -> RunnerFactory:
    factory = _require_mapping_value(values, kind, "runner_factories")
    if not callable(factory):
        raise TypeError("runner_factories values must be callable")
    return factory


__all__ = [
    "BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE",
    "BaselineGateExecution",
    "BaselineGateRunnerError",
    "BaselineGateStudyExecution",
    "RunnerFactory",
    "TrustedBaselineGateRunner",
    "baseline_gate_attempt_budget",
]
