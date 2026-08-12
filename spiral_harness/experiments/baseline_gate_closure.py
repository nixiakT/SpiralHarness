"""Durable structural closure for a four-condition baseline GATE study."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from spiral_harness.benchmark.runner import BenchmarkBatchExecution
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    ModelExecution,
)
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    TrustedExecutionUsage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    SchedulePreflightCertificate,
)
from spiral_harness.experiments.baselines import (
    REQUIRED_BASELINES,
    BaselineKind,
    BaselineProtocolConsistencyReport,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    BaselineUsageReport,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateTrialArm,
    GateTrialBatch,
)

BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.baseline-gate-usage-report.v1+json"
)
BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.baseline-gate-study-closure.v1+json"
)


class BaselineGateClosureError(RuntimeError):
    """Raised when a trusted baseline GATE run cannot be closed exactly."""


class BaselineGateBatchClosure(ImmutableModel):
    """Replayable structural closure for one trusted benchmark batch."""

    schema_version: Literal["1"] = "1"
    schedule: EvaluationBatchSchedule
    candidate_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef
    preflight_ref: ArtifactRef
    parent_batch_ref: ArtifactRef
    candidate_batch_ref: ArtifactRef
    usage: TrustedExecutionUsage
    receipt_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    execution_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    outcome_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def exact_batch_refs_and_usage(self) -> BaselineGateBatchClosure:
        exact_refs = (
            ("parent_harness_ref", self.parent_harness_ref, HARNESS_MANIFEST_MEDIA_TYPE),
            ("candidate_harness_ref", self.candidate_harness_ref, HARNESS_MANIFEST_MEDIA_TYPE),
            ("preflight_ref", self.preflight_ref, SCHEDULE_PREFLIGHT_MEDIA_TYPE),
            ("parent_batch_ref", self.parent_batch_ref, GATE_TRIAL_BATCH_MEDIA_TYPE),
            ("candidate_batch_ref", self.candidate_batch_ref, GATE_TRIAL_BATCH_MEDIA_TYPE),
        )
        for field_name, ref, expected in exact_refs:
            if ref.media_type != expected:
                raise ValueError(f"{field_name} declares the wrong media type")
        _require_json_ref(self.candidate_ref, field_name="candidate_ref")
        _require_json_ref(self.mechanism_evidence_ref, field_name="mechanism_evidence_ref")
        if self.schedule.phase is not EvaluationPhase.GATE:
            raise ValueError("baseline gate batch closure requires a GATE schedule")
        if len(self.schedule.search_runs) != 1:
            raise ValueError("baseline gate batch closure binds exactly one search run")
        if self.schedule.parent_harness_id != self.parent_harness_ref.sha256:
            raise ValueError("parent_harness_ref differs from the frozen schedule")
        if self.schedule.candidate_harness_id != self.candidate_harness_ref.sha256:
            raise ValueError("candidate_harness_ref differs from the frozen schedule")
        if self.usage.schedule_fingerprint != self.schedule.fingerprint:
            raise ValueError("trusted usage belongs to another schedule")
        if self.usage.cell_count != self.schedule.cell_count:
            raise ValueError("trusted usage does not cover the schedule")
        if self.schedule.max_attempts_per_cell != 1:
            raise ValueError("baseline gate closure requires first-attempt-only schedules")
        if self.usage.attempt_count != self.schedule.cell_count:
            raise ValueError("baseline gate closure requires one receipt per schedule cell")
        if {ref.sha256 for ref in self.receipt_refs} != {
            ref.sha256 for ref in self.usage.receipt_refs
        }:
            raise ValueError("receipt_refs differ from trusted usage")
        _require_ref_media(self.receipt_refs, EXECUTION_RECEIPT_MEDIA_TYPE, "receipt_refs")
        _require_ref_media(self.execution_refs, MODEL_EXECUTION_MEDIA_TYPE, "execution_refs")
        _require_ref_media(self.outcome_refs, ATTEMPT_OUTCOME_MEDIA_TYPE, "outcome_refs")
        expected_attempts = self.usage.attempt_count
        if (
            len(self.receipt_refs) != expected_attempts
            or len(self.execution_refs) != expected_attempts
            or len(self.outcome_refs) != expected_attempts
        ):
            raise ValueError("batch closure ref counts must match trusted attempt usage")
        if (
            self.usage.settled_attempts != expected_attempts
            or self.usage.burned_attempts
            or self.usage.poisoned_attempts
        ):
            raise ValueError("baseline gate closure requires only settled attempts")
        return self


class BaselineGateConditionClosure(ImmutableModel):
    """All plan-derived GATE batches and usage for one baseline condition."""

    schema_version: Literal["1"] = "1"
    kind: BaselineKind
    report_ref: ArtifactRef
    report: BaselineUsageReport
    batches: Annotated[tuple[BaselineGateBatchClosure, ...], Field(min_length=1)]

    @field_validator("batches")
    @classmethod
    def canonicalize_batches(
        cls,
        batches: tuple[BaselineGateBatchClosure, ...],
    ) -> tuple[BaselineGateBatchClosure, ...]:
        ordered = tuple(sorted(batches, key=lambda batch: batch.schedule.search_runs[0]))
        if len(ordered) != len({batch.schedule.search_runs[0] for batch in ordered}):
            raise ValueError("condition closure batches must not duplicate search runs")
        return ordered

    @model_validator(mode="after")
    def report_matches_batches(self) -> BaselineGateConditionClosure:
        if self.report_ref.media_type != BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE:
            raise ValueError("report_ref declares the wrong media type")
        if self.report.kind is not self.kind:
            raise ValueError("condition report belongs to another baseline kind")
        search_runs = tuple(batch.schedule.search_runs[0] for batch in self.batches)
        if search_runs != self.report.executed_search_run_seeds:
            raise ValueError("condition batches differ from reported search-run seeds")
        for batch in self.batches:
            if batch.schedule.kind != self.kind.value:
                raise ValueError("condition batch schedule belongs to another baseline kind")
            if batch.schedule.repeat_seeds != self.report.executed_repeat_seeds:
                raise ValueError("condition batch repeat seeds differ from its report")
        return self


class BaselineGateStudyClosure(ImmutableModel):
    """CAS-published structural closure for a complete four-condition GATE study."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    query: Annotated[int, Field(ge=0, strict=True)]
    protocol_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    token_ceiling_per_attempt: Annotated[int, Field(ge=1, strict=True)]
    consistency: BaselineProtocolConsistencyReport
    conditions: Annotated[
        tuple[BaselineGateConditionClosure, ...],
        Field(min_length=4, max_length=4),
    ]
    usage_evidence_scope: Literal["trusted-runner-receipts-and-gate-batches"] = (
        "trusted-runner-receipts-and-gate-batches"
    )
    reportable_benchmark_result: Literal[False] = False

    @field_validator("task_ids")
    @classmethod
    def canonicalize_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("task_ids must not contain duplicates")
        return ordered

    @field_validator("conditions")
    @classmethod
    def canonicalize_conditions(
        cls,
        conditions: tuple[BaselineGateConditionClosure, ...],
    ) -> tuple[BaselineGateConditionClosure, ...]:
        ordered = tuple(sorted(conditions, key=lambda condition: condition.kind.value))
        kinds = frozenset(condition.kind for condition in ordered)
        if kinds != REQUIRED_BASELINES:
            raise ValueError("baseline gate study closure requires exactly four conditions")
        return ordered

    @model_validator(mode="after")
    def exact_study_closure_shape(self) -> BaselineGateStudyClosure:
        _require_json_ref(self.protocol_ref, field_name="protocol_ref")
        _require_json_ref(self.gate_split_ref, field_name="gate_split_ref")
        if self.parent_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("parent_harness_ref declares the wrong media type")
        if self.consistency.plan_fingerprint != self.plan_fingerprint:
            raise ValueError("consistency report belongs to another baseline plan")
        if (
            tuple(condition.kind for condition in self.conditions)
            != self.consistency.baseline_kinds
        ):
            raise ValueError("condition set differs from the consistency report")
        for condition in self.conditions:
            if condition.report.plan_fingerprint != self.plan_fingerprint:
                raise ValueError("condition report belongs to another baseline plan")
            for batch in condition.batches:
                if (
                    batch.schedule.query != self.query
                    or batch.schedule.task_ids != self.task_ids
                    or batch.schedule.token_ceiling_per_attempt != self.token_ceiling_per_attempt
                    or batch.parent_harness_ref != self.parent_harness_ref
                ):
                    raise ValueError("condition batch axes differ from the study closure")
        return self


class BaselineGateExecutionLike(Protocol):
    """Structural input needed to close one executed baseline condition."""

    kind: BaselineKind
    report: BaselineUsageReport
    report_ref: ArtifactRef
    schedules: tuple[EvaluationBatchSchedule, ...]
    batches: tuple[BenchmarkBatchExecution, ...]


def build_baseline_gate_study_closure(
    *,
    plan: BaselineStudyPlan | object,
    query: int,
    protocol_ref: ArtifactRef,
    gate_split_ref: ArtifactRef,
    parent_harness_ref: ArtifactRef,
    task_ids: tuple[str, ...],
    token_ceiling_per_attempt: int,
    executions: tuple[BaselineGateExecutionLike, ...],
    consistency: BaselineProtocolConsistencyReport,
) -> BaselineGateStudyClosure:
    """Build the exact structural closure for a completed baseline GATE study."""

    checked_plan = BaselineProtocolValidator.validate_plan(plan)
    return BaselineGateStudyClosure(
        plan_fingerprint=checked_plan.fingerprint,
        query=query,
        protocol_ref=protocol_ref,
        gate_split_ref=gate_split_ref,
        parent_harness_ref=parent_harness_ref,
        task_ids=task_ids,
        token_ceiling_per_attempt=token_ceiling_per_attempt,
        consistency=consistency,
        conditions=tuple(_condition_closure(execution) for execution in executions),
    )


def publish_baseline_gate_study_closure(
    repository: ArtifactRepository,
    closure: BaselineGateStudyClosure,
    *,
    plan: BaselineStudyPlan | object,
) -> ArtifactRef:
    """Publish and immediately replay-check a baseline GATE study closure."""

    checked = BaselineGateStudyClosure.model_validate(closure, strict=True)
    ref = repository.put_json(checked, media_type=BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE)
    verified = verify_baseline_gate_study_closure(repository, ref, plan=plan)
    if verified != checked:
        raise BaselineGateClosureError("published baseline gate study closure changed content")
    return ArtifactRef.model_validate(ref, strict=True)


def verify_baseline_gate_study_closure(
    repository: ArtifactRepository,
    closure_ref: ArtifactRef,
    *,
    plan: BaselineStudyPlan | object | None = None,
) -> BaselineGateStudyClosure:
    """Reload and structurally verify a published baseline GATE study closure."""

    checked_repository = _require_repository(repository)
    checked_ref = _validate_ref(
        closure_ref,
        BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE,
        "baseline gate study closure",
    )
    closure = _load_model(
        checked_repository,
        checked_ref,
        BaselineGateStudyClosure,
        "baseline gate study closure",
    )
    reports: list[BaselineUsageReport] = []
    for condition in closure.conditions:
        report = _load_model(
            checked_repository,
            condition.report_ref,
            BaselineUsageReport,
            f"{condition.kind.value} usage report",
        )
        if report != condition.report:
            raise BaselineGateClosureError(
                f"{condition.kind.value} embedded usage report differs from report_ref"
            )
        reports.append(report)
        for batch in condition.batches:
            _verify_batch_artifacts(checked_repository, closure=closure, batch=batch)

    if plan is not None:
        checked_plan = BaselineProtocolValidator.validate_plan(plan)
        if closure.plan_fingerprint != checked_plan.fingerprint:
            raise BaselineGateClosureError("baseline gate study closure belongs to another plan")
        consistency = BaselineProtocolValidator.validate_usage(checked_plan, tuple(reports))
        if consistency != closure.consistency:
            raise BaselineGateClosureError("baseline gate study consistency cannot be replayed")
    return closure


def _condition_closure(
    execution: BaselineGateExecutionLike,
) -> BaselineGateConditionClosure:
    return BaselineGateConditionClosure(
        kind=execution.kind,
        report_ref=execution.report_ref,
        report=execution.report,
        batches=tuple(
            _batch_closure(schedule, batch)
            for schedule, batch in zip(execution.schedules, execution.batches, strict=True)
        ),
    )


def _batch_closure(
    schedule: EvaluationBatchSchedule,
    batch: BenchmarkBatchExecution,
) -> BaselineGateBatchClosure:
    if batch.parent_batch.candidate_ref != batch.candidate_batch.candidate_ref:
        raise BaselineGateClosureError("parent and candidate batches bind different candidates")
    if batch.parent_batch.mechanism_evidence_ref != batch.candidate_batch.mechanism_evidence_ref:
        raise BaselineGateClosureError("parent and candidate batches bind different mechanisms")
    return BaselineGateBatchClosure(
        schedule=schedule,
        candidate_ref=batch.candidate_batch.candidate_ref,
        parent_harness_ref=batch.parent_batch.harness_ref,
        candidate_harness_ref=batch.candidate_batch.harness_ref,
        mechanism_evidence_ref=batch.candidate_batch.mechanism_evidence_ref,
        preflight_ref=batch.preflight_ref,
        parent_batch_ref=batch.parent_batch_ref,
        candidate_batch_ref=batch.candidate_batch_ref,
        usage=batch.usage,
        receipt_refs=batch.receipt_refs,
        execution_refs=batch.execution_refs,
        outcome_refs=batch.outcome_refs,
    )


def _verify_batch_artifacts(
    repository: ArtifactRepository,
    *,
    closure: BaselineGateStudyClosure,
    batch: BaselineGateBatchClosure,
) -> None:
    preflight = _load_model(
        repository,
        batch.preflight_ref,
        SchedulePreflightCertificate,
        "preflight",
    )
    if not preflight.binds_schedule(batch.schedule):
        raise BaselineGateClosureError("batch preflight differs from the frozen schedule")
    if preflight.ledger_tail_ref is not None:
        raise BaselineGateClosureError("baseline gate closure requires a fresh preflight ledger")
    parent_batch = _load_model(repository, batch.parent_batch_ref, GateTrialBatch, "parent batch")
    candidate_batch = _load_model(
        repository,
        batch.candidate_batch_ref,
        GateTrialBatch,
        "candidate batch",
    )
    _verify_gate_batch(
        parent_batch,
        closure=closure,
        batch=batch,
        arm=GateTrialArm.PARENT,
        harness_ref=batch.parent_harness_ref,
    )
    _verify_gate_batch(
        candidate_batch,
        closure=closure,
        batch=batch,
        arm=GateTrialArm.CANDIDATE,
        harness_ref=batch.candidate_harness_ref,
    )
    receipts = tuple(
        _load_model(repository, ref, ExecutionReceipt, "execution receipt")
        for ref in batch.receipt_refs
    )
    executions = _load_by_digest(
        repository,
        batch.execution_refs,
        ModelExecution,
        "model execution",
    )
    outcomes = _load_by_digest(repository, batch.outcome_refs, AttemptOutcome, "attempt outcome")
    _verify_receipts(
        repository=repository,
        batch=batch,
        preflight=preflight,
        receipts=receipts,
        executions=executions,
        outcomes=outcomes,
    )


def _verify_gate_batch(
    gate_batch: GateTrialBatch,
    *,
    closure: BaselineGateStudyClosure,
    batch: BaselineGateBatchClosure,
    arm: GateTrialArm,
    harness_ref: ArtifactRef,
) -> None:
    expected = (
        closure.protocol_ref,
        batch.candidate_ref,
        arm,
        harness_ref,
        closure.gate_split_ref,
        batch.mechanism_evidence_ref,
    )
    actual = (
        gate_batch.protocol_ref,
        gate_batch.candidate_ref,
        gate_batch.arm,
        gate_batch.harness_ref,
        gate_batch.gate_split_ref,
        gate_batch.mechanism_evidence_ref,
    )
    if actual != expected:
        raise BaselineGateClosureError("gate batch differs from baseline closure")
    receipt_digests = {ref.sha256 for ref in batch.receipt_refs}
    source_digests = {ref.sha256 for ref in gate_batch.source_refs}
    if not receipt_digests.issubset(source_digests):
        raise BaselineGateClosureError("gate batch omits one or more receipt sources")
    expected_pairs = _expected_observation_pairs(batch.schedule, arm=arm)
    actual_pairs = tuple((item.task_id, item.seed) for item in gate_batch.observations)
    if tuple(sorted(actual_pairs)) != expected_pairs:
        raise BaselineGateClosureError("gate batch observations differ from the frozen schedule")


def _verify_receipts(
    *,
    repository: ArtifactRepository,
    batch: BaselineGateBatchClosure,
    preflight: SchedulePreflightCertificate,
    receipts: tuple[ExecutionReceipt, ...],
    executions: dict[str, ModelExecution],
    outcomes: dict[str, AttemptOutcome],
) -> None:
    execution_digests = {receipt.execution_ref.sha256 for receipt in receipts}
    if execution_digests != set(executions):
        raise BaselineGateClosureError("receipt executions differ from batch closure")
    outcome_digests = {receipt.outcome_ref.sha256 for receipt in receipts}
    if outcome_digests != set(outcomes):
        raise BaselineGateClosureError("receipt outcomes differ from batch closure")
    expected_cells = {cell.fingerprint for cell in batch.schedule.iter_cells()}
    actual_cells = {receipt.cell_fingerprint for receipt in receipts}
    if actual_cells != expected_cells or len(receipts) != len(expected_cells):
        raise BaselineGateClosureError("receipts do not cover the exact schedule cells")
    by_harness = {
        EvaluationSide.PARENT: batch.parent_harness_ref.sha256,
        EvaluationSide.CANDIDATE: batch.candidate_harness_ref.sha256,
    }
    for receipt in receipts:
        execution = executions[receipt.execution_ref.sha256]
        outcome = outcomes[receipt.outcome_ref.sha256]
        expected_harness_id = by_harness[receipt.cell.side]
        if receipt.schedule_fingerprint != batch.schedule.fingerprint:
            raise BaselineGateClosureError("receipt belongs to another schedule")
        if not batch.schedule.contains(receipt.cell):
            raise BaselineGateClosureError("receipt cell is outside the frozen schedule")
        if receipt.cell_fingerprint != receipt.cell.fingerprint:
            raise BaselineGateClosureError("receipt cell fingerprint changed")
        if execution.harness_id != expected_harness_id:
            raise BaselineGateClosureError("receipt side has a mismatched harness")
        if execution.task_id != receipt.cell.task_id:
            raise BaselineGateClosureError("execution task differs from its receipt cell")
        try:
            expected_seed = batch.schedule.seed_for(
                receipt.cell,
                attempt_index=receipt.attempt_index,
            )
        except Exception as exc:
            raise BaselineGateClosureError("receipt attempt index is outside the schedule") from exc
        if execution.seed != expected_seed:
            raise BaselineGateClosureError("execution seed differs from its receipt cell")
        if receipt.preflight_ref != batch.preflight_ref:
            raise BaselineGateClosureError("receipt preflight differs from the batch closure")
        if receipt.preflight_fingerprint != preflight.fingerprint:
            raise BaselineGateClosureError("receipt preflight fingerprint changed")
        if not preflight.binds_execution(execution):
            raise BaselineGateClosureError("execution differs from the batch preflight")
        if outcome.disposition is not AttemptDisposition.SETTLED:
            raise BaselineGateClosureError("baseline gate receipt did not settle")
        if outcome.execution_ref != receipt.execution_ref:
            raise BaselineGateClosureError("outcome execution differs from its receipt")
        if outcome.reservation_ref != receipt.reservation_ref:
            raise BaselineGateClosureError("outcome reservation differs from its receipt")
        reservation = _load_model(
            repository,
            receipt.reservation_ref,
            AttemptReservation,
            "attempt reservation",
        )
        _verify_reservation(
            preflight=preflight,
            receipt=receipt,
            reservation=reservation,
            outcome=outcome,
            execution=execution,
        )
        if (
            outcome.reported_tokens != receipt.reported_tokens
            or outcome.charged_tokens != receipt.charged_tokens
        ):
            raise BaselineGateClosureError("outcome token accounting differs from its receipt")


def _verify_reservation(
    *,
    preflight: SchedulePreflightCertificate,
    receipt: ExecutionReceipt,
    reservation: AttemptReservation,
    outcome: AttemptOutcome,
    execution: ModelExecution,
) -> None:
    if receipt.reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
        raise BaselineGateClosureError("receipt reservation declares the wrong media type")
    shared = (
        reservation.ledger_id,
        reservation.writer_epoch_id,
        reservation.budget_fingerprint,
    )
    if shared != (preflight.ledger_id, preflight.writer_epoch_id, preflight.budget_fingerprint):
        raise BaselineGateClosureError("reservation differs from the batch preflight")
    if shared != (outcome.ledger_id, outcome.writer_epoch_id, outcome.budget_fingerprint):
        raise BaselineGateClosureError("outcome differs from its reservation identity")
    if reservation.sequence != outcome.sequence:
        raise BaselineGateClosureError("outcome sequence differs from its reservation")
    if reservation.task_fingerprint != execution.task.fingerprint:
        raise BaselineGateClosureError("reservation task differs from its execution")
    if reservation.execution_fingerprint != execution.execution_fingerprint:
        raise BaselineGateClosureError("reservation execution fingerprint changed")
    if reservation.request_sha256 != execution.request_sha256:
        raise BaselineGateClosureError("reservation request hash differs from execution")
    if reservation.reserved_tokens != preflight.token_ceiling_per_attempt:
        raise BaselineGateClosureError("reservation token ceiling differs from preflight")


def _expected_observation_pairs(
    schedule: EvaluationBatchSchedule,
    *,
    arm: GateTrialArm,
) -> tuple[tuple[str, int], ...]:
    side = EvaluationSide.PARENT if arm is GateTrialArm.PARENT else EvaluationSide.CANDIDATE
    pairs = tuple(
        (cell.task_id, schedule.seed_for(cell, attempt_index=0))
        for cell in schedule.iter_cells()
        if cell.side is side
    )
    return tuple(sorted(pairs))


def _require_repository(repository: ArtifactRepository) -> ArtifactRepository:
    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    return repository


def _validate_ref(ref: ArtifactRef, expected_media_type: str, label: str) -> ArtifactRef:
    checked = ArtifactRef.model_validate(ref, strict=True)
    if checked.media_type != expected_media_type:
        raise BaselineGateClosureError(f"{label} declares the wrong media type")
    return checked


def _load_model[ModelT](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    try:
        value = repository.get_json(ref, model_type)
        return _strict_model_validate(value, model_type)
    except Exception as exc:
        raise BaselineGateClosureError(f"{label} artifact cannot be verified") from exc


def _load_by_digest[ModelT](
    repository: ArtifactRepository,
    refs: tuple[ArtifactRef, ...],
    model_type: type[ModelT],
    label: str,
) -> dict[str, ModelT]:
    return {ref.sha256: _load_model(repository, ref, model_type, label) for ref in refs}


def _strict_model_validate[ModelT](value: object, model_type: type[ModelT]) -> ModelT:
    validator = getattr(model_type, "model_validate", None)
    if not callable(validator):
        raise TypeError("model_type must expose model_validate")
    return validator(value, strict=True)


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _require_ref_media(
    refs: tuple[ArtifactRef, ...],
    expected_media_type: str,
    field_name: str,
) -> None:
    if len({ref.sha256 for ref in refs}) != len(refs):
        raise ValueError(f"{field_name} must not contain duplicates")
    for ref in refs:
        if ref.media_type != expected_media_type:
            raise ValueError(f"{field_name} contains the wrong media type")


__all__ = [
    "BASELINE_GATE_STUDY_CLOSURE_MEDIA_TYPE",
    "BASELINE_GATE_USAGE_REPORT_MEDIA_TYPE",
    "BaselineGateBatchClosure",
    "BaselineGateClosureError",
    "BaselineGateConditionClosure",
    "BaselineGateStudyClosure",
    "build_baseline_gate_study_closure",
    "publish_baseline_gate_study_closure",
    "verify_baseline_gate_study_closure",
]
