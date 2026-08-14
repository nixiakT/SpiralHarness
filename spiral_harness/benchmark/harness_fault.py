"""Partition-scoped, full-roster receipt grading for v3 one-family slice."""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from spiral_harness.benchmark._harness_fault_cases import (
    PARTITION_OPENING_MEDIA_TYPE,
    PARTITION_ROSTER_MEDIA_TYPE,
    FaultFamily,
    HarnessFaultTask,
    PartitionEvaluationGrant,
    RepairRuleId,
    ScenarioRole,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault_compiler import (
    HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    HarnessFaultCompilationManifest,
    HarnessRole,
    verify_fault_compilation,
)
from spiral_harness.benchmark.harness_fault_runtime import (
    RUNTIME_EVENT_MEDIA_TYPE,
    AttestedRuntimeBranchEvent,
    HarnessFaultMiddlewareBackend,
    RuntimeEventVerificationCapability,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    CandidateTask,
    ExecutionStatus,
    FrozenModelSpec,
    ModelExecution,
)
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    TrustedExecutionUsage,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationCellKey,
    EvaluationPhase,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.models import TrialObservation, TrialStatus

HARNESS_FAULT_EVALUATOR_VERSION = "spiral-harness.harness-fault-evaluator:v3-one-family"
HARNESS_FAULT_SCHEDULE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-schedule-closure.v3+json"
)
HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-graded-outcome.v3+json"
)
HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-graded-batch.v3+json"
)


class HarnessFaultDataError(ValueError):
    """Partition access or authority-owned task data failed closed."""


class HarnessFaultGradingError(ValueError):
    """Full-roster schedule, receipt, runtime, or grading closure failed."""


class HarnessFaultComparisonKind(StrEnum):
    MAIN = "main"
    REVERT = "revert"
    PLACEBO = "placebo"


class HarnessFaultOutput(ImmutableModel):
    """Worker-visible output contains adherence/behavior only, never activation."""

    answer: Literal["MATCH", "DIFFERENT"]
    observable: NonEmptyStr

    @field_validator("observable", mode="before")
    @classmethod
    def exact_observable(cls, value: Any) -> Any:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("observable must be an exact non-empty string")
        return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise HarnessFaultGradingError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def parse_harness_fault_output(text: str) -> HarnessFaultOutput:
    if type(text) is not str:
        raise HarnessFaultGradingError("fault output must be a string")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HarnessFaultGradingError(f"forbidden JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, HarnessFaultGradingError) as exc:
        raise HarnessFaultGradingError(f"invalid strict JSON output: {exc}") from exc
    if type(value) is not dict:
        raise HarnessFaultGradingError("fault output must be one JSON object")
    try:
        return HarnessFaultOutput.model_validate(value, strict=True)
    except ValidationError as exc:
        raise HarnessFaultGradingError(f"invalid output contract: {exc}") from exc


class HarnessFaultScheduleClosure(ImmutableModel):
    """Persisted full-roster schedule joined to authority and compiler refs."""

    schema_version: Literal["3"] = "3"
    evaluator_fingerprint: Sha256
    authority_id: Sha256
    partition: ProtocolPartition
    comparison_kind: HarnessFaultComparisonKind
    opening_ref: ArtifactRef
    roster_ref: ArtifactRef
    compilation_ref: ArtifactRef
    schedule: EvaluationBatchSchedule

    @model_validator(mode="after")
    def exact_media(self) -> Self:
        expected = (
            (self.opening_ref, PARTITION_OPENING_MEDIA_TYPE),
            (self.roster_ref, PARTITION_ROSTER_MEDIA_TYPE),
            (self.compilation_ref, HARNESS_FAULT_COMPILATION_MEDIA_TYPE),
        )
        if any(ref.media_type != media for ref, media in expected):
            raise ValueError("schedule closure contains a ref with the wrong media type")
        return self


class HarnessFaultGradedOutcome(ImmutableModel):
    """One raw-evidence-bound outcome retaining complete cluster/cell coordinates."""

    schema_version: Literal["3"] = "3"
    observation: TrialObservation
    cell: EvaluationCellKey
    partition: ProtocolPartition
    scenario_id: NonEmptyStr
    scenario_commitment: Sha256
    family: FaultFamily
    template_id: NonEmptyStr
    source_id: NonEmptyStr
    group_id: NonEmptyStr
    scenario_role: ScenarioRole
    harness_role: HarnessRole
    rule_id: RepairRuleId
    receipt_ref: ArtifactRef
    execution_ref: ArtifactRef
    attempt_outcome_ref: ArtifactRef
    runtime_event_ref: ArtifactRef
    parsed_output: HarnessFaultOutput | None

    @model_validator(mode="after")
    def exact_media_and_cell(self) -> Self:
        expected = (
            (self.receipt_ref, EXECUTION_RECEIPT_MEDIA_TYPE),
            (self.execution_ref, MODEL_EXECUTION_MEDIA_TYPE),
            (self.attempt_outcome_ref, ATTEMPT_OUTCOME_MEDIA_TYPE),
            (self.runtime_event_ref, RUNTIME_EVENT_MEDIA_TYPE),
        )
        if any(ref.media_type != media for ref, media in expected):
            raise ValueError("graded outcome contains a ref with the wrong media type")
        if self.cell.task_id != self.observation.task_id:
            raise ValueError("outcome cell and observation task differ")
        return self


class HarnessFaultGradedBatch(ImmutableModel):
    """Exact full-roster result of strongest live-ledger receipt replay."""

    schema_version: Literal["3"] = "3"
    evaluator_fingerprint: Sha256
    partition: ProtocolPartition
    comparison_kind: HarnessFaultComparisonKind
    schedule_ref: ArtifactRef
    opening_ref: ArtifactRef
    roster_ref: ArtifactRef
    compilation_ref: ArtifactRef
    usage: TrustedExecutionUsage
    outcome_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def exact_refs(self) -> Self:
        expected = (
            (self.schedule_ref, HARNESS_FAULT_SCHEDULE_MEDIA_TYPE),
            (self.opening_ref, PARTITION_OPENING_MEDIA_TYPE),
            (self.roster_ref, PARTITION_ROSTER_MEDIA_TYPE),
            (self.compilation_ref, HARNESS_FAULT_COMPILATION_MEDIA_TYPE),
        )
        if any(ref.media_type != media for ref, media in expected):
            raise ValueError("graded batch contains a ref with the wrong media type")
        if any(
            ref.media_type != HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE for ref in self.outcome_refs
        ):
            raise ValueError("outcome_refs contains the wrong media type")
        if len({ref.sha256 for ref in self.outcome_refs}) != len(self.outcome_refs):
            raise ValueError("outcome_refs must be unique")
        return self


class HarnessFaultGradedBatchRecord(ImmutableModel):
    batch: HarnessFaultGradedBatch
    batch_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_media(self) -> Self:
        if self.batch_ref.media_type != HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE:
            raise ValueError("batch_ref declares the wrong media type")
        return self


def _load(store: ArtifactStore, ref: ArtifactRef, media: str, model: type, label: str):
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
        if checked.media_type != media:
            raise ValueError("wrong media type")
        loaded = store.get_json(checked, model)
        return model.model_validate(loaded, strict=True)
    except Exception as exc:
        raise HarnessFaultGradingError(f"{label} artifact cannot be verified") from exc


class _PartitionEvaluator:
    """Private common implementation behind phase-specific evaluator types."""

    __slots__ = (
        "__records",
        "_fingerprint",
        "_grant",
        "_roster",
        "_runtime_verification",
        "_store",
    )
    _EXPECTED_PARTITION: ProtocolPartition

    def __init__(
        self,
        store: ArtifactStore,
        grant: PartitionEvaluationGrant,
        runtime_verification: RuntimeEventVerificationCapability,
    ) -> None:
        if type(store) is not ArtifactStore:
            raise TypeError("store must be an exact ArtifactStore")
        if type(runtime_verification) is not RuntimeEventVerificationCapability:
            raise TypeError("runtime_verification must be an exact capability")
        checked_grant = PartitionEvaluationGrant.model_validate(grant, strict=True)
        if checked_grant.partition is not self._EXPECTED_PARTITION:
            raise HarnessFaultDataError("grant belongs to another evaluator partition")
        verified = verify_partition_opening(store, checked_grant)
        self._store = store
        self._grant = checked_grant
        self._roster = verified.roster
        self.__records = {item.task.task_id: item for item in verified.scenarios}
        self._runtime_verification = runtime_verification
        self._fingerprint = canonical_sha256(
            {
                "version": HARNESS_FAULT_EVALUATOR_VERSION,
                "partition": checked_grant.partition,
                "commitment": checked_grant.public_commitment.fingerprint,
                "opening_ref": checked_grant.opening_ref,
                "roster_ref": checked_grant.roster_ref,
                "runtime_producer_id": runtime_verification.producer_id,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def partition_grant(self) -> PartitionEvaluationGrant:
        return self._grant

    @property
    def runtime_verification(self) -> RuntimeEventVerificationCapability:
        return self._runtime_verification

    def task_roster(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self._roster.tasks)

    def load_task(self, task_ref: str) -> HarnessFaultTask:
        if type(task_ref) is not str or not task_ref:
            raise HarnessFaultDataError("task_ref must be an exact non-empty string")
        for task in self._roster.tasks:
            if task.task_id == task_ref:
                return task
        raise HarnessFaultDataError("task is outside this evaluator's exact roster")

    def publish_schedule(
        self,
        schedule: EvaluationBatchSchedule,
        *,
        comparison_kind: HarnessFaultComparisonKind,
        compilation_ref: ArtifactRef,
        spec: FrozenModelSpec,
    ) -> ArtifactRef:
        checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
        checked_kind = HarnessFaultComparisonKind(comparison_kind)
        compilation = verify_fault_compilation(
            self._store,
            spec,
            compilation_ref,
            expected_runtime_producer_id=self._runtime_verification.producer_id,
        )
        if tuple(checked_schedule.task_ids) != tuple(sorted(self.task_roster())):
            raise HarnessFaultGradingError(
                "schedule must cover the exact complete partition roster"
            )
        if checked_schedule.max_attempts_per_cell != 1:
            raise HarnessFaultGradingError(
                "v3 one-family closure requires exactly one attempt per schedule cell"
            )
        self._verify_phase_and_harnesses(checked_schedule, checked_kind, compilation)
        closure = HarnessFaultScheduleClosure(
            evaluator_fingerprint=self.fingerprint,
            authority_id=self._grant.public_commitment.authority_id,
            partition=self._EXPECTED_PARTITION,
            comparison_kind=checked_kind,
            opening_ref=self._grant.opening_ref,
            roster_ref=self._grant.roster_ref,
            compilation_ref=compilation_ref,
            schedule=checked_schedule,
        )
        return self._store.put_json(closure, media_type=HARNESS_FAULT_SCHEDULE_MEDIA_TYPE)

    def _verify_phase_and_harnesses(
        self,
        schedule: EvaluationBatchSchedule,
        kind: HarnessFaultComparisonKind,
        compilation: HarnessFaultCompilationManifest,
    ) -> None:
        if self._EXPECTED_PARTITION is ProtocolPartition.EXPLORATION:
            expected_phase = (
                EvaluationPhase.EXPLORATION
                if kind is HarnessFaultComparisonKind.MAIN
                else EvaluationPhase.PROBE
            )
        else:
            if kind is not HarnessFaultComparisonKind.MAIN:
                raise HarnessFaultGradingError("gate/sealed evaluators admit main batches only")
            expected_phase = (
                EvaluationPhase.GATE
                if self._EXPECTED_PARTITION is ProtocolPartition.GATE
                else EvaluationPhase.SEALED
            )
        if schedule.phase is not expected_phase:
            raise HarnessFaultGradingError("schedule phase differs from evaluator capability")
        parent_role = {
            HarnessFaultComparisonKind.MAIN: HarnessRole.FAULTY_PARENT,
            HarnessFaultComparisonKind.REVERT: HarnessRole.REVERT,
            HarnessFaultComparisonKind.PLACEBO: HarnessRole.PLACEBO,
        }[kind]
        if (
            schedule.parent_harness_id != compilation.entry(parent_role).harness_ref.sha256
            or schedule.candidate_harness_id
            != compilation.entry(HarnessRole.CANDIDATE).harness_ref.sha256
        ):
            raise HarnessFaultGradingError("schedule harness pair differs from compiler graph")

    def grade_receipt_batch(
        self,
        *,
        schedule_ref: ArtifactRef,
        preflight_ref: ArtifactRef,
        attempt_ledger: AttemptLedger,
        receipt_refs: Iterable[ArtifactRef],
        runtime_backend: HarnessFaultMiddlewareBackend,
    ) -> HarnessFaultGradedBatchRecord:
        closure = _load(
            self._store,
            schedule_ref,
            HARNESS_FAULT_SCHEDULE_MEDIA_TYPE,
            HarnessFaultScheduleClosure,
            "schedule closure",
        )
        if (
            closure.evaluator_fingerprint != self.fingerprint
            or closure.opening_ref != self._grant.opening_ref
            or closure.roster_ref != self._grant.roster_ref
            or closure.partition is not self._EXPECTED_PARTITION
        ):
            raise HarnessFaultGradingError("schedule closure belongs to another evaluator")
        if (
            type(attempt_ledger) is not AttemptLedger
            or attempt_ledger.repository is not self._store
        ):
            raise HarnessFaultGradingError("grade requires this store's exact live AttemptLedger")
        if (
            type(runtime_backend) is not HarnessFaultMiddlewareBackend
            or runtime_backend.repository is not self._store
            or runtime_backend.compilation_ref != closure.compilation_ref
            or runtime_backend.producer_id != self._runtime_verification.producer_id
        ):
            raise HarnessFaultGradingError("runtime backend differs from frozen trusted producer")
        preflight = _load(
            self._store,
            preflight_ref,
            SCHEDULE_PREFLIGHT_MEDIA_TYPE,
            SchedulePreflightCertificate,
            "preflight",
        )
        if not preflight.binds_schedule(closure.schedule):
            raise HarnessFaultGradingError("preflight differs from persisted schedule")
        compilation = verify_fault_compilation(
            self._store,
            preflight.model_spec,
            closure.compilation_ref,
            expected_runtime_producer_id=self._runtime_verification.producer_id,
        )
        self._verify_phase_and_harnesses(closure.schedule, closure.comparison_kind, compilation)
        refs = tuple(receipt_refs)
        usage = replay_trusted_usage(
            self._store,
            schedule=closure.schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=attempt_ledger,
            receipt_refs=refs,
        )
        outcome_refs = tuple(
            self._grade_one(
                receipt_ref,
                closure=closure,
                compilation=compilation,
                spec=preflight.model_spec,
                runtime_backend=runtime_backend,
            )
            for receipt_ref in usage.receipt_refs
        )
        if len(outcome_refs) != closure.schedule.cell_count:
            raise HarnessFaultGradingError("graded outcomes do not cover every schedule cell")
        batch = HarnessFaultGradedBatch(
            evaluator_fingerprint=self.fingerprint,
            partition=self._EXPECTED_PARTITION,
            comparison_kind=closure.comparison_kind,
            schedule_ref=schedule_ref,
            opening_ref=self._grant.opening_ref,
            roster_ref=self._grant.roster_ref,
            compilation_ref=closure.compilation_ref,
            usage=usage,
            outcome_refs=outcome_refs,
        )
        batch_ref = self._store.put_json(batch, media_type=HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE)
        return HarnessFaultGradedBatchRecord(batch=batch, batch_ref=batch_ref)

    def _grade_one(
        self,
        receipt_ref: ArtifactRef,
        *,
        closure: HarnessFaultScheduleClosure,
        compilation: HarnessFaultCompilationManifest,
        spec: FrozenModelSpec,
        runtime_backend: HarnessFaultMiddlewareBackend,
    ) -> ArtifactRef:
        receipt = _load(
            self._store,
            receipt_ref,
            EXECUTION_RECEIPT_MEDIA_TYPE,
            ExecutionReceipt,
            "receipt",
        )
        execution = _load(
            self._store,
            receipt.execution_ref,
            MODEL_EXECUTION_MEDIA_TYPE,
            ModelExecution,
            "execution",
        )
        attempt = _load(
            self._store,
            receipt.outcome_ref,
            ATTEMPT_OUTCOME_MEDIA_TYPE,
            AttemptOutcome,
            "attempt outcome",
        )
        try:
            scenario = self.__records[receipt.cell.task_id]
        except KeyError as exc:
            raise HarnessFaultGradingError("receipt task is outside exact roster") from exc
        if (
            receipt.schedule_fingerprint != closure.schedule.fingerprint
            or not closure.schedule.contains(receipt.cell)
            or receipt.attempt_index != 0
            or attempt.execution_ref != receipt.execution_ref
            or execution.task != CandidateTask.from_task_view(scenario.task)
            or execution.seed != closure.schedule.seed_for(receipt.cell)
        ):
            raise HarnessFaultGradingError("receipt/execution/scenario provenance mismatch")
        try:
            role = compilation.role_for_harness(execution.request.harness_ref)
        except KeyError as exc:
            raise HarnessFaultGradingError("execution harness is outside compiler") from exc
        entry = compilation.entry(role)
        HarnessMaterializer(self._store, spec=spec).verify_execution_request(
            entry.harness_ref, execution
        )
        event_ref = runtime_backend.event_ref_for(execution.request_sha256)
        event = _load(
            self._store,
            event_ref,
            RUNTIME_EVENT_MEDIA_TYPE,
            AttestedRuntimeBranchEvent,
            "runtime event",
        )
        event = self._runtime_verification.verify(event)
        if (
            event.compilation_ref != closure.compilation_ref
            or event.request_sha256 != execution.request_sha256
            or event.task_id != execution.task_id
            or event.harness_ref != execution.request.harness_ref
            or event.rule_id is not entry.rule_id
            or event.raw_left_sha256 != sha256_bytes(scenario.left.encode("utf-8"))
            or event.raw_right_sha256 != sha256_bytes(scenario.right.encode("utf-8"))
            or execution.output is None
            or event.final_output_sha256 != sha256_bytes(execution.output.encode("utf-8"))
        ):
            raise HarnessFaultGradingError("runtime branch event does not bind raw execution")

        parsed = None
        score = None
        if execution.status is ExecutionStatus.COMPLETED:
            if attempt.disposition is not AttemptDisposition.SETTLED:
                raise HarnessFaultGradingError("completed execution is not settled")
            try:
                parsed = parse_harness_fault_output(execution.output)
            except HarnessFaultGradingError:
                score = 0.0
            else:
                score = float(
                    parsed.answer == scenario.expected_answer
                    and parsed.observable == scenario.expected_observable
                )
            status = TrialStatus.COMPLETED
        else:
            if attempt.disposition is AttemptDisposition.SETTLED:
                raise HarnessFaultGradingError("failed execution cannot be settled")
            status = TrialStatus.INFRA_ERROR
        observation = TrialObservation(
            task_id=scenario.task.task_id,
            seed=execution.seed,
            harness_id=execution.harness_id,
            status=status,
            score=score,
            slice_tags=(
                "benchmark:harness-fault-v3-one-family",
                f"family:{scenario.family.value}",
                f"group:{scenario.group_id}",
                f"scenario:{scenario.scenario_id}",
                f"source:{scenario.source_id}",
                f"template:{scenario.template_id}",
                f"scenario-role:{scenario.role.value}",
                f"harness-role:{role.value}",
            ),
            tokens=execution.tokens,
            latency_ms=execution.latency_ms,
            tool_calls=execution.tool_calls,
            cost_usd=execution.cost_usd,
            execution_fingerprint=execution.execution_fingerprint,
        )
        outcome = HarnessFaultGradedOutcome(
            observation=observation,
            cell=receipt.cell,
            partition=self._EXPECTED_PARTITION,
            scenario_id=scenario.scenario_id,
            scenario_commitment=scenario.scenario_commitment,
            family=scenario.family,
            template_id=scenario.template_id,
            source_id=scenario.source_id,
            group_id=scenario.group_id,
            scenario_role=scenario.role,
            harness_role=role,
            rule_id=entry.rule_id,
            receipt_ref=receipt_ref,
            execution_ref=receipt.execution_ref,
            attempt_outcome_ref=receipt.outcome_ref,
            runtime_event_ref=event_ref,
            parsed_output=parsed,
        )
        return self._store.put_json(outcome, media_type=HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE)


class HarnessFaultExplorationGrader(_PartitionEvaluator):
    _EXPECTED_PARTITION = ProtocolPartition.EXPLORATION


class HarnessFaultGateEvaluator(_PartitionEvaluator):
    _EXPECTED_PARTITION = ProtocolPartition.GATE


class HarnessFaultSealedEvaluator(_PartitionEvaluator):
    _EXPECTED_PARTITION = ProtocolPartition.SEALED


__all__ = [
    "HARNESS_FAULT_EVALUATOR_VERSION",
    "HARNESS_FAULT_GRADED_BATCH_MEDIA_TYPE",
    "HARNESS_FAULT_GRADED_OUTCOME_MEDIA_TYPE",
    "HARNESS_FAULT_SCHEDULE_MEDIA_TYPE",
    "HarnessFaultComparisonKind",
    "HarnessFaultDataError",
    "HarnessFaultExplorationGrader",
    "HarnessFaultGateEvaluator",
    "HarnessFaultGradedBatch",
    "HarnessFaultGradedBatchRecord",
    "HarnessFaultGradedOutcome",
    "HarnessFaultGradingError",
    "HarnessFaultOutput",
    "HarnessFaultScheduleClosure",
    "HarnessFaultSealedEvaluator",
    "parse_harness_fault_output",
]
