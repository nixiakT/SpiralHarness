"""Trusted scoring loop for the harness-free PURE model reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.benchmark.base import BenchmarkAdapter
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.accounted_execution import load_accounted_execution
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    PURE_REFERENCE_EXECUTION_MEDIA_TYPE,
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    CandidateTask,
    ExecutionStatus,
    FrozenModelSpec,
)
from spiral_harness.execution.model import paired_execution_fingerprint
from spiral_harness.execution.pure_contracts import (
    PureReferenceCapabilities,
    PureReferenceExecution,
    materialize_pure_request,
)
from spiral_harness.execution.pure_model import PureModelRunner
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.models import TrialObservation

PURE_REFERENCE_BATCH_MEDIA_TYPE = "application/vnd.spiral-harness.pure-reference-batch.v1+json"
PURE_REFERENCE_CONTRACT = "spiral-harness/pure-model-reference/v1"


class PureReferenceRunnerError(RuntimeError):
    """A frozen PURE reference batch could not be executed exactly."""


class PureReferenceCell(ImmutableModel):
    """One frozen task-byte and rollout-seed coordinate."""

    task_id: NonEmptyStr
    task_fingerprint: Sha256
    task_payload_sha256: Sha256
    task_payload_size_bytes: Annotated[int, Field(ge=1, strict=True)]
    rollout_seed: Annotated[int, Field(ge=0, strict=True)]

    @classmethod
    def from_task(cls, task: object, *, rollout_seed: int) -> PureReferenceCell:
        checked = CandidateTask.from_task_view(task)
        payload = checked.question.encode("utf-8")
        return cls(
            task_id=checked.task_id,
            task_fingerprint=checked.fingerprint,
            task_payload_sha256=sha256_bytes(payload),
            task_payload_size_bytes=len(payload),
            rollout_seed=rollout_seed,
        )

    def require_exact_task(self, task: object) -> CandidateTask:
        """Reject task-ID or byte drift before any provider invocation."""

        checked = CandidateTask.from_task_view(task)
        payload = checked.question.encode("utf-8")
        actual = (
            checked.task_id,
            checked.fingerprint,
            sha256_bytes(payload),
            len(payload),
        )
        expected = (
            self.task_id,
            self.task_fingerprint,
            self.task_payload_sha256,
            self.task_payload_size_bytes,
        )
        if actual != expected:
            raise PureReferenceRunnerError("trusted task bytes differ from the frozen PURE cell")
        return checked


class PureReferencePlan(ImmutableModel):
    """Complete immutable identity of one dedicated PURE reference ledger."""

    schema_version: Literal["1"] = "1"
    condition: Literal["pure-model-reference"] = "pure-model-reference"
    contract: Literal["spiral-harness/pure-model-reference/v1"] = PURE_REFERENCE_CONTRACT
    capabilities: PureReferenceCapabilities = PureReferenceCapabilities()
    adapter_fingerprint: NonEmptyStr
    partition: ProtocolPartition
    spec: FrozenModelSpec
    cells: Annotated[tuple[PureReferenceCell, ...], Field(min_length=1)]
    ledger_id: NonEmptyStr
    attempt_budget: AttemptBudget

    @field_validator("adapter_fingerprint", "ledger_id", mode="before")
    @classmethod
    def identity_text_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("PURE plan identity text must be exact and non-empty")
        return value

    @model_validator(mode="after")
    def cells_and_dedicated_budget_are_exact(self) -> Self:
        coordinates = tuple((cell.task_id, cell.rollout_seed) for cell in self.cells)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("PURE plan task/seed coordinates must be unique")
        if self.attempt_budget.max_attempts != len(self.cells):
            raise ValueError("PURE plan requires exactly one attempt per frozen cell")
        reserved_total = len(self.cells) * self.attempt_budget.max_tokens_per_attempt
        if self.attempt_budget.max_total_tokens != reserved_total:
            raise ValueError("PURE plan budget must reserve the exact worst-case token total")
        return self

    @property
    def reference_id(self) -> str:
        """Derived condition identity; never aliases a harness manifest."""

        return canonical_sha256(
            {
                "contract": PURE_REFERENCE_CONTRACT,
                "frozen_plan": self,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.reference_id


class PureReferenceBatch(ImmutableModel):
    """Content-addressed engineering batch graded by the supplied adapter.

    The false literals are deliberate: replaying this single-arm artifact does
    not attest the provider, establish cross-arm pairing, or turn development
    output into sealed benchmark evidence.
    """

    schema_version: Literal["1"] = "1"
    condition: Literal["pure-model-reference"] = "pure-model-reference"
    reference_id: Sha256
    plan: PureReferencePlan
    capabilities: PureReferenceCapabilities = PureReferenceCapabilities()
    scoring_plane: Literal["trusted-benchmark-adapter"] = "trusted-benchmark-adapter"
    observations: Annotated[tuple[TrialObservation, ...], Field(min_length=1)]
    execution_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    outcome_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    ledger_tail_ref: ArtifactRef
    all_executions_completed_successfully: bool
    terminal_ledger_poisoned: bool
    reportable_benchmark_result: Literal[False] = False
    sealed_evidence: Literal[False] = False
    paired_cross_arm_receipt_closure: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    standalone_verification: Literal[False] = False

    @model_validator(mode="after")
    def shape_is_self_consistent(self) -> Self:
        if self.reference_id != self.plan.reference_id:
            raise ValueError("PURE batch reference_id was not derived from its frozen plan")
        if self.capabilities != self.plan.capabilities:
            raise ValueError("PURE batch capabilities differ from its plan")
        count = len(self.plan.cells)
        if not (
            len(self.observations) == len(self.execution_refs) == len(self.outcome_refs) == count
        ):
            raise ValueError("PURE batch artifacts do not cover every frozen cell exactly once")
        if any(
            ref.media_type != PURE_REFERENCE_EXECUTION_MEDIA_TYPE for ref in self.execution_refs
        ):
            raise ValueError("PURE batch contains a non-PURE execution reference")
        if any(ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE for ref in self.outcome_refs):
            raise ValueError("PURE batch contains an invalid attempt outcome reference")
        if len({ref.sha256 for ref in self.execution_refs}) != count:
            raise ValueError("PURE batch execution references must be unique")
        if len({ref.sha256 for ref in self.outcome_refs}) != count:
            raise ValueError("PURE batch outcome references must be unique")
        if self.ledger_tail_ref != self.outcome_refs[-1]:
            raise ValueError("PURE batch ledger tail does not close its outcome chain")
        for cell, observation in zip(self.plan.cells, self.observations, strict=True):
            if (
                observation.task_id,
                observation.seed,
                observation.harness_id,
            ) != (cell.task_id, cell.rollout_seed, self.reference_id):
                raise ValueError("PURE observation provenance differs from its frozen cell")
        return self


@dataclass(frozen=True, slots=True)
class PureReferenceBatchExecution:
    """Published PURE batch plus score-free in-memory executions."""

    batch: PureReferenceBatch
    batch_ref: ArtifactRef
    executions: tuple[PureReferenceExecution, ...]


class TrustedPureReferenceRunner[TaskT]:
    """Preflight, execute, and grade a PURE plan without exposing the adapter."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        adapter: BenchmarkAdapter[str, TaskT, object],
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        if not isinstance(adapter, BenchmarkAdapter):
            raise TypeError("adapter must implement BenchmarkAdapter")
        self._repository = repository
        self._adapter = adapter

    def execute(
        self,
        *,
        plan: PureReferencePlan,
        runner: PureModelRunner,
        attempt_ledger: AttemptLedger,
    ) -> PureReferenceBatchExecution:
        """Run one preflighted plan and score outputs inside the trusted plane."""

        checked_plan = PureReferencePlan.model_validate(plan, strict=True)
        if type(runner) is not PureModelRunner:
            raise TypeError("runner must be an exact PureModelRunner")
        if type(attempt_ledger) is not AttemptLedger:
            raise TypeError("attempt_ledger must be an exact AttemptLedger")
        self._require_execution_context(checked_plan, runner, attempt_ledger)
        tasks = self._preflight_tasks(checked_plan)

        records = []
        observations = []
        for cell, task in zip(checked_plan.cells, tasks, strict=True):
            record = runner.execute_record(
                task,
                reference_id=checked_plan.reference_id,
                rollout_seed=cell.rollout_seed,
                reservation_token_ceiling=checked_plan.attempt_budget.max_tokens_per_attempt,
            )
            observation = self._adapter.grade(
                task,
                record.execution,
                harness_id=checked_plan.reference_id,
                seed=cell.rollout_seed,
                execution_fingerprint=record.execution.execution_fingerprint,
            )
            records.append(record)
            observations.append(observation)

        final_state = attempt_ledger.state()
        if final_state.completed_attempts != len(checked_plan.cells):
            raise PureReferenceRunnerError("attempt ledger does not cover every PURE cell")
        if final_state.tail_ref is None:
            raise PureReferenceRunnerError("completed PURE ledger has no terminal outcome")
        batch = PureReferenceBatch(
            reference_id=checked_plan.reference_id,
            plan=checked_plan,
            observations=tuple(observations),
            execution_refs=tuple(record.execution_ref for record in records),
            outcome_refs=tuple(record.outcome_ref for record in records),
            ledger_tail_ref=final_state.tail_ref,
            all_executions_completed_successfully=all(
                record.execution.status is ExecutionStatus.COMPLETED for record in records
            ),
            terminal_ledger_poisoned=final_state.poisoned,
        )
        batch_ref = self._repository.put_json(batch, media_type=PURE_REFERENCE_BATCH_MEDIA_TYPE)
        checked_ref = ArtifactRef.model_validate(batch_ref, strict=True)
        if checked_ref.media_type != PURE_REFERENCE_BATCH_MEDIA_TYPE:
            raise PureReferenceRunnerError("repository returned the wrong PURE batch media type")
        verified = verify_pure_reference_batch(
            self._repository,
            checked_ref,
            adapter=self._adapter,
            expected_plan=checked_plan,
        )
        if verified != batch:
            raise PureReferenceRunnerError("persisted PURE batch content changed")
        return PureReferenceBatchExecution(
            batch=batch,
            batch_ref=checked_ref,
            executions=tuple(record.execution for record in records),
        )

    def _require_execution_context(
        self,
        plan: PureReferencePlan,
        runner: PureModelRunner,
        ledger: AttemptLedger,
    ) -> None:
        if self._adapter.fingerprint != plan.adapter_fingerprint:
            raise PureReferenceRunnerError("adapter differs from the frozen PURE plan")
        if runner.repository is not self._repository or ledger.repository is not self._repository:
            raise PureReferenceRunnerError("PURE runner and ledger must use the batch repository")
        if runner.spec != plan.spec:
            raise PureReferenceRunnerError("runner does not use the exact frozen model spec")
        if ledger.ledger_id != plan.ledger_id or runner.ledger_id != plan.ledger_id:
            raise PureReferenceRunnerError("attempt ledger differs from the frozen PURE plan")
        if ledger.budget != plan.attempt_budget:
            raise PureReferenceRunnerError("attempt budget differs from the frozen PURE plan")
        runner_state = runner.attempt_state()
        ledger_state = ledger.state()
        if runner_state.writer_epoch_id != ledger_state.writer_epoch_id:
            raise PureReferenceRunnerError("runner and attempt ledger are different writer epochs")
        if ledger_state.attempts_used != 0 or ledger_state.tail_ref is not None:
            raise PureReferenceRunnerError("PURE plan requires a fresh dedicated attempt ledger")

    def _preflight_tasks(self, plan: PureReferencePlan) -> tuple[TaskT, ...]:
        roster = frozenset(self._adapter.task_roster(plan.partition))
        if any(cell.task_id not in roster for cell in plan.cells):
            raise PureReferenceRunnerError("PURE plan includes a task outside its frozen partition")
        tasks = tuple(self._adapter.load_task(cell.task_id) for cell in plan.cells)
        for cell, task in zip(plan.cells, tasks, strict=True):
            cell.require_exact_task(task)
        return tasks


def verify_pure_reference_batch[TaskT](
    repository: ArtifactRepository,
    batch_ref: ArtifactRef,
    *,
    adapter: BenchmarkAdapter[str, TaskT, object],
    expected_plan: PureReferencePlan | object | None = None,
) -> PureReferenceBatch:
    """Replay one published PURE batch and independently regrade every cell.

    This closes the repository-local execution, accounting, task-byte, and
    grading chain.  It intentionally does not attest that a remote provider
    served the declared model and does not establish matched cross-arm study
    closure; the returned batch therefore remains non-reportable.
    """

    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    if not isinstance(adapter, BenchmarkAdapter):
        raise TypeError("adapter must implement BenchmarkAdapter")
    try:
        checked_ref = ArtifactRef.model_validate(batch_ref, strict=True)
    except Exception as exc:
        raise PureReferenceRunnerError("PURE batch reference is malformed") from exc
    if checked_ref.media_type != PURE_REFERENCE_BATCH_MEDIA_TYPE:
        raise PureReferenceRunnerError("PURE batch reference declares the wrong media type")
    batch = _load_model(repository, checked_ref, PureReferenceBatch, "PURE batch")
    plan = batch.plan
    if expected_plan is not None:
        try:
            checked_plan = PureReferencePlan.model_validate(expected_plan, strict=True)
        except Exception as exc:
            raise PureReferenceRunnerError("expected PURE plan is malformed") from exc
        if plan != checked_plan:
            raise PureReferenceRunnerError("published PURE batch belongs to another plan")
    if adapter.fingerprint != plan.adapter_fingerprint:
        raise PureReferenceRunnerError("adapter differs from the published PURE plan")

    roster = frozenset(adapter.task_roster(plan.partition))
    if any(cell.task_id not in roster for cell in plan.cells):
        raise PureReferenceRunnerError("published PURE cell is outside the adapter roster")
    tasks = tuple(adapter.load_task(cell.task_id) for cell in plan.cells)
    for cell, task in zip(plan.cells, tasks, strict=True):
        cell.require_exact_task(task)

    try:
        audit_ledger = AttemptLedger(
            repository,
            ledger_id=plan.ledger_id,
            budget=plan.attempt_budget,
            tail_ref=batch.ledger_tail_ref,
        )
        state = audit_ledger.state()
    except Exception as exc:
        raise PureReferenceRunnerError("PURE attempt ledger cannot be replayed") from exc
    if state.pending_reservation_ref is not None:
        raise PureReferenceRunnerError("PURE attempt ledger ends with an open reservation")
    if state.completed_attempts != len(plan.cells) or state.attempts_used != len(plan.cells):
        raise PureReferenceRunnerError("PURE attempt ledger does not cover the frozen cells")
    if state.poisoned != batch.terminal_ledger_poisoned:
        raise PureReferenceRunnerError("PURE batch terminal poison flag differs from its ledger")

    previous_outcome_ref: ArtifactRef | None = None
    writer_epoch_id: str | None = None
    completed = True
    for index, (cell, task, execution_ref, outcome_ref, observation) in enumerate(
        zip(
            plan.cells,
            tasks,
            batch.execution_refs,
            batch.outcome_refs,
            batch.observations,
            strict=True,
        )
    ):
        try:
            execution = load_accounted_execution(repository, execution_ref)
        except Exception as exc:
            raise PureReferenceRunnerError("PURE execution artifact cannot be verified") from exc
        if type(execution) is not PureReferenceExecution:
            raise PureReferenceRunnerError("PURE batch references a non-PURE execution")
        outcome = _load_model(repository, outcome_ref, AttemptOutcome, "PURE attempt outcome")
        reservation = _load_model(
            repository,
            outcome.reservation_ref,
            AttemptReservation,
            "PURE attempt reservation",
        )

        checked_task = cell.require_exact_task(task)
        expected_request = materialize_pure_request(
            checked_task,
            reference_id=plan.reference_id,
            rollout_seed=cell.rollout_seed,
        )
        expected_execution_fingerprint = paired_execution_fingerprint(
            plan.spec,
            checked_task,
            seed=cell.rollout_seed,
            backend_fingerprint=plan.spec.backend_fingerprint,
        )
        if (
            execution.task != checked_task
            or execution.request != expected_request
            or execution.request_sha256 != expected_request.fingerprint
            or execution.spec != plan.spec
            or execution.execution_fingerprint != expected_execution_fingerprint
            or execution.capabilities != plan.capabilities
        ):
            raise PureReferenceRunnerError("PURE execution differs from its frozen cell")
        if outcome.execution_ref != execution_ref:
            raise PureReferenceRunnerError("PURE outcome does not bind its execution reference")
        if (
            reservation.ledger_id != plan.ledger_id
            or outcome.ledger_id != plan.ledger_id
            or reservation.budget_fingerprint != plan.attempt_budget.fingerprint
            or outcome.budget_fingerprint != plan.attempt_budget.fingerprint
            or reservation.sequence != index
            or outcome.sequence != index
            or reservation.previous_outcome_ref != previous_outcome_ref
            or outcome.reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE
            or reservation.task_fingerprint != checked_task.fingerprint
            or reservation.execution_fingerprint != expected_execution_fingerprint
            or reservation.request_sha256 != expected_request.fingerprint
            or reservation.reserved_tokens != plan.attempt_budget.max_tokens_per_attempt
        ):
            raise PureReferenceRunnerError("PURE reservation/outcome chain differs from its plan")
        if reservation.writer_epoch_id != outcome.writer_epoch_id:
            raise PureReferenceRunnerError("PURE reservation and outcome writer epochs differ")
        if writer_epoch_id is None:
            writer_epoch_id = reservation.writer_epoch_id
        elif writer_epoch_id != reservation.writer_epoch_id:
            raise PureReferenceRunnerError("PURE attempt chain crosses writer epochs")
        if outcome.reported_tokens != execution.usage.total_tokens:
            raise PureReferenceRunnerError("PURE outcome usage differs from its execution")

        if execution.status is ExecutionStatus.COMPLETED:
            if (
                outcome.disposition is not AttemptDisposition.SETTLED
                or outcome.error_class is not None
            ):
                raise PureReferenceRunnerError("completed PURE execution was not settled")
        else:
            completed = False
            expected_disposition = (
                AttemptDisposition.POISONED
                if execution.usage.total_tokens > reservation.reserved_tokens
                else AttemptDisposition.BURNED
            )
            if (
                outcome.disposition is not expected_disposition
                or execution.error is None
                or outcome.error_class != execution.error.error_class.value
            ):
                raise PureReferenceRunnerError("failed PURE execution accounting is inconsistent")

        try:
            regraded = TrialObservation.model_validate(
                adapter.grade(
                    checked_task,
                    execution,
                    harness_id=plan.reference_id,
                    seed=cell.rollout_seed,
                    execution_fingerprint=expected_execution_fingerprint,
                ),
                strict=True,
            )
        except Exception as exc:
            raise PureReferenceRunnerError(
                "PURE observation cannot be independently regraded"
            ) from exc
        if regraded != observation:
            raise PureReferenceRunnerError("PURE observation differs from trusted regrading")
        previous_outcome_ref = outcome_ref

    if completed != batch.all_executions_completed_successfully:
        raise PureReferenceRunnerError("PURE batch completion flag differs from its executions")
    return batch


def _load_model[ModelT](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    try:
        loaded = repository.get_json(ref, model_type)
        validator = model_type.model_validate
        return validator(loaded, strict=True)
    except Exception as exc:
        raise PureReferenceRunnerError(f"{label} artifact cannot be verified") from exc


__all__ = [
    "PURE_REFERENCE_BATCH_MEDIA_TYPE",
    "PURE_REFERENCE_CONTRACT",
    "PureReferenceBatch",
    "PureReferenceBatchExecution",
    "PureReferenceCell",
    "PureReferencePlan",
    "PureReferenceRunnerError",
    "TrustedPureReferenceRunner",
    "verify_pure_reference_batch",
]
