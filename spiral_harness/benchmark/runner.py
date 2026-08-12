"""Trusted fixed-model benchmark batch execution."""

from __future__ import annotations

from dataclasses import dataclass

from spiral_harness.benchmark.base import BenchmarkAdapter
from spiral_harness.core.experiment import ProtocolManifest, ProtocolPartition
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.model import FixedModelRunner
from spiral_harness.execution.receipts import (
    ScheduledExecutionRecord,
    TrustedExecutionUsage,
    execute_scheduled_attempt,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateTrialArm,
    GateTrialBatch,
    TrustedGateBatchService,
)
from spiral_harness.verification.models import TrialObservation


class BenchmarkBatchRunnerError(RuntimeError):
    """Raised when a trusted benchmark batch cannot be executed exactly."""


@dataclass(frozen=True, slots=True)
class BenchmarkBatchExecution:
    """Published artifacts and usage for one complete paired benchmark schedule."""

    preflight_ref: ArtifactRef
    parent_batch_ref: ArtifactRef
    candidate_batch_ref: ArtifactRef
    parent_batch: GateTrialBatch
    candidate_batch: GateTrialBatch
    usage: TrustedExecutionUsage
    receipt_refs: tuple[ArtifactRef, ...]
    execution_refs: tuple[ArtifactRef, ...]
    outcome_refs: tuple[ArtifactRef, ...]


class TrustedBenchmarkBatchRunner[TaskT]:
    """Run a frozen schedule while keeping adapter grading in the trusted plane."""

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
        self._adapter = adapter
        self._gate_batches = gate_batch_service

    def execute_paired_batch(
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
        runner: FixedModelRunner,
        attempt_ledger: AttemptLedger,
        source_refs: tuple[ArtifactRef, ...] = (),
    ) -> BenchmarkBatchExecution:
        """Execute, grade, sign, and persist both arms of one frozen schedule."""

        if type(runner) is not FixedModelRunner:
            raise TypeError("runner must be an exact FixedModelRunner")
        if type(attempt_ledger) is not AttemptLedger:
            raise TypeError("attempt_ledger must be an exact AttemptLedger")
        if runner.repository is not self._repository:
            raise BenchmarkBatchRunnerError("runner must use the benchmark repository")
        if attempt_ledger.repository is not self._repository:
            raise BenchmarkBatchRunnerError("attempt ledger must use the benchmark repository")
        if runner.attempt_state().ledger_id != attempt_ledger.ledger_id:
            raise BenchmarkBatchRunnerError("runner and attempt_ledger differ")

        checked_protocol = ProtocolManifest.model_validate(protocol, strict=True)
        checked_schedule = EvaluationBatchSchedule.model_validate(schedule, strict=True)
        self._require_gate_protocol_matches_runner(
            protocol=checked_protocol,
            schedule=checked_schedule,
            runner=runner,
            gate_split_ref=ArtifactRef.model_validate(gate_split_ref, strict=True),
        )
        parent_ref = ArtifactRef.model_validate(parent_harness_ref, strict=True)
        child_ref = ArtifactRef.model_validate(candidate_harness_ref, strict=True)
        if parent_ref.sha256 != checked_schedule.parent_harness_id:
            raise BenchmarkBatchRunnerError("parent_harness_ref does not match schedule")
        if child_ref.sha256 != checked_schedule.candidate_harness_id:
            raise BenchmarkBatchRunnerError("candidate_harness_ref does not match schedule")

        preflight_ref = publish_schedule_preflight(
            self._repository,
            preflight_attempt_budget(checked_schedule, attempt_ledger, runner.spec),
        )
        records: list[ScheduledExecutionRecord] = []
        observations: dict[EvaluationSide, list[TrialObservation]] = {
            EvaluationSide.PARENT: [],
            EvaluationSide.CANDIDATE: [],
        }
        expected_tail = attempt_ledger.tail_ref
        previous_receipt_ref: ArtifactRef | None = None
        for cell in checked_schedule.iter_cells():
            harness_ref = parent_ref if cell.side is EvaluationSide.PARENT else child_ref
            task = self._adapter.load_task(cell.task_id)
            record = execute_scheduled_attempt(
                runner=runner,
                schedule=checked_schedule,
                preflight_ref=preflight_ref,
                expected_previous_ledger_tail_ref=expected_tail,
                previous_receipt_ref=previous_receipt_ref,
                cell=cell,
                attempt_index=0,
                task=task,
                harness_ref=harness_ref,
            )
            seed = checked_schedule.seed_for(cell, attempt_index=0)
            observations[cell.side].append(
                self._adapter.grade(
                    task,
                    record.execution,
                    harness_id=harness_ref.sha256,
                    seed=seed,
                    execution_fingerprint=record.execution.execution_fingerprint,
                )
            )
            records.append(record)
            expected_tail = record.outcome_ref
            previous_receipt_ref = record.receipt_ref

        receipt_refs = tuple(record.receipt_ref for record in records)
        usage = replay_trusted_usage(
            self._repository,
            schedule=checked_schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=attempt_ledger,
            receipt_refs=receipt_refs,
        )
        if usage.attempt_count != checked_schedule.cell_count:
            raise BenchmarkBatchRunnerError("trusted usage does not cover the schedule")
        parent_batch, parent_batch_ref = self._publish_batch(
            protocol_ref=protocol_ref,
            protocol=checked_protocol,
            candidate_ref=candidate_ref,
            arm=GateTrialArm.PARENT,
            harness_ref=parent_ref,
            gate_split_ref=gate_split_ref,
            mechanism_evidence_ref=mechanism_evidence_ref,
            source_refs=(*source_refs, *receipt_refs),
            observations=tuple(observations[EvaluationSide.PARENT]),
        )
        candidate_batch, candidate_batch_ref = self._publish_batch(
            protocol_ref=protocol_ref,
            protocol=checked_protocol,
            candidate_ref=candidate_ref,
            arm=GateTrialArm.CANDIDATE,
            harness_ref=child_ref,
            gate_split_ref=gate_split_ref,
            mechanism_evidence_ref=mechanism_evidence_ref,
            source_refs=(*source_refs, *receipt_refs),
            observations=tuple(observations[EvaluationSide.CANDIDATE]),
        )
        return BenchmarkBatchExecution(
            preflight_ref=preflight_ref,
            parent_batch_ref=parent_batch_ref,
            candidate_batch_ref=candidate_batch_ref,
            parent_batch=parent_batch,
            candidate_batch=candidate_batch,
            usage=usage,
            receipt_refs=receipt_refs,
            execution_refs=tuple(record.execution_ref for record in records),
            outcome_refs=tuple(record.outcome_ref for record in records),
        )

    def _require_gate_protocol_matches_runner(
        self,
        *,
        protocol: ProtocolManifest,
        schedule: EvaluationBatchSchedule,
        runner: FixedModelRunner,
        gate_split_ref: ArtifactRef,
    ) -> None:
        if schedule.phase is not EvaluationPhase.GATE:
            raise BenchmarkBatchRunnerError(
                "trusted benchmark batch runner requires a GATE schedule"
            )

        expected = {
            "benchmark_fingerprint": self._adapter.fingerprint,
            "grader_fingerprint": self._adapter.fingerprint,
            "model_fingerprint": runner.spec.model_fingerprint,
            "inference_fingerprint": runner.spec.inference_fingerprint,
            "runtime_fingerprint": runner.spec.runtime_fingerprint,
            "model_spec_fingerprint": runner.spec.fingerprint,
        }
        mismatched = tuple(
            field_name
            for field_name, expected_value in expected.items()
            if getattr(protocol, field_name) != expected_value
        )
        if mismatched:
            raise BenchmarkBatchRunnerError(
                "protocol executor identity mismatch: " + ", ".join(mismatched)
            )
        if protocol.gate_batch_attestor_id != self._gate_batches.attestor_id:
            raise BenchmarkBatchRunnerError("protocol gate batch attestor differs from runner")

        protocol_gate_split_ref = next(
            split.manifest_ref
            for split in protocol.splits
            if split.partition is ProtocolPartition.GATE
        )
        if gate_split_ref != protocol_gate_split_ref:
            raise BenchmarkBatchRunnerError("gate_split_ref differs from the protocol GATE split")

        gate_roster = frozenset(self._adapter.task_roster(ProtocolPartition.GATE))
        unknown_task_ids = tuple(
            task_id for task_id in schedule.task_ids if task_id not in gate_roster
        )
        if unknown_task_ids:
            raise BenchmarkBatchRunnerError(
                "schedule includes tasks outside the trusted GATE roster: "
                + ", ".join(unknown_task_ids)
            )

    def _publish_batch(
        self,
        *,
        protocol_ref: ArtifactRef,
        protocol: ProtocolManifest,
        candidate_ref: ArtifactRef,
        arm: GateTrialArm,
        harness_ref: ArtifactRef,
        gate_split_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
        source_refs: tuple[ArtifactRef, ...],
        observations: tuple[TrialObservation, ...],
    ) -> tuple[GateTrialBatch, ArtifactRef]:
        if not observations:
            raise BenchmarkBatchRunnerError("cannot publish an empty benchmark arm")
        batch = self._gate_batches.create(
            protocol_ref=ArtifactRef.model_validate(protocol_ref, strict=True),
            protocol=protocol,
            candidate_ref=ArtifactRef.model_validate(candidate_ref, strict=True),
            arm=arm,
            harness_ref=harness_ref,
            gate_split_ref=ArtifactRef.model_validate(gate_split_ref, strict=True),
            mechanism_evidence_ref=ArtifactRef.model_validate(
                mechanism_evidence_ref,
                strict=True,
            ),
            source_refs=source_refs,
            observations=observations,
        )
        batch_ref = self._repository.put_json(batch, media_type=GATE_TRIAL_BATCH_MEDIA_TYPE)
        return batch, batch_ref


__all__ = [
    "BenchmarkBatchExecution",
    "BenchmarkBatchRunnerError",
    "TrustedBenchmarkBatchRunner",
]
