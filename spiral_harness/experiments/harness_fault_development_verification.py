"""Offline fail-closed verifier for model-driven HarnessFault development."""

from __future__ import annotations

from spiral_harness.benchmark._harness_fault_cases import verify_partition_opening
from spiral_harness.benchmark.harness_fault_compiler import (
    HarnessRole,
    parse_fault_repair_action,
    verify_fault_compilation,
)
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import ArtifactRef
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
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE,
    HarnessFaultDevelopmentError,
    HarnessFaultFeedbackProjection,
    HarnessFaultModelDrivenDevelopmentResult,
)
from spiral_harness.experiments.harness_fault_development_support import (
    build_feedback_projection,
    make_gate,
    make_observation,
    proposer_task,
    solver_seed,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def verify_model_driven_harness_fault_development(
    store: ArtifactStore,
    result_ref: ArtifactRef,
) -> HarnessFaultModelDrivenDevelopmentResult:
    """Replay proposal, raw behavior, gate, and the complete accounting chain."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    try:
        checked_ref = ArtifactRef.model_validate(result_ref, strict=True)
        if checked_ref.media_type != HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE:
            raise ValueError("wrong result media type")
        result = store.get_json(checked_ref, HarnessFaultModelDrivenDevelopmentResult)
        result = HarnessFaultModelDrivenDevelopmentResult.model_validate(result, strict=True)
    except Exception as exc:
        raise HarnessFaultDevelopmentError("development result cannot be verified") from exc

    verified = verify_partition_opening(store, result.partition_grant)
    scenarios = {item.task.task_id: item for item in verified.scenarios}
    if len(scenarios) != result.plan.task_count:
        raise HarnessFaultDevelopmentError("authority roster differs from the frozen plan")
    compilation = verify_fault_compilation(
        store,
        result.plan.model_spec,
        result.compilation_ref,
        expected_runtime_producer_id=result.runtime_producer_id,
    )
    if compilation.entry(HarnessRole.FAULTY_PARENT).harness_ref != result.parent_harness_ref:
        raise HarnessFaultDevelopmentError("compiled and observed parents differ")
    materializer = HarnessMaterializer(store, spec=result.plan.model_spec)

    observations_by_key = {}
    for observation in result.observations:
        try:
            scenario = scenarios[observation.task_id]
        except KeyError as exc:
            raise HarnessFaultDevelopmentError("observation task is outside the roster") from exc
        expected_harness = compilation.entry(observation.harness_role).harness_ref
        record = _load_call(store, observation.execution_ref, observation.outcome_ref)
        execution = record.execution
        if (
            execution.spec != result.plan.model_spec
            or execution.task != CandidateTask.from_task_view(scenario.task)
            or execution.request.harness_ref != expected_harness
            or execution.seed != solver_seed(result.plan.solver_master_seed, scenario.task.task_id)
        ):
            raise HarnessFaultDevelopmentError("observation execution provenance differs")
        materializer.verify_execution_request(expected_harness, execution)
        expected = make_observation(
            scenario=scenario,
            role=observation.harness_role,
            seed=execution.seed,
            record=record,
        )
        if observation != expected:
            raise HarnessFaultDevelopmentError("observation differs from its raw model output")
        key = (observation.harness_role, observation.task_id)
        if key in observations_by_key:
            raise HarnessFaultDevelopmentError("duplicate observation role/task cell")
        observations_by_key[key] = observation

    parent = tuple(
        item for item in result.observations if item.harness_role is HarnessRole.FAULTY_PARENT
    )
    expected_feedback = build_feedback_projection(result.plan, parent)
    feedback = _load_feedback(store, result.feedback_ref)
    if feedback != expected_feedback:
        raise HarnessFaultDevelopmentError("feedback differs from complete parent outputs")

    proposal_record = _load_call(
        store,
        result.proposal.execution_ref,
        result.proposal.outcome_ref,
    )
    proposal_execution = proposal_record.execution
    expected_proposer_task = proposer_task(feedback)
    if (
        proposal_execution.spec != result.plan.model_spec
        or proposal_execution.task != expected_proposer_task
        or proposal_execution.request.harness_ref != result.proposer_harness_ref
        or proposal_execution.seed != result.plan.proposal_seed
        or proposal_execution.status is not result.proposal.execution_status
    ):
        raise HarnessFaultDevelopmentError("proposal execution provenance differs")
    materializer.verify_execution_request(result.proposer_harness_ref, proposal_execution)
    parsed = None
    output_hash = None
    if proposal_execution.status is ExecutionStatus.COMPLETED:
        output = proposal_execution.output_text
        if output is None:  # pragma: no cover - execution invariant
            raise HarnessFaultDevelopmentError("completed proposal omitted output")
        output_hash = sha256_bytes(output.encode("utf-8"))
        try:
            parsed = parse_fault_repair_action(output)
        except ValueError:
            parsed = None
    if (
        result.proposal.raw_model_output_sha256 != output_hash
        or result.proposal.parsed_action != parsed
        or result.proposal.proposal_valid is not (parsed is not None)
        or compilation.selected_rule_id is not result.proposal.applied_rule_id
    ):
        raise HarnessFaultDevelopmentError("compiled candidate is not the sole model proposal")

    expected_gate = make_gate(
        plan=result.plan,
        proposal=result.proposal,
        observations=result.observations,
        parent_harness_ref=result.parent_harness_ref,
        candidate_harness_ref=compilation.entry(HarnessRole.CANDIDATE).harness_ref,
        resource_budget_passed=True,
    )
    if result.gate != expected_gate:
        raise HarnessFaultDevelopmentError("promotion differs from the automatic frozen gate")

    audit = AttemptLedger(
        store,
        ledger_id=result.ledger_id,
        budget=result.attempt_budget,
        tail_ref=result.ledger_tail_ref,
    ).state()
    if (
        audit.attempts_used != result.model_call_count
        or audit.completed_attempts != result.model_call_count
        or audit.pending_reservation_ref is not None
        or audit.poisoned
        or audit.charged_tokens > result.plan.max_total_tokens
    ):
        raise HarnessFaultDevelopmentError("attempt budget does not close exactly")
    declared_outcomes = {result.proposal.outcome_ref.sha256} | {
        item.outcome_ref.sha256 for item in result.observations
    }
    if _ledger_outcome_hashes(store, result.ledger_tail_ref) != declared_outcomes:
        raise HarnessFaultDevelopmentError("result calls differ from the exact ledger chain")
    return result


def _load_feedback(store: ArtifactStore, ref: ArtifactRef) -> HarnessFaultFeedbackProjection:
    try:
        if ref.media_type != HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE:
            raise ValueError("wrong feedback media")
        loaded = store.get_json(ref, HarnessFaultFeedbackProjection)
        return HarnessFaultFeedbackProjection.model_validate(loaded, strict=True)
    except Exception as exc:
        raise HarnessFaultDevelopmentError("feedback artifact cannot be verified") from exc


def _load_call(
    store: ArtifactStore,
    execution_ref: ArtifactRef,
    outcome_ref: ArtifactRef,
) -> ModelExecutionRecord:
    try:
        if execution_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise ValueError("wrong execution media")
        if outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("wrong outcome media")
        execution = store.get_json(execution_ref, ModelExecution)
        outcome = store.get_json(outcome_ref, AttemptOutcome)
        execution = ModelExecution.model_validate(execution, strict=True)
        outcome = AttemptOutcome.model_validate(outcome, strict=True)
    except Exception as exc:
        raise HarnessFaultDevelopmentError("model call artifacts cannot be verified") from exc
    expected_disposition = (
        AttemptDisposition.SETTLED
        if execution.status is ExecutionStatus.COMPLETED
        else AttemptDisposition.BURNED
    )
    if outcome.execution_ref != execution_ref or outcome.disposition is not expected_disposition:
        raise HarnessFaultDevelopmentError("execution and attempt outcome do not close")
    return ModelExecutionRecord(
        execution=execution,
        execution_ref=execution_ref,
        outcome_ref=outcome_ref,
    )


def _ledger_outcome_hashes(store: ArtifactStore, tail: ArtifactRef) -> set[str]:
    values: set[str] = set()
    cursor = tail
    while True:
        if cursor.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE or cursor.sha256 in values:
            raise HarnessFaultDevelopmentError("ledger outcome chain is malformed")
        values.add(cursor.sha256)
        outcome = store.get_json(cursor, AttemptOutcome)
        reservation_ref = outcome.reservation_ref
        if reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise HarnessFaultDevelopmentError("ledger reservation link has wrong media")
        reservation = store.get_json(reservation_ref, AttemptReservation)
        previous = reservation.previous_outcome_ref
        if previous is None:
            return values
        cursor = previous


__all__ = ["verify_model_driven_harness_fault_development"]
