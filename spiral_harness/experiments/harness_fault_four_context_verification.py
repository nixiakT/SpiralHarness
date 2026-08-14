"""Offline verifier for the atomic four-context HarnessFault development run."""

from __future__ import annotations

from spiral_harness.benchmark._harness_fault_cases import (
    HiddenScenarioSpec,
    RepairRuleId,
    VerifiedPartitionOpening,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault_compiler import (
    HarnessRole,
    parse_fault_repair_action,
    publish_faulty_parent_harness,
    verify_fault_compilation,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
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
from spiral_harness.experiments.confirmatory_arms import FaultFactorialCell
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    HarnessFaultDevelopmentError,
    HarnessFaultFeedbackProjection,
    ModelAuthoredFaultProposal,
)
from spiral_harness.experiments.harness_fault_development_support import (
    build_feedback_projection_from_coordinates,
    make_gate_from_coordinates,
    make_observation,
    proposer_task,
    publish_proposer_harness,
    solver_seed,
)
from spiral_harness.experiments.harness_fault_four_context_contracts import (
    HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE,
    HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE,
    HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE,
    FaultContextGateRecord,
    FrozenFaultContextCandidate,
    HarnessFaultFourContextDevelopmentResult,
    HarnessFaultFourContextGateBatch,
    JointFaultContextCandidateFreeze,
    fault_context_candidate_freeze_id,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def verify_model_driven_harness_fault_four_context_development(
    store: ArtifactStore,
    result_ref: ArtifactRef,
) -> HarnessFaultFourContextDevelopmentResult:
    """Replay paired tasks, raw outputs, candidates, gates, and ledger order."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    result = _load_result(store, result_ref)
    persisted_freeze = _load_typed(
        store,
        result.candidate_freeze_ref,
        expected_media_type=HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE,
        model_type=JointFaultContextCandidateFreeze,
        label="candidate freeze",
    )
    if persisted_freeze != result.candidate_freeze:
        raise HarnessFaultDevelopmentError("embedded and persisted candidate freezes differ")
    persisted_gate_batch = _load_typed(
        store,
        result.gate_batch_ref,
        expected_media_type=HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE,
        model_type=HarnessFaultFourContextGateBatch,
        label="gate batch",
    )
    if persisted_gate_batch != result.gate_batch:
        raise HarnessFaultDevelopmentError("embedded and persisted gate batches differ")

    opened_fit, opened_gate = _open_and_verify_paired_blocks(store, result)
    contexts = {context.cell: context for context in result.contexts}
    materializer = HarnessMaterializer(store, spec=result.plan.model_spec)
    expected_parent_ref = publish_faulty_parent_harness(store, result.plan.model_spec)
    expected_proposer_ref = publish_proposer_harness(store, result.plan.model_spec)
    expected_candidates = []
    expected_gates = []
    ordered_definition_outcomes: list[ArtifactRef] = []
    ordered_attribution_outcomes: list[ArtifactRef] = []

    for block in result.plan.blocks:
        context = contexts[block.cell]
        fit_scenarios = tuple(
            sorted(opened_fit[block.cell].scenarios, key=lambda item: item.task.task_id)
        )
        gate_scenarios = tuple(
            sorted(opened_gate[block.cell].scenarios, key=lambda item: item.task.task_id)
        )
        compilation = verify_fault_compilation(
            store,
            result.plan.model_spec,
            context.compilation_ref,
            expected_runtime_producer_id=context.runtime_producer_id,
        )
        expected_runtime_producer_id = canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-direct-runtime/v1",
                "four_context_plan_fingerprint": result.plan_fingerprint,
                "cell": block.cell,
                "block_id": block.block_id,
                "behavior_rewrite": False,
            }
        )
        if context.runtime_producer_id != expected_runtime_producer_id:
            raise HarnessFaultDevelopmentError("context runtime producer is not plan-bound")
        if (
            context.parent_harness_ref != expected_parent_ref
            or compilation.entry(HarnessRole.FAULTY_PARENT).harness_ref != expected_parent_ref
        ):
            raise HarnessFaultDevelopmentError("context parent is not the exact frozen parent")
        if context.proposer_harness_ref != expected_proposer_ref:
            raise HarnessFaultDevelopmentError("context proposer is not the exact frozen proposer")

        fit_parent_by_task = {
            observation.task_id: observation for observation in context.fit_parent_observations
        }
        for scenario in fit_scenarios:
            try:
                observation = fit_parent_by_task[scenario.task.task_id]
            except KeyError as exc:  # pragma: no cover - result shape validator
                raise HarnessFaultDevelopmentError("fit parent observation is missing") from exc
            if observation.harness_role is not HarnessRole.FAULTY_PARENT:
                raise HarnessFaultDevelopmentError("fit feedback contains a non-parent role")
            _verify_observation(
                store=store,
                materializer=materializer,
                model_spec=result.plan.model_spec,
                scenario=scenario,
                observation=observation,
                expected_harness_ref=compilation.entry(HarnessRole.FAULTY_PARENT).harness_ref,
                solver_master_seed=block.solver_master_seed,
            )
            ordered_definition_outcomes.append(observation.outcome_ref)
        if len(fit_parent_by_task) != block.fit_task_count:
            raise HarnessFaultDevelopmentError("fit parent observations differ from fit roster")

        parent = context.fit_parent_observations
        expected_feedback = build_feedback_projection_from_coordinates(
            plan_fingerprint=block.fit_stage_fingerprint,
            feedback_linkage=block.feedback_linkage,
            task_count=block.fit_task_count,
            parent=parent,
        )
        feedback = _load_typed(
            store,
            context.feedback_ref,
            expected_media_type=HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
            model_type=HarnessFaultFeedbackProjection,
            label="context feedback",
        )
        if feedback != expected_feedback:
            raise HarnessFaultDevelopmentError("feedback is not the exact context-local projection")
        expected_proposal = _verify_proposal(
            store=store,
            materializer=materializer,
            model_spec=result.plan.model_spec,
            feedback=feedback,
            proposer_harness_ref=context.proposer_harness_ref,
            proposal=context.proposal,
            proposal_seed=block.proposal_seed,
        )
        if context.proposal != expected_proposal:
            raise HarnessFaultDevelopmentError("proposal differs from its raw model output")
        if compilation.selected_rule_id is not expected_proposal.applied_rule_id:
            raise HarnessFaultDevelopmentError("compiled candidate is not the model proposal")
        ordered_definition_outcomes.append(context.proposal.outcome_ref)

        gate_observations = {
            (observation.harness_role, observation.task_id): observation
            for observation in context.gate_observations
        }
        for role in (
            HarnessRole.FAULTY_PARENT,
            HarnessRole.CANDIDATE,
            HarnessRole.REVERT,
            HarnessRole.PLACEBO,
        ):
            for scenario in gate_scenarios:
                key = (role, scenario.task.task_id)
                try:
                    observation = gate_observations[key]
                except KeyError as exc:  # pragma: no cover - result shape validator
                    raise HarnessFaultDevelopmentError(
                        "context observation cell is missing"
                    ) from exc
                _verify_observation(
                    store=store,
                    materializer=materializer,
                    model_spec=result.plan.model_spec,
                    scenario=scenario,
                    observation=observation,
                    expected_harness_ref=compilation.entry(role).harness_ref,
                    solver_master_seed=block.solver_master_seed,
                )
                ordered_attribution_outcomes.append(observation.outcome_ref)
        if len(gate_observations) != 4 * block.gate_task_count:
            raise HarnessFaultDevelopmentError(
                "context gate observations differ from the disjoint gate roster"
            )

        candidate_ref = compilation.entry(HarnessRole.CANDIDATE).harness_ref
        expected_candidate = FrozenFaultContextCandidate(
            cell=block.cell,
            block_id=block.block_id,
            plan_fingerprint=block.fingerprint,
            feedback_ref=context.feedback_ref,
            proposer_harness_ref=context.proposer_harness_ref,
            proposal=context.proposal,
            runtime_producer_id=context.runtime_producer_id,
            parent_harness_ref=context.parent_harness_ref,
            compilation_ref=context.compilation_ref,
            candidate_harness_ref=candidate_ref,
        )
        expected_candidates.append(expected_candidate)
        expected_gate = make_gate_from_coordinates(
            promotion_rule=block.promotion_rule,
            task_count=block.gate_task_count,
            proposal=context.proposal,
            observations=context.gate_observations,
            parent_harness_ref=context.parent_harness_ref,
            candidate_harness_ref=candidate_ref,
            resource_budget_passed=True,
        )
        if context.gate != expected_gate:
            raise HarnessFaultDevelopmentError("context gate differs from the frozen rule")
        expected_gates.append(
            FaultContextGateRecord(
                cell=block.cell,
                block_id=block.block_id,
                gate=expected_gate,
            )
        )

    expected_freeze = JointFaultContextCandidateFreeze(
        plan_fingerprint=result.plan_fingerprint,
        freeze_id=fault_context_candidate_freeze_id(
            plan_fingerprint=result.plan_fingerprint,
            candidates=tuple(expected_candidates),
        ),
        candidates=tuple(expected_candidates),
        candidate_definition_call_count=result.plan.candidate_definition_call_count,
        last_candidate_definition_sequence=result.plan.candidate_definition_call_count - 1,
        first_attribution_sequence=result.plan.candidate_definition_call_count,
    )
    if result.candidate_freeze != expected_freeze:
        raise HarnessFaultDevelopmentError("joint freeze does not bind the four raw proposals")
    expected_gate_batch = HarnessFaultFourContextGateBatch(
        plan_fingerprint=result.plan_fingerprint,
        candidate_freeze_ref=result.candidate_freeze_ref,
        candidate_freeze_id=expected_freeze.freeze_id,
        records=tuple(expected_gates),
        model_call_count_before_gate_batch=result.model_call_count,
    )
    if result.gate_batch != expected_gate_batch:
        raise HarnessFaultDevelopmentError("cross-context gate batch is not reproducible")

    expected_ledger_id = "harness-fault-four-context-" + result.plan_fingerprint[:24]
    if result.ledger_id != expected_ledger_id:
        raise HarnessFaultDevelopmentError("ledger ID is not bound to the four-context plan")
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
        raise HarnessFaultDevelopmentError("global attempt budget does not close exactly")
    expected_order = tuple(ordered_definition_outcomes + ordered_attribution_outcomes)
    actual_order = _ledger_outcomes_in_sequence(store, result.ledger_tail_ref)
    if actual_order != expected_order:
        raise HarnessFaultDevelopmentError(
            "ledger order violates joint candidate freeze or attribution topology"
        )
    if len(actual_order) != result.model_call_count:
        raise HarnessFaultDevelopmentError("ledger chain differs from the declared call count")
    return result


def _load_result(
    store: ArtifactStore,
    result_ref: ArtifactRef,
) -> HarnessFaultFourContextDevelopmentResult:
    return _load_typed(
        store,
        result_ref,
        expected_media_type=HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE,
        model_type=HarnessFaultFourContextDevelopmentResult,
        label="four-context result",
    )


def _load_typed(store, ref, *, expected_media_type, model_type, label):
    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
        if checked_ref.media_type != expected_media_type:
            raise ValueError("wrong media type")
        loaded = store.get_json(checked_ref, model_type)
        return model_type.model_validate(loaded, strict=True)
    except Exception as exc:
        raise HarnessFaultDevelopmentError(f"{label} cannot be verified") from exc


def _open_and_verify_paired_blocks(
    store: ArtifactStore,
    result: HarnessFaultFourContextDevelopmentResult,
) -> tuple[
    dict[FaultFactorialCell, VerifiedPartitionOpening],
    dict[FaultFactorialCell, VerifiedPartitionOpening],
]:
    try:
        opened_fit = {}
        opened_gate = {}
        fit_anchor = None
        gate_anchor = None
        for block in result.plan.blocks:
            current_fit = verify_partition_opening(store, block.fit_partition_grant)
            current_gate = verify_partition_opening(store, block.gate_partition_grant)
            if fit_anchor is None:
                fit_anchor = current_fit
            elif current_fit != fit_anchor:
                raise ValueError("context fit roster differs from paired fit block")
            if gate_anchor is None:
                gate_anchor = current_gate
            elif current_gate != gate_anchor:
                raise ValueError("context gate roster differs from paired gate block")
            _require_fit_gate_disjoint(current_fit, current_gate)
            opened_fit[block.cell] = current_fit
            opened_gate[block.cell] = current_gate
    except Exception as exc:
        raise HarnessFaultDevelopmentError(
            "paired disjoint fit/gate authority blocks cannot be replayed"
        ) from exc
    for block in result.plan.blocks:
        fit_scenarios: tuple[HiddenScenarioSpec, ...] = opened_fit[block.cell].scenarios
        gate_scenarios: tuple[HiddenScenarioSpec, ...] = opened_gate[block.cell].scenarios
        if len(fit_scenarios) != block.fit_task_count:
            raise HarnessFaultDevelopmentError("paired fit block differs from fit_task_count")
        if len(gate_scenarios) != block.gate_task_count:
            raise HarnessFaultDevelopmentError("paired gate block differs from gate_task_count")
    return opened_fit, opened_gate


def _require_fit_gate_disjoint(
    fit: VerifiedPartitionOpening,
    gate: VerifiedPartitionOpening,
) -> None:
    fit_scenarios = fit.scenarios
    gate_scenarios = gate.scenarios
    axes = (
        (
            {item.task.task_id for item in fit_scenarios},
            {item.task.task_id for item in gate_scenarios},
        ),
        (
            {item.scenario_id for item in fit_scenarios},
            {item.scenario_id for item in gate_scenarios},
        ),
        (
            {item.scenario_commitment for item in fit_scenarios},
            {item.scenario_commitment for item in gate_scenarios},
        ),
        ({item.source_id for item in fit_scenarios}, {item.source_id for item in gate_scenarios}),
        ({item.group_id for item in fit_scenarios}, {item.group_id for item in gate_scenarios}),
        (
            {(item.family, item.source_id) for item in fit_scenarios},
            {(item.family, item.source_id) for item in gate_scenarios},
        ),
        (
            {(item.family, item.group_id) for item in fit_scenarios},
            {(item.family, item.group_id) for item in gate_scenarios},
        ),
    )
    if any(fit_values.intersection(gate_values) for fit_values, gate_values in axes):
        raise ValueError("fit and gate blocks overlap")
    fit_strata = sorted(
        (item.family.value, item.surface.value, item.role.value) for item in fit_scenarios
    )
    gate_strata = sorted(
        (item.family.value, item.surface.value, item.role.value) for item in gate_scenarios
    )
    if fit_strata != gate_strata:
        raise ValueError("fit and gate blocks have unmatched strata")


def _verify_observation(
    *,
    store: ArtifactStore,
    materializer: HarnessMaterializer,
    model_spec,
    scenario: HiddenScenarioSpec,
    observation,
    expected_harness_ref: ArtifactRef,
    solver_master_seed: int,
) -> None:
    record = _load_call(store, observation.execution_ref, observation.outcome_ref)
    execution = record.execution
    expected_seed = solver_seed(solver_master_seed, scenario.task.task_id)
    if (
        execution.spec != model_spec
        or execution.task != CandidateTask.from_task_view(scenario.task)
        or execution.request.harness_ref != expected_harness_ref
        or execution.seed != expected_seed
    ):
        raise HarnessFaultDevelopmentError("observation execution provenance differs")
    materializer.verify_execution_request(expected_harness_ref, execution)
    expected = make_observation(
        scenario=scenario,
        role=observation.harness_role,
        seed=expected_seed,
        record=record,
    )
    if observation != expected:
        raise HarnessFaultDevelopmentError("observation differs from its raw model output")


def _verify_proposal(
    *,
    store: ArtifactStore,
    materializer: HarnessMaterializer,
    model_spec,
    feedback: HarnessFaultFeedbackProjection,
    proposer_harness_ref: ArtifactRef,
    proposal: ModelAuthoredFaultProposal,
    proposal_seed: int,
) -> ModelAuthoredFaultProposal:
    record = _load_call(store, proposal.execution_ref, proposal.outcome_ref)
    execution = record.execution
    if (
        execution.spec != model_spec
        or execution.task != proposer_task(feedback)
        or execution.request.harness_ref != proposer_harness_ref
        or execution.seed != proposal_seed
    ):
        raise HarnessFaultDevelopmentError("proposal execution provenance differs")
    materializer.verify_execution_request(proposer_harness_ref, execution)
    parsed = None
    output_hash = None
    if execution.status is ExecutionStatus.COMPLETED:
        output = execution.output_text
        if output is None:  # pragma: no cover - execution invariant
            raise HarnessFaultDevelopmentError("completed proposal omitted output")
        output_hash = sha256_bytes(output.encode("utf-8"))
        try:
            parsed = parse_fault_repair_action(output)
        except ValueError:
            parsed = None
    applied_rule = parsed.rule_id if parsed is not None else RepairRuleId.CONSTANT_LEGACY
    return ModelAuthoredFaultProposal(
        feedback_ref=proposal.feedback_ref,
        proposer_harness_ref=proposer_harness_ref,
        execution_ref=record.execution_ref,
        outcome_ref=record.outcome_ref,
        execution_status=execution.status,
        raw_model_output_sha256=output_hash,
        parsed_action=parsed,
        proposal_valid=parsed is not None,
        applied_rule_id=applied_rule,
    )


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
    disposition = (
        AttemptDisposition.SETTLED
        if execution.status is ExecutionStatus.COMPLETED
        else AttemptDisposition.BURNED
    )
    if outcome.execution_ref != execution_ref or outcome.disposition is not disposition:
        raise HarnessFaultDevelopmentError("execution and attempt outcome do not close")
    return ModelExecutionRecord(
        execution=execution,
        execution_ref=execution_ref,
        outcome_ref=outcome_ref,
    )


def _ledger_outcomes_in_sequence(
    store: ArtifactStore,
    tail_ref: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    reversed_refs: list[ArtifactRef] = []
    cursor = tail_ref
    seen: set[str] = set()
    while True:
        if cursor.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE or cursor.sha256 in seen:
            raise HarnessFaultDevelopmentError("global ledger outcome chain is malformed")
        seen.add(cursor.sha256)
        reversed_refs.append(cursor)
        outcome = store.get_json(cursor, AttemptOutcome)
        reservation_ref = outcome.reservation_ref
        if reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise HarnessFaultDevelopmentError("global ledger reservation link has wrong media")
        reservation = store.get_json(reservation_ref, AttemptReservation)
        if reservation.sequence != outcome.sequence:
            raise HarnessFaultDevelopmentError("ledger reservation/outcome sequence differs")
        previous = reservation.previous_outcome_ref
        if previous is None:
            break
        cursor = previous
    ordered = tuple(reversed(reversed_refs))
    for sequence, ref in enumerate(ordered):
        outcome = store.get_json(ref, AttemptOutcome)
        if outcome.sequence != sequence:
            raise HarnessFaultDevelopmentError("global ledger sequence is not contiguous")
    return ordered


__all__ = ["verify_model_driven_harness_fault_four_context_development"]
