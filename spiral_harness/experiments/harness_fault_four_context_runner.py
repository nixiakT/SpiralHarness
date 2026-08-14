"""Atomic SS/MS/SM/MM runner for model-output-driven HarnessFault development."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from spiral_harness.benchmark._harness_fault_cases import (
    RepairRuleId,
    VerifiedPartitionOpening,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault_compiler import (
    FaultRepairAction,
    HarnessRole,
    compile_fault_repair,
    parse_fault_repair_action,
    publish_faulty_parent_harness,
    verify_fault_compilation,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import ExecutionStatus
from spiral_harness.execution.materialization import HarnessMaterializer
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.experiments.confirmatory_arms import FaultFactorialCell
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    HarnessFaultDevelopmentError,
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
    FAULT_CONTEXT_ORDER,
    HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE,
    HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE,
    HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE,
    FaultContextGateRecord,
    FrozenFaultContextCandidate,
    HarnessFaultContextDevelopmentClosure,
    HarnessFaultFourContextBlock,
    HarnessFaultFourContextDevelopmentResult,
    HarnessFaultFourContextGateBatch,
    HarnessFaultFourContextPlan,
    JointFaultContextCandidateFreeze,
    fault_context_candidate_freeze_id,
)
from spiral_harness.storage.artifact_store import ArtifactStore


@dataclass(frozen=True, slots=True)
class HarnessFaultFourContextRunRecord:
    result: HarnessFaultFourContextDevelopmentResult
    result_ref: ArtifactRef


@dataclass(slots=True)
class _PreparedContext:
    block: HarnessFaultFourContextBlock
    verified_fit: VerifiedPartitionOpening
    runner: FixedModelRunner
    parent_harness_ref: ArtifactRef
    feedback_ref: ArtifactRef
    proposer_harness_ref: ArtifactRef
    proposal: ModelAuthoredFaultProposal
    runtime_producer_id: str
    compilation_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    fit_parent_observations: tuple
    gate_observations: list


def run_model_driven_harness_fault_four_context_development(
    *,
    store: ArtifactStore,
    context_backends: Mapping[FaultFactorialCell, ModelBackend],
    plan: HarnessFaultFourContextPlan,
) -> HarnessFaultFourContextRunRecord:
    """Fit four candidates, freeze jointly, then open one disjoint gate block."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    checked_plan = HarnessFaultFourContextPlan.model_validate(plan, strict=True)
    backends = _validate_context_backends(context_backends, checked_plan)
    verified_fit = _open_paired_fit_block(store, checked_plan)

    ledger_id = "harness-fault-four-context-" + checked_plan.fingerprint[:24]
    ledger = AttemptLedger(
        store,
        ledger_id=ledger_id,
        budget=checked_plan.attempt_budget,
    )
    materializer = HarnessMaterializer(store, spec=checked_plan.model_spec)
    parent_harness_ref = publish_faulty_parent_harness(store, checked_plan.model_spec)
    proposer_harness_ref = publish_proposer_harness(store, checked_plan.model_spec)

    prepared: list[_PreparedContext] = []
    for block in checked_plan.blocks:
        runner = FixedModelRunner(
            spec=checked_plan.model_spec,
            backend=backends[block.cell],
            attempt_ledger=ledger,
        )
        fit_scenarios = tuple(
            sorted(verified_fit[block.cell].scenarios, key=lambda item: item.task.task_id)
        )
        parent_harness = materializer.materialize(parent_harness_ref)
        fit_parent = tuple(
            make_observation(
                scenario=scenario,
                role=HarnessRole.FAULTY_PARENT,
                seed=solver_seed(block.solver_master_seed, scenario.task.task_id),
                record=runner.execute_record(
                    scenario.task,
                    harness=parent_harness,
                    seed=solver_seed(block.solver_master_seed, scenario.task.task_id),
                ),
            )
            for scenario in fit_scenarios
        )
        feedback = build_feedback_projection_from_coordinates(
            plan_fingerprint=block.fit_stage_fingerprint,
            feedback_linkage=block.feedback_linkage,
            task_count=block.fit_task_count,
            parent=fit_parent,
        )
        feedback_ref = store.put_json(
            feedback,
            media_type=HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
        )
        proposal_record = runner.execute_record(
            proposer_task(feedback),
            harness=materializer.materialize(proposer_harness_ref),
            seed=block.proposal_seed,
        )
        parsed_action = None
        output_sha256 = None
        if proposal_record.execution.status is ExecutionStatus.COMPLETED:
            output = proposal_record.execution.output_text
            if output is None:  # pragma: no cover - execution invariant
                raise RuntimeError("completed proposer call omitted output")
            output_sha256 = sha256_bytes(output.encode("utf-8"))
            try:
                parsed_action = parse_fault_repair_action(output)
            except ValueError:
                parsed_action = None
        applied_action = parsed_action or FaultRepairAction(rule_id=RepairRuleId.CONSTANT_LEGACY)
        proposal = ModelAuthoredFaultProposal(
            feedback_ref=feedback_ref,
            proposer_harness_ref=proposer_harness_ref,
            execution_ref=proposal_record.execution_ref,
            outcome_ref=proposal_record.outcome_ref,
            execution_status=proposal_record.execution.status,
            raw_model_output_sha256=output_sha256,
            parsed_action=parsed_action,
            proposal_valid=parsed_action is not None,
            applied_rule_id=applied_action.rule_id,
        )
        runtime_producer_id = canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-direct-runtime/v1",
                "four_context_plan_fingerprint": checked_plan.fingerprint,
                "cell": block.cell,
                "block_id": block.block_id,
                "behavior_rewrite": False,
            }
        )
        compiled = compile_fault_repair(
            store,
            checked_plan.model_spec,
            applied_action,
            runtime_producer_id=runtime_producer_id,
        )
        compilation = verify_fault_compilation(
            store,
            checked_plan.model_spec,
            compiled.manifest_ref,
            expected_runtime_producer_id=runtime_producer_id,
        )
        if compilation.entry(HarnessRole.FAULTY_PARENT).harness_ref != parent_harness_ref:
            raise HarnessFaultDevelopmentError("compiled parent differs from observed parent")
        prepared.append(
            _PreparedContext(
                block=block,
                verified_fit=verified_fit[block.cell],
                runner=runner,
                parent_harness_ref=parent_harness_ref,
                feedback_ref=feedback_ref,
                proposer_harness_ref=proposer_harness_ref,
                proposal=proposal,
                runtime_producer_id=runtime_producer_id,
                compilation_ref=compiled.manifest_ref,
                candidate_harness_ref=compilation.entry(HarnessRole.CANDIDATE).harness_ref,
                fit_parent_observations=fit_parent,
                gate_observations=[],
            )
        )

    definition_state = ledger.state()
    if (
        definition_state.attempts_used != checked_plan.candidate_definition_call_count
        or definition_state.completed_attempts != checked_plan.candidate_definition_call_count
        or definition_state.pending_reservation_ref is not None
        or definition_state.poisoned
    ):
        raise HarnessFaultDevelopmentError("candidate-definition ledger prefix did not close")
    candidates = tuple(_frozen_candidate(item) for item in prepared)
    candidate_freeze = JointFaultContextCandidateFreeze(
        plan_fingerprint=checked_plan.fingerprint,
        freeze_id=fault_context_candidate_freeze_id(
            plan_fingerprint=checked_plan.fingerprint,
            candidates=candidates,
        ),
        candidates=candidates,
        candidate_definition_call_count=definition_state.attempts_used,
        last_candidate_definition_sequence=definition_state.attempts_used - 1,
        first_attribution_sequence=definition_state.attempts_used,
    )
    candidate_freeze_ref = store.put_json(
        candidate_freeze,
        media_type=HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE,
    )

    # This ordering is enforced by this trusted function, but CAS has no trusted
    # publication clock and the reusable authority grant is not a one-shot
    # capability.  Result contracts therefore explicitly decline independent
    # timing, one-time-consumption, and cross-process atomicity attestation.
    verified_gate = _open_paired_gate_block_after_freeze(
        store,
        checked_plan,
        verified_fit,
    )

    for item in prepared:
        compilation = verify_fault_compilation(
            store,
            checked_plan.model_spec,
            item.compilation_ref,
            expected_runtime_producer_id=item.runtime_producer_id,
        )
        gate_scenarios = tuple(
            sorted(
                verified_gate[item.block.cell].scenarios, key=lambda scenario: scenario.task.task_id
            )
        )
        for role in (
            HarnessRole.FAULTY_PARENT,
            HarnessRole.CANDIDATE,
            HarnessRole.REVERT,
            HarnessRole.PLACEBO,
        ):
            harness_ref = compilation.entry(role).harness_ref
            harness = materializer.materialize(harness_ref)
            for scenario in gate_scenarios:
                seed = solver_seed(item.block.solver_master_seed, scenario.task.task_id)
                item.gate_observations.append(
                    make_observation(
                        scenario=scenario,
                        role=role,
                        seed=seed,
                        record=item.runner.execute_record(
                            scenario.task,
                            harness=harness,
                            seed=seed,
                        ),
                    )
                )

    state = ledger.state()
    resource_closed = bool(
        state.attempts_used == checked_plan.max_model_calls
        and state.completed_attempts == checked_plan.max_model_calls
        and state.pending_reservation_ref is None
        and not state.poisoned
        and state.charged_tokens <= checked_plan.max_total_tokens
    )
    if not resource_closed or state.tail_ref is None:
        raise HarnessFaultDevelopmentError(
            "global ledger did not close the exact split-block budget"
        )

    closures = []
    gate_records = []
    for item in prepared:
        gate_observations = tuple(item.gate_observations)
        gate = make_gate_from_coordinates(
            promotion_rule=item.block.promotion_rule,
            task_count=item.block.gate_task_count,
            proposal=item.proposal,
            observations=gate_observations,
            parent_harness_ref=item.parent_harness_ref,
            candidate_harness_ref=item.candidate_harness_ref,
            resource_budget_passed=resource_closed,
        )
        closures.append(
            HarnessFaultContextDevelopmentClosure(
                cell=item.block.cell,
                block_id=item.block.block_id,
                plan_fingerprint=item.block.fingerprint,
                runtime_producer_id=item.runtime_producer_id,
                parent_harness_ref=item.parent_harness_ref,
                compilation_ref=item.compilation_ref,
                feedback_ref=item.feedback_ref,
                proposer_harness_ref=item.proposer_harness_ref,
                proposal=item.proposal,
                fit_parent_observations=item.fit_parent_observations,
                gate_observations=gate_observations,
                gate=gate,
            )
        )
        gate_records.append(
            FaultContextGateRecord(
                cell=item.block.cell,
                block_id=item.block.block_id,
                gate=gate,
            )
        )
    gate_batch = HarnessFaultFourContextGateBatch(
        plan_fingerprint=checked_plan.fingerprint,
        candidate_freeze_ref=candidate_freeze_ref,
        candidate_freeze_id=candidate_freeze.freeze_id,
        records=tuple(gate_records),
        model_call_count_before_gate_batch=state.attempts_used,
    )
    gate_batch_ref = store.put_json(
        gate_batch,
        media_type=HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE,
    )
    result = HarnessFaultFourContextDevelopmentResult(
        plan=checked_plan,
        plan_fingerprint=checked_plan.fingerprint,
        candidate_freeze=candidate_freeze,
        candidate_freeze_ref=candidate_freeze_ref,
        contexts=tuple(closures),
        gate_batch=gate_batch,
        gate_batch_ref=gate_batch_ref,
        ledger_id=ledger_id,
        attempt_budget=checked_plan.attempt_budget,
        ledger_tail_ref=state.tail_ref,
        model_call_count=state.attempts_used,
    )
    result_ref = store.put_json(result, media_type=HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE)
    return HarnessFaultFourContextRunRecord(result=result, result_ref=result_ref)


def _validate_context_backends(
    context_backends: Mapping[FaultFactorialCell, ModelBackend],
    plan: HarnessFaultFourContextPlan,
) -> dict[FaultFactorialCell, ModelBackend]:
    if not isinstance(context_backends, Mapping):
        raise TypeError("context_backends must be a mapping")
    if any(type(cell) is not FaultFactorialCell for cell in context_backends):
        raise TypeError("context_backends keys must be exact FaultFactorialCell values")
    if frozenset(context_backends) != frozenset(FAULT_CONTEXT_ORDER):
        raise ValueError("context_backends must contain exactly SS, MS, SM, and MM")
    values = tuple(context_backends[cell] for cell in FAULT_CONTEXT_ORDER)
    if any(not isinstance(backend, ModelBackend) for backend in values):
        raise TypeError("every context backend must implement ModelBackend")
    if len({id(backend) for backend in values}) != len(FAULT_CONTEXT_ORDER):
        raise HarnessFaultDevelopmentError("every context requires a distinct backend client")
    if any(backend.fingerprint != plan.model_spec.backend_fingerprint for backend in values):
        raise HarnessFaultDevelopmentError("context backend differs from the frozen model spec")
    return {cell: context_backends[cell] for cell in FAULT_CONTEXT_ORDER}


def _open_paired_fit_block(
    store: ArtifactStore,
    plan: HarnessFaultFourContextPlan,
) -> dict[FaultFactorialCell, VerifiedPartitionOpening]:
    opened = {}
    anchor = None
    for block in plan.blocks:
        current = verify_partition_opening(store, block.fit_partition_grant)
        if len(current.scenarios) != block.fit_task_count:
            raise HarnessFaultDevelopmentError("opened fit block differs from fit_task_count")
        if anchor is None:
            anchor = current
        elif current != anchor:
            raise HarnessFaultDevelopmentError(
                "factorial contexts do not share one exact latent scenario block"
            )
        opened[block.cell] = current
    return opened


def _open_paired_gate_block_after_freeze(
    store: ArtifactStore,
    plan: HarnessFaultFourContextPlan,
    fit_opened: dict[FaultFactorialCell, VerifiedPartitionOpening],
) -> dict[FaultFactorialCell, VerifiedPartitionOpening]:
    opened = {}
    anchor = None
    for block in plan.blocks:
        current = verify_partition_opening(store, block.gate_partition_grant)
        if len(current.scenarios) != block.gate_task_count:
            raise HarnessFaultDevelopmentError("opened gate block differs from gate_task_count")
        if anchor is None:
            anchor = current
        elif current != anchor:
            raise HarnessFaultDevelopmentError(
                "factorial contexts do not share one exact gate scenario block"
            )
        _require_fit_gate_disjoint(fit_opened[block.cell], current)
        opened[block.cell] = current
    return opened


def _require_fit_gate_disjoint(
    fit: VerifiedPartitionOpening,
    gate: VerifiedPartitionOpening,
) -> None:
    fit_scenarios = fit.scenarios
    gate_scenarios = gate.scenarios
    axes = (
        (
            "task",
            {item.task.task_id for item in fit_scenarios},
            {item.task.task_id for item in gate_scenarios},
        ),
        (
            "scenario",
            {item.scenario_id for item in fit_scenarios},
            {item.scenario_id for item in gate_scenarios},
        ),
        (
            "scenario commitment",
            {item.scenario_commitment for item in fit_scenarios},
            {item.scenario_commitment for item in gate_scenarios},
        ),
        (
            "source",
            {item.source_id for item in fit_scenarios},
            {item.source_id for item in gate_scenarios},
        ),
        (
            "group",
            {item.group_id for item in fit_scenarios},
            {item.group_id for item in gate_scenarios},
        ),
        (
            "family/source",
            {(item.family, item.source_id) for item in fit_scenarios},
            {(item.family, item.source_id) for item in gate_scenarios},
        ),
        (
            "family/group",
            {(item.family, item.group_id) for item in fit_scenarios},
            {(item.family, item.group_id) for item in gate_scenarios},
        ),
    )
    for label, fit_values, gate_values in axes:
        if fit_values.intersection(gate_values):
            raise HarnessFaultDevelopmentError(f"fit and gate blocks overlap on {label} units")
    fit_strata = sorted(
        (item.family.value, item.surface.value, item.role.value) for item in fit_scenarios
    )
    gate_strata = sorted(
        (item.family.value, item.surface.value, item.role.value) for item in gate_scenarios
    )
    if fit_strata != gate_strata:
        raise HarnessFaultDevelopmentError("fit and gate blocks do not share matched strata")


def _frozen_candidate(item: _PreparedContext) -> FrozenFaultContextCandidate:
    return FrozenFaultContextCandidate(
        cell=item.block.cell,
        block_id=item.block.block_id,
        plan_fingerprint=item.block.fingerprint,
        feedback_ref=item.feedback_ref,
        proposer_harness_ref=item.proposer_harness_ref,
        proposal=item.proposal,
        runtime_producer_id=item.runtime_producer_id,
        parent_harness_ref=item.parent_harness_ref,
        compilation_ref=item.compilation_ref,
        candidate_harness_ref=item.candidate_harness_ref,
    )


__all__ = [
    "HarnessFaultFourContextRunRecord",
    "run_model_driven_harness_fault_four_context_development",
]
