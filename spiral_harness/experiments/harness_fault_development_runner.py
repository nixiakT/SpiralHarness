"""One-round model-output-driven HarnessFault public development runner."""

from __future__ import annotations

from dataclasses import dataclass

from spiral_harness.benchmark._harness_fault_cases import (
    PartitionEvaluationGrant,
    RepairRuleId,
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
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE,
    HarnessFaultDevelopmentError,
    HarnessFaultDevelopmentPlan,
    HarnessFaultModelDrivenDevelopmentResult,
    ModelAuthoredFaultProposal,
)
from spiral_harness.experiments.harness_fault_development_support import (
    build_feedback_projection,
    make_gate,
    make_observation,
    proposer_task,
    publish_proposer_harness,
    solver_seed,
)
from spiral_harness.storage.artifact_store import ArtifactStore


@dataclass(frozen=True, slots=True)
class HarnessFaultDevelopmentRunRecord:
    result: HarnessFaultModelDrivenDevelopmentResult
    result_ref: ArtifactRef


def run_model_driven_harness_fault_development(
    *,
    store: ArtifactStore,
    backend: ModelBackend,
    partition_grant: PartitionEvaluationGrant,
    plan: HarnessFaultDevelopmentPlan,
) -> HarnessFaultDevelopmentRunRecord:
    """Execute 4N+1 calls; no caller-supplied candidate or human ranking exists."""

    if type(store) is not ArtifactStore:
        raise TypeError("store must be an exact ArtifactStore")
    if not isinstance(backend, ModelBackend):
        raise TypeError("backend must implement ModelBackend")
    checked_plan = HarnessFaultDevelopmentPlan.model_validate(plan, strict=True)
    checked_grant = PartitionEvaluationGrant.model_validate(partition_grant, strict=True)
    if checked_grant.partition is not checked_plan.partition:
        raise HarnessFaultDevelopmentError("development grant must be exact exploration data")
    verified = verify_partition_opening(store, checked_grant)
    scenarios = tuple(sorted(verified.scenarios, key=lambda item: item.task.task_id))
    if len(scenarios) != checked_plan.task_count:
        raise HarnessFaultDevelopmentError("plan task_count differs from the authority roster")
    if backend.fingerprint != checked_plan.model_spec.backend_fingerprint:
        raise HarnessFaultDevelopmentError("backend differs from the frozen model spec")

    ledger_id = "harness-fault-model-driven-" + checked_plan.fingerprint[:24]
    ledger = AttemptLedger(
        store,
        ledger_id=ledger_id,
        budget=checked_plan.attempt_budget,
    )
    runner = FixedModelRunner(
        spec=checked_plan.model_spec,
        backend=backend,
        attempt_ledger=ledger,
    )
    materializer = HarnessMaterializer(store, spec=checked_plan.model_spec)
    parent_harness_ref = publish_faulty_parent_harness(store, checked_plan.model_spec)
    parent_harness = materializer.materialize(parent_harness_ref)
    parent = tuple(
        make_observation(
            scenario=scenario,
            role=HarnessRole.FAULTY_PARENT,
            seed=solver_seed(checked_plan.solver_master_seed, scenario.task.task_id),
            record=runner.execute_record(
                scenario.task,
                harness=parent_harness,
                seed=solver_seed(checked_plan.solver_master_seed, scenario.task.task_id),
            ),
        )
        for scenario in scenarios
    )

    feedback = build_feedback_projection(checked_plan, parent)
    feedback_ref = store.put_json(
        feedback,
        media_type=HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    )
    proposer_harness_ref = publish_proposer_harness(store, checked_plan.model_spec)
    proposer_record = runner.execute_record(
        proposer_task(feedback),
        harness=materializer.materialize(proposer_harness_ref),
        seed=checked_plan.proposal_seed,
    )
    parsed_action = None
    proposal_output_sha256 = None
    if proposer_record.execution.status is ExecutionStatus.COMPLETED:
        output = proposer_record.execution.output_text
        if output is None:  # pragma: no cover - execution invariant
            raise RuntimeError("completed proposer call omitted output")
        proposal_output_sha256 = sha256_bytes(output.encode("utf-8"))
        try:
            parsed_action = parse_fault_repair_action(output)
        except ValueError:
            parsed_action = None
    proposal_valid = parsed_action is not None
    applied_action = parsed_action or FaultRepairAction(rule_id=RepairRuleId.CONSTANT_LEGACY)
    proposal = ModelAuthoredFaultProposal(
        feedback_ref=feedback_ref,
        proposer_harness_ref=proposer_harness_ref,
        execution_ref=proposer_record.execution_ref,
        outcome_ref=proposer_record.outcome_ref,
        execution_status=proposer_record.execution.status,
        raw_model_output_sha256=proposal_output_sha256,
        parsed_action=parsed_action,
        proposal_valid=proposal_valid,
        applied_rule_id=applied_action.rule_id,
    )

    runtime_producer_id = canonical_sha256(
        {
            "schema": "spiral-harness/model-driven-fault-direct-runtime/v1",
            "plan_fingerprint": checked_plan.fingerprint,
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

    observations = list(parent)
    for role in (HarnessRole.CANDIDATE, HarnessRole.REVERT, HarnessRole.PLACEBO):
        harness_ref = compilation.entry(role).harness_ref
        harness = materializer.materialize(harness_ref)
        for scenario in scenarios:
            seed = solver_seed(checked_plan.solver_master_seed, scenario.task.task_id)
            observations.append(
                make_observation(
                    scenario=scenario,
                    role=role,
                    seed=seed,
                    record=runner.execute_record(
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
        raise HarnessFaultDevelopmentError("attempt ledger did not close the exact 4N+1 budget")
    candidate_ref = compilation.entry(HarnessRole.CANDIDATE).harness_ref
    frozen_observations = tuple(observations)
    gate = make_gate(
        plan=checked_plan,
        proposal=proposal,
        observations=frozen_observations,
        parent_harness_ref=parent_harness_ref,
        candidate_harness_ref=candidate_ref,
        resource_budget_passed=resource_closed,
    )
    result = HarnessFaultModelDrivenDevelopmentResult(
        plan=checked_plan,
        plan_fingerprint=checked_plan.fingerprint,
        partition_grant=checked_grant,
        runtime_producer_id=runtime_producer_id,
        parent_harness_ref=parent_harness_ref,
        compilation_ref=compiled.manifest_ref,
        feedback_ref=feedback_ref,
        proposer_harness_ref=proposer_harness_ref,
        proposal=proposal,
        observations=frozen_observations,
        gate=gate,
        ledger_id=ledger_id,
        attempt_budget=checked_plan.attempt_budget,
        ledger_tail_ref=state.tail_ref,
        model_call_count=checked_plan.max_model_calls,
    )
    result_ref = store.put_json(result, media_type=HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE)
    return HarnessFaultDevelopmentRunRecord(result=result, result_ref=result_ref)


__all__ = [
    "HarnessFaultDevelopmentRunRecord",
    "run_model_driven_harness_fault_development",
]
