"""Live same-spec four-arm development runner with one observable evolution step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from spiral_harness.benchmark.bbh import BBHLogicalDeductionSevenAdapter
from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    ExecutionStatus,
)
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.execution.pure_model import PureModelBackend, PureModelRunner
from spiral_harness.experiments.development_four_arm_artifacts import (
    publish_development_harness,
)
from spiral_harness.experiments.development_four_arm_contracts import (
    DevelopmentArm,
    DevelopmentArmObservation,
    DevelopmentFourArmClosure,
    DevelopmentFourArmPlan,
    DevelopmentSplit,
    FrozenCandidateProposal,
    JointCandidateFreeze,
    SelectionDecision,
    development_adaptive_stage_fingerprint,
    development_candidate_freeze_id,
)
from spiral_harness.experiments.development_four_arm_execution import (
    EvaluatedDevelopmentCall,
    evaluate_fixed,
    evaluate_pure,
    execute_proposer,
    holdout_metrics,
    provider_identity_summary,
    select_candidate,
)
from spiral_harness.experiments.development_four_arm_execution import (
    full_disclosure as make_full_disclosure,
)
from spiral_harness.experiments.development_four_arm_execution import (
    score_disclosure as make_score_disclosure,
)
from spiral_harness.experiments.development_four_arm_prompts import (
    DEVELOPMENT_SEED_PROMPT,
    FULL_PROPOSER_SYSTEM_PROMPT,
    SCORE_PROPOSER_SYSTEM_PROMPT,
    build_full_proposal_input,
    build_score_proposal_input,
    parse_candidate_prompt,
)
from spiral_harness.storage.artifact_store import ArtifactStore

DEVELOPMENT_FOUR_ARM_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.development-four-arm-result.v1+json"
)
DEVELOPMENT_FOUR_ARM_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.development-four-arm-closure.v1+json"
)
DEVELOPMENT_CANDIDATE_FREEZE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.development-four-arm-candidate-freeze.v1+json"
)
DEVELOPMENT_CALL_TRACE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.development-four-arm-call-trace.v1+json"
)


@dataclass(frozen=True, slots=True)
class DevelopmentFourArmRunResult:
    """One persisted non-reportable development run."""

    closure: DevelopmentFourArmClosure
    closure_ref: ArtifactRef
    result_ref: ArtifactRef
    payload: Mapping[str, object]


def run_development_four_arm(
    *,
    output: Path,
    backend: ModelBackend | PureModelBackend,
    gsm8k_adapter: GSM8KBenchmarkAdapter,
    bbh_adapter: BBHLogicalDeductionSevenAdapter,
    plan: DevelopmentFourArmPlan,
    max_tokens_per_attempt: int | None = None,
) -> DevelopmentFourArmRunResult:
    """Execute the fixed 58-call public development protocol without retries."""

    checked_plan = DevelopmentFourArmPlan.model_validate(plan, strict=True)
    if not isinstance(backend, ModelBackend) or not isinstance(backend, PureModelBackend):
        raise TypeError("backend must implement both fixed-harness and PURE invocation")
    if backend.fingerprint != checked_plan.model_spec.backend_fingerprint:
        raise ValueError("backend fingerprint differs from the frozen development plan")
    if max_tokens_per_attempt is None:
        max_tokens_per_attempt = checked_plan.max_tokens_per_attempt
    if (
        type(max_tokens_per_attempt) is not int
        or max_tokens_per_attempt != checked_plan.max_tokens_per_attempt
    ):
        raise ValueError("max_tokens_per_attempt must equal the frozen plan ceiling")
    _validate_plan_against_adapters(
        checked_plan,
        gsm8k_adapter=gsm8k_adapter,
        bbh_adapter=bbh_adapter,
    )

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("development output directory must be absent or empty")
    store = ArtifactStore(output / "artifacts")
    fixed_ledger = AttemptLedger(
        store,
        ledger_id="development-four-arm-fixed",
        budget=AttemptBudget(
            max_attempts=50,
            max_total_tokens=50 * max_tokens_per_attempt,
            max_tokens_per_attempt=max_tokens_per_attempt,
        ),
    )
    pure_ledger = AttemptLedger(
        store,
        ledger_id="development-four-arm-pure",
        budget=AttemptBudget(
            max_attempts=8,
            max_total_tokens=8 * max_tokens_per_attempt,
            max_tokens_per_attempt=max_tokens_per_attempt,
        ),
    )
    fixed_runner = FixedModelRunner(
        spec=checked_plan.model_spec,
        backend=backend,
        attempt_ledger=fixed_ledger,
    )
    pure_runner = PureModelRunner(
        spec=checked_plan.model_spec,
        backend=backend,
        attempt_ledger=pure_ledger,
    )
    plan_fingerprint = checked_plan.fingerprint
    adaptive_stage_fingerprint = development_adaptive_stage_fingerprint(checked_plan)
    call_trace: list[dict[str, object]] = []

    static_condition = _condition_id(adaptive_stage_fingerprint, DevelopmentArm.STATIC, "seed")
    pure_condition = _condition_id(plan_fingerprint, DevelopmentArm.PURE, "bare-model")
    static_harness = publish_development_harness(
        store,
        spec=checked_plan.model_spec,
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        condition_id=static_condition,
        arm=DevelopmentArm.STATIC,
        purpose="solver",
        feedback_view="none",
        promotion_rule="none",
        system_prompt=DEVELOPMENT_SEED_PROMPT,
    )

    fit_tasks = tuple(task for task in checked_plan.tasks if task.split is DevelopmentSplit.FIT)
    static_fit = tuple(
        evaluate_fixed(
            task=task,
            arm=DevelopmentArm.STATIC,
            condition_id=static_condition,
            harness=static_harness,
            runner=fixed_runner,
            adapters=(gsm8k_adapter, bbh_adapter),
            trace=call_trace,
            phase="fit-parent",
        )
        for task in fit_tasks
    )
    score_disclosure = make_score_disclosure(static_fit)
    full_disclosure = make_full_disclosure(static_fit)

    score_proposer_condition = _condition_id(
        adaptive_stage_fingerprint, DevelopmentArm.SCORE, "proposer"
    )
    full_proposer_condition = _condition_id(
        adaptive_stage_fingerprint, DevelopmentArm.FULL, "proposer"
    )
    score_proposer = publish_development_harness(
        store,
        spec=checked_plan.model_spec,
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        condition_id=score_proposer_condition,
        arm=DevelopmentArm.SCORE,
        purpose="proposer",
        feedback_view="aggregate-score",
        promotion_rule="development-automatic-fit-v1",
        system_prompt=SCORE_PROPOSER_SYSTEM_PROMPT,
    )
    full_proposer = publish_development_harness(
        store,
        spec=checked_plan.model_spec,
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        condition_id=full_proposer_condition,
        arm=DevelopmentArm.FULL,
        purpose="proposer",
        feedback_view="fit-item-evidence",
        promotion_rule="development-automatic-fit-v1",
        system_prompt=FULL_PROPOSER_SYSTEM_PROMPT,
    )
    score_proposal_record = execute_proposer(
        runner=fixed_runner,
        harness=score_proposer,
        arm=DevelopmentArm.SCORE,
        prompt=build_score_proposal_input(score_disclosure),
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        trace=call_trace,
    )
    full_proposal_record = execute_proposer(
        runner=fixed_runner,
        harness=full_proposer,
        arm=DevelopmentArm.FULL,
        prompt=build_full_proposal_input(full_disclosure),
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        trace=call_trace,
    )
    parsed_prompts = {
        DevelopmentArm.SCORE: parse_candidate_prompt(score_proposal_record.execution.output_text),
        DevelopmentArm.FULL: parse_candidate_prompt(full_proposal_record.execution.output_text),
    }
    proposal_records = {
        DevelopmentArm.SCORE: score_proposal_record,
        DevelopmentArm.FULL: full_proposal_record,
    }
    parent_conditions = {
        DevelopmentArm.SCORE: static_condition,
        DevelopmentArm.FULL: static_condition,
    }
    proposals: list[FrozenCandidateProposal] = []
    for arm in (DevelopmentArm.SCORE, DevelopmentArm.FULL):
        record = proposal_records[arm]
        candidate_id = canonical_sha256(
            {
                "schema": "spiral-harness/development-four-arm-candidate/v1",
                "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
                "arm": arm.value,
                "proposal_request_sha256": record.execution.request_sha256,
                "parsed_prompt": parsed_prompts[arm],
            }
        )
        proposals.append(
            FrozenCandidateProposal(
                arm=arm,
                candidate_id=candidate_id,
                proposal_fingerprint=record.execution.request_sha256,
                parent_condition_id=parent_conditions[arm],
                candidate_condition_id=_condition_id(
                    adaptive_stage_fingerprint,
                    arm,
                    "candidate-" + candidate_id,
                ),
            )
        )
    candidate_freeze = JointCandidateFreeze(
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        freeze_id=development_candidate_freeze_id(
            adaptive_stage_fingerprint=adaptive_stage_fingerprint,
            proposals=tuple(proposals),
        ),
        freeze_sequence=len(call_trace),
        candidate_evaluation_start_sequence=len(call_trace) + 1,
        proposals=tuple(proposals),
    )
    candidate_freeze_ref = store.put_json(
        candidate_freeze,
        media_type=DEVELOPMENT_CANDIDATE_FREEZE_MEDIA_TYPE,
    )

    proposal_by_arm = {proposal.arm: proposal for proposal in candidate_freeze.proposals}
    candidate_harnesses = {}
    for arm, feedback_view in (
        (DevelopmentArm.SCORE, "aggregate-score"),
        (DevelopmentArm.FULL, "fit-item-evidence"),
    ):
        proposal = proposal_by_arm[arm]
        candidate_harnesses[arm] = publish_development_harness(
            store,
            spec=checked_plan.model_spec,
            adaptive_stage_fingerprint=adaptive_stage_fingerprint,
            condition_id=proposal.candidate_condition_id,
            arm=arm,
            purpose="solver",
            feedback_view=feedback_view,
            promotion_rule="development-automatic-fit-v1",
            system_prompt=parsed_prompts[arm] or DEVELOPMENT_SEED_PROMPT,
        )

    candidate_fit = {
        arm: tuple(
            evaluate_fixed(
                task=task,
                arm=arm,
                condition_id=proposal_by_arm[arm].candidate_condition_id,
                harness=candidate_harnesses[arm],
                runner=fixed_runner,
                adapters=(gsm8k_adapter, bbh_adapter),
                trace=call_trace,
                phase="fit-candidate",
            )
            for task in fit_tasks
        )
        for arm in (DevelopmentArm.SCORE, DevelopmentArm.FULL)
    }
    selections = tuple(
        select_candidate(
            arm=arm,
            proposal=proposal_by_arm[arm],
            candidate_valid=parsed_prompts[arm] is not None
            and proposal_records[arm].execution.status is ExecutionStatus.COMPLETED,
            parent=static_fit,
            candidate=candidate_fit[arm],
        )
        for arm in (DevelopmentArm.SCORE, DevelopmentArm.FULL)
    )
    selection_by_arm = {selection.arm: selection for selection in selections}
    selection_freeze_ref = store.put_json(
        {
            "schema_version": "1",
            "plan_fingerprint": plan_fingerprint,
            "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
            "candidate_freeze_ref": candidate_freeze_ref.model_dump(mode="json"),
            "selections": [selection.model_dump(mode="json") for selection in selections],
            "selection_freeze_sequence": len(call_trace),
            "first_holdout_sequence": len(call_trace) + 1,
            "shared_parent_condition_id": static_condition,
            "holdout_opened": False,
        },
        media_type="application/vnd.spiral-harness.development-selection-freeze.v1+json",
    )
    selected_harnesses = {
        arm: (
            candidate_harnesses[arm]
            if selection_by_arm[arm].decision is SelectionDecision.PROMOTE
            else static_harness
        )
        for arm in (DevelopmentArm.SCORE, DevelopmentArm.FULL)
    }

    holdout_tasks = tuple(
        task for task in checked_plan.tasks if task.split is DevelopmentSplit.HOLDOUT
    )
    arm_order = (
        DevelopmentArm.PURE,
        DevelopmentArm.STATIC,
        DevelopmentArm.SCORE,
        DevelopmentArm.FULL,
    )
    holdout_calls: list[EvaluatedDevelopmentCall] = []
    for task_index, task in enumerate(holdout_tasks):
        rotated = arm_order[task_index % 4 :] + arm_order[: task_index % 4]
        for arm in rotated:
            if arm is DevelopmentArm.PURE:
                holdout_calls.append(
                    evaluate_pure(
                        task=task,
                        condition_id=pure_condition,
                        runner=pure_runner,
                        adapters=(gsm8k_adapter, bbh_adapter),
                        trace=call_trace,
                    )
                )
            else:
                harness = (
                    static_harness if arm is DevelopmentArm.STATIC else selected_harnesses[arm]
                )
                condition_id = (
                    static_condition
                    if arm is DevelopmentArm.STATIC
                    else selection_by_arm[arm].selected_condition_id
                )
                holdout_calls.append(
                    evaluate_fixed(
                        task=task,
                        arm=arm,
                        condition_id=condition_id,
                        harness=harness,
                        runner=fixed_runner,
                        adapters=(gsm8k_adapter, bbh_adapter),
                        trace=call_trace,
                        phase="development-holdout",
                    )
                )

    if len(call_trace) != checked_plan.max_model_calls:
        raise RuntimeError("development runner did not consume the frozen 58-call topology")
    fixed_attempt_state = fixed_runner.attempt_state()
    pure_attempt_state = pure_runner.attempt_state()
    if (
        fixed_attempt_state.attempts_used != 50
        or pure_attempt_state.attempts_used != 8
        or fixed_attempt_state.pending_reservation_ref is not None
        or pure_attempt_state.pending_reservation_ref is not None
    ):
        raise RuntimeError("development attempt ledgers do not close the 50 + 8 topology")
    call_trace_ref = store.put_json(
        {
            "schema_version": "1",
            "plan_fingerprint": plan_fingerprint,
            "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
            "calls": call_trace,
        },
        media_type=DEVELOPMENT_CALL_TRACE_MEDIA_TYPE,
    )
    observations = tuple(
        DevelopmentArmObservation(
            task_id=item.task_id,
            benchmark=item.benchmark,
            arm=item.arm,
            seed=item.seed,
            model_spec_fingerprint=item.execution.spec.fingerprint,
            condition_id=item.condition_id,
            reference_id=item.execution_ref.sha256,
            correct=item.correct,
        )
        for item in holdout_calls
    )
    closure = DevelopmentFourArmClosure(
        plan=checked_plan,
        plan_fingerprint=plan_fingerprint,
        score_disclosure=score_disclosure,
        full_disclosure=full_disclosure,
        candidate_freeze=candidate_freeze,
        selections=selections,
        observations=observations,
    )
    closure_ref = store.put_json(
        closure,
        media_type=DEVELOPMENT_FOUR_ARM_CLOSURE_MEDIA_TYPE,
    )
    metrics = holdout_metrics(observations)
    identities = provider_identity_summary(call_trace)
    observed_spec_fingerprints = {str(item["model_spec_fingerprint"]) for item in call_trace}
    nominal_spec_shared = observed_spec_fingerprints == {checked_plan.model_spec.fingerprint}
    if not nominal_spec_shared:
        raise RuntimeError("executed calls drifted from the frozen nominal model spec")
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": "permanently_nonreportable_development_four_arm",
        "plan_fingerprint": plan_fingerprint,
        "adaptive_stage_fingerprint": adaptive_stage_fingerprint,
        "closure_ref": closure_ref.model_dump(mode="json"),
        "candidate_freeze_ref": candidate_freeze_ref.model_dump(mode="json"),
        "selection_freeze_ref": selection_freeze_ref.model_dump(mode="json"),
        "call_trace_ref": call_trace_ref.model_dump(mode="json"),
        "model": checked_plan.model_spec.model,
        "model_spec_fingerprint": checked_plan.model_spec.fingerprint,
        "nominal_model_spec_shared_across_all_calls": nominal_spec_shared,
        "provider_identity": identities,
        "provider_identity_attested": False,
        "same_exact_served_revision_claim_allowed": False,
        "unique_task_count": 16,
        "model_call_count": len(call_trace),
        "max_model_calls": checked_plan.max_model_calls,
        "reservation_token_ceiling_per_call": max_tokens_per_attempt,
        "maximum_reserved_tokens": checked_plan.max_total_tokens,
        "resource_usage": _resource_usage(call_trace),
        "attempt_ledgers": {
            "fixed": fixed_attempt_state.model_dump(mode="json"),
            "pure": pure_attempt_state.model_dump(mode="json"),
        },
        "call_topology": {
            "shared_static_parent_fit": 8,
            "pure_holdout": 8,
            "static_holdout": 8,
            "score_proposer": 1,
            "score_candidate_fit": 8,
            "score_holdout": 8,
            "full_proposer": 1,
            "full_candidate_fit": 8,
            "full_holdout": 8,
        },
        "budget_matched_across_all_arms": False,
        "score_and_full_call_slots_matched": True,
        "adaptive_arms_share_one_static_parent_fit": True,
        "adaptive_randomness_uses_fit_stage_commitment_only": True,
        "evidence_scope": checked_plan.evidence_scope,
        "adaptive_contrast": checked_plan.adaptive_contrast,
        "joint_policy_effect_identified": False,
        "execution_attested": False,
        "budget_matched": False,
        "runtime_execution_proof": False,
        "confirmatory_protocol": False,
        "holdout_opened_after_joint_candidate_and_selection_freeze": True,
        "selection_trace": [selection.model_dump(mode="json") for selection in selections],
        "holdout_metrics": metrics,
        "calls": call_trace,
        "reportable_benchmark_result": False,
        "sealed_evidence": False,
        "confirmatory_inference": False,
        "simultaneous_lcb_available": False,
        "disclaimer": (
            "Public development exercise only: eight fit and eight public holdout tasks, no "
            "sealed release, no all-arm budget match, no confirmatory inference, and no >10pp "
            "claim. SCORE versus FULL isolates feedback view only; it is not the paper's joint "
            "feedback-plus-promotion-policy treatment."
        ),
    }
    result_ref = store.put_json(payload, media_type=DEVELOPMENT_FOUR_ARM_RESULT_MEDIA_TYPE)
    return DevelopmentFourArmRunResult(
        closure=closure,
        closure_ref=closure_ref,
        result_ref=result_ref,
        payload=MappingProxyType(payload),
    )


def _condition_id(plan_fingerprint: str, arm: DevelopmentArm, variant: str) -> str:
    return canonical_sha256(
        {
            "schema": "spiral-harness/development-four-arm-condition/v1",
            "plan_fingerprint": plan_fingerprint,
            "arm": arm.value,
            "variant": variant,
        }
    )


def _validate_plan_against_adapters(
    plan: DevelopmentFourArmPlan,
    *,
    gsm8k_adapter: GSM8KBenchmarkAdapter,
    bbh_adapter: BBHLogicalDeductionSevenAdapter,
) -> None:
    for benchmark, adapter in (
        ("gsm8k", gsm8k_adapter),
        ("bbh", bbh_adapter),
    ):
        trusted_roster = frozenset(adapter.task_roster(ProtocolPartition.EXPLORATION))
        for planned in (task for task in plan.tasks if task.benchmark.value == benchmark):
            if planned.task_id not in trusted_roster:
                raise ValueError(
                    f"{benchmark} development task is not in the trusted exploration roster"
                )
            trusted = adapter.load_task(planned.task_id)
            if trusted.question != planned.question:
                raise ValueError(
                    f"{benchmark} development task question differs from the trusted adapter"
                )


def _resource_usage(trace: list[dict[str, object]]) -> dict[str, object]:
    costs = [item["cost_usd"] for item in trace]
    return {
        "input_tokens": sum(int(item["input_tokens"]) for item in trace),
        "output_tokens": sum(int(item["output_tokens"]) for item in trace),
        "latency_ms_sum": sum(float(item["latency_ms"]) for item in trace),
        "tool_calls": sum(int(item["tool_calls"]) for item in trace),
        "cost_usd": (
            sum(float(cost) for cost in costs if cost is not None)
            if all(cost is not None for cost in costs)
            else None
        ),
        "cost_coverage_complete": all(cost is not None for cost in costs),
    }


__all__ = [
    "DEVELOPMENT_CALL_TRACE_MEDIA_TYPE",
    "DEVELOPMENT_FOUR_ARM_CLOSURE_MEDIA_TYPE",
    "DEVELOPMENT_FOUR_ARM_RESULT_MEDIA_TYPE",
    "DevelopmentFourArmRunResult",
    "development_adaptive_stage_fingerprint",
    "run_development_four_arm",
]
