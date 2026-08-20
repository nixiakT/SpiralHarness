"""Deep offline lineage checks for published BFCL public-runner evidence."""

from __future__ import annotations

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicGraderReceipt,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    adapt_bfcl_v4_public_pilot_task,
    select_bfcl_v4_pure_at_b_plurality,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4OpenAiToolAdapterResult,
    BfclV4PublicPilotTask,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
    BfclV4ArmCandidateFreeze,
    BfclV4ArmSelection,
    BfclV4JointCandidateFreeze,
    BfclV4JointSelectionFreeze,
    BfclV4RunClosure,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.execution.contracts import ExecutionStatus
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4CandidateParseResult,
    BfclV4CandidateResolution,
    BfclV4DiagnosisParseResult,
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    build_bfcl_v4_proposal_prompt,
    parse_bfcl_v4_candidate,
    parse_bfcl_v4_diagnosis,
    resolve_bfcl_v4_candidate,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE,
    BFCL_V4_RUNNER_CANDIDATE_MEDIA_TYPE,
    BFCL_V4_RUNNER_DIAGNOSIS_MEDIA_TYPE,
    BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
    BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
    BfclV4PublicPilotRunResult,
    BfclV4RunnerHarnessArtifact,
    BfclV4RunnerHarnessKind,
)
from spiral_harness.experiments.bfcl_v4_public_runner_evidence import (
    build_runner_diagnosis_prompt,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    BfclV4RunnerCallRecord,
    load_canonical_model,
    prediction_from_wire_calls,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4ArmSelectionDecision,
    BfclV4HoldoutEvidence,
    BfclV4HoldoutObservation,
    BfclV4JointSelectionDecision,
    BfclV4JointSelectionEvidence,
    BfclV4PublicDescriptiveMetrics,
)
from spiral_harness.experiments.bfcl_v4_public_selection_logic import (
    build_bfcl_v4_holdout_evidence,
    build_bfcl_v4_joint_selection_evidence,
    compute_bfcl_v4_public_descriptive_metrics,
    select_bfcl_v4_public_candidates,
)
from spiral_harness.storage.protocol import ArtifactRepository


def _candidate_resolutions(
    repository: ArtifactRepository,
    *,
    plan: BfclV4PublicPilotCallPlan,
    result: BfclV4PublicPilotRunResult,
    records: tuple[BfclV4RunnerCallRecord, ...],
    tasks_by_id: dict[str, BfclV4PublicPilotTask],
) -> dict[BfclV4PilotArm, BfclV4CandidateResolution]:
    freeze = load_canonical_model(
        repository,
        result.candidate_freeze_ref,
        BfclV4JointCandidateFreeze,
        media_type=BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    )
    output: dict[BfclV4PilotArm, BfclV4CandidateResolution] = {}
    coordinates = (
        (BfclV4PilotArm.SCORE, 10, 12, freeze.score),
        (BfclV4PilotArm.FULL, 11, 13, freeze.full),
    )
    for arm, diagnosis_index, proposal_index, arm_freeze in coordinates:
        diagnosis_record = records[diagnosis_index]
        proposal_record = records[proposal_index]
        diagnosis_prompt = load_canonical_model(
            repository,
            diagnosis_record.materialization.executed_harness_ref,
            BfclV4DiagnosisPrompt,
            media_type=BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
        )
        expected_diagnosis_prompt = build_runner_diagnosis_prompt(
            arm=arm,
            parent_system_prompt=_parent_prompt(repository, arm_freeze),
            records=records[:10],
            tasks_by_id=tasks_by_id,
        )
        if diagnosis_prompt != expected_diagnosis_prompt:
            raise BfclV4PublicRunnerError(
                "diagnosis prompt differs from the five parent-FIT grades"
            )
        diagnosis_execution = diagnosis_record.execution_record.execution
        diagnosis = parse_bfcl_v4_diagnosis(
            diagnosis_prompt,
            (
                diagnosis_execution.response
                if diagnosis_execution.status is ExecutionStatus.COMPLETED
                else None
            ),
        )
        stored_diagnosis = load_canonical_model(
            repository,
            diagnosis_record.completion.model_output_ref,
            BfclV4DiagnosisParseResult,
            media_type=BFCL_V4_RUNNER_DIAGNOSIS_MEDIA_TYPE,
        )
        if diagnosis != stored_diagnosis:
            raise BfclV4PublicRunnerError("diagnosis parse differs from its native response")

        proposal_prompt = load_canonical_model(
            repository,
            proposal_record.materialization.executed_harness_ref,
            BfclV4ProposalPrompt,
            media_type=BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
        )
        if proposal_prompt != build_bfcl_v4_proposal_prompt(diagnosis_prompt, diagnosis):
            raise BfclV4PublicRunnerError("proposal prompt differs from its diagnosis lineage")
        proposal_execution = proposal_record.execution_record.execution
        candidate = parse_bfcl_v4_candidate(
            proposal_prompt,
            (
                proposal_execution.response
                if proposal_execution.status is ExecutionStatus.COMPLETED
                else None
            ),
        )
        stored_candidate = load_canonical_model(
            repository,
            proposal_record.completion.model_output_ref,
            BfclV4CandidateParseResult,
            media_type=BFCL_V4_RUNNER_CANDIDATE_MEDIA_TYPE,
        )
        if candidate != stored_candidate:
            raise BfclV4PublicRunnerError("candidate parse differs from its native response")
        resolution = resolve_bfcl_v4_candidate(
            diagnosis_result=diagnosis,
            proposal_prompt=proposal_prompt,
            candidate_parse_result=candidate,
        )
        _verify_candidate_freeze_arm(
            repository,
            arm_freeze=arm_freeze,
            proposal_record=proposal_record,
            resolution=resolution,
        )
        output[arm] = resolution
    if freeze.plan_fingerprint != plan.fingerprint:
        raise BfclV4PublicRunnerError("candidate freeze belongs to another plan")
    return output


def _parent_prompt(
    repository: ArtifactRepository,
    arm_freeze: BfclV4ArmCandidateFreeze,
) -> str:
    parent = load_canonical_model(
        repository,
        arm_freeze.parent_harness_ref,
        BfclV4RunnerHarnessArtifact,
        media_type=BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
    )
    if parent.kind is not BfclV4RunnerHarnessKind.PARENT or parent.system_prompt is None:
        raise BfclV4PublicRunnerError("candidate freeze parent is not the exact parent harness")
    return parent.system_prompt


def _verify_candidate_freeze_arm(
    repository: ArtifactRepository,
    *,
    arm_freeze: BfclV4ArmCandidateFreeze,
    proposal_record: BfclV4RunnerCallRecord,
    resolution: BfclV4CandidateResolution,
) -> None:
    expected = (
        proposal_record.completion_ref,
        proposal_record.completion.model_output_ref,
        resolution.candidate_admissible,
        resolution.exact_parent_fallback_used,
    )
    observed = (
        arm_freeze.proposal_completion_ref,
        arm_freeze.candidate_parse_ref,
        arm_freeze.candidate_valid,
        arm_freeze.fallback_used,
    )
    if observed != expected or _parent_prompt(repository, arm_freeze) != (
        resolution.parent_system_prompt
    ):
        raise BfclV4PublicRunnerError("candidate freeze differs from parsed proposal resolution")
    if resolution.candidate_admissible:
        if arm_freeze.candidate_harness_ref is None:
            raise BfclV4PublicRunnerError("admissible candidate harness is absent")
        candidate = load_canonical_model(
            repository,
            arm_freeze.candidate_harness_ref,
            BfclV4RunnerHarnessArtifact,
            media_type=BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
        )
        if (
            candidate.kind is not BfclV4RunnerHarnessKind.CANDIDATE
            or candidate.arm != arm_freeze.arm.value
            or candidate.system_prompt != resolution.evaluation_system_prompt
        ):
            raise BfclV4PublicRunnerError("candidate harness differs from resolved strategy")


def _verify_selection_observations(
    records: tuple[BfclV4RunnerCallRecord, ...],
    evidence: BfclV4JointSelectionEvidence,
) -> None:
    observations = tuple(
        sorted(
            (*evidence.score.observations, *evidence.full.observations),
            key=lambda item: item.slot.global_slot,
        )
    )
    for observation in observations:
        record = records[observation.slot.global_slot]
        receipt = record.grader_receipt
        if record.prediction is None or receipt is None:
            raise BfclV4PublicRunnerError("selection observation points to an ungraded call")
        observed = (
            observation.slot,
            observation.materialization,
            observation.completion,
            observation.completion_ref,
            observation.provider_attempt_succeeded,
            observation.prediction,
            observation.grader_receipt_reference_sha256,
            observation.accepted,
        )
        expected = (
            record.slot,
            record.materialization,
            record.completion,
            record.completion_ref,
            record.provider_succeeded,
            record.prediction,
            receipt.fingerprint,
            receipt.upstream_ast_checker_valid,
        )
        if observed != expected:
            raise BfclV4PublicRunnerError("selection observation is detached from its completion")


def _verify_joint_selection(
    repository: ArtifactRepository,
    *,
    plan: BfclV4PublicPilotCallPlan,
    result: BfclV4PublicPilotRunResult,
    records: tuple[BfclV4RunnerCallRecord, ...],
    evidence: BfclV4JointSelectionEvidence,
    decision: BfclV4JointSelectionDecision,
    resolutions: dict[BfclV4PilotArm, BfclV4CandidateResolution],
) -> None:
    _verify_selection_observations(records, evidence)
    observations = tuple(
        sorted(
            (*evidence.score.observations, *evidence.full.observations),
            key=lambda item: item.slot.global_slot,
        )
    )
    rebuilt = build_bfcl_v4_joint_selection_evidence(
        plan=plan,
        score_resolution=resolutions[BfclV4PilotArm.SCORE],
        full_resolution=resolutions[BfclV4PilotArm.FULL],
        observations=observations,
    )
    if rebuilt != evidence or select_bfcl_v4_public_candidates(plan=plan, evidence=rebuilt) != (
        decision
    ):
        raise BfclV4PublicRunnerError("selection evidence or decision differs from replayed grades")

    joint = load_canonical_model(
        repository,
        result.joint_selection_freeze_ref,
        BfclV4JointSelectionFreeze,
        media_type=BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
    )
    for selection_ref, expected_decision in (
        (joint.score_selection_ref, decision.score),
        (joint.full_selection_ref, decision.full),
    ):
        selection = load_canonical_model(
            repository,
            selection_ref,
            BfclV4ArmSelection,
            media_type=BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
        )
        frozen_decision = load_canonical_model(
            repository,
            selection.decision_ref,
            BfclV4ArmSelectionDecision,
            media_type=BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE,
        )
        if frozen_decision != expected_decision:
            raise BfclV4PublicRunnerError("journal selection decision differs from joint evidence")


def _verify_plain_holdout(
    observation: BfclV4HoldoutObservation,
    record: BfclV4RunnerCallRecord,
) -> None:
    receipt = record.grader_receipt
    if record.prediction is None or receipt is None:
        raise BfclV4PublicRunnerError("HOLDOUT observation points to an ungraded call")
    if (
        observation.selected_prediction != record.prediction
        or observation.selected_prediction_ref != record.completion.prediction_ref
        or observation.grader_receipt_ref != record.completion.grader_receipt_ref
        or observation.accepted is not receipt.upstream_ast_checker_valid
        or receipt.holdout_unlock is None
        or observation.holdout_unlock_fingerprint != receipt.holdout_unlock.fingerprint
    ):
        raise BfclV4PublicRunnerError("HOLDOUT grade is detached from its completion receipt")


def _verify_pure_at_b_holdout(
    repository: ArtifactRepository,
    *,
    plan: BfclV4PublicPilotCallPlan,
    observation: BfclV4HoldoutObservation,
    group: tuple[BfclV4RunnerCallRecord, ...],
    adapter: BfclV4OpenAiToolAdapterResult,
) -> None:
    samples = tuple(item.pure_at_b_sample for item in group)
    if any(item is None for item in samples):
        raise BfclV4PublicRunnerError("PURE@B completion lacks its frozen sample")
    selection = select_bfcl_v4_pure_at_b_plurality(
        samples,  # type: ignore[arg-type]
        adapter.name_bindings,
    )
    prediction = prediction_from_wire_calls(
        task_id=observation.task_id,
        calls=selection.selected_calls,
        adapter=adapter,
    )
    stored_prediction = load_canonical_model(
        repository,
        observation.selected_prediction_ref,
        type(prediction),
        media_type=BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
    )
    receipt = load_canonical_model(
        repository,
        observation.grader_receipt_ref,
        BfclV4PublicGraderReceipt,
        media_type=BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
    )
    selected = group[selection.selected_frozen_index]
    binding = receipt.slot
    expected_binding = (
        plan.fingerprint,
        canonical_sha256(selected.slot),
        selected.slot.call_id,
        BfclV4PilotArm.PURE_AT_B.value,
        "pure-at-b-selected",
        selected.slot.harness_variant,
        selected.materialization.executed_harness_variant,
        selected.materialization.fallback_used,
        observation.task_id,
        prediction.fingerprint,
    )
    actual_binding = (
        binding.plan_fingerprint,
        binding.call_slot_reference_sha256,
        binding.call_id,
        binding.arm,
        binding.grade_role,
        binding.intended_harness_variant,
        binding.executed_harness_variant,
        binding.fallback_used,
        binding.task_id,
        binding.prediction_sha256,
    )
    if (
        observation.pure_at_b_selection_fingerprint != canonical_sha256(selection)
        or observation.selected_prediction != prediction
        or stored_prediction != prediction
        or receipt.prediction != prediction
        or actual_binding != expected_binding
        or observation.accepted is not receipt.upstream_ast_checker_valid
        or receipt.holdout_unlock is None
        or observation.holdout_unlock_fingerprint != receipt.holdout_unlock.fingerprint
        or observation.prediction_imputed_empty_for_failed_attempt
        is not (not any(item.provider_succeeded for item in group))
    ):
        raise BfclV4PublicRunnerError("PURE@B selection or grade is detached from frozen samples")


def _verify_holdout(
    repository: ArtifactRepository,
    *,
    plan: BfclV4PublicPilotCallPlan,
    records: tuple[BfclV4RunnerCallRecord, ...],
    adapters: dict[str, BfclV4OpenAiToolAdapterResult],
    decision: BfclV4JointSelectionDecision,
    holdout: BfclV4HoldoutEvidence,
    metrics: BfclV4PublicDescriptiveMetrics,
) -> None:
    for observation in holdout.observations:
        group = tuple(
            item
            for item in records
            if item.slot.arm is observation.arm and item.slot.task_id == observation.task_id
        )
        if (
            observation.source_call_ids != tuple(item.slot.call_id for item in group)
            or observation.source_completion_refs != tuple(item.completion_ref for item in group)
            or observation.source_provider_attempt_succeeded
            != tuple(item.provider_succeeded for item in group)
        ):
            raise BfclV4PublicRunnerError("HOLDOUT sources differ from closure completions")
        if observation.arm is BfclV4PilotArm.PURE_AT_B:
            _verify_pure_at_b_holdout(
                repository,
                plan=plan,
                observation=observation,
                group=group,
                adapter=adapters[observation.task_id],
            )
        elif len(group) != 1:
            raise BfclV4PublicRunnerError("single-call HOLDOUT group has the wrong size")
        else:
            _verify_plain_holdout(observation, group[0])
    rebuilt_holdout = build_bfcl_v4_holdout_evidence(
        plan=plan,
        joint_selection=decision,
        observations=holdout.observations,
    )
    rebuilt_metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=plan,
        joint_selection=decision,
        evidence=rebuilt_holdout,
    )
    if rebuilt_holdout != holdout or rebuilt_metrics != metrics:
        raise BfclV4PublicRunnerError("HOLDOUT evidence or metrics differ from replayed grades")


def verify_bfcl_v4_public_runner_evidence_lineage(
    repository: ArtifactRepository,
    *,
    plan: BfclV4PublicPilotCallPlan,
    result: BfclV4PublicPilotRunResult,
    closure: BfclV4RunClosure,
    records: tuple[BfclV4RunnerCallRecord, ...],
    tasks_by_id: dict[str, BfclV4PublicPilotTask],
    evidence: BfclV4JointSelectionEvidence,
    decision: BfclV4JointSelectionDecision,
    holdout: BfclV4HoldoutEvidence,
    metrics: BfclV4PublicDescriptiveMetrics,
) -> None:
    """Reconnect every score-bearing summary to the replayed 100-call closure."""

    if (
        result.candidate_freeze_ref != closure.candidate_freeze_ref
        or result.joint_selection_freeze_ref != closure.joint_selection_freeze_ref
        or len(records) != 100
        or tuple(item.completion_ref for item in records) != closure.call_completion_refs
    ):
        raise BfclV4PublicRunnerError("runner result evidence differs from semantic closure")
    resolutions = _candidate_resolutions(
        repository,
        plan=plan,
        result=result,
        records=records,
        tasks_by_id=tasks_by_id,
    )
    _verify_joint_selection(
        repository,
        plan=plan,
        result=result,
        records=records,
        evidence=evidence,
        decision=decision,
        resolutions=resolutions,
    )
    adapters = {
        task_id: adapt_bfcl_v4_public_pilot_task(task) for task_id, task in tasks_by_id.items()
    }
    _verify_holdout(
        repository,
        plan=plan,
        records=records,
        adapters=adapters,
        decision=decision,
        holdout=holdout,
        metrics=metrics,
    )


__all__ = ["verify_bfcl_v4_public_runner_evidence_lineage"]
