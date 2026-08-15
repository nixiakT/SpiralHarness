"""Candidate-safe diagnosis and offline selection joins for the BFCL runner."""

from __future__ import annotations

from spiral_harness.benchmark.bfcl_v4_public_grader import (
    project_bfcl_v4_full_fit_feedback,
    project_bfcl_v4_score_fit_aggregate,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4PublicPilotTask,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BfclV4ArmSelection,
    BfclV4JointCandidateFreeze,
    BfclV4SelectedVariant,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4CandidateResolution,
    BfclV4DiagnosisPrompt,
    BfclV4FullFitDiagnosisBatch,
    BfclV4FullFitDiagnosisObservation,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    build_bfcl_v4_full_diagnosis_prompt,
    build_bfcl_v4_score_diagnosis_prompt,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    BfclV4RunnerCallRecord,
    load_canonical_model,
    publish_model,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4JointSelectionDecision,
    BfclV4SelectionObservation,
)
from spiral_harness.experiments.bfcl_v4_public_selection_logic import (
    build_bfcl_v4_joint_selection_evidence,
    select_bfcl_v4_public_candidates,
)
from spiral_harness.storage.protocol import ArtifactRepository


def build_runner_diagnosis_prompt(
    *,
    arm: BfclV4PilotArm,
    parent_system_prompt: str,
    records: tuple[BfclV4RunnerCallRecord, ...],
    tasks_by_id: dict[str, BfclV4PublicPilotTask],
) -> BfclV4DiagnosisPrompt:
    """Project only the arm-authorized five parent-FIT observations."""

    fit = tuple(
        item
        for item in records
        if item.slot.arm is arm and item.slot.kind is BfclV4PilotCallKind.PARENT_FIT
    )
    if len(fit) != 5:
        raise BfclV4PublicRunnerError("diagnosis requires five completed parent-FIT calls")
    if any(item.prediction is None or item.grader_receipt is None for item in fit):
        raise BfclV4PublicRunnerError("parent-FIT record lacks explicit prediction or grade")

    if arm is BfclV4PilotArm.SCORE:
        receipts = tuple(item.grader_receipt for item in fit)
        assert all(item is not None for item in receipts)
        aggregate = project_bfcl_v4_score_fit_aggregate(
            receipts,  # type: ignore[arg-type]
            tuple(canonical_sha256(item.slot) for item in fit),
        )
        return build_bfcl_v4_score_diagnosis_prompt(
            parent_system_prompt=parent_system_prompt,
            aggregate=aggregate,
        )
    if arm is not BfclV4PilotArm.FULL:
        raise BfclV4PublicRunnerError("only SCORE and FULL have diagnosis calls")

    observations = []
    for item in fit:
        assert item.prediction is not None and item.grader_receipt is not None
        task_id = item.slot.task_id
        if task_id is None or task_id not in tasks_by_id:
            raise BfclV4PublicRunnerError("FULL parent-FIT task is unavailable")
        observations.append(
            BfclV4FullFitDiagnosisObservation(
                task=tasks_by_id[task_id],
                own_prediction=item.prediction,
                feedback=project_bfcl_v4_full_fit_feedback(
                    item.prediction,
                    item.grader_receipt,
                ),
            )
        )
    return build_bfcl_v4_full_diagnosis_prompt(
        parent_system_prompt=parent_system_prompt,
        batch=BfclV4FullFitDiagnosisBatch(observations=tuple(observations)),
    )


def _selection_observation(
    plan: BfclV4PublicPilotCallPlan,
    record: BfclV4RunnerCallRecord,
) -> BfclV4SelectionObservation:
    if record.prediction is None or record.grader_receipt is None:
        raise BfclV4PublicRunnerError("selection record lacks explicit prediction or grade")
    return BfclV4SelectionObservation(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        arm=record.slot.arm,
        slot=record.slot,
        slot_reference_sha256=canonical_sha256(record.slot),
        materialization=record.materialization,
        completion=record.completion,
        completion_ref=record.completion_ref,
        provider_attempt_succeeded=record.provider_succeeded,
        prediction_imputed_empty_for_failed_attempt=not record.provider_succeeded,
        prediction=record.prediction,
        grader_receipt_reference_sha256=record.grader_receipt.fingerprint,
        accepted=record.accepted,
    )


def freeze_runner_selection_inputs(
    *,
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    candidate_freeze_ref: ArtifactRef,
    records: tuple[BfclV4RunnerCallRecord, ...],
    resolutions: dict[BfclV4PilotArm, BfclV4CandidateResolution],
) -> tuple[
    BfclV4JointSelectionDecision,
    ArtifactRef,
    ArtifactRef,
    BfclV4ArmSelection,
    BfclV4ArmSelection,
]:
    """Compute both arm decisions together after all 16 GATE completions."""

    selection_kinds = {
        BfclV4PilotCallKind.PARENT_FIT,
        BfclV4PilotCallKind.CANDIDATE_FIT,
        BfclV4PilotCallKind.GATE,
    }
    selection_records = tuple(item for item in records if item.slot.kind in selection_kinds)
    if len(selection_records) != 36 or selection_records[-1].slot.global_slot != 39:
        raise BfclV4PublicRunnerError("joint selection requires the complete 36-call evidence set")
    observations = tuple(_selection_observation(plan, item) for item in selection_records)
    score_resolution = resolutions[BfclV4PilotArm.SCORE]
    full_resolution = resolutions[BfclV4PilotArm.FULL]
    evidence = build_bfcl_v4_joint_selection_evidence(
        plan=plan,
        score_resolution=score_resolution,
        full_resolution=full_resolution,
        observations=observations,
    )
    evidence_ref = publish_model(
        repository,
        evidence,
        media_type=BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE,
    )
    decision = select_bfcl_v4_public_candidates(plan=plan, evidence=evidence)
    decision_ref = publish_model(
        repository,
        decision,
        media_type=BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
    )
    score_decision_ref = publish_model(
        repository,
        decision.score,
        media_type=BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE,
    )
    full_decision_ref = publish_model(
        repository,
        decision.full,
        media_type=BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE,
    )
    by_arm = {
        BfclV4PilotArm.SCORE: (score_resolution, decision.score, score_decision_ref),
        BfclV4PilotArm.FULL: (full_resolution, decision.full, full_decision_ref),
    }
    gate_refs = {
        arm: tuple(
            item.completion_ref
            for item in selection_records
            if item.slot.arm is arm and item.slot.kind is BfclV4PilotCallKind.GATE
        )
        for arm in by_arm
    }
    selections: dict[BfclV4PilotArm, BfclV4ArmSelection] = {}
    for arm, (resolution, arm_decision, arm_decision_ref) in by_arm.items():
        selected_candidate = arm_decision.selected_variant == "candidate"
        selected_ref = (
            _candidate_harness_ref(repository, candidate_freeze_ref, arm)
            if selected_candidate
            else _parent_harness_ref(repository, candidate_freeze_ref, arm)
        )
        selections[arm] = BfclV4ArmSelection(
            plan_fingerprint=plan.fingerprint,
            arm=arm,
            candidate_freeze_ref=candidate_freeze_ref,
            gate_completion_refs=gate_refs[arm],
            decision_ref=arm_decision_ref,
            selected_variant=(
                BfclV4SelectedVariant.CANDIDATE
                if selected_candidate
                else BfclV4SelectedVariant.PARENT
            ),
            selected_harness_ref=selected_ref,
            forced_rollback=resolution.forced_rollback,
        )
    return (
        decision,
        evidence_ref,
        decision_ref,
        selections[BfclV4PilotArm.SCORE],
        selections[BfclV4PilotArm.FULL],
    )


def _candidate_freeze(repository: ArtifactRepository, ref: ArtifactRef):
    return load_canonical_model(
        repository,
        ref,
        BfclV4JointCandidateFreeze,
        media_type=BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    )


def _parent_harness_ref(
    repository: ArtifactRepository,
    freeze_ref: ArtifactRef,
    arm: BfclV4PilotArm,
) -> ArtifactRef:
    freeze = _candidate_freeze(repository, freeze_ref)
    return (
        freeze.score.parent_harness_ref
        if arm is BfclV4PilotArm.SCORE
        else freeze.full.parent_harness_ref
    )


def _candidate_harness_ref(
    repository: ArtifactRepository,
    freeze_ref: ArtifactRef,
    arm: BfclV4PilotArm,
) -> ArtifactRef:
    freeze = _candidate_freeze(repository, freeze_ref)
    candidate = freeze.score if arm is BfclV4PilotArm.SCORE else freeze.full
    return candidate.effective_candidate_harness_ref


__all__ = ["build_runner_diagnosis_prompt", "freeze_runner_selection_inputs"]
