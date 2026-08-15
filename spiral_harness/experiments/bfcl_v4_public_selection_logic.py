"""Pure offline selection and integer descriptive metrics for the BFCL V4 pilot."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4AdaptiveArm,
    BfclV4CandidateResolution,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BFCL_V4_HOLDOUT_TASK_IDS,
    BFCL_V4_METRIC_ARM_ORDER,
    BFCL_V4_PAIRED_CONTRASTS,
    BfclV4ArmDescriptiveMetric,
    BfclV4ArmSelectionDecision,
    BfclV4ArmSelectionEvidence,
    BfclV4HoldoutEvidence,
    BfclV4HoldoutObservation,
    BfclV4JointSelectionDecision,
    BfclV4JointSelectionEvidence,
    BfclV4PairedDescriptiveDelta,
    BfclV4PublicDescriptiveMetrics,
    BfclV4RollbackReason,
    BfclV4SelectionDecision,
    BfclV4SelectionObservation,
)

_SELECTION_KINDS = {
    BfclV4PilotCallKind.PARENT_FIT,
    BfclV4PilotCallKind.CANDIDATE_FIT,
    BfclV4PilotCallKind.GATE,
}


class BfclV4PublicSelectionError(ValueError):
    """Raised when public-pilot evidence cannot close the frozen offline rule."""


def _checked[ModelT: BaseModel](value: object, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate(value, strict=True)
    except Exception as exc:
        raise BfclV4PublicSelectionError(f"{label} failed strict revalidation") from exc


def _checked_plan(plan: object) -> BfclV4PublicPilotCallPlan:
    return _checked(plan, BfclV4PublicPilotCallPlan, "BFCL public call plan")


def _arm_slots(plan: BfclV4PublicPilotCallPlan, arm: BfclV4PilotArm) -> tuple[object, ...]:
    return tuple(slot for slot in plan.calls if slot.arm is arm and slot.kind in _SELECTION_KINDS)


def build_bfcl_v4_joint_selection_evidence(
    *,
    plan: BfclV4PublicPilotCallPlan,
    score_resolution: BfclV4CandidateResolution,
    full_resolution: BfclV4CandidateResolution,
    observations: tuple[BfclV4SelectionObservation, ...],
) -> BfclV4JointSelectionEvidence:
    """Bind both arms only after the complete global 36-observation FIT/GATE prefix."""

    checked_plan = _checked_plan(plan)
    score = _checked(score_resolution, BfclV4CandidateResolution, "SCORE resolution")
    full = _checked(full_resolution, BfclV4CandidateResolution, "FULL resolution")
    if score.arm is not BfclV4AdaptiveArm.SCORE or full.arm is not BfclV4AdaptiveArm.FULL:
        raise BfclV4PublicSelectionError("candidate resolutions must be SCORE then FULL")
    if not isinstance(observations, tuple):
        raise BfclV4PublicSelectionError("selection observations must be an exact tuple")
    checked_observations = tuple(
        _checked(item, BfclV4SelectionObservation, "selection observation") for item in observations
    )
    expected_slots = tuple(slot for slot in checked_plan.calls if slot.kind in _SELECTION_KINDS)
    if len(expected_slots) != 36 or len(checked_observations) != 36:
        raise BfclV4PublicSelectionError("joint selection requires all 36 FIT/GATE observations")
    for expected, observation in zip(expected_slots, checked_observations, strict=True):
        if observation.slot != expected:
            raise BfclV4PublicSelectionError(
                "selection observations differ from the frozen global slot order"
            )
        if (
            observation.plan_fingerprint != checked_plan.fingerprint
            or observation.schedule_content_sha256 != checked_plan.schedule_content_sha256
        ):
            raise BfclV4PublicSelectionError("selection observation belongs to another plan")

    freeze_refs = {
        item.materialization.candidate_freeze_ref
        for item in checked_observations
        if item.materialization.candidate_freeze_ref is not None
    }
    if len(freeze_refs) != 1:
        raise BfclV4PublicSelectionError("FIT/GATE observations do not share one candidate freeze")
    candidate_freeze_ref = freeze_refs.pop()
    arm_evidence: dict[BfclV4PilotArm, BfclV4ArmSelectionEvidence] = {}
    for arm, resolution in (
        (BfclV4PilotArm.SCORE, score),
        (BfclV4PilotArm.FULL, full),
    ):
        arm_observations = tuple(item for item in checked_observations if item.arm is arm)
        if tuple(item.slot for item in arm_observations) != _arm_slots(checked_plan, arm):
            raise BfclV4PublicSelectionError("arm evidence differs from its frozen slot order")
        arm_evidence[arm] = BfclV4ArmSelectionEvidence(
            plan_fingerprint=checked_plan.fingerprint,
            schedule_content_sha256=checked_plan.schedule_content_sha256,
            arm=arm,
            candidate_freeze_ref=candidate_freeze_ref,
            candidate_resolution=resolution,
            candidate_resolution_fingerprint=resolution.fingerprint,
            observations=arm_observations,
        )
    gate_refs = tuple(
        item.completion_ref
        for item in checked_observations
        if item.slot.kind is BfclV4PilotCallKind.GATE
    )
    return BfclV4JointSelectionEvidence(
        plan_fingerprint=checked_plan.fingerprint,
        schedule_content_sha256=checked_plan.schedule_content_sha256,
        candidate_freeze_ref=candidate_freeze_ref,
        score=arm_evidence[BfclV4PilotArm.SCORE],
        full=arm_evidence[BfclV4PilotArm.FULL],
        global_gate_completion_refs=gate_refs,
    )


def _require_joint_lineage(
    plan: BfclV4PublicPilotCallPlan,
    evidence: BfclV4JointSelectionEvidence,
) -> None:
    if (
        evidence.plan_fingerprint != plan.fingerprint
        or evidence.schedule_content_sha256 != plan.schedule_content_sha256
    ):
        raise BfclV4PublicSelectionError("joint evidence belongs to another call plan")
    all_observations = tuple(
        sorted(
            (*evidence.score.observations, *evidence.full.observations),
            key=lambda item: item.slot.global_slot,
        )
    )
    expected = tuple(slot for slot in plan.calls if slot.kind in _SELECTION_KINDS)
    if tuple(item.slot for item in all_observations) != expected:
        raise BfclV4PublicSelectionError("joint evidence does not cover the plan's FIT/GATE slots")


def _shadow_controls_exact(evidence: BfclV4ArmSelectionEvidence) -> bool:
    gates = {
        (item.slot.task_id, item.slot.harness_variant): item
        for item in evidence.observations
        if item.slot.kind is BfclV4PilotCallKind.GATE
    }
    for task_id in {item.slot.task_id for item in gates.values()}:
        controls = tuple(gates[(task_id, variant)] for variant in ("parent", "revert", "placebo"))
        if any(not item.provider_attempt_succeeded or item.prediction is None for item in controls):
            return False
        predictions = {
            item.prediction.fingerprint for item in controls if item.prediction is not None
        }
        grades = {item.accepted for item in controls}
        if len(predictions) != 1 or len(grades) != 1:
            return False
    return True


def _arm_decision(
    evidence: BfclV4ArmSelectionEvidence,
    *,
    joint_evidence_fingerprint: str,
) -> BfclV4ArmSelectionDecision:
    parent_fit = tuple(
        item for item in evidence.observations if item.slot.kind is BfclV4PilotCallKind.PARENT_FIT
    )
    candidate_fit = tuple(
        item
        for item in evidence.observations
        if item.slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT
    )
    parent_gate = tuple(
        item
        for item in evidence.observations
        if item.slot.kind is BfclV4PilotCallKind.GATE and item.slot.harness_variant == "parent"
    )
    candidate_gate = tuple(
        item
        for item in evidence.observations
        if item.slot.kind is BfclV4PilotCallKind.GATE and item.slot.harness_variant == "candidate"
    )
    parent_fit_correct = sum(item.accepted for item in parent_fit)
    candidate_fit_correct = sum(item.accepted for item in candidate_fit)
    parent_gate_correct = sum(item.accepted for item in parent_gate)
    candidate_gate_correct = sum(item.accepted for item in candidate_gate)
    all_succeeded = all(item.provider_attempt_succeeded for item in evidence.observations)
    controls_exact = _shadow_controls_exact(evidence)
    fit_nondecreasing = candidate_fit_correct >= parent_fit_correct
    gate_better = candidate_gate_correct > parent_gate_correct
    admissible = evidence.candidate_resolution.candidate_admissible
    reasons = tuple(
        reason
        for failed, reason in (
            (not admissible, BfclV4RollbackReason.CANDIDATE_INVALID),
            (not all_succeeded, BfclV4RollbackReason.PROVIDER_ATTEMPT_FAILURE),
            (not controls_exact, BfclV4RollbackReason.SHADOW_CONTROL_MISMATCH),
            (not fit_nondecreasing, BfclV4RollbackReason.CANDIDATE_FIT_REGRESSION),
            (not gate_better, BfclV4RollbackReason.CANDIDATE_GATE_NOT_STRICTLY_BETTER),
        )
        if failed
    )
    promote = not reasons
    resolution = evidence.candidate_resolution
    candidate_sha = resolution.candidate_parse_result.candidate_system_prompt_sha256
    return BfclV4ArmSelectionDecision(
        plan_fingerprint=evidence.plan_fingerprint,
        schedule_content_sha256=evidence.schedule_content_sha256,
        arm=evidence.arm,
        candidate_freeze_reference_sha256=evidence.candidate_freeze_ref.sha256,
        joint_evidence_fingerprint=joint_evidence_fingerprint,
        arm_evidence_fingerprint=evidence.fingerprint,
        candidate_resolution_fingerprint=evidence.candidate_resolution_fingerprint,
        candidate_admissible=admissible,
        all_fit_gate_provider_attempts_succeeded=all_succeeded,
        shadow_controls_exact=controls_exact,
        parent_fit_correct=parent_fit_correct,
        candidate_fit_correct=candidate_fit_correct,
        parent_gate_correct=parent_gate_correct,
        candidate_gate_correct=candidate_gate_correct,
        candidate_fit_nondecreasing=fit_nondecreasing,
        candidate_gate_strictly_better=gate_better,
        decision=(BfclV4SelectionDecision.PROMOTE if promote else BfclV4SelectionDecision.ROLLBACK),
        selected_variant="candidate" if promote else "parent",
        parent_system_prompt_sha256=resolution.parent_system_prompt_sha256,
        candidate_system_prompt_sha256=candidate_sha,
        selected_system_prompt_sha256=(
            resolution.evaluation_system_prompt_sha256
            if promote
            else resolution.parent_system_prompt_sha256
        ),
        forced_rollback=not admissible,
        rollback_reasons=reasons,
    )


def select_bfcl_v4_public_candidates(
    *,
    plan: BfclV4PublicPilotCallPlan,
    evidence: BfclV4JointSelectionEvidence,
) -> BfclV4JointSelectionDecision:
    """Compute SCORE and FULL together; no per-arm early-selection API exists."""

    checked_plan = _checked_plan(plan)
    checked_evidence = _checked(evidence, BfclV4JointSelectionEvidence, "joint evidence")
    _require_joint_lineage(checked_plan, checked_evidence)
    joint_fingerprint = checked_evidence.fingerprint
    score = _arm_decision(checked_evidence.score, joint_evidence_fingerprint=joint_fingerprint)
    full = _arm_decision(checked_evidence.full, joint_evidence_fingerprint=joint_fingerprint)
    return BfclV4JointSelectionDecision(
        plan_fingerprint=checked_plan.fingerprint,
        schedule_content_sha256=checked_plan.schedule_content_sha256,
        candidate_freeze_reference_sha256=checked_evidence.candidate_freeze_ref.sha256,
        joint_evidence_fingerprint=joint_fingerprint,
        global_gate_completion_refs=checked_evidence.global_gate_completion_refs,
        score=score,
        full=full,
    )


def _expected_holdout_call_ids(
    plan: BfclV4PublicPilotCallPlan,
    observation: BfclV4HoldoutObservation,
) -> tuple[str, ...]:
    return tuple(
        slot.call_id
        for slot in plan.calls
        if slot.arm is observation.arm and slot.task_id == observation.task_id
    )


def build_bfcl_v4_holdout_evidence(
    *,
    plan: BfclV4PublicPilotCallPlan,
    joint_selection: BfclV4JointSelectionDecision,
    observations: tuple[BfclV4HoldoutObservation, ...],
) -> BfclV4HoldoutEvidence:
    """Bind the 40 final grades to all 60 frozen post-selection call completions."""

    checked_plan = _checked_plan(plan)
    selection = _checked(joint_selection, BfclV4JointSelectionDecision, "joint selection decision")
    if (
        selection.plan_fingerprint != checked_plan.fingerprint
        or selection.schedule_content_sha256 != checked_plan.schedule_content_sha256
    ):
        raise BfclV4PublicSelectionError("joint selection belongs to another call plan")
    if not isinstance(observations, tuple):
        raise BfclV4PublicSelectionError("HOLDOUT observations must be an exact tuple")
    checked_observations = tuple(
        _checked(item, BfclV4HoldoutObservation, "HOLDOUT observation") for item in observations
    )
    expected_keys = tuple(
        (arm, task_id) for arm in BFCL_V4_METRIC_ARM_ORDER for task_id in BFCL_V4_HOLDOUT_TASK_IDS
    )
    if tuple((item.arm, item.task_id) for item in checked_observations) != expected_keys:
        raise BfclV4PublicSelectionError("HOLDOUT observations require the exact arm-major matrix")
    selection_fingerprint = selection.fingerprint
    arm_fingerprints = {
        BfclV4PilotArm.SCORE: selection.score.fingerprint,
        BfclV4PilotArm.FULL: selection.full.fingerprint,
    }
    for item in checked_observations:
        if (
            item.plan_fingerprint != checked_plan.fingerprint
            or item.schedule_content_sha256 != checked_plan.schedule_content_sha256
            or item.joint_selection_decision_fingerprint != selection_fingerprint
            or item.source_call_ids != _expected_holdout_call_ids(checked_plan, item)
        ):
            raise BfclV4PublicSelectionError("HOLDOUT observation differs from its frozen lineage")
        if (
            item.arm in arm_fingerprints
            and item.arm_selection_decision_fingerprint != arm_fingerprints[item.arm]
        ):
            raise BfclV4PublicSelectionError("adaptive HOLDOUT uses another arm decision")
    unlocks = {
        item.holdout_unlock_fingerprint
        for item in checked_observations
        if item.holdout_unlock_fingerprint is not None
    }
    if len(unlocks) != 1:
        raise BfclV4PublicSelectionError("graded HOLDOUT observations do not share one unlock")
    return BfclV4HoldoutEvidence(
        plan_fingerprint=checked_plan.fingerprint,
        schedule_content_sha256=checked_plan.schedule_content_sha256,
        joint_selection_decision_fingerprint=selection_fingerprint,
        observations=checked_observations,
    )


def compute_bfcl_v4_public_descriptive_metrics(
    *,
    plan: BfclV4PublicPilotCallPlan,
    joint_selection: BfclV4JointSelectionDecision,
    evidence: BfclV4HoldoutEvidence,
) -> BfclV4PublicDescriptiveMetrics:
    """Return only exact public-development accuracies and paired integer deltas."""

    checked_plan = _checked_plan(plan)
    selection = _checked(joint_selection, BfclV4JointSelectionDecision, "joint selection decision")
    holdout = _checked(evidence, BfclV4HoldoutEvidence, "HOLDOUT evidence")
    if (
        selection.plan_fingerprint != checked_plan.fingerprint
        or selection.schedule_content_sha256 != checked_plan.schedule_content_sha256
        or holdout.plan_fingerprint != checked_plan.fingerprint
        or holdout.schedule_content_sha256 != checked_plan.schedule_content_sha256
        or holdout.joint_selection_decision_fingerprint != selection.fingerprint
    ):
        raise BfclV4PublicSelectionError("descriptive inputs do not share one frozen lineage")
    for item in holdout.observations:
        if item.source_call_ids != _expected_holdout_call_ids(checked_plan, item):
            raise BfclV4PublicSelectionError("HOLDOUT source calls differ from the frozen plan")

    correctness = {
        arm: tuple(item.accepted for item in holdout.observations if item.arm is arm)
        for arm in BFCL_V4_METRIC_ARM_ORDER
    }
    arm_metrics = tuple(
        BfclV4ArmDescriptiveMetric(
            arm=arm,
            task_ids=BFCL_V4_HOLDOUT_TASK_IDS,
            correctness=correctness[arm],
            correct_count=sum(correctness[arm]),
            accuracy_basis_points=sum(correctness[arm]) * 1_250,
        )
        for arm in BFCL_V4_METRIC_ARM_ORDER
    )
    paired = []
    for treatment, reference in BFCL_V4_PAIRED_CONTRASTS:
        treatment_vector = correctness[treatment]
        reference_vector = correctness[reference]
        wins = sum(
            left and not right
            for left, right in zip(treatment_vector, reference_vector, strict=True)
        )
        losses = sum(
            right and not left
            for left, right in zip(treatment_vector, reference_vector, strict=True)
        )
        delta = (wins - losses) * 1_250
        paired.append(
            BfclV4PairedDescriptiveDelta(
                treatment_arm=treatment,
                reference_arm=reference,
                task_ids=BFCL_V4_HOLDOUT_TASK_IDS,
                treatment_correctness=treatment_vector,
                reference_correctness=reference_vector,
                wins=wins,
                ties=8 - wins - losses,
                losses=losses,
                delta_basis_points=delta,
                strictly_exceeds_positive_ten_percentage_points=delta > 1_000,
            )
        )
    return BfclV4PublicDescriptiveMetrics(
        plan_fingerprint=checked_plan.fingerprint,
        schedule_content_sha256=checked_plan.schedule_content_sha256,
        joint_selection_decision_fingerprint=selection.fingerprint,
        holdout_evidence_fingerprint=holdout.fingerprint,
        arms=arm_metrics,
        paired_deltas=tuple(paired),
    )


__all__ = [
    "BfclV4PublicSelectionError",
    "build_bfcl_v4_holdout_evidence",
    "build_bfcl_v4_joint_selection_evidence",
    "compute_bfcl_v4_public_descriptive_metrics",
    "select_bfcl_v4_public_candidates",
]
