"""Post-selection HOLDOUT and target-free PURE@B evidence assembly."""

from __future__ import annotations

from pathlib import Path

from spiral_harness.benchmark.bfcl_v4_public_grader import (
    grade_bfcl_v4_public_prediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4GradingSlotBinding,
    BfclV4HoldoutUnlock,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    select_bfcl_v4_pure_at_b_plurality,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4OpenAiToolAdapterResult,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
    BFCL_V4_RUNNER_PURE_AT_B_SELECTION_MEDIA_TYPE,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    BfclV4RunnerCallRecord,
    prediction_from_wire_calls,
    publish_model,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4HoldoutObservation,
    BfclV4JointSelectionDecision,
)
from spiral_harness.experiments.bfcl_v4_public_selection_logic import (
    build_bfcl_v4_holdout_evidence,
    compute_bfcl_v4_public_descriptive_metrics,
)
from spiral_harness.storage.protocol import ArtifactRepository


def _pure_at_b_observation(
    *,
    repository: ArtifactRepository,
    checkout: Path,
    plan: BfclV4PublicPilotCallPlan,
    selection_decision: BfclV4JointSelectionDecision,
    records: tuple[BfclV4RunnerCallRecord, ...],
    adapters: dict[str, BfclV4OpenAiToolAdapterResult],
    task_id: str,
    unlock: BfclV4HoldoutUnlock,
) -> BfclV4HoldoutObservation:
    group = tuple(
        item
        for item in records
        if item.slot.arm is BfclV4PilotArm.PURE_AT_B and item.slot.task_id == task_id
    )
    samples = tuple(item.pure_at_b_sample for item in group)
    if not group or any(item is None for item in samples):
        raise BfclV4PublicRunnerError("PURE@B group lacks frozen samples")
    adapter = adapters[task_id]
    target_free = select_bfcl_v4_pure_at_b_plurality(
        samples,  # type: ignore[arg-type]
        adapter.name_bindings,
    )
    publish_model(
        repository,
        target_free,
        media_type=BFCL_V4_RUNNER_PURE_AT_B_SELECTION_MEDIA_TYPE,
    )
    prediction = prediction_from_wire_calls(
        task_id=task_id,
        calls=target_free.selected_calls,
        adapter=adapter,
    )
    prediction_ref = publish_model(
        repository,
        prediction,
        media_type=BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE,
    )
    selected = group[target_free.selected_frozen_index]
    binding = BfclV4GradingSlotBinding(
        plan_fingerprint=plan.fingerprint,
        call_slot_reference_sha256=canonical_sha256(selected.slot),
        call_id=selected.slot.call_id,
        arm=BfclV4PilotArm.PURE_AT_B.value,
        grade_role="pure-at-b-selected",
        intended_harness_variant=selected.slot.harness_variant,
        executed_harness_variant=selected.materialization.executed_harness_variant,
        fallback_used=selected.materialization.fallback_used,
        task_id=task_id,
        prediction_sha256=prediction.fingerprint,
    )
    receipt = grade_bfcl_v4_public_prediction(
        prediction,
        binding,
        checkout,
        holdout_unlock=unlock,
    )
    receipt_ref = publish_model(
        repository,
        receipt,
        media_type=BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE,
    )
    all_failed = not any(item.provider_succeeded for item in group)
    return BfclV4HoldoutObservation(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        joint_selection_decision_fingerprint=selection_decision.fingerprint,
        task_id=task_id,
        arm=BfclV4PilotArm.PURE_AT_B,
        source_call_ids=tuple(item.slot.call_id for item in group),
        source_completion_refs=tuple(item.completion_ref for item in group),
        source_provider_attempt_succeeded=tuple(item.provider_succeeded for item in group),
        pure_at_b_selection_fingerprint=canonical_sha256(target_free),
        prediction_imputed_empty_for_failed_attempt=all_failed,
        selected_prediction=prediction,
        selected_prediction_ref=prediction_ref,
        grader_receipt_ref=receipt_ref,
        holdout_unlock_fingerprint=unlock.fingerprint,
        accepted=receipt.upstream_ast_checker_valid,
    )


def finish_runner_holdout_evidence(
    *,
    repository: ArtifactRepository,
    checkout: Path,
    plan: BfclV4PublicPilotCallPlan,
    selection_decision: BfclV4JointSelectionDecision,
    records: tuple[BfclV4RunnerCallRecord, ...],
    adapters: dict[str, BfclV4OpenAiToolAdapterResult],
    unlock: BfclV4HoldoutUnlock,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Bind all 60 post-selection calls into 40 graded arm/task cells."""

    if len(records) != 100:
        raise BfclV4PublicRunnerError("HOLDOUT evidence requires all 100 calls")
    holdout_ids = tuple(
        item.task_id
        for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
        if item.split.value == "holdout"
    )
    arm_decisions = {
        BfclV4PilotArm.SCORE: selection_decision.score.fingerprint,
        BfclV4PilotArm.FULL: selection_decision.full.fingerprint,
    }
    observations = []
    for arm in (
        BfclV4PilotArm.PURE,
        BfclV4PilotArm.STATIC,
        BfclV4PilotArm.SCORE,
        BfclV4PilotArm.FULL,
    ):
        for task_id in holdout_ids:
            matches = tuple(
                item for item in records if item.slot.arm is arm and item.slot.task_id == task_id
            )
            if len(matches) != 1:
                raise BfclV4PublicRunnerError("single-call HOLDOUT evidence is incomplete")
            item = matches[0]
            if item.prediction is None or item.grader_receipt is None:
                raise BfclV4PublicRunnerError("HOLDOUT call lacks explicit grade evidence")
            if item.completion.prediction_ref is None or item.completion.grader_receipt_ref is None:
                raise BfclV4PublicRunnerError("HOLDOUT completion lacks grade references")
            observations.append(
                BfclV4HoldoutObservation(
                    plan_fingerprint=plan.fingerprint,
                    schedule_content_sha256=plan.schedule_content_sha256,
                    joint_selection_decision_fingerprint=selection_decision.fingerprint,
                    arm_selection_decision_fingerprint=arm_decisions.get(arm),
                    task_id=task_id,
                    arm=arm,
                    source_call_ids=(item.slot.call_id,),
                    source_completion_refs=(item.completion_ref,),
                    source_provider_attempt_succeeded=(item.provider_succeeded,),
                    prediction_imputed_empty_for_failed_attempt=not item.provider_succeeded,
                    selected_prediction=item.prediction,
                    selected_prediction_ref=item.completion.prediction_ref,
                    grader_receipt_ref=item.completion.grader_receipt_ref,
                    holdout_unlock_fingerprint=unlock.fingerprint,
                    accepted=item.accepted,
                )
            )
    observations.extend(
        _pure_at_b_observation(
            repository=repository,
            checkout=checkout,
            plan=plan,
            selection_decision=selection_decision,
            records=records,
            adapters=adapters,
            task_id=task_id,
            unlock=unlock,
        )
        for task_id in holdout_ids
    )
    evidence = build_bfcl_v4_holdout_evidence(
        plan=plan,
        joint_selection=selection_decision,
        observations=tuple(observations),
    )
    evidence_ref = publish_model(
        repository,
        evidence,
        media_type=BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    )
    metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=plan,
        joint_selection=selection_decision,
        evidence=evidence,
    )
    metrics_ref = publish_model(
        repository,
        metrics,
        media_type=BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    )
    return evidence_ref, metrics_ref


__all__ = ["finish_runner_holdout_evidence"]
