from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4OfficialPredictionCall,
    BfclV4PublicPrediction,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4CallOutcome,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4AdaptiveArm,
    BfclV4CandidateParseFailure,
    BfclV4CandidateParseResult,
    BfclV4CandidateResolution,
    BfclV4DiagnosisFailure,
    BfclV4DiagnosisParseResult,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_CANDIDATE_SUBMIT_TOOL,
    resolve_bfcl_v4_candidate,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BFCL_V4_HOLDOUT_TASK_IDS,
    BFCL_V4_METRIC_ARM_ORDER,
    BfclV4HoldoutObservation,
    BfclV4PairedDescriptiveDelta,
    BfclV4RollbackReason,
    BfclV4SelectionDecision,
    BfclV4SelectionObservation,
)
from spiral_harness.experiments.bfcl_v4_public_selection_logic import (
    BfclV4PublicSelectionError,
    build_bfcl_v4_holdout_evidence,
    build_bfcl_v4_joint_selection_evidence,
    compute_bfcl_v4_public_descriptive_metrics,
    select_bfcl_v4_public_candidates,
)


def _sha(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _opaque_ref(label: str, media_type: str = "application/json") -> ArtifactRef:
    payload = label.encode("utf-8")
    return ArtifactRef(sha256=sha256_bytes(payload), size=len(payload), media_type=media_type)


def _exact_ref(value: object, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(sha256=sha256_bytes(payload), size=len(payload), media_type=media_type)


def _resolution(
    arm: BfclV4AdaptiveArm,
    *,
    admissible: bool = True,
) -> BfclV4CandidateResolution:
    diagnosis_text = f"diagnosis-{arm.value}"
    diagnosis = BfclV4DiagnosisParseResult(
        arm=arm,
        diagnosis_prompt_fingerprint=_sha(f"diagnosis-prompt-{arm.value}"),
        native_response_fingerprint=_sha(f"diagnosis-response-{arm.value}"),
        valid=True,
        failure=BfclV4DiagnosisFailure.NONE,
        diagnosis_text=diagnosis_text,
        diagnosis_text_sha256=_sha(diagnosis_text),
    )
    parent = "frozen parent system prompt"
    proposal = BfclV4ProposalPrompt(
        arm=arm,
        feedback_view="score-only" if arm is BfclV4AdaptiveArm.SCORE else "candidate-safe-full",
        system_prompt="frozen proposer",
        user_prompt="submit one strategy",
        parent_system_prompt=parent,
        parent_system_prompt_sha256=_sha(parent),
        diagnosis_result_fingerprint=diagnosis.fingerprint,
        diagnosis_valid=True,
        submit_tool=BFCL_V4_CANDIDATE_SUBMIT_TOOL,
        submit_tool_fingerprint=canonical_sha256(BFCL_V4_CANDIDATE_SUBMIT_TOOL),
    )
    if admissible:
        strategy = f"strategy-{arm.value}"
        candidate_prompt = f"candidate system prompt for {arm.value}"
        candidate = BfclV4CandidateParseResult(
            arm=arm,
            proposal_prompt_fingerprint=proposal.fingerprint,
            native_response_fingerprint=_sha(f"proposal-response-{arm.value}"),
            valid=True,
            failure=BfclV4CandidateParseFailure.NONE,
            strategy_text=strategy,
            strategy_text_sha256=_sha(strategy),
            candidate_system_prompt=candidate_prompt,
            candidate_system_prompt_sha256=_sha(candidate_prompt),
        )
    else:
        candidate = BfclV4CandidateParseResult(
            arm=arm,
            proposal_prompt_fingerprint=proposal.fingerprint,
            native_response_fingerprint=None,
            valid=False,
            failure=BfclV4CandidateParseFailure.NO_VERIFIED_RESPONSE,
        )
    return resolve_bfcl_v4_candidate(
        diagnosis_result=diagnosis,
        proposal_prompt=proposal,
        candidate_parse_result=candidate,
    )


def _prediction(task_id: str, label: str) -> BfclV4PublicPrediction:
    return BfclV4PublicPrediction(
        task_id=task_id,
        calls=(BfclV4OfficialPredictionCall(function_name=label, arguments_json="{}"),),
    )


def _selection_observation(
    *,
    plan,
    slot,
    resolution: BfclV4CandidateResolution,
    candidate_freeze_ref: ArtifactRef,
    accepted: bool,
    prediction_label: str,
    provider_succeeded: bool = True,
) -> BfclV4SelectionObservation:
    candidate_bound = slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT or (
        slot.kind is BfclV4PilotCallKind.GATE and slot.harness_variant == "candidate"
    )
    if slot.kind is BfclV4PilotCallKind.PARENT_FIT:
        executed_variant, fallback, freeze_ref = "parent", False, None
    elif candidate_bound:
        executed_variant = resolution.executed_harness_variant
        fallback = resolution.exact_parent_fallback_used
        freeze_ref = candidate_freeze_ref
    else:
        executed_variant, fallback, freeze_ref = "parent", False, candidate_freeze_ref
    materialization = BfclV4CallMaterialization(
        plan_fingerprint=plan.fingerprint,
        slot=slot,
        call_slot_reference_sha256=canonical_sha256(slot),
        request_ref=_opaque_ref(f"request:{slot.call_id}"),
        executed_harness_ref=_opaque_ref(f"harness:{slot.arm.value}:{executed_variant}"),
        intended_harness_variant=slot.harness_variant,
        executed_harness_variant=executed_variant,
        fallback_used=fallback,
        candidate_freeze_ref=freeze_ref,
    )
    materialization_ref = _exact_ref(materialization, BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE)
    prediction = (
        _prediction(slot.task_id, prediction_label)
        if provider_succeeded
        else BfclV4PublicPrediction(task_id=slot.task_id)
    )
    prediction_ref = _exact_ref(prediction, "application/json")
    grader_ref = _opaque_ref(f"grade:{slot.call_id}:{accepted}:{provider_succeeded}")
    completion = BfclV4CallCompletion(
        plan_fingerprint=plan.fingerprint,
        call_id=slot.call_id,
        global_slot=slot.global_slot,
        call_slot_reference_sha256=canonical_sha256(slot),
        materialization_ref=materialization_ref,
        attempt_outcome_ref=_opaque_ref(f"attempt:{slot.call_id}"),
        model_output_ref=prediction_ref,
        outcome=(
            BfclV4CallOutcome.SUCCEEDED
            if provider_succeeded
            else BfclV4CallOutcome.PROVIDER_FAILURE
        ),
        prediction_ref=prediction_ref,
        grader_receipt_ref=grader_ref,
    )
    completion_ref = _exact_ref(completion, BFCL_V4_CALL_COMPLETION_MEDIA_TYPE)
    return BfclV4SelectionObservation(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        arm=slot.arm,
        slot=slot,
        slot_reference_sha256=canonical_sha256(slot),
        materialization=materialization,
        completion=completion,
        completion_ref=completion_ref,
        provider_attempt_succeeded=provider_succeeded,
        prediction_imputed_empty_for_failed_attempt=not provider_succeeded,
        prediction=prediction,
        grader_receipt_reference_sha256=grader_ref.sha256,
        accepted=accepted if provider_succeeded else False,
    )


def _selection_observations(
    plan,
    resolutions: Mapping[BfclV4PilotArm, BfclV4CandidateResolution],
    *,
    parent_fit_correct: int = 2,
    candidate_fit_correct: int = 3,
    parent_gate_correct: int = 0,
    candidate_gate_correct: int = 1,
    failed_call_id: str | None = None,
    shadow_mismatch_arm: BfclV4PilotArm | None = None,
) -> tuple[BfclV4SelectionObservation, ...]:
    candidate_freeze = _opaque_ref("joint-candidate-freeze", BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE)
    observations = []
    for slot in plan.calls:
        if slot.kind not in {
            BfclV4PilotCallKind.PARENT_FIT,
            BfclV4PilotCallKind.CANDIDATE_FIT,
            BfclV4PilotCallKind.GATE,
        }:
            continue
        if slot.kind is BfclV4PilotCallKind.PARENT_FIT:
            index = sum(
                previous.arm is slot.arm
                and previous.kind is BfclV4PilotCallKind.PARENT_FIT
                and previous.global_slot < slot.global_slot
                for previous in plan.calls
            )
            accepted = index < parent_fit_correct
            label = f"parent_fit_{slot.task_id}"
        elif slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT:
            index = sum(
                previous.arm is slot.arm
                and previous.kind is BfclV4PilotCallKind.CANDIDATE_FIT
                and previous.global_slot < slot.global_slot
                for previous in plan.calls
            )
            accepted = index < candidate_fit_correct
            label = f"candidate_fit_{slot.task_id}"
        else:
            task_index = 0 if slot.task_id == "multiple_10" else 1
            accepted_count = (
                candidate_gate_correct
                if slot.harness_variant == "candidate"
                else parent_gate_correct
            )
            accepted = task_index < accepted_count
            label = (
                f"candidate_gate_{slot.task_id}"
                if slot.harness_variant == "candidate"
                else f"shadow_{slot.task_id}"
            )
            if (
                slot.arm is shadow_mismatch_arm
                and task_index == 0
                and slot.harness_variant == "placebo"
            ):
                label = f"mismatched_shadow_{slot.task_id}"
        observations.append(
            _selection_observation(
                plan=plan,
                slot=slot,
                resolution=resolutions[slot.arm],
                candidate_freeze_ref=candidate_freeze,
                accepted=accepted,
                prediction_label=label,
                provider_succeeded=slot.call_id != failed_call_id,
            )
        )
    return tuple(observations)


def _select(
    *,
    score_admissible: bool = True,
    full_admissible: bool = True,
    **observation_options,
):
    plan = build_bfcl_v4_public_pilot_call_plan()
    resolutions = {
        BfclV4PilotArm.SCORE: _resolution(BfclV4AdaptiveArm.SCORE, admissible=score_admissible),
        BfclV4PilotArm.FULL: _resolution(BfclV4AdaptiveArm.FULL, admissible=full_admissible),
    }
    observations = _selection_observations(plan, resolutions, **observation_options)
    evidence = build_bfcl_v4_joint_selection_evidence(
        plan=plan,
        score_resolution=resolutions[BfclV4PilotArm.SCORE],
        full_resolution=resolutions[BfclV4PilotArm.FULL],
        observations=observations,
    )
    return (
        plan,
        observations,
        evidence,
        select_bfcl_v4_public_candidates(plan=plan, evidence=evidence),
    )


def test_joint_selector_promotes_only_after_complete_sixteen_gate_evidence() -> None:
    _, _, evidence, selection = _select()

    assert len(evidence.global_gate_completion_refs) == 16
    assert evidence.all_sixteen_gate_completions_present is True
    for decision in (selection.score, selection.full):
        assert decision.decision is BfclV4SelectionDecision.PROMOTE
        assert decision.rollback_reasons == ()
        assert decision.parent_fit_correct == 2
        assert decision.candidate_fit_correct == 3
        assert decision.parent_gate_correct == 0
        assert decision.candidate_gate_correct == 1
        assert decision.shadow_controls_exact is True
        assert decision.selected_variant == "candidate"
        assert decision.forced_rollback is False
    assert selection.hidden_test_evidence is False
    assert selection.reportable_result is False


def test_selector_rejects_incomplete_or_reordered_global_evidence() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    resolutions = {
        BfclV4PilotArm.SCORE: _resolution(BfclV4AdaptiveArm.SCORE),
        BfclV4PilotArm.FULL: _resolution(BfclV4AdaptiveArm.FULL),
    }
    observations = _selection_observations(plan, resolutions)

    with pytest.raises(BfclV4PublicSelectionError, match="all 36"):
        build_bfcl_v4_joint_selection_evidence(
            plan=plan,
            score_resolution=resolutions[BfclV4PilotArm.SCORE],
            full_resolution=resolutions[BfclV4PilotArm.FULL],
            observations=observations[:-1],
        )
    reordered = (observations[1], observations[0], *observations[2:])
    with pytest.raises(BfclV4PublicSelectionError, match="global slot order"):
        build_bfcl_v4_joint_selection_evidence(
            plan=plan,
            score_resolution=resolutions[BfclV4PilotArm.SCORE],
            full_resolution=resolutions[BfclV4PilotArm.FULL],
            observations=reordered,
        )


@pytest.mark.parametrize(
    ("options", "reason"),
    (
        ({"candidate_fit_correct": 1}, BfclV4RollbackReason.CANDIDATE_FIT_REGRESSION),
        (
            {"parent_gate_correct": 1, "candidate_gate_correct": 1},
            BfclV4RollbackReason.CANDIDATE_GATE_NOT_STRICTLY_BETTER,
        ),
        (
            {"shadow_mismatch_arm": BfclV4PilotArm.SCORE},
            BfclV4RollbackReason.SHADOW_CONTROL_MISMATCH,
        ),
    ),
)
def test_each_frozen_rule_failure_rolls_back(options, reason) -> None:
    _, _, _, selection = _select(**options)
    decision = selection.score

    assert decision.decision is BfclV4SelectionDecision.ROLLBACK
    assert reason in decision.rollback_reasons
    assert decision.selected_variant == "parent"
    assert decision.forced_rollback is False


def test_provider_failure_rolls_back_even_when_counts_otherwise_promote() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    failed = next(
        slot.call_id
        for slot in plan.calls
        if slot.arm is BfclV4PilotArm.SCORE
        and slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT
        and slot.task_id == "parallel_multiple_9"
    )
    _, observations, _, selection = _select(failed_call_id=failed)
    failed_observation = next(item for item in observations if item.slot.call_id == failed)

    assert failed_observation.provider_attempt_succeeded is False
    assert failed_observation.prediction_imputed_empty_for_failed_attempt is True
    assert failed_observation.prediction.calls == ()
    assert failed_observation.completion.prediction_ref is not None
    assert failed_observation.completion.grader_receipt_ref is not None
    assert failed_observation.accepted is False
    assert selection.score.decision is BfclV4SelectionDecision.ROLLBACK
    assert BfclV4RollbackReason.PROVIDER_ATTEMPT_FAILURE in selection.score.rollback_reasons
    assert selection.full.decision is BfclV4SelectionDecision.PROMOTE


def test_invalid_candidate_consumes_slots_but_forces_exact_parent_rollback() -> None:
    _, observations, evidence, selection = _select(score_admissible=False)
    score_candidate_slots = tuple(
        item
        for item in observations
        if item.arm is BfclV4PilotArm.SCORE
        and (
            item.slot.kind is BfclV4PilotCallKind.CANDIDATE_FIT
            or (
                item.slot.kind is BfclV4PilotCallKind.GATE
                and item.slot.harness_variant == "candidate"
            )
        )
    )

    assert len(score_candidate_slots) == 7
    assert len(evidence.score.observations) == 18
    assert all(
        item.materialization.executed_harness_variant == "parent" for item in score_candidate_slots
    )
    assert all(item.materialization.fallback_used for item in score_candidate_slots)
    assert selection.score.decision is BfclV4SelectionDecision.ROLLBACK
    assert selection.score.forced_rollback is True
    assert selection.score.rollback_reasons[0] is BfclV4RollbackReason.CANDIDATE_INVALID


def test_selection_observation_rejects_a_forged_completion_reference() -> None:
    plan, observations, _, _ = _select()
    forged = observations[0].model_copy(
        update={
            "completion_ref": _opaque_ref("forged-completion", BFCL_V4_CALL_COMPLETION_MEDIA_TYPE)
        }
    )
    with pytest.raises(ValidationError, match="completion reference"):
        BfclV4SelectionObservation.model_validate(forged, strict=True)
    assert plan.total_model_call_ceiling == 100


def _holdout_observations(plan, selection, correctness, *, failed_key=None):
    observations = []
    selection_sha = selection.fingerprint
    adaptive_sha = {
        BfclV4PilotArm.SCORE: selection.score.fingerprint,
        BfclV4PilotArm.FULL: selection.full.fingerprint,
    }
    for arm in BFCL_V4_METRIC_ARM_ORDER:
        for task_index, task_id in enumerate(BFCL_V4_HOLDOUT_TASK_IDS):
            slots = tuple(
                slot for slot in plan.calls if slot.arm is arm and slot.task_id == task_id
            )
            failed = failed_key == (arm, task_id)
            refs = tuple(
                _opaque_ref(
                    f"holdout-completion:{slot.call_id}", BFCL_V4_CALL_COMPLETION_MEDIA_TYPE
                )
                for slot in slots
            )
            selected_prediction = (
                BfclV4PublicPrediction(task_id=task_id)
                if failed
                else _prediction(task_id, f"holdout_{arm.value.replace('-', '_')}_{task_id}")
            )
            observations.append(
                BfclV4HoldoutObservation(
                    plan_fingerprint=plan.fingerprint,
                    schedule_content_sha256=plan.schedule_content_sha256,
                    joint_selection_decision_fingerprint=selection_sha,
                    arm_selection_decision_fingerprint=adaptive_sha.get(arm),
                    task_id=task_id,
                    arm=arm,
                    source_call_ids=tuple(slot.call_id for slot in slots),
                    source_completion_refs=refs,
                    source_provider_attempt_succeeded=(not failed,) * len(slots),
                    pure_at_b_selection_fingerprint=(
                        _sha(f"plurality:{task_id}") if arm is BfclV4PilotArm.PURE_AT_B else None
                    ),
                    prediction_imputed_empty_for_failed_attempt=failed,
                    selected_prediction=selected_prediction,
                    selected_prediction_ref=_exact_ref(selected_prediction, "application/json"),
                    grader_receipt_ref=_opaque_ref(f"holdout-grade:{arm.value}:{task_id}"),
                    holdout_unlock_fingerprint=_sha("shared-holdout-unlock"),
                    accepted=False if failed else correctness[arm][task_index],
                )
            )
    return tuple(observations)


def test_holdout_metrics_are_exact_integer_bp_and_descriptive_only() -> None:
    plan, _, _, selection = _select()
    correctness = {
        BfclV4PilotArm.PURE: (True, True, False, False, False, False, False, False),
        BfclV4PilotArm.STATIC: (True, True, True, False, False, False, False, False),
        BfclV4PilotArm.SCORE: (True, True, True, True, False, False, False, False),
        BfclV4PilotArm.FULL: (True, True, True, True, True, False, False, False),
        BfclV4PilotArm.PURE_AT_B: (True, True, True, True, False, False, False, False),
    }
    observations = _holdout_observations(plan, selection, correctness)
    evidence = build_bfcl_v4_holdout_evidence(
        plan=plan, joint_selection=selection, observations=observations
    )
    metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=plan, joint_selection=selection, evidence=evidence
    )

    assert [item.correct_count for item in metrics.arms] == [2, 3, 4, 5, 4]
    assert [item.accuracy_basis_points for item in metrics.arms] == [
        2_500,
        3_750,
        5_000,
        6_250,
        5_000,
    ]
    full_pure = next(
        item
        for item in metrics.paired_deltas
        if item.treatment_arm is BfclV4PilotArm.FULL and item.reference_arm is BfclV4PilotArm.PURE
    )
    assert (full_pure.wins, full_pure.ties, full_pure.losses) == (3, 5, 0)
    assert full_pure.delta_basis_points == 3_750
    assert full_pure.strictly_exceeds_positive_ten_percentage_points is True
    assert all(
        item.strictly_exceeds_positive_ten_percentage_points for item in metrics.paired_deltas
    )
    assert metrics.public_development_descriptive_only is True
    assert metrics.statistical_significance_claimed is False
    assert metrics.hidden_test_evidence is False
    assert metrics.reportable_result is False

    def all_values(value):
        if isinstance(value, dict):
            return (value, *(child for item in value.values() for child in all_values(item)))
        if isinstance(value, (list, tuple)):
            return (value, *(child for item in value for child in all_values(item)))
        return (value,)

    assert not any(isinstance(value, float) for value in all_values(metrics.model_dump()))


def test_holdout_builder_rejects_missing_or_foreign_source_calls() -> None:
    plan, _, _, selection = _select()
    correctness = {arm: (False,) * 8 for arm in BFCL_V4_METRIC_ARM_ORDER}
    observations = _holdout_observations(plan, selection, correctness)

    with pytest.raises(BfclV4PublicSelectionError, match="arm-major matrix"):
        build_bfcl_v4_holdout_evidence(
            plan=plan,
            joint_selection=selection,
            observations=observations[:-1],
        )
    forged = observations[0].model_copy(update={"source_call_ids": ("foreign/call",)})
    with pytest.raises(BfclV4PublicSelectionError, match="frozen lineage"):
        build_bfcl_v4_holdout_evidence(
            plan=plan,
            joint_selection=selection,
            observations=(forged, *observations[1:]),
        )


def test_failed_holdout_attempt_is_explicitly_empty_graded_and_incorrect() -> None:
    plan, _, _, selection = _select()
    correctness = {arm: (True,) * 8 for arm in BFCL_V4_METRIC_ARM_ORDER}
    failed_key = (BfclV4PilotArm.FULL, BFCL_V4_HOLDOUT_TASK_IDS[0])
    observations = _holdout_observations(plan, selection, correctness, failed_key=failed_key)
    failed = next(item for item in observations if (item.arm, item.task_id) == failed_key)
    evidence = build_bfcl_v4_holdout_evidence(
        plan=plan, joint_selection=selection, observations=observations
    )
    metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=plan, joint_selection=selection, evidence=evidence
    )

    assert failed.source_provider_attempt_succeeded == (False,)
    assert failed.prediction_imputed_empty_for_failed_attempt is True
    assert failed.selected_prediction.calls == ()
    assert failed.grader_receipt_ref is not None
    assert failed.accepted is False
    full = next(item for item in metrics.arms if item.arm is BfclV4PilotArm.FULL)
    assert full.correct_count == 7
    assert full.accuracy_basis_points == 8_750


def test_all_failed_pure_at_b_is_explicitly_empty_graded_and_incorrect() -> None:
    plan, _, _, selection = _select()
    correctness = {arm: (True,) * 8 for arm in BFCL_V4_METRIC_ARM_ORDER}
    failed_key = (BfclV4PilotArm.PURE_AT_B, BFCL_V4_HOLDOUT_TASK_IDS[0])
    observations = _holdout_observations(plan, selection, correctness, failed_key=failed_key)
    failed = next(item for item in observations if (item.arm, item.task_id) == failed_key)
    evidence = build_bfcl_v4_holdout_evidence(
        plan=plan, joint_selection=selection, observations=observations
    )

    assert failed.source_provider_attempt_succeeded == (False, False, False, False)
    assert failed.prediction_imputed_empty_for_failed_attempt is True
    assert failed.selected_prediction.calls == ()
    assert failed.pure_at_b_selection_fingerprint is not None
    assert failed.accepted is False
    assert evidence.all_sixty_holdout_call_completions_present is True

    forged = failed.model_copy(update={"prediction_imputed_empty_for_failed_attempt": False})
    with pytest.raises(ValidationError, match="all-failed PURE@B"):
        BfclV4HoldoutObservation.model_validate(forged, strict=True)


def test_descriptive_delta_contract_rejects_float_basis_points() -> None:
    with pytest.raises(ValidationError):
        BfclV4PairedDescriptiveDelta(
            treatment_arm=BfclV4PilotArm.FULL,
            reference_arm=BfclV4PilotArm.PURE,
            task_ids=BFCL_V4_HOLDOUT_TASK_IDS,
            treatment_correctness=(True,) + (False,) * 7,
            reference_correctness=(False,) * 8,
            wins=1,
            ties=7,
            losses=0,
            delta_basis_points=1_250.0,
            strictly_exceeds_positive_ten_percentage_points=True,
        )
