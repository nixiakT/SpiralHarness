"""Internal semantic-transition replay for the BFCL V4 public pilot."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4GradingSlotBinding,
    BfclV4HoldoutUnlock,
    BfclV4PublicGraderReceipt,
    BfclV4PublicPrediction,
    checked,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import BfclV4PureAtBSample
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotCallSlot,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
    BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
    BFCL_V4_RUN_EVENT_MEDIA_TYPE,
    BFCL_V4_RUN_STATE_MEDIA_TYPE,
    BfclV4ArmCandidateFreeze,
    BfclV4ArmSelection,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4JointCandidateFreeze,
    BfclV4JointSelectionFreeze,
    BfclV4RunAction,
    BfclV4RunEvent,
    BfclV4RunJournalEntry,
    BfclV4RunState,
    BfclV4SelectedVariant,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository


class BfclV4RunError(RuntimeError):
    """A requested run transition violates the frozen protocol."""


class BfclV4RunIntegrityError(BfclV4RunError):
    """A content-addressed run artifact cannot be replayed exactly."""


class BfclV4RunCycleError(BfclV4RunIntegrityError):
    """The linked journal revisits an entry digest."""


class BfclV4StaleTailError(BfclV4RunError):
    """The caller attempted a compare-and-set from a stale or foreign tail."""


def _checked_ref(ref: ArtifactRef, media_type: str | None, label: str) -> ArtifactRef:
    try:
        value = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise BfclV4RunIntegrityError(f"{label} is not an exact artifact reference") from exc
    if media_type is not None and value.media_type != media_type:
        raise BfclV4RunIntegrityError(f"{label} declares the wrong media type")
    return value


def _load[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model: type[ModelT],
    label: str,
    *,
    media_type: str | None = None,
) -> ModelT:
    checked_ref = _checked_ref(ref, media_type, label)
    try:
        loaded = repository.get_json(checked_ref, model)
        return model.model_validate(loaded, strict=True)
    except Exception as exc:
        raise BfclV4RunIntegrityError(f"{label} cannot be verified") from exc


def _require_artifact(repository: ArtifactRepository, ref: ArtifactRef, label: str) -> None:
    checked_ref = _checked_ref(ref, None, label)
    try:
        repository.get_bytes(checked_ref)
    except Exception as exc:
        raise BfclV4RunIntegrityError(f"{label} cannot be verified") from exc


def _publish[ModelT: BaseModel](
    repository: ArtifactRepository,
    value: ModelT,
    model: type[ModelT],
    media_type: str,
    label: str,
) -> ArtifactRef:
    checked_value = model.model_validate(value, strict=True)
    try:
        raw_ref = repository.put_json(checked_value, media_type=media_type)
        ref = _checked_ref(raw_ref, media_type, f"published {label}")
        loaded = _load(repository, ref, model, f"published {label}", media_type=media_type)
    except BfclV4RunIntegrityError:
        raise
    except Exception as exc:
        raise BfclV4RunIntegrityError(f"{label} publication failed") from exc
    if loaded != checked_value:
        raise BfclV4RunIntegrityError(f"published {label} changed content")
    return ref


def _plan(value: BfclV4PublicPilotCallPlan | None) -> BfclV4PublicPilotCallPlan:
    chosen = build_bfcl_v4_public_pilot_call_plan() if value is None else value
    try:
        checked_plan = BfclV4PublicPilotCallPlan.model_validate(chosen, strict=True)
    except Exception as exc:
        raise BfclV4RunIntegrityError("BFCL call plan is not the frozen typed plan") from exc
    expected = build_bfcl_v4_public_pilot_call_plan()
    if checked_plan != expected:
        raise BfclV4RunIntegrityError("BFCL call plan differs from the repository-frozen plan")
    return checked_plan


def _completion_materialization(
    repository: ArtifactRepository,
    completion_ref: ArtifactRef,
) -> tuple[BfclV4CallCompletion, BfclV4CallMaterialization]:
    completion = _load(
        repository,
        completion_ref,
        BfclV4CallCompletion,
        "call completion",
        media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    )
    materialization = _load(
        repository,
        completion.materialization_ref,
        BfclV4CallMaterialization,
        "call materialization",
        media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    )
    return completion, materialization


def _arm_candidate(
    freeze: BfclV4JointCandidateFreeze,
    arm: BfclV4PilotArm,
) -> BfclV4ArmCandidateFreeze:
    if arm == BfclV4PilotArm.SCORE:
        return freeze.score
    if arm == BfclV4PilotArm.FULL:
        return freeze.full
    raise BfclV4RunIntegrityError("nonadaptive arm has no candidate artifact")


def _expected_grade_role(slot: BfclV4PilotCallSlot) -> str | None:
    if slot.arm == BfclV4PilotArm.PURE_AT_B:
        return None
    if slot.arm in {BfclV4PilotArm.PURE, BfclV4PilotArm.STATIC}:
        return "baseline"
    if slot.kind == BfclV4PilotCallKind.PARENT_FIT:
        return "parent-fit"
    if slot.kind == BfclV4PilotCallKind.CANDIDATE_FIT:
        return "candidate-fit"
    if slot.kind == BfclV4PilotCallKind.HOLDOUT:
        return "holdout"
    if slot.kind == BfclV4PilotCallKind.GATE:
        return f"gate-{slot.harness_variant}"
    return None


def _load_candidate_freeze(
    repository: ArtifactRepository,
    state: BfclV4RunState,
) -> BfclV4JointCandidateFreeze:
    if state.candidate_freeze_ref is None:
        raise BfclV4RunIntegrityError("joint candidate freeze is absent")
    return _load(
        repository,
        state.candidate_freeze_ref,
        BfclV4JointCandidateFreeze,
        "joint candidate freeze",
        media_type=BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    )


def _load_joint_selection(
    repository: ArtifactRepository,
    state: BfclV4RunState,
) -> tuple[BfclV4JointSelectionFreeze, BfclV4ArmSelection, BfclV4ArmSelection]:
    if state.joint_selection_freeze_ref is None:
        raise BfclV4RunIntegrityError("joint selection freeze is absent")
    joint = _load(
        repository,
        state.joint_selection_freeze_ref,
        BfclV4JointSelectionFreeze,
        "joint selection freeze",
        media_type=BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
    )
    score = _load(
        repository,
        joint.score_selection_ref,
        BfclV4ArmSelection,
        "SCORE selection",
        media_type=BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    )
    full = _load(
        repository,
        joint.full_selection_ref,
        BfclV4ArmSelection,
        "FULL selection",
        media_type=BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    )
    return joint, score, full


def _expected_execution(
    repository: ArtifactRepository,
    state: BfclV4RunState,
    slot: BfclV4PilotCallSlot,
) -> tuple[ArtifactRef | None, str, bool]:
    """Return an enforced harness ref (if adaptive), variant, and fallback flag."""

    if slot.arm == BfclV4PilotArm.PURE:
        return None, "bare", False
    if slot.arm == BfclV4PilotArm.STATIC:
        return None, "static-frozen", False
    if slot.arm == BfclV4PilotArm.PURE_AT_B:
        return None, "bare", False
    if slot.kind == BfclV4PilotCallKind.PARENT_FIT:
        return None, "parent", False
    if slot.kind in {BfclV4PilotCallKind.DIAGNOSIS, BfclV4PilotCallKind.PROPOSAL}:
        return None, slot.harness_variant, False

    freeze = _load_candidate_freeze(repository, state)
    candidate = _arm_candidate(freeze, slot.arm)
    if slot.kind == BfclV4PilotCallKind.CANDIDATE_FIT:
        variant = "candidate" if candidate.candidate_valid else "parent"
        return candidate.effective_candidate_harness_ref, variant, not candidate.candidate_valid
    if slot.kind == BfclV4PilotCallKind.GATE:
        if slot.harness_variant == "candidate":
            variant = "candidate" if candidate.candidate_valid else "parent"
            return candidate.effective_candidate_harness_ref, variant, not candidate.candidate_valid
        # Revert and placebo are exact-parent shadow controls in this pilot.
        return candidate.parent_harness_ref, "parent", False
    if slot.kind == BfclV4PilotCallKind.HOLDOUT:
        _, score, full = _load_joint_selection(repository, state)
        selection = score if slot.arm == BfclV4PilotArm.SCORE else full
        return (
            selection.selected_harness_ref,
            selection.selected_variant.value,
            selection.selected_variant == BfclV4SelectedVariant.PARENT,
        )
    raise BfclV4RunIntegrityError("adaptive call kind has no execution rule")


def _verify_materialization(
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    state: BfclV4RunState,
    materialization: BfclV4CallMaterialization,
) -> None:
    if state.closed or state.open_materialization_ref is not None:
        raise BfclV4RunError("cannot materialize while the run is closed or a call is open")
    if state.next_global_slot >= 100:
        raise BfclV4RunError("the frozen 100-call schedule is already complete")
    expected = plan.calls[state.next_global_slot]
    if materialization.plan_fingerprint != plan.fingerprint or materialization.slot != expected:
        raise BfclV4RunIntegrityError("materialization differs from the next frozen call slot")
    if materialization.call_slot_reference_sha256 != canonical_sha256(expected):
        raise BfclV4RunIntegrityError("materialization slot reference changed")
    expected_candidate_ref = state.candidate_freeze_ref if expected.global_slot >= 14 else None
    expected_selection_ref = (
        state.joint_selection_freeze_ref if expected.global_slot >= 40 else None
    )
    if materialization.candidate_freeze_ref != expected_candidate_ref:
        raise BfclV4RunIntegrityError("materialization candidate barrier binding differs")
    if materialization.joint_selection_freeze_ref != expected_selection_ref:
        raise BfclV4RunIntegrityError("materialization selection barrier binding differs")
    if expected.requires_both_candidate_artifacts and expected_candidate_ref is None:
        raise BfclV4RunError("candidate FIT cannot materialize before both candidates freeze")
    if expected.requires_both_selection_artifacts and expected_selection_ref is None:
        raise BfclV4RunError("HOLDOUT cannot materialize before both selections freeze")
    expected_harness, variant, fallback = _expected_execution(repository, state, expected)
    if expected_harness is not None and materialization.executed_harness_ref != expected_harness:
        raise BfclV4RunIntegrityError("adaptive slot materialized the wrong harness")
    if (
        materialization.executed_harness_variant != variant
        or materialization.fallback_used != fallback
    ):
        raise BfclV4RunIntegrityError("executed variant or fallback flag differs from policy")
    _require_artifact(repository, materialization.request_ref, "model request")
    _require_artifact(repository, materialization.executed_harness_ref, "executed harness")


def _verify_grader_binding(
    repository: ArtifactRepository,
    materialization: BfclV4CallMaterialization,
    completion: BfclV4CallCompletion,
) -> None:
    slot = materialization.slot
    role = _expected_grade_role(slot)
    if role is None:
        if completion.prediction_ref is not None or completion.grader_receipt_ref is not None:
            raise BfclV4RunIntegrityError("controller/PURE@B call must not carry a grader receipt")
        if slot.arm == BfclV4PilotArm.PURE_AT_B:
            sample = _load(
                repository,
                completion.model_output_ref,
                BfclV4PureAtBSample,
                "PURE@B sample",
            )
            if sample.sample_id != slot.call_id:
                raise BfclV4RunIntegrityError("PURE@B sample ID differs from its frozen call")
        return
    if completion.prediction_ref is None or completion.grader_receipt_ref is None:
        raise BfclV4RunIntegrityError("gradable call lacks prediction or trusted grader receipt")
    prediction = _load(
        repository,
        completion.prediction_ref,
        BfclV4PublicPrediction,
        "public prediction",
    )
    receipt = _load(
        repository,
        completion.grader_receipt_ref,
        BfclV4PublicGraderReceipt,
        "public grader receipt",
    )
    try:
        receipt = checked(receipt, BfclV4PublicGraderReceipt)
    except Exception as exc:
        raise BfclV4RunIntegrityError("public grader receipt failed strict revalidation") from exc
    binding: BfclV4GradingSlotBinding = receipt.slot
    expected_coordinates = (
        materialization.plan_fingerprint,
        materialization.call_slot_reference_sha256,
        slot.call_id,
        slot.arm.value,
        role,
        materialization.intended_harness_variant,
        materialization.executed_harness_variant,
        materialization.fallback_used,
        slot.task_id,
    )
    actual_coordinates = (
        binding.plan_fingerprint,
        binding.call_slot_reference_sha256,
        binding.call_id,
        binding.arm,
        binding.grade_role,
        binding.intended_harness_variant,
        binding.executed_harness_variant,
        binding.fallback_used,
        binding.task_id,
    )
    if actual_coordinates != expected_coordinates:
        raise BfclV4RunIntegrityError("grader receipt differs from its materialized slot")
    if receipt.prediction != prediction:
        raise BfclV4RunIntegrityError("grader receipt embeds a different prediction")
    if slot.global_slot >= 40:
        if receipt.holdout_unlock is None:
            raise BfclV4RunIntegrityError("HOLDOUT receipt lacks the joint-selection unlock")
        joint = _load(
            repository,
            materialization.joint_selection_freeze_ref,  # type: ignore[arg-type]
            BfclV4JointSelectionFreeze,
            "joint selection freeze",
            media_type=BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
        )
        expected_unlock = BfclV4HoldoutUnlock(
            plan_fingerprint=materialization.plan_fingerprint,
            score_selection_artifact_sha256=joint.score_selection_ref.sha256,
            full_selection_artifact_sha256=joint.full_selection_ref.sha256,
        )
        if receipt.holdout_unlock != expected_unlock:
            raise BfclV4RunIntegrityError("HOLDOUT receipt uses a foreign selection unlock")


def _verify_completion(
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    state: BfclV4RunState,
    completion: BfclV4CallCompletion,
) -> None:
    if state.open_materialization_ref is None:
        raise BfclV4RunError("cannot complete a call without an open materialization")
    if completion.materialization_ref != state.open_materialization_ref:
        raise BfclV4RunIntegrityError("completion does not close the current materialization")
    materialization = _load(
        repository,
        completion.materialization_ref,
        BfclV4CallMaterialization,
        "call materialization",
        media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    )
    slot = plan.calls[state.next_global_slot]
    expected = (
        plan.fingerprint,
        slot.call_id,
        slot.global_slot,
        materialization.call_slot_reference_sha256,
    )
    actual = (
        completion.plan_fingerprint,
        completion.call_id,
        completion.global_slot,
        completion.call_slot_reference_sha256,
    )
    if actual != expected or materialization.slot != slot:
        raise BfclV4RunIntegrityError("completion coordinates differ from the open frozen slot")
    _require_artifact(repository, completion.attempt_outcome_ref, "attempt outcome")
    _require_artifact(repository, completion.model_output_ref, "model output")
    _verify_grader_binding(repository, materialization, completion)


def _verify_candidate_freeze(
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    state: BfclV4RunState,
    freeze: BfclV4JointCandidateFreeze,
) -> None:
    if state.next_global_slot != 14 or state.open_materialization_ref is not None:
        raise BfclV4RunError("both proposals must finish before candidates freeze")
    if state.candidate_freeze_ref is not None:
        raise BfclV4RunError("candidate artifacts are already frozen")
    if freeze.plan_fingerprint != plan.fingerprint:
        raise BfclV4RunIntegrityError("candidate freeze belongs to another plan")
    expected_proposals = (state.call_completion_refs[12], state.call_completion_refs[13])
    if (freeze.score.proposal_completion_ref, freeze.full.proposal_completion_ref) != (
        expected_proposals
    ):
        raise BfclV4RunIntegrityError("candidate freeze does not bind both proposal completions")
    for arm_freeze, indexes in ((freeze.score, range(0, 5)), (freeze.full, range(5, 10))):
        parent_refs = []
        for index in indexes:
            _, materialization = _completion_materialization(
                repository, state.call_completion_refs[index]
            )
            parent_refs.append(materialization.executed_harness_ref)
        if set(parent_refs) != {arm_freeze.parent_harness_ref}:
            raise BfclV4RunIntegrityError("candidate freeze parent differs from parent FIT calls")
        _require_artifact(repository, arm_freeze.candidate_parse_ref, "candidate parse artifact")
        _require_artifact(repository, arm_freeze.parent_harness_ref, "parent harness")
        if arm_freeze.candidate_harness_ref is not None:
            _require_artifact(repository, arm_freeze.candidate_harness_ref, "candidate harness")


def _verify_arm_selection(
    selection: BfclV4ArmSelection,
    arm_freeze: BfclV4ArmCandidateFreeze,
    expected_gate_refs: tuple[ArtifactRef, ...],
    candidate_freeze_ref: ArtifactRef,
    plan_fingerprint: str,
) -> None:
    if (
        selection.plan_fingerprint != plan_fingerprint
        or selection.arm != arm_freeze.arm
        or selection.candidate_freeze_ref != candidate_freeze_ref
        or selection.gate_completion_refs != expected_gate_refs
    ):
        raise BfclV4RunIntegrityError("arm selection differs from its candidate/GATE lineage")
    if not arm_freeze.candidate_valid:
        expected = (BfclV4SelectedVariant.PARENT, arm_freeze.parent_harness_ref, True)
    elif selection.selected_variant == BfclV4SelectedVariant.CANDIDATE:
        expected = (
            BfclV4SelectedVariant.CANDIDATE,
            arm_freeze.effective_candidate_harness_ref,
            False,
        )
    else:
        expected = (BfclV4SelectedVariant.PARENT, arm_freeze.parent_harness_ref, False)
    if (selection.selected_variant, selection.selected_harness_ref, selection.forced_rollback) != (
        expected
    ):
        raise BfclV4RunIntegrityError("selection violates candidate validity/rollback policy")


def _verify_joint_selection(
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    state: BfclV4RunState,
    joint: BfclV4JointSelectionFreeze,
) -> None:
    if state.next_global_slot != 40 or state.open_materialization_ref is not None:
        raise BfclV4RunError("all sixteen GATE calls must finish before joint selection")
    if state.joint_selection_freeze_ref is not None or state.candidate_freeze_ref is None:
        raise BfclV4RunError("joint selection is already frozen or candidates are absent")
    expected_gates = state.call_completion_refs[24:40]
    if (
        joint.plan_fingerprint != plan.fingerprint
        or joint.candidate_freeze_ref != state.candidate_freeze_ref
        or joint.gate_completion_refs != expected_gates
    ):
        raise BfclV4RunIntegrityError("joint selection does not bind the exact sixteen GATE calls")
    freeze = _load_candidate_freeze(repository, state)
    score = _load(
        repository,
        joint.score_selection_ref,
        BfclV4ArmSelection,
        "SCORE selection",
        media_type=BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    )
    full = _load(
        repository,
        joint.full_selection_ref,
        BfclV4ArmSelection,
        "FULL selection",
        media_type=BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    )
    _verify_arm_selection(
        score, freeze.score, expected_gates[:8], state.candidate_freeze_ref, plan.fingerprint
    )
    _verify_arm_selection(
        full, freeze.full, expected_gates[8:], state.candidate_freeze_ref, plan.fingerprint
    )
    _require_artifact(repository, score.decision_ref, "SCORE selection decision")
    _require_artifact(repository, full.decision_ref, "FULL selection decision")


def _apply_event(
    repository: ArtifactRepository,
    plan: BfclV4PublicPilotCallPlan,
    previous: BfclV4RunState | None,
    event: BfclV4RunEvent,
) -> BfclV4RunState:
    if event.plan_fingerprint != plan.fingerprint:
        raise BfclV4RunIntegrityError("journal event belongs to another call plan")
    if previous is None:
        if event.action != BfclV4RunAction.OPEN:
            raise BfclV4RunError("journal root must be OPEN")
        return BfclV4RunState(plan_fingerprint=plan.fingerprint)
    if previous.plan_fingerprint != plan.fingerprint:
        raise BfclV4RunIntegrityError("journal state belongs to another call plan")
    if previous.closed:
        raise BfclV4RunError("no transition may follow a closed run")
    content = previous.model_dump(mode="python")
    if event.action == BfclV4RunAction.MATERIALIZE_CALL:
        materialization = _load(
            repository,
            event.materialization_ref,  # type: ignore[arg-type]
            BfclV4CallMaterialization,
            "call materialization",
            media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
        )
        _verify_materialization(repository, plan, previous, materialization)
        content["open_materialization_ref"] = event.materialization_ref
    elif event.action == BfclV4RunAction.COMPLETE_CALL:
        completion = _load(
            repository,
            event.completion_ref,  # type: ignore[arg-type]
            BfclV4CallCompletion,
            "call completion",
            media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
        )
        _verify_completion(repository, plan, previous, completion)
        content["open_materialization_ref"] = None
        content["call_completion_refs"] = (
            *previous.call_completion_refs,
            event.completion_ref,
        )
        content["next_global_slot"] = previous.next_global_slot + 1
    elif event.action == BfclV4RunAction.FREEZE_CANDIDATES:
        freeze = _load(
            repository,
            event.candidate_freeze_ref,  # type: ignore[arg-type]
            BfclV4JointCandidateFreeze,
            "joint candidate freeze",
            media_type=BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
        )
        _verify_candidate_freeze(repository, plan, previous, freeze)
        content["candidate_freeze_ref"] = event.candidate_freeze_ref
    elif event.action == BfclV4RunAction.FREEZE_SELECTIONS:
        joint = _load(
            repository,
            event.joint_selection_freeze_ref,  # type: ignore[arg-type]
            BfclV4JointSelectionFreeze,
            "joint selection freeze",
            media_type=BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
        )
        _verify_joint_selection(repository, plan, previous, joint)
        content["joint_selection_freeze_ref"] = event.joint_selection_freeze_ref
    elif event.action == BfclV4RunAction.CLOSE:
        if (
            previous.next_global_slot != 100
            or previous.open_materialization_ref is not None
            or previous.joint_selection_freeze_ref is None
        ):
            raise BfclV4RunError("run closure requires all 100 completed calls")
        content["closed"] = True
    else:
        raise BfclV4RunError("OPEN may appear only at the journal root")
    return BfclV4RunState.model_validate(content, strict=True)


def _read_entries(
    repository: ArtifactRepository,
    tail_ref: ArtifactRef,
) -> tuple[tuple[ArtifactRef, BfclV4RunJournalEntry], ...]:
    cursor: ArtifactRef | None = _checked_ref(tail_ref, BFCL_V4_RUN_ENTRY_MEDIA_TYPE, "run tail")
    backwards: list[tuple[ArtifactRef, BfclV4RunJournalEntry]] = []
    seen: set[str] = set()
    while cursor is not None:
        if cursor.sha256 in seen:
            raise BfclV4RunCycleError(f"run journal cycle detected at {cursor.sha256}")
        seen.add(cursor.sha256)
        entry = _load(
            repository,
            cursor,
            BfclV4RunJournalEntry,
            "run journal entry",
            media_type=BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
        )
        backwards.append((cursor, entry))
        cursor = entry.previous_entry_ref
    entries = tuple(reversed(backwards))
    for index, (ref, entry) in enumerate(entries):
        if entry.sequence != index:
            raise BfclV4RunIntegrityError("run journal sequence is not contiguous from zero")
        expected_previous = None if index == 0 else entries[index - 1][0]
        expected_state = None if index == 0 else entries[index - 1][1].state_ref
        if (
            entry.previous_entry_ref != expected_previous
            or entry.previous_state_ref != expected_state
        ):
            raise BfclV4RunIntegrityError("run journal link or previous-state join changed")
        if ref.media_type != BFCL_V4_RUN_ENTRY_MEDIA_TYPE:  # pragma: no cover - checked above
            raise BfclV4RunIntegrityError("run journal entry reference media type changed")
    return entries


def replay_bfcl_v4_public_run(
    repository: ArtifactRepository,
    tail_ref: ArtifactRef,
    *,
    plan: BfclV4PublicPilotCallPlan | None = None,
) -> tuple[tuple[BfclV4RunEvent, ...], BfclV4RunState]:
    """Offline-replay a tail, recomputing every semantic state transition."""

    checked_plan = _plan(plan)
    entries = _read_entries(repository, tail_ref)
    state: BfclV4RunState | None = None
    events: list[BfclV4RunEvent] = []
    for _, entry in entries:
        if entry.plan_fingerprint != checked_plan.fingerprint:
            raise BfclV4RunIntegrityError("journal entry belongs to another call plan")
        event = _load(
            repository,
            entry.event_ref,
            BfclV4RunEvent,
            "run event",
            media_type=BFCL_V4_RUN_EVENT_MEDIA_TYPE,
        )
        recomputed = _apply_event(repository, checked_plan, state, event)
        persisted = _load(
            repository,
            entry.state_ref,
            BfclV4RunState,
            "run state",
            media_type=BFCL_V4_RUN_STATE_MEDIA_TYPE,
        )
        if persisted != recomputed:
            raise BfclV4RunIntegrityError("persisted run state differs from semantic replay")
        state = persisted
        events.append(event)
    if state is None:  # pragma: no cover - a tail necessarily loads one entry
        raise BfclV4RunIntegrityError("run journal is empty")
    return tuple(events), state
