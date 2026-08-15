from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark._bfcl_v4_public_run_replay as replay_subject
from spiral_harness.benchmark.bfcl_v4_public_grader import (
    BfclV4PublicGraderError,
    _checkout_and_git,
    grade_bfcl_v4_public_prediction,
    make_bfcl_v4_public_prediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4GradingSlotBinding,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
    BFCL_V4_RUN_STATE_MEDIA_TYPE,
    BfclV4ArmCandidateFreeze,
    BfclV4ArmSelection,
    BfclV4CallOutcome,
    BfclV4RunJournalEntry,
    BfclV4RunState,
    BfclV4SelectedVariant,
)
from spiral_harness.benchmark.bfcl_v4_public_run_journal import (
    BfclV4PublicRunJournal,
    BfclV4RunError,
    BfclV4RunIntegrityError,
    BfclV4StaleTailError,
    replay_bfcl_v4_public_run,
    verify_bfcl_v4_public_run_closure,
)
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.artifact_store import ArtifactStore


def _artifact(store: ArtifactStore, label: str) -> ArtifactRef:
    return store.put_json({"artifact": label})


def _variant_and_harness(
    journal: BfclV4PublicRunJournal,
    *,
    parent: ArtifactRef,
    candidate: ArtifactRef,
    bare: ArtifactRef,
    static: ArtifactRef,
    controller: ArtifactRef,
) -> tuple[ArtifactRef, str, bool]:
    assert journal.state is not None
    slot = journal.plan.calls[journal.state.next_global_slot]
    if slot.arm == BfclV4PilotArm.PURE:
        return bare, "bare", False
    if slot.arm == BfclV4PilotArm.STATIC:
        return static, "static-frozen", False
    if slot.arm == BfclV4PilotArm.PURE_AT_B:
        return bare, "bare", False
    if slot.kind == BfclV4PilotCallKind.PARENT_FIT:
        return parent, "parent", False
    if slot.kind in {BfclV4PilotCallKind.DIAGNOSIS, BfclV4PilotCallKind.PROPOSAL}:
        return controller, slot.harness_variant, False
    if slot.kind == BfclV4PilotCallKind.CANDIDATE_FIT:
        if slot.arm == BfclV4PilotArm.SCORE:
            return candidate, "candidate", False
        return parent, "parent", True
    if slot.kind == BfclV4PilotCallKind.GATE:
        if slot.harness_variant == "candidate" and slot.arm == BfclV4PilotArm.SCORE:
            return candidate, "candidate", False
        if slot.harness_variant == "candidate":
            return parent, "parent", True
        return parent, "parent", False
    if slot.kind == BfclV4PilotCallKind.HOLDOUT:
        if slot.arm == BfclV4PilotArm.SCORE:
            return candidate, "candidate", False
        return parent, "parent", True
    raise AssertionError(slot)


def _complete_next(
    journal: BfclV4PublicRunJournal,
    store: ArtifactStore,
    tail: ArtifactRef,
    *,
    parent: ArtifactRef,
    candidate: ArtifactRef,
    bare: ArtifactRef,
    static: ArtifactRef,
    controller: ArtifactRef,
) -> tuple[ArtifactRef, ArtifactRef]:
    assert journal.state is not None
    slot = journal.plan.calls[journal.state.next_global_slot]
    harness, variant, fallback = _variant_and_harness(
        journal,
        parent=parent,
        candidate=candidate,
        bare=bare,
        static=static,
        controller=controller,
    )
    materialized_tail, materialization_ref = journal.materialize_next_call(
        expected_tail_ref=tail,
        request_ref=_artifact(store, f"request/{slot.global_slot}"),
        executed_harness_ref=harness,
        executed_harness_variant=variant,
        fallback_used=fallback,
    )
    return journal.complete_call(
        expected_tail_ref=materialized_tail,
        materialization_ref=materialization_ref,
        attempt_outcome_ref=_artifact(store, f"attempt/{slot.global_slot}"),
        model_output_ref=_artifact(store, f"output/{slot.global_slot}"),
        outcome=BfclV4CallOutcome.SUCCEEDED,
    )


def _candidate_arm(
    *,
    store: ArtifactStore,
    arm: BfclV4PilotArm,
    proposal_ref: ArtifactRef,
    parent: ArtifactRef,
    candidate: ArtifactRef | None,
) -> BfclV4ArmCandidateFreeze:
    valid = candidate is not None
    return BfclV4ArmCandidateFreeze(
        arm=arm,
        proposal_completion_ref=proposal_ref,
        parent_harness_ref=parent,
        candidate_parse_ref=_artifact(store, f"parse/{arm.value}"),
        candidate_harness_ref=candidate,
        effective_candidate_harness_ref=candidate if valid else parent,
        candidate_valid=valid,
        fallback_used=not valid,
    )


def _selection(
    *,
    store: ArtifactStore,
    journal: BfclV4PublicRunJournal,
    arm: BfclV4PilotArm,
    gate_refs: tuple[ArtifactRef, ...],
    selected: BfclV4SelectedVariant,
    harness: ArtifactRef,
    forced_rollback: bool,
) -> BfclV4ArmSelection:
    assert journal.state is not None and journal.state.candidate_freeze_ref is not None
    return BfclV4ArmSelection(
        plan_fingerprint=journal.plan.fingerprint,
        arm=arm,
        candidate_freeze_ref=journal.state.candidate_freeze_ref,
        gate_completion_refs=gate_refs,
        decision_ref=_artifact(store, f"decision/{arm.value}"),
        selected_variant=selected,
        selected_harness_ref=harness,
        forced_rollback=forced_rollback,
    )


def test_candidate_contract_forces_exact_parent_fallback(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    parent = _artifact(store, "parent")
    candidate = _artifact(store, "candidate")
    completion = ArtifactRef(
        sha256="a" * 64,
        size=1,
        media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    )

    with pytest.raises(ValidationError, match="exact-parent fallback"):
        BfclV4ArmCandidateFreeze(
            arm=BfclV4PilotArm.SCORE,
            proposal_completion_ref=completion,
            parent_harness_ref=parent,
            candidate_parse_ref=_artifact(store, "parse"),
            candidate_harness_ref=candidate,
            effective_candidate_harness_ref=candidate,
            candidate_valid=False,
            fallback_used=False,
        )


def test_semantic_barriers_cas_and_offline_100_call_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test isolates journal mechanics.  The separate integration test below
    # exercises one real strict grader receipt; production replay never bypasses it.
    monkeypatch.setattr(replay_subject, "_verify_grader_binding", lambda *args: None)
    store = ArtifactStore(tmp_path)
    journal = BfclV4PublicRunJournal(store)
    parent = _artifact(store, "parent")
    candidate = _artifact(store, "score-candidate")
    bare = _artifact(store, "bare")
    static = _artifact(store, "static")
    controller = _artifact(store, "controller")
    tail = journal.open()

    for _ in range(13):
        tail, _ = _complete_next(
            journal,
            store,
            tail,
            parent=parent,
            candidate=candidate,
            bare=bare,
            static=static,
            controller=controller,
        )
    assert journal.state is not None and journal.state.next_global_slot == 13
    score_arm = _candidate_arm(
        store=store,
        arm=BfclV4PilotArm.SCORE,
        proposal_ref=journal.state.call_completion_refs[12],
        parent=parent,
        candidate=candidate,
    )
    missing_full_proposal = ArtifactRef(
        sha256="f" * 64,
        size=1,
        media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    )
    early_full = _candidate_arm(
        store=store,
        arm=BfclV4PilotArm.FULL,
        proposal_ref=missing_full_proposal,
        parent=parent,
        candidate=None,
    )
    with pytest.raises(BfclV4RunError, match="both proposals"):
        journal.freeze_candidates(expected_tail_ref=tail, score=score_arm, full=early_full)

    stale_tail = tail
    tail, _ = _complete_next(
        journal,
        store,
        tail,
        parent=parent,
        candidate=candidate,
        bare=bare,
        static=static,
        controller=controller,
    )
    with pytest.raises(BfclV4StaleTailError, match="stale"):
        journal.materialize_next_call(
            expected_tail_ref=stale_tail,
            request_ref=_artifact(store, "stale-request"),
            executed_harness_ref=candidate,
            executed_harness_variant="candidate",
        )

    assert journal.state is not None
    full_arm = _candidate_arm(
        store=store,
        arm=BfclV4PilotArm.FULL,
        proposal_ref=journal.state.call_completion_refs[13],
        parent=parent,
        candidate=None,
    )
    tail, candidate_freeze_ref = journal.freeze_candidates(
        expected_tail_ref=tail,
        score=score_arm,
        full=full_arm,
    )
    assert journal.state.candidate_freeze_ref == candidate_freeze_ref

    while journal.state.next_global_slot < 39:
        tail, _ = _complete_next(
            journal,
            store,
            tail,
            parent=parent,
            candidate=candidate,
            bare=bare,
            static=static,
            controller=controller,
        )
    fake_gate = ArtifactRef(
        sha256="e" * 64,
        size=1,
        media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    )
    score_selection = _selection(
        store=store,
        journal=journal,
        arm=BfclV4PilotArm.SCORE,
        gate_refs=journal.state.call_completion_refs[24:32],
        selected=BfclV4SelectedVariant.CANDIDATE,
        harness=candidate,
        forced_rollback=False,
    )
    early_full_selection = _selection(
        store=store,
        journal=journal,
        arm=BfclV4PilotArm.FULL,
        gate_refs=(*journal.state.call_completion_refs[32:39], fake_gate),
        selected=BfclV4SelectedVariant.PARENT,
        harness=parent,
        forced_rollback=True,
    )
    with pytest.raises(BfclV4RunError, match="sixteen GATE"):
        journal.freeze_selections(
            expected_tail_ref=tail,
            score=score_selection,
            full=early_full_selection,
        )

    tail, _ = _complete_next(
        journal,
        store,
        tail,
        parent=parent,
        candidate=candidate,
        bare=bare,
        static=static,
        controller=controller,
    )
    assert journal.state.next_global_slot == 40
    with pytest.raises(BfclV4RunError, match="HOLDOUT"):
        journal.materialize_next_call(
            expected_tail_ref=tail,
            request_ref=_artifact(store, "early-holdout"),
            executed_harness_ref=bare,
            executed_harness_variant="bare",
        )

    full_selection = _selection(
        store=store,
        journal=journal,
        arm=BfclV4PilotArm.FULL,
        gate_refs=journal.state.call_completion_refs[32:40],
        selected=BfclV4SelectedVariant.PARENT,
        harness=parent,
        forced_rollback=True,
    )
    tail, joint_selection_ref = journal.freeze_selections(
        expected_tail_ref=tail,
        score=score_selection,
        full=full_selection,
    )
    assert journal.state.joint_selection_freeze_ref == joint_selection_ref
    unlock = journal.holdout_unlock()
    assert unlock.score_selection_artifact_sha256 != unlock.full_selection_artifact_sha256

    while journal.state.next_global_slot < 100:
        tail, _ = _complete_next(
            journal,
            store,
            tail,
            parent=parent,
            candidate=candidate,
            bare=bare,
            static=static,
            controller=controller,
        )
    tail, closure_ref = journal.close(expected_tail_ref=tail)
    verification = verify_bfcl_v4_public_run_closure(store, closure_ref)
    assert verification.replayed_transition_count == 204
    assert verification.completed_model_calls == 100
    events, state = replay_bfcl_v4_public_run(store, tail)
    assert len(events) == 204
    assert state.closed is True
    assert state.next_global_slot == 100


def test_replay_rejects_a_persisted_state_not_derived_from_its_event(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    journal = BfclV4PublicRunJournal(store)
    tail = journal.open()
    entry = store.get_json(tail, BfclV4RunJournalEntry)
    wrong_state = BfclV4RunState(plan_fingerprint="f" * 64)
    wrong_state_ref = store.put_json(wrong_state, media_type=BFCL_V4_RUN_STATE_MEDIA_TYPE)
    forged_entry = BfclV4RunJournalEntry(
        plan_fingerprint=journal.plan.fingerprint,
        sequence=0,
        previous_entry_ref=None,
        previous_state_ref=None,
        event_ref=entry.event_ref,
        state_ref=wrong_state_ref,
    )
    forged_tail = store.put_json(forged_entry, media_type=BFCL_V4_RUN_ENTRY_MEDIA_TYPE)

    with pytest.raises(BfclV4RunIntegrityError, match="semantic replay"):
        replay_bfcl_v4_public_run(store, forged_tail)


def test_one_real_grader_receipt_closes_into_its_exact_call_slot(tmp_path: Path) -> None:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else Path("/tmp/spiral-bfcl-upstream")
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable")
    try:
        _checkout_and_git(checkout)
    except BfclV4PublicGraderError as error:
        pytest.skip(f"pinned BFCL checkout is unusable: {error}")

    store = ArtifactStore(tmp_path)
    journal = BfclV4PublicRunJournal(store)
    tail = journal.open()
    parent = _artifact(store, "parent")
    tail, materialization_ref = journal.materialize_next_call(
        expected_tail_ref=tail,
        request_ref=_artifact(store, "request"),
        executed_harness_ref=parent,
        executed_harness_variant="parent",
    )
    materialization = store.get_json(materialization_ref)
    slot = journal.plan.calls[0]
    prediction = make_bfcl_v4_public_prediction(slot.task_id, ())
    binding = BfclV4GradingSlotBinding(
        plan_fingerprint=journal.plan.fingerprint,
        call_slot_reference_sha256=materialization["call_slot_reference_sha256"],
        call_id=slot.call_id,
        arm="score",
        grade_role="parent-fit",
        intended_harness_variant="parent",
        executed_harness_variant="parent",
        task_id=slot.task_id,
        prediction_sha256=prediction.fingerprint,
    )
    receipt = grade_bfcl_v4_public_prediction(prediction, binding, checkout)
    prediction_ref = store.put_json(prediction)
    receipt_ref = store.put_json(receipt)
    tail, _ = journal.complete_call(
        expected_tail_ref=tail,
        materialization_ref=materialization_ref,
        attempt_outcome_ref=_artifact(store, "attempt"),
        model_output_ref=prediction_ref,
        outcome=BfclV4CallOutcome.SUCCEEDED,
        prediction_ref=prediction_ref,
        grader_receipt_ref=receipt_ref,
    )
    assert journal.tail_ref == tail
    assert journal.state is not None and journal.state.next_global_slot == 1
