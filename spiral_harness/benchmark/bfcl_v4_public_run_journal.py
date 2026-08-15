"""Public semantic CAS journal API for the BFCL V4 development pilot."""

from __future__ import annotations

from threading import RLock

from spiral_harness.benchmark._bfcl_v4_public_run_replay import (
    BfclV4RunCycleError,
    BfclV4RunError,
    BfclV4RunIntegrityError,
    BfclV4StaleTailError,
    _apply_event,
    _checked_ref,
    _load,
    _load_joint_selection,
    _plan,
    _publish,
    _read_entries,
    _verify_candidate_freeze,
    _verify_completion,
    _verify_joint_selection,
    _verify_materialization,
    replay_bfcl_v4_public_run,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import BfclV4HoldoutUnlock
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
    BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
    BFCL_V4_RUN_EVENT_MEDIA_TYPE,
    BFCL_V4_RUN_STATE_MEDIA_TYPE,
    BfclV4ArmCandidateFreeze,
    BfclV4ArmSelection,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4CallOutcome,
    BfclV4JointCandidateFreeze,
    BfclV4JointSelectionFreeze,
    BfclV4RunAction,
    BfclV4RunClosure,
    BfclV4RunClosureVerification,
    BfclV4RunEvent,
    BfclV4RunJournalEntry,
    BfclV4RunState,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository


def verify_bfcl_v4_public_run_closure(
    repository: ArtifactRepository,
    closure_ref: ArtifactRef,
    *,
    plan: BfclV4PublicPilotCallPlan | None = None,
) -> BfclV4RunClosureVerification:
    """Reload a portable closure and strictly replay all 100 call lineages."""

    checked_plan = _plan(plan)
    checked_closure_ref = _checked_ref(closure_ref, BFCL_V4_RUN_CLOSURE_MEDIA_TYPE, "run closure")
    closure = _load(
        repository,
        checked_closure_ref,
        BfclV4RunClosure,
        "run closure",
        media_type=BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    )
    events, state = replay_bfcl_v4_public_run(
        repository, closure.journal_tail_ref, plan=checked_plan
    )
    entries = _read_entries(repository, closure.journal_tail_ref)
    if len(events) != 204:
        raise BfclV4RunIntegrityError("closed run must contain exactly 204 semantic transitions")
    if (
        closure.plan_fingerprint != checked_plan.fingerprint
        or not state.closed
        or entries[-1][1].state_ref != closure.final_state_ref
        or state.candidate_freeze_ref != closure.candidate_freeze_ref
        or state.joint_selection_freeze_ref != closure.joint_selection_freeze_ref
        or state.call_completion_refs != closure.call_completion_refs
    ):
        raise BfclV4RunIntegrityError("closure differs from the replayed terminal state")
    actions = tuple(event.action for event in events)
    expected_counts = {
        BfclV4RunAction.OPEN: 1,
        BfclV4RunAction.MATERIALIZE_CALL: 100,
        BfclV4RunAction.COMPLETE_CALL: 100,
        BfclV4RunAction.FREEZE_CANDIDATES: 1,
        BfclV4RunAction.FREEZE_SELECTIONS: 1,
        BfclV4RunAction.CLOSE: 1,
    }
    if any(actions.count(action) != count for action, count in expected_counts.items()):
        raise BfclV4RunIntegrityError("closure action multiplicities differ from the protocol")
    return BfclV4RunClosureVerification(
        closure_fingerprint=closure.fingerprint,
        plan_fingerprint=checked_plan.fingerprint,
        replayed_transition_count=len(events),
    )


class BfclV4PublicRunJournal:
    """Process-local writer with caller-held-tail compare-and-set checks.

    An optional tail can be replayed for crash recovery, but there is no
    cross-process lock, lease, authenticated mutable head, or durable CAS.
    Semantic activation occurs only when an artifact is appended to this
    journal; individually persisted candidate/selection objects are inert.
    """

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        plan: BfclV4PublicPilotCallPlan | None = None,
        tail_ref: ArtifactRef | None = None,
    ) -> None:
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must implement ArtifactRepository")
        self._repository = repository
        self._plan = _plan(plan)
        self._lock = RLock()
        self._tail_ref: ArtifactRef | None = None
        self._state_ref: ArtifactRef | None = None
        self._state: BfclV4RunState | None = None
        self._sequence = -1
        if tail_ref is not None:
            entries = _read_entries(repository, tail_ref)
            _, state = replay_bfcl_v4_public_run(repository, tail_ref, plan=self._plan)
            self._tail_ref = entries[-1][0]
            self._state_ref = entries[-1][1].state_ref
            self._state = state
            self._sequence = entries[-1][1].sequence

    @property
    def plan(self) -> BfclV4PublicPilotCallPlan:
        return self._plan

    @property
    def tail_ref(self) -> ArtifactRef | None:
        with self._lock:
            return self._tail_ref

    @property
    def state(self) -> BfclV4RunState | None:
        with self._lock:
            return self._state

    def _require_tail(self, expected_tail_ref: ArtifactRef) -> None:
        expected = _checked_ref(expected_tail_ref, BFCL_V4_RUN_ENTRY_MEDIA_TYPE, "expected tail")
        if expected != self._tail_ref:
            raise BfclV4StaleTailError("expected tail is stale or belongs to another run branch")

    def _append(self, event: BfclV4RunEvent) -> ArtifactRef:
        next_state = _apply_event(self._repository, self._plan, self._state, event)
        event_ref = _publish(
            self._repository,
            event,
            BfclV4RunEvent,
            BFCL_V4_RUN_EVENT_MEDIA_TYPE,
            "run event",
        )
        state_ref = _publish(
            self._repository,
            next_state,
            BfclV4RunState,
            BFCL_V4_RUN_STATE_MEDIA_TYPE,
            "run state",
        )
        entry = BfclV4RunJournalEntry(
            plan_fingerprint=self._plan.fingerprint,
            sequence=self._sequence + 1,
            previous_entry_ref=self._tail_ref,
            previous_state_ref=self._state_ref,
            event_ref=event_ref,
            state_ref=state_ref,
        )
        tail_ref = _publish(
            self._repository,
            entry,
            BfclV4RunJournalEntry,
            BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
            "run journal entry",
        )
        self._tail_ref, self._state_ref, self._state = tail_ref, state_ref, next_state
        self._sequence += 1
        return tail_ref

    def open(self) -> ArtifactRef:
        """Create the unique OPEN root for this controller object."""

        with self._lock:
            if self._tail_ref is not None:
                raise BfclV4RunError("run journal is already open")
            return self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.OPEN,
                )
            )

    def materialize_next_call(
        self,
        *,
        expected_tail_ref: ArtifactRef,
        request_ref: ArtifactRef,
        executed_harness_ref: ArtifactRef,
        executed_harness_variant: str,
        fallback_used: bool = False,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        """Authorize and activate exactly the next frozen model-call slot."""

        with self._lock:
            self._require_tail(expected_tail_ref)
            if self._state is None or self._state.next_global_slot >= 100:
                raise BfclV4RunError("run is unopened or its frozen call schedule is complete")
            slot = self._plan.calls[self._state.next_global_slot]
            materialization = BfclV4CallMaterialization(
                plan_fingerprint=self._plan.fingerprint,
                slot=slot,
                call_slot_reference_sha256=canonical_sha256(slot),
                request_ref=request_ref,
                executed_harness_ref=executed_harness_ref,
                intended_harness_variant=slot.harness_variant,
                executed_harness_variant=executed_harness_variant,
                fallback_used=fallback_used,
                candidate_freeze_ref=(
                    self._state.candidate_freeze_ref if slot.global_slot >= 14 else None
                ),
                joint_selection_freeze_ref=(
                    self._state.joint_selection_freeze_ref if slot.global_slot >= 40 else None
                ),
            )
            _verify_materialization(self._repository, self._plan, self._state, materialization)
            ref = _publish(
                self._repository,
                materialization,
                BfclV4CallMaterialization,
                BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
                "call materialization",
            )
            tail = self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.MATERIALIZE_CALL,
                    materialization_ref=ref,
                )
            )
            return tail, ref

    def complete_call(
        self,
        *,
        expected_tail_ref: ArtifactRef,
        materialization_ref: ArtifactRef,
        attempt_outcome_ref: ArtifactRef,
        model_output_ref: ArtifactRef,
        outcome: BfclV4CallOutcome,
        prediction_ref: ArtifactRef | None = None,
        grader_receipt_ref: ArtifactRef | None = None,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        """Close one materialization; failures still consume its one frozen slot."""

        with self._lock:
            self._require_tail(expected_tail_ref)
            if self._state is None or self._state.open_materialization_ref is None:
                raise BfclV4RunError("there is no open call to complete")
            materialization = _load(
                self._repository,
                materialization_ref,
                BfclV4CallMaterialization,
                "call materialization",
                media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
            )
            completion = BfclV4CallCompletion(
                plan_fingerprint=self._plan.fingerprint,
                call_id=materialization.slot.call_id,
                global_slot=materialization.slot.global_slot,
                call_slot_reference_sha256=materialization.call_slot_reference_sha256,
                materialization_ref=materialization_ref,
                attempt_outcome_ref=attempt_outcome_ref,
                model_output_ref=model_output_ref,
                outcome=outcome,
                prediction_ref=prediction_ref,
                grader_receipt_ref=grader_receipt_ref,
            )
            _verify_completion(self._repository, self._plan, self._state, completion)
            ref = _publish(
                self._repository,
                completion,
                BfclV4CallCompletion,
                BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
                "call completion",
            )
            tail = self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.COMPLETE_CALL,
                    completion_ref=ref,
                )
            )
            return tail, ref

    def freeze_candidates(
        self,
        *,
        expected_tail_ref: ArtifactRef,
        score: BfclV4ArmCandidateFreeze,
        full: BfclV4ArmCandidateFreeze,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        """Atomically activate both candidates after both proposal completions."""

        with self._lock:
            self._require_tail(expected_tail_ref)
            if self._state is None:
                raise BfclV4RunError("run journal is not open")
            freeze = BfclV4JointCandidateFreeze(
                plan_fingerprint=self._plan.fingerprint,
                score=score,
                full=full,
            )
            _verify_candidate_freeze(self._repository, self._plan, self._state, freeze)
            ref = _publish(
                self._repository,
                freeze,
                BfclV4JointCandidateFreeze,
                BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
                "joint candidate freeze",
            )
            tail = self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.FREEZE_CANDIDATES,
                    candidate_freeze_ref=ref,
                )
            )
            return tail, ref

    def freeze_selections(
        self,
        *,
        expected_tail_ref: ArtifactRef,
        score: BfclV4ArmSelection,
        full: BfclV4ArmSelection,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        """Activate both selections in one event only after all 16 GATE calls."""

        with self._lock:
            self._require_tail(expected_tail_ref)
            if self._state is None or self._state.candidate_freeze_ref is None:
                raise BfclV4RunError("candidate freeze is absent")
            if self._state.next_global_slot != 40 or self._state.open_materialization_ref:
                raise BfclV4RunError(
                    "all sixteen GATE calls must finish before either selection is persisted"
                )
            score = BfclV4ArmSelection.model_validate(score, strict=True)
            full = BfclV4ArmSelection.model_validate(full, strict=True)
            score_ref = _publish(
                self._repository,
                score,
                BfclV4ArmSelection,
                BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
                "SCORE selection",
            )
            full_ref = _publish(
                self._repository,
                full,
                BfclV4ArmSelection,
                BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
                "FULL selection",
            )
            joint = BfclV4JointSelectionFreeze(
                plan_fingerprint=self._plan.fingerprint,
                candidate_freeze_ref=self._state.candidate_freeze_ref,
                score_selection_ref=score_ref,
                full_selection_ref=full_ref,
                gate_completion_refs=self._state.call_completion_refs[24:40],
            )
            _verify_joint_selection(self._repository, self._plan, self._state, joint)
            joint_ref = _publish(
                self._repository,
                joint,
                BfclV4JointSelectionFreeze,
                BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
                "joint selection freeze",
            )
            tail = self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.FREEZE_SELECTIONS,
                    joint_selection_freeze_ref=joint_ref,
                )
            )
            return tail, joint_ref

    def holdout_unlock(self) -> BfclV4HoldoutUnlock:
        """Derive the exact grader unlock from the current joint selection."""

        with self._lock:
            if self._state is None:
                raise BfclV4RunError("run journal is not open")
            joint, _, _ = _load_joint_selection(self._repository, self._state)
            return BfclV4HoldoutUnlock(
                plan_fingerprint=self._plan.fingerprint,
                score_selection_artifact_sha256=joint.score_selection_ref.sha256,
                full_selection_artifact_sha256=joint.full_selection_ref.sha256,
            )

    def close(self, *, expected_tail_ref: ArtifactRef) -> tuple[ArtifactRef, ArtifactRef]:
        """Close exactly 100 calls and publish a portable replay closure."""

        with self._lock:
            self._require_tail(expected_tail_ref)
            if self._state is None:
                raise BfclV4RunError("run journal is not open")
            tail = self._append(
                BfclV4RunEvent(
                    plan_fingerprint=self._plan.fingerprint,
                    action=BfclV4RunAction.CLOSE,
                )
            )
            if (
                self._state_ref is None
                or self._state.candidate_freeze_ref is None
                or self._state.joint_selection_freeze_ref is None
            ):  # pragma: no cover - CLOSE transition guarantees these values
                raise BfclV4RunIntegrityError("terminal state lacks closure references")
            closure = BfclV4RunClosure(
                plan_fingerprint=self._plan.fingerprint,
                journal_tail_ref=tail,
                final_state_ref=self._state_ref,
                candidate_freeze_ref=self._state.candidate_freeze_ref,
                joint_selection_freeze_ref=self._state.joint_selection_freeze_ref,
                call_completion_refs=self._state.call_completion_refs,
            )
            closure_ref = _publish(
                self._repository,
                closure,
                BfclV4RunClosure,
                BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
                "run closure",
            )
            return tail, closure_ref


__all__ = [
    "BfclV4PublicRunJournal",
    "BfclV4RunCycleError",
    "BfclV4RunError",
    "BfclV4RunIntegrityError",
    "BfclV4StaleTailError",
    "replay_bfcl_v4_public_run",
    "verify_bfcl_v4_public_run_closure",
]
