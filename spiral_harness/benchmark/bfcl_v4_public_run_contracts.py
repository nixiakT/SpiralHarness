"""Immutable semantic-ledger contracts for the BFCL V4 public pilot.

The artifacts in this module control ordering and provenance for the frozen
100-call development pilot.  They deliberately do not upgrade public BFCL
data into hidden, sealed, official-full-suite, or reportable evidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallSlot,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_RUN_EVENT_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-run-event.v1+json"
BFCL_V4_RUN_ENTRY_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-run-entry.v1+json"
BFCL_V4_RUN_STATE_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-run-state.v1+json"
BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-call-materialization.v1+json"
)
BFCL_V4_CALL_COMPLETION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-call-completion.v1+json"
)
BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-joint-candidate-freeze.v1+json"
)
BFCL_V4_ARM_SELECTION_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-arm-selection.v1+json"
BFCL_V4_JOINT_SELECTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-joint-selection.v1+json"
)
BFCL_V4_RUN_CLOSURE_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-run-closure.v1+json"


def _require_media(ref: ArtifactRef, media_type: str, label: str) -> None:
    if ref.media_type != media_type:
        raise ValueError(f"{label} must declare {media_type}")


class BfclV4RunAction(StrEnum):
    """The complete semantic transition vocabulary for one run."""

    OPEN = "open"
    MATERIALIZE_CALL = "materialize-call"
    COMPLETE_CALL = "complete-call"
    FREEZE_CANDIDATES = "freeze-candidates"
    FREEZE_SELECTIONS = "freeze-selections"
    CLOSE = "close"


class BfclV4CallOutcome(StrEnum):
    """Whether the one frozen provider attempt returned a usable response."""

    SUCCEEDED = "succeeded"
    PROVIDER_FAILURE = "provider-failure"


class BfclV4SelectedVariant(StrEnum):
    """The only adaptive-arm choices permitted after GATE."""

    PARENT = "parent"
    CANDIDATE = "candidate"


class BfclV4ArmCandidateFreeze(ImmutableModel):
    """One arm's parsed proposal and exact invalid-candidate fallback."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4PilotArm
    proposal_completion_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_parse_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef | None = None
    effective_candidate_harness_ref: ArtifactRef
    candidate_valid: bool
    fallback_used: bool

    @model_validator(mode="after")
    def _close_fallback(self) -> Self:
        if self.arm not in {BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL}:
            raise ValueError("candidate freeze arm must be SCORE or FULL")
        _require_media(
            self.proposal_completion_ref,
            BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
            "proposal_completion_ref",
        )
        if self.candidate_valid:
            if self.candidate_harness_ref is None:
                raise ValueError("valid candidate must bind a candidate harness")
            if self.candidate_harness_ref == self.parent_harness_ref:
                raise ValueError("valid candidate harness must differ from its parent")
            if self.effective_candidate_harness_ref != self.candidate_harness_ref:
                raise ValueError("valid candidate must execute its candidate harness")
            if self.fallback_used:
                raise ValueError("valid candidate must not claim parent fallback")
        elif (
            self.candidate_harness_ref is not None
            or self.effective_candidate_harness_ref != self.parent_harness_ref
            or not self.fallback_used
        ):
            raise ValueError("invalid candidate must use exact-parent fallback")
        return self


class BfclV4JointCandidateFreeze(ImmutableModel):
    """Atomic activation artifact for both arm candidates."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    score: BfclV4ArmCandidateFreeze
    full: BfclV4ArmCandidateFreeze
    both_proposals_complete: Literal[True] = True
    both_candidates_frozen: Literal[True] = True
    invalid_candidate_slot_policy: Literal["parent-fallback-consumes-all-frozen-slots"] = (
        "parent-fallback-consumes-all-frozen-slots"
    )

    @model_validator(mode="after")
    def _close_joint_candidate(self) -> Self:
        if self.score.arm != BfclV4PilotArm.SCORE or self.full.arm != BfclV4PilotArm.FULL:
            raise ValueError("joint candidate freeze must contain SCORE then FULL")
        if self.score.parent_harness_ref != self.full.parent_harness_ref:
            raise ValueError("SCORE and FULL must start from the same parent harness")
        if self.score.proposal_completion_ref == self.full.proposal_completion_ref:
            raise ValueError("SCORE and FULL proposals must be distinct completions")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4ArmSelection(ImmutableModel):
    """One GATE-derived selection, stored before atomic joint activation."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    arm: BfclV4PilotArm
    candidate_freeze_ref: ArtifactRef
    gate_completion_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=8, max_length=8)]
    decision_ref: ArtifactRef
    selected_variant: BfclV4SelectedVariant
    selected_harness_ref: ArtifactRef
    forced_rollback: bool

    @model_validator(mode="after")
    def _close_selection_shape(self) -> Self:
        if self.arm not in {BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL}:
            raise ValueError("selection arm must be SCORE or FULL")
        _require_media(
            self.candidate_freeze_ref,
            BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
            "candidate_freeze_ref",
        )
        for ref in self.gate_completion_refs:
            _require_media(ref, BFCL_V4_CALL_COMPLETION_MEDIA_TYPE, "gate completion ref")
        if len({ref.sha256 for ref in self.gate_completion_refs}) != 8:
            raise ValueError("selection must bind eight distinct GATE completions")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4JointSelectionFreeze(ImmutableModel):
    """Single semantic commit that activates SCORE and FULL selections together."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    candidate_freeze_ref: ArtifactRef
    score_selection_ref: ArtifactRef
    full_selection_ref: ArtifactRef
    gate_completion_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=16, max_length=16)]
    all_sixteen_gate_calls_complete: Literal[True] = True
    both_selections_frozen: Literal[True] = True
    selections_final_before_holdout_materialization: Literal[True] = True
    holdout_can_continue_search: Literal[False] = False

    @model_validator(mode="after")
    def _close_joint_selection(self) -> Self:
        _require_media(
            self.candidate_freeze_ref,
            BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
            "candidate_freeze_ref",
        )
        _require_media(
            self.score_selection_ref,
            BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
            "score_selection_ref",
        )
        _require_media(
            self.full_selection_ref,
            BFCL_V4_ARM_SELECTION_MEDIA_TYPE,
            "full_selection_ref",
        )
        if self.score_selection_ref == self.full_selection_ref:
            raise ValueError("SCORE and FULL selection artifacts must be distinct")
        for ref in self.gate_completion_refs:
            _require_media(ref, BFCL_V4_CALL_COMPLETION_MEDIA_TYPE, "gate completion ref")
        if len({ref.sha256 for ref in self.gate_completion_refs}) != 16:
            raise ValueError("joint selection must bind sixteen distinct GATE completions")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CallMaterialization(ImmutableModel):
    """Authorization-time binding for exactly one slot in the frozen plan."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    slot: BfclV4PilotCallSlot
    call_slot_reference_sha256: Sha256
    request_ref: ArtifactRef
    executed_harness_ref: ArtifactRef
    intended_harness_variant: NonEmptyStr
    executed_harness_variant: NonEmptyStr
    fallback_used: bool = False
    candidate_freeze_ref: ArtifactRef | None = None
    joint_selection_freeze_ref: ArtifactRef | None = None
    provider_attempt_ceiling: Literal[1] = 1

    @model_validator(mode="after")
    def _bind_slot(self) -> Self:
        if self.call_slot_reference_sha256 != canonical_sha256(self.slot):
            raise ValueError("call slot reference differs from the exact typed slot")
        if self.intended_harness_variant != self.slot.harness_variant:
            raise ValueError("intended harness variant differs from the frozen slot")
        if self.candidate_freeze_ref is not None:
            _require_media(
                self.candidate_freeze_ref,
                BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
                "candidate_freeze_ref",
            )
        if self.joint_selection_freeze_ref is not None:
            _require_media(
                self.joint_selection_freeze_ref,
                BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
                "joint_selection_freeze_ref",
            )
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4CallCompletion(ImmutableModel):
    """One consumed provider slot and its candidate/grader evidence references."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    call_id: NonEmptyStr
    global_slot: Annotated[int, Field(ge=0, lt=100, strict=True)]
    call_slot_reference_sha256: Sha256
    materialization_ref: ArtifactRef
    attempt_outcome_ref: ArtifactRef
    model_output_ref: ArtifactRef
    outcome: BfclV4CallOutcome
    prediction_ref: ArtifactRef | None = None
    grader_receipt_ref: ArtifactRef | None = None
    provider_attempts_consumed: Literal[1] = 1

    @model_validator(mode="after")
    def _close_completion(self) -> Self:
        _require_media(
            self.materialization_ref,
            BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
            "materialization_ref",
        )
        if (self.prediction_ref is None) != (self.grader_receipt_ref is None):
            raise ValueError("prediction and grader receipt references must appear together")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4RunState(ImmutableModel):
    """Replayable semantic state after one accepted journal transition."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    next_global_slot: Annotated[int, Field(ge=0, le=100, strict=True)] = 0
    open_materialization_ref: ArtifactRef | None = None
    call_completion_refs: Annotated[tuple[ArtifactRef, ...], Field(max_length=100)] = ()
    candidate_freeze_ref: ArtifactRef | None = None
    joint_selection_freeze_ref: ArtifactRef | None = None
    closed: bool = False

    @model_validator(mode="after")
    def _close_state(self) -> Self:
        if len(self.call_completion_refs) != self.next_global_slot:
            raise ValueError("completion count must equal the next global slot")
        if len({ref.sha256 for ref in self.call_completion_refs}) != len(self.call_completion_refs):
            raise ValueError("call completion references must not repeat")
        for ref in self.call_completion_refs:
            _require_media(ref, BFCL_V4_CALL_COMPLETION_MEDIA_TYPE, "call completion ref")
        if self.open_materialization_ref is not None:
            _require_media(
                self.open_materialization_ref,
                BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
                "open_materialization_ref",
            )
            if self.next_global_slot == 100:
                raise ValueError("a complete call schedule cannot retain an open materialization")
        if self.candidate_freeze_ref is not None:
            _require_media(
                self.candidate_freeze_ref,
                BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
                "candidate_freeze_ref",
            )
            if self.next_global_slot < 14:
                raise ValueError("candidate freeze cannot precede both proposal completions")
        if self.next_global_slot > 14 and self.candidate_freeze_ref is None:
            raise ValueError("candidate FIT progress requires the joint candidate freeze")
        if self.joint_selection_freeze_ref is not None:
            _require_media(
                self.joint_selection_freeze_ref,
                BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
                "joint_selection_freeze_ref",
            )
            if self.next_global_slot < 40:
                raise ValueError("selection freeze cannot precede all sixteen GATE calls")
        if self.next_global_slot > 40 and self.joint_selection_freeze_ref is None:
            raise ValueError("HOLDOUT progress requires the joint selection freeze")
        if self.closed and (
            self.next_global_slot != 100
            or self.open_materialization_ref is not None
            or self.joint_selection_freeze_ref is None
        ):
            raise ValueError("closed run must contain all 100 calls and both selections")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4RunEvent(ImmutableModel):
    """One typed journal command with exactly one action payload."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    action: BfclV4RunAction
    materialization_ref: ArtifactRef | None = None
    completion_ref: ArtifactRef | None = None
    candidate_freeze_ref: ArtifactRef | None = None
    joint_selection_freeze_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def _one_payload(self) -> Self:
        values = {
            BfclV4RunAction.MATERIALIZE_CALL: self.materialization_ref,
            BfclV4RunAction.COMPLETE_CALL: self.completion_ref,
            BfclV4RunAction.FREEZE_CANDIDATES: self.candidate_freeze_ref,
            BfclV4RunAction.FREEZE_SELECTIONS: self.joint_selection_freeze_ref,
        }
        expected = values.get(self.action)
        populated = tuple(value for value in values.values() if value is not None)
        if self.action in {BfclV4RunAction.OPEN, BfclV4RunAction.CLOSE}:
            if populated:
                raise ValueError("OPEN/CLOSE event must not contain an artifact payload")
        elif expected is None or len(populated) != 1:
            raise ValueError("event must contain exactly its action-specific artifact payload")
        if self.materialization_ref is not None:
            _require_media(
                self.materialization_ref,
                BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
                "materialization_ref",
            )
        if self.completion_ref is not None:
            _require_media(
                self.completion_ref,
                BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
                "completion_ref",
            )
        if self.candidate_freeze_ref is not None:
            _require_media(
                self.candidate_freeze_ref,
                BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
                "candidate_freeze_ref",
            )
        if self.joint_selection_freeze_ref is not None:
            _require_media(
                self.joint_selection_freeze_ref,
                BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
                "joint_selection_freeze_ref",
            )
        return self


class BfclV4RunJournalEntry(ImmutableModel):
    """One immutable state transition in a caller-held-tail CAS chain."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    sequence: Annotated[int, Field(ge=0, strict=True)]
    previous_entry_ref: ArtifactRef | None
    previous_state_ref: ArtifactRef | None
    event_ref: ArtifactRef
    state_ref: ArtifactRef

    @model_validator(mode="after")
    def _close_links(self) -> Self:
        if self.sequence == 0:
            if self.previous_entry_ref is not None or self.previous_state_ref is not None:
                raise ValueError("journal root cannot reference a previous entry or state")
        elif self.previous_entry_ref is None or self.previous_state_ref is None:
            raise ValueError("non-root journal entry requires previous entry and state refs")
        if self.previous_entry_ref is not None:
            _require_media(
                self.previous_entry_ref,
                BFCL_V4_RUN_ENTRY_MEDIA_TYPE,
                "previous_entry_ref",
            )
        if self.previous_state_ref is not None:
            _require_media(
                self.previous_state_ref,
                BFCL_V4_RUN_STATE_MEDIA_TYPE,
                "previous_state_ref",
            )
        _require_media(self.event_ref, BFCL_V4_RUN_EVENT_MEDIA_TYPE, "event_ref")
        _require_media(self.state_ref, BFCL_V4_RUN_STATE_MEDIA_TYPE, "state_ref")
        return self


class BfclV4RunClosure(ImmutableModel):
    """Portable input to the offline closure verifier."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    journal_tail_ref: ArtifactRef
    final_state_ref: ArtifactRef
    candidate_freeze_ref: ArtifactRef
    joint_selection_freeze_ref: ArtifactRef
    call_completion_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=100, max_length=100)]
    total_model_calls: Literal[100] = 100
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_refs(self) -> Self:
        _require_media(self.journal_tail_ref, BFCL_V4_RUN_ENTRY_MEDIA_TYPE, "journal_tail_ref")
        _require_media(self.final_state_ref, BFCL_V4_RUN_STATE_MEDIA_TYPE, "final_state_ref")
        _require_media(
            self.candidate_freeze_ref,
            BFCL_V4_CANDIDATE_FREEZE_MEDIA_TYPE,
            "candidate_freeze_ref",
        )
        _require_media(
            self.joint_selection_freeze_ref,
            BFCL_V4_JOINT_SELECTION_MEDIA_TYPE,
            "joint_selection_freeze_ref",
        )
        for ref in self.call_completion_refs:
            _require_media(ref, BFCL_V4_CALL_COMPLETION_MEDIA_TYPE, "call completion ref")
        if len({ref.sha256 for ref in self.call_completion_refs}) != 100:
            raise ValueError("closure must bind 100 distinct call completions")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4RunClosureVerification(ImmutableModel):
    """Offline replay result, retaining the public-development claim boundary."""

    schema_version: Literal["1"] = "1"
    closure_fingerprint: Sha256
    plan_fingerprint: Sha256
    replayed_transition_count: Annotated[int, Field(ge=204, strict=True)]
    completed_model_calls: Literal[100] = 100
    completed_gate_calls_before_joint_selection: Literal[16] = 16
    holdout_calls_after_joint_selection: Literal[60] = 60
    semantic_cas_chain_verified: Literal[True] = True
    candidate_barrier_verified: Literal[True] = True
    joint_selection_barrier_verified: Literal[True] = True
    grader_receipt_bindings_verified: Literal[True] = True
    offline_closure_verified: Literal[True] = True
    cross_process_durable_cas_attested: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
