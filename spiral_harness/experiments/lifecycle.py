"""Immutable experiment lifecycle records used by the trusted controller."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr
from spiral_harness.storage.protocol import ArtifactRepository

EXPERIMENT_LIFECYCLE_EVENT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-lifecycle-event.v1+json"
)
EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-journal-entry.v1+json"
)
SELECTION_CLOSURE_MEDIA_TYPE = "application/vnd.spiral-harness.selection-closure.v1+json"
SEALED_RUN_AUTHORIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.sealed-run-authorization.v1+json"
)
EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-completion-report.v1+json"
)
SEALED_EVALUATION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.sealed-evaluation-report.v1+json"
)
EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-invalidation-report.v1+json"
)


class ExperimentState(StrEnum):
    """Trusted persisted states of one frozen experiment."""

    FROZEN = "frozen"
    SEARCHING = "searching"
    SELECTION_CLOSED = "selection_closed"
    SEALED_RUNNING = "sealed_running"
    COMPLETE = "complete"
    INVALIDATED = "invalidated"


TERMINAL_EXPERIMENT_STATES = frozenset({ExperimentState.COMPLETE, ExperimentState.INVALIDATED})

_ALLOWED_TRANSITIONS = MappingProxyType(
    {
        ExperimentState.FROZEN: frozenset({ExperimentState.SEARCHING, ExperimentState.INVALIDATED}),
        ExperimentState.SEARCHING: frozenset(
            {ExperimentState.SELECTION_CLOSED, ExperimentState.INVALIDATED}
        ),
        ExperimentState.SELECTION_CLOSED: frozenset(
            {ExperimentState.SEALED_RUNNING, ExperimentState.INVALIDATED}
        ),
        ExperimentState.SEALED_RUNNING: frozenset(
            {ExperimentState.COMPLETE, ExperimentState.INVALIDATED}
        ),
    }
)


class ExperimentViolationCode(StrEnum):
    """Controller-classified causes that invalidate an experiment lineage."""

    INTEGRITY = "integrity"
    LEAKAGE = "leakage"


class SelectionReason(StrEnum):
    """Controller-derived reason that bounded search can be closed."""

    PROMOTED_CHAMPION = "promoted_champion"
    NO_PROMOTABLE_CANDIDATES = "no_promotable_candidates"


class ExperimentLifecycleError(RuntimeError):
    """Raised when an experiment chain is malformed or disconnected."""


class ExperimentLifecycleEvent(ImmutableModel):
    """One immutable semantic transition for a frozen experiment."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    from_state: ExperimentState | None
    to_state: ExperimentState
    evidence_refs: tuple[ArtifactRef, ...] = ()
    usage_tail_ref: ArtifactRef | None = None
    reason: NonEmptyStr

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_evidence_refs(
        cls,
        refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(refs, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("evidence_refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def transition_is_legal(self) -> ExperimentLifecycleEvent:
        if self.from_state is None:
            if self.to_state is not ExperimentState.FROZEN:
                raise ValueError("only FROZEN may be emitted without a prior state")
            if self.evidence_refs:
                raise ValueError("FROZEN root does not accept caller evidence")
            if self.usage_tail_ref is not None:
                raise ValueError("FROZEN root cannot have experiment usage")
            return self
        if self.to_state not in _ALLOWED_TRANSITIONS.get(self.from_state, frozenset()):
            raise ValueError(
                f"illegal experiment transition: {self.from_state.name} -> {self.to_state.name}"
            )
        if self.to_state is not ExperimentState.SEARCHING and not self.evidence_refs:
            raise ValueError(f"transition to {self.to_state.name} requires evidence")
        if self.to_state is ExperimentState.SEARCHING and self.usage_tail_ref is not None:
            raise ValueError("SEARCHING must begin before any recorded gate usage")
        return self


class ExperimentJournalEntry(ImmutableModel):
    """One immutable link in an experiment lifecycle journal."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    sequence: Annotated[int, Field(ge=0, strict=True)]
    event_ref: ArtifactRef
    previous_entry_ref: ArtifactRef | None

    @model_validator(mode="after")
    def root_and_link_shape_matches_sequence(self) -> ExperimentJournalEntry:
        if self.event_ref.media_type != EXPERIMENT_LIFECYCLE_EVENT_MEDIA_TYPE:
            raise ValueError("event_ref declares the wrong experiment lifecycle media type")
        if (
            self.previous_entry_ref is not None
            and self.previous_entry_ref.media_type != EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE
        ):
            raise ValueError("previous_entry_ref declares the wrong experiment journal media type")
        if self.sequence == 0 and self.previous_entry_ref is not None:
            raise ValueError("experiment sequence 0 must not have a previous entry")
        if self.sequence > 0 and self.previous_entry_ref is None:
            raise ValueError("experiment sequence greater than 0 requires a previous entry")
        return self


class SelectionClosure(ImmutableModel):
    """Freeze the selected champion and analysis before sealed access."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    champion_candidate_tail_ref: ArtifactRef | None
    champion_harness_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    usage_tail_ref: ArtifactRef | None
    selection_reason: SelectionReason
    stopping_criteria: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def champion_candidate_refs_are_paired(self) -> SelectionClosure:
        if (self.champion_candidate_ref is None) != (self.champion_candidate_tail_ref is None):
            raise ValueError("champion candidate and tail refs must be present together")
        if (
            self.selection_reason is SelectionReason.PROMOTED_CHAMPION
            and self.champion_candidate_ref is None
        ):
            raise ValueError("promoted champion selection requires candidate refs")
        if (
            self.selection_reason is SelectionReason.NO_PROMOTABLE_CANDIDATES
            and self.champion_candidate_ref is not None
        ):
            raise ValueError("seed fallback must not claim a promoted candidate")
        return self


class SealedRunAuthorization(ImmutableModel):
    """Authorize sealed access only from one exact selection-closed branch."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    selection_closed_tail_ref: ArtifactRef
    selection_closure_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    champion_harness_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    sealed_split_ref: ArtifactRef
    usage_tail_ref: ArtifactRef | None


class SealedEvaluationReport(ImmutableModel):
    """Typed final evidence produced by a trusted sealed evaluator."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    sealed_running_tail_ref: ArtifactRef
    sealed_authorization_ref: ArtifactRef
    selection_closure_ref: ArtifactRef
    champion_candidate_ref: ArtifactRef | None
    champion_harness_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    sealed_split_ref: ArtifactRef
    usage_tail_ref: ArtifactRef | None
    model_fingerprint: NonEmptyStr
    inference_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    sandbox_fingerprint: NonEmptyStr
    grader_fingerprint: NonEmptyStr
    capability_policy_ref: ArtifactRef
    result_ref: ArtifactRef
    evidence_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    completed: Literal[True] = True


class ExperimentCompletionReport(ImmutableModel):
    """Bind an external sealed final report to the exact running branch."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    sealed_running_tail_ref: ArtifactRef
    sealed_authorization_ref: ArtifactRef
    sealed_evaluation_report_ref: ArtifactRef
    usage_tail_ref: ArtifactRef | None


class ExperimentInvalidationReport(ImmutableModel):
    """Controller-authored proof that permanently invalidates a lineage."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    source_tail_ref: ArtifactRef
    source_state: ExperimentState
    violation_code: ExperimentViolationCode
    evidence_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    usage_tail_ref: ArtifactRef | None
    message: NonEmptyStr


def replay_experiment_lifecycle(
    events: Iterable[ExperimentLifecycleEvent],
) -> ExperimentState:
    """Validate one ordered event chain and return its projected state."""

    validated = tuple(ExperimentLifecycleEvent.model_validate(event) for event in events)
    if not validated:
        raise ExperimentLifecycleError("experiment lifecycle requires at least one event")
    experiment_ref = validated[0].experiment_ref
    state: ExperimentState | None = None
    for index, event in enumerate(validated):
        if event.experiment_ref != experiment_ref:
            raise ExperimentLifecycleError(
                f"experiment event {index} belongs to another experiment"
            )
        if state in TERMINAL_EXPERIMENT_STATES:
            raise ExperimentLifecycleError(f"event {index} follows terminal state {state.name}")
        if event.from_state is not state:
            raise ExperimentLifecycleError(f"experiment event {index} is disconnected")
        state = event.to_state
    if state is None:  # pragma: no cover - guarded by non-empty validation
        raise ExperimentLifecycleError("experiment lifecycle produced no state")
    return state


class ExperimentJournal:
    """Structural content-addressed journal for experiment lifecycle events."""

    def __init__(self, repository: ArtifactRepository) -> None:
        self._repository = repository

    def append(
        self,
        *,
        experiment_ref: ArtifactRef,
        event: ExperimentLifecycleEvent,
        previous_entry_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Append one structurally valid event and return its immutable head."""

        event = ExperimentLifecycleEvent.model_validate(event)
        if event.experiment_ref != experiment_ref:
            raise ExperimentLifecycleError("event belongs to another experiment")
        if previous_entry_ref is None:
            events: tuple[ExperimentLifecycleEvent, ...] = ()
            sequence = 0
        else:
            events = self.replay(previous_entry_ref)
            sequence = len(events)
        replay_experiment_lifecycle((*events, event))
        event_ref = self._repository.put_json(
            event,
            media_type=EXPERIMENT_LIFECYCLE_EVENT_MEDIA_TYPE,
        )
        entry = ExperimentJournalEntry(
            experiment_ref=experiment_ref,
            sequence=sequence,
            event_ref=event_ref,
            previous_entry_ref=previous_entry_ref,
        )
        return self._repository.put_json(
            entry,
            media_type=EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE,
        )

    def replay(self, tail_ref: ArtifactRef) -> tuple[ExperimentLifecycleEvent, ...]:
        """Verify links and lifecycle transitions from root through ``tail_ref``."""

        cursor: ArtifactRef | None = ArtifactRef.model_validate(tail_ref)
        backwards: list[tuple[ArtifactRef, ExperimentJournalEntry]] = []
        seen: set[str] = set()
        while cursor is not None:
            if cursor.media_type != EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE:
                raise ExperimentLifecycleError("experiment tail declares the wrong media type")
            if cursor.sha256 in seen:
                raise ExperimentLifecycleError("experiment journal contains a cycle")
            seen.add(cursor.sha256)
            entry = self._repository.get_json(cursor, ExperimentJournalEntry)
            backwards.append((cursor, entry))
            cursor = entry.previous_entry_ref
        entries = tuple(reversed(backwards))
        events: list[ExperimentLifecycleEvent] = []
        previous_ref: ArtifactRef | None = None
        experiment_ref = entries[0][1].experiment_ref
        for sequence, (entry_ref, entry) in enumerate(entries):
            if entry.sequence != sequence:
                raise ExperimentLifecycleError("experiment journal sequence is not contiguous")
            if entry.previous_entry_ref != previous_ref:
                raise ExperimentLifecycleError("experiment journal link is inconsistent")
            if entry.experiment_ref != experiment_ref:
                raise ExperimentLifecycleError("experiment journal changed experiment_ref")
            event = self._repository.get_json(entry.event_ref, ExperimentLifecycleEvent)
            events.append(event)
            previous_ref = entry_ref
        replay_experiment_lifecycle(events)
        return tuple(events)


__all__ = [
    "EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE",
    "EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE",
    "EXPERIMENT_JOURNAL_ENTRY_MEDIA_TYPE",
    "EXPERIMENT_LIFECYCLE_EVENT_MEDIA_TYPE",
    "SEALED_EVALUATION_REPORT_MEDIA_TYPE",
    "SEALED_RUN_AUTHORIZATION_MEDIA_TYPE",
    "SELECTION_CLOSURE_MEDIA_TYPE",
    "TERMINAL_EXPERIMENT_STATES",
    "ExperimentCompletionReport",
    "ExperimentInvalidationReport",
    "ExperimentJournal",
    "ExperimentJournalEntry",
    "ExperimentLifecycleError",
    "ExperimentLifecycleEvent",
    "ExperimentState",
    "ExperimentViolationCode",
    "SealedEvaluationReport",
    "SealedRunAuthorization",
    "SelectionClosure",
    "SelectionReason",
    "replay_experiment_lifecycle",
]
