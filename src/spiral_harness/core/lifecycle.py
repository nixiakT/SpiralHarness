"""Strict candidate lifecycle events and their deterministic projection."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr


class CandidateState(StrEnum):
    """Persisted states for an immutable candidate manifest."""

    REGISTERED = "registered"
    VALID = "valid"
    INVALID = "invalid"
    RUNNING_PROBES = "running_probes"
    RUNNING_GATE = "running_gate"
    EVIDENCE_COMPLETE = "evidence_complete"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


TERMINAL_CANDIDATE_STATES = frozenset(
    {
        CandidateState.INVALID,
        CandidateState.PROMOTED,
        CandidateState.REJECTED,
        CandidateState.INCONCLUSIVE,
    }
)

_EVIDENCE_REQUIRED_TARGETS = frozenset(
    {
        CandidateState.VALID,
        CandidateState.INVALID,
        CandidateState.RUNNING_GATE,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.PROMOTED,
        CandidateState.REJECTED,
        CandidateState.INCONCLUSIVE,
    }
)

_ALLOWED_TRANSITIONS = MappingProxyType(
    {
        CandidateState.REGISTERED: frozenset({CandidateState.VALID, CandidateState.INVALID}),
        CandidateState.VALID: frozenset({CandidateState.RUNNING_PROBES}),
        CandidateState.RUNNING_PROBES: frozenset(
            {CandidateState.RUNNING_GATE, CandidateState.REJECTED}
        ),
        CandidateState.RUNNING_GATE: frozenset({CandidateState.EVIDENCE_COMPLETE}),
        CandidateState.EVIDENCE_COMPLETE: frozenset(
            {
                CandidateState.PROMOTED,
                CandidateState.REJECTED,
                CandidateState.INCONCLUSIVE,
            }
        ),
    }
)


class CandidateLifecycleError(ValueError):
    """Raised when an event sequence cannot be a candidate lifecycle."""


class CandidateLifecycleEvent(ImmutableModel):
    """One immutable transition in a candidate's persisted lifecycle.

    ``from_state=None`` is reserved for the initial registration event.  The
    event deliberately contains no wall-clock timestamp: ordering comes from
    the append-only journal, while evidence timestamps belong to referenced
    artifacts whose clocks can be audited independently.
    """

    candidate_ref: ArtifactRef
    from_state: CandidateState | None
    to_state: CandidateState
    evidence_refs: tuple[ArtifactRef, ...] = ()
    reason: NonEmptyStr | None = None

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_evidence_refs(
        cls,
        refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        """Give set-like evidence a stable identity and reject duplicates."""

        ordered = tuple(sorted(refs, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("evidence_refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def transition_is_legal(self) -> CandidateLifecycleEvent:
        if self.from_state is None:
            if self.to_state is not CandidateState.REGISTERED:
                raise ValueError("only REGISTERED may be emitted without a prior state")
            return self

        allowed = _ALLOWED_TRANSITIONS.get(self.from_state, frozenset())
        if self.to_state not in allowed:
            if self.from_state in TERMINAL_CANDIDATE_STATES:
                raise ValueError(f"terminal state {self.from_state.name} cannot transition further")
            raise ValueError(
                f"illegal candidate transition: {self.from_state.name} -> {self.to_state.name}"
            )
        if self.to_state in _EVIDENCE_REQUIRED_TARGETS and not self.evidence_refs:
            raise ValueError(f"transition to {self.to_state.name} requires evidence_refs")
        return self

    @property
    def state(self) -> CandidateState:
        """The state produced by this transition."""

        return self.to_state


class CandidateLifecycle(ImmutableModel):
    """Validated projection of one candidate's ordered transition events."""

    candidate_ref: ArtifactRef
    state: CandidateState
    event_count: Annotated[int, Field(ge=1, strict=True)]

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is legal for this projection."""

        return self.state in TERMINAL_CANDIDATE_STATES


def allowed_candidate_transitions(state: CandidateState) -> frozenset[CandidateState]:
    """Return the complete legal successor set for ``state``."""

    return _ALLOWED_TRANSITIONS.get(state, frozenset())


def replay_candidate_lifecycle(
    events: Iterable[CandidateLifecycleEvent],
) -> CandidateLifecycle:
    """Validate ordered events and return their deterministic state projection.

    Events are revalidated so instances created with Pydantic's unsafe
    construction helpers cannot cross this trusted boundary.  Besides each
    event's local transition, replay binds the whole sequence to one exact
    ``candidate_ref`` and verifies that adjacent transitions join.
    """

    validated = tuple(CandidateLifecycleEvent.model_validate(event) for event in events)
    if not validated:
        raise CandidateLifecycleError("a candidate lifecycle requires at least one event")

    candidate_ref = validated[0].candidate_ref
    current_state: CandidateState | None = None
    for index, event in enumerate(validated):
        if event.candidate_ref != candidate_ref:
            raise CandidateLifecycleError(
                f"event {index} candidate_ref does not match the lifecycle candidate"
            )
        if current_state in TERMINAL_CANDIDATE_STATES:
            raise CandidateLifecycleError(
                f"event {index} follows terminal state {current_state.name}"
            )
        if event.from_state != current_state:
            expected = "none" if current_state is None else current_state.name
            actual = "none" if event.from_state is None else event.from_state.name
            raise CandidateLifecycleError(
                f"event {index} starts from {actual}; expected {expected}"
            )
        current_state = event.to_state

    if current_state is None:  # pragma: no cover - guarded by the non-empty check
        raise CandidateLifecycleError("candidate lifecycle did not produce a state")
    return CandidateLifecycle(
        candidate_ref=candidate_ref,
        state=current_state,
        event_count=len(validated),
    )


__all__ = [
    "TERMINAL_CANDIDATE_STATES",
    "CandidateLifecycle",
    "CandidateLifecycleError",
    "CandidateLifecycleEvent",
    "CandidateState",
    "allowed_candidate_transitions",
    "replay_candidate_lifecycle",
]
