from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from spiral_harness.core.lifecycle import (
    TERMINAL_CANDIDATE_STATES,
    CandidateLifecycleError,
    CandidateLifecycleEvent,
    CandidateState,
    allowed_candidate_transitions,
    replay_candidate_lifecycle,
)
from spiral_harness.core.models import ArtifactRef


def artifact(digit: str, *, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def lifecycle_events(
    candidate_ref: ArtifactRef,
    states: Iterable[CandidateState],
) -> tuple[CandidateLifecycleEvent, ...]:
    events: list[CandidateLifecycleEvent] = []
    previous: CandidateState | None = None
    for state in states:
        evidence_refs = (
            (artifact("e"),)
            if state
            in {
                CandidateState.VALID,
                CandidateState.INVALID,
                CandidateState.RUNNING_GATE,
                CandidateState.EVIDENCE_COMPLETE,
                CandidateState.PROMOTED,
                CandidateState.REJECTED,
                CandidateState.INCONCLUSIVE,
            }
            else ()
        )
        events.append(
            CandidateLifecycleEvent(
                candidate_ref=candidate_ref,
                from_state=previous,
                to_state=state,
                evidence_refs=evidence_refs,
            )
        )
        previous = state
    return tuple(events)


def test_transition_schema_accepts_exactly_the_declared_graph() -> None:
    expected = {
        None: {CandidateState.REGISTERED},
        CandidateState.REGISTERED: {CandidateState.VALID, CandidateState.INVALID},
        CandidateState.VALID: {CandidateState.RUNNING_PROBES},
        CandidateState.INVALID: set(),
        CandidateState.RUNNING_PROBES: {
            CandidateState.RUNNING_GATE,
            CandidateState.REJECTED,
        },
        CandidateState.RUNNING_GATE: {CandidateState.EVIDENCE_COMPLETE},
        CandidateState.EVIDENCE_COMPLETE: {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        },
        CandidateState.PROMOTED: set(),
        CandidateState.REJECTED: set(),
        CandidateState.INCONCLUSIVE: set(),
    }

    for source, allowed_targets in expected.items():
        for target in CandidateState:
            values = {
                "candidate_ref": artifact("a"),
                "from_state": source,
                "to_state": target,
                "evidence_refs": (artifact("e"),),
            }
            if target in allowed_targets:
                assert CandidateLifecycleEvent(**values).to_state is target
            else:
                with pytest.raises(ValidationError):
                    CandidateLifecycleEvent(**values)

    for state in CandidateState:
        assert allowed_candidate_transitions(state) == frozenset(expected[state])
    assert {
        CandidateState.INVALID,
        CandidateState.PROMOTED,
        CandidateState.REJECTED,
        CandidateState.INCONCLUSIVE,
    } == TERMINAL_CANDIDATE_STATES


@pytest.mark.parametrize(
    "states",
    [
        (CandidateState.REGISTERED, CandidateState.INVALID),
        (
            CandidateState.REGISTERED,
            CandidateState.VALID,
            CandidateState.RUNNING_PROBES,
            CandidateState.REJECTED,
        ),
        (
            CandidateState.REGISTERED,
            CandidateState.VALID,
            CandidateState.RUNNING_PROBES,
            CandidateState.RUNNING_GATE,
            CandidateState.EVIDENCE_COMPLETE,
            CandidateState.PROMOTED,
        ),
        (
            CandidateState.REGISTERED,
            CandidateState.VALID,
            CandidateState.RUNNING_PROBES,
            CandidateState.RUNNING_GATE,
            CandidateState.EVIDENCE_COMPLETE,
            CandidateState.REJECTED,
        ),
        (
            CandidateState.REGISTERED,
            CandidateState.VALID,
            CandidateState.RUNNING_PROBES,
            CandidateState.RUNNING_GATE,
            CandidateState.EVIDENCE_COMPLETE,
            CandidateState.INCONCLUSIVE,
        ),
    ],
)
def test_replay_accepts_each_terminal_path(states: tuple[CandidateState, ...]) -> None:
    candidate_ref = artifact("a")
    events = lifecycle_events(candidate_ref, states)

    lifecycle = replay_candidate_lifecycle(events)

    assert lifecycle.candidate_ref == candidate_ref
    assert lifecycle.state is states[-1]
    assert lifecycle.event_count == len(states)
    assert lifecycle.is_terminal


def test_nonterminal_projection_reports_that_more_work_is_legal() -> None:
    lifecycle = replay_candidate_lifecycle(
        lifecycle_events(
            artifact("a"),
            (CandidateState.REGISTERED, CandidateState.VALID),
        )
    )

    assert lifecycle.state is CandidateState.VALID
    assert not lifecycle.is_terminal


def test_replay_rejects_empty_disconnected_and_non_registration_sequences() -> None:
    candidate_ref = artifact("a")

    with pytest.raises(CandidateLifecycleError, match="at least one"):
        replay_candidate_lifecycle(())

    non_registration = CandidateLifecycleEvent(
        candidate_ref=candidate_ref,
        from_state=CandidateState.VALID,
        to_state=CandidateState.RUNNING_PROBES,
    )
    with pytest.raises(CandidateLifecycleError, match="expected none"):
        replay_candidate_lifecycle((non_registration,))

    registered = lifecycle_events(candidate_ref, (CandidateState.REGISTERED,))[0]
    disconnected = CandidateLifecycleEvent(
        candidate_ref=candidate_ref,
        from_state=CandidateState.VALID,
        to_state=CandidateState.RUNNING_PROBES,
    )
    with pytest.raises(CandidateLifecycleError, match="expected REGISTERED"):
        replay_candidate_lifecycle((registered, disconnected))


def test_replay_binds_every_event_to_one_exact_candidate_ref() -> None:
    registered = lifecycle_events(artifact("a"), (CandidateState.REGISTERED,))[0]
    valid_other_candidate = CandidateLifecycleEvent(
        candidate_ref=artifact("b"),
        from_state=CandidateState.REGISTERED,
        to_state=CandidateState.VALID,
        evidence_refs=(artifact("e"),),
    )

    with pytest.raises(CandidateLifecycleError, match="candidate_ref"):
        replay_candidate_lifecycle((registered, valid_other_candidate))


def test_replay_explicitly_rejects_an_event_after_a_terminal_state() -> None:
    candidate_ref = artifact("a")
    terminal_path = lifecycle_events(
        candidate_ref,
        (CandidateState.REGISTERED, CandidateState.INVALID),
    )
    second_registration = lifecycle_events(
        candidate_ref,
        (CandidateState.REGISTERED,),
    )[0]

    with pytest.raises(CandidateLifecycleError, match="follows terminal state INVALID"):
        replay_candidate_lifecycle((*terminal_path, second_registration))


def test_event_is_frozen_extra_forbid_and_canonicalizes_evidence() -> None:
    event = CandidateLifecycleEvent(
        candidate_ref=artifact("a"),
        from_state=None,
        to_state=CandidateState.REGISTERED,
        evidence_refs=(artifact("e"), artifact("d")),
        reason="  manifest frozen  ",
    )

    assert [ref.sha256 for ref in event.evidence_refs] == ["d" * 64, "e" * 64]
    assert event.reason == "manifest frozen"
    assert event.state is CandidateState.REGISTERED
    with pytest.raises(ValidationError):
        event.to_state = CandidateState.VALID
    with pytest.raises(ValidationError):
        CandidateLifecycleEvent(
            candidate_ref=artifact("a"),
            from_state=None,
            to_state=CandidateState.REGISTERED,
            unexpected=True,
        )
    with pytest.raises(ValidationError, match="duplicate"):
        CandidateLifecycleEvent(
            candidate_ref=artifact("a"),
            from_state=None,
            to_state=CandidateState.REGISTERED,
            evidence_refs=(artifact("e"), artifact("e")),
        )


@pytest.mark.parametrize(
    "target",
    [
        CandidateState.VALID,
        CandidateState.INVALID,
        CandidateState.RUNNING_GATE,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.PROMOTED,
        CandidateState.REJECTED,
        CandidateState.INCONCLUSIVE,
    ],
)
def test_evidence_bearing_transitions_fail_closed_without_refs(
    target: CandidateState,
) -> None:
    source = {
        CandidateState.VALID: CandidateState.REGISTERED,
        CandidateState.INVALID: CandidateState.REGISTERED,
        CandidateState.RUNNING_GATE: CandidateState.RUNNING_PROBES,
        CandidateState.EVIDENCE_COMPLETE: CandidateState.RUNNING_GATE,
        CandidateState.PROMOTED: CandidateState.EVIDENCE_COMPLETE,
        CandidateState.REJECTED: CandidateState.EVIDENCE_COMPLETE,
        CandidateState.INCONCLUSIVE: CandidateState.EVIDENCE_COMPLETE,
    }[target]

    with pytest.raises(ValidationError, match="requires evidence_refs"):
        CandidateLifecycleEvent(
            candidate_ref=artifact("a"),
            from_state=source,
            to_state=target,
        )


def test_replay_revalidates_an_unsafely_copied_event() -> None:
    registered = lifecycle_events(artifact("a"), (CandidateState.REGISTERED,))[0]
    bypassed = registered.model_copy(update={"to_state": CandidateState.PROMOTED})

    with pytest.raises(ValidationError):
        replay_candidate_lifecycle((bypassed,))
