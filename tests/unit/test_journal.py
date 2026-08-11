from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.lifecycle import (
    CandidateLifecycleError,
    CandidateLifecycleEvent,
    CandidateState,
)
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.artifact_store import ArtifactNotFoundError, ArtifactStore
from spiral_harness.storage.journal import (
    JOURNAL_ENTRY_MEDIA_TYPE,
    LIFECYCLE_EVENT_MEDIA_TYPE,
    CandidateJournal,
    JournalCycleError,
    JournalEntry,
    JournalIntegrityError,
)


def artifact(digit: str, *, size: int = 1, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=size, media_type=media_type)


def event(
    candidate_ref: ArtifactRef,
    source: CandidateState | None,
    target: CandidateState,
) -> CandidateLifecycleEvent:
    evidence_refs = (
        (artifact("e"),)
        if target
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
    return CandidateLifecycleEvent(
        candidate_ref=candidate_ref,
        from_state=source,
        to_state=target,
        evidence_refs=evidence_refs,
    )


def store_entry(store: ArtifactStore, entry: JournalEntry) -> ArtifactRef:
    return store.put_json(entry, media_type=JOURNAL_ENTRY_MEDIA_TYPE)


def test_append_persists_exact_links_and_replays_events_in_forward_order(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    candidate_ref = store.put_json({"candidate": "immutable"})
    events = (
        event(candidate_ref, None, CandidateState.REGISTERED),
        event(candidate_ref, CandidateState.REGISTERED, CandidateState.VALID),
        event(candidate_ref, CandidateState.VALID, CandidateState.RUNNING_PROBES),
    )

    first_ref = journal.append(stream_id=" candidate/a ", event=events[0])
    second_ref = journal.append(
        stream_id="candidate/a",
        event=events[1],
        previous_entry_ref=first_ref,
    )
    tail_ref = journal.append(
        stream_id="candidate/a",
        event=events[2],
        previous_entry_ref=second_ref,
    )

    first = store.get_json(first_ref, JournalEntry)
    second = store.get_json(second_ref, JournalEntry)
    tail = store.get_json(tail_ref, JournalEntry)
    assert (first.sequence, second.sequence, tail.sequence) == (0, 1, 2)
    assert first.previous_entry_ref is None
    assert second.previous_entry_ref == first_ref
    assert tail.previous_entry_ref == second_ref
    assert {first.stream_id, second.stream_id, tail.stream_id} == {"candidate/a"}
    assert first_ref.media_type == JOURNAL_ENTRY_MEDIA_TYPE
    assert first.event_ref.media_type == LIFECYCLE_EVENT_MEDIA_TYPE
    assert journal.replay(tail_ref) == events
    assert set(store.get_json(first_ref)) == {
        "event_ref",
        "previous_entry_ref",
        "sequence",
        "stream_id",
    }


def test_append_is_idempotent_and_caller_held_tails_can_branch(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    candidate_ref = store.put_json({"candidate": "branchable"})
    registered = event(candidate_ref, None, CandidateState.REGISTERED)
    valid = event(candidate_ref, CandidateState.REGISTERED, CandidateState.VALID)
    probes = event(candidate_ref, CandidateState.VALID, CandidateState.RUNNING_PROBES)
    gate = event(candidate_ref, CandidateState.RUNNING_PROBES, CandidateState.RUNNING_GATE)
    complete = event(
        candidate_ref,
        CandidateState.RUNNING_GATE,
        CandidateState.EVIDENCE_COMPLETE,
    )

    tail: ArtifactRef | None = None
    for item in (registered, valid, probes, gate, complete):
        tail = journal.append(
            stream_id="candidate/branch",
            event=item,
            previous_entry_ref=tail,
        )
    assert tail is not None

    promoted = event(
        candidate_ref,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.PROMOTED,
    )
    rejected = event(
        candidate_ref,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.REJECTED,
    )
    promoted_tail = journal.append(
        stream_id="candidate/branch",
        event=promoted,
        previous_entry_ref=tail,
    )
    repeated_tail = journal.append(
        stream_id="candidate/branch",
        event=promoted,
        previous_entry_ref=tail,
    )
    rejected_tail = journal.append(
        stream_id="candidate/branch",
        event=rejected,
        previous_entry_ref=tail,
    )

    assert promoted_tail == repeated_tail
    assert promoted_tail != rejected_tail
    assert journal.replay(tail)[-1] == complete
    assert journal.replay(promoted_tail)[-1] == promoted
    assert journal.replay(rejected_tail)[-1] == rejected
    # The journal writes only immutable objects; it never publishes an
    # overwriteable stream head beside the ArtifactStore.
    assert set(store.root.iterdir()) == {store.objects_dir}


def test_journal_entry_is_frozen_extra_forbid_and_has_strict_link_shape() -> None:
    event_ref = artifact("e", media_type=LIFECYCLE_EVENT_MEDIA_TYPE)
    previous_ref = artifact("a", media_type=JOURNAL_ENTRY_MEDIA_TYPE)
    root = JournalEntry(
        stream_id="candidate/a",
        sequence=0,
        event_ref=event_ref,
        previous_entry_ref=None,
    )

    with pytest.raises(ValidationError):
        root.sequence = 1
    with pytest.raises(ValidationError):
        JournalEntry(
            stream_id="candidate/a",
            sequence=0,
            event_ref=event_ref,
            previous_entry_ref=None,
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        JournalEntry(
            stream_id="candidate/a",
            sequence=True,
            event_ref=event_ref,
            previous_entry_ref=previous_ref,
        )
    with pytest.raises(ValidationError, match="sequence 0"):
        JournalEntry(
            stream_id="candidate/a",
            sequence=0,
            event_ref=event_ref,
            previous_entry_ref=previous_ref,
        )
    with pytest.raises(ValidationError, match="requires"):
        JournalEntry(
            stream_id="candidate/a",
            sequence=1,
            event_ref=event_ref,
            previous_entry_ref=None,
        )
    with pytest.raises(ValidationError, match="lifecycle event media type"):
        JournalEntry(
            stream_id="candidate/a",
            sequence=0,
            event_ref=artifact("e"),
            previous_entry_ref=None,
        )
    with pytest.raises(ValidationError, match="journal entry media type"):
        JournalEntry(
            stream_id="candidate/a",
            sequence=1,
            event_ref=event_ref,
            previous_entry_ref=artifact("a"),
        )


def test_append_rejects_stream_candidate_join_and_terminal_violations_before_writing(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    candidate_ref = store.put_json({"candidate": "a"})
    other_candidate_ref = store.put_json({"candidate": "b"})
    root = journal.append(
        stream_id="candidate/a",
        event=event(candidate_ref, None, CandidateState.REGISTERED),
    )
    object_count = len([path for path in store.objects_dir.rglob("*") if path.is_file()])

    with pytest.raises(JournalIntegrityError, match="stream_id"):
        journal.append(
            stream_id="candidate/b",
            event=event(candidate_ref, CandidateState.REGISTERED, CandidateState.VALID),
            previous_entry_ref=root,
        )
    with pytest.raises(CandidateLifecycleError, match="candidate_ref"):
        journal.append(
            stream_id="candidate/a",
            event=event(
                other_candidate_ref,
                CandidateState.REGISTERED,
                CandidateState.VALID,
            ),
            previous_entry_ref=root,
        )
    with pytest.raises(CandidateLifecycleError, match="expected REGISTERED"):
        journal.append(
            stream_id="candidate/a",
            event=event(
                candidate_ref,
                CandidateState.VALID,
                CandidateState.RUNNING_PROBES,
            ),
            previous_entry_ref=root,
        )

    invalid_tail = journal.append(
        stream_id="candidate/a",
        event=event(candidate_ref, CandidateState.REGISTERED, CandidateState.INVALID),
        previous_entry_ref=root,
    )
    with pytest.raises(CandidateLifecycleError, match="follows terminal state"):
        journal.append(
            stream_id="candidate/a",
            event=event(candidate_ref, None, CandidateState.REGISTERED),
            previous_entry_ref=invalid_tail,
        )

    # Only the valid INVALID transition added its event and entry.
    assert len([path for path in store.objects_dir.rglob("*") if path.is_file()]) == (
        object_count + 2
    )


@pytest.mark.parametrize("stream_id", ["", "   "])
def test_append_rejects_empty_stream_without_persisting_event(tmp_path, stream_id: str) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)

    with pytest.raises(ValueError, match="stream_id"):
        journal.append(
            stream_id=stream_id,
            event=event(artifact("a"), None, CandidateState.REGISTERED),
        )

    assert not [path for path in store.objects_dir.rglob("*") if path.is_file()]


def test_append_rejects_a_non_string_stream_id(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)

    with pytest.raises(TypeError, match="stream_id"):
        journal.append(
            stream_id=1,  # type: ignore[arg-type]
            event=event(artifact("a"), None, CandidateState.REGISTERED),
        )

    assert not [path for path in store.objects_dir.rglob("*") if path.is_file()]


def test_replay_rejects_sequence_gap_and_stream_change(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    candidate_ref = store.put_json({"candidate": "a"})
    registered_ref = store.put_json(
        event(candidate_ref, None, CandidateState.REGISTERED),
        media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
    )
    valid_ref = store.put_json(
        event(candidate_ref, CandidateState.REGISTERED, CandidateState.VALID),
        media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
    )
    root_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/a",
            sequence=0,
            event_ref=registered_ref,
            previous_entry_ref=None,
        ),
    )
    gap_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/a",
            sequence=2,
            event_ref=valid_ref,
            previous_entry_ref=root_ref,
        ),
    )
    changed_stream_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/b",
            sequence=1,
            event_ref=valid_ref,
            previous_entry_ref=root_ref,
        ),
    )

    with pytest.raises(JournalIntegrityError, match="not contiguous"):
        journal.replay(gap_ref)
    with pytest.raises(JournalIntegrityError, match="stream changed"):
        journal.replay(changed_stream_ref)


def test_replay_rejects_missing_previous_entry(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    event_ref = store.put_json(
        event(artifact("a"), CandidateState.REGISTERED, CandidateState.VALID),
        media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
    )
    tail_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/a",
            sequence=1,
            event_ref=event_ref,
            previous_entry_ref=artifact("f", media_type=JOURNAL_ENTRY_MEDIA_TYPE),
        ),
    )

    with pytest.raises(ArtifactNotFoundError):
        journal.replay(tail_ref)


def test_replay_rejects_a_tail_with_the_wrong_declared_media_type(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    candidate_ref = store.put_json({"candidate": "a"})
    tail_ref = journal.append(
        stream_id="candidate/a",
        event=event(candidate_ref, None, CandidateState.REGISTERED),
    )
    forged_ref = ArtifactRef(
        sha256=tail_ref.sha256,
        size=tail_ref.size,
        media_type="application/json",
    )

    with pytest.raises(JournalIntegrityError, match="wrong media type"):
        journal.replay(forged_ref)


def test_replay_detects_a_cycle_before_interpreting_events(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    first_ref = artifact("a", media_type=JOURNAL_ENTRY_MEDIA_TYPE)
    second_ref = artifact("b", media_type=JOURNAL_ENTRY_MEDIA_TYPE)
    unused_event_ref = artifact("e", media_type=LIFECYCLE_EVENT_MEDIA_TYPE)
    entries = {
        first_ref.sha256: JournalEntry(
            stream_id="candidate/a",
            sequence=2,
            event_ref=unused_event_ref,
            previous_entry_ref=second_ref,
        ),
        second_ref.sha256: JournalEntry(
            stream_id="candidate/a",
            sequence=1,
            event_ref=unused_event_ref,
            previous_entry_ref=first_ref,
        ),
    }

    def fake_get_json(
        ref: ArtifactRef,
        model_type: type[JournalEntry] | None = None,
    ) -> JournalEntry:
        assert model_type is JournalEntry
        return entries[ref.sha256]

    monkeypatch.setattr(store, "get_json", fake_get_json)

    with pytest.raises(JournalCycleError, match="cycle"):
        journal.replay(first_ref)


def test_replay_revalidates_events_and_candidate_identity(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    first_candidate = artifact("a")
    second_candidate = artifact("b")
    registered_ref = store.put_json(
        event(first_candidate, None, CandidateState.REGISTERED),
        media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
    )
    wrong_candidate_ref = store.put_json(
        event(second_candidate, CandidateState.REGISTERED, CandidateState.VALID),
        media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
    )
    root_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/a",
            sequence=0,
            event_ref=registered_ref,
            previous_entry_ref=None,
        ),
    )
    tail_ref = store_entry(
        store,
        JournalEntry(
            stream_id="candidate/a",
            sequence=1,
            event_ref=wrong_candidate_ref,
            previous_entry_ref=root_ref,
        ),
    )

    with pytest.raises(CandidateLifecycleError, match="candidate_ref"):
        journal.replay(tail_ref)


def test_replay_rejects_an_entry_with_extra_fields(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    journal = CandidateJournal(store)
    malformed_ref = store.put_json(
        {
            "stream_id": "candidate/a",
            "sequence": 0,
            "event_ref": artifact("e", media_type=LIFECYCLE_EVENT_MEDIA_TYPE),
            "previous_entry_ref": None,
            "mutable_head": artifact("f"),
        },
        media_type=JOURNAL_ENTRY_MEDIA_TYPE,
    )

    with pytest.raises(ValidationError, match="extra"):
        journal.replay(malformed_ref)
