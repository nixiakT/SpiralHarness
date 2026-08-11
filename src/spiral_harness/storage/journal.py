"""Content-addressed linked journal for candidate lifecycle events."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from spiral_harness.core.lifecycle import (
    CandidateLifecycleEvent,
    replay_candidate_lifecycle,
)
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr
from spiral_harness.storage.protocol import ArtifactRepository

JOURNAL_ENTRY_MEDIA_TYPE = "application/vnd.spiral-harness.journal-entry.v1+json"
LIFECYCLE_EVENT_MEDIA_TYPE = "application/vnd.spiral-harness.lifecycle-event.v1+json"


class JournalIntegrityError(RuntimeError):
    """Raised when linked journal structure is inconsistent."""


class JournalCycleError(JournalIntegrityError):
    """Raised when a journal link revisits an entry reference."""


class JournalEntry(ImmutableModel):
    """One immutable link from an event to the preceding journal entry."""

    stream_id: NonEmptyStr
    sequence: Annotated[int, Field(ge=0, strict=True)]
    event_ref: ArtifactRef
    previous_entry_ref: ArtifactRef | None

    @model_validator(mode="after")
    def root_and_link_shape_matches_sequence(self) -> JournalEntry:
        if self.event_ref.media_type != LIFECYCLE_EVENT_MEDIA_TYPE:
            raise ValueError("event_ref must declare the lifecycle event media type")
        if (
            self.previous_entry_ref is not None
            and self.previous_entry_ref.media_type != JOURNAL_ENTRY_MEDIA_TYPE
        ):
            raise ValueError("previous_entry_ref must declare the journal entry media type")
        if self.sequence == 0 and self.previous_entry_ref is not None:
            raise ValueError("journal sequence 0 must not have a previous_entry_ref")
        if self.sequence > 0 and self.previous_entry_ref is None:
            raise ValueError("journal sequence greater than 0 requires a previous_entry_ref")
        return self


class CandidateJournal:
    """Append and replay candidate events without a mutable head pointer.

    Callers hold a returned entry reference and pass it explicitly as the next
    append's ``previous_entry_ref``.  Consequently, an append never overwrites
    old state and branching from an older tail remains explicit and replayable.

    This is deliberately a structural ledger, not an authorization service.
    A trusted experiment controller must validate admission reports, mechanism
    evidence, gate decisions, and budget state before appending the
    corresponding semantic transition.
    """

    def __init__(self, store: ArtifactRepository) -> None:
        self.store = store

    def append(
        self,
        *,
        stream_id: str,
        event: CandidateLifecycleEvent,
        previous_entry_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Validate and append ``event``, returning the new immutable tail."""

        validated_event = CandidateLifecycleEvent.model_validate(event)
        normalized_stream_id = self._normalize_stream_id(stream_id)

        if previous_entry_ref is None:
            previous_entries: tuple[tuple[ArtifactRef, JournalEntry], ...] = ()
            previous_events: tuple[CandidateLifecycleEvent, ...] = ()
            sequence = 0
        else:
            validated_previous_ref = ArtifactRef.model_validate(previous_entry_ref)
            previous_entries = self._read_entries(validated_previous_ref)
            if previous_entries[-1][1].stream_id != normalized_stream_id:
                raise JournalIntegrityError(
                    "append stream_id does not match the previous journal entry"
                )
            previous_events = self._read_events(previous_entries)
            replay_candidate_lifecycle(previous_events)
            sequence = previous_entries[-1][1].sequence + 1
            previous_entry_ref = validated_previous_ref

        # Validate the candidate, transition join, and terminal-state rule before
        # publishing either of the two new content-addressed objects.
        replay_candidate_lifecycle((*previous_events, validated_event))

        event_ref = self.store.put_json(
            validated_event,
            media_type=LIFECYCLE_EVENT_MEDIA_TYPE,
        )
        entry = JournalEntry(
            stream_id=normalized_stream_id,
            sequence=sequence,
            event_ref=event_ref,
            previous_entry_ref=previous_entry_ref,
        )
        return self.store.put_json(entry, media_type=JOURNAL_ENTRY_MEDIA_TYPE)

    def replay(self, tail_ref: ArtifactRef) -> tuple[CandidateLifecycleEvent, ...]:
        """Verify the entire chain and return events in chronological order."""

        entries = self._read_entries(ArtifactRef.model_validate(tail_ref))
        events = self._read_events(entries)
        replay_candidate_lifecycle(events)
        return events

    @staticmethod
    def _normalize_stream_id(stream_id: str) -> str:
        if not isinstance(stream_id, str):
            raise TypeError("stream_id must be a string")
        normalized = stream_id.strip()
        if not normalized:
            raise ValueError("stream_id must not be empty")
        return normalized

    def _read_entries(
        self,
        tail_ref: ArtifactRef,
    ) -> tuple[tuple[ArtifactRef, JournalEntry], ...]:
        """Walk and structurally validate a tail-to-root linked chain."""

        backwards: list[tuple[ArtifactRef, JournalEntry]] = []
        seen_digests: set[str] = set()
        cursor: ArtifactRef | None = tail_ref

        while cursor is not None:
            if cursor.media_type != JOURNAL_ENTRY_MEDIA_TYPE:
                raise JournalIntegrityError("journal entry ref declares the wrong media type")
            if cursor.sha256 in seen_digests:
                raise JournalCycleError(f"journal cycle detected at entry {cursor.sha256}")
            seen_digests.add(cursor.sha256)
            loaded = self.store.get_json(cursor, JournalEntry)
            entry = JournalEntry.model_validate(loaded)
            backwards.append((cursor, entry))
            cursor = entry.previous_entry_ref

        entries = tuple(reversed(backwards))
        if not entries:  # pragma: no cover - tail_ref always produces one read attempt
            raise JournalIntegrityError("journal chain is empty")

        expected_stream_id = entries[0][1].stream_id
        for expected_sequence, (_, entry) in enumerate(entries):
            if entry.sequence != expected_sequence:
                raise JournalIntegrityError(
                    "journal sequence is not contiguous from zero: "
                    f"expected {expected_sequence}, got {entry.sequence}"
                )
            if entry.stream_id != expected_stream_id:
                raise JournalIntegrityError(f"journal stream changed at sequence {entry.sequence}")
            expected_previous = (
                None if expected_sequence == 0 else entries[expected_sequence - 1][0]
            )
            if entry.previous_entry_ref != expected_previous:
                raise JournalIntegrityError(
                    f"journal link is inconsistent at sequence {entry.sequence}"
                )

        return entries

    def _read_events(
        self,
        entries: tuple[tuple[ArtifactRef, JournalEntry], ...],
    ) -> tuple[CandidateLifecycleEvent, ...]:
        events: list[CandidateLifecycleEvent] = []
        for _, entry in entries:
            loaded = self.store.get_json(entry.event_ref, CandidateLifecycleEvent)
            events.append(CandidateLifecycleEvent.model_validate(loaded))
        return tuple(events)


__all__ = [
    "JOURNAL_ENTRY_MEDIA_TYPE",
    "LIFECYCLE_EVENT_MEDIA_TYPE",
    "CandidateJournal",
    "JournalCycleError",
    "JournalEntry",
    "JournalIntegrityError",
]
