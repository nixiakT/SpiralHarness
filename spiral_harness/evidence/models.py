"""Typed, content-addressed contracts for trajectories and diagnostic evidence.

Large runtime payloads deliberately live in the artifact store.  The models in
this module only carry bounded labels, causal coordinates, and
:class:`~spiral_harness.core.models.ArtifactRef` values so they remain safe to
hash, compare, and place in an append-only ledger.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel

ShortText = Annotated[str, Field(min_length=1, max_length=256)]
ClaimText = Annotated[str, Field(min_length=1, max_length=1_024)]
SequenceNumber = Annotated[int, Field(ge=0, strict=True)]
ElapsedMilliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class TrajectoryEventKind(StrEnum):
    """Stable vocabulary for evidence-bearing runtime events."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    COMPONENT_ACTIVATED = "component_activated"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    SKILL_SELECTED = "skill_selected"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    MIDDLEWARE = "middleware"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"


class TrajectoryEvent(ImmutableModel):
    """One ordered event whose detailed payload is stored out of line."""

    schema_version: Literal["1"] = "1"
    sequence: SequenceNumber
    elapsed_ms: ElapsedMilliseconds
    kind: TrajectoryEventKind
    payload_ref: ArtifactRef
    component_ref: ArtifactRef | None = None
    causal_parent_sequences: tuple[SequenceNumber, ...] = ()

    @field_validator("causal_parent_sequences")
    @classmethod
    def _canonicalize_causal_parents(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError("causal_parent_sequences must not contain duplicates")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _parents_precede_event(self) -> Self:
        invalid = tuple(
            parent for parent in self.causal_parent_sequences if parent >= self.sequence
        )
        if invalid:
            raise ValueError(
                "causal parent sequences must be strictly earlier than the event sequence; "
                f"invalid: {invalid}"
            )
        if (
            self.kind
            in {
                TrajectoryEventKind.COMPONENT_ACTIVATED,
                TrajectoryEventKind.SKILL_SELECTED,
            }
            and self.component_ref is None
        ):
            raise ValueError(f"{self.kind.value} requires component_ref")
        return self


class Trajectory(ImmutableModel):
    """A complete, causally ordered record for one isolated harness run."""

    schema_version: Literal["1"] = "1"
    run_id: ShortText
    task_id: ShortText
    seed: SequenceNumber
    harness_ref: ArtifactRef
    execution_fingerprint: Annotated[str, Field(min_length=1, max_length=1_024)]
    events: Annotated[tuple[TrajectoryEvent, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def _validate_event_stream(self) -> Self:
        actual_sequences = tuple(event.sequence for event in self.events)
        expected_sequences = tuple(range(len(self.events)))
        if actual_sequences != expected_sequences:
            raise ValueError("trajectory event sequences must be continuous and start at zero")

        if self.events[0].kind is not TrajectoryEventKind.RUN_STARTED:
            raise ValueError("the first trajectory event must be run_started")
        if self.events[-1].kind not in {
            TrajectoryEventKind.RUN_FINISHED,
            TrajectoryEventKind.ERROR,
        }:
            raise ValueError("the last trajectory event must be run_finished or error")

        if any(event.kind is TrajectoryEventKind.RUN_STARTED for event in self.events[1:]):
            raise ValueError("run_started may only appear as the first trajectory event")
        terminal_kinds = {TrajectoryEventKind.RUN_FINISHED, TrajectoryEventKind.ERROR}
        if any(event.kind in terminal_kinds for event in self.events[:-1]):
            raise ValueError("run_finished and error may only appear as the terminal event")

        previous_elapsed = self.events[0].elapsed_ms
        for event in self.events[1:]:
            if event.elapsed_ms < previous_elapsed:
                raise ValueError("trajectory elapsed_ms values must be non-decreasing")
            previous_elapsed = event.elapsed_ms

        known_sequences = set(actual_sequences)
        for event in self.events:
            missing = tuple(
                parent for parent in event.causal_parent_sequences if parent not in known_sequences
            )
            if missing:
                raise ValueError(
                    f"event {event.sequence} refers to unknown causal parents: {missing}"
                )
        return self


class EvidenceSpanRef(ImmutableModel):
    """An inclusive event range supporting one bounded, falsifiable claim."""

    schema_version: Literal["1"] = "1"
    trajectory_ref: ArtifactRef
    start_sequence: SequenceNumber
    end_sequence: SequenceNumber
    claim: ClaimText
    expected_event_kinds: Annotated[
        tuple[TrajectoryEventKind, ...],
        Field(min_length=1),
    ]

    @field_validator("expected_event_kinds")
    @classmethod
    def _canonicalize_expected_kinds(
        cls,
        value: tuple[TrajectoryEventKind, ...],
    ) -> tuple[TrajectoryEventKind, ...]:
        if len(value) != len(set(value)):
            raise ValueError("expected_event_kinds must not contain duplicates")
        return tuple(sorted(value, key=lambda kind: kind.value))

    @model_validator(mode="after")
    def _range_is_ordered(self) -> Self:
        if self.end_sequence < self.start_sequence:
            raise ValueError("end_sequence must be greater than or equal to start_sequence")
        return self


class EvidencePacket(ImmutableModel):
    """A compact evidence bundle; substantive payloads remain artifact references."""

    schema_version: Literal["1"] = "1"
    source_spans: Annotated[tuple[EvidenceSpanRef, ...], Field(min_length=1)]
    positive_anchors: tuple[EvidenceSpanRef, ...] = ()
    failure_signature_ref: ArtifactRef | None = None
    grader_verdict_ref: ArtifactRef | None = None

    @field_validator("source_spans", "positive_anchors")
    @classmethod
    def _canonicalize_spans(
        cls,
        value: tuple[EvidenceSpanRef, ...],
    ) -> tuple[EvidenceSpanRef, ...]:
        keys = tuple(_span_key(span) for span in value)
        if len(keys) != len(set(keys)):
            raise ValueError("evidence span collections must not contain duplicates")
        return tuple(sorted(value, key=_span_key))


class FailureSignature(ImmutableModel):
    """A normalized failure description suitable for grouping and reuse."""

    schema_version: Literal["1"] = "1"
    failure_id: ShortText
    failure_class: ShortText
    summary: ClaimText
    symptom_codes: Annotated[tuple[ShortText, ...], Field(min_length=1)]
    affected_component_refs: tuple[ArtifactRef, ...] = ()
    affected_task_ids: tuple[ShortText, ...] = ()
    slice_tags: tuple[ShortText, ...] = ()

    @field_validator("symptom_codes", "affected_task_ids", "slice_tags")
    @classmethod
    def _canonicalize_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("set-like label fields must not contain duplicates")
        return tuple(sorted(value))

    @field_validator("affected_component_refs")
    @classmethod
    def _canonicalize_component_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        return _canonicalize_artifact_refs(value, field_name="affected_component_refs")


class DiagnosticCluster(ImmutableModel):
    """A durable cluster that references, rather than embeds, diagnostic artifacts."""

    schema_version: Literal["1"] = "1"
    cluster_id: ShortText
    label: ShortText
    failure_signature_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    representative_signature_ref: ArtifactRef
    evidence_packet_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    task_ids: Annotated[tuple[ShortText, ...], Field(min_length=1)]
    slice_tags: tuple[ShortText, ...] = ()

    @field_validator("failure_signature_refs", "evidence_packet_refs")
    @classmethod
    def _canonicalize_refs(
        cls,
        value: tuple[ArtifactRef, ...],
        info: object,
    ) -> tuple[ArtifactRef, ...]:
        field_name = getattr(info, "field_name", "artifact_refs")
        return _canonicalize_artifact_refs(value, field_name=field_name)

    @field_validator("task_ids", "slice_tags")
    @classmethod
    def _canonicalize_cluster_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("set-like label fields must not contain duplicates")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _representative_is_a_member(self) -> Self:
        if self.representative_signature_ref not in self.failure_signature_refs:
            raise ValueError(
                "representative_signature_ref must be present in failure_signature_refs"
            )
        return self


class EvidenceResolutionError(ValueError):
    """Raised when a span does not resolve against its claimed trajectory."""


def _validated_model[ModelT: ImmutableModel](
    model_type: type[ModelT],
    value: ModelT,
) -> ModelT:
    """Revalidate models so unchecked Pydantic copies cannot cross this boundary."""

    if not isinstance(value, model_type):
        raise TypeError(f"expected {model_type.__name__}, got {type(value).__name__}")
    return model_type.model_validate(
        value.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )


def resolve_evidence_span(
    span: EvidenceSpanRef,
    trajectory: Trajectory,
    actual_trajectory_ref: ArtifactRef,
) -> tuple[TrajectoryEvent, ...]:
    """Resolve an inclusive span and verify its artifact binding and event kinds.

    ``expected_event_kinds`` is a required set, not an exact event sequence: a
    span may contain additional context events, but every declared kind must be
    observed at least once.
    """

    checked_span = _validated_model(EvidenceSpanRef, span)
    checked_trajectory = _validated_model(Trajectory, trajectory)
    checked_ref = _validated_model(ArtifactRef, actual_trajectory_ref)

    if checked_span.trajectory_ref != checked_ref:
        raise EvidenceResolutionError(
            "evidence span trajectory_ref does not match the actual trajectory artifact"
        )
    media_type = checked_ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise EvidenceResolutionError("trajectory artifact must declare a JSON media type")
    trajectory_bytes = canonical_json_bytes(checked_trajectory)
    if len(trajectory_bytes) != checked_ref.size:
        raise EvidenceResolutionError("actual trajectory artifact size does not match its content")
    if sha256_bytes(trajectory_bytes) != checked_ref.sha256:
        raise EvidenceResolutionError("actual trajectory artifact hash does not match its content")
    if checked_span.end_sequence >= len(checked_trajectory.events):
        raise EvidenceResolutionError("evidence span is outside the actual trajectory event range")

    events = checked_trajectory.events[checked_span.start_sequence : checked_span.end_sequence + 1]
    if not events or events[0].sequence != checked_span.start_sequence:
        raise EvidenceResolutionError("evidence span start_sequence did not resolve")
    if events[-1].sequence != checked_span.end_sequence:
        raise EvidenceResolutionError("evidence span end_sequence did not resolve")

    observed_kinds = {event.kind for event in events}
    missing_kinds = tuple(
        kind for kind in checked_span.expected_event_kinds if kind not in observed_kinds
    )
    if missing_kinds:
        joined = ", ".join(kind.value for kind in missing_kinds)
        raise EvidenceResolutionError(f"evidence span is missing expected event kinds: {joined}")
    return events


def validate_evidence_span(
    span: EvidenceSpanRef,
    trajectory: Trajectory,
    actual_trajectory_ref: ArtifactRef,
) -> None:
    """Validate a span against a trajectory, raising on any mismatch."""

    resolve_evidence_span(span, trajectory, actual_trajectory_ref)


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return (ref.sha256, ref.size, ref.media_type)


def _canonicalize_artifact_refs(
    value: tuple[ArtifactRef, ...],
    *,
    field_name: str,
) -> tuple[ArtifactRef, ...]:
    digests = tuple(ref.sha256 for ref in value)
    if len(digests) != len(set(digests)):
        raise ValueError(f"{field_name} must not contain duplicate artifacts")
    return tuple(sorted(value, key=_artifact_ref_key))


def _span_key(span: EvidenceSpanRef) -> tuple[str, int, int, str, tuple[str, ...]]:
    return (
        span.trajectory_ref.sha256,
        span.start_sequence,
        span.end_sequence,
        span.claim,
        tuple(kind.value for kind in span.expected_event_kinds),
    )


__all__ = [
    "DiagnosticCluster",
    "EvidencePacket",
    "EvidenceResolutionError",
    "EvidenceSpanRef",
    "FailureSignature",
    "Trajectory",
    "TrajectoryEvent",
    "TrajectoryEventKind",
    "resolve_evidence_span",
    "validate_evidence_span",
]
