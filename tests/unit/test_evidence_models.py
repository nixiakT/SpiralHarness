from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.evidence.models import (
    DiagnosticCluster,
    EvidencePacket,
    EvidenceResolutionError,
    EvidenceSpanRef,
    FailureSignature,
    Trajectory,
    TrajectoryEvent,
    TrajectoryEventKind,
    resolve_evidence_span,
    validate_evidence_span,
)


def artifact(digit: str, *, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def event(
    sequence: int,
    kind: TrajectoryEventKind,
    *,
    elapsed_ms: float | None = None,
    parents: tuple[int, ...] | None = None,
    component_ref: ArtifactRef | None = None,
) -> TrajectoryEvent:
    return TrajectoryEvent(
        sequence=sequence,
        elapsed_ms=float(sequence) if elapsed_ms is None else elapsed_ms,
        kind=kind,
        payload_ref=artifact(hex(sequence + 1)[2:]),
        component_ref=component_ref,
        causal_parent_sequences=(sequence - 1,) if parents is None and sequence else parents or (),
    )


def trajectory(*, terminal: TrajectoryEventKind = TrajectoryEventKind.RUN_FINISHED) -> Trajectory:
    component = artifact("a", media_type="application/vnd.component+json")
    return Trajectory(
        run_id="run-1",
        task_id="task-1",
        seed=7,
        harness_ref=artifact("b", media_type="application/vnd.harness+json"),
        execution_fingerprint="model=x;runtime=y;seed=7",
        events=(
            event(0, TrajectoryEventKind.RUN_STARTED),
            event(1, TrajectoryEventKind.COMPONENT_ACTIVATED, component_ref=component),
            event(2, TrajectoryEventKind.SKILL_SELECTED, component_ref=component),
            event(3, TrajectoryEventKind.MODEL_REQUEST),
            event(4, TrajectoryEventKind.MODEL_RESPONSE),
            event(5, TrajectoryEventKind.FINAL_OUTPUT),
            event(6, terminal),
        ),
    )


def span(trajectory_ref: ArtifactRef | None = None) -> EvidenceSpanRef:
    return EvidenceSpanRef(
        trajectory_ref=trajectory_ref or artifact("c"),
        start_sequence=1,
        end_sequence=4,
        claim="the skill was activated before the model response",
        expected_event_kinds=(
            TrajectoryEventKind.MODEL_RESPONSE,
            TrajectoryEventKind.COMPONENT_ACTIVATED,
            TrajectoryEventKind.SKILL_SELECTED,
        ),
    )


def trajectory_ref(value: Trajectory) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=sha256_bytes(payload),
        size=len(payload),
        media_type="application/vnd.spiral-harness.trajectory.v1+json",
    )


def test_event_kind_vocabulary_covers_the_required_runtime_surface() -> None:
    assert {kind.value for kind in TrajectoryEventKind} == {
        "run_started",
        "run_finished",
        "component_activated",
        "model_request",
        "model_response",
        "skill_selected",
        "tool_call",
        "tool_result",
        "memory_read",
        "memory_write",
        "middleware",
        "final_output",
        "error",
    }


def test_event_canonicalizes_causal_parents_and_is_frozen() -> None:
    value = event(
        4,
        TrajectoryEventKind.MODEL_RESPONSE,
        parents=(2, 0, 3),
    )

    assert value.causal_parent_sequences == (0, 2, 3)
    with pytest.raises(ValidationError):
        value.sequence = 8  # type: ignore[misc]


@pytest.mark.parametrize("parents", [(1, 1), (2,), (3,), (-1,)])
def test_event_rejects_duplicate_nonpreceding_or_negative_parents(
    parents: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        event(2, TrajectoryEventKind.MODEL_RESPONSE, parents=parents)


def test_event_requires_artifact_refs_instead_of_inline_payloads() -> None:
    with pytest.raises(ValidationError):
        TrajectoryEvent(
            sequence=0,
            elapsed_ms=0.0,
            kind=TrajectoryEventKind.RUN_STARTED,
            payload_ref={"message": "large inline payload"},
            unexpected_payload={"also": "forbidden"},
        )

    for kind in (
        TrajectoryEventKind.COMPONENT_ACTIVATED,
        TrajectoryEventKind.SKILL_SELECTED,
    ):
        with pytest.raises(ValidationError, match="requires component_ref"):
            event(1, kind)


def test_complete_trajectory_accepts_success_and_error_terminals() -> None:
    success = trajectory()
    failure = trajectory(terminal=TrajectoryEventKind.ERROR)

    assert tuple(item.sequence for item in success.events) == tuple(range(7))
    assert success.events[-1].kind is TrajectoryEventKind.RUN_FINISHED
    assert failure.events[-1].kind is TrajectoryEventKind.ERROR


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (event(0, TrajectoryEventKind.RUN_STARTED), event(2, TrajectoryEventKind.ERROR)),
            "continuous",
        ),
        (
            (event(0, TrajectoryEventKind.MODEL_REQUEST), event(1, TrajectoryEventKind.ERROR)),
            "first",
        ),
        (
            (
                event(0, TrajectoryEventKind.RUN_STARTED),
                event(1, TrajectoryEventKind.FINAL_OUTPUT),
            ),
            "last",
        ),
        (
            (
                event(0, TrajectoryEventKind.RUN_STARTED),
                event(1, TrajectoryEventKind.RUN_FINISHED),
                event(2, TrajectoryEventKind.ERROR),
            ),
            "terminal",
        ),
        (
            (
                event(0, TrajectoryEventKind.RUN_STARTED, elapsed_ms=2.0),
                event(1, TrajectoryEventKind.ERROR, elapsed_ms=1.0),
            ),
            "non-decreasing",
        ),
    ],
)
def test_trajectory_rejects_invalid_ordering(
    events: tuple[TrajectoryEvent, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Trajectory(
            run_id="run",
            task_id="task",
            seed=0,
            harness_ref=artifact("d"),
            execution_fingerprint="fixed",
            events=events,
        )


def test_trajectory_revalidates_unchecked_event_parent_references() -> None:
    invalid_terminal = event(1, TrajectoryEventKind.ERROR).model_copy(
        update={"causal_parent_sequences": (99,)}
    )

    with pytest.raises(ValidationError, match="strictly earlier"):
        Trajectory(
            run_id="run",
            task_id="task",
            seed=0,
            harness_ref=artifact("d"),
            execution_fingerprint="fixed",
            events=(event(0, TrajectoryEventKind.RUN_STARTED), invalid_terminal),
        )


def test_span_has_inclusive_ordered_range_and_canonical_expected_kinds() -> None:
    value = span()

    assert value.expected_event_kinds == (
        TrajectoryEventKind.COMPONENT_ACTIVATED,
        TrajectoryEventKind.MODEL_RESPONSE,
        TrajectoryEventKind.SKILL_SELECTED,
    )

    with pytest.raises(ValidationError, match="end_sequence"):
        EvidenceSpanRef(
            trajectory_ref=artifact("c"),
            start_sequence=4,
            end_sequence=3,
            claim="invalid range",
            expected_event_kinds=(TrajectoryEventKind.ERROR,),
        )

    with pytest.raises(ValidationError, match="duplicates"):
        EvidenceSpanRef(
            trajectory_ref=artifact("c"),
            start_sequence=0,
            end_sequence=1,
            claim="duplicate expected kinds",
            expected_event_kinds=(
                TrajectoryEventKind.RUN_STARTED,
                TrajectoryEventKind.RUN_STARTED,
            ),
        )


def test_resolve_span_checks_ref_inclusive_range_and_required_kinds() -> None:
    value = trajectory()
    actual_ref = trajectory_ref(value)
    resolved = resolve_evidence_span(span(actual_ref), value, actual_ref)

    assert tuple(item.sequence for item in resolved) == (1, 2, 3, 4)
    validate_evidence_span(span(actual_ref), value, actual_ref)


def test_resolve_span_rejects_ref_range_and_kind_mismatches() -> None:
    value = trajectory()
    actual_ref = trajectory_ref(value)
    with pytest.raises(EvidenceResolutionError, match="does not match"):
        resolve_evidence_span(span(artifact("d")), value, actual_ref)

    outside = span(actual_ref).model_copy(update={"end_sequence": 99})
    with pytest.raises(EvidenceResolutionError, match="outside"):
        resolve_evidence_span(outside, value, actual_ref)

    missing_kind = span(actual_ref).model_copy(
        update={"expected_event_kinds": (TrajectoryEventKind.TOOL_CALL,)}
    )
    with pytest.raises(EvidenceResolutionError, match="tool_call"):
        resolve_evidence_span(missing_kind, value, actual_ref)

    wrong_size = actual_ref.model_copy(update={"size": actual_ref.size + 1})
    with pytest.raises(EvidenceResolutionError, match="size"):
        resolve_evidence_span(span(wrong_size), value, wrong_size)

    wrong_hash = actual_ref.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(EvidenceResolutionError, match="hash"):
        resolve_evidence_span(span(wrong_hash), value, wrong_hash)

    wrong_media = actual_ref.model_copy(update={"media_type": "text/plain"})
    with pytest.raises(EvidenceResolutionError, match="JSON media type"):
        resolve_evidence_span(span(wrong_media), value, wrong_media)


def test_resolve_span_revalidates_unchecked_models() -> None:
    value = trajectory()
    actual_ref = trajectory_ref(value)
    unchecked = span(actual_ref).model_copy(update={"start_sequence": -1})

    with pytest.raises(ValidationError):
        resolve_evidence_span(unchecked, value, actual_ref)


def test_evidence_packet_canonicalizes_spans_and_rejects_duplicates() -> None:
    first = span(artifact("c"))
    second = EvidenceSpanRef(
        trajectory_ref=artifact("d"),
        start_sequence=0,
        end_sequence=1,
        claim="a successful negative-control run",
        expected_event_kinds=(TrajectoryEventKind.RUN_STARTED,),
    )
    packet = EvidencePacket(
        source_spans=(second, first),
        positive_anchors=(second, first),
        failure_signature_ref=artifact("e"),
        grader_verdict_ref=artifact("f"),
    )

    assert packet.source_spans == (first, second)
    assert packet.positive_anchors == (first, second)

    with pytest.raises(ValidationError, match="duplicates"):
        EvidencePacket(source_spans=(first, first))


def test_freeform_payloads_are_bounded_and_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpanRef(
            trajectory_ref=artifact("c"),
            start_sequence=0,
            end_sequence=1,
            claim="x" * 1_025,
            expected_event_kinds=(TrajectoryEventKind.RUN_STARTED,),
        )

    with pytest.raises(ValidationError):
        EvidencePacket(
            source_spans=(span(),),
            raw_payload={"trajectory": "must be stored out of line"},
        )


def test_failure_signature_canonicalizes_every_set_like_field() -> None:
    signature = FailureSignature(
        failure_id="failure-1",
        failure_class="tool-selection",
        summary="the wrong tool was selected after a valid skill activation",
        symptom_codes=("wrong-tool", "low-reward"),
        affected_component_refs=(artifact("f"), artifact("e")),
        affected_task_ids=("task-b", "task-a"),
        slice_tags=("tools", "protected"),
    )

    assert signature.symptom_codes == ("low-reward", "wrong-tool")
    assert tuple(ref.sha256 for ref in signature.affected_component_refs) == (
        "e" * 64,
        "f" * 64,
    )
    assert signature.affected_task_ids == ("task-a", "task-b")
    assert signature.slice_tags == ("protected", "tools")


@pytest.mark.parametrize(
    "update",
    [
        {"symptom_codes": ("same", "same")},
        {"affected_component_refs": (artifact("e"), artifact("e"))},
        {"affected_task_ids": ("same", "same")},
        {"slice_tags": ("same", "same")},
    ],
)
def test_failure_signature_rejects_duplicate_set_members(update: dict[str, object]) -> None:
    values: dict[str, object] = {
        "failure_id": "failure-1",
        "failure_class": "tool-selection",
        "summary": "normalized failure",
        "symptom_codes": ("base",),
    }
    values.update(update)
    with pytest.raises(ValidationError, match="duplicate"):
        FailureSignature(**values)


def test_diagnostic_cluster_is_canonical_and_representative_is_a_member() -> None:
    signature_a = artifact("a")
    signature_b = artifact("b")
    cluster = DiagnosticCluster(
        cluster_id="cluster-1",
        label="tool selection failures",
        failure_signature_refs=(signature_b, signature_a),
        representative_signature_ref=signature_a,
        evidence_packet_refs=(artifact("d"), artifact("c")),
        task_ids=("task-b", "task-a"),
        slice_tags=("tools", "protected"),
    )

    assert cluster.failure_signature_refs == (signature_a, signature_b)
    assert tuple(ref.sha256 for ref in cluster.evidence_packet_refs) == (
        "c" * 64,
        "d" * 64,
    )
    assert cluster.task_ids == ("task-a", "task-b")
    assert cluster.slice_tags == ("protected", "tools")

    with pytest.raises(ValidationError, match="representative_signature_ref"):
        DiagnosticCluster(
            cluster_id="cluster-2",
            label="invalid representative",
            failure_signature_refs=(signature_a,),
            representative_signature_ref=signature_b,
            evidence_packet_refs=(artifact("c"),),
            task_ids=("task-a",),
        )


@pytest.mark.parametrize(
    "field, duplicate",
    [
        ("failure_signature_refs", (artifact("a"), artifact("a"))),
        ("evidence_packet_refs", (artifact("c"), artifact("c"))),
        ("task_ids", ("task-a", "task-a")),
        ("slice_tags", ("tools", "tools")),
    ],
)
def test_diagnostic_cluster_rejects_duplicate_set_members(
    field: str,
    duplicate: tuple[object, object],
) -> None:
    values: dict[str, object] = {
        "cluster_id": "cluster-1",
        "label": "cluster",
        "failure_signature_refs": (artifact("a"),),
        "representative_signature_ref": artifact("a"),
        "evidence_packet_refs": (artifact("c"),),
        "task_ids": ("task-a",),
    }
    values[field] = duplicate
    with pytest.raises(ValidationError, match="duplicate"):
        DiagnosticCluster(**values)
