"""Source-linked trajectory and diagnostic evidence contracts."""

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
