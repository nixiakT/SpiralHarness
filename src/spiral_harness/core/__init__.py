"""Immutable domain objects and deterministic serialization primitives."""

from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from spiral_harness.core.experiment import (
    CandidateManifest,
    ExperimentManifest,
    MutationPolicy,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.lifecycle import (
    TERMINAL_CANDIDATE_STATES,
    CandidateLifecycle,
    CandidateLifecycleError,
    CandidateLifecycleEvent,
    CandidateState,
    allowed_candidate_transitions,
    replay_candidate_lifecycle,
)
from spiral_harness.core.models import (
    ArtifactRef,
    BudgetPolicy,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)

__all__ = [
    "TERMINAL_CANDIDATE_STATES",
    "ArtifactRef",
    "BudgetPolicy",
    "CandidateLifecycle",
    "CandidateLifecycleError",
    "CandidateLifecycleEvent",
    "CandidateManifest",
    "CandidateMutation",
    "CandidateState",
    "ComponentKind",
    "ExperimentManifest",
    "HarnessComponentRef",
    "HarnessManifest",
    "MutationHypothesis",
    "MutationPolicy",
    "ProtocolManifest",
    "ProtocolPartition",
    "ProtocolSplit",
    "allowed_candidate_transitions",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "replay_candidate_lifecycle",
    "sha256_bytes",
]
