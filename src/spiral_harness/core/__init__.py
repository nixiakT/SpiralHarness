"""Immutable domain objects and deterministic serialization primitives."""

from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
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
    "ArtifactRef",
    "BudgetPolicy",
    "CandidateMutation",
    "ComponentKind",
    "HarnessComponentRef",
    "HarnessManifest",
    "MutationHypothesis",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "sha256_bytes",
]
