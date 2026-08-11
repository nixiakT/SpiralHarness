"""Structural contract for an independent candidate verifier."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from spiral_harness.verification.models import (
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
)


@runtime_checkable
class CandidateVerifier(Protocol):
    """Decide promotion from frozen observations, never from proposer opinion."""

    def evaluate(
        self,
        parent_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
        candidate_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
        mechanism_evidence: MechanismEvidence | Sequence[MechanismCheck] | None = None,
        *,
        parent_harness_id: str | None = None,
        candidate_harness_id: str | None = None,
    ) -> GateDecision: ...


__all__ = ["CandidateVerifier"]
