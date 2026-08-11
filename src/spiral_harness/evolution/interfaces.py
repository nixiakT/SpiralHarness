"""Small replaceable interfaces for diagnosis and candidate proposal."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

EvidenceT_contra = TypeVar("EvidenceT_contra", contravariant=True)
DiagnosisT_co = TypeVar("DiagnosisT_co", covariant=True)
DiagnosisT_contra = TypeVar("DiagnosisT_contra", contravariant=True)
ParentT_contra = TypeVar("ParentT_contra", contravariant=True)
CandidateT_co = TypeVar("CandidateT_co", covariant=True)


@runtime_checkable
class Diagnoser(Protocol[EvidenceT_contra, DiagnosisT_co]):
    """Turn source-linked exploration evidence into falsifiable diagnoses."""

    def diagnose(self, evidence: EvidenceT_contra) -> tuple[DiagnosisT_co, ...]: ...


@runtime_checkable
class Proposer(Protocol[DiagnosisT_contra, ParentT_contra, CandidateT_co]):
    """Produce zero or more bounded candidates; an empty tuple is a decline."""

    def propose(
        self,
        diagnosis: DiagnosisT_contra,
        *,
        parent: ParentT_contra,
    ) -> tuple[CandidateT_co, ...]: ...


__all__ = ["Diagnoser", "Proposer"]
