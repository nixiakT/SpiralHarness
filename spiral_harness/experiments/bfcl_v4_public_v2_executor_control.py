"""Leaf control-value helpers shared by BFCL public-v2 replay code."""

from __future__ import annotations

from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2ControlValue,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2ProposalDisposition,
)


def candidate_index_from_control(value: BfclV4PublicV2ControlValue) -> int | None:
    """Project a nomination value to its candidate slot, or parent fallback."""

    return {
        BfclV4PublicV2ControlValue.PARENT_FALLBACK: None,
        BfclV4PublicV2ControlValue.CANDIDATE_0: 0,
        BfclV4PublicV2ControlValue.CANDIDATE_1: 1,
        BfclV4PublicV2ControlValue.CANDIDATE_2: 2,
    }.get(value)


def candidate_event_is_valid(event: BfclV4PublicV2JournalEvent) -> bool:
    """Return the exact local proposal validity used by deterministic controls."""

    return (
        event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
        and event.proposal_disposition is BfclV4PublicV2ProposalDisposition.VALID
        and event.candidate_artifact_sha256 is not None
    )


__all__ = ["candidate_event_is_valid", "candidate_index_from_control"]
