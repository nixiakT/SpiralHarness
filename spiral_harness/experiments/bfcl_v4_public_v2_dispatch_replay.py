"""Independent replay checks for persisted BFCL v2 dispatch projections."""

from __future__ import annotations

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2DagNode,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_dispatch_contracts import (
    BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    BfclV4PublicV2DispatchReceipt,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2JournalEvent,
)


def validate_bfcl_v4_public_v2_persisted_dispatch(
    *,
    node: BfclV4PublicDevelopmentV2DagNode,
    prefix: tuple[BfclV4PublicV2JournalEvent, ...],
    event: BfclV4PublicV2JournalEvent,
) -> bool:
    """Return whether the persisted projection reconstructs one exact receipt."""

    lineage = (
        event.journal_prefix_fingerprint,
        event.request_materialization_fingerprint,
        event.native_request_fingerprint,
        event.proposal_batch_set_fingerprint,
    )
    if any(value is None for value in lineage):
        return False
    prefix_fingerprint, materialization, native_request, proposal_batches = lineage
    prior_batch_sets = tuple(
        item.proposal_batch_set_fingerprint
        for item in prefix
        if item.proposal_batch_set_fingerprint is not None
    )
    expected_batches = (
        prior_batch_sets[-1]
        if prior_batch_sets
        else BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT
    )
    if proposal_batches != expected_batches:
        return False
    try:
        dispatch = BfclV4PublicV2DispatchReceipt(
            node=node,
            node_reference_sha256=canonical_sha256(node),
            journal_prefix_fingerprint=prefix_fingerprint,
            journal_prefix_event_count=len(prefix),
            journal_prefix_tail_event_sha256=prefix[-1].fingerprint if prefix else None,
            campaign_plan_fingerprint=event.campaign_plan_fingerprint,
            node_schedule_content_sha256=event.node_schedule_content_sha256,
            runtime_fingerprint=event.runtime_fingerprint,
            semantic_release_fingerprint=event.semantic_release_fingerprint,
            request_materialization_fingerprint=materialization,
            native_request_fingerprint=native_request,
            request_payload_sha256=event.request_payload_sha256,
            proposal_batch_set_fingerprint=proposal_batches,
        )
    except (TypeError, ValueError):
        return False
    return event.dispatch_fingerprint == dispatch.fingerprint


__all__ = ["validate_bfcl_v4_public_v2_persisted_dispatch"]
