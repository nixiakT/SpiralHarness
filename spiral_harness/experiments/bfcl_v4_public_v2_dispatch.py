"""Compact dispatch-boundary helpers for the BFCL public-v2 executor."""

from __future__ import annotations

from typing import Protocol

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.experiments.bfcl_v4_public_v2_dispatch_contracts import (
    BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    BfclV4PublicV2DispatchContext,
    BfclV4PublicV2DispatchControl,
    BfclV4PublicV2DispatchReceipt,
    bfcl_v4_public_v2_journal_prefix_fingerprint,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2ProposalDisposition,
    BfclV4PublicV2ProviderAttempt,
    BfclV4PublicV2ProviderRequest,
)


class BfclV4PublicV2DispatchError(ValueError):
    """A binder returned another node, prefix, or request contract."""


class _Binder(Protocol):
    def bind(
        self,
        *,
        node: BfclV4PublicDevelopmentV2DagNode,
        context: BfclV4PublicV2DispatchContext,
    ) -> BfclV4PublicV2DispatchReceipt: ...


class _Provider(Protocol):
    def execute(
        self,
        request: BfclV4PublicV2ProviderRequest,
        node: BfclV4PublicDevelopmentV2DagNode,
        dispatch: BfclV4PublicV2DispatchReceipt,
    ) -> BfclV4PublicV2ProviderAttempt: ...


def recover_bfcl_v4_public_v2_dispatch_state(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> tuple[tuple[BfclV4PublicV2DispatchControl, ...], str]:
    """Recover compact binder state once when opening a durable prefix."""

    controls = tuple(
        control
        for event in events
        if (control := project_bfcl_v4_public_v2_dispatch_control(campaign=campaign, event=event))
        is not None
    )
    batch_sets = tuple(
        event.proposal_batch_set_fingerprint
        for event in events
        if event.proposal_batch_set_fingerprint is not None
    )
    return (
        controls,
        batch_sets[-1] if batch_sets else BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    )


def build_bfcl_v4_public_v2_dispatch_context(
    *,
    campaign_plan_fingerprint: str,
    node_schedule_content_sha256: str,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    event_count: int,
    tail_event_sha256: str | None,
    controls: tuple[BfclV4PublicV2DispatchControl, ...],
    proposal_batch_set_fingerprint: str,
) -> BfclV4PublicV2DispatchContext:
    """Build one O(1), response-free view of the current hash-chain prefix."""

    prefix = bfcl_v4_public_v2_journal_prefix_fingerprint(
        campaign_plan_fingerprint=campaign_plan_fingerprint,
        node_schedule_content_sha256=node_schedule_content_sha256,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        event_count=event_count,
        tail_event_sha256=tail_event_sha256,
    )
    return BfclV4PublicV2DispatchContext(
        campaign_plan_fingerprint=campaign_plan_fingerprint,
        node_schedule_content_sha256=node_schedule_content_sha256,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        journal_prefix_event_count=event_count,
        journal_prefix_tail_event_sha256=tail_event_sha256,
        journal_prefix_fingerprint=prefix,
        controls=controls,
        proposal_batch_set_fingerprint=proposal_batch_set_fingerprint,
    )


def recover_bfcl_v4_public_v2_dispatch_context(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> BfclV4PublicV2DispatchContext:
    """Recover one compact context from a replay-verified durable prefix."""

    controls, batches = recover_bfcl_v4_public_v2_dispatch_state(
        campaign=campaign,
        events=events,
    )
    return build_bfcl_v4_public_v2_dispatch_context(
        campaign_plan_fingerprint=campaign.fingerprint,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        event_count=len(events),
        tail_event_sha256=events[-1].fingerprint if events else None,
        controls=controls,
        proposal_batch_set_fingerprint=batches,
    )


def advance_bfcl_v4_public_v2_dispatch_context(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    context: BfclV4PublicV2DispatchContext,
    event: BfclV4PublicV2JournalEvent,
) -> BfclV4PublicV2DispatchContext:
    """Advance the compact context after one already-validated terminal event."""

    control = project_bfcl_v4_public_v2_dispatch_control(campaign=campaign, event=event)
    controls = context.controls if control is None else (*context.controls, control)
    return build_bfcl_v4_public_v2_dispatch_context(
        campaign_plan_fingerprint=context.campaign_plan_fingerprint,
        node_schedule_content_sha256=context.node_schedule_content_sha256,
        runtime_fingerprint=context.runtime_fingerprint,
        semantic_release_fingerprint=context.semantic_release_fingerprint,
        event_count=context.journal_prefix_event_count + 1,
        tail_event_sha256=event.fingerprint,
        controls=controls,
        proposal_batch_set_fingerprint=context.proposal_batch_set_fingerprint,
    )


def project_bfcl_v4_public_v2_dispatch_control(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    event: BfclV4PublicV2JournalEvent,
) -> BfclV4PublicV2DispatchControl | None:
    """Project one replayed control event without exposing call evidence."""

    if event.event_kind is BfclV4PublicV2EventKind.CALL:
        return None
    node = campaign.nodes[event.node_slot]
    if event.control_value is None or node.replicate_id is None:
        raise BfclV4PublicV2DispatchError("control event lacks its frozen coordinate")
    return BfclV4PublicV2DispatchControl(
        node_id=node.node_id,
        replicate_id=node.replicate_id,
        arm=node.arm,
        kind=node.kind.value,
        value=event.control_value.value,
        event_fingerprint=event.fingerprint,
    )


def bind_bfcl_v4_public_v2_dispatch(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node: BfclV4PublicDevelopmentV2DagNode,
    context: BfclV4PublicV2DispatchContext,
    binder: _Binder,
) -> BfclV4PublicV2DispatchReceipt:
    """Validate one exact compact receipt before a durable reservation."""

    if (
        context.campaign_plan_fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or context.node_schedule_content_sha256 != campaign.node_schedule_content_sha256
        or context.journal_prefix_event_count != node.node_slot
        or node.node_slot >= len(campaign.nodes)
        or campaign.nodes[node.node_slot] != node
    ):
        raise BfclV4PublicV2DispatchError(
            "request dispatch context is not the next node of the frozen campaign"
        )
    try:
        raw = binder.bind(node=node, context=context)
        if type(raw) is not BfclV4PublicV2DispatchReceipt:
            raise TypeError("binder returned another dispatch contract")
        dispatch = BfclV4PublicV2DispatchReceipt.model_validate(raw, strict=True)
    except Exception as error:
        raise BfclV4PublicV2DispatchError(
            "request binder failed exact materialization dispatch"
        ) from error
    observed = (
        dispatch.node,
        dispatch.journal_prefix_fingerprint,
        dispatch.journal_prefix_event_count,
        dispatch.journal_prefix_tail_event_sha256,
        dispatch.campaign_plan_fingerprint,
        dispatch.node_schedule_content_sha256,
        dispatch.runtime_fingerprint,
        dispatch.semantic_release_fingerprint,
    )
    expected = (
        node,
        context.journal_prefix_fingerprint,
        context.journal_prefix_event_count,
        context.journal_prefix_tail_event_sha256,
        context.campaign_plan_fingerprint,
        context.node_schedule_content_sha256,
        context.runtime_fingerprint,
        context.semantic_release_fingerprint,
    )
    if observed != expected:
        raise BfclV4PublicV2DispatchError(
            "request dispatch differs from the next node or replayed prefix"
        )
    if dispatch.proposal_batch_set_fingerprint != context.proposal_batch_set_fingerprint:
        raise BfclV4PublicV2DispatchError("dispatch changed the frozen proposal batch set")
    return dispatch


def execute_bfcl_v4_public_v2_dispatch(
    *,
    provider: _Provider,
    request: BfclV4PublicV2ProviderRequest,
    node: BfclV4PublicDevelopmentV2DagNode,
    dispatch: BfclV4PublicV2DispatchReceipt,
) -> BfclV4PublicV2ProviderAttempt:
    """Execute once; a foreign/malformed result deterministically burns the slot."""

    try:
        raw = provider.execute(request, node, dispatch)
        attempt = BfclV4PublicV2ProviderAttempt.model_validate(raw, strict=True)
        if attempt.dispatch_fingerprint != dispatch.fingerprint:
            raise ValueError("provider attempt belongs to another dispatch")
        return attempt
    except Exception:
        return BfclV4PublicV2ProviderAttempt(
            dispatch_fingerprint=dispatch.fingerprint,
            disposition=BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE,
            proposal_disposition=(
                BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
                else None
            ),
        )


__all__ = [
    "BfclV4PublicV2DispatchError",
    "advance_bfcl_v4_public_v2_dispatch_context",
    "bind_bfcl_v4_public_v2_dispatch",
    "build_bfcl_v4_public_v2_dispatch_context",
    "execute_bfcl_v4_public_v2_dispatch",
    "project_bfcl_v4_public_v2_dispatch_control",
    "recover_bfcl_v4_public_v2_dispatch_context",
    "recover_bfcl_v4_public_v2_dispatch_state",
]
