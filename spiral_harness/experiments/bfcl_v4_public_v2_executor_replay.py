"""Network-free independent semantic replay for the BFCL public v2 journal."""

from __future__ import annotations

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2GateVariant,
    BfclV4PublicDevelopmentV2NodeKind,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2ControlRecord,
    BfclV4PublicV2ControlValue,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2JournalSnapshot,
    BfclV4PublicV2ProposalDisposition,
    BfclV4PublicV2ProviderRequest,
    BfclV4PublicV2PureAtBAggregationRecord,
    BfclV4PublicV2ReplayState,
)


class BfclV4PublicV2ExecutorError(RuntimeError):
    """A requested transition violates the frozen v2 rehearsal protocol."""


class BfclV4PublicV2ReplayError(BfclV4PublicV2ExecutorError):
    """The event chain cannot independently reconstruct the frozen campaign."""


_GRADABLE_KINDS = frozenset(
    {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        BfclV4PublicDevelopmentV2NodeKind.GATE,
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
    }
)


def _checked_campaign(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan | None,
) -> BfclV4PublicDevelopmentV2CampaignPlan:
    chosen = build_bfcl_v4_public_development_v2_campaign_plan() if campaign is None else campaign
    try:
        checked = BfclV4PublicDevelopmentV2CampaignPlan.model_validate(chosen, strict=True)
    except Exception as exc:
        raise BfclV4PublicV2ReplayError("v2 campaign failed strict validation") from exc
    if checked != build_bfcl_v4_public_development_v2_campaign_plan():
        raise BfclV4PublicV2ReplayError("v2 campaign differs from the repository-frozen plan")
    return checked


def _event_by_node(
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> dict[str, BfclV4PublicV2JournalEvent]:
    return {event.node_id: event for event in events}


def _proposal_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
    candidate_index: int,
) -> BfclV4PublicV2JournalEvent:
    proposal = next(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
        and candidate.candidate_index == candidate_index
    )
    try:
        return _event_by_node(events)[proposal.node_id]
    except KeyError as exc:
        raise BfclV4PublicV2ReplayError("candidate use precedes its proposal event") from exc


def _nomination_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
) -> BfclV4PublicV2JournalEvent:
    nomination = next(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.NOMINATION
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
    )
    try:
        return _event_by_node(events)[nomination.node_id]
    except KeyError as exc:
        raise BfclV4PublicV2ReplayError("adaptive use precedes its nomination event") from exc


def _decision_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
) -> BfclV4PublicV2JournalEvent:
    decision = next(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
    )
    try:
        return _event_by_node(events)[decision.node_id]
    except KeyError as exc:
        raise BfclV4PublicV2ReplayError("adaptive HOLDOUT precedes its GATE decision") from exc


def _candidate_from_control(value: BfclV4PublicV2ControlValue) -> int | None:
    return {
        BfclV4PublicV2ControlValue.PARENT_FALLBACK: None,
        BfclV4PublicV2ControlValue.CANDIDATE_0: 0,
        BfclV4PublicV2ControlValue.CANDIDATE_1: 1,
        BfclV4PublicV2ControlValue.CANDIDATE_2: 2,
    }.get(value)


def _candidate_valid(event: BfclV4PublicV2JournalEvent) -> bool:
    return (
        event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
        and event.proposal_disposition is BfclV4PublicV2ProposalDisposition.VALID
        and event.candidate_artifact_sha256 is not None
    )


def _expected_nomination(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
) -> BfclV4PublicV2ControlValue:
    by_node = _event_by_node(events)
    ranked: list[tuple[int, int]] = []
    for candidate_index in range(3):
        proposal = _proposal_event(campaign, events, node, candidate_index)
        if not _candidate_valid(proposal):
            continue
        fit_nodes = tuple(
            candidate
            for candidate in campaign.nodes
            if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT
            and candidate.replicate_id == node.replicate_id
            and candidate.arm is node.arm
            and candidate.candidate_index == candidate_index
        )
        if len(fit_nodes) != 10 or any(candidate.node_id not in by_node for candidate in fit_nodes):
            raise BfclV4PublicV2ReplayError("nomination lacks all ten candidate FIT events")
        correct = sum(by_node[candidate.node_id].binary_grade is True for candidate in fit_nodes)
        ranked.append((-correct, candidate_index))
    if not ranked:
        return BfclV4PublicV2ControlValue.PARENT_FALLBACK
    candidate_index = min(ranked)[1]
    return BfclV4PublicV2ControlValue(f"candidate-{candidate_index}")


def _expected_promotion(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
) -> BfclV4PublicV2ControlValue:
    by_node = _event_by_node(events)
    nomination = _nomination_event(campaign, events, node)
    if nomination.control_value is None:
        raise BfclV4PublicV2ReplayError("nomination control value is absent")
    candidate_index = _candidate_from_control(nomination.control_value)
    if candidate_index is None:
        return BfclV4PublicV2ControlValue.PARENT_FALLBACK
    proposal = _proposal_event(campaign, events, node, candidate_index)
    if not _candidate_valid(proposal):
        return BfclV4PublicV2ControlValue.PARENT_FALLBACK

    parent_fit_nodes = tuple(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
    )
    candidate_fit_nodes = tuple(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
        and candidate.candidate_index == candidate_index
    )
    gate_nodes = tuple(
        candidate
        for candidate in campaign.nodes
        if candidate.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
        and candidate.replicate_id == node.replicate_id
        and candidate.arm is node.arm
    )
    relevant_nodes = (*parent_fit_nodes, *candidate_fit_nodes, *gate_nodes)
    if any(candidate.node_id not in by_node for candidate in relevant_nodes):
        raise BfclV4PublicV2ReplayError("GATE decision lacks complete FIT/GATE evidence")
    relevant_events = tuple(by_node[candidate.node_id] for candidate in relevant_nodes)
    all_succeeded = all(
        event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
        for event in relevant_events
    )

    parent_fit = sum(
        by_node[candidate.node_id].binary_grade is True for candidate in parent_fit_nodes
    )
    candidate_fit = sum(
        by_node[candidate.node_id].binary_grade is True for candidate in candidate_fit_nodes
    )
    gate_by_coordinate = {
        (candidate.task_ref, candidate.rollout_index, candidate.gate_variant): by_node[
            candidate.node_id
        ]
        for candidate in gate_nodes
    }
    shadow_exact = True
    for task_ref in {candidate.task_ref for candidate in gate_nodes}:
        for rollout_index in range(3):
            controls = tuple(
                gate_by_coordinate[(task_ref, rollout_index, variant)]
                for variant in (
                    BfclV4PublicDevelopmentV2GateVariant.PARENT,
                    BfclV4PublicDevelopmentV2GateVariant.REVERT,
                    BfclV4PublicDevelopmentV2GateVariant.NEGATIVE_CONTROL,
                )
            )
            shadow_exact &= (
                len({(event.canonical_response, event.binary_grade) for event in controls}) == 1
            )
    parent_gate = sum(
        event.binary_grade is True
        for (task_ref, rollout, variant), event in gate_by_coordinate.items()
        if variant is BfclV4PublicDevelopmentV2GateVariant.PARENT
    )
    candidate_gate = sum(
        event.binary_grade is True
        for (task_ref, rollout, variant), event in gate_by_coordinate.items()
        if variant is BfclV4PublicDevelopmentV2GateVariant.NOMINATED_CANDIDATE
    )
    promote = (
        all_succeeded
        and shadow_exact
        and candidate_fit >= parent_fit
        and candidate_gate > parent_gate
    )
    return (
        BfclV4PublicV2ControlValue.PROMOTE
        if promote
        else BfclV4PublicV2ControlValue.PARENT_FALLBACK
    )


def _expected_executed_variant(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
) -> str:
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT:
        return "parent"
    if node.kind in {
        BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
        BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
    }:
        return node.harness_variant
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT:
        assert node.candidate_index is not None
        proposal = _proposal_event(campaign, events, node, node.candidate_index)
        return f"typed-candidate-{node.candidate_index}" if _candidate_valid(proposal) else "parent"
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE:
        if node.gate_variant is BfclV4PublicDevelopmentV2GateVariant.NOMINATED_CANDIDATE:
            nomination = _nomination_event(campaign, events, node)
            if nomination.control_value is None:
                raise BfclV4PublicV2ReplayError("GATE nomination value is absent")
            candidate_index = _candidate_from_control(nomination.control_value)
            return "parent" if candidate_index is None else f"typed-candidate-{candidate_index}"
        assert node.gate_variant is not None
        return node.gate_variant.value
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.HOLDOUT:
        if node.arm is BfclV4PublicDevelopmentV2Arm.PURE:
            return "bare"
        if node.arm is BfclV4PublicDevelopmentV2Arm.STATIC:
            return "static-frozen"
        decision = _decision_event(campaign, events, node)
        if decision.control_value is BfclV4PublicV2ControlValue.PROMOTE:
            nomination = _nomination_event(campaign, events, node)
            if nomination.control_value is None:
                raise BfclV4PublicV2ReplayError("promoted HOLDOUT nomination value is absent")
            candidate_index = _candidate_from_control(nomination.control_value)
            if candidate_index is None:
                raise BfclV4PublicV2ReplayError("promotion cannot select parent nomination")
            return f"typed-candidate-{candidate_index}"
        return "parent"
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE:
        return "bare"
    raise BfclV4PublicV2ReplayError("control node has no executed harness variant")


def _request(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node: BfclV4PublicDevelopmentV2DagNode,
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    request_payload_sha256: str,
    decision_barrier_evidence_fingerprint: str | None = None,
    evaluation_unlock_fingerprint: str | None = None,
) -> BfclV4PublicV2ProviderRequest:
    if node.campaign_call_slot is None or node.provider_seed_u63 is None:
        raise BfclV4PublicV2ExecutorError("provider request requires a model-call node")
    return BfclV4PublicV2ProviderRequest(
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        node_id=node.node_id,
        node_reference_sha256=canonical_sha256(node),
        campaign_call_slot=node.campaign_call_slot,
        provider_seed_u63=node.provider_seed_u63,
        request_payload_sha256=request_payload_sha256,
        decision_barrier_evidence_fingerprint=decision_barrier_evidence_fingerprint,
        evaluation_unlock_fingerprint=evaluation_unlock_fingerprint,
    )


def _control_request_hashes(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> tuple[str, str]:
    by_node = _event_by_node(events)
    try:
        evidence = tuple(by_node[node_id].fingerprint for node_id in node.allowed_evidence_from)
    except KeyError as exc:
        raise BfclV4PublicV2ReplayError("control input evidence is missing") from exc
    payload_sha256 = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-control-input/v1",
            "node_id": node.node_id,
            "evidence_event_sha256": evidence,
        }
    )
    request_fingerprint = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-control-request/v1",
            "campaign_plan_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
            "node_schedule_content_sha256": campaign.node_schedule_content_sha256,
            "mutation_catalog_fingerprint": campaign.mutation_catalog_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "semantic_release_fingerprint": semantic_release_fingerprint,
            "node_reference_sha256": canonical_sha256(node),
            "request_payload_sha256": payload_sha256,
        }
    )
    return payload_sha256, request_fingerprint


def _build_control_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> BfclV4PublicV2JournalEvent:
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.NOMINATION:
        event_kind = BfclV4PublicV2EventKind.NOMINATION
        control = _expected_nomination(campaign, events, node)
    elif node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION:
        event_kind = BfclV4PublicV2EventKind.DECISION
        control = _expected_promotion(campaign, events, node)
    else:
        raise BfclV4PublicV2ExecutorError("deterministic event requires a control node")
    payload_hash, request_fingerprint = _control_request_hashes(
        campaign,
        events,
        node,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
    )
    return BfclV4PublicV2JournalEvent(
        sequence=node.node_slot,
        previous_event_sha256=events[-1].fingerprint if events else None,
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        node_id=node.node_id,
        node_slot=node.node_slot,
        node_reference_sha256=canonical_sha256(node),
        event_kind=event_kind,
        request_fingerprint=request_fingerprint,
        request_payload_sha256=payload_hash,
        provider_attempts_consumed=0,
        control_value=control,
    )


def _validate_call_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
    event: BfclV4PublicV2JournalEvent,
) -> None:
    if event.event_kind is not BfclV4PublicV2EventKind.CALL:
        raise BfclV4PublicV2ReplayError("model-call node requires a call event")
    expected_variant = _expected_executed_variant(campaign, events, node)
    if event.executed_harness_variant != expected_variant:
        raise BfclV4PublicV2ReplayError("executed harness variant violates frozen fallback policy")
    succeeded = event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
    if succeeded is (event.canonical_response is None) or succeeded is (
        event.provider_response_fingerprint is None
    ):
        raise BfclV4PublicV2ReplayError("provider disposition and response lineage disagree")
    gradable = node.kind in _GRADABLE_KINDS
    if (event.binary_grade is not None) is not gradable:
        raise BfclV4PublicV2ReplayError("binary-grade presence differs from call kind")
    if not succeeded and gradable and event.binary_grade is not False:
        raise BfclV4PublicV2ReplayError("failed gradable call must deterministically grade false")
    graded = event.trusted_grader_receipt_fingerprint is not None
    if graded is not (succeeded and gradable) or event.trusted_grade_attempts_consumed != int(
        graded
    ):
        raise BfclV4PublicV2ReplayError("trusted grade receipt presence differs from call outcome")

    evaluation = node.kind in {
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
    }
    authority = (
        event.decision_barrier_evidence_fingerprint,
        event.evaluation_unlock_fingerprint,
    )
    if (authority[0] is not None) is not evaluation or (authority[1] is not None) is not evaluation:
        raise BfclV4PublicV2ReplayError("evaluation authority presence differs from call kind")
    prior_authorities = {
        (item.decision_barrier_evidence_fingerprint, item.evaluation_unlock_fingerprint)
        for item in events
        if item.evaluation_unlock_fingerprint is not None
    }
    if len(prior_authorities) > 1 or (prior_authorities and authority not in prior_authorities):
        raise BfclV4PublicV2ReplayError("evaluation calls do not share one frozen unlock")

    if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL:
        if event.proposal_disposition is None:
            raise BfclV4PublicV2ReplayError("proposal call lacks typed validation disposition")
        if (
            succeeded
            and event.proposal_disposition is BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
        ):
            raise BfclV4PublicV2ReplayError("successful proposal cannot claim provider failure")
        if not succeeded and (
            event.proposal_disposition is not BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
        ):
            raise BfclV4PublicV2ReplayError(
                "failed proposal must burn as provider-failure candidate"
            )
        valid = event.proposal_disposition is BfclV4PublicV2ProposalDisposition.VALID
        if valid is not (event.candidate_artifact_sha256 is not None):
            raise BfclV4PublicV2ReplayError("valid proposal and candidate artifact disagree")
    elif event.proposal_disposition is not None or event.candidate_artifact_sha256 is not None:
        raise BfclV4PublicV2ReplayError("non-proposal call cannot carry candidate resolution")


def _validate_event(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    event: BfclV4PublicV2JournalEvent,
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> None:
    sequence = len(events)
    if sequence >= len(campaign.nodes):
        raise BfclV4PublicV2ReplayError("journal exceeds the frozen 1,098-node DAG")
    node = campaign.nodes[sequence]
    expected_previous = events[-1].fingerprint if events else None
    expected_lineage = (
        sequence,
        expected_previous,
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        campaign.node_schedule_content_sha256,
        campaign.mutation_catalog_fingerprint,
        runtime_fingerprint,
        semantic_release_fingerprint,
        node.node_id,
        node.node_slot,
        canonical_sha256(node),
    )
    actual_lineage = (
        event.sequence,
        event.previous_event_sha256,
        event.campaign_plan_fingerprint,
        event.node_schedule_content_sha256,
        event.mutation_catalog_fingerprint,
        event.runtime_fingerprint,
        event.semantic_release_fingerprint,
        event.node_id,
        event.node_slot,
        event.node_reference_sha256,
    )
    if actual_lineage != expected_lineage:
        raise BfclV4PublicV2ReplayError("event sequence, hash chain, or frozen lineage changed")

    if node.consumes_model_call:
        request = _request(
            campaign,
            node,
            runtime_fingerprint=runtime_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
            request_payload_sha256=event.request_payload_sha256,
            decision_barrier_evidence_fingerprint=event.decision_barrier_evidence_fingerprint,
            evaluation_unlock_fingerprint=event.evaluation_unlock_fingerprint,
        )
        if event.request_fingerprint != request.fingerprint:
            raise BfclV4PublicV2ReplayError("call request fingerprint differs from exact lineage")
        _validate_call_event(campaign, events, node, event)
    else:
        expected = _build_control_event(
            campaign,
            events,
            node,
            runtime_fingerprint=runtime_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
        )
        if event != expected:
            raise BfclV4PublicV2ReplayError("nomination or promotion differs from recomputation")


def _aggregations(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> tuple[BfclV4PublicV2PureAtBAggregationRecord, ...]:
    if len(events) != 1_098:
        return ()
    by_node = _event_by_node(events)
    records: list[BfclV4PublicV2PureAtBAggregationRecord] = []
    for outer_seed in campaign.outer_seeds_u64:
        for allocation in campaign.pure_at_b_allocation:
            nodes = tuple(
                node
                for node in campaign.nodes
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
                and node.outer_seed_u64 == outer_seed
                and node.task_ref == allocation.task_ref
            )
            source_events = tuple(by_node[node.node_id] for node in nodes)
            responses = tuple(event.canonical_response for event in source_events)
            records.append(
                BfclV4PublicV2PureAtBAggregationRecord(
                    outer_seed_u64=outer_seed,
                    task_ref=allocation.task_ref,
                    source_event_sha256=tuple(event.fingerprint for event in source_events),
                    result=aggregate_bfcl_v4_public_development_v2_pure_at_b(responses),
                )
            )
    return tuple(records)


def _state(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> BfclV4PublicV2ReplayState:
    calls = tuple(event for event in events if event.event_kind is BfclV4PublicV2EventKind.CALL)
    succeeded = sum(
        event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
        for event in calls
    )
    controls: list[BfclV4PublicV2ControlRecord] = []
    for event in events:
        if event.event_kind is BfclV4PublicV2EventKind.CALL:
            continue
        node = campaign.nodes[event.node_slot]
        assert event.control_value is not None
        controls.append(
            BfclV4PublicV2ControlRecord(
                node_id=node.node_id,
                outer_seed_u64=node.outer_seed_u64,
                arm=node.arm,
                kind=node.kind,
                value=event.control_value,
            )
        )
    return BfclV4PublicV2ReplayState(
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        next_node_slot=len(events),
        tail_event_sha256=events[-1].fingerprint if events else None,
        completed_node_count=len(events),
        burned_call_slot_count=len(calls),
        succeeded_call_count=succeeded,
        failed_call_count=len(calls) - succeeded,
        provider_attempt_count=sum(event.provider_attempts_consumed for event in calls),
        crash_recovery_burn_count=sum(
            event.provider_attempt_disposition
            is BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN
            for event in calls
        ),
        nominations=tuple(
            item for item in controls if item.kind is BfclV4PublicDevelopmentV2NodeKind.NOMINATION
        ),
        decisions=tuple(
            item
            for item in controls
            if item.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        ),
        pure_at_b_aggregations=_aggregations(campaign, events),
        complete=len(events) == 1_098,
    )


def replay_bfcl_v4_public_v2_journal(
    snapshot: BfclV4PublicV2JournalSnapshot,
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan | None = None,
    require_complete: bool = True,
) -> BfclV4PublicV2ReplayState:
    checked_campaign = _checked_campaign(campaign)
    try:
        checked_snapshot = BfclV4PublicV2JournalSnapshot.model_validate(snapshot, strict=True)
    except Exception as exc:
        raise BfclV4PublicV2ReplayError("journal snapshot failed strict validation") from exc
    expected_header = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        checked_campaign.node_schedule_content_sha256,
    )
    if (
        checked_snapshot.campaign_plan_fingerprint,
        checked_snapshot.node_schedule_content_sha256,
    ) != expected_header:
        raise BfclV4PublicV2ReplayError("journal header differs from the frozen campaign")
    prefix: tuple[BfclV4PublicV2JournalEvent, ...] = ()
    for event in checked_snapshot.events:
        _validate_event(
            checked_campaign,
            prefix,
            event,
            runtime_fingerprint=checked_snapshot.runtime_fingerprint,
            semantic_release_fingerprint=checked_snapshot.semantic_release_fingerprint,
        )
        prefix = (*prefix, event)
    pending = checked_snapshot.pending_call_reservation
    if pending is not None:
        node = checked_campaign.nodes[len(prefix)]
        expected_request = _request(
            checked_campaign,
            node,
            runtime_fingerprint=checked_snapshot.runtime_fingerprint,
            semantic_release_fingerprint=checked_snapshot.semantic_release_fingerprint,
            request_payload_sha256=pending.request.request_payload_sha256,
            decision_barrier_evidence_fingerprint=(
                pending.request.decision_barrier_evidence_fingerprint
            ),
            evaluation_unlock_fingerprint=pending.request.evaluation_unlock_fingerprint,
        )
        if pending.node != node or pending.request != expected_request:
            raise BfclV4PublicV2ReplayError(
                "pending reservation differs from the exact next DAG call"
            )
    state = _state(
        checked_campaign,
        prefix,
        runtime_fingerprint=checked_snapshot.runtime_fingerprint,
        semantic_release_fingerprint=checked_snapshot.semantic_release_fingerprint,
    )
    if require_complete and not state.complete:
        raise BfclV4PublicV2ReplayError("journal is missing one or more frozen DAG nodes")
    return state


__all__ = [
    "BfclV4PublicV2ExecutorError",
    "BfclV4PublicV2ReplayError",
    "replay_bfcl_v4_public_v2_journal",
]
