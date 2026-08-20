"""Independent replay gate for BFCL public-v2 HOLDOUT evaluation authority."""

from __future__ import annotations

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2Stage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_barrier_capability import (
    BfclV4PublicV2VerifiedDecisionBarrierCapability,
    BfclV4PublicV2VerifiedDecisionBarrierReceipt,
    _mint_bfcl_v4_public_v2_verified_decision_barrier,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import Sha256
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalSnapshot,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_replay import (
    replay_bfcl_v4_public_v2_journal,
)


class BfclV4PublicV2VerifiedBarrierReplayError(RuntimeError):
    """The supplied durable prefix is not the exact pre-evaluation barrier."""


def verify_bfcl_v4_public_v2_decision_barrier_snapshot(
    snapshot: BfclV4PublicV2JournalSnapshot,
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    runtime_fingerprint: Sha256,
    semantic_release_fingerprint: Sha256,
) -> BfclV4PublicV2VerifiedDecisionBarrierCapability:
    """Replay the exact pre-evaluation prefix and mint non-serializable authority."""

    try:
        checked_campaign = BfclV4PublicDevelopmentV2CampaignPlan.model_validate(
            campaign,
            strict=True,
        )
        checked_snapshot = BfclV4PublicV2JournalSnapshot.model_validate(snapshot, strict=True)
        if (
            checked_campaign.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
            or checked_campaign.node_schedule_content_sha256
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
            or checked_snapshot.campaign_plan_fingerprint != checked_campaign.fingerprint
            or checked_snapshot.node_schedule_content_sha256
            != checked_campaign.node_schedule_content_sha256
            or checked_snapshot.runtime_fingerprint != runtime_fingerprint
            or checked_snapshot.semantic_release_fingerprint != semantic_release_fingerprint
            or checked_snapshot.pending_call_reservation is not None
        ):
            raise ValueError("barrier roots changed")
        first_evaluation_slot = next(
            node.node_slot
            for node in checked_campaign.nodes
            if node.stage is BfclV4PublicDevelopmentV2Stage.EVALUATION
        )
        if len(checked_snapshot.events) != first_evaluation_slot:
            raise ValueError("snapshot is not the exact pre-evaluation prefix")
        replay = replay_bfcl_v4_public_v2_journal(
            checked_snapshot,
            campaign=checked_campaign,
            require_complete=False,
        )
        if replay.next_node_slot != first_evaluation_slot or len(replay.decisions) != 6:
            raise ValueError("replay lacks the exact six-decision barrier")

        decision_nodes = tuple(
            node
            for node in checked_campaign.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        )
        events_by_node = {event.node_id: event for event in checked_snapshot.events}
        decision_events = tuple(events_by_node[node.node_id] for node in decision_nodes)
        if len(decision_nodes) != 6 or any(
            event.event_kind is not BfclV4PublicV2EventKind.DECISION for event in decision_events
        ):
            raise ValueError("replayed barrier decisions changed kind")
        evidence = BfclV4PublicV2DecisionBarrierEvidence(
            semantic_release_fingerprint=semantic_release_fingerprint,
            decision_node_references=tuple(canonical_sha256(node) for node in decision_nodes),
            decision_event_fingerprints=tuple(event.fingerprint for event in decision_events),
            final_decision_event_fingerprint=decision_events[-1].fingerprint,
        )
        receipt = BfclV4PublicV2VerifiedDecisionBarrierReceipt(
            evidence=evidence,
            evidence_fingerprint=evidence.fingerprint,
            journal_snapshot_fingerprint=checked_snapshot.fingerprint,
            journal_prefix_event_count=len(checked_snapshot.events),
            journal_tail_event_fingerprint=decision_events[-1].fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
            replay_state_fingerprint=replay.fingerprint,
        )
    except (KeyError, StopIteration, TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicV2VerifiedBarrierReplayError(
            "durable journal failed independent evaluation-barrier replay"
        ) from error
    return _mint_bfcl_v4_public_v2_verified_decision_barrier(receipt)


__all__ = [
    "BfclV4PublicV2VerifiedBarrierReplayError",
    "verify_bfcl_v4_public_v2_decision_barrier_snapshot",
]
