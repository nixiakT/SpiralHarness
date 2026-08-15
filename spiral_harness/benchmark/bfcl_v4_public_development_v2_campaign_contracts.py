"""Campaign-level hashes and validation for the provider-free BFCL v2 DAG."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_ID,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_TOPOLOGY_SHA256,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2ExecutionProfile,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2NominationRule,
    BfclV4PublicDevelopmentV2OuterSeed,
    BfclV4PublicDevelopmentV2PromotionRule,
    BfclV4PublicDevelopmentV2PureAtBAggregationSpec,
    BfclV4PublicDevelopmentV2PureAtBAllocation,
    BfclV4PublicDevelopmentV2Stage,
    expected_bfcl_v4_public_development_v2_pure_at_b_allocation,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256


class BfclV4PublicDevelopmentV2ReplicateDigest(ImmutableModel):
    """Exact per-seed projection inside the global barrier DAG."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=3, strict=True)]
    replicate_id: NonEmptyStr
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed
    schedule_content_sha256: Sha256
    topology_sha256: Sha256

    @model_validator(mode="after")
    def _bind_registration(self) -> Self:
        expected = (
            self.ordinal,
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS[self.ordinal],
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64[self.ordinal],
        )
        if (self.ordinal, self.replicate_id, self.outer_seed_u64) != expected:
            raise ValueError("v2 replicate digest differs from frozen seed registration")
        return self


def bfcl_v4_public_development_v2_node_schedule_sha256(
    nodes: tuple[BfclV4PublicDevelopmentV2DagNode, ...],
) -> str:
    """Hash every exact global DAG node under a versioned domain."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-development-v2-global-dag/v1",
            "manifest_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            "outer_seeds_u64": BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
            "nodes": nodes,
        }
    )


def _normalize_replicate_ref(value: str, replicate_id: str) -> str:
    return value.replace(f"{replicate_id}/", "replicate/")


def _normalize_topology_ref(value: str, replicate_id: str) -> str:
    for registered_id in BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS:
        if value.startswith(f"{registered_id}/"):
            relation = "replicate" if registered_id == replicate_id else "other-replicate"
            return value.replace(f"{registered_id}/", f"{relation}/", 1)
    return value


def bfcl_v4_public_development_v2_replicate_schedule_sha256(
    nodes: tuple[BfclV4PublicDevelopmentV2DagNode, ...],
    *,
    replicate_id: str,
) -> str:
    """Hash one seed's exact projection, retaining provider seeds and barriers."""

    projection = tuple(
        {
            **node.model_dump(
                mode="python",
                exclude={"node_slot", "campaign_call_slot"},
            ),
            "node_id": _normalize_replicate_ref(node.node_id, replicate_id),
            "depends_on": tuple(
                _normalize_replicate_ref(value, replicate_id) for value in node.depends_on
            ),
            "allowed_evidence_from": tuple(
                _normalize_replicate_ref(value, replicate_id)
                for value in node.allowed_evidence_from
            ),
        }
        for node in nodes
        if node.replicate_id == replicate_id
    )
    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-development-v2-replicate-schedule/v1",
            "manifest_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            "replicate_id": replicate_id,
            "nodes": projection,
        }
    )


def bfcl_v4_public_development_v2_replicate_topology_sha256(
    nodes: tuple[BfclV4PublicDevelopmentV2DagNode, ...],
    *,
    replicate_id: str,
) -> str:
    """Project away only replicate identity and provider-seed values."""

    projection = tuple(
        {
            **node.model_dump(
                mode="python",
                exclude={
                    "node_slot",
                    "campaign_call_slot",
                    "replicate_id",
                    "outer_seed_u64",
                    "provider_seed_u63",
                },
            ),
            "node_id": _normalize_topology_ref(node.node_id, replicate_id),
            "depends_on": tuple(
                sorted(_normalize_topology_ref(value, replicate_id) for value in node.depends_on)
            ),
            "allowed_evidence_from": tuple(
                sorted(
                    _normalize_topology_ref(value, replicate_id)
                    for value in node.allowed_evidence_from
                )
            ),
        }
        for node in nodes
        if node.replicate_id == replicate_id
    )
    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-development-v2-replicate-topology/v1",
            "manifest_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            "nodes": projection,
        }
    )


class BfclV4PublicDevelopmentV2CampaignPlan(ImmutableModel):
    """Exact prospective 3-seed, 1,086-call global DAG; no execution evidence."""

    schema_version: Literal["1"] = "1"
    campaign_id: Literal["bfcl-v4-public-development-v2-three-search-seeds-global-dag-v1"] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_ID
    )
    manifest_fingerprint: Literal[
        "3f6e410bd26466381090ced0c6144515281a9653e5d36429685ad898f168eae7"
    ] = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    structural_task_ref_resolution: Literal[
        "zero-based-index-within-split-in-frozen-manifest-order"
    ] = "zero-based-index-within-split-in-frozen-manifest-order"
    outer_seeds_u64: Annotated[tuple[int, ...], Field(min_length=3, max_length=3)]
    replicate_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=3, max_length=3)]
    nodes: Annotated[
        tuple[BfclV4PublicDevelopmentV2DagNode, ...],
        Field(min_length=1_098, max_length=1_098),
    ]
    node_schedule_content_sha256: Sha256
    replicate_digests: Annotated[
        tuple[BfclV4PublicDevelopmentV2ReplicateDigest, ...],
        Field(min_length=3, max_length=3),
    ]
    execution_profile: BfclV4PublicDevelopmentV2ExecutionProfile
    execution_profile_fingerprint: Sha256
    nomination_rule: BfclV4PublicDevelopmentV2NominationRule
    nomination_rule_fingerprint: Sha256
    promotion_rule: BfclV4PublicDevelopmentV2PromotionRule
    promotion_rule_fingerprint: Sha256
    pure_at_b_allocation: Annotated[
        tuple[BfclV4PublicDevelopmentV2PureAtBAllocation, ...],
        Field(min_length=16, max_length=16),
    ]
    pure_at_b_aggregation: BfclV4PublicDevelopmentV2PureAtBAggregationSpec
    pure_at_b_aggregation_fingerprint: Sha256
    replicate_count: Literal[3] = 3
    model_calls_per_replicate: Literal[362] = 362
    total_model_call_ceiling: Literal[1_086] = 1_086
    deterministic_control_nodes_per_replicate: Literal[4] = 4
    total_dag_node_count: Literal[1_098] = 1_098
    pure_calls_per_replicate: Literal[16] = 16
    static_calls_per_replicate: Literal[16] = 16
    score_calls_per_replicate: Literal[110] = 110
    full_calls_per_replicate: Literal[110] = 110
    pure_at_b_calls_per_replicate: Literal[110] = 110
    all_seeds_candidates_and_nominations_frozen_before_any_gate: Literal[True] = True
    all_seeds_gates_and_both_arm_decisions_frozen_before_any_evaluation: Literal[True] = True
    candidate_proposal_failure_consumes_slot_without_retry_or_backfill: Literal[True] = True
    duplicate_no_op_or_invalid_candidate_consumes_slot: Literal[True] = True
    invalid_candidate_forces_parent_fallback: Literal[True] = True
    pure_at_b_base_samples_per_task: Literal[6] = 6
    pure_at_b_remainder_task_count: Literal[14] = 14
    pure_at_b_remainder_calls: Literal[14] = 14
    pure_at_b_total_calls_per_replicate: Literal[110] = 110
    typed_atomic_mutation_catalog_required_before_execution: Literal[True] = True
    mutation_catalog_fingerprint: Literal[
        "a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5"
    ] = BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.fingerprint
    mutation_catalog_fingerprint_bound: Literal[True] = True
    every_model_request_must_bind_campaign_plan_fingerprint: Literal[True] = True
    independent_semantic_audit_release_bound: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    provider_calls_executed: Literal[False] = False
    model_outputs_present: Literal[False] = False
    scores_present: Literal[False] = False
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    confirmatory_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_campaign(self) -> Self:
        if self.outer_seeds_u64 != BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64:
            raise ValueError("v2 campaign outer seeds or their order changed")
        if self.replicate_ids != BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS:
            raise ValueError("v2 campaign replicate IDs or their order changed")
        if self.execution_profile != BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE or (
            self.execution_profile_fingerprint != self.execution_profile.fingerprint
        ):
            raise ValueError("v2 campaign execution profile changed")
        if self.nomination_rule != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE or (
            self.nomination_rule_fingerprint != self.nomination_rule.fingerprint
        ):
            raise ValueError("v2 nomination rule changed")
        if self.promotion_rule != BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE or (
            self.promotion_rule_fingerprint != self.promotion_rule.fingerprint
        ):
            raise ValueError("v2 promotion rule changed")
        if (
            self.pure_at_b_aggregation != BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION
            or self.pure_at_b_aggregation_fingerprint != self.pure_at_b_aggregation.fingerprint
        ):
            raise ValueError("v2 PURE@B aggregation changed")

        expected_allocation = expected_bfcl_v4_public_development_v2_pure_at_b_allocation()
        if self.pure_at_b_allocation != expected_allocation:
            raise ValueError("v2 PURE@B prospective 6-plus-remainder allocation changed")
        if (
            sum(item.sample_count for item in self.pure_at_b_allocation) != 110
            or sum(item.receives_remainder_sample for item in self.pure_at_b_allocation) != 14
        ):
            raise ValueError("v2 PURE@B allocation does not total 110 across fourteen remainders")

        if tuple(node.node_slot for node in self.nodes) != tuple(range(1_098)):
            raise ValueError("v2 global DAG node slots must be the exact sequence 0..1097")
        calls = tuple(node for node in self.nodes if node.consumes_model_call)
        if tuple(node.campaign_call_slot for node in calls) != tuple(range(1_086)):
            raise ValueError("v2 campaign call slots must be the exact sequence 0..1085")
        if len({node.node_id for node in self.nodes}) != 1_098:
            raise ValueError("v2 global DAG node IDs must be unique")
        known = {node.node_id for node in self.nodes}
        if any(ref not in known for node in self.nodes for ref in node.depends_on):
            raise ValueError("v2 global DAG dependency references an unknown node")
        node_slots = {node.node_id: node.node_slot for node in self.nodes}
        if any(node_slots[ref] >= node.node_slot for node in self.nodes for ref in node.depends_on):
            raise ValueError("v2 global DAG dependencies must point strictly backward")

        expected_stage_order = tuple(sorted(self.nodes, key=lambda node: node.stage.value))
        if self.nodes != expected_stage_order:
            raise ValueError("v2 global DAG stages differ from the frozen global barriers")
        nomination_slots = tuple(
            node.node_slot
            for node in self.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.NOMINATION
        )
        gate_slots = tuple(
            node.node_slot
            for node in self.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
        )
        decision_slots = tuple(
            node.node_slot
            for node in self.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        )
        evaluation_slots = tuple(
            node.node_slot
            for node in self.nodes
            if node.stage is BfclV4PublicDevelopmentV2Stage.EVALUATION
        )
        if not (
            len(nomination_slots) == 6
            and len(gate_slots) == 288
            and len(decision_slots) == 6
            and len(evaluation_slots) == 522
            and max(nomination_slots) < min(gate_slots)
            and max(gate_slots) < min(decision_slots)
            and max(decision_slots) < min(evaluation_slots)
        ):
            raise ValueError("v2 nomination/GATE/decision/evaluation global barriers changed")

        stage_call_counts = Counter(node.stage for node in calls)
        expected_stage_call_counts = {
            BfclV4PublicDevelopmentV2Stage.PARENT_FIT: 60,
            BfclV4PublicDevelopmentV2Stage.DIAGNOSIS: 18,
            BfclV4PublicDevelopmentV2Stage.PROPOSAL: 18,
            BfclV4PublicDevelopmentV2Stage.CANDIDATE_FIT: 180,
            BfclV4PublicDevelopmentV2Stage.GATE: 288,
            BfclV4PublicDevelopmentV2Stage.EVALUATION: 522,
        }
        if dict(stage_call_counts) != expected_stage_call_counts:
            raise ValueError("v2 stage call counts differ from the 1,086-call registration")

        expected_digests: list[BfclV4PublicDevelopmentV2ReplicateDigest] = []
        topology_hashes: set[str] = set()
        provider_seed_sets: list[set[int]] = []
        for ordinal, (replicate_id, outer_seed) in enumerate(
            zip(self.replicate_ids, self.outer_seeds_u64, strict=True)
        ):
            replicate_nodes = tuple(
                node for node in self.nodes if node.replicate_id == replicate_id
            )
            replicate_calls = tuple(node for node in replicate_nodes if node.consumes_model_call)
            if len(replicate_nodes) != 366 or len(replicate_calls) != 362:
                raise ValueError("v2 replicate must contain 362 calls and four control nodes")
            if {node.outer_seed_u64 for node in replicate_nodes} != {outer_seed}:
                raise ValueError("v2 replicate node outer seed changed")
            if tuple(node.replicate_node_slot for node in replicate_nodes) != tuple(range(366)):
                raise ValueError("v2 replicate node slots must be the exact sequence 0..365")
            if tuple(node.replicate_call_slot for node in replicate_calls) != tuple(range(362)):
                raise ValueError("v2 replicate call slots must be the exact sequence 0..361")

            for arm, expected_count in BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS.items():
                arm_calls = tuple(node for node in replicate_calls if node.arm is arm)
                if len(arm_calls) != expected_count or tuple(
                    node.arm_slot for node in arm_calls
                ) != tuple(range(expected_count)):
                    raise ValueError("v2 per-arm call count or within-arm slot order changed")

            schedule_hash = bfcl_v4_public_development_v2_replicate_schedule_sha256(
                self.nodes,
                replicate_id=replicate_id,
            )
            topology_hash = bfcl_v4_public_development_v2_replicate_topology_sha256(
                self.nodes,
                replicate_id=replicate_id,
            )
            if schedule_hash != BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_SCHEDULE_SHA256[ordinal]:
                raise ValueError("v2 replicate schedule differs from its frozen digest")
            if topology_hash != BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_TOPOLOGY_SHA256:
                raise ValueError("v2 replicate topology differs from its frozen digest")
            topology_hashes.add(topology_hash)
            expected_digests.append(
                BfclV4PublicDevelopmentV2ReplicateDigest(
                    ordinal=ordinal,
                    replicate_id=replicate_id,
                    outer_seed_u64=outer_seed,
                    schedule_content_sha256=schedule_hash,
                    topology_sha256=topology_hash,
                )
            )
            provider_seed_sets.append(
                {
                    node.provider_seed_u63
                    for node in replicate_calls
                    if node.provider_seed_u63 is not None
                }
            )

        if self.replicate_digests != tuple(expected_digests):
            raise ValueError("v2 per-replicate schedule digests changed")
        if len(topology_hashes) != 1:
            raise ValueError("v2 replicate topology differs across outer seeds")
        if any(
            not provider_seed_sets[left].isdisjoint(provider_seed_sets[right])
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("v2 provider-seed sets overlap across outer seeds")

        computed_schedule = bfcl_v4_public_development_v2_node_schedule_sha256(self.nodes)
        if (
            self.node_schedule_content_sha256 != computed_schedule
            or computed_schedule != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
        ):
            raise ValueError("v2 node schedule digest differs from exact DAG content")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicDevelopmentV2NodeRequestLineage(ImmutableModel):
    """Provider-free lineage that must accompany one materialized model request."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    node_id: NonEmptyStr
    node_reference_sha256: Sha256
    campaign_call_slot: Annotated[int, Field(ge=0, lt=1_086, strict=True)]
    provider_seed_u63: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
    mutation_catalog_fingerprint: Literal[
        "a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5"
    ] = BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.fingerprint
    request_payload_present: Literal[False] = False
    provider_call_executed: Literal[False] = False


def derive_bfcl_v4_public_development_v2_node_request_lineage(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node_id: str,
) -> BfclV4PublicDevelopmentV2NodeRequestLineage:
    """Bind a call node to the exact plan before any request is materialized."""

    checked = BfclV4PublicDevelopmentV2CampaignPlan.model_validate(campaign, strict=True)
    matches = tuple(node for node in checked.nodes if node.node_id == node_id)
    if len(matches) != 1 or not matches[0].consumes_model_call:
        raise ValueError("v2 request lineage requires exactly one registered model-call node")
    node = matches[0]
    assert node.campaign_call_slot is not None
    assert node.provider_seed_u63 is not None
    return BfclV4PublicDevelopmentV2NodeRequestLineage(
        campaign_plan_fingerprint=checked.fingerprint,
        node_schedule_content_sha256=checked.node_schedule_content_sha256,
        node_id=node.node_id,
        node_reference_sha256=canonical_sha256(node),
        campaign_call_slot=node.campaign_call_slot,
        provider_seed_u63=node.provider_seed_u63,
    )


__all__ = [name for name in globals() if name.startswith("Bfcl") or name.startswith("bfcl")]
