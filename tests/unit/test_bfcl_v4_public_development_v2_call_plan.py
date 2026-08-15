from __future__ import annotations

from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_GATE_TASK_REFS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_TOPOLOGY_SHA256,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2GateVariant,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2Stage,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
    expected_bfcl_v4_public_development_v2_pure_at_b_allocation,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)


@pytest.fixture(scope="module")
def plan() -> BfclV4PublicDevelopmentV2CampaignPlan:
    return build_bfcl_v4_public_development_v2_campaign_plan()


def test_exact_global_dag_counts_and_fingerprints(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    calls = tuple(node for node in plan.nodes if node.consumes_model_call)
    controls = tuple(node for node in plan.nodes if not node.consumes_model_call)

    assert len(plan.nodes) == 1_098
    assert len(calls) == 1_086
    assert len(controls) == 12
    assert plan.node_schedule_content_sha256 == BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    assert plan.fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    assert tuple(item.schedule_content_sha256 for item in plan.replicate_digests) == (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_SCHEDULE_SHA256
    )
    assert {item.topology_sha256 for item in plan.replicate_digests} == {
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_TOPOLOGY_SHA256
    }
    assert tuple(node.campaign_call_slot for node in calls) == tuple(range(1_086))
    assert tuple(node.node_slot for node in plan.nodes) == tuple(range(1_098))


def test_each_seed_has_exact_362_call_five_arm_budget(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    for replicate_id, outer_seed in zip(
        plan.replicate_ids,
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
        strict=True,
    ):
        nodes = tuple(node for node in plan.nodes if node.replicate_id == replicate_id)
        calls = tuple(node for node in nodes if node.consumes_model_call)
        assert len(nodes) == 366
        assert len(calls) == 362
        assert {node.outer_seed_u64 for node in nodes} == {outer_seed}
        assert Counter(node.arm for node in calls) == Counter(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS
        )
        assert tuple(node.replicate_call_slot for node in calls) == tuple(range(362))
        for arm, count in BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS.items():
            arm_calls = tuple(node for node in calls if node.arm is arm)
            assert tuple(node.arm_slot for node in arm_calls) == tuple(range(count))


def test_each_adaptive_arm_has_three_typed_pipelines_and_exact_inner_budget(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    expected = Counter(
        {
            BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT: 10,
            BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS: 3,
            BfclV4PublicDevelopmentV2NodeKind.PROPOSAL: 3,
            BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT: 30,
            BfclV4PublicDevelopmentV2NodeKind.GATE: 48,
            BfclV4PublicDevelopmentV2NodeKind.HOLDOUT: 16,
        }
    )
    for replicate_id in plan.replicate_ids:
        for arm in (BfclV4PublicDevelopmentV2Arm.SCORE, BfclV4PublicDevelopmentV2Arm.FULL):
            arm_nodes = tuple(
                node for node in plan.nodes if node.replicate_id == replicate_id and node.arm is arm
            )
            calls = tuple(node for node in arm_nodes if node.consumes_model_call)
            assert Counter(node.kind for node in calls) == expected
            assert Counter(
                node.candidate_index
                for node in calls
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT
            ) == Counter({0: 10, 1: 10, 2: 10})
            assert {
                node.pipeline_index
                for node in calls
                if node.kind
                in {
                    BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
                    BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
                }
            } == {0, 1, 2}
            assert all(
                node.typed_atomic_edit_required
                for node in calls
                if node.kind
                in {
                    BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
                    BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
                }
            )
            controls = tuple(node for node in arm_nodes if not node.consumes_model_call)
            assert tuple(node.kind for node in controls) == (
                BfclV4PublicDevelopmentV2NodeKind.NOMINATION,
                BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION,
            )


def test_global_candidate_gate_and_evaluation_barriers_are_causal_and_input_scoped(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    proposals = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
    )
    candidates = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT
    )
    nominations = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.NOMINATION
    )
    gates = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
    )
    decisions = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
    )
    evaluations = tuple(
        node for node in plan.nodes if node.stage is BfclV4PublicDevelopmentV2Stage.EVALUATION
    )

    proposal_ids = {node.node_id for node in proposals}
    nomination_ids = {node.node_id for node in nominations}
    decision_ids = {node.node_id for node in decisions}
    assert all(set(node.depends_on) == proposal_ids for node in candidates)
    assert all(set(node.depends_on) == nomination_ids for node in gates)
    assert all(set(node.depends_on) == decision_ids for node in evaluations)
    assert max(node.node_slot for node in nominations) < min(node.node_slot for node in gates)
    assert max(node.node_slot for node in gates) < min(node.node_slot for node in decisions)
    assert max(node.node_slot for node in decisions) < min(node.node_slot for node in evaluations)

    for node in gates:
        assert len(node.allowed_evidence_from) == 1
        own_nomination = next(
            candidate
            for candidate in nominations
            if candidate.replicate_id == node.replicate_id and candidate.arm is node.arm
        )
        assert node.allowed_evidence_from == (own_nomination.node_id,)
    for node in evaluations:
        if node.arm in {
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2Arm.FULL,
        }:
            assert len(node.allowed_evidence_from) == 1
        else:
            assert node.allowed_evidence_from == ()


def test_provider_seeds_pair_comparisons_and_are_disjoint_across_outer_seeds(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    outer_seed_sets: list[set[int]] = []
    for replicate_id in plan.replicate_ids:
        calls = tuple(
            node
            for node in plan.nodes
            if node.replicate_id == replicate_id and node.consumes_model_call
        )
        outer_seed_sets.append(
            {node.provider_seed_u63 for node in calls if node.provider_seed_u63 is not None}
        )

        fit_groups: defaultdict[tuple[str, int], set[int]] = defaultdict(set)
        for node in calls:
            if node.kind in {
                BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
                BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
            }:
                assert node.task_ref is not None and node.rollout_index is not None
                assert node.provider_seed_u63 is not None
                fit_groups[(node.task_ref, node.rollout_index)].add(node.provider_seed_u63)
        assert all(len(values) == 1 for values in fit_groups.values())

        for kind in (
            BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
            BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        ):
            for pipeline_index in range(3):
                paired = {
                    node.provider_seed_u63
                    for node in calls
                    if node.kind is kind and node.pipeline_index == pipeline_index
                }
                assert len(paired) == 1

        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_GATE_TASK_REFS:
            for rollout_index in range(3):
                paired = {
                    node.provider_seed_u63
                    for node in calls
                    if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
                    and node.task_ref == task_ref
                    and node.rollout_index == rollout_index
                }
                assert len(paired) == 1

        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS:
            paired = {
                node.provider_seed_u63
                for node in calls
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.HOLDOUT
                and node.task_ref == task_ref
            }
            assert len(paired) == 1

        pure_at_b = tuple(
            node
            for node in calls
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
        )
        assert len({node.provider_seed_u63 for node in pure_at_b}) == 110

    assert outer_seed_sets[0].isdisjoint(outer_seed_sets[1])
    assert outer_seed_sets[0].isdisjoint(outer_seed_sets[2])
    assert outer_seed_sets[1].isdisjoint(outer_seed_sets[2])


def test_gate_variants_and_failure_policy_are_exact(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    gates = tuple(
        node for node in plan.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
    )
    assert {node.gate_variant for node in gates} == set(BfclV4PublicDevelopmentV2GateVariant)
    assert len(gates) == 3 * 2 * 4 * 3 * 4
    assert all(
        node.failure_consumes_frozen_slot
        and node.max_provider_attempts == 1
        and not node.retry_allowed
        and not node.backfill_allowed
        for node in plan.nodes
        if node.consumes_model_call
    )
    assert plan.nomination_rule.duplicate_is_invalid is True
    assert plan.nomination_rule.no_op_is_invalid is True
    assert plan.nomination_rule.proposal_provider_failure_is_invalid is True
    assert plan.nomination_rule.no_valid_candidate_nominates_parent_fallback is True
    assert plan.nomination_rule.binary_grader_outcomes_used is True
    assert plan.nomination_rule.target_derived_fit_scores_used is True
    assert plan.nomination_rule.trusted_grader_score_provenance_required is True
    assert plan.nomination_rule.possible_answers_visible_to_nomination_controller is False
    assert plan.promotion_rule.invalid_candidate_forces_parent_fallback is True


def test_pure_at_b_allocation_and_total_label_free_mode_are_frozen(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    assert plan.pure_at_b_allocation == (
        expected_bfcl_v4_public_development_v2_pure_at_b_allocation()
    )
    assert Counter(item.sample_count for item in plan.pure_at_b_allocation) == Counter(
        {7: 14, 6: 2}
    )
    assert sum(item.sample_count for item in plan.pure_at_b_allocation) == 110

    majority = aggregate_bfcl_v4_public_development_v2_pure_at_b(
        ('[{"name":"a"}]', None, '[{"name":"a"}]', '[{"name":"b"}]')
    )
    assert majority.selected_canonical_response == '[{"name":"a"}]'
    assert majority.modal_count == 2
    assert majority.tied_for_mode is False

    tie_input = (None, "response-a", "response-b")
    assert aggregate_bfcl_v4_public_development_v2_pure_at_b(
        tie_input
    ) == aggregate_bfcl_v4_public_development_v2_pure_at_b(tie_input)
    all_failed = aggregate_bfcl_v4_public_development_v2_pure_at_b((None, None, None))
    assert all_failed.selected_no_response is True
    assert all_failed.modal_count == 3
    empty = aggregate_bfcl_v4_public_development_v2_pure_at_b(())
    assert empty.selected_no_response is True
    assert empty.sample_count == empty.modal_count == 0


def test_same_model_budget_and_nonreportable_scope_are_explicit(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    profile = plan.execution_profile
    assert profile.model_route == "qwen36-35b-a3b"
    assert profile.inference.temperature == 0.2
    assert profile.inference.top_p == 0.95
    assert profile.inference.max_output_tokens == 2_048
    assert profile.per_call_total_token_ceiling == 32_768
    assert profile.provider_attempts_per_call == 1
    assert profile.automatic_retries_allowed is False
    assert profile.adaptive_stopping_allowed is False
    assert plan.score_bearing_execution_allowed is False
    assert plan.mutation_catalog_fingerprint == (
        "a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5"
    )
    assert plan.mutation_catalog_fingerprint_bound is True
    assert plan.independent_semantic_audit_release_bound is False
    assert plan.provider_calls_executed is False
    assert plan.model_outputs_present is False
    assert plan.scores_present is False
    assert plan.public_development_only is True
    assert plan.official_bfcl_evidence is False
    assert plan.hidden_test_evidence is False
    assert plan.confirmatory_evidence is False
    assert plan.reportable_result is False

    first_call = next(node for node in plan.nodes if node.consumes_model_call)
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=plan,
        node_id=first_call.node_id,
    )
    assert lineage.campaign_plan_fingerprint == plan.fingerprint
    assert lineage.node_schedule_content_sha256 == plan.node_schedule_content_sha256
    assert lineage.node_id == first_call.node_id
    assert lineage.campaign_call_slot == first_call.campaign_call_slot
    assert lineage.provider_seed_u63 == first_call.provider_seed_u63
    assert lineage.request_payload_present is False
    assert lineage.provider_call_executed is False


def test_campaign_rejects_seed_dependency_order_budget_and_claim_tampering(
    plan: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    seed_tamper = plan.model_dump(mode="python")
    first_call = next(node for node in seed_tamper["nodes"] if node["provider_seed_u63"])
    first_call["provider_seed_u63"] ^= 1
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(seed_tamper, strict=True)

    dependency_tamper = plan.model_dump(mode="python")
    first_gate = next(
        node
        for node in dependency_tamper["nodes"]
        if node["kind"] is BfclV4PublicDevelopmentV2NodeKind.GATE
    )
    first_gate["depends_on"] = first_gate["depends_on"][:-1]
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(dependency_tamper, strict=True)

    order_tamper = plan.model_dump(mode="python")
    order_tamper["nodes"] = (
        order_tamper["nodes"][1],
        order_tamper["nodes"][0],
        *order_tamper["nodes"][2:],
    )
    with pytest.raises(ValidationError, match="node slots"):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(order_tamper, strict=True)

    budget_tamper = plan.model_dump(mode="python")
    budget_tamper["execution_profile"]["per_call_total_token_ceiling"] = 32_767
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(budget_tamper, strict=True)

    claim_tamper = plan.model_dump(mode="python")
    claim_tamper["reportable_result"] = True
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(claim_tamper, strict=True)

    false_label_free_nomination = plan.model_dump(mode="python")
    false_label_free_nomination["nomination_rule"]["grader_labels_or_answers_used"] = False
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(
            false_label_free_nomination,
            strict=True,
        )
