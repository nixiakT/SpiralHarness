"""Build the provider-free BFCL V4 public-development v2 global call DAG."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_FIT_TASK_REFS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_GATE_TASK_REFS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROVIDER_SEED_DOMAIN,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2FeedbackView,
    BfclV4PublicDevelopmentV2GateVariant,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2OuterSeed,
    BfclV4PublicDevelopmentV2Stage,
    bfcl_v4_public_development_v2_node_id,
    expected_bfcl_v4_public_development_v2_pure_at_b_allocation,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    BfclV4PublicDevelopmentV2ReplicateDigest,
    bfcl_v4_public_development_v2_node_schedule_sha256,
    bfcl_v4_public_development_v2_replicate_schedule_sha256,
    bfcl_v4_public_development_v2_replicate_topology_sha256,
)
from spiral_harness.core.canonical import canonical_json_bytes

_ADAPTIVE_ARMS = (
    BfclV4PublicDevelopmentV2Arm.SCORE,
    BfclV4PublicDevelopmentV2Arm.FULL,
)
_HOLDOUT_ARMS = (
    BfclV4PublicDevelopmentV2Arm.PURE,
    BfclV4PublicDevelopmentV2Arm.STATIC,
    BfclV4PublicDevelopmentV2Arm.SCORE,
    BfclV4PublicDevelopmentV2Arm.FULL,
)
_GATE_VARIANTS = tuple(BfclV4PublicDevelopmentV2GateVariant)
_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class _DraftNode:
    replicate_id: str
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed
    arm: BfclV4PublicDevelopmentV2Arm
    stage: BfclV4PublicDevelopmentV2Stage
    kind: BfclV4PublicDevelopmentV2NodeKind
    task_ref: str | None
    rollout_index: int | None
    sample_index: int | None
    pipeline_index: int | None
    candidate_index: int | None
    gate_variant: BfclV4PublicDevelopmentV2GateVariant | None
    harness_variant: str
    feedback_view: BfclV4PublicDevelopmentV2FeedbackView
    seed_coordinate: tuple[object, ...] | None
    depends_on: tuple[str, ...]
    allowed_evidence_from: tuple[str, ...]
    grader_feedback_available: bool
    typed_atomic_edit_required: bool

    @property
    def node_id(self) -> str:
        return bfcl_v4_public_development_v2_node_id(
            replicate_id=self.replicate_id,
            arm=self.arm,
            kind=self.kind,
            task_ref=self.task_ref,
            rollout_index=self.rollout_index,
            sample_index=self.sample_index,
            pipeline_index=self.pipeline_index,
            candidate_index=self.candidate_index,
            gate_variant=self.gate_variant,
        )


def _provider_seed_u63(
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed,
    coordinate: tuple[object, ...],
) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROVIDER_SEED_DOMAIN,
                "outer_seed_u64": outer_seed_u64,
                "coordinate": coordinate,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _feedback(arm: BfclV4PublicDevelopmentV2Arm) -> BfclV4PublicDevelopmentV2FeedbackView:
    return {
        BfclV4PublicDevelopmentV2Arm.SCORE: (BfclV4PublicDevelopmentV2FeedbackView.SCORE_ONLY),
        BfclV4PublicDevelopmentV2Arm.FULL: (
            BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL
        ),
    }[arm]


def _draft(
    *,
    replicate_id: str,
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed,
    arm: BfclV4PublicDevelopmentV2Arm,
    stage: BfclV4PublicDevelopmentV2Stage,
    kind: BfclV4PublicDevelopmentV2NodeKind,
    harness_variant: str,
    task_ref: str | None = None,
    rollout_index: int | None = None,
    sample_index: int | None = None,
    pipeline_index: int | None = None,
    candidate_index: int | None = None,
    gate_variant: BfclV4PublicDevelopmentV2GateVariant | None = None,
    feedback_view: BfclV4PublicDevelopmentV2FeedbackView = (
        BfclV4PublicDevelopmentV2FeedbackView.NONE
    ),
    seed_coordinate: tuple[object, ...] | None = None,
    depends_on: tuple[str, ...] = (),
    allowed_evidence_from: tuple[str, ...] = (),
    grader_feedback_available: bool = False,
    typed_atomic_edit_required: bool = False,
) -> _DraftNode:
    return _DraftNode(
        replicate_id=replicate_id,
        outer_seed_u64=outer_seed_u64,
        arm=arm,
        stage=stage,
        kind=kind,
        task_ref=task_ref,
        rollout_index=rollout_index,
        sample_index=sample_index,
        pipeline_index=pipeline_index,
        candidate_index=candidate_index,
        gate_variant=gate_variant,
        harness_variant=harness_variant,
        feedback_view=feedback_view,
        seed_coordinate=seed_coordinate,
        depends_on=depends_on,
        allowed_evidence_from=allowed_evidence_from,
        grader_feedback_available=grader_feedback_available,
        typed_atomic_edit_required=typed_atomic_edit_required,
    )


def _registered_replicates() -> tuple[tuple[str, BfclV4PublicDevelopmentV2OuterSeed], ...]:
    return tuple(
        (replicate_id, outer_seed)
        for replicate_id, outer_seed in zip(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS,
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
            strict=True,
        )
    )


def _build_drafts() -> tuple[_DraftNode, ...]:
    parents: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_FIT_TASK_REFS:
            for rollout_index in range(2):
                for arm in _ADAPTIVE_ARMS:
                    parents.append(
                        _draft(
                            replicate_id=replicate_id,
                            outer_seed_u64=outer_seed,
                            arm=arm,
                            stage=BfclV4PublicDevelopmentV2Stage.PARENT_FIT,
                            kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
                            task_ref=task_ref,
                            rollout_index=rollout_index,
                            harness_variant="parent",
                            seed_coordinate=("fit-evaluation", task_ref, rollout_index),
                        )
                    )
    all_parent_ids = tuple(node.node_id for node in parents)

    diagnoses: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for pipeline_index in range(3):
            for arm in _ADAPTIVE_ARMS:
                own_parent_ids = tuple(
                    node.node_id
                    for node in parents
                    if node.replicate_id == replicate_id and node.arm is arm
                )
                diagnoses.append(
                    _draft(
                        replicate_id=replicate_id,
                        outer_seed_u64=outer_seed,
                        arm=arm,
                        stage=BfclV4PublicDevelopmentV2Stage.DIAGNOSIS,
                        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
                        pipeline_index=pipeline_index,
                        harness_variant=f"diagnosis-pipeline-{pipeline_index}",
                        feedback_view=_feedback(arm),
                        seed_coordinate=("controller", "diagnosis", pipeline_index),
                        depends_on=all_parent_ids,
                        allowed_evidence_from=own_parent_ids,
                        grader_feedback_available=True,
                    )
                )
    all_diagnosis_ids = tuple(node.node_id for node in diagnoses)

    proposals: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for pipeline_index in range(3):
            for arm in _ADAPTIVE_ARMS:
                own_diagnosis = next(
                    node.node_id
                    for node in diagnoses
                    if node.replicate_id == replicate_id
                    and node.arm is arm
                    and node.pipeline_index == pipeline_index
                )
                proposals.append(
                    _draft(
                        replicate_id=replicate_id,
                        outer_seed_u64=outer_seed,
                        arm=arm,
                        stage=BfclV4PublicDevelopmentV2Stage.PROPOSAL,
                        kind=BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
                        pipeline_index=pipeline_index,
                        candidate_index=pipeline_index,
                        harness_variant=f"typed-atomic-proposal-{pipeline_index}",
                        feedback_view=_feedback(arm),
                        seed_coordinate=("controller", "proposal", pipeline_index),
                        depends_on=all_diagnosis_ids,
                        allowed_evidence_from=(own_diagnosis,),
                        grader_feedback_available=True,
                        typed_atomic_edit_required=True,
                    )
                )
    all_proposal_ids = tuple(node.node_id for node in proposals)

    candidates: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for candidate_index in range(3):
            for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_FIT_TASK_REFS:
                for rollout_index in range(2):
                    for arm in _ADAPTIVE_ARMS:
                        own_proposal = next(
                            node.node_id
                            for node in proposals
                            if node.replicate_id == replicate_id
                            and node.arm is arm
                            and node.candidate_index == candidate_index
                        )
                        candidates.append(
                            _draft(
                                replicate_id=replicate_id,
                                outer_seed_u64=outer_seed,
                                arm=arm,
                                stage=BfclV4PublicDevelopmentV2Stage.CANDIDATE_FIT,
                                kind=BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
                                task_ref=task_ref,
                                rollout_index=rollout_index,
                                pipeline_index=candidate_index,
                                candidate_index=candidate_index,
                                harness_variant=(
                                    f"typed-candidate-{candidate_index}-or-parent-fallback"
                                ),
                                seed_coordinate=("fit-evaluation", task_ref, rollout_index),
                                depends_on=all_proposal_ids,
                                allowed_evidence_from=(own_proposal,),
                                typed_atomic_edit_required=True,
                            )
                        )
    all_candidate_ids = tuple(node.node_id for node in candidates)

    nominations: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for arm in _ADAPTIVE_ARMS:
            own_parents = tuple(
                node.node_id
                for node in parents
                if node.replicate_id == replicate_id and node.arm is arm
            )
            own_proposals = tuple(
                node.node_id
                for node in proposals
                if node.replicate_id == replicate_id and node.arm is arm
            )
            own_candidates = tuple(
                node.node_id
                for node in candidates
                if node.replicate_id == replicate_id and node.arm is arm
            )
            own_evidence = (*own_parents, *own_proposals, *own_candidates)
            nominations.append(
                _draft(
                    replicate_id=replicate_id,
                    outer_seed_u64=outer_seed,
                    arm=arm,
                    stage=BfclV4PublicDevelopmentV2Stage.NOMINATION,
                    kind=BfclV4PublicDevelopmentV2NodeKind.NOMINATION,
                    harness_variant="deterministic-fit-nomination-or-parent-fallback",
                    depends_on=_unique((*all_candidate_ids, *own_parents, *own_proposals)),
                    allowed_evidence_from=own_evidence,
                )
            )
    all_nomination_ids = tuple(node.node_id for node in nominations)

    gates: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_GATE_TASK_REFS:
            for rollout_index in range(3):
                for gate_variant in _GATE_VARIANTS:
                    for arm in _ADAPTIVE_ARMS:
                        own_nomination = next(
                            node.node_id
                            for node in nominations
                            if node.replicate_id == replicate_id and node.arm is arm
                        )
                        gates.append(
                            _draft(
                                replicate_id=replicate_id,
                                outer_seed_u64=outer_seed,
                                arm=arm,
                                stage=BfclV4PublicDevelopmentV2Stage.GATE,
                                kind=BfclV4PublicDevelopmentV2NodeKind.GATE,
                                task_ref=task_ref,
                                rollout_index=rollout_index,
                                gate_variant=gate_variant,
                                harness_variant=gate_variant.value,
                                seed_coordinate=("gate-evaluation", task_ref, rollout_index),
                                depends_on=all_nomination_ids,
                                allowed_evidence_from=(own_nomination,),
                            )
                        )
    all_gate_ids = tuple(node.node_id for node in gates)

    decisions: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for arm in _ADAPTIVE_ARMS:
            own_nomination = next(
                node.node_id
                for node in nominations
                if node.replicate_id == replicate_id and node.arm is arm
            )
            own_gates = tuple(
                node.node_id
                for node in gates
                if node.replicate_id == replicate_id and node.arm is arm
            )
            decisions.append(
                _draft(
                    replicate_id=replicate_id,
                    outer_seed_u64=outer_seed,
                    arm=arm,
                    stage=BfclV4PublicDevelopmentV2Stage.DECISION,
                    kind=BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION,
                    harness_variant="deterministic-promote-or-parent-fallback",
                    depends_on=(*all_gate_ids, own_nomination),
                    allowed_evidence_from=(own_nomination, *own_gates),
                )
            )
    all_decision_ids = tuple(node.node_id for node in decisions)

    evaluations: list[_DraftNode] = []
    for replicate_id, outer_seed in _registered_replicates():
        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS:
            for arm in _HOLDOUT_ARMS:
                if arm is BfclV4PublicDevelopmentV2Arm.PURE:
                    harness_variant = "bare"
                elif arm is BfclV4PublicDevelopmentV2Arm.STATIC:
                    harness_variant = "static-frozen"
                else:
                    harness_variant = "selected-or-parent-fallback"
                own_decision = tuple(
                    node.node_id
                    for node in decisions
                    if node.replicate_id == replicate_id and node.arm is arm
                )
                evaluations.append(
                    _draft(
                        replicate_id=replicate_id,
                        outer_seed_u64=outer_seed,
                        arm=arm,
                        stage=BfclV4PublicDevelopmentV2Stage.EVALUATION,
                        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
                        task_ref=task_ref,
                        rollout_index=0,
                        harness_variant=harness_variant,
                        seed_coordinate=("holdout-evaluation", task_ref, 0),
                        depends_on=all_decision_ids,
                        allowed_evidence_from=own_decision,
                    )
                )

        for allocation in expected_bfcl_v4_public_development_v2_pure_at_b_allocation():
            for sample_index in range(allocation.sample_count):
                evaluations.append(
                    _draft(
                        replicate_id=replicate_id,
                        outer_seed_u64=outer_seed,
                        arm=BfclV4PublicDevelopmentV2Arm.PURE_AT_B,
                        stage=BfclV4PublicDevelopmentV2Stage.EVALUATION,
                        kind=BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
                        task_ref=allocation.task_ref,
                        sample_index=sample_index,
                        harness_variant=f"bare-sample-{sample_index}",
                        seed_coordinate=(
                            "pure-at-b-evaluation",
                            allocation.task_ref,
                            sample_index,
                        ),
                        depends_on=all_decision_ids,
                    )
                )

    return (
        *parents,
        *diagnoses,
        *proposals,
        *candidates,
        *nominations,
        *gates,
        *decisions,
        *evaluations,
    )


def _finalize_nodes(drafts: tuple[_DraftNode, ...]) -> tuple[BfclV4PublicDevelopmentV2DagNode, ...]:
    campaign_call_slot = 0
    replicate_node_slots: defaultdict[str, int] = defaultdict(int)
    replicate_call_slots: defaultdict[str, int] = defaultdict(int)
    arm_slots: defaultdict[tuple[str, BfclV4PublicDevelopmentV2Arm], int] = defaultdict(int)
    nodes: list[BfclV4PublicDevelopmentV2DagNode] = []

    for node_slot, draft in enumerate(drafts):
        consumes_call = draft.seed_coordinate is not None
        replicate_node_slot = replicate_node_slots[draft.replicate_id]
        replicate_node_slots[draft.replicate_id] += 1
        if consumes_call:
            current_campaign_call_slot: int | None = campaign_call_slot
            campaign_call_slot += 1
            current_replicate_call_slot: int | None = replicate_call_slots[draft.replicate_id]
            replicate_call_slots[draft.replicate_id] += 1
            arm_key = (draft.replicate_id, draft.arm)
            current_arm_slot: int | None = arm_slots[arm_key]
            arm_slots[arm_key] += 1
            provider_seed: int | None = _provider_seed_u63(
                draft.outer_seed_u64,
                draft.seed_coordinate,
            )
            max_provider_attempts: int | None = 1
        else:
            current_campaign_call_slot = None
            current_replicate_call_slot = None
            current_arm_slot = None
            provider_seed = None
            max_provider_attempts = None

        nodes.append(
            BfclV4PublicDevelopmentV2DagNode(
                node_slot=node_slot,
                campaign_call_slot=current_campaign_call_slot,
                replicate_node_slot=replicate_node_slot,
                replicate_call_slot=current_replicate_call_slot,
                arm_slot=current_arm_slot,
                node_id=draft.node_id,
                replicate_id=draft.replicate_id,
                outer_seed_u64=draft.outer_seed_u64,
                arm=draft.arm,
                stage=draft.stage,
                kind=draft.kind,
                task_ref=draft.task_ref,
                rollout_index=draft.rollout_index,
                sample_index=draft.sample_index,
                pipeline_index=draft.pipeline_index,
                candidate_index=draft.candidate_index,
                gate_variant=draft.gate_variant,
                harness_variant=draft.harness_variant,
                feedback_view=draft.feedback_view,
                provider_seed_u63=provider_seed,
                depends_on=draft.depends_on,
                allowed_evidence_from=draft.allowed_evidence_from,
                consumes_model_call=consumes_call,
                max_provider_attempts=max_provider_attempts,
                failure_consumes_frozen_slot=consumes_call,
                grader_feedback_available=draft.grader_feedback_available,
                typed_atomic_edit_required=draft.typed_atomic_edit_required,
            )
        )
    return tuple(nodes)


def build_bfcl_v4_public_development_v2_campaign_plan() -> BfclV4PublicDevelopmentV2CampaignPlan:
    """Build all three seeds and global barriers without invoking a provider."""

    nodes = _finalize_nodes(_build_drafts())
    replicate_digests = tuple(
        BfclV4PublicDevelopmentV2ReplicateDigest(
            ordinal=ordinal,
            replicate_id=replicate_id,
            outer_seed_u64=outer_seed,
            schedule_content_sha256=(
                bfcl_v4_public_development_v2_replicate_schedule_sha256(
                    nodes,
                    replicate_id=replicate_id,
                )
            ),
            topology_sha256=(
                bfcl_v4_public_development_v2_replicate_topology_sha256(
                    nodes,
                    replicate_id=replicate_id,
                )
            ),
        )
        for ordinal, (replicate_id, outer_seed) in enumerate(_registered_replicates())
    )
    campaign = BfclV4PublicDevelopmentV2CampaignPlan(
        manifest_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
        outer_seeds_u64=BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
        replicate_ids=BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS,
        nodes=nodes,
        node_schedule_content_sha256=(bfcl_v4_public_development_v2_node_schedule_sha256(nodes)),
        replicate_digests=replicate_digests,
        execution_profile=BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE,
        execution_profile_fingerprint=(BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE.fingerprint),
        nomination_rule=BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE,
        nomination_rule_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE.fingerprint,
        promotion_rule=BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE,
        promotion_rule_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE.fingerprint,
        pure_at_b_allocation=(expected_bfcl_v4_public_development_v2_pure_at_b_allocation()),
        pure_at_b_aggregation=BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION,
        pure_at_b_aggregation_fingerprint=(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
        ),
    )
    if campaign.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT:
        raise RuntimeError("v2 campaign content differs from its prospective fingerprint")
    return campaign


__all__ = ["build_bfcl_v4_public_development_v2_campaign_plan"]
