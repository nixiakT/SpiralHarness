from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.confirmatory_arms import (
    AdaptiveExecutionCeilings,
    AdaptiveProtocolCommitments,
    ConfirmatoryFourArmDesign,
    EvidenceComputationMode,
    ExAnteAdaptiveTopology,
    FaultFactorialCell,
    FaultFactorialCellPlan,
    FaultFactorialDesign,
    FaultFactorialProfile,
    HarnessMode,
    ModelMediatedRole,
    ModelRoleCeiling,
    OptimizerFeedbackMode,
    PromotionRule,
    RealTaskArm,
    RealTaskArmPlan,
    RealTaskArmProfile,
    SearchBudgetScope,
    SearchMode,
    make_confirmatory_four_arm_design,
    make_fault_factorial_design,
    make_fault_factorial_profile,
    make_real_task_arm_profile,
)
from spiral_harness.experiments.confirmatory_pure_at_b import (
    AggregationMethod,
    PureAtBAggregationRule,
    PureAtBEvaluationUnitAllocation,
    PureAtBPlan,
    PureAtBSampleAllocation,
    make_pure_at_b_plan,
)
from spiral_harness.experiments.confirmatory_resources import (
    CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
    AdaptiveConditionContext,
    ConfirmatoryTaskSplitManifest,
    TaskSplitEvaluationUnit,
    TaskSplitTask,
)
from spiral_harness.experiments.confirmatory_roster_artifacts import (
    ConfirmatoryTaskSplitArtifactError,
    load_confirmatory_task_split,
    publish_confirmatory_task_split,
    verify_confirmatory_task_split,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def digest(label: str) -> str:
    return canonical_sha256({"test": "confirmatory-arms", "label": label})


def task_manifest_ref(task_id: str) -> ArtifactRef:
    return ArtifactRef(
        sha256=digest(f"task-manifest:{task_id}"),
        size=1_000,
        media_type="application/vnd.spiral-harness.task-manifest.v1+json",
    )


def evaluation_unit_ref(unit_id: str) -> ArtifactRef:
    return ArtifactRef(
        sha256=digest(f"evaluation-unit:{unit_id}"),
        size=2_000,
        media_type="application/vnd.spiral-harness.evaluation-unit.v1+json",
    )


def task_split_manifest() -> ConfirmatoryTaskSplitManifest:
    return ConfirmatoryTaskSplitManifest(
        split_id="confirmatory-real-task-v1",
        tasks=tuple(
            TaskSplitTask(
                task_id=task_id,
                task_manifest_ref=task_manifest_ref(task_id),
                evaluation_units=(
                    TaskSplitEvaluationUnit(
                        evaluation_unit_id=f"{task_id}:seed-0",
                        evaluation_unit_ref=evaluation_unit_ref(f"{task_id}:seed-0"),
                    ),
                ),
            )
            for task_id in ("gsm8k", "bbh")
        ),
    )


def protocol_commitments() -> AdaptiveProtocolCommitments:
    split = task_split_manifest()
    return AdaptiveProtocolCommitments(
        model_spec_fingerprint=digest("exact-model-spec"),
        solver_config_fingerprint=digest("solver-config"),
        optimizer_config_fingerprint=digest("optimizer-config"),
        seed_harness_ref=ArtifactRef(
            sha256=digest("seed-harness"),
            size=100,
            media_type="application/vnd.spiral-harness.harness-manifest.v2+json",
        ),
        mutation_policy_ref=ArtifactRef(
            sha256=digest("mutation-policy"),
            size=200,
            media_type=CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
        ),
        task_split_fingerprint=split.fingerprint,
        task_split_manifest=split,
        task_split_manifest_ref=split.artifact_ref,
        seed_schedule_fingerprint=digest("seed-schedule"),
        candidate_parser_fingerprint=digest("candidate-parser"),
        grader_fingerprint=digest("grader"),
        query_dag_fingerprint=digest("query-dag"),
        retry_policy_fingerprint=digest("retry-policy"),
    )


def role_ceilings() -> tuple[ModelRoleCeiling, ...]:
    calls = {
        ModelMediatedRole.SOLVER: 80,
        ModelMediatedRole.DIAGNOSER: 4,
        ModelMediatedRole.PROPOSER: 8,
        ModelMediatedRole.MATERIALIZER: 0,
        ModelMediatedRole.RANKER: 4,
        ModelMediatedRole.NOMINATOR: 4,
        ModelMediatedRole.JUDGE: 0,
        ModelMediatedRole.GRADER: 0,
    }
    return tuple(
        ModelRoleCeiling(
            role=role,
            max_calls=calls[role],
            max_tokens=calls[role] * 1_000,
        )
        for role in reversed(tuple(ModelMediatedRole))
    )


def ceilings(**updates: object) -> AdaptiveExecutionCeilings:
    roles = role_ceilings()
    values: dict[str, object] = {
        "max_rounds": 4,
        "max_proposals_per_round": 2,
        "max_total_proposals": 8,
        "max_total_nominations": 4,
        "max_gate_queries": 4,
        "max_feedback_releases": 4,
        "max_evaluations": 80,
        "max_attempts_per_model_call": 2,
        "token_ceiling_per_model_call": 1_000,
        "role_model_calls": roles,
        "max_total_model_calls": sum(item.max_calls for item in roles),
        "max_total_tokens": sum(item.max_tokens for item in roles),
        "max_total_model_attempts": (sum(item.max_calls for item in roles) * 2),
        "max_total_attempt_tokens": (sum(item.max_tokens for item in roles) * 2),
        "max_wall_time_seconds": 3_600.0,
        "max_cost_usd": 20.0,
    }
    values.update(updates)
    return AdaptiveExecutionCeilings(**values)


def topology(*, conditions: int) -> ExAnteAdaptiveTopology:
    context_ids = {
        2: (AdaptiveConditionContext.SCORE, AdaptiveConditionContext.FULL),
        4: (
            AdaptiveConditionContext.SS,
            AdaptiveConditionContext.MS,
            AdaptiveConditionContext.SM,
            AdaptiveConditionContext.MM,
        ),
    }.get(conditions, tuple(AdaptiveConditionContext)[:conditions])
    return ExAnteAdaptiveTopology(
        condition_context_count=conditions,
        condition_context_ids=context_ids,
        protocol_commitments=protocol_commitments(),
    )


def four_arm_design() -> ConfirmatoryFourArmDesign:
    return make_confirmatory_four_arm_design(
        adaptive_topology=topology(conditions=2),
        adaptive_ceilings=ceilings(),
    )


def factorial_design() -> FaultFactorialDesign:
    return make_fault_factorial_design(
        adaptive_topology=topology(conditions=4),
        adaptive_ceilings=ceilings(),
    )


def test_real_task_and_fault_factorial_enums_are_separate_domains() -> None:
    assert set(RealTaskArm).isdisjoint(FaultFactorialCell)

    with pytest.raises(TypeError, match="exact RealTaskArm"):
        make_real_task_arm_profile(FaultFactorialCell.SS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact FaultFactorialCell"):
        make_fault_factorial_profile(RealTaskArm.FULL)  # type: ignore[arg-type]


def test_four_arm_design_encodes_joint_full_score_treatment_without_false_attestation() -> None:
    design = four_arm_design()

    assert tuple(plan.arm for plan in design.arms) == (
        RealTaskArm.PURE,
        RealTaskArm.STATIC,
        RealTaskArm.SCORE,
        RealTaskArm.FULL,
    )
    pure = design.arm(RealTaskArm.PURE)
    static = design.arm(RealTaskArm.STATIC)
    score = design.arm(RealTaskArm.SCORE)
    full = design.arm(RealTaskArm.FULL)

    assert pure.profile.harness_mode is HarnessMode.PROVIDER_MINIMAL
    assert static.profile.harness_mode is HarnessMode.FROZEN_SEED
    assert pure.adaptive_topology is static.adaptive_topology is None
    assert pure.adaptive_ceilings is static.adaptive_ceilings is None
    assert pure.profile.search_budget_scope is SearchBudgetScope.NOT_APPLICABLE
    assert static.profile.search_budget_scope is SearchBudgetScope.NOT_APPLICABLE
    assert all(plan.evaluation_commitments == full.evaluation_commitments for plan in design.arms)
    assert pure.evaluation_commitments.model_spec_fingerprint == digest("exact-model-spec")
    assert pure.evaluation_commitments.solver_config_fingerprint == digest("solver-config")
    assert pure.evaluation_commitments.task_split_manifest == task_split_manifest()
    assert pure.evaluation_commitments.seed_schedule_fingerprint == digest("seed-schedule")
    assert pure.evaluation_commitments.grader_fingerprint == digest("grader")
    assert pure.evaluation_commitments.query_dag_fingerprint == digest("query-dag")
    assert pure.evaluation_commitments.retry_policy_fingerprint == digest("retry-policy")

    assert score.profile.optimizer_feedback is OptimizerFeedbackMode.SCORE_ONLY
    assert score.profile.promotion_rule is PromotionRule.PERFORMANCE_ONLY
    assert full.profile.optimizer_feedback is OptimizerFeedbackMode.MECHANISM_VISIBLE
    assert full.profile.promotion_rule is PromotionRule.MECHANISM_GATED
    assert score.profile.evidence_computation is EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET
    assert full.profile.evidence_computation is EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET
    assert score.adaptive_topology == full.adaptive_topology
    assert score.adaptive_ceilings == full.adaptive_ceilings
    assert score.adaptive_topology is not None
    assert score.adaptive_topology.protocol_commitments == protocol_commitments()
    assert design.full_score_treatment == "joint-feedback-and-promotion-policy"
    assert design.full_score_treatment_fields == ("optimizer_feedback", "promotion_rule")
    assert design.component_mediation_identified is False
    assert design.execution_attested is False
    assert design.runtime_topology_attested is False
    assert design.runtime_commitments_attested is False
    assert design.provider_identity_attested is False
    assert design.sealed_evidence is False
    assert design.reportable_result is False


@pytest.mark.parametrize(
    ("arm", "update", "message"),
    [
        (
            RealTaskArm.SCORE,
            {"promotion_rule": PromotionRule.MECHANISM_GATED},
            "score differs",
        ),
        (
            RealTaskArm.FULL,
            {"optimizer_feedback": OptimizerFeedbackMode.SCORE_ONLY},
            "full differs",
        ),
        (
            RealTaskArm.PURE,
            {"harness_mode": HarnessMode.FROZEN_SEED},
            "pure differs",
        ),
        (
            RealTaskArm.STATIC,
            {"search_budget_scope": SearchBudgetScope.SCORE_FULL_EX_ANTE},
            "static differs",
        ),
    ],
)
def test_real_task_profiles_fail_closed_on_treatment_or_reference_drift(
    arm: RealTaskArm,
    update: dict[str, object],
    message: str,
) -> None:
    values = make_real_task_arm_profile(arm).model_dump(mode="python")
    values.update(update)

    with pytest.raises(ValidationError, match=message):
        RealTaskArmProfile(**values)


def test_reference_arms_cannot_be_padded_with_adaptive_search_resources() -> None:
    for arm in (RealTaskArm.PURE, RealTaskArm.STATIC):
        with pytest.raises(ValidationError, match="must not claim adaptive search resources"):
            RealTaskArmPlan(
                profile=make_real_task_arm_profile(arm),
                evaluation_commitments=protocol_commitments().evaluation_commitments,
                adaptive_topology=topology(conditions=2),
                adaptive_ceilings=ceilings(),
            )


def test_adaptive_arms_require_both_topology_and_ceilings() -> None:
    profile = make_real_task_arm_profile(RealTaskArm.SCORE)
    shared = protocol_commitments().evaluation_commitments
    with pytest.raises(ValidationError, match="require adaptive topology and ceilings"):
        RealTaskArmPlan(profile=profile, evaluation_commitments=shared)
    with pytest.raises(ValidationError, match="supplied together"):
        RealTaskArmPlan(
            profile=profile,
            evaluation_commitments=shared,
            adaptive_topology=topology(conditions=2),
        )


def replace_arm(
    design: ConfirmatoryFourArmDesign,
    arm: RealTaskArm,
    transform: Callable[[RealTaskArmPlan], RealTaskArmPlan],
) -> tuple[RealTaskArmPlan, ...]:
    return tuple(transform(plan) if plan.arm is arm else plan for plan in design.arms)


def test_four_arm_design_rejects_missing_duplicate_and_context_drift() -> None:
    design = four_arm_design()
    duplicate = tuple(
        design.arm(RealTaskArm.PURE) if plan.arm is RealTaskArm.STATIC else plan
        for plan in design.arms
    )
    with pytest.raises(ValidationError, match="duplicate arms"):
        ConfirmatoryFourArmDesign(arms=duplicate)

    full = design.arm(RealTaskArm.FULL)
    with pytest.raises(ValidationError, match="exact SCORE and FULL contexts"):
        RealTaskArmPlan(
            profile=full.profile,
            evaluation_commitments=full.evaluation_commitments,
            adaptive_topology=topology(conditions=3),
            adaptive_ceilings=full.adaptive_ceilings,
        )


def test_four_arm_design_rejects_score_full_ceiling_drift() -> None:
    design = four_arm_design()
    full = design.arm(RealTaskArm.FULL)
    drifted_roles = tuple(
        item.model_copy(update={"max_calls": 5, "max_tokens": 5_000})
        if item.role is ModelMediatedRole.NOMINATOR
        else item
        for item in role_ceilings()
    )
    drifted_ceilings = ceilings(
        role_model_calls=drifted_roles,
        max_total_model_calls=sum(item.max_calls for item in drifted_roles),
        max_total_tokens=sum(item.max_tokens for item in drifted_roles),
        max_total_model_attempts=2 * sum(item.max_calls for item in drifted_roles),
        max_total_attempt_tokens=2 * sum(item.max_tokens for item in drifted_roles),
    )
    drifted_full = RealTaskArmPlan(
        profile=full.profile,
        evaluation_commitments=full.evaluation_commitments,
        adaptive_topology=full.adaptive_topology,
        adaptive_ceilings=drifted_ceilings,
    )

    with pytest.raises(ValidationError, match="ceilings differ"):
        ConfirmatoryFourArmDesign(
            arms=replace_arm(design, RealTaskArm.FULL, lambda _: drifted_full)
        )


def test_four_arm_design_freezes_all_minimal_runtime_commitments() -> None:
    design = four_arm_design()
    full = design.arm(RealTaskArm.FULL)
    assert full.adaptive_topology is not None
    changed_commitments = full.adaptive_topology.protocol_commitments.model_copy(
        update={"grader_fingerprint": digest("different-grader")}
    )
    changed_topology = full.adaptive_topology.model_copy(
        update={"protocol_commitments": changed_commitments}
    )
    changed_full = RealTaskArmPlan(
        profile=full.profile,
        evaluation_commitments=changed_commitments.evaluation_commitments,
        adaptive_topology=changed_topology,
        adaptive_ceilings=full.adaptive_ceilings,
    )

    with pytest.raises(ValidationError, match="evaluation commitments must be exact"):
        ConfirmatoryFourArmDesign(
            arms=replace_arm(design, RealTaskArm.FULL, lambda _: changed_full)
        )


def test_four_arm_design_rejects_seed_schedule_drift_in_only_pure() -> None:
    design = four_arm_design()
    pure = design.arm(RealTaskArm.PURE)
    changed_pure = RealTaskArmPlan(
        profile=pure.profile,
        evaluation_commitments=pure.evaluation_commitments.model_copy(
            update={"seed_schedule_fingerprint": digest("different-pure-seed-schedule")}
        ),
    )

    with pytest.raises(ValidationError, match="evaluation commitments must be exact"):
        ConfirmatoryFourArmDesign(
            arms=replace_arm(design, RealTaskArm.PURE, lambda _: changed_pure)
        )


def test_four_arm_canonical_order_has_stable_fingerprint() -> None:
    design = four_arm_design()
    reversed_design = ConfirmatoryFourArmDesign(arms=tuple(reversed(design.arms)))

    assert reversed_design.arms == design.arms
    assert reversed_design.fingerprint == design.fingerprint


def test_topology_rejects_shortened_stage_or_attribution_control_set() -> None:
    base = topology(conditions=2).model_dump(mode="python")
    with pytest.raises(ValidationError, match="every frozen stage"):
        ExAnteAdaptiveTopology(**{**base, "stages": tuple(base["stages"][:-1])})
    with pytest.raises(ValidationError, match="parent/candidate/revert/placebo"):
        ExAnteAdaptiveTopology(**{**base, "attribution_sides": ("parent", "candidate", "revert")})
    with pytest.raises(ValidationError):
        ExAnteAdaptiveTopology(**{**base, "execution_attested": True})


def test_adaptive_ceilings_canonicalize_roles_and_close_totals() -> None:
    value = ceilings()
    assert tuple(item.role for item in value.role_model_calls) == tuple(ModelMediatedRole)
    assert value.max_total_model_calls == sum(item.max_calls for item in value.role_model_calls)
    assert value.max_total_tokens == sum(item.max_tokens for item in value.role_model_calls)
    assert value.max_total_model_attempts == (
        value.max_total_model_calls * value.max_attempts_per_model_call
    )
    assert value.max_total_attempt_tokens == (
        value.max_total_tokens * value.max_attempts_per_model_call
    )
    assert value.retry_attempts_count_as_provider_calls is True
    assert value.failed_attempts_consume_attempt_and_token_ceilings is True


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_total_proposals": 9}, "round schedule"),
        ({"max_gate_queries": 3}, "exactly one fresh gate query"),
        ({"max_feedback_releases": 3}, "exactly one feedback release"),
        ({"max_total_model_calls": 101}, "role ceilings"),
        ({"max_total_tokens": 101_000}, "role ceilings"),
        ({"max_total_model_attempts": 199}, "every permitted retry attempt"),
        ({"max_total_attempt_tokens": 199_000}, "every permitted retry attempt"),
        ({"max_evaluations": 81}, "solver call ceiling"),
    ],
)
def test_adaptive_ceilings_reject_arithmetic_drift(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ceilings(**updates)


def test_adaptive_ceilings_require_complete_unique_role_roster() -> None:
    roles = role_ceilings()
    duplicate = (*roles[:-1], roles[0])
    with pytest.raises(ValidationError, match="duplicate roles"):
        ceilings(role_model_calls=duplicate)

    missing = roles[:-1]
    with pytest.raises(ValidationError):
        ceilings(role_model_calls=missing)


def test_role_ceiling_enables_calls_and_tokens_together() -> None:
    with pytest.raises(ValidationError, match="enable or disable"):
        ModelRoleCeiling(role=ModelMediatedRole.RANKER, max_calls=0, max_tokens=1)
    with pytest.raises(ValidationError, match="enable or disable"):
        ModelRoleCeiling(role=ModelMediatedRole.RANKER, max_calls=1, max_tokens=0)


def aggregation_rule(task_id: str = "gsm8k") -> PureAtBAggregationRule:
    return PureAtBAggregationRule(
        task_id=task_id,
        task_manifest_ref=task_manifest_ref(task_id),
        rule_id=f"{task_id}-majority-v1",
        method=AggregationMethod.MAJORITY_BINARY,
        implementation_fingerprint=digest(f"majority-implementation:{task_id}"),
        normalizer_fingerprint=digest(f"normalizer:{task_id}"),
        output_domain_fingerprint=digest(f"binary-output-domain:{task_id}"),
    )


def pure_at_b_unit(
    *,
    task_id: str,
    unit_id: str,
    sample_count: int,
) -> PureAtBEvaluationUnitAllocation:
    samples = tuple(
        PureAtBSampleAllocation(
            sample_id=f"{unit_id}:sample-{index:03d}",
            seed_fingerprint=digest(f"pure-at-b-seed:{unit_id}:{index:03d}"),
            token_ceiling=1_000,
            max_provider_attempts=2,
            provider_attempt_token_ceiling=2_000,
        )
        for index in range(sample_count)
    )
    return PureAtBEvaluationUnitAllocation(
        evaluation_unit_id=unit_id,
        evaluation_unit_ref=evaluation_unit_ref(unit_id),
        task_id=task_id,
        task_manifest_ref=task_manifest_ref(task_id),
        samples=samples,
        sample_count_ceiling=sample_count,
        model_call_ceiling=sample_count,
        token_ceiling=sample_count * 1_000,
        provider_attempt_ceiling=sample_count * 2,
        provider_attempt_token_ceiling=sample_count * 2_000,
    )


def pure_at_b_inputs() -> tuple[
    tuple[PureAtBAggregationRule, ...],
    tuple[PureAtBEvaluationUnitAllocation, ...],
]:
    return (
        (aggregation_rule("bbh"), aggregation_rule("gsm8k")),
        (
            pure_at_b_unit(task_id="bbh", unit_id="bbh:seed-0", sample_count=50),
            pure_at_b_unit(task_id="gsm8k", unit_id="gsm8k:seed-0", sample_count=50),
        ),
    )


def pure_at_b_plan() -> PureAtBPlan:
    aggregations, evaluation_units = pure_at_b_inputs()
    return make_pure_at_b_plan(
        four_arm_design=four_arm_design(),
        aggregations=aggregations,
        evaluation_units=evaluation_units,
    )


def test_pure_at_b_is_separate_and_derives_full_ex_ante_budget() -> None:
    design = four_arm_design()
    aggregations, evaluation_units = pure_at_b_inputs()
    plan = make_pure_at_b_plan(
        four_arm_design=design,
        aggregations=aggregations,
        evaluation_units=evaluation_units,
    )
    full_ceilings = design.arm(RealTaskArm.FULL).adaptive_ceilings
    assert full_ceilings is not None

    assert plan not in design.arms
    assert plan.pure_sample_count_ceiling == full_ceilings.max_total_model_calls
    assert plan.pure_model_call_ceiling == full_ceilings.max_total_model_calls
    assert plan.pure_token_ceiling == full_ceilings.max_total_tokens
    assert plan.max_attempts_per_sample == full_ceilings.max_attempts_per_model_call
    assert plan.pure_model_attempt_ceiling == full_ceilings.max_total_model_attempts
    assert plan.pure_attempt_token_ceiling == full_ceilings.max_total_attempt_tokens
    full_topology = design.arm(RealTaskArm.FULL).adaptive_topology
    assert full_topology is not None
    assert plan.model_spec_fingerprint == (
        full_topology.protocol_commitments.model_spec_fingerprint
    )
    assert plan.solver_config_fingerprint == (
        full_topology.protocol_commitments.solver_config_fingerprint
    )
    assert tuple(item.task_id for item in plan.aggregations) == ("bbh", "gsm8k")
    assert all(item.adaptive is False for item in plan.aggregations)
    assert all(item.grader_feedback_used is False for item in plan.aggregations)
    assert sum(item.model_call_ceiling for item in plan.evaluation_units) == 100
    assert sum(item.token_ceiling for item in plan.evaluation_units) == 100_000
    assert plan.task_split_fingerprint == (
        full_topology.protocol_commitments.task_split_fingerprint
    )
    assert plan.grader_fingerprint == full_topology.protocol_commitments.grader_fingerprint
    assert plan.query_dag_fingerprint == full_topology.protocol_commitments.query_dag_fingerprint
    assert plan.retry_policy_fingerprint == (
        full_topology.protocol_commitments.retry_policy_fingerprint
    )
    assert plan.hidden_grader_best_of_k_permitted is False
    assert plan.same_solver_configuration_attested is False
    assert plan.sample_independence_attested is False
    assert plan.aggregation_execution_attested is False
    assert plan.execution_attested is False
    assert plan.provider_identity_attested is False
    assert plan.sealed_evidence is False
    assert plan.reportable_result is False


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("pure_sample_count_ceiling", 99, "differs from evaluation-unit allocations"),
        ("pure_model_call_ceiling", 99, "differs from evaluation-unit allocations"),
        ("pure_token_ceiling", 99, "differs from evaluation-unit allocations"),
        ("max_attempts_per_sample", 99, "attempt ceiling differs from FULL"),
        ("pure_model_attempt_ceiling", 99, "differs from evaluation-unit allocations"),
        ("pure_attempt_token_ceiling", 99, "differs from evaluation-unit allocations"),
        ("model_spec_fingerprint", "0" * 64, "model spec fingerprint differs from FULL"),
        ("solver_config_fingerprint", "0" * 64, "solver config fingerprint differs from FULL"),
        ("task_split_fingerprint", "0" * 64, "task split fingerprint differs from FULL"),
        (
            "evaluation_seed_schedule_fingerprint",
            "0" * 64,
            "evaluation seed schedule fingerprint differs from FULL",
        ),
        ("grader_fingerprint", "0" * 64, "grader fingerprint differs from FULL"),
        ("query_dag_fingerprint", "0" * 64, "query dag fingerprint differs from FULL"),
        ("retry_policy_fingerprint", "0" * 64, "retry policy fingerprint differs from FULL"),
        (
            "sample_seed_schedule_fingerprint",
            "0" * 64,
            "seed schedule fingerprint differs from exact sample seeds",
        ),
        ("four_arm_design_fingerprint", "0" * 64, "fingerprint differs"),
    ],
)
def test_pure_at_b_rejects_budget_or_design_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    plan = make_pure_at_b_plan(
        four_arm_design=four_arm_design(),
        aggregations=pure_at_b_inputs()[0],
        evaluation_units=pure_at_b_inputs()[1],
    )
    values = plan.model_dump(mode="python")
    values[field] = replacement

    with pytest.raises(ValidationError, match=message):
        PureAtBPlan(**values)


def test_pure_at_b_aggregation_cannot_be_adaptive_or_use_a_hidden_grader() -> None:
    base = aggregation_rule().model_dump(mode="python")
    with pytest.raises(ValidationError):
        PureAtBAggregationRule(**{**base, "adaptive": True})
    with pytest.raises(ValidationError):
        PureAtBAggregationRule(**{**base, "grader_feedback_used": True})
    with pytest.raises(ValidationError, match="exact and non-empty"):
        PureAtBAggregationRule(**{**base, "rule_id": " drift "})
    missing_implementation = dict(base)
    missing_implementation.pop("implementation_fingerprint")
    with pytest.raises(ValidationError, match="implementation_fingerprint"):
        PureAtBAggregationRule(**missing_implementation)


def test_pure_at_b_rejects_unallocated_task_or_task_artifact_drift() -> None:
    aggregations, units = pure_at_b_inputs()
    with pytest.raises(ValidationError, match="aggregation task roster differs"):
        make_pure_at_b_plan(
            four_arm_design=four_arm_design(),
            aggregations=aggregations[:1],
            evaluation_units=units,
        )

    changed_task_ref = units[0].model_copy(
        update={"task_manifest_ref": task_manifest_ref("wrong-task-artifact")}
    )
    with pytest.raises(ValidationError, match="roster is not bijective"):
        make_pure_at_b_plan(
            four_arm_design=four_arm_design(),
            aggregations=aggregations,
            evaluation_units=(changed_task_ref, units[1]),
        )


def test_task_split_manifest_rejects_duplicate_task_or_unit_identity() -> None:
    manifest = task_split_manifest()
    with pytest.raises(ValidationError, match="task IDs must not repeat"):
        ConfirmatoryTaskSplitManifest(
            split_id=manifest.split_id,
            tasks=(manifest.tasks[0], manifest.tasks[0]),
        )

    repeated_unit = manifest.tasks[1].model_copy(
        update={"evaluation_units": manifest.tasks[0].evaluation_units}
    )
    with pytest.raises(ValidationError, match="evaluation-unit IDs must be globally unique"):
        ConfirmatoryTaskSplitManifest(
            split_id=manifest.split_id,
            tasks=(manifest.tasks[0], repeated_unit),
        )


def test_pure_at_b_rejects_missing_changed_duplicate_and_foreign_roster_entries() -> None:
    aggregations, units = pure_at_b_inputs()
    design = four_arm_design()

    with pytest.raises(ValidationError, match="aggregation task roster differs"):
        make_pure_at_b_plan(
            four_arm_design=design,
            aggregations=aggregations[1:],
            evaluation_units=units[1:],
        )

    changed_ref = units[0].model_copy(
        update={"evaluation_unit_ref": evaluation_unit_ref("bbh:changed-ref")}
    )
    with pytest.raises(ValidationError, match="roster is not bijective"):
        make_pure_at_b_plan(
            four_arm_design=design,
            aggregations=aggregations,
            evaluation_units=(changed_ref, units[1]),
        )

    with pytest.raises(ValidationError, match="exactly one aggregation rule per task"):
        make_pure_at_b_plan(
            four_arm_design=design,
            aggregations=(aggregations[0], aggregations[0]),
            evaluation_units=units,
        )

    foreign_rule = aggregation_rule("foreign")
    foreign_unit = pure_at_b_unit(
        task_id="foreign",
        unit_id="foreign:seed-0",
        sample_count=50,
    )
    with pytest.raises(ValidationError, match="aggregation task roster differs"):
        make_pure_at_b_plan(
            four_arm_design=design,
            aggregations=(foreign_rule, aggregations[1]),
            evaluation_units=(foreign_unit, units[1]),
        )


def test_task_split_manifest_publishes_reloads_and_verifies_from_store(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = task_split_manifest()

    ref = publish_confirmatory_task_split(store, manifest)

    assert ref == manifest.artifact_ref
    assert load_confirmatory_task_split(store, ref) == manifest
    assert verify_confirmatory_task_split(store, ref, manifest) == manifest

    different = manifest.model_copy(update={"split_id": "different-split"})
    with pytest.raises(ConfirmatoryTaskSplitArtifactError, match="differs from the expected"):
        verify_confirmatory_task_split(store, ref, different)


def test_pure_at_b_rejects_duplicate_seeds_and_per_sample_budget_drift() -> None:
    aggregations, units = pure_at_b_inputs()
    duplicate_seed_sample = (
        units[1]
        .samples[0]
        .model_copy(update={"seed_fingerprint": units[0].samples[0].seed_fingerprint})
    )
    duplicate_seed_unit = units[1].model_copy(
        update={"samples": (duplicate_seed_sample, *units[1].samples[1:])}
    )
    with pytest.raises(ValidationError, match="sample seeds must be globally distinct"):
        make_pure_at_b_plan(
            four_arm_design=four_arm_design(),
            aggregations=aggregations,
            evaluation_units=(units[0], duplicate_seed_unit),
        )

    over_cap_sample = (
        units[1]
        .samples[0]
        .model_copy(update={"token_ceiling": 1_001, "provider_attempt_token_ceiling": 2_002})
    )
    over_cap_unit = units[1].model_copy(
        update={
            "samples": (over_cap_sample, *units[1].samples[1:]),
            "token_ceiling": units[1].token_ceiling + 1,
            "provider_attempt_token_ceiling": units[1].provider_attempt_token_ceiling + 2,
        }
    )
    with pytest.raises(ValidationError, match="per-sample token ceiling exceeds"):
        make_pure_at_b_plan(
            four_arm_design=four_arm_design(),
            aggregations=aggregations,
            evaluation_units=(units[0], over_cap_unit),
        )


def test_pure_at_b_unit_rejects_scalar_totals_not_backed_by_samples() -> None:
    unit = pure_at_b_inputs()[1][0]
    with pytest.raises(ValidationError, match="token ceiling differs from its samples"):
        PureAtBEvaluationUnitAllocation(
            **{**unit.model_dump(mode="python"), "token_ceiling": unit.token_ceiling - 1}
        )


def test_pure_at_b_canonicalizes_task_unit_and_seed_schedule_order() -> None:
    plan = pure_at_b_plan()
    aggregations, units = pure_at_b_inputs()
    reversed_plan = make_pure_at_b_plan(
        four_arm_design=four_arm_design(),
        aggregations=tuple(reversed(aggregations)),
        evaluation_units=tuple(reversed(units)),
    )

    assert reversed_plan == plan
    assert reversed_plan.fingerprint == plan.fingerprint
    assert reversed_plan.sample_seed_schedule_fingerprint == plan.sample_seed_schedule_fingerprint


def test_unchecked_pure_at_b_attestation_cannot_cross_publication_boundary() -> None:
    plan = pure_at_b_plan()
    content = plan.model_dump(mode="python", round_trip=True)
    content["execution_attested"] = True
    forged = PureAtBPlan.model_construct(**content)

    with pytest.raises(ValidationError, match="execution_attested"):
        forged.model_dump_json()


def test_fault_factorial_maps_feedback_and_promotion_without_mixing_references() -> None:
    design = factorial_design()
    assert tuple(plan.cell for plan in design.cells) == (
        FaultFactorialCell.SS,
        FaultFactorialCell.MS,
        FaultFactorialCell.SM,
        FaultFactorialCell.MM,
    )
    expected = {
        FaultFactorialCell.SS: (
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.MS: (
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.SM: (
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.MECHANISM_GATED,
        ),
        FaultFactorialCell.MM: (
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.MECHANISM_GATED,
        ),
    }
    for cell, pair in expected.items():
        plan = design.cell(cell)
        assert (plan.profile.optimizer_feedback, plan.profile.promotion_rule) == pair
        assert plan.profile.harness_mode is HarnessMode.MUTABLE
        assert plan.profile.search_mode is SearchMode.MATCHED_ADAPTIVE
        assert plan.profile.evidence_computation is EvidenceComputationMode.FULL_ATTRIBUTION_QUARTET

    assert design.conditional_guidance_effect == "MS-minus-SS"
    assert design.conditional_promotion_effect == "SM-minus-SS"
    assert design.interaction_estimand == "(MM-minus-SM)-minus-(MS-minus-SS)"
    assert design.adaptive_policy_intention_to_treat is True
    assert design.same_candidate_mediation_claimed is False
    assert design.execution_attested is False
    assert design.runtime_topology_attested is False
    assert design.runtime_commitments_attested is False
    assert design.provider_identity_attested is False
    assert design.sealed_evidence is False
    assert design.reportable_result is False


def test_fault_factorial_profile_rejects_relabelled_cell() -> None:
    profile = make_fault_factorial_profile(FaultFactorialCell.MS)
    values = profile.model_dump(mode="python")
    values["promotion_rule"] = PromotionRule.MECHANISM_GATED

    with pytest.raises(ValidationError, match="MS differs"):
        FaultFactorialProfile(**values)


def test_fault_factorial_rejects_duplicate_or_unmatched_cells() -> None:
    design = factorial_design()
    duplicate = tuple(
        design.cell(FaultFactorialCell.SS) if plan.cell is FaultFactorialCell.MS else plan
        for plan in design.cells
    )
    with pytest.raises(ValidationError, match="duplicate cells"):
        FaultFactorialDesign(cells=duplicate)

    mm = design.cell(FaultFactorialCell.MM)
    drifted = FaultFactorialCellPlan(
        profile=mm.profile,
        adaptive_topology=mm.adaptive_topology,
        adaptive_ceilings=ceilings(max_cost_usd=21.0),
    )
    cells = tuple(drifted if plan.cell is FaultFactorialCell.MM else plan for plan in design.cells)
    with pytest.raises(ValidationError, match="different ex-ante ceilings"):
        FaultFactorialDesign(cells=cells)


def test_fault_factorial_requires_four_isolated_condition_contexts() -> None:
    with pytest.raises(ValidationError, match="exact SS, MS, SM, and MM"):
        make_fault_factorial_design(
            adaptive_topology=topology(conditions=2),
            adaptive_ceilings=ceilings(),
        )


def test_design_models_forbid_unknown_fields_and_positive_attestation_claims() -> None:
    design = four_arm_design()
    with pytest.raises(ValidationError):
        ConfirmatoryFourArmDesign(arms=design.arms, hidden_result="win")
    with pytest.raises(ValidationError):
        ConfirmatoryFourArmDesign(arms=design.arms, execution_attested=True)

    factorial = factorial_design()
    with pytest.raises(ValidationError):
        FaultFactorialDesign(cells=factorial.cells, sealed_evidence=True)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (four_arm_design, "execution_attested"),
        (factorial_design, "sealed_evidence"),
        (lambda: topology(conditions=2), "runtime_proof_available"),
        (ceilings, "runtime_proof_available"),
    ],
)
def test_model_copy_cannot_create_positive_evidence_claims(
    factory: Callable[[], object],
    field: str,
) -> None:
    value = factory()
    with pytest.raises(ValidationError, match=field):
        value.model_copy(update={field: True})  # type: ignore[attr-defined]


@pytest.mark.parametrize("boundary", ["model_dump", "model_dump_json", "fingerprint"])
def test_unchecked_construction_cannot_cross_any_publication_boundary(boundary: str) -> None:
    design = four_arm_design()
    content = design.model_dump(mode="python", round_trip=True)
    content["execution_attested"] = True
    forged = ConfirmatoryFourArmDesign.model_construct(**content)

    with pytest.raises(ValidationError, match="execution_attested"):
        result = getattr(forged, boundary)
        result() if callable(result) else result


@pytest.mark.parametrize(
    "boundary",
    [
        "type-adapter-python",
        "type-adapter-json",
        "core-serializer-python",
        "core-serializer-json",
    ],
)
def test_unchecked_construction_cannot_bypass_the_wrap_serializer(boundary: str) -> None:
    design = four_arm_design()
    content = design.model_dump(mode="python", round_trip=True)
    content["execution_attested"] = True
    forged = ConfirmatoryFourArmDesign.model_construct(**content)

    with pytest.raises(PydanticSerializationError, match="execution_attested"):
        if boundary == "type-adapter-python":
            TypeAdapter(ConfirmatoryFourArmDesign).dump_python(forged)
        elif boundary == "type-adapter-json":
            TypeAdapter(ConfirmatoryFourArmDesign).dump_json(forged)
        elif boundary == "core-serializer-python":
            ConfirmatoryFourArmDesign.__pydantic_serializer__.to_python(forged)
        else:
            ConfirmatoryFourArmDesign.__pydantic_serializer__.to_json(forged)


def test_nested_unchecked_design_value_cannot_cross_parent_publication_boundary() -> None:
    design = four_arm_design()
    full = design.arm(RealTaskArm.FULL)
    assert full.adaptive_topology is not None
    topology_content = full.adaptive_topology.model_dump(mode="python", round_trip=True)
    topology_content["execution_attested"] = True
    forged_topology = ExAnteAdaptiveTopology.model_construct(**topology_content)

    forged_full = RealTaskArmPlan.model_construct(
        profile=full.profile,
        evaluation_commitments=full.evaluation_commitments,
        adaptive_topology=forged_topology,
        adaptive_ceilings=full.adaptive_ceilings,
    )
    forged_design = ConfirmatoryFourArmDesign.model_construct(
        arms=tuple(forged_full if plan.arm is RealTaskArm.FULL else plan for plan in design.arms)
    )

    with pytest.raises(PydanticSerializationError, match="execution_attested"):
        TypeAdapter(ConfirmatoryFourArmDesign).dump_json(forged_design)
