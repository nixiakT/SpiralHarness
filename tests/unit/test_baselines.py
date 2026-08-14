from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE, ArtifactRef, ComponentKind
from spiral_harness.experiments.baselines import (
    LEGACY_BASELINE_KINDS,
    BaselineArmPlan,
    BaselineKind,
    BaselineProtocolError,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    BaselineUsageReport,
    FeedbackType,
    FrozenMutationPolicy,
    FrozenRunContext,
    MutationCapability,
    MutationMode,
    PairedEvaluationPlan,
    ResourceCeilings,
    ResourceUsage,
    plan_four_baselines,
)


def ref(character: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=128, media_type=media_type)


def study_plan() -> BaselineStudyPlan:
    context = FrozenRunContext(
        benchmark_ref=ref(
            "a",
            "application/vnd.spiral-harness.benchmark-manifest.v1+json",
        ),
        model_fingerprint="fixed-model:sha256:111",
        inference_fingerprint="temperature=0;top_p=1;max_tokens=2048",
        runtime_fingerprint="runner-image:sha256:222",
        seed_harness_ref=ref(
            "b",
            HARNESS_MANIFEST_MEDIA_TYPE,
        ),
        mutation_policy=FrozenMutationPolicy(
            grammar_version="atomic-replace-v1",
            allowed_component_kinds=(ComponentKind.SKILL, ComponentKind.PROMPT),
            max_artifact_size_bytes=65_536,
        ),
        proposal_random_seed=20260811,
    )
    evaluation = PairedEvaluationPlan(
        search_run_seeds=(103, 101, 107),
        repeat_seeds=(29, 11, 17),
    )
    ceilings = ResourceCeilings(
        max_evaluations=1_000,
        max_feedback_queries=20,
        max_proposals=40,
        max_optimizer_model_calls=80,
        max_tokens=2_000_000,
        max_wall_time_seconds=7_200.0,
        max_cost_usd=100.0,
    )
    return plan_four_baselines(
        context=context,
        evaluation=evaluation,
        ceilings=ceilings,
    )


def usage_reports(plan: BaselineStudyPlan) -> tuple[BaselineUsageReport, ...]:
    used_by_kind = {
        BaselineKind.STATIC: ResourceUsage(
            evaluations=120,
            tokens=12_000,
            wall_time_seconds=60.0,
            cost_usd=1.0,
        ),
        BaselineKind.RANDOM_VALID: ResourceUsage(
            evaluations=240,
            feedback_queries=4,
            proposals=8,
            tokens=24_000,
            wall_time_seconds=120.0,
            cost_usd=2.0,
        ),
        BaselineKind.PROMPT_ONLY: ResourceUsage(
            evaluations=240,
            feedback_queries=4,
            proposals=8,
            optimizer_model_calls=8,
            tokens=30_000,
            wall_time_seconds=140.0,
            cost_usd=3.0,
        ),
        BaselineKind.EVIDENCE_TARGETED: ResourceUsage(
            evaluations=240,
            feedback_queries=4,
            proposals=8,
            optimizer_model_calls=12,
            tokens=36_000,
            wall_time_seconds=160.0,
            cost_usd=4.0,
        ),
    }
    feedback_by_kind = {
        BaselineKind.STATIC: (FeedbackType.BENCHMARK_METADATA,),
        BaselineKind.RANDOM_VALID: (
            FeedbackType.BENCHMARK_METADATA,
            FeedbackType.GATE_AGGREGATES,
        ),
        BaselineKind.PROMPT_ONLY: (
            FeedbackType.EXPLORATION_AGGREGATES,
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
            FeedbackType.GATE_AGGREGATES,
        ),
        BaselineKind.EVIDENCE_TARGETED: (
            FeedbackType.EXPLORATION_TRAJECTORIES,
            FeedbackType.DIAGNOSTIC_EVIDENCE,
            FeedbackType.GATE_AGGREGATES,
        ),
    }
    mutated_by_kind = {
        BaselineKind.STATIC: (),
        BaselineKind.RANDOM_VALID: (ComponentKind.SKILL,),
        BaselineKind.PROMPT_ONLY: (ComponentKind.PROMPT,),
        BaselineKind.EVIDENCE_TARGETED: (ComponentKind.PROMPT, ComponentKind.SKILL),
    }
    return tuple(
        BaselineUsageReport(
            plan_fingerprint=plan.fingerprint,
            kind=kind,
            available=plan.arm(kind).ceilings,
            used=used_by_kind[kind],
            executed_search_run_seeds=plan.arm(kind).evaluation.search_run_seeds,
            executed_repeat_seeds=plan.arm(kind).evaluation.repeat_seeds,
            feedback_used=feedback_by_kind[kind],
            mutated_component_kinds=mutated_by_kind[kind],
        )
        for kind in LEGACY_BASELINE_KINDS
    )


def replace_report(
    reports: tuple[BaselineUsageReport, ...],
    kind: BaselineKind,
    transform: Callable[[BaselineUsageReport], BaselineUsageReport],
) -> tuple[BaselineUsageReport, ...]:
    return tuple(transform(report) if report.kind is kind else report for report in reports)


def test_planner_freezes_exactly_four_matched_condition_profiles() -> None:
    plan = study_plan()

    assert tuple(arm.kind for arm in plan.arms) == tuple(
        sorted(LEGACY_BASELINE_KINDS, key=lambda kind: kind.value)
    )
    assert len({arm.context for arm in plan.arms}) == 1
    assert len({arm.evaluation for arm in plan.arms}) == 1
    assert len({arm.ceilings for arm in plan.arms}) == 1
    assert plan.arms[0].evaluation.search_run_seeds == (101, 103, 107)
    assert plan.arms[0].evaluation.repeat_seeds == (11, 17, 29)

    static = plan.arm(BaselineKind.STATIC)
    random_valid = plan.arm(BaselineKind.RANDOM_VALID)
    prompt_only = plan.arm(BaselineKind.PROMPT_ONLY)
    targeted = plan.arm(BaselineKind.EVIDENCE_TARGETED)
    assert static.mutation.mode is MutationMode.NONE
    assert static.mutation.mutable_component_kinds == ()
    assert random_valid.mutation.mode is MutationMode.UNIFORM_RANDOM_VALID
    assert random_valid.mutation.mutable_component_kinds == (
        ComponentKind.PROMPT,
        ComponentKind.SKILL,
    )
    assert prompt_only.mutation.mutable_component_kinds == (ComponentKind.PROMPT,)
    assert targeted.mutation.may_use_diagnostic_evidence is True
    assert FeedbackType.DIAGNOSTIC_EVIDENCE not in prompt_only.available_feedback
    assert FeedbackType.DIAGNOSTIC_EVIDENCE in targeted.available_feedback
    assert FeedbackType.EXPLORATION_ITEM_FEEDBACK in prompt_only.available_feedback
    assert FeedbackType.EXPLORATION_TRAJECTORIES in prompt_only.available_feedback
    assert all(
        FeedbackType.GATE_ITEM_CONTENT not in arm.available_feedback
        and FeedbackType.SEALED_ITEM_CONTENT not in arm.available_feedback
        for arm in plan.arms
    )


def test_study_rejects_missing_or_duplicate_conditions() -> None:
    plan = study_plan()

    with pytest.raises(ValidationError, match="exactly four conditions"):
        BaselineStudyPlan(arms=plan.arms[:-1])
    with pytest.raises(ValidationError, match="duplicate conditions"):
        BaselineStudyPlan(arms=(*plan.arms[:-1], plan.arms[0]))


@pytest.mark.parametrize(
    "field",
    [
        "max_evaluations",
        "max_feedback_queries",
        "max_proposals",
        "max_optimizer_model_calls",
        "max_tokens",
        "max_wall_time_seconds",
    ],
)
def test_required_study_resource_ceiling_must_be_positive(field: str) -> None:
    values = study_plan().arms[0].ceilings.model_dump(mode="python")
    values[field] = 0.0 if field == "max_wall_time_seconds" else 0

    with pytest.raises(ValidationError, match="greater than 0"):
        ResourceCeilings(**values)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_resource_floats_must_be_finite(value: float) -> None:
    ceilings = study_plan().arms[0].ceilings.model_dump(mode="python")
    ceilings["max_cost_usd"] = value
    with pytest.raises(ValidationError, match="finite number"):
        ResourceCeilings(**ceilings)
    with pytest.raises(ValidationError, match="finite number"):
        ResourceUsage(cost_usd=value)

    assert ResourceCeilings(**{**ceilings, "max_cost_usd": 0.0}).max_cost_usd == 0.0


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "context",
            lambda arm: arm.context.model_copy(update={"model_fingerprint": "drifted-model"}),
            "frozen context drifted",
        ),
        (
            "evaluation",
            lambda arm: arm.evaluation.model_copy(update={"repeat_seeds": (11, 17, 31)}),
            "paired evaluation plan drifted",
        ),
        (
            "evaluation",
            lambda arm: arm.evaluation.model_copy(update={"search_run_seeds": (101, 103, 109)}),
            "paired evaluation plan drifted",
        ),
        (
            "ceilings",
            lambda arm: arm.ceilings.model_copy(update={"max_feedback_queries": 21}),
            "resource ceilings drifted",
        ),
    ],
)
def test_study_rejects_context_seed_or_ceiling_drift(
    field: str,
    replacement: Callable[[BaselineArmPlan], object],
    error: str,
) -> None:
    plan = study_plan()
    original = plan.arm(BaselineKind.EVIDENCE_TARGETED)
    drifted = original.model_copy(update={field: replacement(original)})

    with pytest.raises(ValidationError, match=error):
        BaselineStudyPlan(
            arms=tuple(drifted if arm.kind is original.kind else arm for arm in plan.arms)
        )


def test_condition_profile_rejects_item_feedback_and_mutation_escalation() -> None:
    plan = study_plan()
    prompt = plan.arm(BaselineKind.PROMPT_ONLY)

    with pytest.raises(ValidationError, match="gate/sealed item feedback is forbidden"):
        BaselineArmPlan(
            **prompt.model_dump(
                mode="python",
                exclude={"available_feedback"},
            ),
            available_feedback=(*prompt.available_feedback, FeedbackType.GATE_ITEM_CONTENT),
        )
    with pytest.raises(ValidationError, match="mutation capability"):
        BaselineArmPlan(
            **prompt.model_dump(mode="python", exclude={"mutation"}),
            mutation=MutationCapability(
                mode=MutationMode.PROMPT_OPTIMIZATION,
                mutable_component_kinds=(ComponentKind.PROMPT, ComponentKind.SKILL),
                may_call_optimizer_model=True,
            ),
        )

    with pytest.raises(ValidationError, match="Input should be 1"):
        FrozenMutationPolicy(
            grammar_version="non-atomic",
            allowed_component_kinds=(ComponentKind.PROMPT,),
            max_components_per_proposal=2,
            max_artifact_size_bytes=65_536,
        )


def test_protocol_validator_accepts_available_but_not_equal_declared_usage() -> None:
    plan = study_plan()
    reports = usage_reports(plan)

    result = BaselineProtocolValidator.validate_usage(plan, reports)

    assert result.reports_consistent is True
    assert result.execution_attested is False
    assert result.evidence_scope == "self-reported-aggregate-claims"
    assert result.plan_fingerprint == plan.fingerprint
    static = next(report for report in reports if report.kind is BaselineKind.STATIC)
    assert static.available.max_proposals == 40
    assert static.used.proposals == 0
    assert static.used.optimizer_model_calls == 0
    assert static.used.evaluations > 0


def test_protocol_report_shape_cannot_omit_conditions_or_checks() -> None:
    plan = study_plan()
    valid = BaselineProtocolValidator.validate_usage(plan, usage_reports(plan))

    with pytest.raises(ValidationError, match="exactly the four"):
        type(valid)(
            plan_fingerprint=plan.fingerprint,
            baseline_kinds=(BaselineKind.STATIC,),
        )
    with pytest.raises(ValidationError, match="complete canonical"):
        type(valid)(
            plan_fingerprint=plan.fingerprint,
            baseline_kinds=LEGACY_BASELINE_KINDS,
            checks=("complete-condition-set",),
        )


@pytest.mark.parametrize(
    "used",
    [
        ResourceUsage(evaluations=120, feedback_queries=1),
        ResourceUsage(evaluations=120, optimizer_model_calls=1),
        ResourceUsage(evaluations=120, proposals=1),
    ],
)
def test_static_cannot_claim_search_budget_consumption(used: ResourceUsage) -> None:
    plan = study_plan()
    reports = usage_reports(plan)
    reports = replace_report(
        reports,
        BaselineKind.STATIC,
        lambda report: report.model_copy(update={"used": used}),
    )

    with pytest.raises(BaselineProtocolError, match=r"static.*(search|optimizer|proposal|mutat)"):
        BaselineProtocolValidator.validate_usage(plan, reports)


def test_repeat_seeds_are_unique_and_every_report_must_name_the_frozen_pairs() -> None:
    with pytest.raises(ValidationError, match="repeat_seeds must be unique"):
        PairedEvaluationPlan(search_run_seeds=(101, 103), repeat_seeds=(11, 11))

    plan = study_plan()
    reports = usage_reports(plan)
    reports = replace_report(
        reports,
        BaselineKind.RANDOM_VALID,
        lambda report: report.model_copy(update={"executed_repeat_seeds": (11, 17)}),
    )
    with pytest.raises(BaselineProtocolError, match="paired repeat seeds"):
        BaselineProtocolValidator.validate_usage(plan, reports)


def test_reported_search_run_seeds_are_independent_unique_and_complete() -> None:
    with pytest.raises(ValidationError, match="search_run_seeds must be unique"):
        PairedEvaluationPlan(search_run_seeds=(101, 101), repeat_seeds=(11, 17))
    with pytest.raises(ValidationError, match="must be disjoint"):
        PairedEvaluationPlan(search_run_seeds=(101, 103), repeat_seeds=(17, 101))

    plan = study_plan()
    reports = usage_reports(plan)
    reports = replace_report(
        reports,
        BaselineKind.EVIDENCE_TARGETED,
        lambda report: report.model_copy(update={"executed_search_run_seeds": (101, 103)}),
    )
    with pytest.raises(BaselineProtocolError, match="independent search runs"):
        BaselineProtocolValidator.validate_usage(plan, reports)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluations", 1_001),
        ("feedback_queries", 21),
        ("proposals", 41),
        ("optimizer_model_calls", 81),
        ("tokens", 2_000_001),
        ("wall_time_seconds", 7_200.1),
        ("cost_usd", 100.01),
    ],
)
def test_usage_rejects_every_resource_overrun(field: str, value: int | float) -> None:
    plan = study_plan()
    reports = usage_reports(plan)

    def overrun(report: BaselineUsageReport) -> BaselineUsageReport:
        values = report.used.model_dump(mode="python")
        values[field] = value
        return report.model_copy(update={"used": ResourceUsage(**values)})

    reports = replace_report(reports, BaselineKind.EVIDENCE_TARGETED, overrun)
    with pytest.raises(BaselineProtocolError, match=field):
        BaselineProtocolValidator.validate_usage(plan, reports)


def test_usage_rejects_missing_duplicate_or_foreign_allocations() -> None:
    plan = study_plan()
    reports = usage_reports(plan)

    with pytest.raises(BaselineProtocolError, match="incomplete"):
        BaselineProtocolValidator.validate_usage(plan, reports[:-1])
    with pytest.raises(BaselineProtocolError, match="duplicate"):
        BaselineProtocolValidator.validate_usage(plan, (*reports[:-1], reports[0]))

    drifted_available = plan.arm(BaselineKind.PROMPT_ONLY).ceilings.model_copy(
        update={"max_proposals": 39}
    )
    drifted = replace_report(
        reports,
        BaselineKind.PROMPT_ONLY,
        lambda report: report.model_copy(update={"available": drifted_available}),
    )
    with pytest.raises(BaselineProtocolError, match="available ceilings"):
        BaselineProtocolValidator.validate_usage(plan, drifted)


def test_usage_rejects_information_or_component_permission_violations() -> None:
    plan = study_plan()
    reports = usage_reports(plan)
    feedback_violation = replace_report(
        reports,
        BaselineKind.PROMPT_ONLY,
        lambda report: report.model_copy(
            update={
                "feedback_used": (
                    *report.feedback_used,
                    FeedbackType.DIAGNOSTIC_EVIDENCE,
                )
            }
        ),
    )
    with pytest.raises(BaselineProtocolError, match="information permissions"):
        BaselineProtocolValidator.validate_usage(plan, feedback_violation)

    mutation_violation = replace_report(
        reports,
        BaselineKind.PROMPT_ONLY,
        lambda report: report.model_copy(
            update={"mutated_component_kinds": (ComponentKind.SKILL,)}
        ),
    )
    with pytest.raises(BaselineProtocolError, match="outside its condition profile"):
        BaselineProtocolValidator.validate_usage(plan, mutation_violation)


def test_unchecked_plan_copy_is_revalidated_by_protocol_validator() -> None:
    plan = study_plan()
    target = plan.arm(BaselineKind.EVIDENCE_TARGETED)
    drifted_target = target.model_copy(
        update={
            "context": target.context.model_copy(
                update={"runtime_fingerprint": "different-runtime"}
            )
        }
    )
    unchecked = plan.model_copy(
        update={
            "arms": tuple(drifted_target if arm.kind is target.kind else arm for arm in plan.arms)
        }
    )

    with pytest.raises(BaselineProtocolError, match="frozen context drifted"):
        BaselineProtocolValidator.validate_plan(unchecked)
