from __future__ import annotations

import pytest

from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE, ArtifactRef, ComponentKind
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    TrustedExecutionUsage,
)
from spiral_harness.execution.schedule import EvaluationPhase
from spiral_harness.experiments.baseline_execution import (
    BaselineExecutionBindingError,
    BaselineGateBatchUsage,
    baseline_gate_study_label,
    derive_baseline_gate_schedule,
    derive_baseline_gate_schedules,
    summarize_baseline_gate_usage,
)
from spiral_harness.experiments.baselines import (
    BaselineKind,
    BaselineProtocolValidator,
    BaselineStudyPlan,
    FeedbackType,
    FrozenMutationPolicy,
    FrozenRunContext,
    PairedEvaluationPlan,
    ResourceCeilings,
    ResourceUsage,
    plan_four_baselines,
)


def ref(character: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=128, media_type=media_type)


def study_plan() -> BaselineStudyPlan:
    return plan_four_baselines(
        context=FrozenRunContext(
            benchmark_ref=ref(
                "a",
                "application/vnd.spiral-harness.search-benchmark-binding.v1+json",
            ),
            model_fingerprint="fixed-model:sha256:111",
            inference_fingerprint="temperature=0;top_p=1;max_tokens=2048",
            runtime_fingerprint="runner-image:sha256:222",
            seed_harness_ref=ref("b", HARNESS_MANIFEST_MEDIA_TYPE),
            mutation_policy=FrozenMutationPolicy(
                grammar_version="atomic-prompt-replace-v1",
                allowed_component_kinds=(ComponentKind.PROMPT,),
                max_artifact_size_bytes=65_536,
            ),
            proposal_random_seed=20260812,
        ),
        evaluation=PairedEvaluationPlan(
            search_run_seeds=(103, 101),
            repeat_seeds=(29, 11),
        ),
        ceilings=ResourceCeilings(
            max_evaluations=32,
            max_feedback_queries=4,
            max_proposals=4,
            max_optimizer_model_calls=4,
            max_tokens=10_000,
            max_wall_time_seconds=600.0,
            max_cost_usd=0.0,
        ),
    )


def usage_for(schedule, *, tokens: int = 40) -> TrustedExecutionUsage:
    receipt_refs = tuple(
        ref(str(index + 1), EXECUTION_RECEIPT_MEDIA_TYPE) for index in range(schedule.cell_count)
    )
    ledger_tail_refs = (ref("f", "application/vnd.spiral-harness.attempt-outcome.v3+json"),)
    return TrustedExecutionUsage(
        schedule_fingerprint=schedule.fingerprint,
        receipt_refs=receipt_refs,
        ledger_tail_refs=ledger_tail_refs,
        cell_count=schedule.cell_count,
        attempt_count=schedule.cell_count,
        settled_attempts=schedule.cell_count,
        burned_attempts=0,
        poisoned_attempts=0,
        reported_tokens=tokens,
        charged_tokens=tokens,
    )


def test_derives_canonical_gate_schedules_for_every_search_run() -> None:
    plan = study_plan()
    parent_ref = plan.arms[0].context.seed_harness_ref
    candidate_ref = ref("c", HARNESS_MANIFEST_MEDIA_TYPE)

    schedules = derive_baseline_gate_schedules(
        plan,
        kind=BaselineKind.PROMPT_ONLY,
        query=2,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=candidate_ref,
        task_ids=("task-b", "task-a"),
        token_ceiling_per_attempt=64,
    )

    assert tuple(schedule.search_runs[0] for schedule in schedules) == (101, 103)
    assert {schedule.phase for schedule in schedules} == {EvaluationPhase.GATE}
    assert {schedule.kind for schedule in schedules} == {BaselineKind.PROMPT_ONLY.value}
    assert {schedule.study for schedule in schedules} == {baseline_gate_study_label(plan)}
    assert {schedule.query for schedule in schedules} == {2}
    assert {schedule.parent_harness_id for schedule in schedules} == {parent_ref.sha256}
    assert {schedule.candidate_harness_id for schedule in schedules} == {candidate_ref.sha256}
    assert {schedule.task_ids for schedule in schedules} == {("task-a", "task-b")}
    assert {schedule.repeat_seeds for schedule in schedules} == {(11, 29)}
    assert {schedule.cell_count for schedule in schedules} == {8}
    assert len({schedule.master_seed for schedule in schedules}) == 2


def test_summarizes_complete_receipt_backed_gate_usage() -> None:
    plan = study_plan()
    parent_ref = plan.arms[0].context.seed_harness_ref
    candidate_ref = ref("c", HARNESS_MANIFEST_MEDIA_TYPE)
    schedules = derive_baseline_gate_schedules(
        plan,
        kind=BaselineKind.PROMPT_ONLY,
        query=0,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=candidate_ref,
        task_ids=("task-a",),
        token_ceiling_per_attempt=64,
    )
    batches = tuple(
        BaselineGateBatchUsage(schedule=schedule, usage=usage_for(schedule, tokens=25))
        for schedule in schedules
    )

    report = summarize_baseline_gate_usage(
        plan,
        kind=BaselineKind.PROMPT_ONLY,
        batches=batches,
        additional_usage=ResourceUsage(
            feedback_queries=1,
            proposals=1,
            optimizer_model_calls=1,
            tokens=7,
        ),
        feedback_used=(
            FeedbackType.BENCHMARK_METADATA,
            FeedbackType.EXPLORATION_INPUTS,
        ),
        mutated_component_kinds=(ComponentKind.PROMPT,),
    )

    assert report.plan_fingerprint == plan.fingerprint
    assert report.executed_search_run_seeds == (101, 103)
    assert report.executed_repeat_seeds == (11, 29)
    assert report.used.evaluations == sum(schedule.cell_count for schedule in schedules)
    assert report.used.tokens == 57
    assert report.used.feedback_queries == 1
    assert report.used.optimizer_model_calls == 1
    assert (
        BaselineProtocolValidator.validate_usage(
            plan,
            (
                summarize_baseline_gate_usage(
                    plan,
                    kind=kind,
                    batches=tuple(
                        BaselineGateBatchUsage(schedule=s, usage=usage_for(s))
                        for s in derive_baseline_gate_schedules(
                            plan,
                            kind=kind,
                            query=0,
                            parent_harness_ref=parent_ref,
                            candidate_harness_ref=candidate_ref,
                            task_ids=("task-a",),
                            token_ceiling_per_attempt=64,
                        )
                    ),
                )
                for kind in BaselineKind
            ),
        ).reports_consistent
        is True
    )


def test_rejects_schedule_that_would_exceed_full_plan_budget() -> None:
    plan = study_plan().model_copy(
        update={
            "arms": tuple(
                arm.model_copy(
                    update={"ceilings": arm.ceilings.model_copy(update={"max_evaluations": 7})}
                )
                for arm in study_plan().arms
            )
        }
    )

    with pytest.raises(BaselineExecutionBindingError, match="max_evaluations"):
        derive_baseline_gate_schedule(
            plan,
            kind=BaselineKind.PROMPT_ONLY,
            search_run_seed=101,
            query=0,
            parent_harness_ref=ref("b", HARNESS_MANIFEST_MEDIA_TYPE),
            candidate_harness_ref=ref("c", HARNESS_MANIFEST_MEDIA_TYPE),
            task_ids=("task-a",),
            token_ceiling_per_attempt=64,
        )


def test_rejects_gate_schedule_parent_that_differs_from_plan_seed_harness() -> None:
    plan = study_plan()

    with pytest.raises(BaselineExecutionBindingError, match="plan seed harness"):
        derive_baseline_gate_schedule(
            plan,
            kind=BaselineKind.PROMPT_ONLY,
            search_run_seed=101,
            query=0,
            parent_harness_ref=ref("d", HARNESS_MANIFEST_MEDIA_TYPE),
            candidate_harness_ref=ref("c", HARNESS_MANIFEST_MEDIA_TYPE),
            task_ids=("task-a",),
            token_ceiling_per_attempt=64,
        )


def test_rejects_incomplete_or_drifted_gate_usage() -> None:
    plan = study_plan()
    parent_ref = plan.arms[0].context.seed_harness_ref
    candidate_ref = ref("c", HARNESS_MANIFEST_MEDIA_TYPE)
    schedules = derive_baseline_gate_schedules(
        plan,
        kind=BaselineKind.PROMPT_ONLY,
        query=0,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=candidate_ref,
        task_ids=("task-a",),
        token_ceiling_per_attempt=64,
    )

    with pytest.raises(BaselineExecutionBindingError, match="every frozen search run"):
        summarize_baseline_gate_usage(
            plan,
            kind=BaselineKind.PROMPT_ONLY,
            batches=(BaselineGateBatchUsage(schedule=schedules[0], usage=usage_for(schedules[0])),),
        )

    drifted_usage = usage_for(schedules[0]).model_copy(update={"schedule_fingerprint": "e" * 64})
    with pytest.raises(BaselineExecutionBindingError, match="another schedule"):
        summarize_baseline_gate_usage(
            plan,
            kind=BaselineKind.PROMPT_ONLY,
            batches=(
                BaselineGateBatchUsage(schedule=schedules[0], usage=drifted_usage),
                BaselineGateBatchUsage(schedule=schedules[1], usage=usage_for(schedules[1])),
            ),
        )


def test_rejects_static_search_claims_and_non_gate_schedule() -> None:
    plan = study_plan()
    parent_ref = plan.arms[0].context.seed_harness_ref
    candidate_ref = ref("c", HARNESS_MANIFEST_MEDIA_TYPE)
    schedules = derive_baseline_gate_schedules(
        plan,
        kind=BaselineKind.STATIC,
        query=0,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=candidate_ref,
        task_ids=("task-a",),
        token_ceiling_per_attempt=64,
    )
    batches = tuple(
        BaselineGateBatchUsage(schedule=schedule, usage=usage_for(schedule))
        for schedule in schedules
    )

    with pytest.raises(BaselineExecutionBindingError, match="forbidden mutations"):
        summarize_baseline_gate_usage(
            plan,
            kind=BaselineKind.STATIC,
            batches=batches,
            additional_usage=ResourceUsage(proposals=1),
            mutated_component_kinds=(ComponentKind.PROMPT,),
        )

    non_gate = schedules[0].model_copy(update={"phase": EvaluationPhase.PROBE})
    with pytest.raises(BaselineExecutionBindingError, match="GATE"):
        summarize_baseline_gate_usage(
            plan,
            kind=BaselineKind.STATIC,
            batches=(
                BaselineGateBatchUsage(schedule=non_gate, usage=usage_for(non_gate)),
                BaselineGateBatchUsage(schedule=schedules[1], usage=usage_for(schedules[1])),
            ),
        )
