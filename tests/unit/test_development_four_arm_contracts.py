from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.development_four_arm_contracts import (
    AutomaticCandidateSelection,
    BenchmarkCandidateComparison,
    BenchmarkKind,
    DevelopmentArm,
    DevelopmentArmObservation,
    DevelopmentFourArmClosure,
    DevelopmentFourArmPlan,
    DevelopmentSplit,
    DevelopmentTask,
    FrozenCandidateProposal,
    FullDisclosure,
    FullFitTaskDisclosure,
    JointCandidateFreeze,
    ScoreBenchmarkAggregate,
    ScoreDisclosure,
    SelectionDecision,
    development_adaptive_stage_fingerprint,
    development_candidate_freeze_id,
)


def _digest(label: str) -> str:
    return canonical_sha256({"test": "development-four-arm", "label": label})


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="fixture-openai-compatible",
        backend_fingerprint="fixture-backend@sha256:development-v1",
        model="fixture/exact-open-model",
        revision="2026-08-14-development-snapshot",
        tokenizer="fixture-tokenizer",
        tokenizer_revision="2026-08-14-tokenizer-snapshot",
        runtime="fixture-runtime@sha256:development-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=512,
            timeout_seconds=60.0,
        ),
    )


def _tasks() -> tuple[DevelopmentTask, ...]:
    tasks: list[DevelopmentTask] = []
    seed = 100
    for benchmark in BenchmarkKind:
        for split in DevelopmentSplit:
            for index in range(4):
                tasks.append(
                    DevelopmentTask(
                        task_id=f"{benchmark.value}-{split.value}-{index}",
                        benchmark=benchmark,
                        split=split,
                        question=f"Question {benchmark.value} {split.value} {index}?",
                        seed=seed,
                    )
                )
                seed += 1
    return tuple(tasks)


def _plan(*, tasks: tuple[DevelopmentTask, ...] | None = None) -> DevelopmentFourArmPlan:
    return DevelopmentFourArmPlan(
        tasks=_tasks() if tasks is None else tasks,
        model_spec=_spec(),
        max_tokens_per_attempt=4_096,
        max_total_tokens=58 * 4_096,
    )


def _score_disclosure() -> ScoreDisclosure:
    return ScoreDisclosure(
        aggregates=(
            ScoreBenchmarkAggregate(
                benchmark=BenchmarkKind.GSM8K,
                score=0.5,
            ),
            ScoreBenchmarkAggregate(
                benchmark=BenchmarkKind.BBH,
                score=0.75,
            ),
        )
    )


def _full_disclosure(plan: DevelopmentFourArmPlan) -> FullDisclosure:
    correct_limit = {BenchmarkKind.GSM8K: 2, BenchmarkKind.BBH: 3}
    seen = {benchmark: 0 for benchmark in BenchmarkKind}
    items: list[FullFitTaskDisclosure] = []
    for task in plan.tasks:
        if task.split is not DevelopmentSplit.FIT:
            continue
        correct = seen[task.benchmark] < correct_limit[task.benchmark]
        seen[task.benchmark] += 1
        items.append(
            FullFitTaskDisclosure(
                task_id=task.task_id,
                benchmark=task.benchmark,
                question=task.question,
                output=f"Model output for {task.task_id}",
                correct=correct,
            )
        )
    return FullDisclosure(observations=tuple(items))


def _proposal(arm: DevelopmentArm) -> FrozenCandidateProposal:
    return FrozenCandidateProposal(
        arm=arm,
        candidate_id=_digest(f"{arm.value}-candidate-id"),
        proposal_fingerprint=_digest(f"{arm.value}-proposal"),
        parent_condition_id=_digest("static-shared-parent-condition"),
        candidate_condition_id=_digest(f"{arm.value}-candidate-condition"),
    )


def _freeze(plan: DevelopmentFourArmPlan | None = None) -> JointCandidateFreeze:
    checked_plan = _plan() if plan is None else plan
    proposals = (_proposal(DevelopmentArm.SCORE), _proposal(DevelopmentArm.FULL))
    adaptive_stage_fingerprint = development_adaptive_stage_fingerprint(checked_plan)
    return JointCandidateFreeze(
        adaptive_stage_fingerprint=adaptive_stage_fingerprint,
        freeze_id=development_candidate_freeze_id(
            adaptive_stage_fingerprint=adaptive_stage_fingerprint,
            proposals=proposals,
        ),
        freeze_sequence=4,
        candidate_evaluation_start_sequence=5,
        proposals=proposals,
    )


def _comparisons(
    *,
    gsm_parent: int,
    gsm_candidate: int,
    bbh_parent: int,
    bbh_candidate: int,
) -> tuple[BenchmarkCandidateComparison, ...]:
    return (
        BenchmarkCandidateComparison(
            benchmark=BenchmarkKind.GSM8K,
            parent_correct=gsm_parent,
            candidate_correct=gsm_candidate,
        ),
        BenchmarkCandidateComparison(
            benchmark=BenchmarkKind.BBH,
            parent_correct=bbh_parent,
            candidate_correct=bbh_candidate,
        ),
    )


def _selections(
    freeze: JointCandidateFreeze,
) -> tuple[AutomaticCandidateSelection, ...]:
    proposals = {proposal.arm: proposal for proposal in freeze.proposals}
    score = proposals[DevelopmentArm.SCORE]
    full = proposals[DevelopmentArm.FULL]
    return (
        AutomaticCandidateSelection(
            arm=DevelopmentArm.SCORE,
            candidate_id=score.candidate_id,
            candidate_valid=True,
            comparisons=_comparisons(
                gsm_parent=2,
                gsm_candidate=3,
                bbh_parent=3,
                bbh_candidate=3,
            ),
            decision=SelectionDecision.PROMOTE,
            selected_condition_id=score.candidate_condition_id,
        ),
        AutomaticCandidateSelection(
            arm=DevelopmentArm.FULL,
            candidate_id=full.candidate_id,
            candidate_valid=True,
            comparisons=_comparisons(
                gsm_parent=2,
                gsm_candidate=2,
                bbh_parent=3,
                bbh_candidate=3,
            ),
            decision=SelectionDecision.ROLLBACK,
            selected_condition_id=full.parent_condition_id,
        ),
    )


def _condition_ids(
    selections: Iterable[AutomaticCandidateSelection],
) -> dict[DevelopmentArm, str]:
    adaptive = {selection.arm: selection.selected_condition_id for selection in selections}
    return {
        DevelopmentArm.PURE: _digest("pure-condition"),
        DevelopmentArm.STATIC: _digest("static-shared-parent-condition"),
        DevelopmentArm.SCORE: adaptive[DevelopmentArm.SCORE],
        DevelopmentArm.FULL: adaptive[DevelopmentArm.FULL],
    }


def _observations(
    plan: DevelopmentFourArmPlan,
    selections: tuple[AutomaticCandidateSelection, ...],
) -> tuple[DevelopmentArmObservation, ...]:
    conditions = _condition_ids(selections)
    observations: list[DevelopmentArmObservation] = []
    for task in plan.tasks:
        if task.split is not DevelopmentSplit.HOLDOUT:
            continue
        for arm in DevelopmentArm:
            observations.append(
                DevelopmentArmObservation(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    arm=arm,
                    seed=task.seed,
                    model_spec_fingerprint=plan.model_spec.fingerprint,
                    condition_id=conditions[arm],
                    reference_id=_digest(f"observation/{task.task_id}/{arm.value}"),
                    correct=(arm in {DevelopmentArm.SCORE, DevelopmentArm.FULL}),
                )
            )
    return tuple(observations)


def _closure() -> DevelopmentFourArmClosure:
    plan = _plan()
    freeze = _freeze(plan)
    selections = _selections(freeze)
    return DevelopmentFourArmClosure(
        plan=plan,
        plan_fingerprint=plan.fingerprint,
        score_disclosure=_score_disclosure(),
        full_disclosure=_full_disclosure(plan),
        candidate_freeze=freeze,
        selections=selections,
        observations=_observations(plan, selections),
    )


def _content(model: object) -> dict[str, object]:
    return model.model_dump(mode="python", round_trip=True, warnings="none")  # type: ignore[attr-defined,no-any-return]


def test_plan_freezes_exact_public_task_matrix_model_budget_and_nonsealed_status() -> None:
    plan = _plan(tasks=tuple(reversed(_tasks())))

    assert len(plan.tasks) == 16
    assert len({task.task_id for task in plan.tasks}) == 16
    assert plan.benchmarks == (BenchmarkKind.GSM8K, BenchmarkKind.BBH)
    assert plan.arms == tuple(DevelopmentArm)
    assert plan.max_model_calls == 58
    assert plan.max_tokens_per_attempt == 4_096
    assert plan.max_total_tokens == 58 * 4_096
    assert plan.sealed is False
    assert plan.evidence_scope == "public-development-feedback-view-only"
    assert plan.adaptive_contrast == "item-evidence-vs-aggregate-score-feedback-view-only"
    assert plan.score_full_shared_parent_rollouts is True
    assert plan.joint_policy_effect_identified is False
    assert plan.execution_attested is False
    assert plan.budget_matched is False
    assert plan.runtime_execution_proof is False
    assert plan.confirmatory_protocol is False
    assert plan.fingerprint == _plan().fingerprint

    for benchmark in BenchmarkKind:
        selected = [task for task in plan.tasks if task.benchmark is benchmark]
        assert len(selected) == 8
        assert sum(task.split is DevelopmentSplit.FIT for task in selected) == 4
        assert sum(task.split is DevelopmentSplit.HOLDOUT for task in selected) == 4


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"max_model_calls": 57}, "max_model_calls"),
        ({"max_tokens_per_attempt": 511}, "cover the model output-token ceiling"),
        ({"max_total_tokens": 1}, r"max_model_calls \* max_tokens_per_attempt"),
        ({"sealed": True}, "sealed"),
        ({"arms": (DevelopmentArm.PURE,)}, "arms"),
        ({"benchmarks": (BenchmarkKind.GSM8K,)}, "benchmarks"),
        ({"evidence_scope": "confirmatory"}, "evidence_scope"),
        ({"adaptive_contrast": "joint-policy"}, "adaptive_contrast"),
        ({"score_full_shared_parent_rollouts": False}, "shared_parent"),
        ({"joint_policy_effect_identified": True}, "joint_policy"),
        ({"execution_attested": True}, "execution_attested"),
        ({"budget_matched": True}, "budget_matched"),
        ({"runtime_execution_proof": True}, "runtime_execution_proof"),
        ({"confirmatory_protocol": True}, "confirmatory_protocol"),
    ],
)
def test_plan_rejects_any_relaxation(update: dict[str, object], message: str) -> None:
    values = _content(_plan())
    values.update(update)
    with pytest.raises(ValidationError, match=message):
        DevelopmentFourArmPlan(**values)


def test_plan_rejects_duplicate_or_unbalanced_tasks() -> None:
    tasks = list(_tasks())
    tasks[-1] = tasks[0]
    with pytest.raises(ValidationError, match="globally unique"):
        _plan(tasks=tuple(tasks))

    tasks = list(_tasks())
    changed = _content(tasks[-1])
    changed["split"] = DevelopmentSplit.FIT
    tasks[-1] = DevelopmentTask(**changed)
    with pytest.raises(ValidationError, match="four fit and four holdout"):
        _plan(tasks=tuple(tasks))


def test_adaptive_stage_commitment_excludes_holdout_but_binds_fit_content() -> None:
    plan = _plan()
    holdout_tasks = list(plan.tasks)
    holdout_index = next(
        index for index, task in enumerate(holdout_tasks) if task.split is DevelopmentSplit.HOLDOUT
    )
    changed = _content(holdout_tasks[holdout_index])
    changed["question"] = "Changed holdout content that must not steer proposal sampling"
    holdout_tasks[holdout_index] = DevelopmentTask(**changed)
    holdout_changed = _plan(tasks=tuple(holdout_tasks))

    assert holdout_changed.fingerprint != plan.fingerprint
    assert development_adaptive_stage_fingerprint(
        holdout_changed
    ) == development_adaptive_stage_fingerprint(plan)

    fit_tasks = list(plan.tasks)
    fit_index = next(
        index for index, task in enumerate(fit_tasks) if task.split is DevelopmentSplit.FIT
    )
    changed = _content(fit_tasks[fit_index])
    changed["question"] = "Changed fit content that must change the adaptive commitment"
    fit_tasks[fit_index] = DevelopmentTask(**changed)
    fit_changed = _plan(tasks=tuple(fit_tasks))

    assert development_adaptive_stage_fingerprint(
        fit_changed
    ) != development_adaptive_stage_fingerprint(plan)


def test_score_disclosure_can_express_only_two_aggregate_counts_and_scores() -> None:
    disclosure = _score_disclosure()
    assert set(ScoreBenchmarkAggregate.model_fields) == {
        "schema_version",
        "benchmark",
        "evaluated_count",
        "score",
    }
    assert set(ScoreDisclosure.model_fields) == {"schema_version", "aggregates"}
    assert disclosure.aggregates[0].benchmark is BenchmarkKind.BBH

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ScoreBenchmarkAggregate(
            benchmark=BenchmarkKind.GSM8K,
            score=0.5,
            task_id="forbidden",
        )
    with pytest.raises(ValidationError, match="realizable"):
        ScoreBenchmarkAggregate(benchmark=BenchmarkKind.GSM8K, score=0.6)
    with pytest.raises(ValidationError, match="one aggregate per benchmark"):
        ScoreDisclosure(
            aggregates=(
                ScoreBenchmarkAggregate(benchmark=BenchmarkKind.GSM8K, score=0.5),
                ScoreBenchmarkAggregate(benchmark=BenchmarkKind.GSM8K, score=0.75),
            )
        )


def test_full_disclosure_is_fit_only_and_has_no_representable_gold() -> None:
    plan = _plan()
    disclosure = _full_disclosure(plan)
    assert len(disclosure.observations) == 8
    assert "gold" not in FullFitTaskDisclosure.model_fields
    assert "answer" not in FullFitTaskDisclosure.model_fields
    assert all(item.split is DevelopmentSplit.FIT for item in disclosure.observations)

    fit = disclosure.observations[0]
    values = _content(fit)
    values["split"] = DevelopmentSplit.HOLDOUT
    with pytest.raises(ValidationError, match="split"):
        FullFitTaskDisclosure(**values)

    values = _content(fit)
    values["gold"] = "forbidden target"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FullFitTaskDisclosure(**values)


def test_both_candidate_proposals_freeze_atomically_before_evaluation() -> None:
    freeze = _freeze()
    assert {proposal.arm for proposal in freeze.proposals} == {
        DevelopmentArm.SCORE,
        DevelopmentArm.FULL,
    }
    assert freeze.frozen_before_candidate_evaluation is True
    assert freeze.freeze_sequence < freeze.candidate_evaluation_start_sequence
    assert len({proposal.parent_condition_id for proposal in freeze.proposals}) == 1
    assert freeze.freeze_id == development_candidate_freeze_id(
        adaptive_stage_fingerprint=freeze.adaptive_stage_fingerprint,
        proposals=freeze.proposals,
    )

    values = _content(freeze)
    values["candidate_evaluation_start_sequence"] = freeze.freeze_sequence
    with pytest.raises(ValidationError, match="before candidate evaluation"):
        JointCandidateFreeze(**values)

    score = _proposal(DevelopmentArm.SCORE)
    values = _content(freeze)
    values["proposals"] = (score, score)
    with pytest.raises(ValidationError, match="SCORE and FULL"):
        JointCandidateFreeze(**values)

    values = _content(freeze)
    values["freeze_id"] = _digest("forged-joint-freeze")
    with pytest.raises(ValidationError, match="freeze_id must bind"):
        JointCandidateFreeze(**values)

    full_values = _content(_proposal(DevelopmentArm.FULL))
    full_values["parent_condition_id"] = _digest("different-parent-condition")
    values = _content(freeze)
    values["proposals"] = (
        _proposal(DevelopmentArm.SCORE),
        FrozenCandidateProposal(**full_values),
    )
    with pytest.raises(ValidationError, match="one shared parent condition"):
        JointCandidateFreeze(**values)


@pytest.mark.parametrize(
    ("candidate_valid", "comparisons", "decision"),
    [
        (
            True,
            _comparisons(
                gsm_parent=2,
                gsm_candidate=3,
                bbh_parent=3,
                bbh_candidate=3,
            ),
            SelectionDecision.PROMOTE,
        ),
        (
            False,
            _comparisons(
                gsm_parent=2,
                gsm_candidate=3,
                bbh_parent=3,
                bbh_candidate=3,
            ),
            SelectionDecision.ROLLBACK,
        ),
        (
            True,
            _comparisons(
                gsm_parent=2,
                gsm_candidate=2,
                bbh_parent=3,
                bbh_candidate=3,
            ),
            SelectionDecision.ROLLBACK,
        ),
        (
            True,
            _comparisons(
                gsm_parent=1,
                gsm_candidate=3,
                bbh_parent=4,
                bbh_candidate=3,
            ),
            SelectionDecision.ROLLBACK,
        ),
    ],
)
def test_selection_rule_is_fully_automatic(
    candidate_valid: bool,
    comparisons: tuple[BenchmarkCandidateComparison, ...],
    decision: SelectionDecision,
) -> None:
    selection = AutomaticCandidateSelection(
        arm=DevelopmentArm.SCORE,
        candidate_id=_digest("selection-candidate"),
        candidate_valid=candidate_valid,
        comparisons=comparisons,
        decision=decision,
        selected_condition_id=_digest("selection-condition"),
    )
    assert selection.manual_override is False

    values = _content(selection)
    values["decision"] = (
        SelectionDecision.ROLLBACK
        if decision is SelectionDecision.PROMOTE
        else SelectionDecision.PROMOTE
    )
    with pytest.raises(ValidationError, match="automatic rule"):
        AutomaticCandidateSelection(**values)

    values = _content(selection)
    values["manual_override"] = True
    with pytest.raises(ValidationError, match="manual_override"):
        AutomaticCandidateSelection(**values)


def test_observation_keeps_condition_and_record_reference_separate() -> None:
    closure = _closure()
    observation = closure.observations[0]
    assert observation.condition_id != observation.reference_id

    values = _content(observation)
    values["reference_id"] = observation.condition_id
    with pytest.raises(ValidationError, match="remain distinct"):
        DevelopmentArmObservation(**values)


def test_complete_closure_is_canonical_and_permanently_nonreportable() -> None:
    closure = _closure()
    assert len(closure.observations) == 8 * 4
    assert closure.plan_fingerprint == closure.plan.fingerprint
    assert closure.evidence_scope == "public-development-feedback-view-only"
    assert closure.adaptive_contrast == "item-evidence-vs-aggregate-score-feedback-view-only"
    assert closure.score_full_shared_parent_projection is True
    assert closure.joint_policy_effect_identified is False
    assert closure.execution_attested is False
    assert closure.budget_matched is False
    assert closure.runtime_execution_proof is False
    assert closure.confirmatory_protocol is False
    assert closure.reportable_benchmark_result is False
    assert closure.sealed_evidence is False
    assert closure.confirmatory_inference is False
    assert closure.simultaneous_lcb_available is False
    assert closure.provider_identity_attested is False

    reordered = _content(closure)
    reordered["observations"] = tuple(reversed(closure.observations))
    reordered["selections"] = tuple(reversed(closure.selections))
    assert DevelopmentFourArmClosure(**reordered).fingerprint == closure.fingerprint


@pytest.mark.parametrize(
    "flag",
    [
        "reportable_benchmark_result",
        "sealed_evidence",
        "confirmatory_inference",
        "simultaneous_lcb_available",
        "provider_identity_attested",
        "joint_policy_effect_identified",
        "execution_attested",
        "budget_matched",
        "runtime_execution_proof",
        "confirmatory_protocol",
    ],
)
def test_closure_cannot_upgrade_any_evidence_flag(flag: str) -> None:
    values = _content(_closure())
    values[flag] = True
    with pytest.raises(ValidationError, match=flag):
        DevelopmentFourArmClosure(**values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "32 items"),
        ("wrong-seed", "frozen task and seed"),
        ("wrong-spec", "one FrozenModelSpec"),
        ("condition-drift", "drift between condition IDs"),
        ("duplicate-reference", "unique reference_id"),
    ],
)
def test_closure_rejects_incomplete_or_unmatched_holdout_grid(
    mutation: str,
    message: str,
) -> None:
    closure = _closure()
    observations = list(closure.observations)
    if mutation == "missing":
        observations.pop()
    elif mutation == "wrong-seed":
        values = _content(observations[0])
        values["seed"] = observations[0].seed + 1
        observations[0] = DevelopmentArmObservation(**values)
    elif mutation == "wrong-spec":
        values = _content(observations[0])
        values["model_spec_fingerprint"] = _digest("wrong-spec")
        observations[0] = DevelopmentArmObservation(**values)
    elif mutation == "condition-drift":
        values = _content(observations[0])
        values["condition_id"] = _digest("drifted-condition")
        observations[0] = DevelopmentArmObservation(**values)
    elif mutation == "duplicate-reference":
        values = _content(observations[1])
        values["reference_id"] = observations[0].reference_id
        observations[1] = DevelopmentArmObservation(**values)
    else:  # pragma: no cover - closed test parameter roster.
        raise AssertionError(mutation)

    values = _content(closure)
    values["observations"] = tuple(observations)
    with pytest.raises(ValidationError, match=message):
        DevelopmentFourArmClosure(**values)


def test_closure_binds_disclosures_freeze_selection_and_selected_condition() -> None:
    closure = _closure()

    full_values = _content(closure.full_disclosure)
    full_items = list(closure.full_disclosure.observations)
    changed = _content(full_items[0])
    changed["question"] = "A substituted fit question"
    full_items[0] = FullFitTaskDisclosure(**changed)
    full_values["observations"] = tuple(full_items)
    values = _content(closure)
    values["full_disclosure"] = FullDisclosure(**full_values)
    with pytest.raises(ValidationError, match="frozen fit task"):
        DevelopmentFourArmClosure(**values)

    selection_values = _content(closure.selections[0])
    selection_values["candidate_id"] = _digest("unfrozen-candidate")
    changed_selection = AutomaticCandidateSelection(**selection_values)
    values = _content(closure)
    values["selections"] = (changed_selection, closure.selections[1])
    with pytest.raises(ValidationError, match="jointly frozen proposal"):
        DevelopmentFourArmClosure(**values)

    observations = list(closure.observations)
    score_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.arm is DevelopmentArm.SCORE
    )
    changed = _content(observations[score_index])
    changed["condition_id"] = _digest("score-unselected-condition")
    observations[score_index] = DevelopmentArmObservation(**changed)
    values = _content(closure)
    values["observations"] = tuple(observations)
    with pytest.raises(
        ValidationError,
        match=r"drift between condition IDs|automatically selected",
    ):
        DevelopmentFourArmClosure(**values)


def test_closure_requires_score_and_full_views_of_the_same_parent_rollouts() -> None:
    closure = _closure()
    aggregates = list(closure.score_disclosure.aggregates)
    gsm_index = next(
        index
        for index, aggregate in enumerate(aggregates)
        if aggregate.benchmark is BenchmarkKind.GSM8K
    )
    changed = _content(aggregates[gsm_index])
    changed["score"] = 0.25
    aggregates[gsm_index] = ScoreBenchmarkAggregate(**changed)
    score_values = _content(closure.score_disclosure)
    score_values["aggregates"] = tuple(aggregates)
    values = _content(closure)
    values["score_disclosure"] = ScoreDisclosure(**score_values)

    with pytest.raises(ValidationError, match="same shared parent rollouts"):
        DevelopmentFourArmClosure(**values)


def test_static_and_rollback_share_the_frozen_parent_condition() -> None:
    closure = _closure()
    conditions = {
        arm: {item.condition_id for item in closure.observations if item.arm is arm}
        for arm in DevelopmentArm
    }
    shared_parent = closure.candidate_freeze.proposals[0].parent_condition_id
    assert conditions[DevelopmentArm.STATIC] == {shared_parent}
    assert conditions[DevelopmentArm.FULL] == {shared_parent}
    assert len(set.union(*conditions.values())) == 3

    observations = list(closure.observations)
    static_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.arm is DevelopmentArm.STATIC
    ]
    for index in static_indices:
        changed = _content(observations[index])
        changed["condition_id"] = _digest("unfrozen-static-condition")
        observations[index] = DevelopmentArmObservation(**changed)
    values = _content(closure)
    values["observations"] = tuple(observations)
    with pytest.raises(ValidationError, match="jointly frozen shared parent"):
        DevelopmentFourArmClosure(**values)


def test_closure_rejects_a_self_consistent_freeze_for_another_adaptive_stage() -> None:
    closure = _closure()
    freeze_values = _content(closure.candidate_freeze)
    foreign_stage = _digest("foreign-adaptive-stage")
    freeze_values["adaptive_stage_fingerprint"] = foreign_stage
    freeze_values["freeze_id"] = development_candidate_freeze_id(
        adaptive_stage_fingerprint=foreign_stage,
        proposals=closure.candidate_freeze.proposals,
    )
    values = _content(closure)
    values["candidate_freeze"] = JointCandidateFreeze(**freeze_values)

    with pytest.raises(ValidationError, match="plan's fit-only adaptive stage"):
        DevelopmentFourArmClosure(**values)


def test_fingerprint_revalidates_a_constructed_claim_upgrade() -> None:
    closure = _closure()
    tampered = closure.model_copy(update={"reportable_benchmark_result": True})
    with pytest.raises(ValidationError, match="reportable_benchmark_result"):
        _ = tampered.fingerprint
