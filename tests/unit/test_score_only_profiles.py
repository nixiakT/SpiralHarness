from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.models import ArtifactRef, ComponentKind
from spiral_harness.evolution.feedback_media_types import (
    EXPLORATION_INPUTS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
)
from spiral_harness.evolution.feedback_views import (
    PerformanceDecisionBasis,
    ScoreAggregateView,
    ScoreGateDecisionView,
    ScoreOnlyFeedbackView,
    ScoreResourceTotals,
)
from spiral_harness.evolution.models import (
    expected_strategy_feedback,
    expected_strategy_mutation,
)
from spiral_harness.evolution.seeds import (
    UNBOUND_PAIRED_PROPOSER_SEED_DOMAIN,
    derive_strategy_seed,
    derive_unbound_paired_proposer_seed,
)
from spiral_harness.experiments.baseline_profiles import (
    PAIRED_PROPOSER_GROUP,
    V2_BASELINE_KINDS,
    ActionSelectionMode,
    BaselineConditionProfile,
    MatchedContrastProfile,
    MatchedContrastReport,
    V2ActionCapability,
    action_capability_profile,
    feedback_profile,
    make_condition_profile,
    make_matched_contrast_profile,
    make_matched_contrast_report,
)
from spiral_harness.experiments.baselines import (
    BaselineArmPlan,
    BaselineKind,
    FeedbackType,
    FrozenMutationPolicy,
    FrozenRunContext,
    MutationCapability,
    MutationMode,
    PairedEvaluationPlan,
    ResourceCeilings,
)
from spiral_harness.verification.mechanism import ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE
from spiral_harness.verification.models import Decision


def ref(character: str, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=64, media_type=media_type)


def mutation_policy(*, max_size: int = 8_192) -> FrozenMutationPolicy:
    return FrozenMutationPolicy(
        grammar_version="matched-atomic-replace-v2",
        allowed_component_kinds=(ComponentKind.SKILL, ComponentKind.PROMPT),
        max_artifact_size_bytes=max_size,
    )


def resource_totals() -> ScoreResourceTotals:
    return ScoreResourceTotals(
        total_tokens=2_000,
        total_latency_ms=400.0,
        total_tool_calls=3,
        total_model_calls=12,
        failed_model_calls=1,
        retry_count=1,
        total_cost_usd=0.25,
    )


def aggregate(character: str = "a") -> ScoreAggregateView:
    return ScoreAggregateView(
        candidate_sha256=character * 64,
        n_valid_pairs=20,
        n_tasks=10,
        parent_score_mean=0.60,
        candidate_score_mean=0.70,
        mean_delta=0.10,
        confidence_level=0.95,
        confidence_lower=0.02,
        confidence_upper=0.18,
        resources=resource_totals(),
    )


def score_view(*, with_decision: bool = True) -> ScoreOnlyFeedbackView:
    score = aggregate()
    decision = (
        ScoreGateDecisionView(
            candidate_sha256=score.candidate_sha256,
            query_index=0,
            decision=Decision.PROMOTE,
            performance_policy_version="utility-protected-cost-v1",
            performance_policy_config_sha256="b" * 64,
            aggregate=score,
        )
        if with_decision
        else None
    )
    return ScoreOnlyFeedbackView(
        round_index=1 if with_decision else 0,
        benchmark_metadata_ref=ref("c", SAFE_BENCHMARK_METADATA_MEDIA_TYPE),
        exploration_inputs_ref=ref("d", EXPLORATION_INPUTS_MEDIA_TYPE),
        exploration_aggregate=score,
        prior_gate_decision=decision,
    )


def test_v2_profiles_explicitly_cover_every_known_condition() -> None:
    assert frozenset(V2_BASELINE_KINDS) == frozenset(BaselineKind)
    assert len(V2_BASELINE_KINDS) == len(BaselineKind)
    assert (
        tuple(
            make_condition_profile(kind=kind, mutation_policy=mutation_policy()).kind
            for kind in V2_BASELINE_KINDS
        )
        == V2_BASELINE_KINDS
    )

    with pytest.raises(TypeError, match="exact BaselineKind"):
        feedback_profile("future-condition")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact BaselineKind"):
        action_capability_profile("future-condition", mutation_policy())  # type: ignore[arg-type]


def test_score_and_full_share_clean_action_capability_but_not_feedback() -> None:
    policy = mutation_policy()
    score_action = action_capability_profile(BaselineKind.SCORE_ONLY_MATCHED, policy)
    full_action = action_capability_profile(BaselineKind.EVIDENCE_TARGETED, policy)
    assert score_action == full_action
    assert score_action.selection_mode is ActionSelectionMode.MATCHED_OPTIMIZATION
    assert score_action.mutable_component_kinds == (
        ComponentKind.PROMPT,
        ComponentKind.SKILL,
    )
    assert score_action.may_call_optimizer_model is True
    assert "may_use_diagnostic_evidence" not in V2ActionCapability.model_fields
    with pytest.raises(ValidationError, match="Extra inputs"):
        V2ActionCapability(
            **score_action.model_dump(mode="python"),
            may_use_diagnostic_evidence=True,
        )

    score_feedback = frozenset(feedback_profile(BaselineKind.SCORE_ONLY_MATCHED))
    full_feedback = frozenset(feedback_profile(BaselineKind.EVIDENCE_TARGETED))
    assert score_feedback < full_feedback
    assert {
        FeedbackType.EXPLORATION_AGGREGATES,
        FeedbackType.GATE_AGGREGATES,
    }.issubset(score_feedback)
    assert score_feedback.isdisjoint(
        {
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
            FeedbackType.DIAGNOSTIC_EVIDENCE,
            FeedbackType.MECHANISM_EVIDENCE,
        }
    )


def test_prompt_only_remains_distinct_from_score_only() -> None:
    prompt = frozenset(feedback_profile(BaselineKind.PROMPT_ONLY))
    score = frozenset(feedback_profile(BaselineKind.SCORE_ONLY_MATCHED))
    assert FeedbackType.EXPLORATION_ITEM_FEEDBACK in prompt
    assert FeedbackType.EXPLORATION_TRAJECTORIES in prompt
    assert FeedbackType.EXPLORATION_ITEM_FEEDBACK not in score
    assert FeedbackType.EXPLORATION_TRAJECTORIES not in score
    assert action_capability_profile(
        BaselineKind.PROMPT_ONLY,
        mutation_policy(),
    ) != action_capability_profile(BaselineKind.SCORE_ONLY_MATCHED, mutation_policy())


def test_condition_profile_rejects_grant_or_pairing_drift() -> None:
    profile = make_condition_profile(
        kind=BaselineKind.SCORE_ONLY_MATCHED,
        mutation_policy=mutation_policy(),
    )
    assert profile.paired_proposer_group == PAIRED_PROPOSER_GROUP

    values = profile.model_dump(mode="python")
    with pytest.raises(ValidationError, match="feedback differs"):
        BaselineConditionProfile(
            **{
                **values,
                "available_feedback": (
                    *profile.available_feedback,
                    FeedbackType.DIAGNOSTIC_EVIDENCE,
                ),
            }
        )
    with pytest.raises(ValidationError, match="pairing group"):
        BaselineConditionProfile(**{**values, "paired_proposer_group": None})


def test_matched_contrast_proves_only_kind_and_feedback_differ() -> None:
    contrast = make_matched_contrast_profile(mutation_policy=mutation_policy())
    score = contrast.score
    full = contrast.full
    assert score.kind is BaselineKind.SCORE_ONLY_MATCHED
    assert full.kind is BaselineKind.EVIDENCE_TARGETED
    assert score.mutation_policy == full.mutation_policy
    assert score.action_capability == full.action_capability
    assert score.available_feedback != full.available_feedback
    assert score.model_dump(exclude={"kind", "available_feedback"}) == full.model_dump(
        exclude={"kind", "available_feedback"}
    )

    report = make_matched_contrast_report(mutation_policy=mutation_policy())
    assert report.contrast == contrast
    assert report.structurally_matched is True
    assert report.execution_attested is False
    assert report.runtime_topology_matched is False
    assert report.paired_proposer_seed_runtime_bound is False


def test_matched_contrast_rejects_grammar_capability_and_report_drift() -> None:
    score = make_condition_profile(
        kind=BaselineKind.SCORE_ONLY_MATCHED,
        mutation_policy=mutation_policy(max_size=8_192),
    )
    full = make_condition_profile(
        kind=BaselineKind.EVIDENCE_TARGETED,
        mutation_policy=mutation_policy(max_size=4_096),
    )
    with pytest.raises(ValidationError, match="mutation policies must be identical"):
        MatchedContrastProfile(score=score, full=full)

    valid_full = make_condition_profile(
        kind=BaselineKind.EVIDENCE_TARGETED,
        mutation_policy=mutation_policy(),
    )
    forged_action = valid_full.action_capability.model_copy(
        update={"mutable_component_kinds": (ComponentKind.PROMPT,)}
    )
    forged_full = valid_full.model_copy(update={"action_capability": forged_action})
    with pytest.raises(ValidationError, match="action capability"):
        MatchedContrastProfile(score=score, full=forged_full)

    contrast = make_matched_contrast_profile(mutation_policy=mutation_policy())
    with pytest.raises(ValidationError, match="contrast_fingerprint"):
        MatchedContrastReport(
            contrast=contrast,
            contrast_fingerprint="f" * 64,
        )
    with pytest.raises(ValidationError, match="Input should be 'mutation-policy-equal'"):
        MatchedContrastReport(
            contrast=contrast,
            contrast_fingerprint=contrast.fingerprint,
            checks=(
                "action-grammar-equal",
                "mutation-policy-equal",
                "action-capability-equal",
                "only-kind-and-feedback-grant-differ",
            ),
        )


def test_legacy_contracts_explicitly_reject_score_only() -> None:
    with pytest.raises(ValueError, match="protocol-v2 feedback view"):
        expected_strategy_feedback(BaselineKind.SCORE_ONLY_MATCHED)
    with pytest.raises(ValueError, match="protocol-v2 mutation profile"):
        expected_strategy_mutation(BaselineKind.SCORE_ONLY_MATCHED)
    with pytest.raises(ValueError, match="legacy strategy seed"):
        derive_strategy_seed(
            proposal_master_seed=17,
            search_run_seed=101,
            baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        )

    context = FrozenRunContext(
        benchmark_ref=ref("1"),
        model_fingerprint="fixed-model",
        inference_fingerprint="temperature=0",
        runtime_fingerprint="fixed-runtime",
        seed_harness_ref=ref("2"),
        mutation_policy=mutation_policy(),
        proposal_random_seed=17,
    )
    with pytest.raises(ValidationError, match="legacy four-arm protocol"):
        BaselineArmPlan(
            kind=BaselineKind.SCORE_ONLY_MATCHED,
            context=context,
            evaluation=PairedEvaluationPlan(
                search_run_seeds=(101, 103),
                repeat_seeds=(11, 13),
            ),
            ceilings=ResourceCeilings(
                max_evaluations=10,
                max_feedback_queries=2,
                max_proposals=2,
                max_optimizer_model_calls=2,
                max_tokens=10_000,
                max_wall_time_seconds=60.0,
                max_cost_usd=1.0,
            ),
            available_feedback=feedback_profile(BaselineKind.SCORE_ONLY_MATCHED),
            mutation=MutationCapability(mode=MutationMode.NONE),
        )


def test_unbound_seed_helper_is_equal_but_not_runtime_evidence() -> None:
    assert "unbound" in UNBOUND_PAIRED_PROPOSER_SEED_DOMAIN
    values = {
        kind: derive_unbound_paired_proposer_seed(
            proposal_master_seed=17,
            search_run_seed=101,
            baseline_kind=kind,
        )
        for kind in (
            BaselineKind.SCORE_ONLY_MATCHED,
            BaselineKind.EVIDENCE_TARGETED,
        )
    }
    assert len(set(values.values())) == 1
    assert values[BaselineKind.SCORE_ONLY_MATCHED] != derive_unbound_paired_proposer_seed(
        proposal_master_seed=17,
        search_run_seed=103,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    with pytest.raises(ValueError, match="only for SCORE and FULL"):
        derive_unbound_paired_proposer_seed(
            proposal_master_seed=17,
            search_run_seed=101,
            baseline_kind=BaselineKind.PROMPT_ONLY,
        )
    with pytest.raises(TypeError, match="must be an integer"):
        derive_unbound_paired_proposer_seed(
            proposal_master_seed=True,  # type: ignore[arg-type]
            search_run_seed=101,
            baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        )
    assert (
        make_matched_contrast_report(
            mutation_policy=mutation_policy()
        ).paired_proposer_seed_runtime_bound
        is False
    )


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "exploration_item_feedback_ref",
        "exploration_trajectories_ref",
        "diagnostic_evidence_ref",
        "mechanism_evidence_ref",
        "activation_ref",
        "adherence_ref",
        "behavior_ref",
        "item_outputs",
    ),
)
def test_score_view_rejects_forbidden_evidence_fields(forbidden_field: str) -> None:
    values = score_view().model_dump(mode="python")
    values[forbidden_field] = ref("e")
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreOnlyFeedbackView(**values)


def test_score_view_requires_exact_role_media_and_rejects_mechanism_json_spoof() -> None:
    values = score_view().model_dump(mode="python")
    with pytest.raises(ValidationError, match="exact safe-metadata media type"):
        ScoreOnlyFeedbackView(
            **{
                **values,
                "benchmark_metadata_ref": ref(
                    "c",
                    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
                ),
            }
        )
    with pytest.raises(ValidationError, match="exact exploration-inputs media type"):
        ScoreOnlyFeedbackView(
            **{
                **values,
                "exploration_inputs_ref": ref(
                    "d",
                    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
                ),
            }
        )


def test_gate_view_is_performance_only_and_never_claims_full_gate_redaction() -> None:
    view = score_view()
    decision = view.prior_gate_decision
    assert decision is not None
    assert decision.projection_scope == "performance-only-not-full-gate"
    assert decision.basis == PerformanceDecisionBasis()
    assert decision.basis.included_checks == ("utility", "protected-behavior", "cost")
    assert decision.basis.excluded_mechanism_checks == (
        "activation",
        "adherence",
        "behavior",
    )
    assert "gate_version" not in ScoreGateDecisionView.model_fields
    assert "gate_config_sha256" not in ScoreGateDecisionView.model_fields

    values = decision.model_dump(mode="python")
    for forbidden_field in (
        "full_gate_decision",
        "activation_passed",
        "adherence_passed",
        "behavior_passed",
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            ScoreGateDecisionView(**values, **{forbidden_field: True})
    with pytest.raises(ValidationError, match="Input should be 'utility-protected-cost-v1'"):
        ScoreGateDecisionView(
            **{
                **values,
                "basis": {"basis_id": "full-gate-redaction-v1"},
            }
        )


def test_score_view_and_nested_types_are_strict_extra_forbid() -> None:
    view = score_view()
    assert view.runtime_role_binding_attested is False
    assert view.performance_projection_attested is False
    assert view.model_config["extra"] == "forbid"
    assert view.exploration_aggregate.model_config["extra"] == "forbid"
    assert view.prior_gate_decision is not None
    assert view.prior_gate_decision.model_config["extra"] == "forbid"

    resources = resource_totals().model_dump(mode="python")
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreResourceTotals(**resources, trajectory_count=4)
    metrics = aggregate().model_dump(mode="python")
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreAggregateView(**metrics, per_task_scores=(0.5, 0.7))


def test_score_view_binds_candidate_and_decision_sequence() -> None:
    score = aggregate()
    with pytest.raises(ValidationError, match="different candidates"):
        ScoreGateDecisionView(
            candidate_sha256="f" * 64,
            query_index=0,
            decision=Decision.REJECT,
            performance_policy_version="utility-protected-cost-v1",
            performance_policy_config_sha256="b" * 64,
            aggregate=score,
        )

    view = score_view()
    assert view.prior_gate_decision is not None
    values = view.model_dump(mode="python")
    prior = view.prior_gate_decision.model_copy(update={"query_index": view.round_index})
    with pytest.raises(ValidationError, match="must precede"):
        ScoreOnlyFeedbackView(**{**values, "prior_gate_decision": prior})


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("parent_score_mean", -0.01),
        ("parent_score_mean", 1.01),
        ("candidate_score_mean", -0.01),
        ("candidate_score_mean", 1.01),
        ("mean_delta", -1.01),
        ("mean_delta", 1.01),
        ("confidence_lower", -1.01),
        ("confidence_upper", 1.01),
    ),
)
def test_score_aggregate_enforces_normalized_bounds(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ScoreAggregateView(**{**aggregate().model_dump(mode="python"), field_name: value})


def test_score_aggregate_rejects_inconsistent_counts_delta_interval_and_resources() -> None:
    values = aggregate().model_dump(mode="python")
    with pytest.raises(ValidationError, match="at least one task"):
        ScoreAggregateView(**{**values, "n_tasks": 0})
    with pytest.raises(ValidationError, match="greater than or equal"):
        ScoreAggregateView(**{**values, "n_valid_pairs": 9})
    with pytest.raises(ValidationError, match="must not exceed"):
        ScoreAggregateView(
            **{
                **values,
                "confidence_lower": 0.2,
                "confidence_upper": 0.1,
            }
        )
    with pytest.raises(ValidationError, match="mean_delta must equal"):
        ScoreAggregateView(**{**values, "mean_delta": 0.11})
    with pytest.raises(ValidationError, match="must contain mean_delta"):
        ScoreAggregateView(
            **{
                **values,
                "confidence_lower": -0.1,
                "confidence_upper": 0.05,
            }
        )
    with pytest.raises(ValidationError, match="failed_model_calls"):
        ScoreResourceTotals(
            **{
                **resource_totals().model_dump(mode="python"),
                "failed_model_calls": 13,
            }
        )
