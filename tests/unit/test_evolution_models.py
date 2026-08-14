from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE
from spiral_harness.core.models import ArtifactRef, ComponentKind
from spiral_harness.evolution.models import (
    DIAGNOSIS_MEDIA_TYPE,
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    PROMPT_PROPOSAL_MEDIA_TYPE,
    RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE,
    SEARCH_POLICY_MEDIA_TYPE,
    SEARCH_STOPPING_POLICY_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    CandidateScreen,
    CandidateScreenFailure,
    CandidateScreenStatus,
    DeclineReason,
    Diagnosis,
    GateAggregateMetrics,
    GateAggregateView,
    PromptProposal,
    ProposalBatch,
    ProposalDecline,
    SearchRunManifest,
    SearchStoppingPolicy,
    StrategyFeedbackView,
)
from spiral_harness.evolution.seeds import derive_strategy_seed
from spiral_harness.evolution.strategies import (
    make_search_policy,
    make_strategy_plugin_manifest,
    validate_strategy_permissions,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    LEGACY_BASELINE_KINDS,
    BaselineKind,
    FeedbackType,
    FrozenMutationPolicy,
)
from spiral_harness.verification.models import Decision


def ref(character: str, media_type: str = "application/json", *, size: int = 32) -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=size, media_type=media_type)


def prompt_policy() -> FrozenMutationPolicy:
    return FrozenMutationPolicy(
        grammar_version="atomic-prompt-replace-v1",
        allowed_component_kinds=(ComponentKind.PROMPT,),
        max_artifact_size_bytes=4_096,
    )


def search_policy(kind: BaselineKind):
    is_static = kind is BaselineKind.STATIC
    return make_search_policy(
        baseline_kind=kind,
        mutation_policy=prompt_policy(),
        max_rounds=0 if is_static else 4,
        max_gate_queries=0 if is_static else 4,
        patience_rounds=0 if is_static else 2,
        max_consecutive_declines=0 if is_static else 2,
        max_diagnoses=8 if kind is BaselineKind.EVIDENCE_TARGETED else 0,
        max_proposals=0 if is_static else 8,
        max_screens=0 if is_static else 8,
        max_proposals_per_round=0 if is_static else 2,
        max_candidates_screened_per_round=0 if is_static else 2,
        family_alpha=0.05,
    )


def feedback_view(kind: BaselineKind) -> StrategyFeedbackView:
    values: dict[str, object] = {
        "baseline_kind": kind,
        "benchmark_metadata_ref": ref("1"),
    }
    exposed = {FeedbackType.BENCHMARK_METADATA}
    if kind is not BaselineKind.STATIC:
        values["exploration_inputs_ref"] = ref("2")
        exposed.add(FeedbackType.EXPLORATION_INPUTS)
    if kind in {BaselineKind.PROMPT_ONLY, BaselineKind.EVIDENCE_TARGETED}:
        values.update(
            exploration_aggregates_ref=ref("3"),
            exploration_item_feedback_ref=ref("4"),
            exploration_trajectories_ref=ref("5"),
        )
        exposed.update(
            {
                FeedbackType.EXPLORATION_AGGREGATES,
                FeedbackType.EXPLORATION_ITEM_FEEDBACK,
                FeedbackType.EXPLORATION_TRAJECTORIES,
            }
        )
    if kind is BaselineKind.EVIDENCE_TARGETED:
        values["diagnostic_evidence_ref"] = ref("6")
        exposed.add(FeedbackType.DIAGNOSTIC_EVIDENCE)
    values["exposed_feedback"] = tuple(exposed)
    return StrategyFeedbackView(**values)


def gate_metrics() -> GateAggregateMetrics:
    return GateAggregateMetrics(
        n_valid_pairs=20,
        n_tasks=10,
        parent_score_mean=0.60,
        candidate_score_mean=0.70,
        mean_delta=0.10,
        confidence_level=0.95,
        confidence_lower=0.02,
        confidence_upper=0.18,
        wins=6,
        ties=2,
        losses=2,
        regression_rate=0.2,
        worst_task_delta=-0.1,
        parent_tokens_mean=100.0,
        candidate_tokens_mean=105.0,
        tokens_ratio=1.05,
        parent_latency_ms_mean=50.0,
        candidate_latency_ms_mean=52.0,
        latency_ratio=1.04,
        parent_tool_calls_mean=0.0,
        candidate_tool_calls_mean=0.0,
        tool_calls_ratio=None,
    )


def prompt_proposal(
    *,
    kind: BaselineKind = BaselineKind.PROMPT_ONLY,
    proposal_id: str = "proposal-a",
) -> PromptProposal:
    values: dict[str, object] = {
        "proposal_id": proposal_id,
        "baseline_kind": kind,
        "round_index": 0,
        "parent_harness_ref": ref("7"),
        "target_component_name": "system",
        "before_prompt_ref": ref("8", "text/plain"),
        "after_prompt_ref": ref("9", "text/plain"),
        "hypothesis_ref": ref("a"),
        "mechanism_family": "explicit-verification",
    }
    if kind is BaselineKind.RANDOM_VALID:
        values.update(
            catalogue_ref=ref("b", PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE),
            catalogue_entry_id="entry-a",
        )
    if kind is BaselineKind.EVIDENCE_TARGETED:
        values["diagnosis_ref"] = ref("c", DIAGNOSIS_MEDIA_TYPE)
    return PromptProposal(**values)


@pytest.mark.parametrize("kind", LEGACY_BASELINE_KINDS)
def test_search_policy_and_plugin_freeze_exact_four_arm_profiles(kind: BaselineKind) -> None:
    policy = search_policy(kind)
    plugin = make_strategy_plugin_manifest(
        plugin_id=f"builtin-{kind.value}",
        plugin_version="1",
        implementation_ref=ref("d", "application/python"),
        baseline_kind=kind,
    )

    assert policy.mutation_policy.allowed_component_kinds == (ComponentKind.PROMPT,)
    assert policy.selector == "exploration-lcb-v1"
    if kind is BaselineKind.STATIC:
        assert policy.gate_confidence_level is None
        assert policy.stopping_policy == SearchStoppingPolicy(
            baseline_kind=kind,
            max_rounds=0,
            max_gate_queries=0,
            patience_rounds=0,
            max_consecutive_declines=0,
        )
    else:
        assert policy.gate_confidence_level == (1.0 - policy.family_alpha / policy.max_gate_queries)
    checked_policy, checked_plugin, checked_feedback = validate_strategy_permissions(
        policy=policy,
        plugin=plugin,
        feedback=feedback_view(kind),
    )
    assert checked_policy == policy
    assert checked_plugin == plugin
    assert checked_feedback == feedback_view(kind)


def test_search_policy_rejects_grammar_permission_and_analysis_drift() -> None:
    base = search_policy(BaselineKind.PROMPT_ONLY)

    with pytest.raises(ValidationError, match="prompt-only grammar"):
        type(base)(
            **{
                **base.model_dump(),
                "mutation_policy": FrozenMutationPolicy(
                    grammar_version="too-wide",
                    allowed_component_kinds=(ComponentKind.PROMPT, ComponentKind.SKILL),
                    max_artifact_size_bytes=4_096,
                ),
            }
        )
    with pytest.raises(ValidationError, match="feedback permissions"):
        type(base)(
            **{
                **base.model_dump(),
                "available_feedback": (FeedbackType.BENCHMARK_METADATA,),
            }
        )
    with pytest.raises(ValidationError, match="gate_confidence_level"):
        type(base)(**{**base.model_dump(), "gate_confidence_level": 0.95})
    with pytest.raises(ValidationError, match="non-static stopping counts"):
        type(base)(**{**base.model_dump(), "patience_rounds": 0})


def test_strategy_feedback_checks_actual_fields_not_only_permission_claim() -> None:
    prompt = feedback_view(BaselineKind.PROMPT_ONLY)
    values = prompt.model_dump()
    values["diagnostic_evidence_ref"] = ref("e")

    with pytest.raises(ValidationError, match="exactly describe"):
        StrategyFeedbackView(**values)

    values["exposed_feedback"] = (*prompt.exposed_feedback, FeedbackType.DIAGNOSTIC_EVIDENCE)
    with pytest.raises(ValidationError, match="forbidden fields"):
        StrategyFeedbackView(**values)

    targeted = feedback_view(BaselineKind.EVIDENCE_TARGETED).model_dump()
    targeted["diagnostic_evidence_ref"] = None
    targeted["exposed_feedback"] = tuple(
        value
        for value in targeted["exposed_feedback"]
        if value is not FeedbackType.DIAGNOSTIC_EVIDENCE
    )
    with pytest.raises(ValidationError, match="missing required fields"):
        StrategyFeedbackView(**targeted)


def test_gate_feedback_is_strictly_aggregate_only() -> None:
    aggregate = GateAggregateView(
        candidate_ref=ref("a"),
        analysis_plan_ref=ref("b"),
        query_index=1,
        decision=Decision.PROMOTE,
        gate_version="paired-bootstrap-v1",
        gate_config_sha256="c" * 64,
        metrics=gate_metrics(),
        passed_check_count=4,
        failed_check_count=0,
        inconclusive_check_count=0,
    )
    assert "task" not in " ".join(aggregate.model_dump().keys())

    with pytest.raises(ValidationError, match="Extra inputs"):
        GateAggregateView(
            **aggregate.model_dump(),
            task_ids=("secret-task",),
        )
    with pytest.raises(ValidationError, match="failed check"):
        GateAggregateView(
            **{
                **aggregate.model_dump(),
                "decision": Decision.REJECT,
                "failed_check_count": 0,
            }
        )

    superseded = GateAggregateView(
        **{
            **aggregate.model_dump(),
            "decision": Decision.INCONCLUSIVE,
            "resolution": "superseded-promotion",
        }
    )
    assert superseded.inconclusive_check_count == 0
    with pytest.raises(ValidationError, match="requires an inconclusive check"):
        GateAggregateView(
            **{
                **superseded.model_dump(),
                "resolution": "gate-decision",
            }
        )


def test_diagnosis_is_source_linked_canonical_and_immutable() -> None:
    diagnosis = Diagnosis(
        diagnosis_id="missing-verification",
        source_feedback_ref=ref("1", RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE),
        failure_signature_refs=(ref("3"), ref("2")),
        evidence_packet_refs=(ref("5"), ref("4")),
        protected_anchor_refs=(ref("7"), ref("6")),
        target_component_name="system",
        observed_failure="unsupported arithmetic is returned",
        root_cause="the prompt omits a verification step",
        mechanism="require an independent final calculation",
        predicted_benefit="fewer arithmetic mistakes",
        predicted_risk="additional tokens",
        falsifier="verification activates without changing mistakes",
    )
    reordered = Diagnosis(
        **{
            **diagnosis.model_dump(),
            "failure_signature_refs": tuple(reversed(diagnosis.failure_signature_refs)),
            "evidence_packet_refs": tuple(reversed(diagnosis.evidence_packet_refs)),
            "protected_anchor_refs": tuple(reversed(diagnosis.protected_anchor_refs)),
        }
    )

    assert reordered == diagnosis
    assert canonical_sha256(reordered) == canonical_sha256(diagnosis)
    with pytest.raises((ValidationError, FrozenInstanceError)):
        diagnosis.root_cause = "post-hoc rewrite"
    with pytest.raises(ValidationError, match="wrong media type"):
        Diagnosis(**{**diagnosis.model_dump(), "source_feedback_ref": ref("1")})


def test_prompt_proposal_profiles_and_batch_proposal_decline_xor() -> None:
    source = ref("f", RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE)
    proposal = prompt_proposal()
    batch = ProposalBatch(
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        source_feedback_ref=source,
        proposals=(proposal,),
    )
    assert batch.proposals == (proposal,)

    decline = ProposalDecline(
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        source_feedback_ref=source,
        reason=DeclineReason.OPTIMIZER_DECLINED,
        rationale="no bounded change is justified",
    )
    assert (
        ProposalBatch(
            baseline_kind=BaselineKind.PROMPT_ONLY,
            round_index=0,
            source_feedback_ref=source,
            decline=decline,
        ).decline
        == decline
    )

    for values in (
        {"proposals": (), "decline": None},
        {"proposals": (proposal,), "decline": decline},
    ):
        with pytest.raises(ValidationError, match="exactly one"):
            ProposalBatch(
                baseline_kind=BaselineKind.PROMPT_ONLY,
                round_index=0,
                source_feedback_ref=source,
                **values,
            )

    static_decline = ProposalDecline(
        baseline_kind=BaselineKind.STATIC,
        round_index=0,
        source_feedback_ref=source,
        reason=DeclineReason.STATIC_CONDITION,
        rationale="the static condition retains the seed harness",
    )
    assert (
        ProposalBatch(
            baseline_kind=BaselineKind.STATIC,
            round_index=0,
            source_feedback_ref=source,
            decline=static_decline,
        ).proposals
        == ()
    )


def test_prompt_proposal_enforces_arm_provenance_and_non_noop() -> None:
    proposal = prompt_proposal()
    with pytest.raises(ValidationError, match="non-noop"):
        PromptProposal(
            **{
                **proposal.model_dump(),
                "after_prompt_ref": proposal.before_prompt_ref,
            }
        )
    with pytest.raises(ValidationError, match="finite catalogue"):
        PromptProposal(
            **{
                **proposal.model_dump(),
                "baseline_kind": BaselineKind.RANDOM_VALID,
            }
        )
    with pytest.raises(ValidationError, match="typed diagnosis"):
        PromptProposal(
            **{
                **proposal.model_dump(),
                "baseline_kind": BaselineKind.EVIDENCE_TARGETED,
            }
        )
    assert prompt_proposal(kind=BaselineKind.RANDOM_VALID).catalogue_entry_id == "entry-a"
    assert prompt_proposal(kind=BaselineKind.EVIDENCE_TARGETED).diagnosis_ref is not None


def test_search_run_manifest_binds_preregistered_refs_fingerprints_and_seeds() -> None:
    kind = BaselineKind.RANDOM_VALID
    master_seed = 20260811
    run_seed = 101
    study_ref = ref("1", BASELINE_STUDY_PLAN_MEDIA_TYPE)
    policy_ref = ref("2", SEARCH_POLICY_MEDIA_TYPE)
    plugin_ref = ref("3", STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE)
    values = {
        "baseline_study_plan_ref": study_ref,
        "experiment_ref": ref("4", EXPERIMENT_MANIFEST_MEDIA_TYPE),
        "search_policy_ref": policy_ref,
        "search_policy_fingerprint": policy_ref.sha256,
        "strategy_plugin_ref": plugin_ref,
        "strategy_plugin_fingerprint": plugin_ref.sha256,
        "analysis_plan_ref": ref("5"),
        "stopping_policy_ref": ref("6", SEARCH_STOPPING_POLICY_MEDIA_TYPE),
        "stopping_policy_fingerprint": "6" * 64,
        "seed_harness_ref": ref("7"),
        "baseline_kind": kind,
        "baseline_plan_fingerprint": study_ref.sha256,
        "proposal_master_seed": master_seed,
        "search_run_seed": run_seed,
        "repeat_seeds": (29, 11),
        "strategy_seed": derive_strategy_seed(
            proposal_master_seed=master_seed,
            search_run_seed=run_seed,
            baseline_kind=kind,
        ),
        "prompt_mutation_catalogue_ref": ref(
            "8",
            PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
        ),
    }
    manifest = SearchRunManifest(**values)
    assert manifest.repeat_seeds == (11, 29)

    for field_name, value, message in (
        ("search_policy_fingerprint", "9" * 64, "search_policy_fingerprint"),
        ("strategy_seed", manifest.strategy_seed + 1, "strategy_seed"),
        ("repeat_seeds", (11, 101), "disjoint"),
        ("analysis_plan_ref", ref("5", "text/plain"), "JSON media"),
        ("search_policy_ref", ref("2"), "wrong media type"),
    ):
        with pytest.raises(ValidationError, match=message):
            SearchRunManifest(**{**values, field_name: value})


def test_candidate_screen_requires_complete_lcb_and_resource_metrics() -> None:
    values = {
        "baseline_kind": BaselineKind.PROMPT_ONLY,
        "round_index": 0,
        "proposal_ref": ref("1", PROMPT_PROPOSAL_MEDIA_TYPE),
        "candidate_ref": ref("2"),
        "candidate_harness_ref": ref("3"),
        "evaluation_ref": ref("4"),
        "status": CandidateScreenStatus.ELIGIBLE,
        "primary_score": 0.8,
        "mean_delta": 0.1,
        "confidence_lower": 0.02,
        "regression_rate": 0.1,
        "tokens_ratio": 1.05,
        "latency_ratio": 1.02,
    }
    screen = CandidateScreen(**values)
    assert screen.confidence_lower == 0.02

    with pytest.raises(ValidationError, match="present together"):
        CandidateScreen(**{**values, "confidence_lower": None})
    with pytest.raises(ValidationError, match="failure code"):
        CandidateScreen(
            **{
                **values,
                "status": CandidateScreenStatus.REJECTED,
                "failure_codes": (),
            }
        )
    rejected = CandidateScreen(
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        proposal_ref=ref("5", PROMPT_PROPOSAL_MEDIA_TYPE),
        status=CandidateScreenStatus.REJECTED,
        failure_codes=(CandidateScreenFailure.STRUCTURALLY_INVALID,),
    )
    assert rejected.primary_score is None
