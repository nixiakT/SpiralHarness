from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    BfclV4RunClosureVerification,
)
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import AttemptBudget, FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis import (
    BfclV4PublicCampaignAnalysisError,
    compute_bfcl_v4_public_campaign_descriptive_analysis,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis_contracts import (
    BfclV4CampaignPerCallBudget,
    BfclV4CampaignReplicateAnalysisInput,
    BfclV4ExactRational,
    BfclV4PublicCampaignAnalysisInput,
    BfclV4PublicCampaignDescriptiveAnalysis,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BFCL_V4_HOLDOUT_TASK_IDS,
    BFCL_V4_METRIC_ARM_ORDER,
    BFCL_V4_PAIRED_CONTRASTS,
    BfclV4ArmDescriptiveMetric,
    BfclV4ArmSelectionDecision,
    BfclV4JointSelectionDecision,
    BfclV4PairedDescriptiveDelta,
    BfclV4PublicDescriptiveMetrics,
    BfclV4RollbackReason,
    BfclV4SelectionDecision,
)


def _sha(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _ref(label: str, media_type: str) -> ArtifactRef:
    payload = label.encode("utf-8")
    return ArtifactRef(sha256=sha256_bytes(payload), size=len(payload), media_type=media_type)


def _model_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="openai-compatible-native",
        backend_fingerprint="fixture-backend-v1",
        model="fixture-open-model",
        revision="sha256:fixture-model-revision",
        tokenizer="fixture-tokenizer",
        tokenizer_revision="sha256:fixture-tokenizer-revision",
        runtime="fixture-runtime-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=256,
            timeout_seconds=30.0,
        ),
    )


def _arm_decision(plan, arm: BfclV4PilotArm, *, promote: bool) -> BfclV4ArmSelectionDecision:
    parent = _sha(f"parent:{arm.value}")
    candidate = _sha(f"candidate:{arm.value}")
    reasons = () if promote else (BfclV4RollbackReason.CANDIDATE_GATE_NOT_STRICTLY_BETTER,)
    return BfclV4ArmSelectionDecision(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        arm=arm,
        candidate_freeze_reference_sha256=_sha(f"candidate-freeze:{plan.outer_seed_u64}"),
        joint_evidence_fingerprint=_sha(f"joint-evidence:{plan.outer_seed_u64}"),
        arm_evidence_fingerprint=_sha(f"arm-evidence:{plan.outer_seed_u64}:{arm.value}"),
        candidate_resolution_fingerprint=_sha(
            f"candidate-resolution:{plan.outer_seed_u64}:{arm.value}"
        ),
        candidate_admissible=True,
        all_fit_gate_provider_attempts_succeeded=True,
        shadow_controls_exact=True,
        parent_fit_correct=2,
        candidate_fit_correct=3,
        parent_gate_correct=0,
        candidate_gate_correct=1 if promote else 0,
        candidate_fit_nondecreasing=True,
        candidate_gate_strictly_better=promote,
        decision=(BfclV4SelectionDecision.PROMOTE if promote else BfclV4SelectionDecision.ROLLBACK),
        selected_variant="candidate" if promote else "parent",
        parent_system_prompt_sha256=parent,
        candidate_system_prompt_sha256=candidate,
        selected_system_prompt_sha256=candidate if promote else parent,
        forced_rollback=False,
        rollback_reasons=reasons,
    )


def _joint_selection(
    plan,
    *,
    score_promote: bool,
    full_promote: bool,
) -> BfclV4JointSelectionDecision:
    refs = tuple(
        _ref(
            f"gate:{plan.outer_seed_u64}:{index}",
            "application/vnd.spiral-harness.bfcl-v4-call-completion.v1+json",
        )
        for index in range(16)
    )
    return BfclV4JointSelectionDecision(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        candidate_freeze_reference_sha256=_sha(f"candidate-freeze:{plan.outer_seed_u64}"),
        joint_evidence_fingerprint=_sha(f"joint-evidence:{plan.outer_seed_u64}"),
        global_gate_completion_refs=refs,
        score=_arm_decision(plan, BfclV4PilotArm.SCORE, promote=score_promote),
        full=_arm_decision(plan, BfclV4PilotArm.FULL, promote=full_promote),
    )


_CORRECTNESS = (
    {
        BfclV4PilotArm.PURE: (True, False, False, False, False, False, False, False),
        BfclV4PilotArm.STATIC: (True, True, False, False, False, False, False, False),
        BfclV4PilotArm.SCORE: (True, True, True, True, False, False, False, False),
        BfclV4PilotArm.FULL: (True, True, True, True, True, True, False, False),
        BfclV4PilotArm.PURE_AT_B: (True, True, True, False, False, False, False, False),
    },
    {
        BfclV4PilotArm.PURE: (False, True, False, False, False, False, False, False),
        BfclV4PilotArm.STATIC: (False, True, True, False, False, False, False, False),
        BfclV4PilotArm.SCORE: (True, True, False, False, True, True, False, False),
        BfclV4PilotArm.FULL: (True, True, True, True, True, True, False, False),
        BfclV4PilotArm.PURE_AT_B: (False, True, True, True, False, False, False, False),
    },
    {
        BfclV4PilotArm.PURE: (False, False, True, False, False, False, False, False),
        BfclV4PilotArm.STATIC: (False, False, True, True, False, False, False, False),
        BfclV4PilotArm.SCORE: (True, False, True, False, False, False, True, True),
        BfclV4PilotArm.FULL: (True, True, True, True, True, False, False, False),
        BfclV4PilotArm.PURE_AT_B: (False, False, True, True, True, False, False, False),
    },
)


def _metrics(plan, joint, correctness) -> BfclV4PublicDescriptiveMetrics:
    arms = tuple(
        BfclV4ArmDescriptiveMetric(
            arm=arm,
            task_ids=BFCL_V4_HOLDOUT_TASK_IDS,
            correctness=correctness[arm],
            correct_count=sum(correctness[arm]),
            accuracy_basis_points=sum(correctness[arm]) * 1_250,
        )
        for arm in BFCL_V4_METRIC_ARM_ORDER
    )
    paired = []
    for treatment, reference in BFCL_V4_PAIRED_CONTRASTS:
        left = correctness[treatment]
        right = correctness[reference]
        wins = sum(a and not b for a, b in zip(left, right, strict=True))
        losses = sum(b and not a for a, b in zip(left, right, strict=True))
        delta = (wins - losses) * 1_250
        paired.append(
            BfclV4PairedDescriptiveDelta(
                treatment_arm=treatment,
                reference_arm=reference,
                task_ids=BFCL_V4_HOLDOUT_TASK_IDS,
                treatment_correctness=left,
                reference_correctness=right,
                wins=wins,
                ties=8 - wins - losses,
                losses=losses,
                delta_basis_points=delta,
                strictly_exceeds_positive_ten_percentage_points=delta > 1_000,
            )
        )
    return BfclV4PublicDescriptiveMetrics(
        plan_fingerprint=plan.fingerprint,
        schedule_content_sha256=plan.schedule_content_sha256,
        joint_selection_decision_fingerprint=joint.fingerprint,
        holdout_evidence_fingerprint=_sha(f"holdout:{plan.outer_seed_u64}"),
        arms=arms,
        paired_deltas=tuple(paired),
    )


def _analysis_input() -> BfclV4PublicCampaignAnalysisInput:
    campaign = build_bfcl_v4_public_pilot_campaign()
    spec = _model_spec()
    attempt_budget = AttemptBudget(
        max_attempts=100,
        max_total_tokens=51_200,
        max_tokens_per_attempt=512,
    )
    budget = BfclV4CampaignPerCallBudget(
        max_output_tokens=256,
        attempt_budget=attempt_budget,
        attempt_budget_fingerprint=attempt_budget.fingerprint,
    )
    promotion_patterns = ((True, True), (False, True), (True, False))
    replicates = []
    for registered, correctness, (score_promote, full_promote) in zip(
        campaign.replicates, _CORRECTNESS, promotion_patterns, strict=True
    ):
        plan = registered.call_plan
        joint = _joint_selection(
            plan,
            score_promote=score_promote,
            full_promote=full_promote,
        )
        metrics = _metrics(plan, joint, correctness)
        closure_ref = _ref(f"closure:{registered.replicate_id}", BFCL_V4_RUN_CLOSURE_MEDIA_TYPE)
        verification = BfclV4RunClosureVerification(
            closure_fingerprint=closure_ref.sha256,
            plan_fingerprint=plan.fingerprint,
            replayed_transition_count=204,
        )
        replicates.append(
            BfclV4CampaignReplicateAnalysisInput(
                ordinal=registered.ordinal,
                replicate_id=registered.replicate_id,
                outer_seed_u64=registered.outer_seed_u64,
                plan_fingerprint=plan.fingerprint,
                schedule_content_sha256=plan.schedule_content_sha256,
                model_spec_fingerprint=spec.fingerprint,
                backend_fingerprint=spec.backend_fingerprint,
                inference_fingerprint=spec.inference_fingerprint,
                attempt_budget=attempt_budget,
                attempt_budget_fingerprint=attempt_budget.fingerprint,
                per_call_budget_fingerprint=budget.fingerprint,
                closure_ref=closure_ref,
                closure_verification=verification,
                joint_selection=joint,
                joint_selection_fingerprint=joint.fingerprint,
                descriptive_metrics=metrics,
                descriptive_metrics_fingerprint=metrics.fingerprint,
            )
        )
    return BfclV4PublicCampaignAnalysisInput(
        campaign=campaign,
        campaign_fingerprint=BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
        model_spec=spec,
        model_spec_fingerprint=spec.fingerprint,
        backend=spec.backend,
        backend_fingerprint=spec.backend_fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        per_call_budget=budget,
        per_call_budget_fingerprint=budget.fingerprint,
        ordered_closure_refs=tuple(item.closure_ref for item in replicates),
        replicates=tuple(replicates),
    )


def _all_values(value):
    if isinstance(value, dict):
        return (value, *(child for item in value.values() for child in _all_values(item)))
    if isinstance(value, (list, tuple)):
        return (value, *(child for item in value for child in _all_values(item)))
    return (value,)


def test_campaign_analysis_preserves_clusters_promotions_and_exact_rational_delta() -> None:
    analysis_input = _analysis_input()
    result = compute_bfcl_v4_public_campaign_descriptive_analysis(analysis_input)

    assert result.campaign_fingerprint == BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT
    assert result.plan_fingerprints == tuple(
        item.call_plan.fingerprint for item in analysis_input.campaign.replicates
    )
    assert result.ordered_closure_refs == analysis_input.ordered_closure_refs
    assert (
        result.attempt_budget_fingerprint
        == analysis_input.per_call_budget.attempt_budget_fingerprint
    )
    full = next(item for item in result.arms if item.arm is BfclV4PilotArm.FULL)
    assert full.correct_task_seed_cells == 17
    assert full.accuracy == BfclV4ExactRational(numerator=17, denominator=24)
    assert full.task_clusters[0].correctness == (True, True, True)
    assert full.task_clusters[5].correctness == (True, True, False)
    assert [(item.arm, item.promotion_count) for item in result.promotions] == [
        (BfclV4PilotArm.SCORE, 2),
        (BfclV4PilotArm.FULL, 2),
    ]

    full_pure = next(
        item
        for item in result.paired_deltas
        if (item.treatment_arm, item.reference_arm) == (BfclV4PilotArm.FULL, BfclV4PilotArm.PURE)
    )
    assert (full_pure.wins, full_pure.ties, full_pure.losses) == (14, 10, 0)
    assert full_pure.exact_delta == BfclV4ExactRational(numerator=7, denominator=12)
    assert full_pure.strictly_exceeds_positive_ten_percentage_points is True
    assert len(full_pure.task_clusters) == 8
    assert all(len(item.treatment_correctness) == 3 for item in full_pure.task_clusters)
    assert not any(isinstance(value, float) for value in _all_values(result.model_dump()))


def test_campaign_output_states_public_descriptive_and_provider_identity_limits() -> None:
    result = compute_bfcl_v4_public_campaign_descriptive_analysis(_analysis_input())

    assert result.task_cluster_count == 8
    assert result.stochastic_search_replicates == 3
    assert result.task_seed_cell_count == 24
    assert result.task_seed_cells_are_independent_inferential_units is False
    assert result.greater_than_ten_pp_is_descriptive_only is True
    assert result.confidence_interval_available is False
    assert result.statistical_significance_claimed is False
    assert result.multiplicity_adjusted_inference_available is False
    assert result.hidden_test_evidence is False
    assert result.reportable_result is False
    assert result.frozen_spec_equality_does_not_attest_same_provider_weights is True
    assert result.provider_identity_observation_counts == (0, 0, 0)
    assert result.provider_declared_identity_consistent_within_replicates == (
        False,
        False,
        False,
    )
    assert result.all_call_response_identity_coverage_complete is False
    assert result.provider_declared_identity_coordinates_not_exposed is True
    assert result.provider_declared_identity_cross_replicate_consistency_attested is False
    assert result.same_provider_weights_attested is False


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"replicates": None}, "seed, order, or plan"),
        ({"ordered_closure_refs": None}, "closure refs differ"),
        ({"backend_fingerprint": "another-backend"}, "model, backend, or inference"),
    ),
)
def test_campaign_input_fails_closed_on_order_closure_or_execution_drift(update, message) -> None:
    original = _analysis_input()
    values = dict(update)
    if "replicates" in values:
        values["replicates"] = tuple(reversed(original.replicates))
    if "ordered_closure_refs" in values:
        values["ordered_closure_refs"] = tuple(reversed(original.ordered_closure_refs))
    forged = original.model_copy(update=values)

    with pytest.raises(ValidationError, match=message):
        BfclV4PublicCampaignAnalysisInput.model_validate(forged, strict=True)
    with pytest.raises(BfclV4PublicCampaignAnalysisError, match="strict revalidation"):
        compute_bfcl_v4_public_campaign_descriptive_analysis(forged)


def test_replicate_rejects_foreign_metrics_selection_or_closure_lineage() -> None:
    analysis_input = _analysis_input()
    first, second = analysis_input.replicates[:2]

    for update, message in (
        ({"descriptive_metrics": second.descriptive_metrics}, "descriptive-metrics lineage"),
        ({"joint_selection": second.joint_selection}, "joint-selection lineage"),
        ({"closure_ref": second.closure_ref}, "closure verification lineage"),
    ):
        forged = first.model_copy(update=update)
        with pytest.raises(ValidationError, match=message):
            BfclV4CampaignReplicateAnalysisInput.model_validate(forged, strict=True)

    forged_identity = first.model_copy(update={"provider_declared_identity_consistent": True})
    with pytest.raises(ValidationError, match="requires an observed response"):
        BfclV4CampaignReplicateAnalysisInput.model_validate(forged_identity, strict=True)


def test_campaign_input_rejects_foreign_plan_and_duplicate_verified_closure() -> None:
    analysis_input = _analysis_input()
    first, second, third = analysis_input.replicates
    foreign_plan = second.model_copy(
        update={
            "ordinal": first.ordinal,
            "replicate_id": first.replicate_id,
            "outer_seed_u64": first.outer_seed_u64,
        }
    )
    checked_foreign_plan = BfclV4CampaignReplicateAnalysisInput.model_validate(
        foreign_plan, strict=True
    )
    forged_plan_input = analysis_input.model_copy(
        update={"replicates": (checked_foreign_plan, second, third)}
    )
    with pytest.raises(ValidationError, match="seed, order, or plan"):
        BfclV4PublicCampaignAnalysisInput.model_validate(forged_plan_input, strict=True)

    duplicate_verification = second.closure_verification.model_copy(
        update={"closure_fingerprint": first.closure_ref.sha256}
    )
    duplicate_second = second.model_copy(
        update={
            "closure_ref": first.closure_ref,
            "closure_verification": duplicate_verification,
        }
    )
    checked_duplicate = BfclV4CampaignReplicateAnalysisInput.model_validate(
        duplicate_second, strict=True
    )
    duplicate_replicates = (first, checked_duplicate, third)
    duplicate_input = analysis_input.model_copy(
        update={
            "replicates": duplicate_replicates,
            "ordered_closure_refs": tuple(item.closure_ref for item in duplicate_replicates),
        }
    )
    with pytest.raises(ValidationError, match="three distinct 100-call closure refs"):
        BfclV4PublicCampaignAnalysisInput.model_validate(duplicate_input, strict=True)


def test_campaign_input_binds_the_complete_same_attempt_budget_per_replicate() -> None:
    analysis_input = _analysis_input()
    foreign_budget = AttemptBudget(
        max_attempts=100,
        max_total_tokens=102_400,
        max_tokens_per_attempt=1_024,
    )
    foreign_replicate = analysis_input.replicates[0].model_copy(
        update={
            "attempt_budget": foreign_budget,
            "attempt_budget_fingerprint": foreign_budget.fingerprint,
        }
    )
    forged = analysis_input.model_copy(
        update={"replicates": (foreign_replicate, *analysis_input.replicates[1:])}
    )

    with pytest.raises(ValidationError, match="execution identity and budget"):
        BfclV4PublicCampaignAnalysisInput.model_validate(forged, strict=True)

    incomplete = AttemptBudget(
        max_attempts=100,
        max_total_tokens=51_199,
        max_tokens_per_attempt=512,
    )
    with pytest.raises(ValidationError, match="100 complete equal per-call ceilings"):
        BfclV4CampaignPerCallBudget(
            max_output_tokens=256,
            attempt_budget=incomplete,
            attempt_budget_fingerprint=incomplete.fingerprint,
        )


def test_provider_identity_projection_can_record_coverage_without_attesting_weights() -> None:
    analysis_input = _analysis_input()
    replicates = tuple(
        item.model_copy(
            update={
                "provider_identity_observation_count": 100,
                "provider_declared_identity_consistent": True,
            }
        )
        for item in analysis_input.replicates
    )
    projected = analysis_input.model_copy(
        update={
            "replicates": replicates,
            "all_call_response_identity_coverage_complete": True,
        }
    )
    checked = BfclV4PublicCampaignAnalysisInput.model_validate(projected, strict=True)
    result = compute_bfcl_v4_public_campaign_descriptive_analysis(checked)

    assert result.provider_identity_observation_counts == (100, 100, 100)
    assert result.provider_declared_identity_consistent_within_replicates == (True, True, True)
    assert result.all_call_response_identity_coverage_complete is True
    assert result.provider_declared_identity_cross_replicate_consistency_attested is False
    assert result.same_provider_weights_attested is False


def test_exact_rational_contract_rejects_float_or_unreduced_values() -> None:
    assert BfclV4ExactRational.from_ratio(-6, 24) == BfclV4ExactRational(
        numerator=-1, denominator=4
    )
    with pytest.raises(ValidationError, match="canonical reduced"):
        BfclV4ExactRational(numerator=2, denominator=4)
    with pytest.raises(ValidationError):
        BfclV4ExactRational(numerator=1.0, denominator=2)


def test_analysis_output_rejects_paired_vectors_detached_from_arm_clusters() -> None:
    result = compute_bfcl_v4_public_campaign_descriptive_analysis(_analysis_input())
    first = result.paired_deltas[0]
    forged_cluster = first.task_clusters[0].model_copy(
        update={"treatment_correctness": (False, False, False)}
    )
    forged_delta = first.model_copy(
        update={"task_clusters": (forged_cluster, *first.task_clusters[1:])}
    )
    forged = result.model_copy(update={"paired_deltas": (forged_delta, *result.paired_deltas[1:])})

    with pytest.raises(ValidationError):
        BfclV4PublicCampaignDescriptiveAnalysis.model_validate(forged, strict=True)
