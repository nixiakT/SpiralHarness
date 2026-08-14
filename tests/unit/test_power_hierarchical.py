from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from spiral_harness.experiments.power_bootstrap_statistics import holm_three
from spiral_harness.experiments.power_hierarchical import (
    HYPOTHESIS_ORDER,
    DatasetSource,
    EndpointHierarchyPlan,
    EndpointNestedInference,
    HierarchicalBootstrapPreregistration,
    HierarchicalFamilyInference,
    HierarchicalHypothesis,
    HierarchicalPairedDataset,
    PairedLeafObservation,
    PreregistrationPurpose,
    TaskGroupPlan,
    TaskHierarchyPlan,
    analyze_hierarchical_dataset,
)
from spiral_harness.experiments.power_nested_generation import simulate_nested_dataset
from spiral_harness.experiments.power_nested_simulation import (
    ContinuousHierarchyAssumption,
    HierarchicalPowerSimulationConfig,
    HierarchicalPowerSimulationReport,
    MonteCarloWilsonRate,
    PairedBinaryAssumption,
    PowerSimulationPurpose,
    SharedContinuousCorrelationAssumption,
    _wilson_rate,
    simulate_hierarchical_power,
)


def _task(task_id: str = "task") -> TaskHierarchyPlan:
    return TaskHierarchyPlan(
        task_id=task_id,
        groups=(
            TaskGroupPlan(group_id="group-b", repeat_seeds=(11, 10)),
            TaskGroupPlan(group_id="group-a", repeat_seeds=(11, 10)),
        ),
    )


def _endpoints() -> tuple[EndpointHierarchyPlan, ...]:
    real_tasks = (_task("real-task"),)
    return (
        EndpointHierarchyPlan(
            hypothesis=HierarchicalHypothesis.FULL_VS_SCORE_REAL,
            endpoint_kind="normalized-score-difference",
            sesoi=0.02,
            tasks=real_tasks,
        ),
        EndpointHierarchyPlan(
            hypothesis=HierarchicalHypothesis.FULL_VS_STATIC_REAL,
            endpoint_kind="normalized-score-difference",
            sesoi=0.02,
            tasks=real_tasks,
        ),
        EndpointHierarchyPlan(
            hypothesis=HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
            endpoint_kind="paired-binary-difference",
            sesoi=0.15,
            tasks=(_task("repair-stratum"),),
        ),
    )


def _preregistration(
    *,
    bootstrap_seed: int = 17,
    bootstrap_draws: int = 200,
    purpose: PreregistrationPurpose = PreregistrationPurpose.DETERMINISTIC_TEST_FIXTURE,
    search_seeds: tuple[int, ...] = (3, 1, 2),
    endpoints: tuple[EndpointHierarchyPlan, ...] | None = None,
) -> HierarchicalBootstrapPreregistration:
    return HierarchicalBootstrapPreregistration(
        study_id="hierarchical-test",
        purpose=purpose,
        model_ids=("model-b", "model-a"),
        independent_search_seeds=search_seeds,
        endpoints=_endpoints() if endpoints is None else endpoints,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )


Difference = Callable[[HierarchicalHypothesis, int, str, int], float]


def _dataset(
    preregistration: HierarchicalBootstrapPreregistration,
    difference: Difference,
    *,
    source: DatasetSource = DatasetSource.SYNTHETIC_VALIDATION,
) -> HierarchicalPairedDataset:
    observations = tuple(
        PairedLeafObservation(
            hypothesis=endpoint.hypothesis,
            model_id=model_id,
            task_id=task.task_id,
            search_seed=search_seed,
            group_id=group.group_id,
            repeat_seed=repeat_seed,
            difference=difference(
                endpoint.hypothesis,
                search_seed,
                group.group_id,
                repeat_seed,
            ),
        )
        for endpoint in preregistration.endpoints
        for model_id in preregistration.model_ids
        for task in endpoint.tasks
        for search_seed in preregistration.independent_search_seeds
        for group in task.groups
        for repeat_seed in group.repeat_seeds
    )
    return HierarchicalPairedDataset(
        preregistration_fingerprint=preregistration.fingerprint,
        source=source,
        observations=observations,
    )


def _constant_difference(
    hypothesis: HierarchicalHypothesis,
    _search_seed: int,
    _group_id: str,
    _repeat_seed: int,
) -> float:
    return 1.0 if hypothesis is HYPOTHESIS_ORDER[2] else 0.2


def _variable_difference(
    hypothesis: HierarchicalHypothesis,
    search_seed: int,
    group_id: str,
    repeat_seed: int,
) -> float:
    if hypothesis is HYPOTHESIS_ORDER[2]:
        return 1.0 if (search_seed + repeat_seed) % 2 else 0.0
    base = {1: -0.1, 2: 0.11, 3: 0.32}[search_seed]
    group_offset = -0.02 if group_id == "group-a" else 0.02
    repeat_offset = -0.01 if repeat_seed == 10 else 0.01
    return base + group_offset + repeat_offset


def _sesoi_boundary_difference(
    hypothesis: HierarchicalHypothesis,
    _search_seed: int,
    _group_id: str,
    _repeat_seed: int,
) -> float:
    return 1.0 if hypothesis is HYPOTHESIS_ORDER[2] else 0.02


def _uncertain_above_sesoi_difference(
    hypothesis: HierarchicalHypothesis,
    search_seed: int,
    _group_id: str,
    _repeat_seed: int,
) -> float:
    if hypothesis is HYPOTHESIS_ORDER[2]:
        return 1.0
    return {1: -0.3, 2: 0.03, 3: 0.36}[search_seed]


def test_nested_bootstrap_is_deterministic_for_one_frozen_artifact() -> None:
    preregistration = _preregistration()
    dataset = _dataset(preregistration, _variable_difference)

    first = analyze_hierarchical_dataset(preregistration, dataset)
    second = analyze_hierarchical_dataset(preregistration, dataset)

    assert first == second


def test_bootstrap_seed_changes_nonconstant_bootstrap_output() -> None:
    first_preregistration = _preregistration(bootstrap_seed=17)
    second_preregistration = _preregistration(bootstrap_seed=18)
    first = analyze_hierarchical_dataset(
        first_preregistration,
        _dataset(first_preregistration, _variable_difference),
    )
    second = analyze_hierarchical_dataset(
        second_preregistration,
        _dataset(second_preregistration, _variable_difference),
    )

    assert tuple(item.standard_error for item in first.endpoints) != tuple(
        item.standard_error for item in second.endpoints
    )


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    (
        ("task_id", "foreign-task"),
        ("search_seed", 999),
        ("group_id", "foreign-group"),
        ("repeat_seed", 999),
    ),
)
def test_complete_roster_rejects_foreign_and_missing_hierarchy_leaf(
    field: str,
    foreign_value: str | int,
) -> None:
    preregistration = _preregistration()
    dataset = _dataset(preregistration, _constant_difference)
    replacement = dataset.observations[0].model_copy(update={field: foreign_value})
    altered = HierarchicalPairedDataset(
        preregistration_fingerprint=preregistration.fingerprint,
        source=dataset.source,
        observations=(replacement, *dataset.observations[1:]),
    )

    with pytest.raises(ValueError, match=r"missing=1, foreign=1"):
        analyze_hierarchical_dataset(preregistration, altered)


def test_complete_roster_rejects_a_missing_leaf() -> None:
    preregistration = _preregistration()
    dataset = _dataset(preregistration, _constant_difference)
    missing = HierarchicalPairedDataset(
        preregistration_fingerprint=preregistration.fingerprint,
        source=dataset.source,
        observations=dataset.observations[:-1],
    )

    with pytest.raises(ValueError, match=r"missing=1, foreign=0"):
        analyze_hierarchical_dataset(preregistration, missing)


def test_dataset_rejects_a_duplicate_leaf_coordinate() -> None:
    preregistration = _preregistration()
    dataset = _dataset(preregistration, _constant_difference)

    with pytest.raises(ValidationError, match="must not duplicate"):
        HierarchicalPairedDataset(
            preregistration_fingerprint=preregistration.fingerprint,
            source=dataset.source,
            observations=(*dataset.observations, dataset.observations[0]),
        )


def test_binary_endpoint_rejects_a_continuous_difference() -> None:
    with pytest.raises(ValidationError, match="must be -1, 0, or 1"):
        PairedLeafObservation(
            hypothesis=HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
            model_id="model",
            task_id="task",
            search_seed=1,
            group_id="group",
            repeat_seed=1,
            difference=0.5,
        )


def test_real_endpoints_must_share_the_exact_hierarchy() -> None:
    endpoints = list(_endpoints())
    endpoints[1] = EndpointHierarchyPlan(
        hypothesis=HierarchicalHypothesis.FULL_VS_STATIC_REAL,
        endpoint_kind="normalized-score-difference",
        sesoi=0.02,
        tasks=(_task("another-real-task"),),
    )

    with pytest.raises(ValidationError, match="must share one exact task hierarchy"):
        _preregistration(endpoints=tuple(endpoints))


@pytest.mark.parametrize(
    "purpose",
    (
        PreregistrationPurpose.DISJOINT_PILOT_ANALYSIS,
        PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION,
    ),
)
def test_formal_analysis_rejects_fewer_than_twenty_thousand_draws(
    purpose: PreregistrationPurpose,
) -> None:
    with pytest.raises(ValidationError, match="at least 20,000"):
        _preregistration(purpose=purpose, bootstrap_draws=19_999)


def test_deterministic_fixture_allows_two_hundred_draws() -> None:
    assert _preregistration(bootstrap_draws=200).bootstrap_draws == 200


def test_formal_prototype_rejects_two_search_seed_clusters_without_coverage_claim() -> None:
    with pytest.raises(ValidationError, match="fewer than four"):
        _preregistration(
            purpose=PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION,
            bootstrap_draws=20_000,
            search_seeds=(1, 2),
        )


def test_holm_uses_the_frozen_tie_break_and_controls_the_full_family() -> None:
    adjusted, rejected = holm_three((0.01, 0.04, 0.03), alpha=0.05)

    assert adjusted == pytest.approx((0.03, 0.06, 0.06))
    assert rejected == (True, False, False)


def test_degenerate_estimate_at_sesoi_boundary_has_p_one() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _sesoi_boundary_difference),
    )

    for endpoint in inference.endpoints[:2]:
        assert endpoint.estimate == pytest.approx(endpoint.sesoi)
        assert endpoint.one_sided_sesoi_null_p_value == 1.0
        assert endpoint.holm_rejected is False
    assert inference.all_primary_margin_tests_passed is False


def test_estimate_above_sesoi_with_boundary_crossing_uncertainty_fails_margin_test() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _uncertain_above_sesoi_difference),
    )

    for endpoint in inference.endpoints[:2]:
        assert endpoint.estimate == pytest.approx(0.03)
        assert endpoint.point_estimate_exceeds_sesoi is True
        assert endpoint.marginal_two_sided_lower < endpoint.sesoi
        assert endpoint.holm_rejected is False
    assert inference.all_primary_margin_tests_passed is False


def test_point_estimate_over_ten_percent_is_not_itself_a_margin_claim() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _variable_difference),
    )

    for endpoint in inference.endpoints[:2]:
        assert endpoint.estimate == pytest.approx(0.11)
        assert endpoint.estimate > 0.10
        assert endpoint.project_margin_familywise_one_sided_lower is not None
        assert endpoint.project_margin_familywise_one_sided_lower <= 0.10
        assert endpoint.project_margin_claim_supported is False


def test_h1_h2_margin_claim_requires_the_two_endpoint_simultaneous_lower_bound() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _constant_difference),
    )

    assert all(item.project_margin_claim_supported for item in inference.endpoints[:2])
    assert all(
        item.project_margin_familywise_one_sided_lower is not None
        and item.project_margin_familywise_one_sided_lower > 0.10
        for item in inference.endpoints[:2]
    )
    assert inference.endpoints[2].project_margin_applicable is False
    assert inference.endpoints[2].project_margin_familywise_one_sided_lower is None
    assert inference.endpoints[2].project_margin_claim_supported is False
    assert inference.all_project_margin_claims_supported is True


def test_synthetic_results_cannot_masquerade_as_confirmatory_results() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _constant_difference),
    )

    assert inference.dataset_source is DatasetSource.SYNTHETIC_VALIDATION
    assert inference.synthetic_validation_only is True
    assert inference.confirmatory_result is False
    assert inference.reportable_sealed_evidence is False
    assert inference.verified_sealed_evidence is False
    assert inference.attested_sealed_closure_verified is False
    assert inference.formal_confirmatory_analysis_blocked is True
    assert inference.coverage_calibrated is False


def test_family_output_recomputes_p_value_to_holm_bindings() -> None:
    preregistration = _preregistration()
    inference = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _constant_difference),
    )
    payload = inference.model_dump(mode="python")
    payload["endpoints"][0]["one_sided_sesoi_null_p_value"] = 1.0
    payload["endpoints"][0]["holm_adjusted_p_value"] = 0.001
    payload["endpoints"][0]["holm_rejected"] = True

    with pytest.raises(ValidationError, match="Holm"):
        HierarchicalFamilyInference.model_validate(payload, strict=True)


def test_endpoint_output_binds_sesoi_and_ten_percent_applicability() -> None:
    preregistration = _preregistration()
    endpoint = analyze_hierarchical_dataset(
        preregistration,
        _dataset(preregistration, _constant_difference),
    ).endpoints[2]
    payload = endpoint.model_dump(mode="python")
    payload["project_margin_applicable"] = True
    payload["project_margin_familywise_one_sided_lower"] = 0.2
    payload["project_margin_claim_supported"] = True

    with pytest.raises(ValidationError, match="applicability"):
        EndpointNestedInference.model_validate(payload, strict=True)


def test_caller_cannot_label_a_dataset_as_confirmatory() -> None:
    preregistration = _preregistration()
    payload = _dataset(preregistration, _constant_difference).model_dump(mode="python")
    payload["source"] = "confirmatory"

    with pytest.raises(ValidationError, match="source"):
        HierarchicalPairedDataset.model_validate(payload, strict=True)


def test_preregistration_fingerprint_is_canonical_for_set_like_rosters() -> None:
    first = _preregistration(search_seeds=(3, 1, 2))
    second = HierarchicalBootstrapPreregistration(
        study_id=first.study_id,
        purpose=first.purpose,
        model_ids=tuple(reversed(first.model_ids)),
        independent_search_seeds=(2, 3, 1),
        endpoints=tuple(reversed(first.endpoints)),
        bootstrap_draws=first.bootstrap_draws,
        bootstrap_seed=first.bootstrap_seed,
    )

    assert first == second
    assert first.fingerprint == second.fingerprint


def _power_config(
    *,
    simulation_seed: int = 91,
    candidates: tuple[tuple[int, ...], ...] = ((1, 2), (1, 2, 3)),
    purpose: PowerSimulationPurpose = PowerSimulationPurpose.DETERMINISTIC_TEST_FIXTURE,
    bootstrap_draws: int = 200,
    null_replicates: int = 20,
    calibration_replicates: int = 20,
    validation_replicates: int = 20,
    first_effect: float = 0.3,
    maximum_leaf_value_lookups: int = 100_000_000,
) -> HierarchicalPowerSimulationConfig:
    return HierarchicalPowerSimulationConfig(
        study_id="nested-power-test",
        purpose=purpose,
        assumption_source_label="unit-test synthetic assumptions",
        assumption_artifact_sha256="a" * 64,
        model_ids=("model",),
        endpoints=_endpoints(),
        candidate_search_seed_sets=candidates,
        continuous_assumptions=(
            ContinuousHierarchyAssumption(
                hypothesis=HierarchicalHypothesis.FULL_VS_STATIC_REAL,
                alternative_effect=0.3,
                search_seed_sd=0.005,
                task_group_sd=0.005,
                repeat_sd=0.005,
            ),
            ContinuousHierarchyAssumption(
                hypothesis=HierarchicalHypothesis.FULL_VS_SCORE_REAL,
                alternative_effect=first_effect,
                search_seed_sd=0.005,
                task_group_sd=0.005,
                repeat_sd=0.005,
            ),
        ),
        shared_continuous_correlation=SharedContinuousCorrelationAssumption(
            search_seed_correlation=0.5,
            task_group_correlation=0.25,
            repeat_correlation=0.1,
        ),
        paired_binary_assumption=PairedBinaryAssumption(
            hypothesis=HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
            alternative_positive_probability=1.0,
            alternative_negative_probability=0.0,
            null_discordant_probability=0.15,
        ),
        analysis_bootstrap_draws=bootstrap_draws,
        analysis_bootstrap_seed=23,
        null_audit_replicates=null_replicates,
        alternative_calibration_replicates=calibration_replicates,
        alternative_validation_replicates=validation_replicates,
        simulation_seed=simulation_seed,
        target_power=0.6,
        maximum_leaf_value_lookups=maximum_leaf_value_lookups,
    )


def test_least_favourable_null_theory_and_generated_roster() -> None:
    preregistration = _preregistration()
    continuous = tuple(
        ContinuousHierarchyAssumption(
            hypothesis=hypothesis,
            alternative_effect=0.3,
            search_seed_sd=0.0,
            task_group_sd=0.0,
            repeat_sd=0.0,
        )
        for hypothesis in HYPOTHESIS_ORDER[:2]
    )
    correlation = SharedContinuousCorrelationAssumption(
        search_seed_correlation=0.0,
        task_group_correlation=0.0,
        repeat_correlation=0.0,
    )
    binary = PairedBinaryAssumption(
        hypothesis=HYPOTHESIS_ORDER[2],
        alternative_positive_probability=1.0,
        alternative_negative_probability=0.0,
        null_discordant_probability=0.15,
    )

    assert binary.null_positive_probability == pytest.approx(0.15)
    assert binary.null_negative_probability == pytest.approx(0.0)
    dataset = simulate_nested_dataset(
        preregistration,
        continuous,
        correlation,
        binary,
        alternative=False,
        seed=123,
    )
    real = [
        item.difference for item in dataset.observations if item.hypothesis in HYPOTHESIS_ORDER[:2]
    ]
    repair = [
        item.difference for item in dataset.observations if item.hypothesis is HYPOTHESIS_ORDER[2]
    ]
    assert set(real) == {0.02}
    assert set(repair).issubset({0.0, 1.0})
    assert sum(repair) / len(repair) == pytest.approx(0.15, abs=0.10)


def test_binary_null_discordance_must_support_the_point_fifteen_boundary() -> None:
    with pytest.raises(ValidationError, match=r"greater than or equal to 0\.15"):
        PairedBinaryAssumption(
            hypothesis=HYPOTHESIS_ORDER[2],
            alternative_positive_probability=1.0,
            alternative_negative_probability=0.0,
            null_discordant_probability=0.14,
        )


def test_wilson_output_rejects_fake_one_of_one_certainty() -> None:
    with pytest.raises(ValidationError, match="Wilson bounds differ"):
        MonteCarloWilsonRate(
            successes=1,
            replicates=1,
            estimate=1.0,
            one_sided_confidence_level=0.95,
            one_sided_lower=1.0,
            one_sided_upper=1.0,
        )


def test_power_simulation_is_seeded_reproducible_and_selects_smallest_candidate() -> None:
    config = _power_config()

    first = simulate_hierarchical_power(config)
    second = simulate_hierarchical_power(config)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.conditional_selection_succeeded is True
    assert first.calibration_selected_search_seeds == (1, 2)
    assert first.conditional_recommended_search_seeds == (1, 2)
    assert first.selected_candidate_validation_statistical_endpoint_pass_power is not None
    assert first.selected_candidate_validation_statistical_endpoint_pass_power.replicates == 20
    assert tuple(item.independent_search_seeds for item in first.candidate_diagnostics) == (
        (1, 2),
        (1, 2, 3),
    )
    assert first.conditional_on_declared_simulation_assumptions is True
    assert first.synthetic_validation_only is True
    assert first.observed_harness_effect_evidence is False
    assert first.unconditional_power_guarantee is False
    assert first.calibration_validation_and_audit_streams_domain_separated is True
    assert first.pilot_closure_verified is False
    assert first.attested_sealed_closure_verified is False
    assert first.reportable_sealed_evidence is False
    assert first.verified_sealed_evidence is False
    assert first.formal_design_ready is False
    assert first.formal_sample_size_frozen is False
    assert first.coverage_calibrated is False
    assert first.partial_null_fwer_audit_implemented is False
    assert first.protected_slice_pass_included is False
    assert first.cost_pass_included is False
    assert first.provenance_pass_included is False
    assert first.estimated_leaf_value_lookups <= first.leaf_value_lookup_budget
    for candidate in first.candidate_diagnostics:
        assert candidate.joint_h3_dependence_assumed_independent is True
        assert candidate.calibration_statistical_endpoint_pass_power.replicates == 20
        assert candidate.passes_calibration_threshold is True
        assert candidate.null_holm_fwer_audit.one_sided_lower <= (
            candidate.null_holm_fwer_audit.estimate
        )
        assert candidate.null_holm_fwer_audit.one_sided_upper >= (
            candidate.null_holm_fwer_audit.estimate
        )
    payload = first.model_dump(mode="python")
    payload["candidate_diagnostics"][0]["null_holm_fwer_audit"] = _wilson_rate(
        0,
        1,
        0.95,
    ).model_dump(mode="python")
    payload["candidate_diagnostics"][0]["null_fwer_exceeds_nominal_point_estimate"] = False
    with pytest.raises(ValidationError, match="wrong replicate count"):
        HierarchicalPowerSimulationReport.model_validate(payload, strict=True)


def test_candidate_search_seed_sets_are_frozen_as_exact_prefixes() -> None:
    canonical = _power_config(candidates=((3, 2, 1), (2, 1)))
    assert canonical.candidate_search_seed_sets == ((1, 2), (1, 2, 3))

    with pytest.raises(ValidationError, match="exact prefix"):
        _power_config(candidates=((1, 2), (1, 3, 4)))


def test_ten_percent_effect_cannot_size_a_strictly_greater_ten_percent_claim() -> None:
    with pytest.raises(ValidationError, match=r"greater than 0\.1"):
        _power_config(first_effect=0.1)


def test_frozen_power_design_enforces_formal_draw_and_replicate_floors() -> None:
    with pytest.raises(ValidationError, match="at least 20,000"):
        _power_config(purpose=PowerSimulationPurpose.PREREGISTERED_PROSPECTIVE_SIMULATION)

    with pytest.raises(ValidationError, match="at least 1,000"):
        _power_config(
            purpose=PowerSimulationPurpose.PREREGISTERED_PROSPECTIVE_SIMULATION,
            bootstrap_draws=20_000,
            maximum_leaf_value_lookups=10**13,
        )

    with pytest.raises(ValidationError, match="small search-seed"):
        _power_config(
            purpose=PowerSimulationPurpose.PREREGISTERED_PROSPECTIVE_SIMULATION,
            bootstrap_draws=20_000,
            null_replicates=1_000,
            calibration_replicates=1_000,
            validation_replicates=1_000,
            maximum_leaf_value_lookups=10**13,
        )


def test_simulation_workload_budget_fails_closed() -> None:
    with pytest.raises(ValidationError, match="exceeds its leaf-lookup budget"):
        _power_config(maximum_leaf_value_lookups=1)
