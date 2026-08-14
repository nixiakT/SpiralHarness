"""Analytic sensitivity proxy for prospective search-run sample sizes.

This bounded-Gaussian proxy stress-tests assumptions before the exact
hierarchical and nested-bootstrap pipeline exists; it cannot freeze sample size.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.power_joint import (
    JointPrimarySensitivityAssumptions,
    gaussian_copula_resample,
)
from spiral_harness.experiments.power_pilot import (
    DeclaredPowerPilotManifestExpectation,
    DigestBoundPowerPilotManifest,
    bind_optional_declared_power_pilot_manifest,
    validate_digest_bound_report_claim,
)

REAL_SCORE_SESOI = 0.02
VERIFIED_REPAIR_SESOI = 0.15
PROJECT_DISPLAY_MARGIN = 0.10
FAMILY_WISE_ALPHA = 0.05


class SensitivityPurpose(StrEnum):
    ANALYTIC_SENSITIVITY = "analytic-sensitivity"
    DETERMINISTIC_TEST_FIXTURE = "deterministic-test-fixture"


class PrimaryHypothesis(StrEnum):
    FULL_VS_SCORE_REAL = "full-vs-score-real"
    FULL_VS_STATIC_REAL = "full-vs-static-real"
    FULL_VS_SCORE_VERIFIED_REPAIR = "full-vs-score-verified-repair"


_PRIMARY_ORDER = (
    PrimaryHypothesis.FULL_VS_SCORE_REAL,
    PrimaryHypothesis.FULL_VS_STATIC_REAL,
    PrimaryHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
)
_EXPECTED_SESOI = {
    PrimaryHypothesis.FULL_VS_SCORE_REAL: REAL_SCORE_SESOI,
    PrimaryHypothesis.FULL_VS_STATIC_REAL: REAL_SCORE_SESOI,
    PrimaryHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR: VERIFIED_REPAIR_SESOI,
}


class FailureSensitivityAssumptions(ImmutableModel):
    """One plug-in failure scenario; these values are not confidence bounds."""

    full_only_probability: Annotated[float, Field(ge=0.0, lt=1.0, strict=True)]
    control_only_probability: Annotated[float, Field(ge=0.0, lt=1.0, strict=True)]
    both_probability: Annotated[float, Field(ge=0.0, lt=1.0, strict=True)]
    full_only_difference: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
    control_only_difference: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
    both_difference: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]

    @model_validator(mode="after")
    def probabilities_leave_a_non_failure_stratum(self) -> FailureSensitivityAssumptions:
        if self.total_probability >= 1.0:
            raise ValueError("failure probabilities must sum to less than one")
        return self

    @property
    def total_probability(self) -> float:
        return self.full_only_probability + self.control_only_probability + self.both_probability


class EndpointSensitivityAssumptions(ImmutableModel):
    """Explicit plug-in variance, ICC, and failure values for one contrast."""

    hypothesis: PrimaryHypothesis
    assumption_artifact_sha256: Sha256
    effect_threshold: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    rollout_difference_variance: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    search_seed_icc: Annotated[float, Field(ge=0.0, lt=1.0, strict=True)]
    task_group_icc: Annotated[float, Field(ge=0.0, lt=1.0, strict=True)]
    cell_effect_sd: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    task_groups_per_search_run: Annotated[int, Field(ge=2, strict=True)]
    rollouts_per_task_group: Annotated[int, Field(ge=1, strict=True)]
    failure_scenario: FailureSensitivityAssumptions

    @model_validator(mode="after")
    def validate_hierarchy_and_frozen_margin(self) -> EndpointSensitivityAssumptions:
        if self.task_group_icc < self.search_seed_icc:
            raise ValueError("task_group_icc must be at least search_seed_icc")
        expected = _EXPECTED_SESOI[self.hypothesis]
        if self.effect_threshold != expected:
            raise ValueError(
                f"{self.hypothesis.value} effect_threshold must remain frozen at {expected}"
            )
        failure_contribution = (
            self.failure_scenario.full_only_probability * self.failure_scenario.full_only_difference
            + self.failure_scenario.control_only_probability
            * self.failure_scenario.control_only_difference
            + self.failure_scenario.both_probability * self.failure_scenario.both_difference
        )
        non_failure_probability = 1.0 - self.failure_scenario.total_probability
        for label, desired_effect in (("null", 0.0), ("design", expected)):
            mean = (desired_effect - failure_contribution) / non_failure_probability
            if not -1.0 <= mean <= 1.0:
                raise ValueError(f"failure scenario implies an impossible {label} non-failure mean")
        return self

    @property
    def search_seed_variance(self) -> float:
        return self.rollout_difference_variance * self.search_seed_icc

    @property
    def averaged_task_group_variance(self) -> float:
        group_fraction = self.task_group_icc - self.search_seed_icc
        return self.rollout_difference_variance * group_fraction / self.task_groups_per_search_run

    @property
    def averaged_rollout_variance(self) -> float:
        residual_fraction = 1.0 - self.task_group_icc
        return (
            self.rollout_difference_variance
            * residual_fraction
            / (self.task_groups_per_search_run * self.rollouts_per_task_group)
        )


class PowerSensitivityConfig(ImmutableModel):
    """Content-addressable inputs to a non-authoritative sensitivity run."""

    schema_version: Literal["4"] = "4"
    study_id: NonEmptyStr
    purpose: SensitivityPurpose
    sensitivity_scenario_label: NonEmptyStr
    declared_pilot_manifest_expectation: DeclaredPowerPilotManifestExpectation | None = None
    model_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=3, max_length=3)]
    real_task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=4, max_length=4)]
    repair_strata: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    real_cell_weighting: Literal["equal-model-by-task-cell"]
    verified_repair_cell_weighting: Literal["equal-model-by-repair-stratum-cell"]
    real_condition_count: Literal[5]
    search_seed_counts_per_cell: Annotated[tuple[int, ...], Field(min_length=1)]
    calibration_null_replicates: Annotated[int, Field(ge=500, strict=True)]
    audit_null_replicates: Annotated[int, Field(ge=500, strict=True)]
    design_effect_replicates: Annotated[int, Field(ge=500, strict=True)]
    simulation_seed: Annotated[int, Field(ge=0, strict=True)]
    analysis_method: Literal["bounded-equal-cell-rank-copula-analytic-proxy"]
    family_wise_alpha: Literal[0.05]
    project_display_margin: Literal[0.1]
    joint_primary_assumptions: JointPrimarySensitivityAssumptions
    endpoint_assumptions: Annotated[
        tuple[EndpointSensitivityAssumptions, ...], Field(min_length=3, max_length=3)
    ]

    @field_validator("model_ids", "real_task_ids", "repair_strata")
    @classmethod
    def canonicalize_unique_rosters(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("sensitivity rosters must not contain duplicate labels")
        return tuple(sorted(values))

    @field_validator("search_seed_counts_per_cell")
    @classmethod
    def validate_search_seed_counts(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 2 for value in values):
            raise ValueError("each proxy point requires at least two complete search runs")
        if len(values) != len(set(values)):
            raise ValueError("search-seed counts must be unique")
        return tuple(sorted(values))

    @field_validator("endpoint_assumptions")
    @classmethod
    def canonicalize_primary_family(
        cls,
        values: tuple[EndpointSensitivityAssumptions, ...],
    ) -> tuple[EndpointSensitivityAssumptions, ...]:
        by_hypothesis = {value.hypothesis: value for value in values}
        if len(by_hypothesis) != len(values) or set(by_hypothesis) != set(_PRIMARY_ORDER):
            raise ValueError("endpoint_assumptions must contain each primary hypothesis once")
        return tuple(by_hypothesis[hypothesis] for hypothesis in _PRIMARY_ORDER)

    @model_validator(mode="after")
    def fixture_cannot_bind_declared_pilot_metadata(self) -> PowerSensitivityConfig:
        if (
            self.purpose is SensitivityPurpose.DETERMINISTIC_TEST_FIXTURE
            and self.declared_pilot_manifest_expectation is not None
        ):
            raise ValueError("a deterministic test fixture cannot bind declared pilot metadata")
        return self


class MonteCarloProxyRate(ImmutableModel):
    estimate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    standard_error: Annotated[float, Field(ge=0.0, strict=True)]
    replicates: Annotated[int, Field(ge=1, strict=True)]


class EndpointSensitivityDiagnostic(ImmutableModel):
    hypothesis: PrimaryHypothesis
    target_sesoi: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    proxy_mean_estimate_at_design_effect: Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
    proxy_mean_minus_target: Annotated[float, Field(ge=-2.0, le=2.0, strict=True)]
    proxy_holm_rejection_rate_at_sesoi: MonteCarloProxyRate
    point_estimate_exceeds_sesoi_rate: MonteCarloProxyRate
    point_estimate_exceeds_project_display_0_10_rate: MonteCarloProxyRate


class BoundedGenerationDiagnostic(ImmutableModel):
    method: Literal["winsorized-gaussian-random-effects-proxy"]
    support: Literal["[-1,1]"]
    calibration_null_clipping_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    audit_null_clipping_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    design_effect_clipping_rate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]


class CandidateSensitivityDiagnostic(ImmutableModel):
    search_seeds_per_model_task_cell: Annotated[int, Field(ge=2, strict=True)]
    independent_unit: Literal["complete_end_to_end_search_run"]
    real_model_task_cells: Annotated[int, Field(ge=1, strict=True)]
    repair_model_stratum_cells: Annotated[int, Field(ge=1, strict=True)]
    real_grid_condition_run_slots: Annotated[int, Field(ge=1, strict=True)]
    verified_repair_cell_weighting: Literal["equal-model-by-repair-stratum-cell"]
    calibration_null_replicates: Annotated[int, Field(ge=500, strict=True)]
    audit_null_replicates: Annotated[int, Field(ge=500, strict=True)]
    independent_audit_bernoulli_replicates: Annotated[int, Field(ge=500, strict=True)]
    design_effect_replicates: Annotated[int, Field(ge=500, strict=True)]
    audit_null_proxy_holm_fwer_rate: MonteCarloProxyRate
    proxy_all_three_holm_rejection_rate_at_sesoi: MonteCarloProxyRate
    proxy_headline_pass_rate_at_sesoi: MonteCarloProxyRate
    bounded_generation: BoundedGenerationDiagnostic
    endpoint_diagnostics: Annotated[
        tuple[EndpointSensitivityDiagnostic, ...], Field(min_length=3, max_length=3)
    ]
    formal_sample_size_candidate: Literal[False]


class PowerSensitivityReport(ImmutableModel):
    """Approximation diagnostics that cannot authorize a formal design."""

    schema_version: Literal["4"] = "4"
    analysis_status: Literal["analytic-sensitivity-proxy-only"]
    config_sha256: Sha256
    config: PowerSensitivityConfig
    digest_bound_declared_pilot_manifest: DigestBoundPowerPilotManifest | None
    declared_pilot_manifest_digest_bound: bool
    pilot_artifact_existence_verified: Literal[False]
    pilot_artifact_authenticity_verified: Literal[False]
    pilot_disjointness_verified: Literal[False]
    pilot_closure_verified: Literal[False]
    candidate_diagnostics: Annotated[
        tuple[CandidateSensitivityDiagnostic, ...], Field(min_length=1)
    ]
    formal_required_search_seeds_per_cell: Literal[None] = None
    formal_design_ready: Literal[False]
    primary_hierarchical_pipeline_design_ready: Literal[False]
    exact_nested_bootstrap_implemented: Literal[False]
    formal_fwer_or_power_coverage_claimed: Literal[False]
    margin_confidence_support_claimed: Literal[False]
    project_display_0_10_is_verified_repair_primary_margin: Literal[False]
    approximation_warning: Literal[
        "Analytic rank-copula sensitivity proxy only; it assumes independent cell aggregates, "
        "treats repair as a continuous bounded endpoint, and cannot freeze a formal S."
    ]
    dependence_warning: Literal[
        "The copula matrix is a latent Gaussian rank parameter, not an exact covariance or "
        "shared-run dependence model."
    ]
    failure_warning: Literal[
        "Failure inputs are one plug-in sensitivity scenario, not a conservative uncertainty set."
    ]
    audit_warning: Literal[
        "Audit Bernoulli draws are independent conditional on separate empirical marginals; "
        "their Monte Carlo SE remains proxy-only."
    ]
    margin_warning: Literal[
        "The 0.10 display threshold is not confidence support and is not the 0.15 repair SESOI."
    ]
    pilot_provenance_warning: Literal[
        "Only canonical declared-manifest syntax and digest equality are checked; referenced "
        "artifact existence, content, authenticity, observation, disjointness, closure, and "
        "blinding are not verified."
    ]

    @model_validator(mode="after")
    def validate_digest_binding_and_config_consistency(self) -> PowerSensitivityReport:
        validate_digest_bound_report_claim(
            embedded_config_sha256=self.config_sha256,
            actual_config_sha256=canonical_sha256(self.config),
            fixture=self.config.purpose is SensitivityPurpose.DETERMINISTIC_TEST_FIXTURE,
            expectation=self.config.declared_pilot_manifest_expectation,
            binding=self.digest_bound_declared_pilot_manifest,
            digest_bound_claimed=self.declared_pilot_manifest_digest_bound,
        )
        return self


@dataclass(frozen=True, slots=True)
class _Statistic:
    estimate: float
    standard_error: float
    test_statistic: float
    clipped_fraction: float


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(total: float, total_squares: float, count: int) -> float:
    return max(0.0, (total_squares - total * total / count) / (count - 1))


def _normal_stratum_mean(
    desired_effect: float,
    failure: FailureSensitivityAssumptions,
) -> float:
    contribution = (
        failure.full_only_probability * failure.full_only_difference
        + failure.control_only_probability * failure.control_only_difference
        + failure.both_probability * failure.both_difference
    )
    return (desired_effect - contribution) / (1.0 - failure.total_probability)


def _bounded_failure_or_normal(
    rng: random.Random,
    *,
    normal_mean: float,
    assumptions: EndpointSensitivityAssumptions,
) -> tuple[float, bool]:
    failure = assumptions.failure_scenario
    draw = rng.random()
    if draw < failure.full_only_probability:
        return failure.full_only_difference, False
    draw -= failure.full_only_probability
    if draw < failure.control_only_probability:
        return failure.control_only_difference, False
    draw -= failure.control_only_probability
    if draw < failure.both_probability:
        return failure.both_difference, False
    raw = (
        normal_mean
        + rng.gauss(0.0, math.sqrt(assumptions.search_seed_variance))
        + rng.gauss(0.0, math.sqrt(assumptions.averaged_task_group_variance))
        + rng.gauss(0.0, math.sqrt(assumptions.averaged_rollout_variance))
    )
    bounded = max(-1.0, min(1.0, raw))
    return bounded, bounded != raw


def _cell_count(config: PowerSensitivityConfig, hypothesis: PrimaryHypothesis) -> int:
    if hypothesis is PrimaryHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR:
        return len(config.model_ids) * len(config.repair_strata)
    return len(config.model_ids) * len(config.real_task_ids)


def _simulate_endpoint(
    rng: random.Random,
    *,
    config: PowerSensitivityConfig,
    assumptions: EndpointSensitivityAssumptions,
    desired_effect: float,
) -> dict[int, _Statistic]:
    candidates = config.search_seed_counts_per_cell
    max_seeds = candidates[-1]
    cell_count = _cell_count(config, assumptions.hypothesis)
    offsets = [rng.gauss(0.0, assumptions.cell_effect_sd) for _ in range(cell_count)]
    offset_mean = _mean(offsets)
    offsets = [value - offset_mean for value in offsets]
    normal_mean = _normal_stratum_mean(desired_effect, assumptions.failure_scenario)
    means = {candidate: [] for candidate in candidates}
    variances = {candidate: [] for candidate in candidates}
    clipped_totals = {candidate: 0 for candidate in candidates}

    for offset in offsets:
        total = total_squares = 0.0
        clipped = 0
        for search_index in range(1, max_seeds + 1):
            value, was_clipped = _bounded_failure_or_normal(
                rng,
                normal_mean=normal_mean + offset,
                assumptions=assumptions,
            )
            clipped += was_clipped
            total += value
            total_squares += value * value
            if search_index in means:
                means[search_index].append(total / search_index)
                variances[search_index].append(_sample_variance(total, total_squares, search_index))
                clipped_totals[search_index] += clipped

    statistics: dict[int, _Statistic] = {}
    for candidate in candidates:
        estimate = _mean(means[candidate])
        variance = math.fsum(variances[candidate]) / (cell_count * cell_count * candidate)
        standard_error = math.sqrt(variance)
        if standard_error == 0.0:
            test_statistic = math.inf if estimate > 0.0 else -math.inf
        else:
            test_statistic = estimate / standard_error
        statistics[candidate] = _Statistic(
            estimate=estimate,
            standard_error=standard_error,
            test_statistic=test_statistic,
            clipped_fraction=clipped_totals[candidate] / (cell_count * candidate),
        )
    return statistics


def _simulate_phase(
    config: PowerSensitivityConfig,
    *,
    phase_seed: int,
    desired_effects: bool,
    replicates: int,
) -> dict[int, dict[PrimaryHypothesis, list[_Statistic]]]:
    simulations = {
        candidate: {hypothesis: [] for hypothesis in _PRIMARY_ORDER}
        for candidate in config.search_seed_counts_per_cell
    }
    rng = random.Random(phase_seed)
    for _ in range(replicates):
        for assumptions in config.endpoint_assumptions:
            desired = assumptions.effect_threshold if desired_effects else 0.0
            endpoint = _simulate_endpoint(
                rng,
                config=config,
                assumptions=assumptions,
                desired_effect=desired,
            )
            for candidate, statistic in endpoint.items():
                simulations[candidate][assumptions.hypothesis].append(statistic)
    for candidate in config.search_seed_counts_per_cell:
        correlated = gaussian_copula_resample(
            tuple(simulations[candidate][hypothesis] for hypothesis in _PRIMARY_ORDER),
            assumptions=config.joint_primary_assumptions,
            seed=phase_seed + candidate * 104_729,
            key=lambda statistic: statistic.test_statistic,
        )
        for hypothesis, values in zip(_PRIMARY_ORDER, correlated, strict=True):
            simulations[candidate][hypothesis] = values
    return simulations


def _upper_tail_p(sorted_calibration: list[float], statistic: float) -> float:
    first_at_least = bisect.bisect_left(sorted_calibration, statistic)
    return (1 + len(sorted_calibration) - first_at_least) / (len(sorted_calibration) + 1)


def _holm_rejections(p_values: tuple[float, ...], alpha: float) -> tuple[bool, ...]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return tuple(rejected)


def _rate(count: int, replicates: int) -> MonteCarloProxyRate:
    estimate = count / replicates
    return MonteCarloProxyRate(
        estimate=estimate,
        standard_error=math.sqrt(estimate * (1.0 - estimate) / replicates),
        replicates=replicates,
    )


def _clipping_rate(
    phase: dict[PrimaryHypothesis, list[_Statistic]],
) -> float:
    values = [item.clipped_fraction for hypothesis in _PRIMARY_ORDER for item in phase[hypothesis]]
    return _mean(values)


def _candidate_diagnostic(
    config: PowerSensitivityConfig,
    *,
    candidate: int,
    calibration: dict[PrimaryHypothesis, list[_Statistic]],
    audit: dict[PrimaryHypothesis, list[_Statistic]],
    design: dict[PrimaryHypothesis, list[_Statistic]],
) -> CandidateSensitivityDiagnostic:
    sorted_calibration = {
        hypothesis: sorted(item.test_statistic for item in calibration[hypothesis])
        for hypothesis in _PRIMARY_ORDER
    }
    audit_any_rejected = 0
    design_all_rejected = 0
    headline_passed = 0
    endpoint_rejections = [0, 0, 0]
    endpoint_sesoi = [0, 0, 0]
    endpoint_display = [0, 0, 0]
    endpoint_estimate_sums = [0.0, 0.0, 0.0]

    for replicate in range(config.audit_null_replicates):
        p_values = tuple(
            _upper_tail_p(
                sorted_calibration[hypothesis],
                audit[hypothesis][replicate].test_statistic,
            )
            for hypothesis in _PRIMARY_ORDER
        )
        audit_any_rejected += any(_holm_rejections(p_values, config.family_wise_alpha))

    for replicate in range(config.design_effect_replicates):
        p_values = tuple(
            _upper_tail_p(
                sorted_calibration[hypothesis],
                design[hypothesis][replicate].test_statistic,
            )
            for hypothesis in _PRIMARY_ORDER
        )
        rejections = _holm_rejections(p_values, config.family_wise_alpha)
        design_all_rejected += all(rejections)
        exceeds = tuple(
            design[hypothesis][replicate].estimate > _EXPECTED_SESOI[hypothesis]
            for hypothesis in _PRIMARY_ORDER
        )
        headline_passed += all(rejections) and all(exceeds)
        for index, hypothesis in enumerate(_PRIMARY_ORDER):
            estimate = design[hypothesis][replicate].estimate
            endpoint_rejections[index] += rejections[index]
            endpoint_sesoi[index] += estimate > _EXPECTED_SESOI[hypothesis]
            endpoint_display[index] += estimate > config.project_display_margin
            endpoint_estimate_sums[index] += estimate

    endpoint_diagnostics = tuple(
        EndpointSensitivityDiagnostic(
            hypothesis=hypothesis,
            target_sesoi=_EXPECTED_SESOI[hypothesis],
            proxy_mean_estimate_at_design_effect=(
                endpoint_estimate_sums[index] / config.design_effect_replicates
            ),
            proxy_mean_minus_target=(
                endpoint_estimate_sums[index] / config.design_effect_replicates
                - _EXPECTED_SESOI[hypothesis]
            ),
            proxy_holm_rejection_rate_at_sesoi=_rate(
                endpoint_rejections[index], config.design_effect_replicates
            ),
            point_estimate_exceeds_sesoi_rate=_rate(
                endpoint_sesoi[index], config.design_effect_replicates
            ),
            point_estimate_exceeds_project_display_0_10_rate=_rate(
                endpoint_display[index], config.design_effect_replicates
            ),
        )
        for index, hypothesis in enumerate(_PRIMARY_ORDER)
    )
    return CandidateSensitivityDiagnostic(
        search_seeds_per_model_task_cell=candidate,
        independent_unit="complete_end_to_end_search_run",
        real_model_task_cells=len(config.model_ids) * len(config.real_task_ids),
        repair_model_stratum_cells=len(config.model_ids) * len(config.repair_strata),
        real_grid_condition_run_slots=(
            len(config.model_ids)
            * len(config.real_task_ids)
            * config.real_condition_count
            * candidate
        ),
        verified_repair_cell_weighting=config.verified_repair_cell_weighting,
        calibration_null_replicates=config.calibration_null_replicates,
        audit_null_replicates=config.audit_null_replicates,
        independent_audit_bernoulli_replicates=config.audit_null_replicates,
        design_effect_replicates=config.design_effect_replicates,
        audit_null_proxy_holm_fwer_rate=_rate(audit_any_rejected, config.audit_null_replicates),
        proxy_all_three_holm_rejection_rate_at_sesoi=_rate(
            design_all_rejected, config.design_effect_replicates
        ),
        proxy_headline_pass_rate_at_sesoi=_rate(headline_passed, config.design_effect_replicates),
        bounded_generation=BoundedGenerationDiagnostic(
            method="winsorized-gaussian-random-effects-proxy",
            support="[-1,1]",
            calibration_null_clipping_rate=_clipping_rate(calibration),
            audit_null_clipping_rate=_clipping_rate(audit),
            design_effect_clipping_rate=_clipping_rate(design),
        ),
        endpoint_diagnostics=endpoint_diagnostics,
        formal_sample_size_candidate=False,
    )


def run_power_sensitivity(
    config: PowerSensitivityConfig,
    *,
    declared_pilot_manifest_payload: bytes | None = None,
) -> PowerSensitivityReport:
    """Run three streams after internally digest-binding any declared manifest bytes."""

    checked = PowerSensitivityConfig.model_validate(
        config.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    binding = bind_optional_declared_power_pilot_manifest(
        declared_pilot_manifest_payload,
        expectation=checked.declared_pilot_manifest_expectation,
        fixture=checked.purpose is SensitivityPurpose.DETERMINISTIC_TEST_FIXTURE,
        expected_study_id=checked.study_id,
        expected_endpoint_assumption_hashes=tuple(
            item.assumption_artifact_sha256 for item in checked.endpoint_assumptions
        ),
        expected_joint_assumption_hash=checked.joint_primary_assumptions.assumption_artifact_sha256,
    )
    calibration = _simulate_phase(
        checked,
        phase_seed=checked.simulation_seed,
        desired_effects=False,
        replicates=checked.calibration_null_replicates,
    )
    audit = _simulate_phase(
        checked,
        phase_seed=checked.simulation_seed + 1_000_003,
        desired_effects=False,
        replicates=checked.audit_null_replicates,
    )
    design = _simulate_phase(
        checked,
        phase_seed=checked.simulation_seed + 2_000_033,
        desired_effects=True,
        replicates=checked.design_effect_replicates,
    )
    candidates = tuple(
        _candidate_diagnostic(
            checked,
            candidate=candidate,
            calibration=calibration[candidate],
            audit=audit[candidate],
            design=design[candidate],
        )
        for candidate in checked.search_seed_counts_per_cell
    )
    return PowerSensitivityReport(
        analysis_status="analytic-sensitivity-proxy-only",
        config_sha256=canonical_sha256(checked),
        config=checked,
        digest_bound_declared_pilot_manifest=binding,
        declared_pilot_manifest_digest_bound=binding is not None,
        pilot_artifact_existence_verified=False,
        pilot_artifact_authenticity_verified=False,
        pilot_disjointness_verified=False,
        pilot_closure_verified=False,
        candidate_diagnostics=candidates,
        formal_required_search_seeds_per_cell=None,
        formal_design_ready=False,
        primary_hierarchical_pipeline_design_ready=False,
        exact_nested_bootstrap_implemented=False,
        formal_fwer_or_power_coverage_claimed=False,
        margin_confidence_support_claimed=False,
        project_display_0_10_is_verified_repair_primary_margin=False,
        approximation_warning=(
            "Analytic rank-copula sensitivity proxy only; it assumes independent cell "
            "aggregates, treats repair as a continuous bounded endpoint, and cannot freeze "
            "a formal S."
        ),
        dependence_warning=(
            "The copula matrix is a latent Gaussian rank parameter, not an exact "
            "covariance or shared-run dependence model."
        ),
        failure_warning=(
            "Failure inputs are one plug-in sensitivity scenario, not a conservative "
            "uncertainty set."
        ),
        audit_warning=(
            "Audit Bernoulli draws are independent conditional on separate empirical "
            "marginals; their Monte Carlo SE remains proxy-only."
        ),
        margin_warning=(
            "The 0.10 display threshold is not confidence support and is not the 0.15 repair SESOI."
        ),
        pilot_provenance_warning=(
            "Only canonical declared-manifest syntax and digest equality are checked; referenced "
            "artifact existence, content, authenticity, observation, disjointness, closure, and "
            "blinding are not verified."
        ),
    )


__all__ = [
    "BoundedGenerationDiagnostic",
    "CandidateSensitivityDiagnostic",
    "EndpointSensitivityAssumptions",
    "EndpointSensitivityDiagnostic",
    "FailureSensitivityAssumptions",
    "MonteCarloProxyRate",
    "PowerSensitivityConfig",
    "PowerSensitivityReport",
    "PrimaryHypothesis",
    "SensitivityPurpose",
    "run_power_sensitivity",
]
