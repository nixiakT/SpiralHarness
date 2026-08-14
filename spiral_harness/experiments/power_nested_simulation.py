"""Conditional prospective power using the same basic nested-bootstrap analyzer."""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.power_bootstrap_statistics import wilson_bounds
from spiral_harness.experiments.power_hierarchical import (
    FAMILY_WISE_ALPHA,
    FORMAL_MINIMUM_BOOTSTRAP_DRAWS,
    FORMAL_MINIMUM_SEARCH_SEEDS,
    HYPOTHESIS_ORDER,
    PROJECT_MARGIN,
    PROJECT_MARGIN_HYPOTHESES,
    TEST_MINIMUM_BOOTSTRAP_DRAWS,
    EndpointHierarchyPlan,
    HierarchicalBootstrapPreregistration,
    HierarchicalFamilyInference,
    HierarchicalHypothesis,
    PreregistrationPurpose,
    analyze_hierarchical_dataset,
)
from spiral_harness.experiments.power_nested_generation import (
    ContinuousHierarchyAssumption,
    PairedBinaryAssumption,
    SharedContinuousCorrelationAssumption,
    simulate_nested_dataset,
)

FORMAL_MINIMUM_SIMULATION_REPLICATES = 1_000
TEST_MINIMUM_SIMULATION_REPLICATES = 20


class PowerSimulationPurpose(StrEnum):
    PREREGISTERED_PROSPECTIVE_SIMULATION = "preregistered-prospective-simulation"
    DETERMINISTIC_TEST_FIXTURE = "deterministic-test-fixture"


SearchSeedSet = Annotated[tuple[int, ...], Field(min_length=2)]


class HierarchicalPowerSimulationConfig(ImmutableModel):
    """Content-addressable prospective design and frozen selection rule."""

    schema_version: Literal["1"] = "1"
    study_id: NonEmptyStr
    purpose: PowerSimulationPurpose
    assumption_source_label: NonEmptyStr
    assumption_artifact_sha256: Sha256
    model_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    endpoints: Annotated[tuple[EndpointHierarchyPlan, ...], Field(min_length=3, max_length=3)]
    candidate_search_seed_sets: Annotated[tuple[SearchSeedSet, ...], Field(min_length=1)]
    continuous_assumptions: Annotated[
        tuple[ContinuousHierarchyAssumption, ...],
        Field(min_length=2, max_length=2),
    ]
    shared_continuous_correlation: SharedContinuousCorrelationAssumption
    paired_binary_assumption: PairedBinaryAssumption
    analysis_bootstrap_draws: Annotated[
        int,
        Field(ge=TEST_MINIMUM_BOOTSTRAP_DRAWS, strict=True),
    ]
    analysis_bootstrap_seed: Annotated[int, Field(ge=0, strict=True)]
    null_audit_replicates: Annotated[
        int,
        Field(ge=TEST_MINIMUM_SIMULATION_REPLICATES, strict=True),
    ]
    alternative_calibration_replicates: Annotated[
        int,
        Field(ge=TEST_MINIMUM_SIMULATION_REPLICATES, strict=True),
    ]
    alternative_validation_replicates: Annotated[
        int,
        Field(ge=TEST_MINIMUM_SIMULATION_REPLICATES, strict=True),
    ]
    simulation_seed: Annotated[int, Field(ge=0, strict=True)]
    target_power: Annotated[
        float,
        Field(gt=0.5, lt=1.0, strict=True, allow_inf_nan=False),
    ]
    monte_carlo_confidence_level: Literal[0.95] = 0.95
    family_wise_alpha: Literal[0.05] = FAMILY_WISE_ALPHA
    project_margin: Literal[0.1] = PROJECT_MARGIN
    candidate_order: Literal["strict-prefix-by-increasing-cardinality"] = (
        "strict-prefix-by-increasing-cardinality"
    )
    maximum_leaf_value_lookups: Annotated[int, Field(ge=1, strict=True)]
    selection_rule: Literal[
        "first-calibration-combined-sizing-pass-then-one-independent-validation"
    ] = "first-calibration-combined-sizing-pass-then-one-independent-validation"
    stream_partition: Literal["domain-separated-calibration-validation-and-null-audit"] = (
        "domain-separated-calibration-validation-and-null-audit"
    )
    null_audit_scope: Literal["global-least-favourable-boundary-no-partial-null"] = (
        "global-least-favourable-boundary-no-partial-null"
    )

    @field_validator("model_ids")
    @classmethod
    def _canonicalize_models(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("model_ids must not contain duplicates")
        return ordered

    @field_validator("endpoints")
    @classmethod
    def _canonicalize_endpoints(
        cls,
        values: tuple[EndpointHierarchyPlan, ...],
    ) -> tuple[EndpointHierarchyPlan, ...]:
        by_hypothesis = {value.hypothesis: value for value in values}
        if len(by_hypothesis) != 3 or set(by_hypothesis) != set(HYPOTHESIS_ORDER):
            raise ValueError("endpoints must contain the exact three-primary family")
        return tuple(by_hypothesis[hypothesis] for hypothesis in HYPOTHESIS_ORDER)

    @field_validator("candidate_search_seed_sets")
    @classmethod
    def _freeze_candidate_seed_sets(
        cls,
        values: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        normalized: list[tuple[int, ...]] = []
        for candidate in values:
            if any(type(seed) is not int or seed < 0 for seed in candidate):
                raise ValueError("candidate search seeds must be non-negative exact integers")
            ordered = tuple(sorted(candidate))
            if len(ordered) != len(set(ordered)):
                raise ValueError("a candidate search-seed set contains duplicates")
            normalized.append(ordered)
        normalized.sort(key=lambda item: (len(item), item))
        if len({len(item) for item in normalized}) != len(normalized):
            raise ValueError("candidate search-seed sets must have distinct cardinalities")
        for smaller, larger in pairwise(normalized):
            if larger[: len(smaller)] != smaller:
                raise ValueError("candidate search-seed sets must form an exact prefix sequence")
        return tuple(normalized)

    @field_validator("continuous_assumptions")
    @classmethod
    def _canonicalize_continuous_assumptions(
        cls,
        values: tuple[ContinuousHierarchyAssumption, ...],
    ) -> tuple[ContinuousHierarchyAssumption, ...]:
        by_hypothesis = {value.hypothesis: value for value in values}
        if len(by_hypothesis) != 2 or set(by_hypothesis) != set(PROJECT_MARGIN_HYPOTHESES):
            raise ValueError("continuous assumptions must contain H1 and H2 exactly once")
        return tuple(by_hypothesis[hypothesis] for hypothesis in PROJECT_MARGIN_HYPOTHESES)

    @model_validator(mode="after")
    def _formal_floors_and_shared_hierarchy(self) -> Self:
        first, second, _ = self.endpoints
        if first.tasks != second.tasks:
            raise ValueError("H1 and H2 must share one exact task/group/repeat hierarchy")
        if self.purpose is PowerSimulationPurpose.PREREGISTERED_PROSPECTIVE_SIMULATION:
            if self.analysis_bootstrap_draws < FORMAL_MINIMUM_BOOTSTRAP_DRAWS:
                raise ValueError(
                    "a preregistered simulation requires at least 20,000 bootstrap draws"
                )
            if (
                min(
                    self.null_audit_replicates,
                    self.alternative_calibration_replicates,
                    self.alternative_validation_replicates,
                )
                < FORMAL_MINIMUM_SIMULATION_REPLICATES
            ):
                raise ValueError(
                    "a preregistered simulation requires at least 1,000 replicates per phase"
                )
            if len(self.candidate_search_seed_sets[0]) < FORMAL_MINIMUM_SEARCH_SEEDS:
                raise ValueError("prospective simulation rejects small search-seed candidates")
        if self.estimated_leaf_value_lookups > self.maximum_leaf_value_lookups:
            raise ValueError("declared simulation workload exceeds its leaf-lookup budget")
        return self

    def _leaf_count(self, search_seeds: tuple[int, ...]) -> int:
        leaves_per_model_seed = sum(
            len(group.repeat_seeds)
            for endpoint in self.endpoints
            for task in endpoint.tasks
            for group in task.groups
        )
        return len(self.model_ids) * len(search_seeds) * leaves_per_model_seed

    @property
    def estimated_analysis_replicates(self) -> int:
        calibration = self.null_audit_replicates + self.alternative_calibration_replicates
        return len(self.candidate_search_seed_sets) * calibration + (
            self.alternative_validation_replicates
        )

    @property
    def estimated_bootstrap_endpoint_draws(self) -> int:
        return (
            self.estimated_analysis_replicates
            * len(HYPOTHESIS_ORDER)
            * (self.analysis_bootstrap_draws)
        )

    @property
    def estimated_leaf_value_lookups(self) -> int:
        calibration = self.null_audit_replicates + self.alternative_calibration_replicates
        candidate_work = sum(
            calibration * self._leaf_count(candidate)
            for candidate in self.candidate_search_seed_sets
        )
        validation_work = self.alternative_validation_replicates * max(
            self._leaf_count(candidate) for candidate in self.candidate_search_seed_sets
        )
        return (self.analysis_bootstrap_draws + 1) * (candidate_work + validation_work)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class MonteCarloWilsonRate(ImmutableModel):
    successes: Annotated[int, Field(ge=0, strict=True)]
    replicates: Annotated[int, Field(ge=1, strict=True)]
    estimate: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    one_sided_confidence_level: Literal[0.95]
    one_sided_lower: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    one_sided_upper: Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
    method: Literal["wilson-score-one-sided"] = "wilson-score-one-sided"

    @model_validator(mode="after")
    def _counts_and_interval_are_consistent(self) -> Self:
        if self.successes > self.replicates:
            raise ValueError("Monte Carlo successes exceed replicates")
        estimate = self.successes / self.replicates
        if self.estimate != estimate:
            raise ValueError("Monte Carlo estimate differs from its counts")
        lower, upper = wilson_bounds(
            self.successes,
            self.replicates,
            self.one_sided_confidence_level,
        )
        if self.one_sided_lower != lower or self.one_sided_upper != upper:
            raise ValueError("Wilson bounds differ from the declared counts")
        return self


class EndpointPowerDiagnostic(ImmutableModel):
    hypothesis: HierarchicalHypothesis
    holm_sesoi_null_rejection_power_diagnostic: MonteCarloWilsonRate
    project_margin_claim_power_diagnostic: MonteCarloWilsonRate | None
    used_for_candidate_selection: Literal[False]

    @model_validator(mode="after")
    def _margin_rate_only_for_h1_h2(self) -> Self:
        applicable = self.hypothesis in PROJECT_MARGIN_HYPOTHESES
        if applicable != (self.project_margin_claim_power_diagnostic is not None):
            raise ValueError("10pp power is defined exactly for H1 and H2")
        return self


class CandidatePowerDiagnostic(ImmutableModel):
    independent_search_seeds: SearchSeedSet
    null_holm_fwer_audit: MonteCarloWilsonRate
    null_fwer_exceeds_nominal_point_estimate: bool
    endpoint_diagnostics: Annotated[
        tuple[EndpointPowerDiagnostic, ...],
        Field(min_length=3, max_length=3),
    ]
    calibration_combined_sizing_event_power: MonteCarloWilsonRate
    combined_sizing_event_definition: Literal[
        "all-three-holm-sesoi-rejections-and-both-h1-h2-simultaneous-10pp-claims"
    ]
    joint_h3_dependence_assumed_independent: Literal[True]
    protected_slice_pass_included: Literal[False]
    cost_pass_included: Literal[False]
    provenance_pass_included: Literal[False]
    passes_calibration_threshold: bool

    @model_validator(mode="after")
    def _derived_flags_are_consistent(self) -> Self:
        if tuple(item.hypothesis for item in self.endpoint_diagnostics) != HYPOTHESIS_ORDER:
            raise ValueError("power endpoints do not use the frozen primary order")
        if self.null_fwer_exceeds_nominal_point_estimate != (
            self.null_holm_fwer_audit.estimate > FAMILY_WISE_ALPHA
        ):
            raise ValueError("null FWER warning flag is inconsistent")
        return self


class HierarchicalPowerSimulationReport(ImmutableModel):
    schema_version: Literal["1"] = "1"
    analysis_status: Literal["synthetic-equal-cell-basic-nested-bootstrap-power"]
    config_sha256: Sha256
    config: HierarchicalPowerSimulationConfig
    candidate_diagnostics: Annotated[tuple[CandidatePowerDiagnostic, ...], Field(min_length=1)]
    calibration_selected_search_seeds: tuple[int, ...] | None
    selected_candidate_validation_combined_sizing_event_power: MonteCarloWilsonRate | None
    conditional_recommended_search_seeds: tuple[int, ...] | None
    conditional_selection_succeeded: bool
    conditional_on_declared_simulation_assumptions: Literal[True]
    synthetic_validation_only: Literal[True]
    observed_harness_effect_evidence: Literal[False]
    unconditional_power_guarantee: Literal[False]
    null_audit_used_for_selection: Literal[False]
    calibration_validation_and_audit_streams_domain_separated: Literal[True]
    combined_sizing_event_scope_only: Literal[True]
    protected_slice_pass_included: Literal[False]
    cost_pass_included: Literal[False]
    provenance_pass_included: Literal[False]
    pilot_artifact_existence_verified: Literal[False]
    pilot_disjointness_verified: Literal[False]
    pilot_closure_verified: Literal[False]
    attested_sealed_closure_verified: Literal[False]
    reportable_sealed_evidence: Literal[False]
    verified_sealed_evidence: Literal[False]
    formal_design_ready: Literal[False]
    formal_sample_size_frozen: Literal[False]
    coverage_calibrated: Literal[False]
    partial_null_fwer_audit_implemented: Literal[False]
    resumable_checkpointing_implemented: Literal[False]
    estimated_analysis_replicates: Annotated[int, Field(ge=1, strict=True)]
    estimated_bootstrap_endpoint_draws: Annotated[int, Field(ge=1, strict=True)]
    estimated_leaf_value_lookups: Annotated[int, Field(ge=1, strict=True)]
    leaf_value_lookup_budget: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def _selection_matches_first_passing_candidate(self) -> Self:
        calibration_selected = next(
            (
                item.independent_search_seeds
                for item in self.candidate_diagnostics
                if item.passes_calibration_threshold
            ),
            None,
        )
        if tuple(item.independent_search_seeds for item in self.candidate_diagnostics) != (
            self.config.candidate_search_seed_sets
        ):
            raise ValueError("power diagnostics do not match the preregistered candidate roster")
        for candidate in self.candidate_diagnostics:
            if candidate.null_holm_fwer_audit.replicates != self.config.null_audit_replicates:
                raise ValueError("null audit rate has the wrong replicate count")
            if candidate.calibration_combined_sizing_event_power.replicates != (
                self.config.alternative_calibration_replicates
            ):
                raise ValueError("calibration rate has the wrong replicate count")
            for endpoint in candidate.endpoint_diagnostics:
                if endpoint.holm_sesoi_null_rejection_power_diagnostic.replicates != (
                    self.config.alternative_calibration_replicates
                ):
                    raise ValueError("endpoint diagnostic has the wrong replicate count")
                margin = endpoint.project_margin_claim_power_diagnostic
                if margin is not None and margin.replicates != (
                    self.config.alternative_calibration_replicates
                ):
                    raise ValueError("10pp diagnostic has the wrong replicate count")
            passed = candidate.calibration_combined_sizing_event_power.one_sided_lower >= (
                self.config.target_power
            )
            if candidate.passes_calibration_threshold != passed:
                raise ValueError("calibration threshold flag is inconsistent")
        if self.calibration_selected_search_seeds != calibration_selected:
            raise ValueError("calibration selection is not the first passing candidate")
        validation = self.selected_candidate_validation_combined_sizing_event_power
        if (calibration_selected is None) != (validation is None):
            raise ValueError(
                "independent validation must exist exactly for a calibration selection"
            )
        if validation is not None and validation.replicates != (
            self.config.alternative_validation_replicates
        ):
            raise ValueError("independent validation has the wrong replicate count")
        expected = (
            calibration_selected
            if validation is not None and validation.one_sided_lower >= self.config.target_power
            else None
        )
        if self.conditional_recommended_search_seeds != expected:
            raise ValueError("conditional recommendation differs from independent validation")
        if self.conditional_selection_succeeded != (expected is not None):
            raise ValueError("conditional_selection_succeeded is inconsistent")
        workload = (
            self.config.estimated_analysis_replicates,
            self.config.estimated_bootstrap_endpoint_draws,
            self.config.estimated_leaf_value_lookups,
            self.config.maximum_leaf_value_lookups,
        )
        if workload != (
            self.estimated_analysis_replicates,
            self.estimated_bootstrap_endpoint_draws,
            self.estimated_leaf_value_lookups,
            self.leaf_value_lookup_budget,
        ):
            raise ValueError("reported workload differs from the preregistered config")
        if self.config_sha256 != self.config.fingerprint:
            raise ValueError("power report is bound to another config")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _domain_seed(config: HierarchicalPowerSimulationConfig, *parts: object) -> int:
    return int(
        canonical_sha256(
            {
                "domain": "spiral-harness/nested-bootstrap-power/v1",
                "config_sha256": config.fingerprint,
                "simulation_seed": config.simulation_seed,
                "parts": tuple(str(part) for part in parts),
            }
        )[:16],
        16,
    )


def _analysis_preregistration(
    config: HierarchicalPowerSimulationConfig,
    search_seeds: tuple[int, ...],
    *,
    bootstrap_seed: int,
) -> HierarchicalBootstrapPreregistration:
    purpose = (
        PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION
        if config.purpose is PowerSimulationPurpose.PREREGISTERED_PROSPECTIVE_SIMULATION
        else PreregistrationPurpose.DETERMINISTIC_TEST_FIXTURE
    )
    return HierarchicalBootstrapPreregistration(
        study_id=f"{config.study_id}/power-candidate-{len(search_seeds)}",
        purpose=purpose,
        model_ids=config.model_ids,
        independent_search_seeds=search_seeds,
        endpoints=config.endpoints,
        bootstrap_draws=config.analysis_bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )


def _wilson_rate(successes: int, replicates: int, confidence_level: float) -> MonteCarloWilsonRate:
    estimate = successes / replicates
    lower, upper = wilson_bounds(successes, replicates, confidence_level)
    return MonteCarloWilsonRate(
        successes=successes,
        replicates=replicates,
        estimate=estimate,
        one_sided_confidence_level=confidence_level,
        one_sided_lower=lower,
        one_sided_upper=upper,
    )


def _simulation_inference(
    config: HierarchicalPowerSimulationConfig,
    search_seeds: tuple[int, ...],
    *,
    phase: str,
    replicate: int,
    alternative: bool,
) -> HierarchicalFamilyInference:
    candidate_key = ",".join(str(seed) for seed in search_seeds)
    preregistration = _analysis_preregistration(
        config,
        search_seeds,
        bootstrap_seed=_domain_seed(
            config,
            "bootstrap",
            candidate_key,
            phase,
            replicate,
            config.analysis_bootstrap_seed,
        ),
    )
    dataset = simulate_nested_dataset(
        preregistration,
        config.continuous_assumptions,
        config.shared_continuous_correlation,
        config.paired_binary_assumption,
        alternative=alternative,
        seed=_domain_seed(config, "dataset", phase, replicate),
    )
    return analyze_hierarchical_dataset(preregistration, dataset)


def _candidate_diagnostic(
    config: HierarchicalPowerSimulationConfig,
    search_seeds: tuple[int, ...],
) -> CandidatePowerDiagnostic:
    null_fwer = 0
    alternative_holm = [0, 0, 0]
    alternative_margin = [0, 0]
    combined_sizing_event_passes = 0
    phases = (
        ("null-audit", False, config.null_audit_replicates),
        (
            "alternative-calibration",
            True,
            config.alternative_calibration_replicates,
        ),
    )
    for phase, alternative, replicates in phases:
        for replicate in range(replicates):
            inference = _simulation_inference(
                config,
                search_seeds,
                phase=phase,
                replicate=replicate,
                alternative=alternative,
            )
            if not alternative:
                null_fwer += any(endpoint.holm_rejected for endpoint in inference.endpoints)
                continue
            for index, endpoint in enumerate(inference.endpoints):
                alternative_holm[index] += endpoint.holm_rejected
            for index, endpoint in enumerate(inference.endpoints[:2]):
                alternative_margin[index] += endpoint.project_margin_claim_supported
            combined_sizing_event_passes += (
                inference.all_primary_holm_rejected
                and inference.all_project_margin_claims_supported
            )

    power_replicates = config.alternative_calibration_replicates
    endpoint_diagnostics: list[EndpointPowerDiagnostic] = []
    for index, hypothesis in enumerate(HYPOTHESIS_ORDER):
        rejection_rate = _wilson_rate(
            alternative_holm[index],
            power_replicates,
            config.monte_carlo_confidence_level,
        )
        margin_rate = (
            _wilson_rate(
                alternative_margin[index],
                power_replicates,
                config.monte_carlo_confidence_level,
            )
            if index < len(PROJECT_MARGIN_HYPOTHESES)
            else None
        )
        endpoint_diagnostics.append(
            EndpointPowerDiagnostic(
                hypothesis=hypothesis,
                holm_sesoi_null_rejection_power_diagnostic=rejection_rate,
                project_margin_claim_power_diagnostic=margin_rate,
                used_for_candidate_selection=False,
            )
        )
    null_rate = _wilson_rate(
        null_fwer,
        config.null_audit_replicates,
        config.monte_carlo_confidence_level,
    )
    return CandidatePowerDiagnostic(
        independent_search_seeds=search_seeds,
        null_holm_fwer_audit=null_rate,
        null_fwer_exceeds_nominal_point_estimate=null_rate.estimate > config.family_wise_alpha,
        endpoint_diagnostics=tuple(endpoint_diagnostics),
        calibration_combined_sizing_event_power=_wilson_rate(
            combined_sizing_event_passes,
            power_replicates,
            config.monte_carlo_confidence_level,
        ),
        combined_sizing_event_definition=(
            "all-three-holm-sesoi-rejections-and-both-h1-h2-simultaneous-10pp-claims"
        ),
        joint_h3_dependence_assumed_independent=True,
        protected_slice_pass_included=False,
        cost_pass_included=False,
        provenance_pass_included=False,
        passes_calibration_threshold=(
            _wilson_rate(
                combined_sizing_event_passes,
                power_replicates,
                config.monte_carlo_confidence_level,
            ).one_sided_lower
            >= config.target_power
        ),
    )


def _independent_validation_rate(
    config: HierarchicalPowerSimulationConfig,
    search_seeds: tuple[int, ...],
) -> MonteCarloWilsonRate:
    successes = 0
    for replicate in range(config.alternative_validation_replicates):
        inference = _simulation_inference(
            config,
            search_seeds,
            phase="alternative-validation",
            replicate=replicate,
            alternative=True,
        )
        successes += (
            inference.all_primary_holm_rejected and inference.all_project_margin_claims_supported
        )
    return _wilson_rate(
        successes,
        config.alternative_validation_replicates,
        config.monte_carlo_confidence_level,
    )


def simulate_hierarchical_power(
    config: HierarchicalPowerSimulationConfig,
) -> HierarchicalPowerSimulationReport:
    """Run deterministic null audits and alternative power simulations."""

    checked = HierarchicalPowerSimulationConfig.model_validate(
        config.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    diagnostics = tuple(
        _candidate_diagnostic(checked, candidate)
        for candidate in checked.candidate_search_seed_sets
    )
    calibration_selected = next(
        (
            item.independent_search_seeds
            for item in diagnostics
            if item.passes_calibration_threshold
        ),
        None,
    )
    validation = (
        _independent_validation_rate(checked, calibration_selected)
        if calibration_selected is not None
        else None
    )
    recommended = (
        calibration_selected
        if validation is not None and validation.one_sided_lower >= checked.target_power
        else None
    )
    return HierarchicalPowerSimulationReport(
        analysis_status="synthetic-equal-cell-basic-nested-bootstrap-power",
        config_sha256=checked.fingerprint,
        config=checked,
        candidate_diagnostics=diagnostics,
        calibration_selected_search_seeds=calibration_selected,
        selected_candidate_validation_combined_sizing_event_power=validation,
        conditional_recommended_search_seeds=recommended,
        conditional_selection_succeeded=recommended is not None,
        conditional_on_declared_simulation_assumptions=True,
        synthetic_validation_only=True,
        observed_harness_effect_evidence=False,
        unconditional_power_guarantee=False,
        null_audit_used_for_selection=False,
        calibration_validation_and_audit_streams_domain_separated=True,
        combined_sizing_event_scope_only=True,
        protected_slice_pass_included=False,
        cost_pass_included=False,
        provenance_pass_included=False,
        pilot_artifact_existence_verified=False,
        pilot_disjointness_verified=False,
        pilot_closure_verified=False,
        attested_sealed_closure_verified=False,
        reportable_sealed_evidence=False,
        verified_sealed_evidence=False,
        formal_design_ready=False,
        formal_sample_size_frozen=False,
        coverage_calibrated=False,
        partial_null_fwer_audit_implemented=False,
        resumable_checkpointing_implemented=False,
        estimated_analysis_replicates=checked.estimated_analysis_replicates,
        estimated_bootstrap_endpoint_draws=checked.estimated_bootstrap_endpoint_draws,
        estimated_leaf_value_lookups=checked.estimated_leaf_value_lookups,
        leaf_value_lookup_budget=checked.maximum_leaf_value_lookups,
    )


__all__ = [
    "FORMAL_MINIMUM_SIMULATION_REPLICATES",
    "TEST_MINIMUM_SIMULATION_REPLICATES",
    "CandidatePowerDiagnostic",
    "ContinuousHierarchyAssumption",
    "EndpointPowerDiagnostic",
    "HierarchicalPowerSimulationConfig",
    "HierarchicalPowerSimulationReport",
    "MonteCarloWilsonRate",
    "PairedBinaryAssumption",
    "PowerSimulationPurpose",
    "SharedContinuousCorrelationAssumption",
    "simulate_hierarchical_power",
]
