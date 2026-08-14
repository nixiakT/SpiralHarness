"""Fixed-roster equal-cell basic nested-bootstrap method-development prototype.

This module is intentionally separate from :mod:`power_analysis`, whose
bounded Gaussian calculation remains an analytic sensitivity proxy.  The
models below require the complete paired leaf-level roster and run the same
declared estimator for every confidence interval and simulated study.

Models and tasks are fixed.  Bootstrap draws resample independent end-to-end
search seeds within each model-by-task cell, task groups within the selected
run, and rollout repeats within the selected group.  They never count a leaf
rollout as an independent search replicate.
"""

from __future__ import annotations

import math
import random
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.power_bootstrap_statistics import (
    holm_three,
    mean,
    quantile_r7,
)

FORMAL_MINIMUM_BOOTSTRAP_DRAWS = 20_000
FORMAL_MINIMUM_SEARCH_SEEDS = 4
TEST_MINIMUM_BOOTSTRAP_DRAWS = 200
PROJECT_MARGIN = 0.10
FAMILY_WISE_ALPHA = 0.05


class HierarchicalHypothesis(StrEnum):
    FULL_VS_SCORE_REAL = "full-vs-score-real"
    FULL_VS_STATIC_REAL = "full-vs-static-real"
    FULL_VS_SCORE_VERIFIED_REPAIR = "full-vs-score-verified-repair"


HYPOTHESIS_ORDER = (
    HierarchicalHypothesis.FULL_VS_SCORE_REAL,
    HierarchicalHypothesis.FULL_VS_STATIC_REAL,
    HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR,
)
PROJECT_MARGIN_HYPOTHESES = (
    HierarchicalHypothesis.FULL_VS_SCORE_REAL,
    HierarchicalHypothesis.FULL_VS_STATIC_REAL,
)
_SESOI = {
    HierarchicalHypothesis.FULL_VS_SCORE_REAL: 0.02,
    HierarchicalHypothesis.FULL_VS_STATIC_REAL: 0.02,
    HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR: 0.15,
}


class PreregistrationPurpose(StrEnum):
    DISJOINT_PILOT_ANALYSIS = "disjoint-pilot-analysis"
    PROSPECTIVE_POWER_SIMULATION = "prospective-power-simulation"
    DETERMINISTIC_TEST_FIXTURE = "deterministic-test-fixture"


class DatasetSource(StrEnum):
    DISJOINT_PILOT = "disjoint-pilot"
    SYNTHETIC_VALIDATION = "synthetic-validation"
    DETERMINISTIC_TEST_FIXTURE = "deterministic-test-fixture"


class TaskGroupPlan(ImmutableModel):
    """One item-side cluster and its exact paired rollout-repeat roster."""

    schema_version: Literal["1"] = "1"
    group_id: NonEmptyStr
    repeat_seeds: Annotated[tuple[int, ...], Field(min_length=1)]

    @field_validator("repeat_seeds")
    @classmethod
    def _canonicalize_repeat_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("repeat seeds must be non-negative exact integers")
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("repeat seeds must not contain duplicates")
        return ordered


class TaskHierarchyPlan(ImmutableModel):
    """Frozen task/stratum hierarchy shared by every model and search seed."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    groups: Annotated[tuple[TaskGroupPlan, ...], Field(min_length=2)]

    @field_validator("groups")
    @classmethod
    def _canonicalize_groups(
        cls,
        values: tuple[TaskGroupPlan, ...],
    ) -> tuple[TaskGroupPlan, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.group_id))
        if len(ordered) != len({value.group_id for value in ordered}):
            raise ValueError("task groups must not contain duplicate identifiers")
        return ordered


class EndpointHierarchyPlan(ImmutableModel):
    """One primary endpoint with its fixed task-side hierarchy."""

    schema_version: Literal["1"] = "1"
    hypothesis: HierarchicalHypothesis
    endpoint_kind: Literal["normalized-score-difference", "paired-binary-difference"]
    sesoi: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    tasks: Annotated[tuple[TaskHierarchyPlan, ...], Field(min_length=1)]

    @field_validator("tasks")
    @classmethod
    def _canonicalize_tasks(
        cls,
        values: tuple[TaskHierarchyPlan, ...],
    ) -> tuple[TaskHierarchyPlan, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.task_id))
        if len(ordered) != len({value.task_id for value in ordered}):
            raise ValueError("endpoint tasks must not contain duplicate identifiers")
        return ordered

    @model_validator(mode="after")
    def _freeze_endpoint_semantics(self) -> Self:
        if self.sesoi != _SESOI[self.hypothesis]:
            raise ValueError(f"{self.hypothesis.value} has the wrong frozen SESOI")
        expected_kind = (
            "paired-binary-difference"
            if self.hypothesis is HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR
            else "normalized-score-difference"
        )
        if self.endpoint_kind != expected_kind:
            raise ValueError(f"{self.hypothesis.value} has the wrong endpoint kind")
        return self


class HierarchicalBootstrapPreregistration(ImmutableModel):
    """Machine-readable prototype contract; sealed confirmatory use is blocked."""

    schema_version: Literal["1"] = "1"
    study_id: NonEmptyStr
    purpose: PreregistrationPurpose
    model_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    independent_search_seeds: Annotated[tuple[int, ...], Field(min_length=2)]
    endpoints: Annotated[tuple[EndpointHierarchyPlan, ...], Field(min_length=3, max_length=3)]
    bootstrap_draws: Annotated[int, Field(ge=TEST_MINIMUM_BOOTSTRAP_DRAWS, strict=True)]
    bootstrap_seed: Annotated[int, Field(ge=0, strict=True)]
    confidence_level: Literal[0.95] = 0.95
    family_wise_alpha: Literal[0.05] = FAMILY_WISE_ALPHA
    project_margin: Literal[0.1] = PROJECT_MARGIN
    project_margin_hypotheses: tuple[
        Literal[HierarchicalHypothesis.FULL_VS_SCORE_REAL],
        Literal[HierarchicalHypothesis.FULL_VS_STATIC_REAL],
    ] = PROJECT_MARGIN_HYPOTHESES
    independent_unit: Literal["complete-end-to-end-search-run"] = "complete-end-to-end-search-run"
    model_task_estimand: Literal["equal-model-by-task-cell-macro-mean"] = (
        "equal-model-by-task-cell-macro-mean"
    )
    model_task_roster: Literal["fixed-not-resampled"] = "fixed-not-resampled"
    within_run_group_estimand: Literal["equal-group-mean"] = "equal-group-mean"
    within_group_repeat_estimand: Literal["equal-repeat-mean"] = "equal-repeat-mean"
    nested_resampling: Literal[
        "search-seed-within-cell,task-group-within-run,repeat-within-group"
    ] = "search-seed-within-cell,task-group-within-run,repeat-within-group"
    interval_method: Literal["basic-nested-bootstrap-r7"] = "basic-nested-bootstrap-r7"
    primary_multiplicity: Literal["one-sided-holm-fwer"] = "one-sided-holm-fwer"
    primary_null: Literal["delta-less-than-or-equal-to-endpoint-sesoi"] = (
        "delta-less-than-or-equal-to-endpoint-sesoi"
    )
    margin_interval_multiplicity: Literal["bonferroni-simultaneous-one-sided-lower"] = (
        "bonferroni-simultaneous-one-sided-lower"
    )
    margin_claim_rule: Literal["simultaneous-lower-bound-strictly-greater-than-0.10"] = (
        "simultaneous-lower-bound-strictly-greater-than-0.10"
    )
    missing_or_failed_leaf_policy: Literal["retain-as-preregistered-score"] = (
        "retain-as-preregistered-score"
    )
    small_cluster_guard: Literal["operational-floor-not-coverage-proof"] = (
        "operational-floor-not-coverage-proof"
    )
    confirmatory_analysis_status: Literal["blocked-pending-attested-sealed-closure"] = (
        "blocked-pending-attested-sealed-closure"
    )

    @field_validator("model_ids")
    @classmethod
    def _canonicalize_models(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("model_ids must not contain duplicates")
        return ordered

    @field_validator("independent_search_seeds")
    @classmethod
    def _canonicalize_search_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("search seeds must be non-negative exact integers")
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("search seeds must not contain duplicates")
        return ordered

    @field_validator("endpoints")
    @classmethod
    def _canonicalize_endpoints(
        cls,
        values: tuple[EndpointHierarchyPlan, ...],
    ) -> tuple[EndpointHierarchyPlan, ...]:
        by_hypothesis = {value.hypothesis: value for value in values}
        if len(by_hypothesis) != len(values) or set(by_hypothesis) != set(HYPOTHESIS_ORDER):
            raise ValueError("endpoints must contain each primary hypothesis exactly once")
        return tuple(by_hypothesis[hypothesis] for hypothesis in HYPOTHESIS_ORDER)

    @model_validator(mode="after")
    def _formal_draw_floor_and_shared_real_hierarchy(self) -> Self:
        if (
            self.purpose
            in {
                PreregistrationPurpose.DISJOINT_PILOT_ANALYSIS,
                PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION,
            }
            and self.bootstrap_draws < FORMAL_MINIMUM_BOOTSTRAP_DRAWS
        ):
            raise ValueError("formal pilot or power analysis requires at least 20,000 draws")
        if (
            self.purpose
            in {
                PreregistrationPurpose.DISJOINT_PILOT_ANALYSIS,
                PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION,
            }
            and len(self.independent_search_seeds) < FORMAL_MINIMUM_SEARCH_SEEDS
        ):
            raise ValueError(
                "formal pilot or power analysis rejects fewer than four independent search seeds"
            )
        first, second, _ = self.endpoints
        if first.tasks != second.tasks:
            raise ValueError("the two real-score endpoints must share one exact task hierarchy")
        if self.project_margin_hypotheses != PROJECT_MARGIN_HYPOTHESES:
            raise ValueError("the 0.10 claim family must be exactly H1 and H2")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class PairedLeafObservation(ImmutableModel):
    """One retained intention-to-treat paired difference at the leaf level."""

    schema_version: Literal["1"] = "1"
    hypothesis: HierarchicalHypothesis
    model_id: NonEmptyStr
    task_id: NonEmptyStr
    search_seed: Annotated[int, Field(ge=0, strict=True)]
    group_id: NonEmptyStr
    repeat_seed: Annotated[int, Field(ge=0, strict=True)]
    difference: Annotated[float, Field(ge=-1.0, le=1.0, strict=True, allow_inf_nan=False)]

    @model_validator(mode="after")
    def _binary_endpoint_is_discrete(self) -> Self:
        if (
            self.hypothesis is HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR
            and self.difference not in (-1.0, 0.0, 1.0)
        ):
            raise ValueError("verified-repair paired differences must be -1, 0, or 1")
        return self

    @property
    def coordinate(self) -> tuple[str, str, str, int, str, int]:
        return (
            self.hypothesis.value,
            self.model_id,
            self.task_id,
            self.search_seed,
            self.group_id,
            self.repeat_seed,
        )


class HierarchicalPairedDataset(ImmutableModel):
    """Canonical complete roster consumed by inference or prospective simulation."""

    schema_version: Literal["1"] = "1"
    preregistration_fingerprint: Sha256
    source: DatasetSource
    observations: Annotated[tuple[PairedLeafObservation, ...], Field(min_length=1)]

    @field_validator("observations")
    @classmethod
    def _canonicalize_observations(
        cls,
        values: tuple[PairedLeafObservation, ...],
    ) -> tuple[PairedLeafObservation, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.coordinate))
        coordinates = tuple(value.coordinate for value in ordered)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("paired observations must not duplicate a leaf coordinate")
        return ordered

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class EndpointNestedInference(ImmutableModel):
    hypothesis: HierarchicalHypothesis
    estimate: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    standard_error: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    confidence_level: Literal[0.95]
    marginal_two_sided_lower: Annotated[float, Field(allow_inf_nan=False)]
    marginal_two_sided_upper: Annotated[float, Field(allow_inf_nan=False)]
    project_margin_familywise_one_sided_lower: (
        Annotated[
            float,
            Field(allow_inf_nan=False),
        ]
        | None
    )
    primary_null: Literal["delta-less-than-or-equal-to-endpoint-sesoi"]
    one_sided_sesoi_null_p_value: Annotated[
        float,
        Field(gt=0.0, le=1.0, allow_inf_nan=False),
    ]
    holm_adjusted_p_value: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    holm_rejected: bool
    sesoi: Annotated[float, Field(gt=0.0, le=1.0, strict=True, allow_inf_nan=False)]
    point_estimate_exceeds_sesoi: bool
    project_margin_applicable: bool
    project_margin: Literal[0.1]
    project_margin_claim_supported: bool

    @model_validator(mode="after")
    def _derived_claim_flags_are_consistent(self) -> Self:
        if self.marginal_two_sided_lower > self.marginal_two_sided_upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        if self.sesoi != _SESOI[self.hypothesis]:
            raise ValueError("endpoint inference uses the wrong frozen SESOI")
        if self.point_estimate_exceeds_sesoi != (self.estimate > self.sesoi):
            raise ValueError("SESOI flag differs from the strict point-estimate rule")
        applicable = self.hypothesis in PROJECT_MARGIN_HYPOTHESES
        if self.project_margin_applicable != applicable:
            raise ValueError("10pp applicability differs from the frozen H1/H2 family")
        if applicable != (self.project_margin_familywise_one_sided_lower is not None):
            raise ValueError("10pp lower bound applicability is inconsistent")
        expected_margin = self.project_margin_applicable and (
            self.project_margin_familywise_one_sided_lower is not None
            and self.project_margin_familywise_one_sided_lower > self.project_margin
        )
        if self.project_margin_claim_supported != expected_margin:
            raise ValueError("0.10 claim flag differs from the simultaneous lower-bound rule")
        return self


class HierarchicalFamilyInference(ImmutableModel):
    """Deterministic prototype output; not sealed confirmatory evidence."""

    schema_version: Literal["1"] = "1"
    analysis_status: Literal["equal-cell-basic-nested-bootstrap-prototype"]
    preregistration_fingerprint: Sha256
    preregistration_purpose: PreregistrationPurpose
    dataset_fingerprint: Sha256
    dataset_source: DatasetSource
    bootstrap_draws: Annotated[int, Field(ge=TEST_MINIMUM_BOOTSTRAP_DRAWS, strict=True)]
    bootstrap_seed: Annotated[int, Field(ge=0, strict=True)]
    endpoints: Annotated[tuple[EndpointNestedInference, ...], Field(min_length=3, max_length=3)]
    all_primary_holm_rejected: bool
    all_primary_margin_tests_passed: bool
    any_project_margin_claim_supported: bool
    all_project_margin_claims_supported: bool
    confirmatory_result: Literal[False]
    reportable_sealed_evidence: Literal[False]
    verified_sealed_evidence: Literal[False]
    attested_sealed_closure_verified: Literal[False]
    formal_confirmatory_analysis_blocked: Literal[True]
    coverage_calibrated: Literal[False]
    synthetic_validation_only: bool

    @model_validator(mode="after")
    def _family_flags_match_endpoints(self) -> Self:
        if tuple(item.hypothesis for item in self.endpoints) != HYPOTHESIS_ORDER:
            raise ValueError("inference endpoints do not use the frozen primary order")
        adjusted, rejected = holm_three(
            tuple(item.one_sided_sesoi_null_p_value for item in self.endpoints),
            alpha=FAMILY_WISE_ALPHA,
        )
        for index, endpoint in enumerate(self.endpoints):
            if endpoint.holm_adjusted_p_value != adjusted[index]:
                raise ValueError("Holm adjusted p-value is inconsistent")
            if endpoint.holm_rejected != rejected[index]:
                raise ValueError("Holm rejection flag is inconsistent")
        all_rejected = all(rejected)
        all_margin_tests_passed = all_rejected and all(
            item.point_estimate_exceeds_sesoi for item in self.endpoints
        )
        margin_items = tuple(item for item in self.endpoints if item.project_margin_applicable)
        if self.all_primary_holm_rejected != all_rejected:
            raise ValueError("all-primary rejection flag is inconsistent")
        if self.all_primary_margin_tests_passed != all_margin_tests_passed:
            raise ValueError("all-primary margin-test flag is inconsistent")
        if self.any_project_margin_claim_supported != any(
            item.project_margin_claim_supported for item in margin_items
        ):
            raise ValueError("any-margin flag is inconsistent")
        if self.all_project_margin_claims_supported != all(
            item.project_margin_claim_supported for item in margin_items
        ):
            raise ValueError("all-margin flag is inconsistent")
        if self.synthetic_validation_only != (
            self.dataset_source is DatasetSource.SYNTHETIC_VALIDATION
        ):
            raise ValueError("synthetic validation flag differs from the dataset source")
        return self


def _expected_coordinates(
    preregistration: HierarchicalBootstrapPreregistration,
) -> set[tuple[str, str, str, int, str, int]]:
    return {
        (
            endpoint.hypothesis.value,
            model_id,
            task.task_id,
            search_seed,
            group.group_id,
            repeat_seed,
        )
        for endpoint in preregistration.endpoints
        for model_id in preregistration.model_ids
        for task in endpoint.tasks
        for search_seed in preregistration.independent_search_seeds
        for group in task.groups
        for repeat_seed in group.repeat_seeds
    }


def validate_complete_hierarchical_dataset(
    preregistration: HierarchicalBootstrapPreregistration,
    dataset: HierarchicalPairedDataset,
) -> None:
    """Fail closed unless the data equal the complete preregistered leaf roster."""

    if dataset.preregistration_fingerprint != preregistration.fingerprint:
        raise ValueError("dataset belongs to another hierarchical preregistration")
    if (
        preregistration.purpose is PreregistrationPurpose.DISJOINT_PILOT_ANALYSIS
        and dataset.source is not DatasetSource.DISJOINT_PILOT
    ):
        raise ValueError("pilot analysis preregistrations only accept disjoint-pilot data")
    if (
        preregistration.purpose is PreregistrationPurpose.PROSPECTIVE_POWER_SIMULATION
        and dataset.source is not DatasetSource.SYNTHETIC_VALIDATION
    ):
        raise ValueError("prospective power preregistrations only accept synthetic data")
    if (
        preregistration.purpose is PreregistrationPurpose.DETERMINISTIC_TEST_FIXTURE
        and dataset.source
        not in {DatasetSource.SYNTHETIC_VALIDATION, DatasetSource.DETERMINISTIC_TEST_FIXTURE}
    ):
        raise ValueError("test fixture preregistrations only accept test or synthetic data")
    expected = _expected_coordinates(preregistration)
    actual = {observation.coordinate for observation in dataset.observations}
    if actual != expected:
        missing = len(expected.difference(actual))
        foreign = len(actual.difference(expected))
        raise ValueError(
            f"dataset differs from the complete hierarchical roster: missing={missing}, "
            f"foreign={foreign}"
        )


def _lookup(
    dataset: HierarchicalPairedDataset,
) -> dict[tuple[str, str, str, int, str, int], float]:
    return {observation.coordinate: observation.difference for observation in dataset.observations}


def _endpoint_estimate(
    preregistration: HierarchicalBootstrapPreregistration,
    endpoint: EndpointHierarchyPlan,
    values: dict[tuple[str, str, str, int, str, int], float],
    *,
    rng: random.Random | None,
) -> float:
    cell_means: list[float] = []
    for model_id in preregistration.model_ids:
        for task in endpoint.tasks:
            seeds = preregistration.independent_search_seeds
            selected_seeds = (
                seeds if rng is None else tuple(seeds[rng.randrange(len(seeds))] for _ in seeds)
            )
            run_means: list[float] = []
            for search_seed in selected_seeds:
                groups = task.groups
                selected_groups = (
                    groups
                    if rng is None
                    else tuple(groups[rng.randrange(len(groups))] for _ in groups)
                )
                group_means: list[float] = []
                for group in selected_groups:
                    repeats = group.repeat_seeds
                    selected_repeats = (
                        repeats
                        if rng is None
                        else tuple(repeats[rng.randrange(len(repeats))] for _ in repeats)
                    )
                    group_means.append(
                        mean(
                            [
                                values[
                                    (
                                        endpoint.hypothesis.value,
                                        model_id,
                                        task.task_id,
                                        search_seed,
                                        group.group_id,
                                        repeat_seed,
                                    )
                                ]
                                for repeat_seed in selected_repeats
                            ]
                        )
                    )
                run_means.append(mean(group_means))
            cell_means.append(mean(run_means))
    return mean(cell_means)


def _bootstrap_estimates(
    preregistration: HierarchicalBootstrapPreregistration,
    endpoint: EndpointHierarchyPlan,
    values: dict[tuple[str, str, str, int, str, int], float],
) -> tuple[float, list[float]]:
    estimate = _endpoint_estimate(preregistration, endpoint, values, rng=None)
    domain_seed = int(
        canonical_sha256(
            {
                "domain": "spiral-harness/hierarchical-nested-bootstrap/v1",
                "bootstrap_seed": preregistration.bootstrap_seed,
                "preregistration_fingerprint": preregistration.fingerprint,
                "hypothesis": endpoint.hypothesis,
            }
        )[:16],
        16,
    )
    rng = random.Random(domain_seed)
    draws = [
        _endpoint_estimate(preregistration, endpoint, values, rng=rng)
        for _ in range(preregistration.bootstrap_draws)
    ]
    return estimate, draws


def analyze_hierarchical_dataset(
    preregistration: HierarchicalBootstrapPreregistration,
    dataset: HierarchicalPairedDataset,
) -> HierarchicalFamilyInference:
    """Run the equal-cell basic bootstrap, SESOI-null Holm, and 10pp rule."""

    checked_preregistration = HierarchicalBootstrapPreregistration.model_validate(
        preregistration.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    checked_dataset = HierarchicalPairedDataset.model_validate(
        dataset.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    validate_complete_hierarchical_dataset(checked_preregistration, checked_dataset)
    values = _lookup(checked_dataset)
    raw: list[tuple[float, float, float, float, float | None, float]] = []
    for endpoint in checked_preregistration.endpoints:
        estimate, draws = _bootstrap_estimates(checked_preregistration, endpoint, values)
        errors = [draw - estimate for draw in draws]
        if len(errors) > 1:
            error_mean = mean(errors)
            standard_error = math.sqrt(
                math.fsum((value - error_mean) ** 2 for value in errors) / (len(errors) - 1)
            )
        else:  # pragma: no cover - schema requires many draws
            standard_error = 0.0
        alpha = checked_preregistration.family_wise_alpha
        marginal_lower = estimate - quantile_r7(errors, 1.0 - alpha / 2.0)
        marginal_upper = estimate - quantile_r7(errors, alpha / 2.0)
        margin_family_lower = (
            estimate
            - quantile_r7(
                errors,
                1.0 - alpha / len(PROJECT_MARGIN_HYPOTHESES),
            )
            if endpoint.hypothesis in PROJECT_MARGIN_HYPOTHESES
            else None
        )
        null_distance = estimate - endpoint.sesoi
        exceedances = sum(error >= null_distance for error in errors)
        p_value = (exceedances + 1.0) / (len(errors) + 1.0)
        raw.append(
            (
                estimate,
                standard_error,
                marginal_lower,
                marginal_upper,
                margin_family_lower,
                p_value,
            )
        )

    p_values = tuple(item[5] for item in raw)
    adjusted, rejected = holm_three(
        p_values,
        alpha=checked_preregistration.family_wise_alpha,
    )
    endpoints = tuple(
        EndpointNestedInference(
            hypothesis=endpoint.hypothesis,
            estimate=raw[index][0],
            standard_error=raw[index][1],
            confidence_level=checked_preregistration.confidence_level,
            marginal_two_sided_lower=raw[index][2],
            marginal_two_sided_upper=raw[index][3],
            project_margin_familywise_one_sided_lower=raw[index][4],
            primary_null="delta-less-than-or-equal-to-endpoint-sesoi",
            one_sided_sesoi_null_p_value=p_values[index],
            holm_adjusted_p_value=adjusted[index],
            holm_rejected=rejected[index],
            sesoi=endpoint.sesoi,
            point_estimate_exceeds_sesoi=raw[index][0] > endpoint.sesoi,
            project_margin_applicable=endpoint.hypothesis in PROJECT_MARGIN_HYPOTHESES,
            project_margin=checked_preregistration.project_margin,
            project_margin_claim_supported=(
                endpoint.hypothesis in PROJECT_MARGIN_HYPOTHESES
                and raw[index][4] is not None
                and raw[index][4] > checked_preregistration.project_margin
            ),
        )
        for index, endpoint in enumerate(checked_preregistration.endpoints)
    )
    margin_items = tuple(item for item in endpoints if item.project_margin_applicable)
    all_rejected = all(item.holm_rejected for item in endpoints)
    return HierarchicalFamilyInference(
        analysis_status="equal-cell-basic-nested-bootstrap-prototype",
        preregistration_fingerprint=checked_preregistration.fingerprint,
        preregistration_purpose=checked_preregistration.purpose,
        dataset_fingerprint=checked_dataset.fingerprint,
        dataset_source=checked_dataset.source,
        bootstrap_draws=checked_preregistration.bootstrap_draws,
        bootstrap_seed=checked_preregistration.bootstrap_seed,
        endpoints=endpoints,
        all_primary_holm_rejected=all_rejected,
        all_primary_margin_tests_passed=all_rejected
        and all(item.point_estimate_exceeds_sesoi for item in endpoints),
        any_project_margin_claim_supported=any(
            item.project_margin_claim_supported for item in margin_items
        ),
        all_project_margin_claims_supported=all(
            item.project_margin_claim_supported for item in margin_items
        ),
        confirmatory_result=False,
        reportable_sealed_evidence=False,
        verified_sealed_evidence=False,
        attested_sealed_closure_verified=False,
        formal_confirmatory_analysis_blocked=True,
        coverage_calibrated=False,
        synthetic_validation_only=checked_dataset.source is DatasetSource.SYNTHETIC_VALIDATION,
    )


__all__ = [
    "FAMILY_WISE_ALPHA",
    "FORMAL_MINIMUM_BOOTSTRAP_DRAWS",
    "HYPOTHESIS_ORDER",
    "PROJECT_MARGIN",
    "PROJECT_MARGIN_HYPOTHESES",
    "DatasetSource",
    "EndpointHierarchyPlan",
    "EndpointNestedInference",
    "HierarchicalBootstrapPreregistration",
    "HierarchicalFamilyInference",
    "HierarchicalHypothesis",
    "HierarchicalPairedDataset",
    "PairedLeafObservation",
    "PreregistrationPurpose",
    "TaskGroupPlan",
    "TaskHierarchyPlan",
    "analyze_hierarchical_dataset",
    "validate_complete_hierarchical_dataset",
]
