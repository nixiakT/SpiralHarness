"""Bounded synthetic generator for hierarchical power validation."""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel
from spiral_harness.experiments.power_hierarchical import (
    PROJECT_MARGIN,
    PROJECT_MARGIN_HYPOTHESES,
    DatasetSource,
    HierarchicalBootstrapPreregistration,
    HierarchicalHypothesis,
    HierarchicalPairedDataset,
    PairedLeafObservation,
)

_STANDARD_UNIFORM_HALF_RANGE = math.sqrt(3.0)


class ContinuousHierarchyAssumption(ImmutableModel):
    """Bounded additive random effects for one real-score contrast."""

    hypothesis: HierarchicalHypothesis
    alternative_effect: Annotated[
        float,
        Field(gt=PROJECT_MARGIN, lt=1.0, strict=True, allow_inf_nan=False),
    ]
    search_seed_sd: Annotated[
        float,
        Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    task_group_sd: Annotated[
        float,
        Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    repeat_sd: Annotated[
        float,
        Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    marginal_generator: Literal["bounded-standardised-uniform"] = "bounded-standardised-uniform"

    @model_validator(mode="after")
    def _real_endpoint_and_bounded_support(self) -> Self:
        if self.hypothesis not in PROJECT_MARGIN_HYPOTHESES:
            raise ValueError("continuous assumptions are only defined for H1 and H2")
        maximum_noise = _STANDARD_UNIFORM_HALF_RANGE * (
            self.search_seed_sd + self.task_group_sd + self.repeat_sd
        )
        if self.alternative_effect + maximum_noise > 1.0:
            raise ValueError("continuous alternative distribution exceeds the [-1, 1] support")
        if maximum_noise > 1.0:
            raise ValueError("continuous null distribution exceeds the [-1, 1] support")
        return self


class SharedContinuousCorrelationAssumption(ImmutableModel):
    """H1/H2 Gaussian-copula correlations at each hierarchy level."""

    correlation_scale: Literal["latent-gaussian-copula"] = "latent-gaussian-copula"
    search_seed_correlation: Annotated[
        float,
        Field(ge=-1.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    task_group_correlation: Annotated[
        float,
        Field(ge=-1.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    repeat_correlation: Annotated[
        float,
        Field(ge=-1.0, le=1.0, strict=True, allow_inf_nan=False),
    ]


class PairedBinaryAssumption(ImmutableModel):
    """Paired categorical probabilities for the verified-repair contrast."""

    hypothesis: HierarchicalHypothesis
    alternative_positive_probability: Annotated[
        float,
        Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    alternative_negative_probability: Annotated[
        float,
        Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
    ]
    null_discordant_probability: Annotated[
        float,
        Field(ge=0.15, le=1.0, strict=True, allow_inf_nan=False),
    ]
    null_boundary_effect: Literal[0.15] = 0.15
    generator: Literal["iid-paired-categorical-within-fixed-hierarchy"] = (
        "iid-paired-categorical-within-fixed-hierarchy"
    )

    @model_validator(mode="after")
    def _verified_repair_probabilities(self) -> Self:
        expected = HierarchicalHypothesis.FULL_VS_SCORE_VERIFIED_REPAIR
        if self.hypothesis is not expected:
            raise ValueError("paired-binary assumptions are only defined for H3")
        if self.alternative_positive_probability + self.alternative_negative_probability > 1.0:
            raise ValueError("alternative paired-binary probabilities exceed one")
        effect = self.alternative_positive_probability - self.alternative_negative_probability
        if effect <= 0.15:
            raise ValueError("H3 alternative effect must be strictly greater than its 0.15 SESOI")
        return self

    @property
    def null_positive_probability(self) -> float:
        return (self.null_discordant_probability + self.null_boundary_effect) / 2.0

    @property
    def null_negative_probability(self) -> float:
        return (self.null_discordant_probability - self.null_boundary_effect) / 2.0


def _uniform_copula_pair(rng: random.Random, correlation: float) -> tuple[float, float]:
    first = rng.gauss(0.0, 1.0)
    second = correlation * first + math.sqrt(max(0.0, 1.0 - correlation**2)) * rng.gauss(0.0, 1.0)
    normal = NormalDist()
    return (
        _STANDARD_UNIFORM_HALF_RANGE * (2.0 * normal.cdf(first) - 1.0),
        _STANDARD_UNIFORM_HALF_RANGE * (2.0 * normal.cdf(second) - 1.0),
    )


def _coordinate_rng(base_seed: int, *parts: object) -> random.Random:
    seed = int(
        canonical_sha256(
            {
                "domain": "spiral-harness/nested-bootstrap-power-coordinate/v1",
                "base_seed": base_seed,
                "parts": tuple(str(part) for part in parts),
            }
        )[:16],
        16,
    )
    return random.Random(seed)


def _bounded(value: float) -> float:
    if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        raise AssertionError("validated synthetic continuous support was exceeded")
    return min(1.0, max(-1.0, value))


def simulate_nested_dataset(
    preregistration: HierarchicalBootstrapPreregistration,
    continuous_assumptions: tuple[
        ContinuousHierarchyAssumption,
        ContinuousHierarchyAssumption,
    ],
    correlation: SharedContinuousCorrelationAssumption,
    binary: PairedBinaryAssumption,
    *,
    alternative: bool,
    seed: int,
) -> HierarchicalPairedDataset:
    """Generate one bounded complete roster with coordinate-domain RNG streams."""

    effects = tuple(
        item.alternative_effect if alternative else preregistration.endpoints[index].sesoi
        for index, item in enumerate(continuous_assumptions)
    )
    first_endpoint = preregistration.endpoints[0]
    observations: list[PairedLeafObservation] = []
    for model_id in preregistration.model_ids:
        for task in first_endpoint.tasks:
            for search_seed in preregistration.independent_search_seeds:
                seed_pair = _uniform_copula_pair(
                    _coordinate_rng(seed, "search", model_id, task.task_id, search_seed),
                    correlation.search_seed_correlation,
                )
                for group in task.groups:
                    group_key = (model_id, task.task_id, search_seed, group.group_id)
                    group_pair = _uniform_copula_pair(
                        _coordinate_rng(seed, "group", *group_key),
                        correlation.task_group_correlation,
                    )
                    for repeat_seed in group.repeat_seeds:
                        repeat_pair = _uniform_copula_pair(
                            _coordinate_rng(seed, "repeat", *group_key, repeat_seed),
                            correlation.repeat_correlation,
                        )
                        for index, assumption in enumerate(continuous_assumptions):
                            difference = _bounded(
                                effects[index]
                                + assumption.search_seed_sd * seed_pair[index]
                                + assumption.task_group_sd * group_pair[index]
                                + assumption.repeat_sd * repeat_pair[index]
                            )
                            observations.append(
                                PairedLeafObservation(
                                    hypothesis=assumption.hypothesis,
                                    model_id=model_id,
                                    task_id=task.task_id,
                                    search_seed=search_seed,
                                    group_id=group.group_id,
                                    repeat_seed=repeat_seed,
                                    difference=difference,
                                )
                            )

    positive = (
        binary.alternative_positive_probability if alternative else binary.null_positive_probability
    )
    negative = (
        binary.alternative_negative_probability if alternative else binary.null_negative_probability
    )
    for model_id in preregistration.model_ids:
        for task in preregistration.endpoints[2].tasks:
            for search_seed in preregistration.independent_search_seeds:
                for group in task.groups:
                    for repeat_seed in group.repeat_seeds:
                        draw = _coordinate_rng(
                            seed,
                            "binary",
                            model_id,
                            task.task_id,
                            search_seed,
                            group.group_id,
                            repeat_seed,
                        ).random()
                        difference = (
                            1.0 if draw < positive else -1.0 if draw < positive + negative else 0.0
                        )
                        observations.append(
                            PairedLeafObservation(
                                hypothesis=binary.hypothesis,
                                model_id=model_id,
                                task_id=task.task_id,
                                search_seed=search_seed,
                                group_id=group.group_id,
                                repeat_seed=repeat_seed,
                                difference=difference,
                            )
                        )
    return HierarchicalPairedDataset(
        preregistration_fingerprint=preregistration.fingerprint,
        source=DatasetSource.SYNTHETIC_VALIDATION,
        observations=tuple(observations),
    )


__all__ = [
    "ContinuousHierarchyAssumption",
    "PairedBinaryAssumption",
    "SharedContinuousCorrelationAssumption",
    "simulate_nested_dataset",
]
