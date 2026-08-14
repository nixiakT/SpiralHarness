"""Small deterministic statistics used by hierarchical power prototypes."""

from __future__ import annotations

import math
from statistics import NormalDist


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def quantile_r7(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("a quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be between zero and one")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def holm_three(
    p_values: tuple[float, float, float],
    *,
    alpha: float,
) -> tuple[tuple[float, float, float], tuple[bool, bool, bool]]:
    ordered = sorted(range(3), key=lambda index: (p_values[index], index))
    adjusted = [1.0, 1.0, 1.0]
    rejected = [False, False, False]
    running = 0.0
    still_rejecting = True
    for rank, index in enumerate(ordered):
        multiplier = 3 - rank
        running = max(running, min(1.0, multiplier * p_values[index]))
        adjusted[index] = running
        if still_rejecting and p_values[index] <= alpha / multiplier:
            rejected[index] = True
        else:
            still_rejecting = False
    return (
        (adjusted[0], adjusted[1], adjusted[2]),
        (rejected[0], rejected[1], rejected[2]),
    )


def wilson_bounds(
    successes: int,
    replicates: int,
    confidence_level: float,
) -> tuple[float, float]:
    estimate = successes / replicates
    z_score = NormalDist().inv_cdf(confidence_level)
    z_squared = z_score**2
    denominator = 1.0 + z_squared / replicates
    center = (estimate + z_squared / (2.0 * replicates)) / denominator
    half_width = (
        z_score
        * math.sqrt(estimate * (1.0 - estimate) / replicates + z_squared / (4.0 * replicates**2))
        / denominator
    )
    return (
        min(estimate, max(0.0, center - half_width)),
        max(estimate, min(1.0, center + half_width)),
    )


__all__ = ["holm_three", "mean", "quantile_r7", "wilson_bounds"]
