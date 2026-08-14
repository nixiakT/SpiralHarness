"""Latent rank-copula sensitivity assumptions for three primary estimators."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

PRIMARY_ESTIMATOR_ORDER = (
    "full-vs-score-real",
    "full-vs-static-real",
    "full-vs-score-verified-repair",
)
Correlation = Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
CorrelationRow = Annotated[tuple[Correlation, ...], Field(min_length=3, max_length=3)]
CorrelationMatrix = Annotated[tuple[CorrelationRow, ...], Field(min_length=3, max_length=3)]


def _cholesky_psd(matrix: CorrelationMatrix) -> tuple[tuple[float, ...], ...]:
    """Return a Cholesky factor, including valid singular PSD matrices."""

    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    tolerance = 1e-12
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - math.fsum(
                factor[row][index] * factor[column][index] for index in range(column)
            )
            if row == column:
                if residual < -tolerance:
                    raise ValueError(
                        "joint primary correlation matrix must be positive semidefinite"
                    )
                factor[row][column] = math.sqrt(max(0.0, residual))
            elif factor[column][column] > tolerance:
                factor[row][column] = residual / factor[column][column]
            elif abs(residual) > tolerance:
                raise ValueError("joint primary correlation matrix must be positive semidefinite")
    return tuple(tuple(row) for row in factor)


class JointPrimarySensitivityAssumptions(ImmutableModel):
    """Latent Gaussian rank parameter; not an exact shared-run covariance.

    The order is explicit because H1 and H2 share FULL, while the third entry
    binds their dependence with the verified-repair endpoint.
    """

    assumption_artifact_sha256: Sha256
    source_label: NonEmptyStr
    estimation_method: NonEmptyStr
    correlation_scale: Literal["latent-gaussian-rank-copula-parameter"]
    hypothesis_order: Annotated[tuple[NonEmptyStr, ...], Field(min_length=3, max_length=3)]
    correlation_matrix: CorrelationMatrix

    @model_validator(mode="after")
    def validate_order_symmetry_and_psd(self) -> JointPrimarySensitivityAssumptions:
        if self.hypothesis_order != PRIMARY_ESTIMATOR_ORDER:
            raise ValueError("joint correlation hypothesis_order does not match the primary family")
        for row in range(3):
            if self.correlation_matrix[row][row] != 1.0:
                raise ValueError("joint primary correlation matrix diagonal must equal one")
            for column in range(row):
                if self.correlation_matrix[row][column] != self.correlation_matrix[column][row]:
                    raise ValueError("joint primary correlation matrix must be symmetric")
        _cholesky_psd(self.correlation_matrix)
        return self


def gaussian_copula_resample[T](
    samples: tuple[Sequence[T], Sequence[T], Sequence[T]],
    *,
    assumptions: JointPrimarySensitivityAssumptions,
    seed: int,
    key: Callable[[T], float],
) -> tuple[list[T], list[T], list[T]]:
    """Draw conditionally IID tuples from empirical marginals and the copula."""

    sample_size = len(samples[0])
    if sample_size == 0 or any(len(sample) != sample_size for sample in samples[1:]):
        raise ValueError("joint copula samples must be non-empty and equally sized")
    factor = _cholesky_psd(assumptions.correlation_matrix)
    rng = random.Random(seed)
    ordered = tuple(sorted(sample, key=key) for sample in samples)
    output: tuple[list[T], list[T], list[T]] = ([], [], [])
    standard_normal = NormalDist()
    for _ in range(sample_size):
        independent = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        latent = tuple(
            math.fsum(factor[row][column] * independent[column] for column in range(3))
            for row in range(3)
        )
        for dimension in range(3):
            probability = standard_normal.cdf(latent[dimension])
            index = min(sample_size - 1, math.floor(probability * sample_size))
            output[dimension].append(ordered[dimension][index])
    return output


__all__ = [
    "PRIMARY_ESTIMATOR_ORDER",
    "JointPrimarySensitivityAssumptions",
    "gaussian_copula_resample",
]
