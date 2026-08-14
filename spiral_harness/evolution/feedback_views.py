"""Narrow structural disclosures for future score-only-matched search.

The schema forbids generic evidence fields and requires role-specific media
type labels.  A label does not authenticate the referenced bytes, however, and
the decision is not producer-attested in phase 1.  Trusted role-scoped loading
and a receipt-bound performance projector remain mandatory runtime work.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.evolution.feedback_media_types import (
    EXPLORATION_INPUTS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
)
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.verification.models import Decision

NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
NormalizedScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NormalizedDelta = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class ScoreResourceTotals(ImmutableModel):
    """Aggregate accounting with no call, task, or trajectory coordinates."""

    schema_version: Literal["1"] = "1"
    total_tokens: NonNegativeInt
    total_latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    total_tool_calls: NonNegativeInt
    total_model_calls: NonNegativeInt
    failed_model_calls: NonNegativeInt
    retry_count: NonNegativeInt
    total_cost_usd: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def _failed_calls_fit_total(self) -> Self:
        if self.failed_model_calls > self.total_model_calls:
            raise ValueError("failed_model_calls must not exceed total_model_calls")
        return self


class ScoreAggregateView(ImmutableModel):
    """One candidate-level score, uncertainty interval, and resource total."""

    schema_version: Literal["1"] = "1"
    candidate_sha256: Sha256
    n_valid_pairs: NonNegativeInt
    n_tasks: NonNegativeInt
    parent_score_mean: NormalizedScore
    candidate_score_mean: NormalizedScore
    mean_delta: NormalizedDelta
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    confidence_lower: NormalizedDelta
    confidence_upper: NormalizedDelta
    resources: ScoreResourceTotals

    @model_validator(mode="after")
    def _counts_and_interval_are_consistent(self) -> Self:
        if self.n_tasks == 0:
            raise ValueError("a score aggregate requires at least one task")
        if self.n_valid_pairs < self.n_tasks:
            raise ValueError("n_valid_pairs must be greater than or equal to n_tasks")
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("confidence_lower must not exceed confidence_upper")
        expected_delta = self.candidate_score_mean - self.parent_score_mean
        if not math.isclose(self.mean_delta, expected_delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("mean_delta must equal candidate_score_mean - parent_score_mean")
        if not self.confidence_lower <= self.mean_delta <= self.confidence_upper:
            raise ValueError("confidence interval must contain mean_delta")
        return self


class PerformanceDecisionBasis(ImmutableModel):
    """Frozen basis for a performance projection, never the full mechanism gate."""

    schema_version: Literal["1"] = "1"
    basis_id: Literal["utility-protected-cost-v1"] = "utility-protected-cost-v1"
    included_checks: tuple[
        Literal["utility"],
        Literal["protected-behavior"],
        Literal["cost"],
    ] = ("utility", "protected-behavior", "cost")
    excluded_mechanism_checks: tuple[
        Literal["activation"],
        Literal["adherence"],
        Literal["behavior"],
    ] = ("activation", "adherence", "behavior")


class ScoreGateDecisionView(ImmutableModel):
    """Declared performance-only shape, not yet an attested projection."""

    schema_version: Literal["1"] = "1"
    projection_scope: Literal["performance-only-not-full-gate"] = "performance-only-not-full-gate"
    candidate_sha256: Sha256
    query_index: NonNegativeInt
    decision: Decision
    performance_policy_version: NonEmptyStr
    performance_policy_config_sha256: Sha256
    basis: PerformanceDecisionBasis = PerformanceDecisionBasis()
    aggregate: ScoreAggregateView

    @model_validator(mode="after")
    def _candidate_binding_matches(self) -> Self:
        if self.candidate_sha256 != self.aggregate.candidate_sha256:
            raise ValueError("gate decision and aggregate refer to different candidates")
        return self


class ScoreOnlyFeedbackView(ImmutableModel):
    """The complete optimizer-visible SCORE disclosure for one search round.

    Every score-bearing value is inline and typed, and the only references have
    exact role labels.  Phase 2 must still authenticate their canonical typed
    contents against the frozen benchmark binding and attest the decision.
    """

    schema_version: Literal["2"] = "2"
    disclosure_schema: Literal["score-only-aggregate-and-performance-decision-v2"] = (
        "score-only-aggregate-and-performance-decision-v2"
    )
    runtime_role_binding_attested: Literal[False] = False
    performance_projection_attested: Literal[False] = False
    baseline_kind: Literal[BaselineKind.SCORE_ONLY_MATCHED] = BaselineKind.SCORE_ONLY_MATCHED
    round_index: NonNegativeInt
    benchmark_metadata_ref: ArtifactRef
    exploration_inputs_ref: ArtifactRef
    exploration_aggregate: ScoreAggregateView
    prior_gate_decision: ScoreGateDecisionView | None = None

    @model_validator(mode="after")
    def _refs_and_gate_sequence_are_valid(self) -> Self:
        if self.benchmark_metadata_ref.media_type != SAFE_BENCHMARK_METADATA_MEDIA_TYPE:
            raise ValueError("benchmark_metadata_ref must use the exact safe-metadata media type")
        if self.exploration_inputs_ref.media_type != EXPLORATION_INPUTS_MEDIA_TYPE:
            raise ValueError(
                "exploration_inputs_ref must use the exact exploration-inputs media type"
            )
        prior = self.prior_gate_decision
        if prior is not None and prior.query_index >= self.round_index:
            raise ValueError("prior gate query_index must precede the current round")
        return self


__all__ = [
    "PerformanceDecisionBasis",
    "ScoreAggregateView",
    "ScoreGateDecisionView",
    "ScoreOnlyFeedbackView",
    "ScoreResourceTotals",
]
