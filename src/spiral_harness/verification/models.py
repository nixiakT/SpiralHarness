"""Typed contracts for the trusted verification plane.

The verifier deliberately owns its own small domain model.  Candidate code may
produce observations, but it does not get to reinterpret malformed or missing
evidence after the fact.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    field_validator,
    model_validator,
)


class VerificationModel(BaseModel):
    """Base class for immutable, typo-resistant verification artifacts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class TrialStatus(StrEnum):
    """Terminal status emitted by a benchmark worker."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    TASK_ERROR = "task_error"
    INFRA_ERROR = "infra_error"
    POLICY_VIOLATION = "policy_violation"


class TrialObservation(VerificationModel):
    """One harness rollout on one task and one predeclared seed."""

    task_id: str
    seed: int
    harness_id: str
    status: TrialStatus = TrialStatus.COMPLETED
    score: float | None = Field(default=None, allow_inf_nan=False)
    slice_tags: tuple[str, ...] = ()
    tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    tool_calls: int = Field(default=0, ge=0)
    violations: tuple[str, ...] = ()
    execution_fingerprint: str

    @field_validator("task_id", "harness_id", "execution_fingerprint")
    @classmethod
    def _non_empty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("slice_tags")
    @classmethod
    def _canonical_slice_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(item.strip() for item in value))
        if any(not item for item in normalized):
            raise ValueError("slice tags must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("slice tags must be unique")
        return normalized

    @field_validator("violations")
    @classmethod
    def _non_empty_violations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("violations must not be empty")
        return value


class MechanismCheck(VerificationModel):
    """One falsifiable check in a candidate's proposed mechanism chain.

    ``passed=None`` means the check was not observed or could not be evaluated;
    it is intentionally different from a failed check.
    """

    name: str = Field(validation_alias=AliasChoices("name", "check_id"))
    passed: StrictBool | None = None
    details: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def _valid_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("evidence refs must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence refs must be unique")
        return normalized

    @model_validator(mode="after")
    def _passed_check_requires_evidence(self) -> MechanismCheck:
        if self.passed is True and not self.evidence_refs:
            raise ValueError("a passed mechanism check requires evidence_refs")
        return self

    @property
    def check_id(self) -> str:
        """Compatibility/readability alias for callers using ``check_id``."""

        return self.name


class MechanismEvidence(VerificationModel):
    """Mechanism evidence collected independently of aggregate score."""

    checks: tuple[MechanismCheck, ...] = ()
    candidate_harness_id: str | None = None

    @field_validator("candidate_harness_id")
    @classmethod
    def _non_empty_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class GateConfig(VerificationModel):
    """Versioned, predeclared promotion policy.

    A protected-slice value is its minimum tolerated score delta.  For example,
    ``{"safety": -0.02}`` permits at most a two-point regression, and the
    slice confidence lower bound must clear that floor.
    """

    version: str = "m0"
    min_tasks: int = Field(default=10, ge=1)
    min_effect: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    bootstrap_seed: int = 0

    require_complete_pairs: bool = True
    expected_task_ids: tuple[str, ...] = ()
    expected_seeds: tuple[int, ...] = ()

    required_mechanism_checks: tuple[str, ...] = ()
    protected_slice_floors: dict[str, FiniteFloat] = Field(default_factory=dict)
    min_slice_tasks: int = Field(default=1, ge=1)

    max_single_task_regression: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        validation_alias=AliasChoices("max_single_task_regression", "max_single_task_drop"),
    )
    max_regression_rate: float | None = Field(default=None, ge=0, le=1)
    regression_tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    max_tokens_ratio: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_latency_ratio: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tool_calls_ratio: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("version")
    @classmethod
    def _non_empty_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("expected_task_ids", "required_mechanism_checks")
    @classmethod
    def _unique_non_empty_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("items must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("items must be unique")
        return normalized

    @field_validator("expected_seeds")
    @classmethod
    def _unique_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("expected_seeds must be unique")
        return value

    @field_validator("protected_slice_floors")
    @classmethod
    def _valid_slice_floors(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in value):
            raise ValueError("protected slice names must not be empty")
        if any(not math.isfinite(floor) for floor in value.values()):
            raise ValueError("protected slice floors must be finite")
        return value

    @model_validator(mode="after")
    def _expected_seeds_need_tasks(self) -> GateConfig:
        if self.expected_seeds and not self.expected_task_ids:
            raise ValueError("expected_seeds requires expected_task_ids")
        return self

    @property
    def max_single_task_drop(self) -> float | None:
        return self.max_single_task_regression


class ConfidenceInterval(VerificationModel):
    confidence_level: float = Field(gt=0, lt=1)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)


class TaskComparison(VerificationModel):
    """Task-level statistical unit after averaging all matched seeds."""

    task_id: str
    seeds: tuple[int, ...]
    slice_tags: tuple[str, ...]
    parent_score: float
    candidate_score: float
    delta: float
    parent_tokens: float
    candidate_tokens: float
    parent_latency_ms: float
    candidate_latency_ms: float
    parent_tool_calls: float
    candidate_tool_calls: float

    @field_validator("slice_tags")
    @classmethod
    def _canonical_slice_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(item.strip() for item in value))
        if any(not item for item in normalized):
            raise ValueError("slice tags must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("slice tags must be unique")
        return normalized


class SliceMetrics(VerificationModel):
    slice_name: str
    n_tasks: int
    parent_mean: float
    candidate_mean: float
    mean_delta: float
    confidence_interval: ConfidenceInterval

    @property
    def ci(self) -> ConfidenceInterval:
        return self.confidence_interval


class ComparisonMetrics(VerificationModel):
    """Fully auditable paired statistics; ``n_tasks`` is the sample size."""

    n_valid_pairs: int
    n_tasks: int
    parent_mean: float
    candidate_mean: float
    mean_delta: float
    confidence_interval: ConfidenceInterval
    wins: int
    ties: int
    losses: int
    regression_rate: float
    worst_task_delta: float

    parent_tokens_mean: float
    candidate_tokens_mean: float
    tokens_ratio: float | None
    parent_latency_ms_mean: float
    candidate_latency_ms_mean: float
    latency_ratio: float | None
    parent_tool_calls_mean: float
    candidate_tool_calls_mean: float
    tool_calls_ratio: float | None

    task_comparisons: tuple[TaskComparison, ...]
    slices: dict[str, SliceMetrics]

    @property
    def ci(self) -> ConfidenceInterval:
        return self.confidence_interval


class PairingAudit(VerificationModel):
    """Everything excluded from, or capable of invalidating, comparison."""

    parent_harness_ids: tuple[str, ...] = ()
    candidate_harness_ids: tuple[str, ...] = ()
    parent_status_counts: dict[str, int] = Field(default_factory=dict)
    candidate_status_counts: dict[str, int] = Field(default_factory=dict)

    duplicate_parent_pairs: tuple[str, ...] = ()
    duplicate_candidate_pairs: tuple[str, ...] = ()
    missing_parent_pairs: tuple[str, ...] = ()
    missing_candidate_pairs: tuple[str, ...] = ()
    unexpected_parent_pairs: tuple[str, ...] = ()
    unexpected_candidate_pairs: tuple[str, ...] = ()
    incomplete_pairs: tuple[str, ...] = ()
    fingerprint_mismatches: tuple[str, ...] = ()
    slice_tag_mismatches: tuple[str, ...] = ()
    expected_missing_pairs: tuple[str, ...] = ()
    expected_missing_tasks: tuple[str, ...] = ()
    integrity_errors: tuple[str, ...] = ()

    @property
    def structurally_valid(self) -> bool:
        return not self.integrity_errors

    @property
    def complete(self) -> bool:
        return not (
            self.missing_parent_pairs
            or self.missing_candidate_pairs
            or self.unexpected_parent_pairs
            or self.unexpected_candidate_pairs
            or self.incomplete_pairs
            or self.expected_missing_pairs
            or self.expected_missing_tasks
        )


class ComparisonResult(VerificationModel):
    audit: PairingAudit
    metrics: ComparisonMetrics | None = None


class GateCheckOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class GateCheck(VerificationModel):
    name: str
    outcome: GateCheckOutcome
    reasons: tuple[str, ...]
    metrics: dict[str, Any] = Field(default_factory=dict)


class Decision(StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class GateDecision(VerificationModel):
    """Versioned decision artifact returned by the deterministic gate."""

    decision: Decision
    gate_version: str
    gate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_harness_id: str | None
    candidate_harness_id: str | None
    checks: tuple[GateCheck, ...]
    reasons: tuple[str, ...]
    comparison: ComparisonResult

    @property
    def metrics(self) -> ComparisonMetrics | None:
        """Convenient access to the comparison metrics in the audit artifact."""

        return self.comparison.metrics


__all__ = [
    "ComparisonMetrics",
    "ComparisonResult",
    "ConfidenceInterval",
    "Decision",
    "GateCheck",
    "GateCheckOutcome",
    "GateConfig",
    "GateDecision",
    "MechanismCheck",
    "MechanismEvidence",
    "PairingAudit",
    "SliceMetrics",
    "TaskComparison",
    "TrialObservation",
    "TrialStatus",
]
