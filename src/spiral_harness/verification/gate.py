"""Deterministic, auditable promotion gate."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.verification.models import (
    ComparisonResult,
    Decision,
    GateCheck,
    GateCheckOutcome,
    GateConfig,
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
    TrialStatus,
)
from spiral_harness.verification.statistics import compare_trials


def _validated_model_copy[ModelT](model_type: type[ModelT], value: ModelT) -> ModelT:
    """Rebuild a model so Pydantic's unchecked copy/construct APIs cannot cross the gate."""

    model_dump = getattr(value, "model_dump", None)
    if model_dump is None:
        raise TypeError(f"expected {model_type.__name__}, got {type(value).__name__}")
    return model_type.model_validate(
        model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )


def _validated_optional_harness_id(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _check(
    name: str,
    outcome: GateCheckOutcome,
    *reasons: str,
    metrics: dict[str, object] | None = None,
) -> GateCheck:
    return GateCheck(
        name=name,
        outcome=outcome,
        reasons=tuple(reasons),
        metrics=metrics or {},
    )


def _integrity_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    audit = comparison.audit
    hard_reasons = list(audit.integrity_errors)
    incomplete_reasons: list[str] = []
    if config.require_complete_pairs:
        if audit.missing_parent_pairs:
            incomplete_reasons.append(
                f"{len(audit.missing_parent_pairs)} pair(s) are missing a parent observation"
            )
        if audit.missing_candidate_pairs:
            incomplete_reasons.append(
                f"{len(audit.missing_candidate_pairs)} pair(s) are missing a candidate observation"
            )
        if audit.incomplete_pairs:
            incomplete_reasons.append(
                f"{len(audit.incomplete_pairs)} pair(s) did not complete with numeric scores"
            )
        if audit.expected_missing_pairs:
            incomplete_reasons.append(
                f"{len(audit.expected_missing_pairs)} preregistered pair(s) are absent"
            )
        if audit.expected_missing_tasks:
            incomplete_reasons.append(
                f"{len(audit.expected_missing_tasks)} preregistered task(s) are absent"
            )

    metrics: dict[str, object] = {
        "duplicate_parent_pairs": list(audit.duplicate_parent_pairs),
        "duplicate_candidate_pairs": list(audit.duplicate_candidate_pairs),
        "missing_parent_pairs": list(audit.missing_parent_pairs),
        "missing_candidate_pairs": list(audit.missing_candidate_pairs),
        "unexpected_parent_pairs": list(audit.unexpected_parent_pairs),
        "unexpected_candidate_pairs": list(audit.unexpected_candidate_pairs),
        "incomplete_pairs": list(audit.incomplete_pairs),
        "fingerprint_mismatches": list(audit.fingerprint_mismatches),
        "slice_tag_mismatches": list(audit.slice_tag_mismatches),
        "expected_missing_pairs": list(audit.expected_missing_pairs),
        "expected_missing_tasks": list(audit.expected_missing_tasks),
        "parent_status_counts": audit.parent_status_counts,
        "candidate_status_counts": audit.candidate_status_counts,
    }
    if hard_reasons:
        return _check(
            "integrity",
            GateCheckOutcome.FAIL,
            *hard_reasons,
            *incomplete_reasons,
            metrics=metrics,
        )
    if incomplete_reasons:
        return _check(
            "integrity",
            GateCheckOutcome.INCONCLUSIVE,
            *incomplete_reasons,
            metrics=metrics,
        )
    return _check(
        "integrity",
        GateCheckOutcome.PASS,
        "all supplied observations are unique, matched, and complete",
        metrics=metrics,
    )


def _sample_size_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    n_tasks = comparison.metrics.n_tasks if comparison.metrics is not None else 0
    metrics = {"n_tasks": n_tasks, "min_tasks": config.min_tasks}
    if n_tasks < config.min_tasks:
        return _check(
            "sample_size",
            GateCheckOutcome.INCONCLUSIVE,
            f"only {n_tasks} valid task-level units; at least {config.min_tasks} are required",
            metrics=metrics,
        )
    return _check(
        "sample_size",
        GateCheckOutcome.PASS,
        f"{n_tasks} valid task-level units meet the minimum",
        metrics=metrics,
    )


def _policy_check(
    parent: Sequence[TrialObservation], candidate: Sequence[TrialObservation]
) -> GateCheck:
    violations: list[str] = []
    for side, observations in (("parent", parent), ("candidate", candidate)):
        for observation in observations:
            label = f"{side}:{observation.task_id}::seed={observation.seed}"
            if observation.status is TrialStatus.POLICY_VIOLATION:
                violations.append(f"{label} has policy_violation status")
            violations.extend(f"{label}: {violation}" for violation in observation.violations)
    if violations:
        return _check(
            "policy",
            GateCheckOutcome.FAIL,
            "one or more trials reported a policy violation",
            metrics={"violations": violations, "count": len(violations)},
        )
    return _check(
        "policy",
        GateCheckOutcome.PASS,
        "no policy violations were reported",
        metrics={"violations": [], "count": 0},
    )


def _normalize_mechanism_evidence(
    evidence: MechanismEvidence | Sequence[MechanismCheck] | None,
) -> MechanismEvidence | None:
    if evidence is None:
        return evidence
    if isinstance(evidence, Mapping):
        raise TypeError(
            "raw mechanism mappings are not trusted evidence; provide MechanismEvidence"
        )
    if isinstance(evidence, MechanismEvidence):
        return _validated_model_copy(MechanismEvidence, evidence)
    checks = tuple(_validated_model_copy(MechanismCheck, check) for check in evidence)
    return MechanismEvidence(checks=checks)


def _mechanism_check(
    evidence: MechanismEvidence | None,
    config: GateConfig,
    candidate_harness_id: str | None,
) -> GateCheck:
    required = config.required_mechanism_checks
    if evidence is None:
        by_name: dict[str, MechanismCheck] = {}
        duplicates: tuple[str, ...] = ()
    else:
        counts = Counter(check.name for check in evidence.checks)
        duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))
        by_name = {check.name: check for check in evidence.checks}

    failed = tuple(name for name in required if name in by_name and by_name[name].passed is False)
    missing = tuple(
        name for name in required if name not in by_name or by_name[name].passed is None
    )
    hard_reasons: list[str] = []
    if duplicates:
        hard_reasons.append(f"duplicate mechanism checks: {', '.join(duplicates)}")
    if evidence is not None:
        if required and evidence.candidate_harness_id is None:
            hard_reasons.append("required mechanism evidence is not bound to a candidate harness")
        elif required and candidate_harness_id is None:
            hard_reasons.append(
                "candidate harness ID is unavailable for mechanism evidence binding"
            )
        elif (
            evidence.candidate_harness_id is not None
            and candidate_harness_id is not None
            and evidence.candidate_harness_id != candidate_harness_id
        ):
            hard_reasons.append("mechanism evidence belongs to a different candidate harness")
    if failed:
        hard_reasons.append(f"required mechanism checks failed: {', '.join(failed)}")

    metrics: dict[str, object] = {
        "required": list(required),
        "passed": [name for name in required if name in by_name and by_name[name].passed is True],
        "failed": list(failed),
        "missing": list(missing),
        "duplicates": list(duplicates),
        "evidence_candidate_harness_id": (
            evidence.candidate_harness_id if evidence is not None else None
        ),
        "observed": {
            name: check.passed for name, check in sorted(by_name.items(), key=lambda item: item[0])
        },
    }
    if hard_reasons:
        reasons = [*hard_reasons]
        if missing:
            reasons.append(f"required mechanism checks missing: {', '.join(missing)}")
        return _check(
            "mechanism",
            GateCheckOutcome.FAIL,
            *reasons,
            metrics=metrics,
        )
    if missing:
        return _check(
            "mechanism",
            GateCheckOutcome.INCONCLUSIVE,
            f"required mechanism checks missing: {', '.join(missing)}",
            metrics=metrics,
        )
    return _check(
        "mechanism",
        GateCheckOutcome.PASS,
        "all required mechanism checks passed" if required else "no mechanism checks are required",
        metrics=metrics,
    )


def _primary_effect_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    metrics = comparison.metrics
    if metrics is None:
        return _check(
            "primary_effect",
            GateCheckOutcome.INCONCLUSIVE,
            "no valid task-level score comparison is available",
            metrics={"min_effect": config.min_effect, "required_lcb": 0.0},
        )

    effect_metrics = {
        "parent_mean": metrics.parent_mean,
        "candidate_mean": metrics.candidate_mean,
        "mean_delta": metrics.mean_delta,
        "min_effect": config.min_effect,
        "ci_lower": metrics.confidence_interval.lower,
        "ci_upper": metrics.confidence_interval.upper,
        "confidence_level": metrics.confidence_interval.confidence_level,
        "required_lcb_exclusive": 0.0,
    }
    if metrics.mean_delta < config.min_effect:
        return _check(
            "primary_effect",
            GateCheckOutcome.FAIL,
            f"mean delta {metrics.mean_delta:.6g} is below minimum effect {config.min_effect:.6g}",
            metrics=effect_metrics,
        )
    if metrics.confidence_interval.lower <= 0:
        return _check(
            "primary_effect",
            GateCheckOutcome.INCONCLUSIVE,
            f"confidence lower bound {metrics.confidence_interval.lower:.6g} does not clear zero",
            metrics=effect_metrics,
        )
    return _check(
        "primary_effect",
        GateCheckOutcome.PASS,
        "point effect and confidence lower bound satisfy the promotion thresholds",
        metrics=effect_metrics,
    )


def _protected_slices_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    if not config.protected_slice_floors:
        return _check(
            "protected_slices",
            GateCheckOutcome.PASS,
            "no protected slices are configured",
            metrics={"slices": {}},
        )
    if comparison.metrics is None:
        return _check(
            "protected_slices",
            GateCheckOutcome.INCONCLUSIVE,
            "no valid task-level slice comparison is available",
            metrics={"slices": {}},
        )

    failures: list[str] = []
    uncertain: list[str] = []
    audit: dict[str, object] = {}
    for slice_name, floor in sorted(config.protected_slice_floors.items()):
        slice_metrics = comparison.metrics.slices.get(slice_name)
        if slice_metrics is None:
            uncertain.append(f"protected slice {slice_name!r} has no valid tasks")
            audit[slice_name] = {"floor": floor, "n_tasks": 0}
            continue
        audit[slice_name] = {
            "floor": floor,
            "n_tasks": slice_metrics.n_tasks,
            "mean_delta": slice_metrics.mean_delta,
            "ci_lower": slice_metrics.confidence_interval.lower,
            "ci_upper": slice_metrics.confidence_interval.upper,
        }
        if slice_metrics.n_tasks < config.min_slice_tasks:
            uncertain.append(
                f"protected slice {slice_name!r} has {slice_metrics.n_tasks} task(s); "
                f"{config.min_slice_tasks} required"
            )
        elif slice_metrics.mean_delta < floor:
            failures.append(
                f"protected slice {slice_name!r} mean delta {slice_metrics.mean_delta:.6g} "
                f"is below floor {floor:.6g}"
            )
        elif slice_metrics.confidence_interval.lower <= floor or math.isclose(
            slice_metrics.confidence_interval.lower,
            floor,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            uncertain.append(
                f"protected slice {slice_name!r} lower bound "
                f"{slice_metrics.confidence_interval.lower:.6g} does not clear floor {floor:.6g}"
            )

    if failures:
        return _check(
            "protected_slices",
            GateCheckOutcome.FAIL,
            *failures,
            *uncertain,
            metrics={"slices": audit, "min_slice_tasks": config.min_slice_tasks},
        )
    if uncertain:
        return _check(
            "protected_slices",
            GateCheckOutcome.INCONCLUSIVE,
            *uncertain,
            metrics={"slices": audit, "min_slice_tasks": config.min_slice_tasks},
        )
    return _check(
        "protected_slices",
        GateCheckOutcome.PASS,
        "all protected slices satisfy their non-inferiority floors",
        metrics={"slices": audit, "min_slice_tasks": config.min_slice_tasks},
    )


def _task_regression_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    configured = (
        config.max_single_task_regression is not None or config.max_regression_rate is not None
    )
    if not configured:
        return _check(
            "task_regressions",
            GateCheckOutcome.PASS,
            "no per-task regression constraints are configured",
            metrics={},
        )
    if comparison.metrics is None:
        return _check(
            "task_regressions",
            GateCheckOutcome.INCONCLUSIVE,
            "no valid task-level deltas are available",
            metrics={},
        )

    deltas = [task.delta for task in comparison.metrics.task_comparisons]
    worst_delta = min(deltas)
    regression_rate = sum(delta < -config.regression_tolerance for delta in deltas) / len(deltas)
    failures: list[str] = []
    if (
        config.max_single_task_regression is not None
        and worst_delta < -config.max_single_task_regression
    ):
        failures.append(
            f"worst task delta {worst_delta:.6g} exceeds allowed regression "
            f"{config.max_single_task_regression:.6g}"
        )
    if config.max_regression_rate is not None and regression_rate > config.max_regression_rate:
        failures.append(
            f"regression rate {regression_rate:.6g} exceeds limit {config.max_regression_rate:.6g}"
        )
    audit = {
        "worst_task_delta": worst_delta,
        "max_single_task_regression": config.max_single_task_regression,
        "regression_rate": regression_rate,
        "max_regression_rate": config.max_regression_rate,
        "regression_tolerance": config.regression_tolerance,
    }
    if failures:
        return _check("task_regressions", GateCheckOutcome.FAIL, *failures, metrics=audit)
    return _check(
        "task_regressions",
        GateCheckOutcome.PASS,
        "per-task regression constraints are satisfied",
        metrics=audit,
    )


def _resource_check(comparison: ComparisonResult, config: GateConfig) -> GateCheck:
    limits = {
        "tokens": config.max_tokens_ratio,
        "latency": config.max_latency_ratio,
        "tool_calls": config.max_tool_calls_ratio,
    }
    configured = {name: limit for name, limit in limits.items() if limit is not None}
    if not configured:
        return _check(
            "resources",
            GateCheckOutcome.PASS,
            "no resource ratio constraints are configured",
            metrics={"limits": limits},
        )
    if comparison.metrics is None:
        return _check(
            "resources",
            GateCheckOutcome.INCONCLUSIVE,
            "no valid paired resource measurements are available",
            metrics={"limits": limits},
        )

    metrics = comparison.metrics
    values = {
        "tokens": (
            metrics.parent_tokens_mean,
            metrics.candidate_tokens_mean,
            metrics.tokens_ratio,
        ),
        "latency": (
            metrics.parent_latency_ms_mean,
            metrics.candidate_latency_ms_mean,
            metrics.latency_ratio,
        ),
        "tool_calls": (
            metrics.parent_tool_calls_mean,
            metrics.candidate_tool_calls_mean,
            metrics.tool_calls_ratio,
        ),
    }
    failures: list[str] = []
    audit: dict[str, object] = {}
    for name, limit in limits.items():
        parent_mean, candidate_mean, ratio = values[name]
        audit[name] = {
            "parent_mean": parent_mean,
            "candidate_mean": candidate_mean,
            "ratio": ratio,
            "max_ratio": limit,
        }
        if limit is None:
            continue
        if not (
            math.isfinite(parent_mean)
            and math.isfinite(candidate_mean)
            and parent_mean >= 0
            and candidate_mean >= 0
        ):
            failures.append(f"{name} measurements must be finite and non-negative")
        elif parent_mean == 0 and candidate_mean > 0:
            failures.append(
                f"{name} ratio is unbounded because parent use is zero "
                "and candidate use is positive"
            )
        elif candidate_mean > limit * parent_mean:
            ratio_text = "unbounded" if ratio is None else f"{ratio:.6g}"
            failures.append(f"{name} ratio {ratio_text} exceeds limit {limit:.6g}")

    if failures:
        return _check("resources", GateCheckOutcome.FAIL, *failures, metrics=audit)
    return _check(
        "resources",
        GateCheckOutcome.PASS,
        "all configured resource ratios are within their hard limits",
        metrics=audit,
    )


class PromotionGate:
    """Apply a frozen :class:`GateConfig` to independently produced evidence."""

    def __init__(self, config: GateConfig | None = None) -> None:
        supplied = config or GateConfig()
        validated = _validated_model_copy(GateConfig, supplied)
        self._config_payload = canonical_json_bytes(validated)
        self._config_sha256 = sha256_bytes(self._config_payload)

    @property
    def config(self) -> GateConfig:
        """Return an independent validated copy of the gate's frozen policy snapshot."""

        return GateConfig.model_validate_json(self._config_payload)

    def evaluate(
        self,
        parent_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
        candidate_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
        mechanism_evidence: MechanismEvidence | Sequence[MechanismCheck] | None = None,
        *,
        parent_harness_id: str | None = None,
        candidate_harness_id: str | None = None,
    ) -> GateDecision:
        parent = tuple(
            _validated_model_copy(TrialObservation, observation) for observation in parent_trials
        )
        candidate = tuple(
            _validated_model_copy(TrialObservation, observation) for observation in candidate_trials
        )
        evidence = _normalize_mechanism_evidence(mechanism_evidence)
        parent_harness_id = _validated_optional_harness_id(
            parent_harness_id, field_name="parent_harness_id"
        )
        candidate_harness_id = _validated_optional_harness_id(
            candidate_harness_id, field_name="candidate_harness_id"
        )
        config = self.config
        comparison = compare_trials(
            parent,
            candidate,
            config=config,
            parent_harness_id=parent_harness_id,
            candidate_harness_id=candidate_harness_id,
        )

        inferred_parent_id = (
            comparison.audit.parent_harness_ids[0]
            if len(comparison.audit.parent_harness_ids) == 1
            else parent_harness_id
        )
        inferred_candidate_id = (
            comparison.audit.candidate_harness_ids[0]
            if len(comparison.audit.candidate_harness_ids) == 1
            else candidate_harness_id
        )
        checks = (
            _integrity_check(comparison, config),
            _sample_size_check(comparison, config),
            _policy_check(parent, candidate),
            _mechanism_check(evidence, config, inferred_candidate_id),
            _primary_effect_check(comparison, config),
            _protected_slices_check(comparison, config),
            _task_regression_check(comparison, config),
            _resource_check(comparison, config),
        )
        failed = [check for check in checks if check.outcome is GateCheckOutcome.FAIL]
        uncertain = [check for check in checks if check.outcome is GateCheckOutcome.INCONCLUSIVE]
        if failed:
            decision = Decision.REJECT
        elif uncertain:
            decision = Decision.INCONCLUSIVE
        else:
            decision = Decision.PROMOTE

        nonpassing = [*failed, *uncertain]
        reasons = tuple(
            f"{check.name}: {reason}" for check in nonpassing for reason in check.reasons
        )
        if not reasons:
            reasons = ("all configured promotion checks passed",)
        result = GateDecision(
            decision=decision,
            gate_version=config.version,
            gate_config_sha256=self._config_sha256,
            parent_harness_id=inferred_parent_id,
            candidate_harness_id=inferred_candidate_id,
            checks=checks,
            reasons=reasons,
            comparison=comparison,
        )
        return _validated_model_copy(GateDecision, result)


def evaluate_gate(
    parent_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    candidate_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    config: GateConfig | None = None,
    mechanism_evidence: MechanismEvidence | Sequence[MechanismCheck] | None = None,
    *,
    parent_harness_id: str | None = None,
    candidate_harness_id: str | None = None,
) -> GateDecision:
    """Functional entry point for one promotion decision."""

    return PromotionGate(config).evaluate(
        parent_trials,
        candidate_trials,
        mechanism_evidence,
        parent_harness_id=parent_harness_id,
        candidate_harness_id=candidate_harness_id,
    )


__all__ = ["PromotionGate", "evaluate_gate"]
