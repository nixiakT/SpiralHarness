"""Matched task-level statistics for candidate verification.

Rollouts are paired by ``(task_id, seed)`` for comparability, then averaged
inside each task.  Only those task averages enter inference.  This is the key
guard against presenting several stochastic seeds as several independent
benchmark tasks.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from spiral_harness.verification.models import (
    ComparisonMetrics,
    ComparisonResult,
    ConfidenceInterval,
    GateConfig,
    PairingAudit,
    SliceMetrics,
    TaskComparison,
    TrialObservation,
    TrialStatus,
)

PairKey = tuple[str, int]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolated quantile with deterministic endpoint behavior."""

    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def paired_bootstrap_ci(
    deltas: Sequence[float] | Iterable[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a seeded percentile bootstrap CI for a paired mean.

    ``deltas`` must already represent independent statistical units.  In the
    verifier these are task-level deltas, never raw rollout/seed deltas.
    """

    values = tuple(float(value) for value in deltas)
    if not values:
        raise ValueError("paired bootstrap requires at least one delta")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if samples < 1_000:
        raise ValueError("samples must be at least 1000")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("deltas must be finite")

    if len(values) == 1 or all(value == values[0] for value in values[1:]):
        lower = upper = values[0]
    else:
        rng = random.Random(seed)
        size = len(values)
        distribution = [
            math.fsum(values[rng.randrange(size)] for _ in range(size)) / size
            for _ in range(samples)
        ]
        distribution.sort()
        alpha = (1.0 - confidence_level) / 2.0
        lower = _quantile(distribution, alpha)
        upper = _quantile(distribution, 1.0 - alpha)

    return ConfidenceInterval(
        confidence_level=confidence_level,
        lower=lower,
        upper=upper,
    )


def _pair_label(key: PairKey) -> str:
    return f"{key[0]}::seed={key[1]}"


def _unique_observations(
    observations: Sequence[TrialObservation],
) -> tuple[dict[PairKey, TrialObservation], tuple[str, ...]]:
    grouped: dict[PairKey, list[TrialObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.task_id, observation.seed)].append(observation)
    duplicates = tuple(
        _pair_label(key) for key, group in sorted(grouped.items()) if len(group) != 1
    )
    unique = {key: group[0] for key, group in grouped.items() if len(group) == 1}
    return unique, duplicates


def _status_counts(observations: Sequence[TrialObservation]) -> dict[str, int]:
    counts = Counter(observation.status.value for observation in observations)
    return dict(sorted(counts.items()))


def _ratio(candidate: float, parent: float) -> float | None:
    if parent == 0:
        return 0.0 if candidate == 0 else None
    ratio = candidate / parent
    return ratio if math.isfinite(ratio) else None


def _slice_seed(base_seed: int, slice_name: str) -> int:
    # Unlike hash(), this is stable across processes and PYTHONHASHSEED values.
    return base_seed + sum((index + 1) * byte for index, byte in enumerate(slice_name.encode()))


def _build_metrics(
    tasks: Sequence[TaskComparison],
    *,
    n_valid_pairs: int,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ComparisonMetrics:
    parent_scores = [task.parent_score for task in tasks]
    candidate_scores = [task.candidate_score for task in tasks]
    deltas = [task.delta for task in tasks]

    slices: dict[str, SliceMetrics] = {}
    slice_names = sorted({tag for task in tasks for tag in task.slice_tags})
    for slice_name in slice_names:
        slice_tasks = [task for task in tasks if slice_name in task.slice_tags]
        slice_deltas = [task.delta for task in slice_tasks]
        slices[slice_name] = SliceMetrics(
            slice_name=slice_name,
            n_tasks=len(slice_tasks),
            parent_mean=_mean([task.parent_score for task in slice_tasks]),
            candidate_mean=_mean([task.candidate_score for task in slice_tasks]),
            mean_delta=_mean(slice_deltas),
            confidence_interval=paired_bootstrap_ci(
                slice_deltas,
                confidence_level=confidence_level,
                samples=bootstrap_samples,
                seed=_slice_seed(bootstrap_seed, slice_name),
            ),
        )

    parent_tokens = _mean([task.parent_tokens for task in tasks])
    candidate_tokens = _mean([task.candidate_tokens for task in tasks])
    parent_latency = _mean([task.parent_latency_ms for task in tasks])
    candidate_latency = _mean([task.candidate_latency_ms for task in tasks])
    parent_tool_calls = _mean([task.parent_tool_calls for task in tasks])
    candidate_tool_calls = _mean([task.candidate_tool_calls for task in tasks])
    tie_tolerance = 1e-12

    return ComparisonMetrics(
        n_valid_pairs=n_valid_pairs,
        n_tasks=len(tasks),
        parent_mean=_mean(parent_scores),
        candidate_mean=_mean(candidate_scores),
        mean_delta=_mean(deltas),
        confidence_interval=paired_bootstrap_ci(
            deltas,
            confidence_level=confidence_level,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        wins=sum(delta > tie_tolerance for delta in deltas),
        ties=sum(abs(delta) <= tie_tolerance for delta in deltas),
        losses=sum(delta < -tie_tolerance for delta in deltas),
        regression_rate=sum(delta < -tie_tolerance for delta in deltas) / len(deltas),
        worst_task_delta=min(deltas),
        parent_tokens_mean=parent_tokens,
        candidate_tokens_mean=candidate_tokens,
        tokens_ratio=_ratio(candidate_tokens, parent_tokens),
        parent_latency_ms_mean=parent_latency,
        candidate_latency_ms_mean=candidate_latency,
        latency_ratio=_ratio(candidate_latency, parent_latency),
        parent_tool_calls_mean=parent_tool_calls,
        candidate_tool_calls_mean=candidate_tool_calls,
        tool_calls_ratio=_ratio(candidate_tool_calls, parent_tool_calls),
        task_comparisons=tuple(tasks),
        slices=slices,
    )


def compare_trials(
    parent_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    candidate_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    *,
    config: GateConfig | None = None,
    confidence_level: float | None = None,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
    expected_task_ids: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    parent_harness_id: str | None = None,
    candidate_harness_id: str | None = None,
) -> ComparisonResult:
    """Validate, pair, and summarize parent/candidate observations.

    Structural mismatches remain in the returned audit rather than being
    silently dropped.  The promotion gate decides their severity while callers
    can still inspect any unaffected task-level statistics.
    """

    parent = tuple(
        TrialObservation.model_validate(
            observation.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        for observation in parent_trials
    )
    candidate = tuple(
        TrialObservation.model_validate(
            observation.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        for observation in candidate_trials
    )
    supplied_config = config or GateConfig()
    effective_config = GateConfig.model_validate(
        supplied_config.model_dump(mode="python", round_trip=True, warnings="none"),
        strict=True,
    )
    confidence_level = (
        effective_config.confidence_level if confidence_level is None else confidence_level
    )
    bootstrap_samples = (
        effective_config.bootstrap_samples if bootstrap_samples is None else bootstrap_samples
    )
    bootstrap_seed = effective_config.bootstrap_seed if bootstrap_seed is None else bootstrap_seed
    expected_task_ids = tuple(
        effective_config.expected_task_ids if expected_task_ids is None else expected_task_ids
    )
    expected_seeds = tuple(
        effective_config.expected_seeds if expected_seeds is None else expected_seeds
    )

    parent_by_key, duplicate_parent = _unique_observations(parent)
    candidate_by_key, duplicate_candidate = _unique_observations(candidate)
    observed_parent_keys = {(observation.task_id, observation.seed) for observation in parent}
    observed_candidate_keys = {(observation.task_id, observation.seed) for observation in candidate}

    expected_keys: set[PairKey] | None = None
    unexpected_parent_keys: set[PairKey] = set()
    unexpected_candidate_keys: set[PairKey] = set()
    if expected_seeds:
        expected_keys = {
            (task_id, seed) for task_id in expected_task_ids for seed in expected_seeds
        }
        unexpected_parent_keys = observed_parent_keys - expected_keys
        unexpected_candidate_keys = observed_candidate_keys - expected_keys
    elif expected_task_ids:
        expected_task_set = set(expected_task_ids)
        unexpected_parent_keys = {
            key for key in observed_parent_keys if key[0] not in expected_task_set
        }
        unexpected_candidate_keys = {
            key for key in observed_candidate_keys if key[0] not in expected_task_set
        }

    # The preregistered roster is exact. Unexpected observations remain visible
    # in the audit, but can never alter score, confidence, slice, or resource metrics.
    parent_by_key = {
        key: observation
        for key, observation in parent_by_key.items()
        if key not in unexpected_parent_keys
    }
    candidate_by_key = {
        key: observation
        for key, observation in candidate_by_key.items()
        if key not in unexpected_candidate_keys
    }
    parent_keys = set(parent_by_key)
    candidate_keys = set(candidate_by_key)
    common_keys = sorted(parent_keys & candidate_keys)

    parent_ids = tuple(sorted({observation.harness_id for observation in parent}))
    candidate_ids = tuple(sorted({observation.harness_id for observation in candidate}))
    integrity_errors: list[str] = []
    if duplicate_parent:
        integrity_errors.append("parent observations contain duplicate task/seed pairs")
    if duplicate_candidate:
        integrity_errors.append("candidate observations contain duplicate task/seed pairs")
    if unexpected_parent_keys:
        integrity_errors.append("parent observations contain unregistered task/seed pairs")
    if unexpected_candidate_keys:
        integrity_errors.append("candidate observations contain unregistered task/seed pairs")
    if len(parent_ids) > 1:
        integrity_errors.append("parent observations contain multiple harness IDs")
    if len(candidate_ids) > 1:
        integrity_errors.append("candidate observations contain multiple harness IDs")
    if len(parent_ids) == 1 and len(candidate_ids) == 1 and parent_ids[0] == candidate_ids[0]:
        integrity_errors.append("parent and candidate observations use the same harness ID")
    if (
        parent_harness_id is not None
        and candidate_harness_id is not None
        and parent_harness_id == candidate_harness_id
    ):
        integrity_errors.append("requested parent and candidate harness IDs are identical")
    if parent_harness_id is not None and any(
        observation.harness_id != parent_harness_id for observation in parent
    ):
        integrity_errors.append("parent observation harness ID does not match requested parent")
    if candidate_harness_id is not None and any(
        observation.harness_id != candidate_harness_id for observation in candidate
    ):
        integrity_errors.append(
            "candidate observation harness ID does not match requested candidate"
        )

    fingerprint_mismatches: list[str] = []
    slice_tag_mismatches: list[str] = []
    incomplete_pairs: list[str] = []
    valid_by_task: dict[str, list[tuple[TrialObservation, TrialObservation]]] = defaultdict(list)

    for key in common_keys:
        parent_observation = parent_by_key[key]
        candidate_observation = candidate_by_key[key]
        label = _pair_label(key)
        if parent_observation.execution_fingerprint != candidate_observation.execution_fingerprint:
            fingerprint_mismatches.append(label)
            continue
        if parent_observation.slice_tags != candidate_observation.slice_tags:
            slice_tag_mismatches.append(label)
            continue
        if (
            parent_observation.status is not TrialStatus.COMPLETED
            or candidate_observation.status is not TrialStatus.COMPLETED
            or parent_observation.score is None
            or candidate_observation.score is None
        ):
            incomplete_pairs.append(
                f"{label} (parent={parent_observation.status.value}, "
                f"candidate={candidate_observation.status.value}, "
                f"scores={parent_observation.score!r}/{candidate_observation.score!r})"
            )
            continue
        valid_by_task[key[0]].append((parent_observation, candidate_observation))

    if fingerprint_mismatches:
        integrity_errors.append("paired observations have different execution fingerprints")
    if slice_tag_mismatches:
        integrity_errors.append("paired observations have different protected-slice tags")

    task_comparisons: list[TaskComparison] = []
    valid_pair_count = 0
    for task_id, seed_pairs in sorted(valid_by_task.items()):
        seed_pairs.sort(key=lambda pair: pair[0].seed)
        tag_sets = {pair[0].slice_tags for pair in seed_pairs}
        if len(tag_sets) != 1:
            slice_tag_mismatches.append(f"{task_id}::tags-vary-across-seeds")
            integrity_errors.append(f"task {task_id!r} has slice tags that vary across seeds")
            continue
        parents = [pair[0] for pair in seed_pairs]
        candidates = [pair[1] for pair in seed_pairs]
        parent_score = _mean(
            [observation.score for observation in parents if observation.score is not None]
        )
        candidate_score = _mean(
            [observation.score for observation in candidates if observation.score is not None]
        )
        valid_pair_count += len(seed_pairs)
        task_comparisons.append(
            TaskComparison(
                task_id=task_id,
                seeds=tuple(observation.seed for observation in parents),
                slice_tags=parents[0].slice_tags,
                parent_score=parent_score,
                candidate_score=candidate_score,
                delta=candidate_score - parent_score,
                parent_tokens=_mean([float(observation.tokens) for observation in parents]),
                candidate_tokens=_mean([float(observation.tokens) for observation in candidates]),
                parent_latency_ms=_mean([observation.latency_ms for observation in parents]),
                candidate_latency_ms=_mean([observation.latency_ms for observation in candidates]),
                parent_tool_calls=_mean([float(observation.tool_calls) for observation in parents]),
                candidate_tool_calls=_mean(
                    [float(observation.tool_calls) for observation in candidates]
                ),
            )
        )

    expected_missing_pairs: tuple[str, ...] = ()
    expected_missing_tasks: tuple[str, ...] = ()
    if expected_keys is not None:
        expected_missing_pairs = tuple(
            _pair_label(key)
            for key in sorted(expected_keys)
            if key not in parent_keys or key not in candidate_keys
        )
    elif expected_task_ids:
        parent_task_ids = {key[0] for key in parent_keys}
        candidate_task_ids = {key[0] for key in candidate_keys}
        expected_missing_tasks = tuple(
            task_id
            for task_id in expected_task_ids
            if task_id not in parent_task_ids or task_id not in candidate_task_ids
        )

    audit = PairingAudit(
        parent_harness_ids=parent_ids,
        candidate_harness_ids=candidate_ids,
        parent_status_counts=_status_counts(parent),
        candidate_status_counts=_status_counts(candidate),
        duplicate_parent_pairs=duplicate_parent,
        duplicate_candidate_pairs=duplicate_candidate,
        missing_parent_pairs=tuple(
            _pair_label(key) for key in sorted(candidate_keys - parent_keys)
        ),
        missing_candidate_pairs=tuple(
            _pair_label(key) for key in sorted(parent_keys - candidate_keys)
        ),
        unexpected_parent_pairs=tuple(_pair_label(key) for key in sorted(unexpected_parent_keys)),
        unexpected_candidate_pairs=tuple(
            _pair_label(key) for key in sorted(unexpected_candidate_keys)
        ),
        incomplete_pairs=tuple(incomplete_pairs),
        fingerprint_mismatches=tuple(fingerprint_mismatches),
        slice_tag_mismatches=tuple(slice_tag_mismatches),
        expected_missing_pairs=expected_missing_pairs,
        expected_missing_tasks=expected_missing_tasks,
        integrity_errors=tuple(dict.fromkeys(integrity_errors)),
    )
    metrics = None
    if task_comparisons:
        metrics = _build_metrics(
            task_comparisons,
            n_valid_pairs=valid_pair_count,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    return ComparisonResult(audit=audit, metrics=metrics)


def compute_paired_statistics(
    parent_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    candidate_trials: Sequence[TrialObservation] | Iterable[TrialObservation],
    **kwargs: object,
) -> ComparisonResult:
    """Descriptive alias for :func:`compare_trials`."""

    return compare_trials(parent_trials, candidate_trials, **kwargs)


paired_comparison = compare_trials


__all__ = [
    "compare_trials",
    "compute_paired_statistics",
    "paired_bootstrap_ci",
    "paired_comparison",
]
