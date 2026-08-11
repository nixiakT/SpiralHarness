from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.verification.models import GateConfig, TrialObservation
from spiral_harness.verification.statistics import compare_trials, paired_bootstrap_ci


def observation(
    task_id: str,
    seed: int,
    harness_id: str,
    score: float,
    *,
    fingerprint: str | None = None,
    tags: tuple[str, ...] = (),
) -> TrialObservation:
    return TrialObservation(
        task_id=task_id,
        seed=seed,
        harness_id=harness_id,
        score=score,
        slice_tags=tags,
        tokens=100,
        latency_ms=20,
        tool_calls=2,
        execution_fingerprint=fingerprint or f"runtime:{task_id}:{seed}",
    )


def test_bootstrap_is_seeded_and_reproducible() -> None:
    first = paired_bootstrap_ci([0.2, -0.1, 0.4, 0.0], samples=1_000, seed=17)
    second = paired_bootstrap_ci([0.2, -0.1, 0.4, 0.0], samples=1_000, seed=17)

    assert first == second
    assert first.lower <= 0.125 <= first.upper


def test_multiple_seeds_are_averaged_within_task_before_inference() -> None:
    parent = [observation("many-seeds", seed, "parent", 0.0) for seed in (1, 2, 3)] + [
        observation("one-seed", 1, "parent", 0.0)
    ]
    candidate = [observation("many-seeds", seed, "candidate", 1.0) for seed in (1, 2, 3)] + [
        observation("one-seed", 1, "candidate", -1.0)
    ]

    result = compare_trials(
        parent,
        candidate,
        config=GateConfig(min_tasks=2, bootstrap_samples=1_000),
    )

    assert result.metrics is not None
    assert result.metrics.n_valid_pairs == 4
    assert result.metrics.n_tasks == 2
    # A rollout-level calculation would incorrectly report +0.5.
    assert result.metrics.mean_delta == pytest.approx(0.0)
    assert [task.delta for task in result.metrics.task_comparisons] == [1.0, -1.0]


def test_duplicate_task_seed_pair_is_audited_and_not_cherry_picked() -> None:
    parent = [
        observation("duplicate", 7, "parent", 0.1),
        observation("duplicate", 7, "parent", 0.9),
        observation("valid", 7, "parent", 0.2),
    ]
    candidate = [
        observation("duplicate", 7, "candidate", 1.0),
        observation("valid", 7, "candidate", 0.4),
    ]

    result = compare_trials(parent, candidate, bootstrap_samples=1_000)

    assert result.audit.duplicate_parent_pairs == ("duplicate::seed=7",)
    assert not result.audit.structurally_valid
    assert result.metrics is not None
    assert result.metrics.n_tasks == 1
    assert result.metrics.task_comparisons[0].task_id == "valid"


def test_execution_fingerprint_must_match_exactly() -> None:
    parent = [observation("task", 3, "parent", 0.4, fingerprint="runtime-A")]
    candidate = [observation("task", 3, "candidate", 0.8, fingerprint="runtime-B")]

    result = compare_trials(parent, candidate, bootstrap_samples=1_000)

    assert result.metrics is None
    assert result.audit.fingerprint_mismatches == ("task::seed=3",)
    assert "different execution fingerprints" in " ".join(result.audit.integrity_errors)


def test_task_weighted_resource_ratios_also_aggregate_seeds_first() -> None:
    parent = [
        observation("a", 1, "parent", 0.0),
        observation("a", 2, "parent", 0.0),
        observation("b", 1, "parent", 0.0),
    ]
    candidate = [
        observation("a", 1, "candidate", 0.1),
        observation("a", 2, "candidate", 0.1),
        observation("b", 1, "candidate", 0.1),
    ]
    # Frozen models make the intended per-observation resource edits explicit.
    parent = [
        item.model_copy(update={"tokens": 10 if item.task_id == "a" else 100}) for item in parent
    ]
    candidate = [
        item.model_copy(update={"tokens": 20 if item.task_id == "a" else 100}) for item in candidate
    ]

    result = compare_trials(parent, candidate, bootstrap_samples=1_000)

    assert result.metrics is not None
    assert result.metrics.parent_tokens_mean == 55
    assert result.metrics.candidate_tokens_mean == 60
    assert result.metrics.tokens_ratio == pytest.approx(60 / 55)


def test_bootstrap_and_config_reject_too_few_resamples() -> None:
    with pytest.raises(ValueError, match="at least 1000"):
        paired_bootstrap_ci([0.2, -0.1], samples=999)

    with pytest.raises(ValidationError):
        GateConfig(bootstrap_samples=999)


def test_preregistered_roster_excludes_and_audits_extra_tasks_and_seeds() -> None:
    parent = [
        observation("registered", 1, "parent", 0.0),
        observation("registered", 2, "parent", 0.0),
        observation("extra", 1, "parent", 0.0),
    ]
    candidate = [
        observation("registered", 1, "candidate", -0.1),
        observation("registered", 2, "candidate", 1.0),
        observation("extra", 1, "candidate", 1.0),
    ]

    result = compare_trials(
        parent,
        candidate,
        config=GateConfig(
            expected_task_ids=("registered",),
            expected_seeds=(1,),
            bootstrap_samples=1_000,
        ),
    )

    assert result.audit.unexpected_parent_pairs == (
        "extra::seed=1",
        "registered::seed=2",
    )
    assert result.audit.unexpected_candidate_pairs == result.audit.unexpected_parent_pairs
    assert not result.audit.structurally_valid
    assert result.metrics is not None
    assert result.metrics.n_valid_pairs == 1
    assert result.metrics.n_tasks == 1
    assert result.metrics.mean_delta == pytest.approx(-0.1)


def test_compare_trials_revalidates_unchecked_pydantic_copies() -> None:
    parent = [observation("task", 1, "parent", 0.0)]
    candidate = [
        observation("task", 1, "candidate", 0.1).model_copy(update={"tokens": float("nan")})
    ]

    with pytest.raises(ValidationError):
        compare_trials(parent, candidate)
