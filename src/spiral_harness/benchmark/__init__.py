"""Benchmark adapters, including explicitly synthetic infrastructure fixtures."""

from spiral_harness.benchmark.controlled_fixture import (
    CANDIDATE_PROMPT,
    CONTROLLED_TASKS,
    EXPLORATION_SEED,
    EXPLORATION_TASKS,
    FIXTURE_KIND,
    FIXTURE_SEED,
    GATE_TASKS,
    NORMALIZATION_SLICE,
    PROTECTED_CANONICAL_SLICE,
    SEED_PROMPT,
    BenchmarkTask,
    DeterministicExecution,
    DeterministicExecutor,
)

__all__ = [
    "CANDIDATE_PROMPT",
    "CONTROLLED_TASKS",
    "EXPLORATION_SEED",
    "EXPLORATION_TASKS",
    "FIXTURE_KIND",
    "FIXTURE_SEED",
    "GATE_TASKS",
    "NORMALIZATION_SLICE",
    "PROTECTED_CANONICAL_SLICE",
    "SEED_PROMPT",
    "BenchmarkTask",
    "DeterministicExecution",
    "DeterministicExecutor",
]
