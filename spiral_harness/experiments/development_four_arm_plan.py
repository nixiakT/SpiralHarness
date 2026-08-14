"""Deterministic public-task planning for the four-arm development study."""

from __future__ import annotations

from spiral_harness.benchmark.bbh import BBHLogicalDeductionSevenAdapter
from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.development_four_arm_contracts import (
    DEVELOPMENT_MAX_MODEL_CALLS,
    BenchmarkKind,
    DevelopmentFourArmPlan,
    DevelopmentSplit,
    DevelopmentTask,
)


def build_development_model_spec(
    *,
    backend_fingerprint: str,
    model: str,
    max_output_tokens: int = 1_024,
    timeout_seconds: float = 120.0,
) -> FrozenModelSpec:
    """Build a credential-free nominal-route spec that makes non-attestation explicit."""

    return FrozenModelSpec(
        backend="openai-compatible-chat",
        backend_fingerprint=backend_fingerprint,
        model=model,
        revision="gateway-route-unattested-2026-08-14",
        tokenizer="provider-managed-unattested",
        tokenizer_revision="gateway-route-unattested-2026-08-14",
        runtime="spiral-harness-development-four-arm-py3.12@v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        ),
    )


def build_development_four_arm_plan(
    *,
    gsm8k_adapter: GSM8KBenchmarkAdapter,
    bbh_adapter: BBHLogicalDeductionSevenAdapter,
    model_spec: FrozenModelSpec,
    sample_seed: int,
    max_tokens_per_attempt: int = 16_384,
) -> DevelopmentFourArmPlan:
    """Freeze four fit and four holdout items per public exploration benchmark."""

    if type(sample_seed) is not int or sample_seed < 0:
        raise ValueError("sample_seed must be a non-negative integer")
    if type(max_tokens_per_attempt) is not int or max_tokens_per_attempt < 1:
        raise ValueError("max_tokens_per_attempt must be a positive integer")
    checked_spec = FrozenModelSpec.model_validate(model_spec, strict=True)
    tasks: list[DevelopmentTask] = []
    adapters = (
        (BenchmarkKind.GSM8K, gsm8k_adapter),
        (BenchmarkKind.BBH, bbh_adapter),
    )
    for benchmark, adapter in adapters:
        roster = adapter.task_roster(ProtocolPartition.EXPLORATION)
        if len(roster) < 8:
            raise ValueError(f"{benchmark.value} exploration roster has fewer than eight tasks")
        selected = tuple(
            sorted(
                roster,
                key=lambda task_id: canonical_sha256(
                    {
                        "schema": "spiral-harness/development-four-arm-sample/v1",
                        "benchmark": benchmark.value,
                        "sample_seed": sample_seed,
                        "task_id": task_id,
                    }
                ),
            )[:8]
        )
        for offset, task_id in enumerate(selected):
            task = adapter.load_task(task_id)
            split = DevelopmentSplit.FIT if offset < 4 else DevelopmentSplit.HOLDOUT
            rollout_seed = int(
                canonical_sha256(
                    {
                        "schema": "spiral-harness/development-four-arm-rollout-seed/v1",
                        "sample_seed": sample_seed,
                        "benchmark": benchmark.value,
                        "task_id": task_id,
                    }
                )[:8],
                16,
            )
            tasks.append(
                DevelopmentTask(
                    task_id=task_id,
                    benchmark=benchmark,
                    split=split,
                    question=task.question,
                    seed=rollout_seed,
                )
            )
    return DevelopmentFourArmPlan(
        tasks=tuple(tasks),
        model_spec=checked_spec,
        max_tokens_per_attempt=max_tokens_per_attempt,
        max_total_tokens=DEVELOPMENT_MAX_MODEL_CALLS * max_tokens_per_attempt,
    )


__all__ = ["build_development_four_arm_plan", "build_development_model_spec"]
