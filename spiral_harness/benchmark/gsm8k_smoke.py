"""Small non-reportable GSM8K smoke runs against a live fixed-model backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE, ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    FrozenModelSpec,
    InferenceConfig,
    ResolvedHarness,
)
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.models import TrialObservation

GSM8K_SMOKE_MEDIA_TYPE = "application/vnd.spiral-harness.gsm8k-smoke.v1+json"
DEFAULT_GSM8K_SMOKE_PROMPT = (
    "You solve grade-school math problems carefully. "
    "Show concise reasoning, then end your answer with exactly one line of the form "
    "#### <number>."
)


@dataclass(frozen=True, slots=True)
class GSM8KSmokeResult:
    """One non-reportable live benchmark smoke artifact."""

    artifact_ref: ArtifactRef
    payload: dict[str, object]
    observations: tuple[TrialObservation, ...]


def build_live_gsm8k_spec(
    *,
    backend_fingerprint: str,
    model: str,
    max_output_tokens: int,
    timeout_seconds: float,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> FrozenModelSpec:
    """Create a credential-free frozen spec for one live smoke run."""

    return FrozenModelSpec(
        backend="openai-compatible-chat",
        backend_fingerprint=backend_fingerprint,
        model=model,
        revision="hosted-snapshot-2026-08-13",
        tokenizer="provider-reported",
        tokenizer_revision="hosted-snapshot-2026-08-13",
        runtime="spiral-harness-live-smoke-py3.12@2026-08-13",
        inference=InferenceConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        ),
    )


def run_gsm8k_smoke(
    *,
    train_path: Path,
    test_path: Path,
    output: Path,
    backend: ModelBackend,
    model: str,
    limit: int,
    partition: ProtocolPartition = ProtocolPartition.EXPLORATION,
    seed: int = 0,
    max_output_tokens: int = 512,
    max_tokens_per_attempt: int = 2_048,
    timeout_seconds: float = 60.0,
    system_prompt: str = DEFAULT_GSM8K_SMOKE_PROMPT,
    verify_pinned: bool = True,
) -> GSM8KSmokeResult:
    """Run a tiny live GSM8K slice through the score-free runner and trusted grader."""

    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if type(max_tokens_per_attempt) is not int or max_tokens_per_attempt < max_output_tokens:
        raise ValueError("max_tokens_per_attempt must be at least max_output_tokens")
    if not isinstance(partition, ProtocolPartition):
        raise TypeError("partition must be a ProtocolPartition")

    output = Path(output)
    artifact_store = ArtifactStore(output / "artifacts")
    adapter = GSM8KBenchmarkAdapter(train_path, test_path, verify_pinned=verify_pinned)
    task_ids = adapter.task_roster(partition)[:limit]
    if len(task_ids) != limit:
        raise ValueError(f"partition {partition.value!r} has only {len(task_ids)} tasks")

    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    ledger = AttemptLedger(
        artifact_store,
        ledger_id="gsm8k-live-smoke",
        budget=AttemptBudget(
            max_attempts=limit,
            max_total_tokens=max_tokens_per_attempt * limit,
            max_tokens_per_attempt=max_tokens_per_attempt,
        ),
    )
    runner = FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)
    harness_ref = _publish_prompt_harness(
        artifact_store,
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        system_prompt=system_prompt,
    )
    harness = ResolvedHarness.from_prompt(
        harness_ref=harness_ref,
        system_prompt=system_prompt,
    )

    observations: list[TrialObservation] = []
    execution_refs: list[dict[str, object]] = []
    for offset, task_id in enumerate(task_ids):
        task = adapter.load_task(task_id)
        execution = runner.execute(task, harness=harness, seed=seed + offset)
        observation = adapter.grade(
            task,
            execution,
            harness_id=harness_ref.sha256,
            seed=seed + offset,
            execution_fingerprint=execution.execution_fingerprint,
        )
        observations.append(observation)
        execution_refs.append(
            {
                "task_id": task_id,
                "seed": seed + offset,
                "execution_ref": (
                    None
                    if runner.last_execution_ref is None
                    else runner.last_execution_ref.model_dump(mode="json")
                ),
                "outcome_ref": (
                    None
                    if runner.last_outcome_ref is None
                    else runner.last_outcome_ref.model_dump(mode="json")
                ),
            }
        )

    completed = tuple(item for item in observations if item.score is not None)
    correct = sum(1 for item in completed if item.score == 1.0)
    total_tokens = sum(item.tokens for item in observations)
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": "non_reportable_gsm8k_smoke",
        "reportable": False,
        "benchmark": "gsm8k",
        "partition": partition.value,
        "limit": limit,
        "model": model,
        "spec_fingerprint": spec.fingerprint,
        "backend_fingerprint": backend.fingerprint,
        "adapter_fingerprint": adapter.fingerprint,
        "split_manifest_fingerprint": adapter.split_manifest.fingerprint,
        "harness_ref": harness_ref.model_dump(mode="json"),
        "attempt_state": ledger.state().model_dump(mode="json"),
        "execution_refs": execution_refs,
        "observations": [item.model_dump(mode="json") for item in observations],
        "completed_count": len(completed),
        "correct_count": correct,
        "accuracy": None if not completed else correct / len(completed),
        "total_tokens": total_tokens,
        "disclaimer": (
            "Exploration-only live smoke run for wiring/debugging; not a sealed or "
            "reportable benchmark result."
        ),
    }
    artifact_ref = artifact_store.put_json(payload, media_type=GSM8K_SMOKE_MEDIA_TYPE)
    return GSM8KSmokeResult(
        artifact_ref=artifact_ref,
        payload=payload,
        observations=tuple(observations),
    )


def _publish_prompt_harness(
    artifact_store: ArtifactStore,
    *,
    model_fingerprint: str,
    runtime_fingerprint: str,
    system_prompt: str,
) -> ArtifactRef:
    prompt_ref = artifact_store.put_bytes(
        system_prompt.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
    )
    manifest = {
        "schema_version": "2",
        "model_fingerprint": model_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "trusted_plane_version": "spiral-harness-live-smoke-v1",
        "components": [
            {
                "name": "system_prompt",
                "kind": "prompt",
                "artifact": prompt_ref.model_dump(mode="json"),
            }
        ],
        "budget": {"max_tokens": 0},
    }
    return ArtifactRef.model_validate(
        artifact_store.put_json(manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE),
        strict=True,
    )


def default_smoke_output_dir(model: str) -> Path:
    """Return a credential-free, model-stable local output directory."""

    return Path("runs") / "gsm8k-smoke" / canonical_sha256({"model": model})[:12]


__all__ = [
    "DEFAULT_GSM8K_SMOKE_PROMPT",
    "GSM8KSmokeResult",
    "build_live_gsm8k_spec",
    "default_smoke_output_dir",
    "run_gsm8k_smoke",
]
