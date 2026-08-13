"""Non-reportable live smoke runner for pinned BBH development data."""

from __future__ import annotations

from pathlib import Path

from spiral_harness.benchmark.bbh import BBHLogicalDeductionSevenAdapter
from spiral_harness.benchmark.bbh_middleware import (
    BBH_OUTPUT_MIDDLEWARE_MEDIA_TYPE,
    BBHOutputContractMiddleware,
)
from spiral_harness.benchmark.gsm8k_smoke import (
    GSM8KSmokeResult,
    _publish_prompt_harness,
    build_live_gsm8k_spec,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import AttemptBudget, ResolvedHarness
from spiral_harness.execution.model import FixedModelRunner, ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

BBH_SMOKE_MEDIA_TYPE = "application/vnd.spiral-harness.bbh-smoke.v1+json"
DEFAULT_BBH_SMOKE_PROMPT = (
    "Solve the multiple-choice logic problem carefully. Explain briefly, then end with "
    "the selected option alone on its own line, for example: (A)"
)


def run_bbh_smoke(
    *,
    dataset_path: Path,
    output: Path,
    backend: ModelBackend,
    model: str,
    limit: int,
    seed: int = 0,
    max_output_tokens: int = 512,
    max_tokens_per_attempt: int = 1_024,
    timeout_seconds: float = 90.0,
    system_prompt: str = DEFAULT_BBH_SMOKE_PROMPT,
    verify_pinned: bool = True,
    output_middleware: BBHOutputContractMiddleware | None = None,
) -> GSM8KSmokeResult:
    """Run a deterministic sample from public BBH development data."""

    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_tokens_per_attempt < max_output_tokens:
        raise ValueError("max_tokens_per_attempt must be at least max_output_tokens")
    adapter = BBHLogicalDeductionSevenAdapter(dataset_path, verify_pinned=verify_pinned)
    roster = adapter.task_roster()
    if limit > len(roster):
        raise ValueError(f"BBH development roster has only {len(roster)} tasks")
    task_ids = tuple(
        sorted(
            roster,
            key=lambda task_id: canonical_sha256(
                {"schema": "spiral-harness/bbh-smoke-sample/v1", "seed": seed, "task_id": task_id}
            ),
        )[:limit]
    )
    store = ArtifactStore(Path(output) / "artifacts")
    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    ledger = AttemptLedger(
        store,
        ledger_id="bbh-live-smoke",
        budget=AttemptBudget(
            max_attempts=limit,
            max_total_tokens=max_tokens_per_attempt * limit,
            max_tokens_per_attempt=max_tokens_per_attempt,
        ),
    )
    runner = FixedModelRunner(spec=spec, backend=backend, attempt_ledger=ledger)
    middleware_ref = None
    if output_middleware is not None:
        checked_middleware = BBHOutputContractMiddleware.model_validate(
            output_middleware, strict=True
        )
        middleware_ref = store.put_json(
            checked_middleware,
            media_type=BBH_OUTPUT_MIDDLEWARE_MEDIA_TYPE,
        )
    harness_ref = _publish_prompt_harness(
        store,
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        system_prompt=system_prompt,
        middleware_ref=middleware_ref,
    )
    harness = ResolvedHarness.from_prompt(harness_ref=harness_ref, system_prompt=system_prompt)
    observations = []
    execution_refs = []
    for offset, task_id in enumerate(task_ids):
        task = adapter.load_task(task_id)
        execution = runner.execute(task, harness=harness, seed=seed + offset)
        grading_execution = execution
        middleware_evidence = None
        if output_middleware is not None and execution.output_text is not None:
            normalized = output_middleware.apply(execution.output_text)
            grading_execution = execution.model_copy(update={"output": normalized})
            middleware_evidence = {
                "middleware_ref": middleware_ref.model_dump(mode="json"),
                "raw_output_sha256": sha256_bytes(execution.output_text.encode("utf-8")),
                "normalized_output_sha256": sha256_bytes(normalized.encode("utf-8")),
                "changed": normalized != execution.output_text,
            }
        observations.append(
            adapter.grade(
                task,
                grading_execution,
                harness_id=harness_ref.sha256,
                seed=seed + offset,
                execution_fingerprint=execution.execution_fingerprint,
            )
        )
        execution_refs.append(
            {
                "task_id": task_id,
                "seed": seed + offset,
                "execution_ref": runner.last_execution_ref.model_dump(mode="json")
                if runner.last_execution_ref
                else None,
                "outcome_ref": runner.last_outcome_ref.model_dump(mode="json")
                if runner.last_outcome_ref
                else None,
                "middleware_evidence": middleware_evidence,
            }
        )
    completed = tuple(item for item in observations if item.score is not None)
    correct = sum(item.score == 1.0 for item in completed)
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": "non_reportable_bbh_development_smoke",
        "reportable": False,
        "benchmark": "bbh",
        "task_name": "logical_deduction_seven_objects",
        "partition": "public-development",
        "limit": limit,
        "sample_seed": seed,
        "sampling_method": "sha256-order-without-replacement-v1",
        "sampled_task_ids": list(task_ids),
        "model": model,
        "adapter_fingerprint": adapter.fingerprint,
        "harness_ref": harness_ref.model_dump(mode="json"),
        "output_middleware_ref": (
            None if middleware_ref is None else middleware_ref.model_dump(mode="json")
        ),
        "attempt_state": ledger.state().model_dump(mode="json"),
        "execution_refs": execution_refs,
        "observations": [item.model_dump(mode="json") for item in observations],
        "completed_count": len(completed),
        "correct_count": correct,
        "accuracy": None if not completed else correct / len(completed),
        "total_tokens": sum(item.tokens for item in observations),
        "disclaimer": (
            "Public-development smoke only; BBH has no sealed split and this is not "
            "a reportable result."
        ),
    }
    ref = store.put_json(payload, media_type=BBH_SMOKE_MEDIA_TYPE)
    return GSM8KSmokeResult(artifact_ref=ref, payload=payload, observations=tuple(observations))


__all__ = ["DEFAULT_BBH_SMOKE_PROMPT", "run_bbh_smoke"]
