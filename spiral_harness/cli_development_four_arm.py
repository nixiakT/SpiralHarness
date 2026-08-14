"""CLI surface for the public, non-reportable four-arm development study."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness.benchmark.bbh import BBHLogicalDeductionSevenAdapter
from spiral_harness.benchmark.datasets import (
    BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE,
    GSM8K_PROVENANCE,
)
from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.experiments.development_four_arm_contracts import (
    DEVELOPMENT_EVIDENCE_SCOPE,
    DevelopmentFourArmPlan,
    DevelopmentSplit,
)
from spiral_harness.experiments.development_four_arm_plan import (
    build_development_four_arm_plan,
    build_development_model_spec,
)
from spiral_harness.experiments.development_four_arm_runner import (
    DevelopmentFourArmRunResult,
    run_development_four_arm,
)
from spiral_harness.providers.openai_compatible import OpenAICompatibleChatBackend

_BASE_URL_ENV = "LITELLM_BASE_URL"
_API_KEY_ENV = "LITELLM_API_KEY"
_DRY_RUN_BACKEND_FINGERPRINT = "dry-run-openai-compatible-backend-unbound"


def _require_exploration_partition(value: str) -> ProtocolPartition:
    try:
        partition = ProtocolPartition(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown partition {value!r}; dev-four-arm accepts only exploration"
        ) from exc
    if partition is not ProtocolPartition.EXPLORATION:
        raise typer.BadParameter(
            "dev-four-arm refuses gate and sealed partitions; it is public development only"
        )
    return partition


def _require_absent_or_empty_output(output: Path) -> None:
    if output.is_symlink():
        raise typer.BadParameter("development output must not be a symbolic link")
    if not output.exists():
        return
    if not output.is_dir():
        raise typer.BadParameter("development output must be a directory")
    try:
        first_entry = next(output.iterdir(), None)
    except OSError as exc:
        raise typer.BadParameter(f"cannot inspect development output: {exc}") from exc
    if first_entry is not None:
        raise typer.BadParameter("development output directory must be absent or empty")


def _load_development_adapters(
    *,
    gsm8k_dir: Path,
    bbh_dir: Path,
) -> tuple[GSM8KBenchmarkAdapter, BBHLogicalDeductionSevenAdapter]:
    gsm8k_train = gsm8k_dir / GSM8K_PROVENANCE.artifact("train").filename
    gsm8k_test = gsm8k_dir / GSM8K_PROVENANCE.artifact("test").filename
    bbh_path = bbh_dir / BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE.artifact("development").filename
    missing = tuple(path for path in (gsm8k_train, gsm8k_test, bbh_path) if not path.is_file())
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise ValueError(
            "pinned local benchmark file(s) missing; this command never downloads implicitly: "
            + rendered
        )
    return (
        GSM8KBenchmarkAdapter(gsm8k_train, gsm8k_test),
        BBHLogicalDeductionSevenAdapter(bbh_path),
    )


def _build_plan(
    *,
    gsm8k_adapter: GSM8KBenchmarkAdapter,
    bbh_adapter: BBHLogicalDeductionSevenAdapter,
    backend_fingerprint: str,
    model: str,
    sample_seed: int,
    max_output_tokens: int,
    max_tokens_per_attempt: int,
    timeout_seconds: float,
) -> DevelopmentFourArmPlan:
    model_spec = build_development_model_spec(
        backend_fingerprint=backend_fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    return build_development_four_arm_plan(
        gsm8k_adapter=gsm8k_adapter,
        bbh_adapter=bbh_adapter,
        model_spec=model_spec,
        sample_seed=sample_seed,
        max_tokens_per_attempt=max_tokens_per_attempt,
    )


def _roster(plan: DevelopmentFourArmPlan, split: DevelopmentSplit) -> list[dict[str, object]]:
    selected = tuple(task for task in plan.tasks if task.split is split)
    return [
        {
            "position": position,
            "benchmark": task.benchmark.value,
            "task_id": task.task_id,
            "rollout_seed": task.seed,
        }
        for position, task in enumerate(selected, start=1)
    ]


def _holdout_call_order(plan: DevelopmentFourArmPlan) -> list[dict[str, object]]:
    holdout = tuple(task for task in plan.tasks if task.split is DevelopmentSplit.HOLDOUT)
    arms = tuple(arm.value for arm in plan.arms)
    return [
        {
            "task_position": position,
            "benchmark": task.benchmark.value,
            "task_id": task.task_id,
            "arm_order": list(arms[index % len(arms) :] + arms[: index % len(arms)]),
        }
        for index, (position, task) in enumerate(
            zip(range(1, len(holdout) + 1), holdout, strict=True)
        )
    ]


def _plan_payload(
    plan: DevelopmentFourArmPlan,
    *,
    dry_run: bool,
    output: Path,
) -> dict[str, object]:
    fit_roster = _roster(plan, DevelopmentSplit.FIT)
    holdout_roster = _roster(plan, DevelopmentSplit.HOLDOUT)
    unique_tasks = [task.task_id for task in plan.tasks]
    return {
        "schema_version": "1",
        "kind": "development_four_arm_cli_plan",
        "mode": "dry-run" if dry_run else "live",
        "dry_run": dry_run,
        "partition": ProtocolPartition.EXPLORATION.value,
        "model": plan.model_spec.model,
        "plan_fingerprint": plan.fingerprint,
        "plan_binding": (
            "dry-run-unbound-placeholder" if dry_run else "live-endpoint-bound-unattested"
        ),
        "unique_task_count": len(unique_tasks),
        "unique_tasks": unique_tasks,
        "task_roster": {"fit": fit_roster, "holdout": holdout_roster},
        "fit_task_order": [item["task_id"] for item in fit_roster],
        "holdout_task_order": [item["task_id"] for item in holdout_roster],
        "arms": [arm.value for arm in plan.arms],
        "holdout_call_order": _holdout_call_order(plan),
        "planned_model_calls": plan.max_model_calls,
        "call_topology": {
            "shared_static_parent_fit": 8,
            "score_proposer": 1,
            "full_proposer": 1,
            "score_candidate_fit": 8,
            "full_candidate_fit": 8,
            "holdout_tasks_x_four_arms": 32,
        },
        "model_max_output_tokens": plan.model_spec.inference.max_output_tokens,
        "reservation_token_ceiling_per_call": plan.max_tokens_per_attempt,
        "plan_token_ceiling": plan.max_total_tokens,
        "evidence_scope": plan.evidence_scope,
        "flags": {
            "nonreportable": True,
            "nonsealed": True,
            "nonattested": True,
            "feedback_view_only": True,
        },
        "reportable_benchmark_result": False,
        "sealed_evidence": False,
        "execution_attested": False,
        "provider_identity_attested": False,
        "feedback_view_only": plan.evidence_scope == DEVELOPMENT_EVIDENCE_SCOPE,
        "output": str(output.resolve()),
    }


def _live_summary(
    plan: DevelopmentFourArmPlan,
    result: DevelopmentFourArmRunResult,
    *,
    output: Path,
) -> dict[str, object]:
    payload = _plan_payload(plan, dry_run=False, output=output)
    payload.update(
        {
            "kind": result.payload["kind"],
            "executed_model_calls": result.payload["model_call_count"],
            "nominal_model_spec_shared_across_all_calls": result.payload[
                "nominal_model_spec_shared_across_all_calls"
            ],
            "provider_identity": result.payload["provider_identity"],
            "resource_usage": result.payload["resource_usage"],
            "selection_trace": result.payload["selection_trace"],
            "holdout_metrics": result.payload["holdout_metrics"],
            "result_ref": result.result_ref.model_dump(mode="json"),
            "closure_ref": result.closure_ref.model_dump(mode="json"),
            "disclaimer": result.payload["disclaimer"],
        }
    )
    return payload


def dev_four_arm(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="One nominal OpenAI-compatible route shared by all solver and proposer calls.",
        ),
    ] = "dashscope/qwen36-35b-a3b",
    gsm8k_dir: Annotated[
        Path,
        typer.Option("--gsm8k-dir", help="Directory containing the pinned GSM8K files."),
    ] = Path("data/benchmarks/gsm8k"),
    bbh_dir: Annotated[
        Path,
        typer.Option("--bbh-dir", help="Directory containing the pinned BBH file."),
    ] = Path("data/benchmarks/bbh"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Absent or empty development artifact directory."),
    ] = Path("runs/development-four-arm"),
    partition: Annotated[
        str,
        typer.Option(
            "--partition",
            help="Must be exploration; gate and sealed partitions are refused.",
        ),
    ] = ProtocolPartition.EXPLORATION.value,
    sample_seed: Annotated[
        int,
        typer.Option("--sample-seed", min=0, help="Deterministic public-task sample seed."),
    ] = 0,
    max_output_tokens: Annotated[
        int,
        typer.Option("--max-output-tokens", min=1, help="Frozen provider output ceiling."),
    ] = 1_024,
    max_tokens_per_attempt: Annotated[
        int,
        typer.Option(
            "--max-tokens-per-attempt",
            min=1,
            help="Attempt reservation ceiling covering provider input plus output tokens.",
        ),
    ] = 16_384,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout-seconds", min=0.001, help="Frozen timeout for every call."),
    ] = 120.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Plan from pinned local files without credentials, network, writes, or model calls."
            ),
        ),
    ] = False,
) -> None:
    """Plan or run the fixed 58-call, permanently non-reportable development exercise."""

    _require_exploration_partition(partition)
    _require_absent_or_empty_output(output)
    if max_tokens_per_attempt < max_output_tokens:
        raise typer.BadParameter(
            "max-tokens-per-attempt must cover the provider max-output-tokens ceiling"
        )
    try:
        gsm8k_adapter, bbh_adapter = _load_development_adapters(
            gsm8k_dir=gsm8k_dir,
            bbh_dir=bbh_dir,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if dry_run:
        try:
            plan = _build_plan(
                gsm8k_adapter=gsm8k_adapter,
                bbh_adapter=bbh_adapter,
                backend_fingerprint=_DRY_RUN_BACKEND_FINGERPRINT,
                model=model,
                sample_seed=sample_seed,
                max_output_tokens=max_output_tokens,
                max_tokens_per_attempt=max_tokens_per_attempt,
                timeout_seconds=timeout_seconds,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        payload = _plan_payload(plan, dry_run=True, output=output)
        payload["executed_model_calls"] = 0
        payload["network_calls"] = 0
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    base_url = os.environ.get(_BASE_URL_ENV)
    api_key = os.environ.get(_API_KEY_ENV)
    if not base_url:
        raise typer.BadParameter(f"missing environment variable {_BASE_URL_ENV}")
    if not api_key:
        raise typer.BadParameter(f"missing environment variable {_API_KEY_ENV}")
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    try:
        plan = _build_plan(
            gsm8k_adapter=gsm8k_adapter,
            bbh_adapter=bbh_adapter,
            backend_fingerprint=backend.fingerprint,
            model=model,
            sample_seed=sample_seed,
            max_output_tokens=max_output_tokens,
            max_tokens_per_attempt=max_tokens_per_attempt,
            timeout_seconds=timeout_seconds,
        )
        result = run_development_four_arm(
            output=output,
            backend=backend,
            gsm8k_adapter=gsm8k_adapter,
            bbh_adapter=bbh_adapter,
            plan=plan,
            max_tokens_per_attempt=max_tokens_per_attempt,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(_live_summary(plan, result, output=output), indent=2, sort_keys=True))


__all__ = ["dev_four_arm"]
