"""Command-line entry point for SpiralHarness."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness import __version__
from spiral_harness.benchmark.bbh_smoke import run_bbh_smoke
from spiral_harness.benchmark.datasets import (
    BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE,
    DATASET_REGISTRY,
    GSM8K_PROVENANCE,
    materialize_dataset,
)
from spiral_harness.benchmark.gsm8k import GSM8KBenchmarkAdapter
from spiral_harness.benchmark.gsm8k_smoke import default_smoke_output_dir, run_gsm8k_smoke
from spiral_harness.benchmark.meta_harness_law import run_law_evaluation
from spiral_harness.benchmark.meta_harness_symptom import run_symptom_evaluation
from spiral_harness.benchmark.penguin_public import (
    run_public_self_evolution,
    verify_penguin_source,
)
from spiral_harness.benchmark.skillsbench_smoke import run_dialogue_parser_smoke
from spiral_harness.cli_development_four_arm import dev_four_arm
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.evolution.controlled_demo import run_controlled_demo
from spiral_harness.providers.openai_compatible import OpenAICompatibleChatBackend
from spiral_harness.providers.openai_native_function import (
    OpenAICompatibleNativeFunctionBackend,
)

app = typer.Typer(
    name="spiral",
    help="Evidence-gated, training-free evolution for agent harnesses.",
    no_args_is_help=True,
)
benchmark_app = typer.Typer(
    help="Materialize and verify trusted benchmark snapshots.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")
benchmark_app.command("dev-four-arm")(dev_four_arm)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Run SpiralHarness commands."""


@app.command("demo")
def controlled_demo(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory for the immutable synthetic run artifacts.",
        ),
    ] = Path("runs/controlled-demo"),
) -> None:
    """Run the synthetic controlled-fault verification demo."""

    result = run_controlled_demo(output)
    metrics = result.decision.metrics
    payload = {
        "candidate_mean": metrics.candidate_mean if metrics is not None else None,
        "decision": result.decision.decision.value,
        "disclaimer": result.disclaimer,
        "gate_config_sha256": result.decision.gate_config_sha256,
        "mean_delta": metrics.mean_delta if metrics is not None else None,
        "output": str(output.resolve()),
        "parent_mean": metrics.parent_mean if metrics is not None else None,
        "run_manifest_sha256": result.run_manifest_ref.sha256,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@benchmark_app.command("list-models")
def list_models(
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout-seconds", help="Timeout for the availability-only request."),
    ] = 15.0,
    base_url_env: Annotated[
        str,
        typer.Option("--base-url-env", help="Environment variable holding the LiteLLM base URL."),
    ] = "LITELLM_BASE_URL",
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Environment variable holding the LiteLLM API key."),
    ] = "LITELLM_API_KEY",
) -> None:
    """List advertised gateway models without running a score-bearing request."""

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url:
        raise typer.BadParameter(f"missing environment variable {base_url_env}")
    if not api_key:
        raise typer.BadParameter(f"missing environment variable {api_key_env}")
    backend = OpenAICompatibleNativeFunctionBackend.from_endpoint(
        base_url=base_url,
        api_key=api_key,
    )
    model_ids = backend.list_models(timeout_seconds=timeout_seconds)
    routes = {
        "dashscope": [model_id for model_id in model_ids if model_id.startswith("dashscope/")],
        "openai": [model_id for model_id in model_ids if model_id.startswith("openai/")],
        "direct": [
            model_id for model_id in model_ids if not model_id.startswith(("dashscope/", "openai/"))
        ],
    }
    payload = {
        "kind": "availability_only_model_catalog",
        "score_bearing_calls": 0,
        "backend_fingerprint": backend.fingerprint,
        "catalog_sha256": canonical_sha256(
            {
                "schema": "spiral-harness/model-catalog/v1",
                "backend_fingerprint": backend.fingerprint,
                "model_ids": model_ids,
            }
        ),
        "count": len(model_ids),
        "models": list(model_ids),
        "routes": routes,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@benchmark_app.command("fetch")
def fetch_benchmark(
    dataset: Annotated[
        str,
        typer.Argument(help="Registered benchmark name; currently only 'gsm8k'."),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Trusted local directory for downloaded dataset files.",
        ),
    ] = Path("data/benchmarks/gsm8k"),
) -> None:
    """Download a pinned dataset, verify every byte, and freeze its split identity."""

    if dataset.casefold() != "gsm8k":
        raise typer.BadParameter(f"unknown benchmark {dataset!r}; expected 'gsm8k'")

    paths = materialize_dataset(GSM8K_PROVENANCE, output)
    paths_by_name = {path.name: path for path in paths}
    adapter = GSM8KBenchmarkAdapter(
        paths_by_name[GSM8K_PROVENANCE.artifact("train").filename],
        paths_by_name[GSM8K_PROVENANCE.artifact("test").filename],
    )
    payload = {
        "adapter_fingerprint": adapter.fingerprint,
        "benchmark": "gsm8k",
        "dataset_id": GSM8K_PROVENANCE.dataset_id,
        "dataset_registry_fingerprint": DATASET_REGISTRY.fingerprint,
        "files": {
            artifact.role: str(paths_by_name[artifact.filename].resolve())
            for artifact in GSM8K_PROVENANCE.artifacts
        },
        "partitions": {
            partition.value: len(adapter.task_roster(partition)) for partition in ProtocolPartition
        },
        "split_manifest_fingerprint": adapter.split_manifest.fingerprint,
        "warning": (
            "The official test roster is sealed evaluation data; do not release its "
            "item-level content or feedback to search workers."
        ),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@benchmark_app.command("smoke-gsm8k")
def smoke_gsm8k(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="OpenAI-compatible model name, for example dashscope/qwen36-35b-a3b.",
        ),
    ] = "dashscope/qwen36-35b-a3b",
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help="Pinned GSM8K data directory; missing files are materialized and verified.",
        ),
    ] = Path("data/benchmarks/gsm8k"),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Directory for non-reportable smoke artifacts.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Number of exploration tasks to run."),
    ] = 3,
    partition: Annotated[
        str,
        typer.Option(
            "--partition",
            help="Protocol partition for smoke runs. Use exploration or gate; sealed is blocked.",
        ),
    ] = ProtocolPartition.EXPLORATION.value,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Initial deterministic runner seed."),
    ] = 0,
    max_output_tokens: Annotated[
        int,
        typer.Option("--max-output-tokens", help="Frozen provider output-token ceiling."),
    ] = 512,
    max_tokens_per_attempt: Annotated[
        int,
        typer.Option("--max-tokens-per-attempt", help="Trusted attempt reservation ceiling."),
    ] = 2_048,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout-seconds", help="Per-call provider timeout."),
    ] = 60.0,
    base_url_env: Annotated[
        str,
        typer.Option("--base-url-env", help="Environment variable holding the LiteLLM base URL."),
    ] = "LITELLM_BASE_URL",
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Environment variable holding the LiteLLM API key."),
    ] = "LITELLM_API_KEY",
) -> None:
    """Run a tiny non-reportable GSM8K smoke slice through a live model gateway."""

    try:
        checked_partition = ProtocolPartition(partition)
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown partition {partition!r}; expected exploration, gate, or sealed"
        ) from exc
    if checked_partition is ProtocolPartition.SEALED:
        raise typer.BadParameter("smoke-gsm8k refuses sealed partition runs")

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url:
        raise typer.BadParameter(f"missing environment variable {base_url_env}")
    if not api_key:
        raise typer.BadParameter(f"missing environment variable {api_key_env}")

    paths = materialize_dataset(GSM8K_PROVENANCE, dataset_dir)
    paths_by_name = {path.name: path for path in paths}
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    result = run_gsm8k_smoke(
        train_path=paths_by_name[GSM8K_PROVENANCE.artifact("train").filename],
        test_path=paths_by_name[GSM8K_PROVENANCE.artifact("test").filename],
        output=output if output is not None else default_smoke_output_dir(model),
        backend=backend,
        model=model,
        limit=limit,
        partition=checked_partition,
        seed=seed,
        max_output_tokens=max_output_tokens,
        max_tokens_per_attempt=max_tokens_per_attempt,
        timeout_seconds=timeout_seconds,
    )
    output_dir = output if output is not None else default_smoke_output_dir(model)
    summary = {
        "accuracy": result.payload["accuracy"],
        "artifact_ref": result.artifact_ref.model_dump(mode="json"),
        "completed_count": result.payload["completed_count"],
        "correct_count": result.payload["correct_count"],
        "disclaimer": result.payload["disclaimer"],
        "limit": result.payload["limit"],
        "model": result.payload["model"],
        "output": str(output_dir.resolve()),
        "partition": result.payload["partition"],
        "reportable": False,
        "total_tokens": result.payload["total_tokens"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("meta-harness-law")
def meta_harness_law(
    data_dir: Annotated[Path, typer.Option("--data-dir")],
    split: Annotated[str, typer.Option("--split")] = "val",
    model: Annotated[str, typer.Option("--model")] = "dashscope/qwen3-coder-flash",
    arm: Annotated[str, typer.Option("--arm")] = "pure",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("runs/meta-harness-law"),
    retrieval_k: Annotated[int, typer.Option("--retrieval-k")] = 8,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens")] = 128,
    base_url_env: Annotated[str, typer.Option("--base-url-env")] = "LITELLM_BASE_URL",
    api_key_env: Annotated[str, typer.Option("--api-key-env")] = "LITELLM_API_KEY",
) -> None:
    """Evaluate same-model pure or retrieval arms on pinned LawBench data."""

    base_url, api_key = os.environ.get(base_url_env), os.environ.get(api_key_env)
    if not base_url or not api_key:
        raise typer.BadParameter("missing LiteLLM endpoint or API key environment variable")
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    payload = run_law_evaluation(
        train_path=data_dir / "train.jsonl",
        evaluation_path=data_dir / f"{split}.jsonl",
        split=split,
        output=output,
        backend=backend,
        model=model,
        arm=arm,
        retrieval_k=retrieval_k,
        max_output_tokens=max_output_tokens,
    )
    typer.echo(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "benchmark",
                    "split",
                    "model",
                    "arm",
                    "retrieval_k",
                    "correct",
                    "total",
                    "accuracy",
                    "total_tokens",
                    "artifact_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


@benchmark_app.command("smoke-bbh")
def smoke_bbh(
    model: Annotated[str, typer.Option("--model")] = "dashscope/qwen-flash",
    dataset_dir: Annotated[Path, typer.Option("--dataset-dir")] = Path("data/benchmarks/bbh"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("runs/bbh-smoke/default"),
    limit: Annotated[int, typer.Option("--limit")] = 16,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens")] = 512,
    max_tokens_per_attempt: Annotated[int, typer.Option("--max-tokens-per-attempt")] = 1_024,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 90.0,
    base_url_env: Annotated[str, typer.Option("--base-url-env")] = "LITELLM_BASE_URL",
    api_key_env: Annotated[str, typer.Option("--api-key-env")] = "LITELLM_API_KEY",
) -> None:
    """Run a non-reportable public-development BBH logic sample."""

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url:
        raise typer.BadParameter(f"missing environment variable {base_url_env}")
    if not api_key:
        raise typer.BadParameter(f"missing environment variable {api_key_env}")
    paths = materialize_dataset(BBH_LOGICAL_DEDUCTION_SEVEN_PROVENANCE, dataset_dir)
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    result = run_bbh_smoke(
        dataset_path=paths[0],
        output=output,
        backend=backend,
        model=model,
        limit=limit,
        seed=seed,
        max_output_tokens=max_output_tokens,
        max_tokens_per_attempt=max_tokens_per_attempt,
        timeout_seconds=timeout_seconds,
    )
    summary = {
        key: result.payload[key]
        for key in (
            "accuracy",
            "completed_count",
            "correct_count",
            "disclaimer",
            "limit",
            "model",
            "total_tokens",
        )
    }
    summary["artifact_ref"] = result.artifact_ref.model_dump(mode="json")
    summary["output"] = str(output.resolve())
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("smoke-skillsbench")
def smoke_skillsbench(
    skillsbench_repo: Annotated[Path, typer.Option("--skillsbench-repo")],
    model: Annotated[str, typer.Option("--model")] = "dashscope/qwen3-coder-flash",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "runs/skillsbench/dialogue-parser"
    ),
    skill_file: Annotated[Path | None, typer.Option("--skill-file")] = None,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens")] = 8_192,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 180.0,
    base_url_env: Annotated[str, typer.Option("--base-url-env")] = "LITELLM_BASE_URL",
    api_key_env: Annotated[str, typer.Option("--api-key-env")] = "LITELLM_API_KEY",
) -> None:
    """Run the pinned dialogue-parser native-compatible smoke task."""

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url:
        raise typer.BadParameter(f"missing environment variable {base_url_env}")
    if not api_key:
        raise typer.BadParameter(f"missing environment variable {api_key_env}")
    skill_text = None if skill_file is None else skill_file.read_text(encoding="utf-8")
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    result = run_dialogue_parser_smoke(
        skillsbench_repo=skillsbench_repo,
        output=output,
        backend=backend,
        model=model,
        skill_text=skill_text,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    summary = {
        key: result.payload[key]
        for key in (
            "benchmark",
            "task",
            "model",
            "harness_variant",
            "passed",
            "input_tokens",
            "output_tokens",
            "protocol",
            "reportable_as_official",
            "disclaimer",
        )
    }
    summary["artifact_sha256"] = result.artifact_sha256
    summary["output"] = str(output.resolve())
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("meta-harness-symptom")
def meta_harness_symptom(
    data_dir: Annotated[Path, typer.Option("--data-dir")],
    split: Annotated[str, typer.Option("--split")] = "val",
    model: Annotated[str, typer.Option("--model")] = "dashscope/qwen-flash",
    arm: Annotated[str, typer.Option("--arm")] = "evolved",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("runs/meta-harness-symptom"),
    retrieval_k: Annotated[int, typer.Option("--retrieval-k")] = 8,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens")] = 128,
    base_url_env: Annotated[str, typer.Option("--base-url-env")] = "LITELLM_BASE_URL",
    api_key_env: Annotated[str, typer.Option("--api-key-env")] = "LITELLM_API_KEY",
) -> None:
    """Evaluate a retrieval harness on pinned Meta-Harness Symptom2Disease data."""

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url or not api_key:
        raise typer.BadParameter("missing LiteLLM endpoint or API key environment variable")
    if split not in {"val", "test"}:
        raise typer.BadParameter("split must be val or test")
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    payload = run_symptom_evaluation(
        train_path=data_dir / "train.jsonl",
        evaluation_path=data_dir / f"{split}.jsonl",
        split=split,
        output=output,
        backend=backend,
        model=model,
        arm=arm,
        retrieval_k=retrieval_k,
        max_output_tokens=max_output_tokens,
    )
    summary = {
        key: payload[key]
        for key in (
            "benchmark",
            "revision",
            "split",
            "model",
            "arm",
            "retrieval_k",
            "correct",
            "total",
            "accuracy",
            "reference_accuracy",
            "exceeds_reference",
            "total_tokens",
            "artifact_sha256",
        )
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("penguin-public")
def penguin_public(
    model: Annotated[str, typer.Option("--model")] = "dashscope/qwen3-coder-flash",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "runs/penguin-public/qwen3-coder-flash"
    ),
    penguin_source: Annotated[
        Path | None,
        typer.Option(
            "--penguin-source",
            help="Optional pinned self-evolve-recursive.ts path to verify before the run.",
        ),
    ] = None,
    runs_per_generation: Annotated[
        int,
        typer.Option(
            "--runs-per-generation",
            help=(
                "Runs in each generation; 5 is the canonical 17-call public schedule, while "
                "1-4 are reduced noncanonical exploratory variants."
            ),
        ),
    ] = 5,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens")] = 32_000,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 120.0,
    base_url_env: Annotated[str, typer.Option("--base-url-env")] = "LITELLM_BASE_URL",
    api_key_env: Annotated[str, typer.Option("--api-key-env")] = "LITELLM_API_KEY",
) -> None:
    """Run Penguin's public recursive demo via Spiral's text-state middleware."""

    base_url, api_key = os.environ.get(base_url_env), os.environ.get(api_key_env)
    if not base_url or not api_key:
        raise typer.BadParameter("missing LiteLLM endpoint or API key environment variable")
    if penguin_source is not None:
        try:
            verify_penguin_source(penguin_source)
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    backend = OpenAICompatibleChatBackend.from_endpoint(base_url=base_url, api_key=api_key)
    result = run_public_self_evolution(
        output=output,
        backend=backend,
        model=model,
        runs_per_generation=runs_per_generation,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    generations = [
        {
            key: generation[key]
            for key in (
                "label",
                "scores",
                "mean",
            )
        }
        for generation in result.payload["generations"]
    ]
    typer.echo(
        json.dumps(
            {
                "benchmark": result.payload["benchmark"],
                "model": model,
                "kind": result.payload["kind"],
                "protocol_sha256": result.payload["protocol_sha256"],
                "protocol_class": result.payload["protocol_class"],
                "reportable_as_canonical": result.payload["reportable_as_canonical"],
                "runs_per_generation": result.payload["runs_per_generation"],
                "call_schedule": result.payload["call_schedule"],
                "worker_harness_ref": result.payload["worker_harness_ref"],
                "reflection_harness_ref": result.payload["reflection_harness_ref"],
                "generations": generations,
                "rounds": result.payload["rounds"],
                "total_tokens": result.payload["total_tokens"],
                "artifact_sha256": result.artifact_ref.sha256,
                "output": str(output.resolve()),
                "disclaimer": result.payload["disclaimer"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
