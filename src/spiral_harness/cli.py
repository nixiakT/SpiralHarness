"""Command-line entry point for SpiralHarness."""

import json
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness import __version__
from spiral_harness.benchmark import (
    DATASET_REGISTRY,
    GSM8K_PROVENANCE,
    GSM8KBenchmarkAdapter,
    materialize_dataset,
)
from spiral_harness.core import ProtocolPartition
from spiral_harness.evolution import run_controlled_demo

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


if __name__ == "__main__":  # pragma: no cover
    app()
