"""Command-line entry point for SpiralHarness."""

import json
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness import __version__
from spiral_harness.evolution import run_controlled_demo

app = typer.Typer(
    name="spiral",
    help="Evidence-gated, training-free evolution for agent harnesses.",
    no_args_is_help=True,
)


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


if __name__ == "__main__":  # pragma: no cover
    app()
