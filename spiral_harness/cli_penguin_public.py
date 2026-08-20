"""CLI boundary for the Penguin public recursive-development runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness.benchmark.penguin_public import (
    run_public_self_evolution,
    verify_penguin_source,
)
from spiral_harness.providers.openai_compatible import OpenAICompatibleChatBackend


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
        {key: generation[key] for key in ("label", "scores", "mean")}
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


__all__ = ["penguin_public"]
