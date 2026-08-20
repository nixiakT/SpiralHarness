"""CLI for the fresh BFCL bare-model versus multi-agent comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.bfcl_v4_multi_agent_comparison import (
    comparison_protocol_summary,
    run_bfcl_v4_multi_agent_comparison,
)
from spiral_harness.experiments.bfcl_v4_multi_agent_verification import (
    verify_bfcl_v4_multi_agent_comparison,
)
from spiral_harness.providers.openai_compatible import OpenAICompatibleChatBackend
from spiral_harness.storage.artifact_store import ArtifactStore


class BfclMultiAgentComparisonCliError(RuntimeError):
    """Credential-free failure exposed at the command boundary."""


def _model_spec(
    *,
    backend: OpenAICompatibleChatBackend,
    model: str,
    observation_revision: str,
) -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="openai-compatible-litellm",
        backend_fingerprint=backend.fingerprint,
        model=model,
        revision=observation_revision,
        tokenizer=f"provider-managed/{model}",
        tokenizer_revision=observation_revision,
        runtime="spiral-openai-compatible-chat-v1",
        inference=InferenceConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2_048,
            timeout_seconds=180.0,
        ),
    )


def bfcl_multi_agent_compare(
    checkout: Annotated[
        Path,
        typer.Option("--checkout", help="Pinned local Gorilla/BFCL checkout."),
    ] = Path("data/benchmarks/bfcl-v4/gorilla"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Absent or empty local result directory."),
    ] = Path("runs/bfcl-v4-multi-agent-qwen36-20260820"),
    model: Annotated[
        str,
        typer.Option("--model", help="Exact currently advertised LiteLLM model route."),
    ] = "qwen36-35b-a3b",
    observation_revision: Annotated[
        str,
        typer.Option(
            "--observation-revision",
            help="Immutable experiment observation label; not a served-weights attestation.",
        ),
    ] = "gateway-observation-2026-08-20",
    root_seed: Annotated[
        int,
        typer.Option("--root-seed", help="Root for the frozen paired three-call seeds."),
    ] = 2026082001,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate roster and print the 54-call plan only."),
    ] = False,
    allow_paid_model: Annotated[
        bool,
        typer.Option(
            "--allow-paid-model",
            help="Explicitly permit OpenAI/Anthropic routes; off by default.",
        ),
    ] = False,
) -> None:
    """Compare one bare model with a budget-matched three-agent Harness."""

    try:
        protocol = comparison_protocol_summary(checkout)
        if dry_run:
            typer.echo(json.dumps(protocol, indent=2, sort_keys=True))
            return
        if model.startswith(("openai/", "anthropic/")) and not allow_paid_model:
            raise BfclMultiAgentComparisonCliError(
                "paid model route requires the explicit --allow-paid-model flag"
            )
        base_url = os.environ.get("LITELLM_BASE_URL")
        api_key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not api_key:
            raise BfclMultiAgentComparisonCliError("missing LITELLM_BASE_URL or LITELLM_API_KEY")
        backend = OpenAICompatibleChatBackend.from_endpoint(
            base_url=base_url,
            api_key=api_key,
            enable_thinking=False,
        )
        if model not in backend.list_models(timeout_seconds=20.0):
            raise BfclMultiAgentComparisonCliError("requested model route is not advertised")
        spec = _model_spec(
            backend=backend,
            model=model,
            observation_revision=observation_revision,
        )
        result, result_ref = run_bfcl_v4_multi_agent_comparison(
            checkout=checkout,
            output=output,
            spec=spec,
            backend=backend,
            root_seed=root_seed,
        )
        verified = verify_bfcl_v4_multi_agent_comparison(
            store=ArtifactStore(output / "artifacts"),
            result_ref=result_ref,
        )
        if verified != result:
            raise BfclMultiAgentComparisonCliError("offline result verification changed the result")
        typer.echo(
            json.dumps(
                {
                    "baseline_budget_matched": f"{result.baseline_budget_matched_correct}/9",
                    "baseline_single": f"{result.baseline_single_correct}/9",
                    "consumed_model_calls": result.consumed_model_calls,
                    "harness": f"{result.harness_correct}/9",
                    "output": str(output.resolve()),
                    "reportable_result": False,
                    "result_ref": result_ref.model_dump(mode="json"),
                    "total_reported_tokens": result.total_reported_tokens,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except BfclMultiAgentComparisonCliError:
        raise
    except Exception:
        raise BfclMultiAgentComparisonCliError(
            "BFCL multi-agent comparison failed; inspect local credential-free artifacts"
        ) from None


__all__ = ["BfclMultiAgentComparisonCliError", "bfcl_multi_agent_compare"]
