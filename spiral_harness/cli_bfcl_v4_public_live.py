"""Registered, fail-closed CLI boundary for the BFCL V4 public live campaign.

Exactly one mode must be explicit: dry-run is local and write-free, smoke runs
one ungraded FIT request, and full runs 300 calls for public-development-only evidence.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from spiral_harness.benchmark.bfcl_v4_fixture_bridge import (
    _checkout_and_git,
    _git_blob,
)
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import strict_json
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    adapt_bfcl_v4_public_pilot_task,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID,
    BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
    BfclV4PublicPilotCampaign,
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
    BfclV4PublicPilotTask,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
)
from spiral_harness.cli_bfcl_v4_public_summary import (
    build_bfcl_v4_public_terminal_summary,
)
from spiral_harness.cli_exception_boundary import discard_exception_graph
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor import (
    run_bfcl_v4_public_campaign,
    verify_bfcl_v4_public_campaign_execution,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor_contracts import (
    BfclV4CampaignExecutionStatus,
    BfclV4PublicCampaignExecutionRecord,
    BfclV4PublicCampaignExecutionVerification,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_SEED_SYSTEM_PROMPT,
)
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET,
    BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
    BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,
    BFCL_V4_PUBLIC_LIVE_INFERENCE,
    BfclV4PublicLiveExecutionConfig,
    freeze_bfcl_v4_public_live_execution_config,
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_native import (
    BfclV4PublicNativePromptKind,
    materialize_bfcl_v4_public_native_request,
)
from spiral_harness.providers.openai_native_contracts import NativeFunctionCallResponse
from spiral_harness.providers.openai_native_function import (
    OpenAICompatibleNativeFunctionBackend,
)
from spiral_harness.storage.artifact_store import ArtifactStore

LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_API_KEY_ENV = "LITELLM_API_KEY"
BFCL_V4_PUBLIC_LIVE_LEDGER_IDS = tuple(
    f"{BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID}/{replicate_id}"
    for replicate_id in BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS
)


class BfclV4PublicLiveCliError(RuntimeError):
    """A sanitized live-boundary validation or execution failure."""


@dataclass(frozen=True, slots=True)
class _LiveActivation:
    backend: OpenAICompatibleNativeFunctionBackend = field(repr=False)
    config: BfclV4PublicLiveExecutionConfig
    advertised_model_count: int


def _physical_cwd() -> Path:
    try:
        return Path.cwd().resolve(strict=True)
    except OSError:
        raise BfclV4PublicLiveCliError("current working directory cannot be resolved") from None


def _absolute_without_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        candidate = Path(os.path.normpath(expanded))
    else:
        candidate = Path(os.path.normpath(_physical_cwd() / expanded))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            raise BfclV4PublicLiveCliError("output path must not traverse symbolic links")
        if not current.exists():
            break
    return candidate


def _safe_output_path(output: Path, *, checkout: Path) -> Path:
    candidate = _absolute_without_symlinks(output)
    if candidate.exists():
        if not candidate.is_dir():
            raise BfclV4PublicLiveCliError("output must be a directory")
        try:
            occupied = next(candidate.iterdir(), None) is not None
        except OSError:
            raise BfclV4PublicLiveCliError("output directory cannot be inspected") from None
        if occupied:
            raise BfclV4PublicLiveCliError("output directory must be absent or empty")
    nearest = candidate
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    if not nearest.is_dir():
        raise BfclV4PublicLiveCliError("output parent must be a directory")
    if candidate == checkout or checkout in candidate.parents or candidate in checkout.parents:
        raise BfclV4PublicLiveCliError("output and pinned checkout must not overlap")
    return candidate


def _validated_checkout(checkout: Path) -> tuple[Path, Path]:
    """Verify only the pinned Git identity; do not inspect any benchmark answer."""

    try:
        return _checkout_and_git(checkout)
    except Exception:
        raise BfclV4PublicLiveCliError("pinned BFCL checkout validation failed") from None


def _frozen_fit_task(checkout: Path, git: Path, *, task_id: str) -> BfclV4PublicPilotTask:
    """Decode one digest-selected FIT row and no HOLDOUT or possible-answer row."""

    try:
        roster = next(
            item for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster if item.task_id == task_id
        )
    except StopIteration:
        raise BfclV4PublicLiveCliError("smoke task is absent from the frozen roster") from None
    if roster.split is not BfclV4PilotSplit.FIT or "possible_answer" in roster.question_git_path:
        raise BfclV4PublicLiveCliError("smoke task is not an answer-free frozen FIT task")
    try:
        blob = _git_blob(git, checkout, roster.question_git_path)
    except Exception:
        raise BfclV4PublicLiveCliError("frozen FIT question blob could not be read") from None
    if len(blob) != roster.question_blob_size or sha256_bytes(blob) != roster.question_blob_sha256:
        raise BfclV4PublicLiveCliError("frozen FIT question blob identity changed")
    matches = tuple(
        line
        for line in blob.splitlines(keepends=True)
        if len(line) == roster.row_size and sha256_bytes(line) == roster.row_sha256
    )
    if len(matches) != 1:
        raise BfclV4PublicLiveCliError("frozen FIT row identity is not unique")
    row = matches[0]
    payload = row[:-1] if row.endswith(b"\n") else row
    try:
        value = strict_json(payload.decode("utf-8"), "BFCL smoke FIT row")
    except (UnicodeDecodeError, ValueError):
        raise BfclV4PublicLiveCliError("frozen FIT row is invalid") from None
    if not isinstance(value, dict) or set(value) != {"id", "question", "function"}:
        raise BfclV4PublicLiveCliError("frozen FIT row schema changed")
    functions = value["function"]
    if value["id"] != task_id or not isinstance(functions, list) or not functions:
        raise BfclV4PublicLiveCliError("frozen FIT row payload changed")
    names = tuple(item.get("name") for item in functions if isinstance(item, dict))
    if len(names) != len(functions) or any(not isinstance(name, str) or not name for name in names):
        raise BfclV4PublicLiveCliError("frozen FIT function roster is invalid")
    try:
        return BfclV4PublicPilotTask(
            task_id=roster.task_id,
            category=roster.category,
            split=roster.split,
            semantic_family=roster.semantic_family,
            question_git_path=roster.question_git_path,
            question_blob_size=roster.question_blob_size,
            question_blob_sha256=roster.question_blob_sha256,
            row_size=roster.row_size,
            row_sha256=roster.row_sha256,
            candidate_payload_sha256=canonical_sha256(value),
            question_json=canonical_json(value["question"]),
            function_schemas_json=canonical_json(functions),
            official_function_names=names,
        )
    except Exception:
        raise BfclV4PublicLiveCliError("frozen FIT task contract validation failed") from None


def _observed_at_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _activate_live(
    environment: Mapping[str, str],
    *,
    catalog_timeout_seconds: float,
    observed_at_utc: str | None,
) -> _LiveActivation:
    base_url = environment.get(LITELLM_BASE_URL_ENV)
    api_key = environment.get(LITELLM_API_KEY_ENV)
    if not base_url:
        raise BfclV4PublicLiveCliError(f"missing environment variable {LITELLM_BASE_URL_ENV}")
    if not api_key:
        raise BfclV4PublicLiveCliError(f"missing environment variable {LITELLM_API_KEY_ENV}")
    activation_failed = False
    try:
        backend = OpenAICompatibleNativeFunctionBackend.from_endpoint(
            base_url=base_url,
            api_key=api_key,
        )
        model_ids = backend.list_models(timeout_seconds=catalog_timeout_seconds)
        observation = observe_bfcl_v4_public_live_model_catalog(
            model_ids,
            observed_at_utc=observed_at_utc or _observed_at_utc(),
        )
        config = freeze_bfcl_v4_public_live_execution_config(
            catalog_observation=observation,
            backend="litellm-openai-compatible-native",
            backend_fingerprint=backend.fingerprint,
            serializer_fingerprint=backend.serializer_fingerprint,
            parser_fingerprint=backend.parser_fingerprint,
            transport_fingerprint=backend.transport_fingerprint,
        )
    except Exception:
        activation_failed = True
    if activation_failed:
        raise BfclV4PublicLiveCliError("direct-only LiteLLM catalog activation failed") from None
    identities = (
        config.backend_fingerprint,
        config.serializer_fingerprint,
        config.parser_fingerprint,
        config.transport_fingerprint,
    )
    if identities != (
        backend.fingerprint,
        backend.serializer_fingerprint,
        backend.parser_fingerprint,
        backend.transport_fingerprint,
    ):
        raise BfclV4PublicLiveCliError("live backend identities differ from frozen config")
    return _LiveActivation(
        backend=backend,
        config=config,
        advertised_model_count=len(model_ids),
    )


def _campaign_summary(campaign: BfclV4PublicPilotCampaign) -> dict[str, object]:
    first_plan = campaign.replicates[0].call_plan
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_fingerprint": campaign.fingerprint,
        "outer_seeds_u64": [item.outer_seed_u64 for item in campaign.replicates],
        "replicate_ids": [item.replicate_id for item in campaign.replicates],
        "plan_fingerprints": [item.call_plan.fingerprint for item in campaign.replicates],
        "schedule_content_sha256": [
            item.call_plan.schedule_content_sha256 for item in campaign.replicates
        ],
        "calls_per_replicate": campaign.model_calls_per_replicate,
        "total_call_ceiling": campaign.total_model_call_ceiling,
        "first_plan_arm_calls": dict(
            sorted(Counter(item.arm.value for item in first_plan.calls).items())
        ),
        "first_plan_call_kinds": dict(
            sorted(Counter(item.kind.value for item in first_plan.calls).items())
        ),
        "attempt_ledger_ids": list(BFCL_V4_PUBLIC_LIVE_LEDGER_IDS),
    }


def _base_summary(
    *,
    mode: str,
    checkout: Path,
    output: Path,
    campaign: BfclV4PublicPilotCampaign,
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "kind": "bfcl_v4_public_live_cli",
        "mode": mode,
        "checkout": str(checkout),
        "output": str(output),
        "campaign": _campaign_summary(campaign),
        "frozen_route_policy": {
            "direct_first": True,
            "direct": BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
            "fallback": BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,
            "direct_fallback_weight_equivalence_assumed": False,
        },
        "frozen_inference": BFCL_V4_PUBLIC_LIVE_INFERENCE.model_dump(mode="json"),
        "inference_fingerprint": BFCL_V4_PUBLIC_LIVE_INFERENCE.fingerprint,
        "attempt_budget": BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.model_dump(mode="json"),
        "attempt_budget_fingerprint": BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.fingerprint,
        "public_development_only": True,
        "hidden_test_evidence": False,
        "reportable_result": False,
    }


def _live_summary(activation: _LiveActivation) -> dict[str, object]:
    config = activation.config
    return {
        "selected_model_route": config.selected_model_route,
        "catalog_model_count": activation.advertised_model_count,
        "catalog_observation_fingerprint": config.catalog_observation_fingerprint,
        "live_config_fingerprint": config.fingerprint,
        "model_spec_fingerprint": config.model_spec_fingerprint,
        "backend_fingerprint": config.backend_fingerprint,
        "serializer_fingerprint": config.serializer_fingerprint,
        "parser_fingerprint": config.parser_fingerprint,
        "transport_fingerprint": config.transport_fingerprint,
        "credentials_recorded": False,
    }


def _run_smoke(
    *,
    checkout: Path,
    git: Path,
    campaign: BfclV4PublicPilotCampaign,
    activation: _LiveActivation,
) -> dict[str, object]:
    slot = campaign.replicates[0].call_plan.calls[0]
    if slot.kind is not BfclV4PilotCallKind.PARENT_FIT or slot.arm is not BfclV4PilotArm.SCORE:
        raise BfclV4PublicLiveCliError("frozen smoke slot is no longer the first SCORE FIT call")
    if slot.task_id is None:
        raise BfclV4PublicLiveCliError("frozen smoke FIT slot has no task")
    task = _frozen_fit_task(checkout, git, task_id=slot.task_id)
    invocation_failed = False
    try:
        adapter = adapt_bfcl_v4_public_pilot_task(task)
        request = materialize_bfcl_v4_public_native_request(
            task=task,
            adapter=adapter,
            spec=activation.config.model_spec,
            backend=activation.backend,
            seed=slot.seed_u63,
            prompt_kind=BfclV4PublicNativePromptKind.PARENT,
            system_prompt=BFCL_V4_SEED_SYSTEM_PROMPT,
        )
        response = NativeFunctionCallResponse.model_validate(
            activation.backend.invoke(request=request),
            strict=True,
        )
    except Exception:
        invocation_failed = True
    if invocation_failed:
        raise BfclV4PublicLiveCliError("one-call nonreportable smoke invocation failed") from None
    if (
        response.request_fingerprint != request.fingerprint
        or response.serializer_fingerprint != activation.config.serializer_fingerprint
        or response.parser_fingerprint != activation.config.parser_fingerprint
        or response.transport_fingerprint != activation.config.transport_fingerprint
    ):
        raise BfclV4PublicLiveCliError("smoke response differs from the frozen native request")
    return {
        "call_id": slot.call_id,
        "global_slot": slot.global_slot,
        "outer_seed_u64": campaign.replicates[0].outer_seed_u64,
        "task_id": task.task_id,
        "split": task.split.value,
        "request_fingerprint": request.fingerprint,
        "response_fingerprint": response.fingerprint,
        "tool_call_count": response.tool_call_count,
        "usage": response.usage.model_dump(mode="json"),
        "possible_answer_paths_requested": 0,
        "holdout_rows_decoded": 0,
        "holdout_tasks_materialized": 0,
        "holdout_content_exposed_to_model": False,
        "grader_invocations": 0,
        "campaign_journal_transitions": 0,
        "campaign_slot_consumed": False,
        "campaign_evidence_produced": False,
        "must_not_be_reused_as_campaign_evidence": True,
        "raw_model_output_emitted": False,
        "raw_model_output_persisted": False,
    }


def _run_full_campaign(
    *,
    repository: ArtifactStore,
    checkout: Path,
    activation: _LiveActivation,
    campaign: BfclV4PublicPilotCampaign,
    campaign_runner: Callable[..., BfclV4PublicCampaignExecutionRecord],
    campaign_verifier: Callable[..., BfclV4PublicCampaignExecutionVerification],
) -> dict[str, object]:
    execution_failed = False
    try:
        raw_record = campaign_runner(
            repository,
            checkout=checkout,
            live_config=activation.config,
            backend=activation.backend,
        )
    except Exception:
        execution_failed = True
    if execution_failed:
        raise BfclV4PublicLiveCliError("full campaign executor failed") from None

    terminal_failed = False
    try:
        record = BfclV4PublicCampaignExecutionRecord.model_validate(raw_record, strict=True)
        if repository.get_bytes(record.result_ref) != canonical_json_bytes(record.result):
            raise ValueError("terminal root bytes differ")
        verification = BfclV4PublicCampaignExecutionVerification.model_validate(
            campaign_verifier(repository, record.result_ref),
            strict=True,
        )
    except Exception:
        terminal_failed = True
    if terminal_failed:
        raise BfclV4PublicLiveCliError("campaign terminal offline verification failed") from None

    expected = (
        campaign.fingerprint,
        activation.config.fingerprint,
        activation.config.model_spec.fingerprint,
        activation.config.backend_fingerprint,
        activation.config.inference_fingerprint,
        activation.config.attempt_budget.fingerprint,
        record.result.fingerprint,
        record.result.status,
        record.result.verified_closed_model_calls,
        len(record.result.completed_replicates),
        len(record.result.checkpoint_refs),
    )
    observed = (
        record.result.campaign_fingerprint,
        record.result.live_execution_config_fingerprint,
        record.result.model_spec_fingerprint,
        record.result.backend_fingerprint,
        record.result.inference_fingerprint,
        record.result.attempt_budget_fingerprint,
        verification.execution_result_fingerprint,
        verification.status,
        verification.verified_model_calls,
        verification.verified_replicate_count,
        verification.verified_checkpoint_count,
    )
    if observed != expected:
        raise BfclV4PublicLiveCliError("verified campaign terminal differs from live activation")
    analysis_expected = record.result.status is BfclV4CampaignExecutionStatus.COMPLETE
    if verification.analysis_recomputed_and_matched is not analysis_expected:
        raise BfclV4PublicLiveCliError("campaign terminal analysis verification is inconsistent")
    summary_failed = False
    try:
        terminal_summary = build_bfcl_v4_public_terminal_summary(
            repository,
            record,
            verification,
        )
    except Exception:
        summary_failed = True
    if summary_failed:
        raise BfclV4PublicLiveCliError(
            "verified campaign terminal summary could not be rendered"
        ) from None
    return terminal_summary


def _run_bfcl_v4_public_live_impl(
    *,
    checkout: Path,
    output: Path,
    dry_run: bool,
    smoke: bool,
    full: bool = False,
    catalog_timeout_seconds: float = 15.0,
    campaign_runner: Callable[..., BfclV4PublicCampaignExecutionRecord] | None = None,
    campaign_verifier: Callable[..., BfclV4PublicCampaignExecutionVerification] | None = None,
    environment: Mapping[str, str] | None = None,
    observed_at_utc: str | None = None,
) -> dict[str, object]:
    """Validate, smoke, or execute the exact public-development campaign."""

    selected_modes = sum((dry_run, smoke, full))
    if selected_modes != 1:
        raise BfclV4PublicLiveCliError(
            "exactly one of --dry-run, --smoke, or --full must be selected"
        )
    if (
        isinstance(catalog_timeout_seconds, bool)
        or not isinstance(catalog_timeout_seconds, (int, float))
        or not 0 < catalog_timeout_seconds <= 60
    ):
        raise BfclV4PublicLiveCliError("catalog timeout must be in (0, 60]")
    checkout_path, git = _validated_checkout(checkout)
    output_path = _safe_output_path(output, checkout=checkout_path)
    try:
        campaign = build_bfcl_v4_public_pilot_campaign()
    except Exception:
        raise BfclV4PublicLiveCliError("frozen BFCL campaign validation failed") from None
    if BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET.max_attempts != campaign.model_calls_per_replicate:
        raise BfclV4PublicLiveCliError("frozen attempt budget differs from call plan")
    mode = "dry-run" if dry_run else ("smoke" if smoke else "full")
    summary = _base_summary(
        mode=mode,
        checkout=checkout_path,
        output=output_path,
        campaign=campaign,
    )
    if dry_run:
        summary.update(
            {
                "live_config_frozen": False,
                "live_activation_requires_authenticated_direct_catalog": True,
                "network_calls": 0,
                "model_calls": 0,
                "score_bearing_calls": 0,
                "grader_invocations": 0,
                "output_writes": 0,
            }
        )
        return summary

    activation = _activate_live(
        os.environ if environment is None else environment,
        catalog_timeout_seconds=float(catalog_timeout_seconds),
        observed_at_utc=observed_at_utc,
    )
    summary["live"] = _live_summary(activation)
    if smoke:
        summary.update(
            {
                "network_calls": 2,
                "catalog_calls": 1,
                "model_calls": 1,
                "score_bearing_calls": 0,
                "smoke": _run_smoke(
                    checkout=checkout_path,
                    git=git,
                    campaign=campaign,
                    activation=activation,
                ),
            }
        )
        return summary

    repository = ArtifactStore(output_path)
    terminal = _run_full_campaign(
        repository=repository,
        checkout=checkout_path,
        activation=activation,
        campaign=campaign,
        campaign_runner=campaign_runner or run_bfcl_v4_public_campaign,
        campaign_verifier=campaign_verifier or verify_bfcl_v4_public_campaign_execution,
    )
    summary.update(
        {
            "catalog_calls": 1,
            "registered_model_calls": campaign.total_model_call_ceiling,
            "terminal": terminal,
        }
    )
    return summary


def run_bfcl_v4_public_live(
    *,
    checkout: Path,
    output: Path,
    dry_run: bool,
    smoke: bool,
    full: bool = False,
    catalog_timeout_seconds: float = 15.0,
    campaign_runner: Callable[..., BfclV4PublicCampaignExecutionRecord] | None = None,
    campaign_verifier: Callable[..., BfclV4PublicCampaignExecutionVerification] | None = None,
    environment: Mapping[str, str] | None = None,
    observed_at_utc: str | None = None,
) -> dict[str, object]:
    """Run with a sanitized exception boundary that releases credential-bearing frames."""

    failure_message: str | None = None
    try:
        return _run_bfcl_v4_public_live_impl(
            checkout=checkout,
            output=output,
            dry_run=dry_run,
            smoke=smoke,
            full=full,
            catalog_timeout_seconds=catalog_timeout_seconds,
            campaign_runner=campaign_runner,
            campaign_verifier=campaign_verifier,
            environment=environment,
            observed_at_utc=observed_at_utc,
        )
    except BfclV4PublicLiveCliError as error:
        failure_message = str(error)
        discard_exception_graph(error)
    environment = None
    campaign_runner = None
    campaign_verifier = None
    raise BfclV4PublicLiveCliError(failure_message) from None


def bfcl_v4_public_live(
    checkout: Annotated[
        Path,
        typer.Option("--checkout", help="Exact pinned BFCL/Gorilla Git checkout."),
    ] = Path("/tmp/spiral-bfcl-upstream"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Absent or empty content-addressed output root."),
    ] = Path("runs/bfcl-v4-public-live"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate pinned local inputs and all 300 slots; make no network call or write.",
        ),
    ] = False,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help="Run exactly one ungraded, nonreportable FIT native-function request.",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Execute 300 model calls (three by 100); public-development evidence only.",
        ),
    ] = False,
    catalog_timeout_seconds: Annotated[
        float,
        typer.Option("--catalog-timeout-seconds", help="Direct-only GET /models timeout."),
    ] = 15.0,
) -> None:
    """Explicitly choose validation, smoke, or a 300-call public-development run."""

    bad_parameter_message: str | None = None
    try:
        payload = run_bfcl_v4_public_live(
            checkout=checkout,
            output=output,
            dry_run=dry_run,
            smoke=smoke,
            full=full,
            catalog_timeout_seconds=catalog_timeout_seconds,
        )
    except BfclV4PublicLiveCliError as error:
        bad_parameter_message = str(error)
        discard_exception_graph(error)
    if bad_parameter_message is not None:
        raise typer.BadParameter(bad_parameter_message) from None
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("mode") == "full" and payload["terminal"]["status"] == "incomplete":
        raise typer.Exit(code=1)


app = typer.Typer(name="bfcl-v4-public-live", no_args_is_help=True)
app.command("run")(bfcl_v4_public_live)


__all__ = [
    "BFCL_V4_PUBLIC_LIVE_LEDGER_IDS",
    "BfclV4PublicLiveCliError",
    "app",
    "bfcl_v4_public_live",
    "run_bfcl_v4_public_live",
]
