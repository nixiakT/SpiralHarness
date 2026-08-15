from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType, MethodType, ModuleType, SimpleNamespace, TracebackType

import pytest
import typer
from typer.testing import CliRunner

import spiral_harness.cli_bfcl_v4_public_live as subject
from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import BfclV4PilotSplit
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
)
from spiral_harness.cli import app as root_app
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import BackendTokenUsage
from spiral_harness.experiments.bfcl_v4_public_campaign_executor_contracts import (
    BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE,
    BfclV4CampaignExecutionFailure,
    BfclV4CampaignExecutionStatus,
    BfclV4CampaignFailureStage,
    BfclV4CampaignVerifiedReplicate,
    BfclV4PublicCampaignExecutionRecord,
    BfclV4PublicCampaignExecutionResult,
    BfclV4PublicCampaignExecutionVerification,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
)
from spiral_harness.providers.openai_native_contracts import NativeFunctionCallResponse


class _FakeNativeBackend:
    fingerprint = "a" * 64
    serializer_fingerprint = "b" * 64
    parser_fingerprint = "c" * 64
    transport_fingerprint = "d" * 64

    def __init__(self, *, model_ids: tuple[str, ...]) -> None:
        self.model_ids = model_ids
        self.catalog_timeouts: list[float] = []
        self.requests: list[object] = []
        self.response = NativeFunctionCallResponse(
            request_fingerprint="f" * 64,
            serializer_fingerprint=self.serializer_fingerprint,
            parser_fingerprint=self.parser_fingerprint,
            transport_fingerprint=self.transport_fingerprint,
            tools_fingerprint="e" * 64,
            tool_calls=(),
            assistant_text="fixture raw model output must remain private",
            finish_reason="stop",
            usage=BackendTokenUsage(input_tokens=17, output_tokens=3),
        )

    def list_models(self, *, timeout_seconds: float) -> tuple[str, ...]:
        self.catalog_timeouts.append(timeout_seconds)
        return tuple(sorted(set(self.model_ids)))

    def invoke(self, *, request: object) -> NativeFunctionCallResponse:
        self.requests.append(request)
        return self.response


def _install_checkout_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    checkout = tmp_path / "pinned-bfcl"
    checkout.mkdir()
    git = tmp_path / "git"
    git.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(subject, "_validated_checkout", lambda path: (checkout, git))
    return checkout, git


def _install_live_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: _FakeNativeBackend,
    *,
    expected_key: str,
) -> None:
    def from_endpoint(*, base_url: str, api_key: str) -> _FakeNativeBackend:
        assert base_url == "http://litellm.internal/v1"
        assert api_key == expected_key
        return backend

    monkeypatch.setattr(
        subject,
        "OpenAICompatibleNativeFunctionBackend",
        SimpleNamespace(from_endpoint=from_endpoint),
    )


def _environment(secret: str) -> dict[str, str]:
    return {
        subject.LITELLM_BASE_URL_ENV: "http://litellm.internal/v1",
        subject.LITELLM_API_KEY_ENV: secret,
    }


def _exception_graph_text(error: BaseException) -> str:
    pending = [error]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), repr(current)))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    return "\n".join(rendered)


def _reachable_traceback_local_text(error: BaseException) -> str:
    """Render reachable exception/frame state without invoking arbitrary attributes."""

    pending: list[object] = [error]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if len(seen) > 20_000:
            pytest.fail("exception reachability audit exceeded its bounded object graph")
        for renderer in (str, repr):
            try:
                rendered.append(renderer(current))
            except Exception:
                rendered.append(f"<{type(current).__name__}:unrenderable>")
        if isinstance(current, BaseException):
            pending.extend(current.args)
            pending.extend(
                item for item in (current.__cause__, current.__context__) if item is not None
            )
            if current.__traceback__ is not None:
                pending.append(current.__traceback__)
            continue
        if isinstance(current, TracebackType):
            if not current.tb_frame.f_code.co_name.startswith("test_"):
                pending.extend(tuple(current.tb_frame.f_locals.values()))
            if current.tb_next is not None:
                pending.append(current.tb_next)
            continue
        if isinstance(current, Mapping):
            for key, value in current.items():
                pending.extend((key, value))
            continue
        if isinstance(current, (tuple, list, set, frozenset)):
            pending.extend(current)
            continue
        if isinstance(current, (ModuleType, type, FunctionType, MethodType)):
            continue
        try:
            state = object.__getattribute__(current, "__dict__")
        except (AttributeError, TypeError):
            continue
        if isinstance(state, dict):
            pending.append(state)
    return "\n".join(rendered)


def test_traceback_reachability_auditor_detects_nested_frame_local() -> None:
    secret = "auditor-must-detect-this-frame-local"

    def explode() -> None:
        retained_backend = SimpleNamespace(api_key=secret)
        if retained_backend.api_key:
            raise RuntimeError("sanitized")

    with pytest.raises(RuntimeError) as captured:
        explode()

    assert secret in _reachable_traceback_local_text(captured.value)


def _ref(label: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        sha256=canonical_sha256({"fixture": label}),
        size=1,
        media_type=media_type,
    )


def _terminal_record(repository, config, *, complete: bool) -> BfclV4PublicCampaignExecutionRecord:
    campaign = build_bfcl_v4_public_pilot_campaign()
    campaign_ref = repository.put_json(
        campaign,
        media_type=BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE,
    )
    config_ref = repository.put_json(
        config,
        media_type=BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE,
    )
    completed: tuple[BfclV4CampaignVerifiedReplicate, ...] = ()
    checkpoints: tuple[ArtifactRef, ...] = ()
    analysis_input_ref = None
    analysis_ref = None
    failure = BfclV4CampaignExecutionFailure(
        stage=BfclV4CampaignFailureStage.REPLICATE_EXECUTION,
        completed_replicate_count=0,
        active_replicate_ordinal=0,
        active_replicate_id=campaign.replicates[0].replicate_id,
        active_outer_seed_u64=campaign.replicates[0].outer_seed_u64,
    )
    if complete:
        completed = tuple(
            BfclV4CampaignVerifiedReplicate(
                ordinal=replicate.ordinal,
                replicate_id=replicate.replicate_id,
                outer_seed_u64=replicate.outer_seed_u64,
                plan_fingerprint=replicate.call_plan.fingerprint,
                schedule_content_sha256=replicate.call_plan.schedule_content_sha256,
                attempt_ledger_id=f"{campaign.campaign_id}/{replicate.replicate_id}",
                model_spec_fingerprint=config.model_spec.fingerprint,
                backend_fingerprint=config.backend_fingerprint,
                inference_fingerprint=config.inference_fingerprint,
                attempt_budget_fingerprint=config.attempt_budget.fingerprint,
                run_result_ref=_ref(
                    f"run-{replicate.ordinal}",
                    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
                ),
                run_result_fingerprint=canonical_sha256({"fixture": f"run-{replicate.ordinal}"}),
                verification_ref=_ref(
                    f"verification-{replicate.ordinal}",
                    BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE,
                ),
                verification_fingerprint=canonical_sha256(
                    {"fixture": f"verification-{replicate.ordinal}"}
                ),
                closure_ref=_ref(
                    f"closure-{replicate.ordinal}",
                    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
                ),
                joint_selection_decision_ref=_ref(
                    f"selection-{replicate.ordinal}",
                    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
                ),
                descriptive_metrics_ref=_ref(
                    f"metrics-{replicate.ordinal}",
                    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
                ),
                provider_attempts_succeeded=100,
                provider_attempts_failed=0,
                provider_identity_observation_count=100,
                provider_declared_identity_consistent=True,
            )
            for replicate in campaign.replicates
        )
        checkpoints = tuple(
            _ref(f"checkpoint-{index}", BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE)
            for index in range(3)
        )
        analysis_input_ref = _ref("analysis-input", BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE)
        analysis_ref = _ref("analysis", BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE)
        failure = None
    result = BfclV4PublicCampaignExecutionResult(
        status=(
            BfclV4CampaignExecutionStatus.COMPLETE
            if complete
            else BfclV4CampaignExecutionStatus.INCOMPLETE
        ),
        campaign_fingerprint=campaign.fingerprint,
        campaign_ref=campaign_ref,
        live_execution_config_ref=config_ref,
        live_execution_config_fingerprint=config.fingerprint,
        model_spec_fingerprint=config.model_spec.fingerprint,
        backend_fingerprint=config.backend_fingerprint,
        inference_fingerprint=config.inference_fingerprint,
        attempt_budget_fingerprint=config.attempt_budget.fingerprint,
        completed_replicates=completed,
        checkpoint_refs=checkpoints,
        latest_checkpoint_ref=checkpoints[-1] if checkpoints else None,
        verified_closed_model_calls=300 if complete else 0,
        analysis_input_ref=analysis_input_ref,
        analysis_input_fingerprint=(
            None if analysis_input_ref is None else analysis_input_ref.sha256
        ),
        analysis_ref=analysis_ref,
        analysis_fingerprint=None if analysis_ref is None else analysis_ref.sha256,
        failure=failure,
    )
    result_ref = repository.put_json(
        result,
        media_type=BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
    )
    return BfclV4PublicCampaignExecutionRecord(result=result, result_ref=result_ref)


def _terminal_verification(
    record: BfclV4PublicCampaignExecutionRecord,
) -> BfclV4PublicCampaignExecutionVerification:
    complete = record.result.status is BfclV4CampaignExecutionStatus.COMPLETE
    return BfclV4PublicCampaignExecutionVerification(
        execution_result_fingerprint=record.result.fingerprint,
        status=record.result.status,
        verified_replicate_count=3 if complete else 0,
        verified_model_calls=300 if complete else 0,
        verified_checkpoint_count=3 if complete else 0,
        analysis_recomputed_and_matched=complete,
    )


def test_dry_run_validates_frozen_campaign_without_credentials_network_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "must-remain-absent"

    def forbidden_activation(*args, **kwargs):
        pytest.fail(f"dry-run reached network activation: {args!r}, {kwargs!r}")

    monkeypatch.setattr(subject, "_activate_live", forbidden_activation)
    payload = subject.run_bfcl_v4_public_live(
        checkout=checkout,
        output=output,
        dry_run=True,
        smoke=False,
        environment={},
    )

    assert payload["mode"] == "dry-run"
    assert payload["network_calls"] == 0
    assert payload["model_calls"] == 0
    assert payload["score_bearing_calls"] == 0
    assert payload["grader_invocations"] == 0
    assert payload["output_writes"] == 0
    assert payload["live_config_frozen"] is False
    assert payload["live_activation_requires_authenticated_direct_catalog"] is True
    assert payload["campaign"]["outer_seeds_u64"] == [2026081501, 2026081502, 2026081503]
    assert payload["campaign"]["calls_per_replicate"] == 100
    assert payload["campaign"]["total_call_ceiling"] == 300
    assert len(payload["campaign"]["plan_fingerprints"]) == 3
    assert payload["campaign"]["first_plan_arm_calls"] == {
        "full": 28,
        "pure": 8,
        "pure-at-b": 28,
        "score": 28,
        "static": 8,
    }
    assert not output.exists()


def test_standalone_typer_dry_run_emits_secret_free_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "dry-output"
    secret = "fixture-key-that-must-not-appear"
    monkeypatch.setenv(subject.LITELLM_BASE_URL_ENV, "http://unused.invalid/v1")
    monkeypatch.setenv(subject.LITELLM_API_KEY_ENV, secret)

    result = CliRunner().invoke(
        subject.app,
        ["--checkout", str(checkout), "--output", str(output), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["network_calls"] == payload["model_calls"] == 0
    assert secret not in result.stdout
    assert not output.exists()


def test_smoke_uses_direct_first_catalog_and_one_ungraded_fit_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "smoke-output-must-remain-absent"
    secret = "fixture-smoke-secret"
    backend = _FakeNativeBackend(
        model_ids=(
            subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
            subject.BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,
        )
    )
    _install_live_backend(monkeypatch, backend, expected_key=secret)

    loaded_task_ids: list[str] = []

    def fit_task(checkout_path: Path, git_path: Path, *, task_id: str):
        assert checkout_path == checkout
        assert git_path.name == "git"
        loaded_task_ids.append(task_id)
        return SimpleNamespace(task_id=task_id, split=BfclV4PilotSplit.FIT)

    request = SimpleNamespace(fingerprint="f" * 64)
    monkeypatch.setattr(subject, "_frozen_fit_task", fit_task)
    monkeypatch.setattr(subject, "adapt_bfcl_v4_public_pilot_task", lambda task: object())

    def materialize(**kwargs):
        assert kwargs["task"].split is BfclV4PilotSplit.FIT
        assert kwargs["prompt_kind"] is subject.BfclV4PublicNativePromptKind.PARENT
        assert kwargs["system_prompt"] == subject.BFCL_V4_SEED_SYSTEM_PROMPT
        assert kwargs["spec"].model == subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
        assert kwargs["backend"] is backend
        return request

    monkeypatch.setattr(subject, "materialize_bfcl_v4_public_native_request", materialize)

    payload = subject.run_bfcl_v4_public_live(
        checkout=checkout,
        output=output,
        dry_run=False,
        smoke=True,
        catalog_timeout_seconds=7.5,
        environment=_environment(secret),
        observed_at_utc="2026-08-15T17:00:00Z",
    )
    rendered = json.dumps(payload, sort_keys=True)

    assert backend.catalog_timeouts == [7.5]
    assert backend.requests == [request]
    assert loaded_task_ids == ["simple_python_0"]
    assert payload["live"]["selected_model_route"] == (
        subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE
    )
    assert payload["catalog_calls"] == 1
    assert payload["model_calls"] == 1
    assert payload["score_bearing_calls"] == 0
    assert payload["smoke"]["split"] == "fit"
    assert payload["smoke"]["possible_answer_paths_requested"] == 0
    assert payload["smoke"]["holdout_rows_decoded"] == 0
    assert payload["smoke"]["holdout_tasks_materialized"] == 0
    assert payload["smoke"]["holdout_content_exposed_to_model"] is False
    assert payload["smoke"]["grader_invocations"] == 0
    assert payload["smoke"]["campaign_journal_transitions"] == 0
    assert payload["smoke"]["campaign_slot_consumed"] is False
    assert payload["smoke"]["campaign_evidence_produced"] is False
    assert payload["smoke"]["must_not_be_reused_as_campaign_evidence"] is True
    assert payload["smoke"]["raw_model_output_emitted"] is False
    assert payload["smoke"]["raw_model_output_persisted"] is False
    assert "fixture raw model output" not in rendered
    assert secret not in rendered
    assert "api_key" not in rendered.casefold()
    assert not output.exists()


def test_smoke_response_must_bind_the_exact_materialized_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    secret = "binding-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,))
    _install_live_backend(monkeypatch, backend, expected_key=secret)
    monkeypatch.setattr(
        subject,
        "_frozen_fit_task",
        lambda *args, task_id, **kwargs: SimpleNamespace(
            task_id=task_id,
            split=BfclV4PilotSplit.FIT,
        ),
    )
    monkeypatch.setattr(subject, "adapt_bfcl_v4_public_pilot_task", lambda task: object())
    monkeypatch.setattr(
        subject,
        "materialize_bfcl_v4_public_native_request",
        lambda **kwargs: SimpleNamespace(fingerprint="0" * 64),
    )

    with pytest.raises(subject.BfclV4PublicLiveCliError, match="smoke response differs"):
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=tmp_path / "unused",
            dry_run=False,
            smoke=True,
            environment=_environment(secret),
            observed_at_utc="2026-08-15T17:00:00Z",
        )


def test_smoke_failure_releases_backend_key_from_reachable_traceback_locals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "failed-smoke-cas"
    secret = "smoke-frame-local-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,))
    backend.retained_api_key = secret
    _install_live_backend(monkeypatch, backend, expected_key=secret)
    monkeypatch.setattr(
        subject,
        "_frozen_fit_task",
        lambda *args, task_id, **kwargs: SimpleNamespace(
            task_id=task_id,
            split=BfclV4PilotSplit.FIT,
        ),
    )

    def explode(task):
        provider_payload = {"api_key": secret, "task": task}
        raise RuntimeError(repr(provider_payload))

    monkeypatch.setattr(subject, "adapt_bfcl_v4_public_pilot_task", explode)
    with pytest.raises(subject.BfclV4PublicLiveCliError) as captured:
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=output,
            dry_run=False,
            smoke=True,
            environment=_environment(secret),
            observed_at_utc="2026-08-15T17:00:00Z",
        )

    assert str(captured.value) == "one-call nonreportable smoke invocation failed"
    assert secret not in _exception_graph_text(captured.value)
    assert secret not in _reachable_traceback_local_text(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not output.exists()


def test_live_failures_are_sanitized_and_never_echo_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    secret = "secret-inside-provider-exception"

    def explode(**kwargs):
        raise RuntimeError(secret + repr(kwargs))

    monkeypatch.setattr(
        subject,
        "OpenAICompatibleNativeFunctionBackend",
        SimpleNamespace(from_endpoint=explode),
    )
    with pytest.raises(subject.BfclV4PublicLiveCliError) as captured:
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=tmp_path / "unused",
            dry_run=False,
            smoke=True,
            environment=_environment(secret),
            observed_at_utc="2026-08-15T17:00:00Z",
        )

    assert str(captured.value) == "direct-only LiteLLM catalog activation failed"
    assert secret not in _exception_graph_text(captured.value)
    assert secret not in _reachable_traceback_local_text(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_typer_boundary_discards_caught_error_frames_before_bad_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "typer-boundary-frame-secret"

    def explode(**kwargs):
        leaked_environment = {"api_key": secret, "kwargs": kwargs}
        raise subject.BfclV4PublicLiveCliError("sanitized command failure") from RuntimeError(
            repr(leaked_environment)
        )

    monkeypatch.setattr(subject, "run_bfcl_v4_public_live", explode)
    with pytest.raises(typer.BadParameter) as captured:
        subject.bfcl_v4_public_live(
            checkout=tmp_path / "unused-checkout",
            output=tmp_path / "unused-output",
            dry_run=False,
            smoke=True,
            full=False,
        )

    assert str(captured.value) == "sanitized command failure"
    assert secret not in _exception_graph_text(captured.value)
    assert secret not in _reachable_traceback_local_text(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_registered_typer_command_never_emits_activation_key_to_stderr_or_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "failed-registered-smoke-cas"
    secret = "registered-activation-frame-secret"

    def explode(**kwargs):
        leaked_request = {"api_key": secret, "kwargs": kwargs}
        raise RuntimeError(repr(leaked_request))

    monkeypatch.setattr(
        subject,
        "OpenAICompatibleNativeFunctionBackend",
        SimpleNamespace(from_endpoint=explode),
    )
    monkeypatch.setenv(subject.LITELLM_BASE_URL_ENV, "http://litellm.internal/v1")
    monkeypatch.setenv(subject.LITELLM_API_KEY_ENV, secret)
    result = CliRunner().invoke(
        root_app,
        [
            "benchmark",
            "bfcl-v4-public-live",
            "--checkout",
            str(checkout),
            "--output",
            str(output),
            "--smoke",
        ],
    )
    rendered = result.stdout + result.stderr

    assert result.exit_code == 2
    assert "direct-only LiteLLM catalog activation failed" in rendered
    assert secret not in rendered
    if result.exception is not None:
        assert secret not in _reachable_traceback_local_text(result.exception)
    assert not output.exists()


@pytest.mark.parametrize("missing", [subject.LITELLM_BASE_URL_ENV, subject.LITELLM_API_KEY_ENV])
def test_live_modes_require_only_the_two_fixed_environment_names(
    missing: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    environment = _environment("present-secret")
    del environment[missing]

    def forbidden_factory(**kwargs):
        pytest.fail(f"missing environment reached backend construction: {kwargs!r}")

    monkeypatch.setattr(
        subject,
        "OpenAICompatibleNativeFunctionBackend",
        SimpleNamespace(from_endpoint=forbidden_factory),
    )
    with pytest.raises(subject.BfclV4PublicLiveCliError, match=missing) as captured:
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=tmp_path / "unused",
            dry_run=False,
            smoke=True,
            environment=environment,
        )

    assert "present-secret" not in _exception_graph_text(captured.value)
    assert "present-secret" not in _reachable_traceback_local_text(captured.value)


@pytest.mark.parametrize("complete", [True, False])
def test_full_mode_runs_and_offline_verifies_typed_terminal_root(
    complete: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "full-cas"
    secret = "executor-bound-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE,))
    _install_live_backend(monkeypatch, backend, expected_key=secret)
    records: list[BfclV4PublicCampaignExecutionRecord] = []
    rendered_terminals: list[BfclV4PublicCampaignExecutionRecord] = []

    def runner(repository, *, checkout: Path, live_config, backend):
        assert checkout == tmp_path / "pinned-bfcl"
        assert backend is backend_object
        assert live_config.selected_model_route == (
            subject.BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
        )
        record = _terminal_record(repository, live_config, complete=complete)
        records.append(record)
        return record

    backend_object = backend

    def verifier(repository, result_ref):
        assert repository.get_bytes(result_ref)
        assert result_ref == records[0].result_ref
        return _terminal_verification(records[0])

    real_summary_builder = subject.build_bfcl_v4_public_terminal_summary

    def summary_builder(repository, record, verification):
        rendered_terminals.append(record)
        if not complete:
            return real_summary_builder(repository, record, verification)
        return {
            "status": "complete",
            "protocol_complete": True,
            "verified_closed_model_calls": verification.verified_model_calls,
            "terminal_offline_verification_passed": True,
            "analysis_recomputed_and_matched": True,
            "analysis_ref": record.result.analysis_ref.model_dump(mode="json"),
            "analysis_summary": {"source": "unit-fixture"},
            "failure": None,
        }

    monkeypatch.setattr(
        subject,
        "build_bfcl_v4_public_terminal_summary",
        summary_builder,
    )

    payload = subject.run_bfcl_v4_public_live(
        checkout=checkout,
        output=output,
        dry_run=False,
        smoke=False,
        full=True,
        campaign_runner=runner,
        campaign_verifier=verifier,
        environment=_environment(secret),
        observed_at_utc="2026-08-15T17:00:00Z",
    )
    rendered = json.dumps(payload, sort_keys=True)

    assert len(records) == len(rendered_terminals) == 1
    assert backend.catalog_timeouts == [15.0]
    assert backend.requests == []
    assert payload["live"]["selected_model_route"] == (
        subject.BFCL_V4_PUBLIC_LIVE_FALLBACK_MODEL_ROUTE
    )
    assert payload["registered_model_calls"] == 300
    assert payload["terminal"]["status"] == ("complete" if complete else "incomplete")
    assert payload["terminal"]["protocol_complete"] is complete
    assert payload["terminal"]["verified_closed_model_calls"] == (300 if complete else 0)
    assert payload["terminal"]["terminal_offline_verification_passed"] is True
    assert payload["terminal"]["analysis_recomputed_and_matched"] is complete
    if complete:
        assert payload["terminal"]["analysis_ref"] is not None
        assert payload["terminal"]["analysis_summary"] == {"source": "unit-fixture"}
        assert payload["terminal"]["failure"] is None
    else:
        assert payload["terminal"]["analysis_ref"] is None
        assert payload["terminal"]["analysis_summary"] is None
        assert payload["terminal"]["failure"] == {
            "active_outer_seed_u64": 2026081501,
            "active_replicate_id": "search-seed-2026081501",
            "active_replicate_ordinal": 0,
            "automatic_retry_attempted": False,
            "completed_replicate_count": 0,
            "exception_text_persisted": False,
            "resumable_checkpoint_claimed": False,
            "stage": "replicate-execution",
            "unverified_run_result_ref": None,
        }
    assert secret not in rendered
    assert (output / "objects").is_dir()


def test_full_executor_exception_cannot_retain_key_in_traceback_or_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "failed-full-cas"
    secret = "full-executor-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,))
    _install_live_backend(monkeypatch, backend, expected_key=secret)

    def explode(*args, **kwargs):
        raise RuntimeError(secret + repr((args, kwargs)))

    with pytest.raises(subject.BfclV4PublicLiveCliError) as captured:
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=output,
            dry_run=False,
            smoke=False,
            full=True,
            campaign_runner=explode,
            campaign_verifier=lambda *args, **kwargs: pytest.fail("verifier was reached"),
            environment=_environment(secret),
            observed_at_utc="2026-08-15T17:00:00Z",
        )

    assert str(captured.value) == "full campaign executor failed"
    assert secret not in _exception_graph_text(captured.value)
    assert secret not in _reachable_traceback_local_text(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(
        secret.encode() not in path.read_bytes() for path in output.rglob("*") if path.is_file()
    )


def test_terminal_summary_exception_cannot_retain_key_in_traceback_or_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "failed-summary-cas"
    secret = "summary-bound-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,))
    _install_live_backend(monkeypatch, backend, expected_key=secret)
    records: list[BfclV4PublicCampaignExecutionRecord] = []

    def runner(repository, *, checkout: Path, live_config, backend):
        del checkout, backend
        record = _terminal_record(repository, live_config, complete=False)
        records.append(record)
        return record

    def verifier(repository, result_ref):
        assert repository.get_bytes(result_ref)
        return _terminal_verification(records[0])

    def explode(*args, **kwargs):
        raise RuntimeError(secret + repr((args, kwargs)))

    monkeypatch.setattr(subject, "build_bfcl_v4_public_terminal_summary", explode)
    with pytest.raises(subject.BfclV4PublicLiveCliError) as captured:
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=output,
            dry_run=False,
            smoke=False,
            full=True,
            campaign_runner=runner,
            campaign_verifier=verifier,
            environment=_environment(secret),
            observed_at_utc="2026-08-15T17:00:00Z",
        )

    assert str(captured.value) == "verified campaign terminal summary could not be rendered"
    assert secret not in _exception_graph_text(captured.value)
    assert secret not in _reachable_traceback_local_text(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(
        secret.encode() not in path.read_bytes() for path in output.rglob("*") if path.is_file()
    )


def test_registered_production_command_emits_incomplete_terminal_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    output = tmp_path / "registered-full-cas"
    secret = "registered-command-secret"
    backend = _FakeNativeBackend(model_ids=(subject.BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,))
    _install_live_backend(monkeypatch, backend, expected_key=secret)
    monkeypatch.setenv(subject.LITELLM_BASE_URL_ENV, "http://litellm.internal/v1")
    monkeypatch.setenv(subject.LITELLM_API_KEY_ENV, secret)
    records: list[BfclV4PublicCampaignExecutionRecord] = []

    def runner(repository, *, checkout: Path, live_config, backend):
        del checkout, backend
        record = _terminal_record(repository, live_config, complete=False)
        records.append(record)
        return record

    def verifier(repository, result_ref):
        assert repository.get_bytes(result_ref)
        return _terminal_verification(records[0])

    monkeypatch.setattr(subject, "run_bfcl_v4_public_campaign", runner)
    monkeypatch.setattr(subject, "verify_bfcl_v4_public_campaign_execution", verifier)
    result = CliRunner().invoke(
        root_app,
        [
            "benchmark",
            "bfcl-v4-public-live",
            "--checkout",
            str(checkout),
            "--output",
            str(output),
            "--full",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "full"
    assert payload["terminal"]["status"] == "incomplete"
    assert payload["terminal"]["verified_closed_model_calls"] == 0
    assert payload["terminal"]["analysis_ref"] is None
    assert payload["terminal"]["failure"]["stage"] == "replicate-execution"
    assert secret not in result.stdout


@pytest.mark.parametrize("mode_args", [[], ["--dry-run", "--full"]])
def test_registered_command_rejects_missing_or_multiple_modes_before_side_effects(
    mode_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "must-remain-absent"

    def forbidden_checkout(path: Path):
        pytest.fail(f"invalid root CLI mode touched checkout: {path}")

    monkeypatch.setattr(subject, "_validated_checkout", forbidden_checkout)
    result = CliRunner().invoke(
        root_app,
        [
            "benchmark",
            "bfcl-v4-public-live",
            "--output",
            str(output),
            *mode_args,
        ],
    )

    assert result.exit_code == 2
    assert "exactly one" in result.output
    assert not output.exists()


@pytest.mark.parametrize("kind", ["nonempty", "symlink"])
def test_output_safety_fails_before_network_or_cas_writes(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = _install_checkout_validation(monkeypatch, tmp_path)
    if kind == "nonempty":
        output = tmp_path / "occupied"
        output.mkdir()
        (output / "prior.json").write_text("{}", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output = tmp_path / "output-link"
        output.symlink_to(target, target_is_directory=True)

    def forbidden_activation(*args, **kwargs):
        pytest.fail(f"unsafe output reached network: {args!r}, {kwargs!r}")

    monkeypatch.setattr(subject, "_activate_live", forbidden_activation)
    with pytest.raises(subject.BfclV4PublicLiveCliError):
        subject.run_bfcl_v4_public_live(
            checkout=checkout,
            output=output,
            dry_run=True,
            smoke=False,
            environment={},
        )


@pytest.mark.parametrize(
    ("dry_run", "smoke", "full"),
    [(False, False, False), (True, True, False), (True, False, True), (False, True, True)],
)
def test_exactly_one_mode_is_required_before_any_side_effect(
    dry_run: bool,
    smoke: bool,
    full: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_checkout(path: Path):
        pytest.fail(f"invalid mode touched checkout: {path}")

    monkeypatch.setattr(subject, "_validated_checkout", forbidden_checkout)
    with pytest.raises(subject.BfclV4PublicLiveCliError, match="exactly one"):
        subject.run_bfcl_v4_public_live(
            checkout=tmp_path / "unused",
            output=tmp_path / "unused-output",
            dry_run=dry_run,
            smoke=smoke,
            full=full,
        )


def test_relative_output_starts_from_physical_cwd_not_a_logical_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = tmp_path / "physical-workspace"
    physical.mkdir()
    logical = tmp_path / "logical-workspace"
    logical.symlink_to(physical, target_is_directory=True)
    assert logical.is_symlink()

    monkeypatch.setattr(subject, "_physical_cwd", lambda: logical.resolve(strict=True))
    resolved = subject._absolute_without_symlinks(Path("runs/bfcl-live"))

    assert resolved == physical / "runs" / "bfcl-live"
    assert logical not in resolved.parents


def test_fit_loader_decodes_exactly_one_public_fit_task_when_checkout_is_available() -> None:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else Path("/tmp/spiral-bfcl-upstream")
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact-source smoke-loader test")

    resolved, git = subject._validated_checkout(checkout)
    task = subject._frozen_fit_task(resolved, git, task_id="simple_python_0")

    assert task.task_id == "simple_python_0"
    assert task.split is BfclV4PilotSplit.FIT
    assert task.possible_answer_data_in_contract is False
    assert task.answer_derived_identity_present is False
    assert "possible_answer" not in task.question_git_path
