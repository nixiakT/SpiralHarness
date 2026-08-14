from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from spiral_harness import cli_development_four_arm as command_module
from spiral_harness.cli import app
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.execution.model import ModelBackend
from spiral_harness.execution.pure_model import PureModelBackend
from spiral_harness.providers.openai_compatible import OpenAICompatibleChatBackend


class _FakeAdapter:
    def __init__(self, label: str) -> None:
        self._label = label
        self._task_ids = tuple(f"{label}-public-{index:02d}" for index in range(10))

    def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
        assert partition is ProtocolPartition.EXPLORATION
        return self._task_ids

    def load_task(self, task_id: str) -> SimpleNamespace:
        assert task_id in self._task_ids
        return SimpleNamespace(question=f"Public {self._label} question for {task_id}?")


class _FakeRef:
    def __init__(self, label: str) -> None:
        self._label = label

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "sha256": ("a" if self._label == "result" else "b") * 64,
            "size": 1,
            "media_type": f"application/x-{self._label}",
        }


def _fake_adapters() -> tuple[_FakeAdapter, _FakeAdapter]:
    return _FakeAdapter("gsm8k"), _FakeAdapter("bbh")


def _install_local_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    gsm8k, bbh = _fake_adapters()

    def fake_load(*, gsm8k_dir: Path, bbh_dir: Path):
        assert gsm8k_dir == Path("local/gsm8k")
        assert bbh_dir == Path("local/bbh")
        return gsm8k, bbh

    monkeypatch.setattr(command_module, "_load_development_adapters", fake_load)


def _base_args(output: Path) -> list[str]:
    return [
        "benchmark",
        "dev-four-arm",
        "--gsm8k-dir",
        "local/gsm8k",
        "--bbh-dir",
        "local/bbh",
        "--output",
        str(output),
        "--max-output-tokens",
        "64",
        "--max-tokens-per-attempt",
        "128",
    ]


def test_dry_run_needs_no_credentials_network_calls_or_output_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "must-remain-absent"
    _install_local_adapters(monkeypatch)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def forbidden_backend(*args, **kwargs):
        pytest.fail(f"dry-run constructed a live backend: {args!r}, {kwargs!r}")

    def forbidden_run(*args, **kwargs):
        pytest.fail(f"dry-run executed the study: {args!r}, {kwargs!r}")

    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        forbidden_backend,
    )
    monkeypatch.setattr(command_module, "run_development_four_arm", forbidden_run)

    result = CliRunner().invoke(app, [*_base_args(output), "--dry-run"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["network_calls"] == payload["executed_model_calls"] == 0
    assert payload["unique_task_count"] == len(set(payload["unique_tasks"])) == 16
    assert len(payload["task_roster"]["fit"]) == 8
    assert len(payload["task_roster"]["holdout"]) == 8
    assert payload["fit_task_order"] == [task["task_id"] for task in payload["task_roster"]["fit"]]
    assert payload["holdout_task_order"] == [
        task["task_id"] for task in payload["task_roster"]["holdout"]
    ]
    assert payload["arms"] == ["pure", "static", "score", "full"]
    assert payload["planned_model_calls"] == 58
    assert payload["plan_token_ceiling"] == 58 * 128
    assert payload["flags"] == {
        "feedback_view_only": True,
        "nonattested": True,
        "nonreportable": True,
        "nonsealed": True,
    }
    assert payload["reportable_benchmark_result"] is False
    assert payload["sealed_evidence"] is False
    assert payload["execution_attested"] is False
    assert payload["provider_identity_attested"] is False
    assert payload["feedback_view_only"] is True
    assert not output.exists()


@pytest.mark.parametrize("partition", ["gate", "sealed"])
def test_command_refuses_gate_and_sealed_before_loading_or_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail(f"refused partition reached a side-effect boundary: {args!r}, {kwargs!r}")

    monkeypatch.setattr(command_module, "_load_development_adapters", forbidden)
    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        forbidden,
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "dev-four-arm",
            "--partition",
            partition,
            "--output",
            str(tmp_path / "unused"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "refuses gate and sealed" in result.output


def test_command_refuses_nonempty_output_before_loading_or_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "prior-artifact.json").write_text("{}", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail(f"nonempty output reached a side-effect boundary: {args!r}, {kwargs!r}")

    monkeypatch.setattr(command_module, "_load_development_adapters", forbidden)
    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        forbidden,
    )

    result = CliRunner().invoke(app, [*_base_args(output), "--dry-run"])

    assert result.exit_code == 2
    assert "must be absent or empty" in result.output
    assert (output / "prior-artifact.json").read_text(encoding="utf-8") == "{}"


def test_command_refuses_symlink_output_before_loading_or_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output-link"
    output.symlink_to(target, target_is_directory=True)

    def forbidden(*args, **kwargs):
        pytest.fail(f"symlink output reached a side-effect boundary: {args!r}, {kwargs!r}")

    monkeypatch.setattr(command_module, "_load_development_adapters", forbidden)
    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        forbidden,
    )

    result = CliRunner().invoke(app, [*_base_args(output), "--dry-run"])

    assert result.exit_code == 2
    assert "must not be a symbolic link" in result.output


def test_live_mode_requires_fixed_environment_names_before_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_local_adapters(monkeypatch)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "fixture-secret")

    def forbidden_backend(*args, **kwargs):
        pytest.fail(f"missing environment reached backend construction: {args!r}, {kwargs!r}")

    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        forbidden_backend,
    )

    result = CliRunner().invoke(app, _base_args(tmp_path / "unused"))

    assert result.exit_code == 2
    assert "LITELLM_BASE_URL" in result.output
    assert "fixture-secret" not in result.output


def test_live_mode_reads_credentials_only_from_environment_and_emits_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "live"
    _install_local_adapters(monkeypatch)
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "fixture-secret")
    backend = SimpleNamespace(fingerprint="fixture-openai-compatible-backend")

    def fake_from_endpoint(*, base_url: str, api_key: str):
        assert base_url == "http://gateway.example/v1"
        assert api_key == "fixture-secret"
        return backend

    def fake_run(**kwargs):
        assert kwargs["output"] == output
        assert kwargs["backend"] is backend
        assert kwargs["plan"].max_model_calls == 58
        assert kwargs["max_tokens_per_attempt"] == 128
        return SimpleNamespace(
            payload={
                "kind": "permanently_nonreportable_development_four_arm",
                "model_call_count": 58,
                "nominal_model_spec_shared_across_all_calls": True,
                "provider_identity": {"all_observations_identical": False},
                "resource_usage": {"input_tokens": 100, "output_tokens": 50},
                "selection_trace": [],
                "holdout_metrics": {"accuracy": {"pure": 0.0, "full": 0.5}},
                "disclaimer": "development only",
            },
            result_ref=_FakeRef("result"),
            closure_ref=_FakeRef("closure"),
        )

    monkeypatch.setattr(
        command_module.OpenAICompatibleChatBackend,
        "from_endpoint",
        fake_from_endpoint,
    )
    monkeypatch.setattr(command_module, "run_development_four_arm", fake_run)

    result = CliRunner().invoke(app, _base_args(output))

    assert result.exit_code == 0, result.output
    assert "fixture-secret" not in result.output
    assert "http://gateway.example" not in result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "live"
    assert payload["executed_model_calls"] == payload["planned_model_calls"] == 58
    assert payload["nominal_model_spec_shared_across_all_calls"] is True
    assert payload["flags"]["nonreportable"] is True
    assert payload["flags"]["nonattested"] is True


def test_command_has_no_cli_endpoint_or_secret_options() -> None:
    result = CliRunner().invoke(
        app,
        ["benchmark", "dev-four-arm", "--api-key", "should-not-be-accepted"],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "should-not-be-accepted" not in result.output


def test_openai_compatible_backend_satisfies_fixed_and_true_pure_interfaces() -> None:
    backend = OpenAICompatibleChatBackend.from_endpoint(
        base_url="http://gateway.example/v1",
        api_key="fixture-only",
    )

    assert isinstance(backend, ModelBackend)
    assert isinstance(backend, PureModelBackend)
