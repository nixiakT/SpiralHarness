import json
import re
from pathlib import Path

from typer.testing import CliRunner

from spiral_harness import __version__
from spiral_harness import cli as cli_module
from spiral_harness.benchmark.datasets import GSM8K_PROVENANCE
from spiral_harness.cli import app
from spiral_harness.core.experiment import ProtocolPartition

_ANSI_OSC = re.compile(r"\x1b\](?:[^\x07\x1b]|\x1b(?!\\))*(?:\x07|\x1b\\)")
_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_terminal_escapes(value: str) -> str:
    """Remove OSC and CSI sequences without importing Typer's private Click fork."""

    return _ANSI_CSI.sub("", _ANSI_OSC.sub("", value))


def test_version_option() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_penguin_help_distinguishes_canonical_and_reduced_schedules() -> None:
    result = CliRunner().invoke(
        app,
        ["benchmark", "penguin-public", "--help"],
        color=True,
    )

    assert result.exit_code == 0
    without_ansi = _strip_terminal_escapes(result.output)
    normalized_output = " ".join(without_ansi.replace("│", " ").split())
    assert "canonical 17-call public schedule" in normalized_output
    assert "1-4 are" in normalized_output
    assert "reduced noncanonical exploratory variants" in normalized_output


def test_terminal_escape_stripper_covers_rich_colors_and_osc_titles() -> None:
    decorated = "\x1b]0;SpiralHarness\x07\x1b[1;31mvisible\x1b[0m"

    assert _strip_terminal_escapes(decorated) == "visible"


def test_controlled_demo_command_emits_replay_root(tmp_path) -> None:
    output = tmp_path / "demo-artifacts"

    result = CliRunner().invoke(app, ["demo", "--output", str(output)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["decision"] == "promote"
    assert payload["mean_delta"] == 0.75
    assert len(payload["run_manifest_sha256"]) == 64
    assert "not real benchmark results" in payload["disclaimer"]
    assert (output / "objects").is_dir()


def test_benchmark_fetch_materializes_and_reports_frozen_identity(tmp_path, monkeypatch) -> None:
    output = tmp_path / "gsm8k"

    def fake_materialize(provenance, directory: Path):
        assert provenance == GSM8K_PROVENANCE
        directory.mkdir(parents=True)
        paths = tuple(directory / artifact.filename for artifact in provenance.artifacts)
        for path in paths:
            path.write_text("fixture", encoding="utf-8")
        return paths

    class FakeSplitManifest:
        fingerprint = "b" * 64

    class FakeAdapter:
        fingerprint = "a" * 64
        split_manifest = FakeSplitManifest()

        def __init__(self, train_path: Path, test_path: Path) -> None:
            assert train_path.name == "train.jsonl"
            assert test_path.name == "test.jsonl"

        def task_roster(self, partition: ProtocolPartition) -> tuple[str, ...]:
            counts = {
                ProtocolPartition.EXPLORATION: 3,
                ProtocolPartition.GATE: 2,
                ProtocolPartition.SEALED: 1,
            }
            return tuple(f"{partition.value}-{index}" for index in range(counts[partition]))

    monkeypatch.setattr(cli_module, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(cli_module, "GSM8KBenchmarkAdapter", FakeAdapter)

    result = CliRunner().invoke(
        app,
        ["benchmark", "fetch", "gsm8k", "--output", str(output)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["adapter_fingerprint"] == "a" * 64
    assert payload["split_manifest_fingerprint"] == "b" * 64
    assert payload["partitions"] == {"exploration": 3, "gate": 2, "sealed": 1}
    assert "sealed evaluation data" in payload["warning"]


def test_benchmark_fetch_rejects_an_unregistered_name() -> None:
    result = CliRunner().invoke(app, ["benchmark", "fetch", "unknown"])

    assert result.exit_code == 2
    assert "unknown benchmark" in result.output


def test_list_models_is_availability_only_and_classifies_gateway_routes(monkeypatch) -> None:
    class FakeBackend:
        fingerprint = "f" * 64

        def list_models(self, *, timeout_seconds: float) -> tuple[str, ...]:
            assert timeout_seconds == 9.0
            return (
                "MiniMax-M2.5",
                "dashscope/qwen36-35b-a3b",
                "openai/gpt-test",
            )

    def fake_from_endpoint(*, base_url: str, api_key: str) -> FakeBackend:
        assert base_url == "http://gateway.example/v1"
        assert api_key == "fixture-secret"
        return FakeBackend()

    monkeypatch.setenv("TEST_LITELLM_URL", "http://gateway.example/v1")
    monkeypatch.setenv("TEST_LITELLM_KEY", "fixture-secret")
    monkeypatch.setattr(
        cli_module.OpenAICompatibleNativeFunctionBackend,
        "from_endpoint",
        fake_from_endpoint,
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "list-models",
            "--timeout-seconds",
            "9",
            "--base-url-env",
            "TEST_LITELLM_URL",
            "--api-key-env",
            "TEST_LITELLM_KEY",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kind"] == "availability_only_model_catalog"
    assert payload["score_bearing_calls"] == 0
    assert payload["count"] == 3
    assert payload["routes"] == {
        "dashscope": ["dashscope/qwen36-35b-a3b"],
        "direct": ["MiniMax-M2.5"],
        "openai": ["openai/gpt-test"],
    }
    assert len(payload["catalog_sha256"]) == 64


def test_list_models_requires_credentials_without_printing_them(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_LITELLM_URL", raising=False)
    monkeypatch.setenv("PRESENT_LITELLM_KEY", "fixture-secret")

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "list-models",
            "--base-url-env",
            "MISSING_LITELLM_URL",
            "--api-key-env",
            "PRESENT_LITELLM_KEY",
        ],
    )

    assert result.exit_code == 2
    assert "MISSING_LITELLM_URL" in result.output
    assert "fixture-secret" not in result.output
