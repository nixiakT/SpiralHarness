import json
from pathlib import Path

from typer.testing import CliRunner

from spiral_harness import __version__
from spiral_harness import cli as cli_module
from spiral_harness.benchmark import GSM8K_PROVENANCE
from spiral_harness.cli import app
from spiral_harness.core import ProtocolPartition


def test_version_option() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


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
