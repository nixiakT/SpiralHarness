import json

from typer.testing import CliRunner

from spiral_harness import __version__
from spiral_harness.cli import app


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
