from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spiral_harness import cli as cli_module
from spiral_harness.benchmark.gsm8k_smoke import run_gsm8k_smoke
from spiral_harness.cli import app
from spiral_harness.execution.contracts import BackendResponse, BackendTokenUsage


class FixtureBackend:
    fingerprint = "fixture-live-backend-v1"

    def __init__(self, output: str = "Work.\n#### 5") -> None:
        self.output = output
        self.calls = 0

    def invoke(self, *, spec: object, request: object) -> BackendResponse:
        del spec, request
        self.calls += 1
        return BackendResponse(
            output=self.output,
            usage=BackendTokenUsage(input_tokens=12, output_tokens=8),
        )


def write_fixture_gsm8k(tmp_path: Path) -> tuple[Path, Path]:
    train_rows = (
        {
            "question": "Lina has 2 shells and finds 3 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 5",
        },
        {
            "question": "Lina has 20 shells and finds 30 more. How many shells does she have?",
            "answer": "Add the shells.\n#### 50",
        },
        {
            "question": "Four boxes hold 6 pens each. How many pens are there?",
            "answer": "Multiply.\n#### 24",
        },
        {
            "question": "A class has 11 girls and 9 boys. How many students are there?",
            "answer": "Add the students.\n#### 20",
        },
    )
    test_rows = (
        {
            "question": "A shelf has 7 red and 8 blue books. How many books are there?",
            "answer": "Add the books.\n#### 15",
        },
    )
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    return train_path, test_path


def test_run_gsm8k_smoke_publishes_non_reportable_artifact(tmp_path: Path) -> None:
    train_path, test_path = write_fixture_gsm8k(tmp_path)
    backend = FixtureBackend()

    result = run_gsm8k_smoke(
        train_path=train_path,
        test_path=test_path,
        output=tmp_path / "run",
        backend=backend,
        model="dashscope/qwen36-35b-a3b",
        limit=1,
        max_output_tokens=64,
        max_tokens_per_attempt=128,
        verify_pinned=False,
    )

    assert backend.calls == 1
    assert result.payload["kind"] == "non_reportable_gsm8k_smoke"
    assert result.payload["reportable"] is False
    assert result.payload["completed_count"] == 1
    assert result.payload["total_tokens"] == 20
    assert result.artifact_ref.media_type == "application/vnd.spiral-harness.gsm8k-smoke.v1+json"
    assert (tmp_path / "run" / "artifacts" / "objects").is_dir()


def test_run_gsm8k_smoke_validates_limit_and_partition_size(tmp_path: Path) -> None:
    train_path, test_path = write_fixture_gsm8k(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        run_gsm8k_smoke(
            train_path=train_path,
            test_path=test_path,
            output=tmp_path / "run",
            backend=FixtureBackend(),
            model="dashscope/qwen36-35b-a3b",
            limit=0,
            verify_pinned=False,
        )


def test_smoke_cli_requires_env_before_materializing_dataset(monkeypatch, tmp_path: Path) -> None:
    called = False

    def fake_materialize(*args: object, **kwargs: object) -> tuple[Path, ...]:
        nonlocal called
        called = True
        del args, kwargs
        return ()

    monkeypatch.setattr(cli_module, "materialize_dataset", fake_materialize)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        ["benchmark", "smoke-gsm8k", "--dataset-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "missing environment variable LITELLM_BASE_URL" in result.output
    assert called is False


def test_smoke_cli_refuses_sealed_partition(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "smoke-gsm8k",
            "--dataset-dir",
            str(tmp_path),
            "--partition",
            "sealed",
        ],
    )

    assert result.exit_code == 2
    assert "refuses sealed partition" in result.output
