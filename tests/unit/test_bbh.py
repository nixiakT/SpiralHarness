from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiral_harness.benchmark.bbh import (
    BBHDataError,
    BBHLogicalDeductionSevenAdapter,
    extract_bbh_answer,
)
from spiral_harness.benchmark.bbh_smoke import run_bbh_smoke
from spiral_harness.execution.contracts import BackendResponse, BackendTokenUsage


class FixtureBackend:
    fingerprint = "fixture-bbh-backend-v1"

    def invoke(self, *, spec: object, request: object) -> BackendResponse:
        del spec, request
        return BackendResponse(
            output="Brief reasoning.\n(B)",
            usage=BackendTokenUsage(input_tokens=20, output_tokens=8),
        )


def write_bbh(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "canary": "public benchmark canary",
                "examples": [
                    {"input": "Question one\nOptions:\n(A) x\n(B) y", "target": "(B)"},
                    {"input": "Question two\nOptions:\n(A) x\n(B) y", "target": "(A)"},
                    {"input": "Question three\nOptions:\n(A) x\n(B) y", "target": "(B)"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [("reasoning\n(B)", "(B)"), ("The answer is (C).", "(C)"), ("I considered (A).", None)],
)
def test_extract_bbh_answer_requires_a_standalone_final_marker(
    text: str, expected: str | None
) -> None:
    assert extract_bbh_answer(text) == expected


def test_bbh_adapter_rejects_invalid_targets(tmp_path: Path) -> None:
    path = write_bbh(tmp_path / "data.json")
    value = json.loads(path.read_text())
    value["examples"][0]["target"] = "B"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BBHDataError, match="invalid BBH example"):
        BBHLogicalDeductionSevenAdapter(path, verify_pinned=False)


def test_bbh_smoke_runs_a_deterministic_development_sample(tmp_path: Path) -> None:
    path = write_bbh(tmp_path / "data.json")
    result = run_bbh_smoke(
        dataset_path=path,
        output=tmp_path / "run",
        backend=FixtureBackend(),
        model="fixture-model",
        limit=2,
        seed=19,
        max_output_tokens=32,
        max_tokens_per_attempt=64,
        verify_pinned=False,
    )

    assert result.payload["partition"] == "public-development"
    assert result.payload["sampling_method"] == "sha256-order-without-replacement-v1"
    assert len(result.payload["sampled_task_ids"]) == 2
    assert result.payload["completed_count"] == 2
    assert result.payload["total_tokens"] == 56
    assert result.payload["reportable"] is False
