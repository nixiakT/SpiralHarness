from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiral_harness.evolution.benchmark_selection import select_validation_champion


def _artifact(path: Path, *, arm: str, accuracy: float, split: str = "val") -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark": "fixture",
                "model": "fixed-model",
                "split": split,
                "arm": arm,
                "retrieval_k": 8,
                "accuracy": accuracy,
                "total_tokens": 100,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_selects_validation_champion_and_records_rejection(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    evolved = _artifact(tmp_path / "evolved.json", arm="evolved", accuracy=0.6)

    trace = select_validation_champion((pure, evolved), output=tmp_path / "trace", minimum_gain=0.1)

    assert trace["accepted"] is True
    assert trace["gain"] == pytest.approx(0.2)
    assert trace["champion"]["name"] == "evolved@k=8"
    assert trace["rejected"] == ["pure@k=8"]


def test_rejects_test_artifact_from_evolution(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    test = _artifact(tmp_path / "test.json", arm="evolved", accuracy=0.9, split="test")

    with pytest.raises(ValueError, match="validation artifacts only"):
        select_validation_champion((pure, test), output=tmp_path / "trace")
