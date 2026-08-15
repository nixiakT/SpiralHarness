from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiral_harness.evolution.benchmark_selection import select_validation_champion


def _artifact(
    path: Path,
    *,
    arm: str,
    accuracy: object,
    split: str = "val",
    total_tokens: object = 100,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark": "fixture",
                "model": "fixed-model",
                "split": split,
                "arm": arm,
                "retrieval_k": 8,
                "accuracy": accuracy,
                "total_tokens": total_tokens,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_selects_validation_champion_and_records_rejection(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    evolved = _artifact(tmp_path / "evolved.json", arm="evolved", accuracy=0.6)

    trace = select_validation_champion(
        (pure, evolved),
        parent_artifact_path=pure,
        output=tmp_path / "trace",
        minimum_gain=0.1,
    )

    assert trace["accepted"] is True
    assert trace["gain"] == pytest.approx(0.2)
    assert trace["parent"]["name"] == "pure@k=8"
    assert trace["challenger"]["name"] == "evolved@k=8"
    assert trace["champion"]["name"] == "evolved@k=8"
    assert trace["rejected"] == ["pure@k=8"]


def test_equal_score_cannot_promote_only_by_using_fewer_tokens(tmp_path: Path) -> None:
    parent = _artifact(
        tmp_path / "parent.json",
        arm="current",
        accuracy=0.6,
        total_tokens=200,
    )
    cheaper = _artifact(
        tmp_path / "cheaper.json",
        arm="cheaper",
        accuracy=0.6,
        total_tokens=50,
    )

    trace = select_validation_champion(
        (cheaper, parent),
        parent_artifact_path=parent,
        output=tmp_path / "trace",
        minimum_gain=0.0,
    )

    assert trace["challenger"]["name"] == "cheaper@k=8"
    assert trace["gain"] == 0.0
    assert trace["accepted"] is False
    assert trace["champion"]["name"] == "current@k=8"
    assert trace["rejected"] == ["cheaper@k=8"]


def test_non_pure_parent_is_the_gain_reference(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    parent = _artifact(tmp_path / "parent.json", arm="evolved-a", accuracy=0.6)
    candidate = _artifact(tmp_path / "candidate.json", arm="evolved-b", accuracy=0.61)

    trace = select_validation_champion(
        (pure, candidate, parent),
        parent_artifact_path=parent,
        output=tmp_path / "trace",
    )

    assert trace["parent"]["name"] == "evolved-a@k=8"
    assert trace["challenger"]["name"] == "evolved-b@k=8"
    assert trace["gain"] == pytest.approx(0.01)
    assert trace["accepted"] is True
    assert trace["champion"]["name"] == "evolved-b@k=8"


def test_rejected_round_keeps_current_champion_for_the_next_round(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    current = _artifact(tmp_path / "current.json", arm="current", accuracy=0.6)
    rejected = _artifact(tmp_path / "rejected.json", arm="rejected", accuracy=0.59)

    rejected_trace = select_validation_champion(
        (pure, rejected, current),
        parent_artifact_path=current,
        output=tmp_path / "rejected-trace",
    )

    assert rejected_trace["accepted"] is False
    assert rejected_trace["champion"]["name"] == "current@k=8"
    assert rejected_trace["gain"] == pytest.approx(-0.01)

    next_candidate = _artifact(tmp_path / "next.json", arm="next", accuracy=0.605)
    next_trace = select_validation_champion(
        (pure, rejected, current, next_candidate),
        parent_artifact_path=current,
        output=tmp_path / "next-trace",
    )

    assert next_trace["parent"]["name"] == "current@k=8"
    assert next_trace["challenger"]["name"] == "next@k=8"
    assert next_trace["gain"] == pytest.approx(0.005)
    assert next_trace["accepted"] is True
    assert next_trace["champion"]["name"] == "next@k=8"


@pytest.mark.parametrize("accuracy", [-0.01, 1.01, True, "0.5"])
def test_rejects_invalid_accuracy_without_coercion(
    tmp_path: Path,
    accuracy: object,
) -> None:
    parent = _artifact(tmp_path / "parent.json", arm="parent", accuracy=0.5)
    invalid = _artifact(tmp_path / "invalid.json", arm="invalid", accuracy=accuracy)

    with pytest.raises(ValueError, match="candidate accuracy"):
        select_validation_champion(
            (parent, invalid),
            parent_artifact_path=parent,
            output=tmp_path / "trace",
        )


@pytest.mark.parametrize("total_tokens", [True, 100.0, "100"])
def test_rejects_non_integer_token_counts_without_coercion(
    tmp_path: Path,
    total_tokens: object,
) -> None:
    parent = _artifact(tmp_path / "parent.json", arm="parent", accuracy=0.5)
    invalid = _artifact(
        tmp_path / "invalid.json",
        arm="invalid",
        accuracy=0.6,
        total_tokens=total_tokens,
    )

    with pytest.raises(ValueError, match="total_tokens must be a non-negative integer"):
        select_validation_champion(
            (parent, invalid),
            parent_artifact_path=parent,
            output=tmp_path / "trace",
        )


def test_rejects_test_artifact_from_evolution(tmp_path: Path) -> None:
    pure = _artifact(tmp_path / "pure.json", arm="pure", accuracy=0.4)
    test = _artifact(tmp_path / "test.json", arm="evolved", accuracy=0.9, split="test")

    with pytest.raises(ValueError, match="validation artifacts only"):
        select_validation_champion(
            (pure, test),
            parent_artifact_path=pure,
            output=tmp_path / "trace",
        )
