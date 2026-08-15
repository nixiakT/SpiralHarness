from __future__ import annotations

from fractions import Fraction
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4 import (
    _EXPECTED_PACKAGE_TREE,
    BFCL_V4_ALL_SCORING_CATEGORIES,
    BFCL_V4_CATEGORY_COUNTS,
    BFCL_V4_EFFECTIVE_MAX_MODEL_STEPS_PER_TURN,
    BFCL_V4_OFFICIAL_METRIC_TERMS,
    BFCL_V4_PACKAGE_ROOT,
    BfclV4MetricError,
    BfclV4SnapshotIntegrityError,
    BfclV4SnapshotVerification,
    FileSetIdentity,
    _file_set_identity,
    _tree_paths,
    bfcl_v4_official_overall,
)


def _accuracies(value: float = 0.0) -> dict[str, float]:
    return {category: value for category in BFCL_V4_ALL_SCORING_CATEGORIES}


def test_pinned_all_scoring_roster_has_22_categories_and_5106_units() -> None:
    assert len(BFCL_V4_ALL_SCORING_CATEGORIES) == 22
    assert tuple(item.category for item in BFCL_V4_CATEGORY_COUNTS) == (
        BFCL_V4_ALL_SCORING_CATEGORIES
    )
    assert sum(item.count for item in BFCL_V4_CATEGORY_COUNTS) == 5106
    assert BFCL_V4_EFFECTIVE_MAX_MODEL_STEPS_PER_TURN == 21


def test_package_identity_is_from_clean_git_archive_not_runtime_cache() -> None:
    assert (
        FileSetIdentity(
            file_count=183,
            total_bytes=13_513_439,
            sha256="3753addd78c10a6e59e3488ffdc5fb38cb46929380925ccfaecfcdb1d8b533b2",
        )
        == _EXPECTED_PACKAGE_TREE
    )


def test_official_metric_terms_are_exactly_normalized() -> None:
    assert sum((term.weight for term in BFCL_V4_OFFICIAL_METRIC_TERMS), Fraction()) == 1
    assert "live_relevance" not in {term.category for term in BFCL_V4_OFFICIAL_METRIC_TERMS}
    assert bfcl_v4_official_overall(_accuracies(0.0)) == 0.0
    assert bfcl_v4_official_overall(_accuracies(1.0)) == pytest.approx(1.0)


def test_official_metric_reproduces_hierarchical_category_weights() -> None:
    values = _accuracies()
    values["multi_turn_base"] = 1.0
    assert bfcl_v4_official_overall(values) == pytest.approx(0.075)

    values = _accuracies()
    values["web_search_base"] = 1.0
    assert bfcl_v4_official_overall(values) == pytest.approx(0.1)

    values = _accuracies()
    values["live_relevance"] = 1.0
    assert bfcl_v4_official_overall(values) == 0.0


@pytest.mark.parametrize("missing", ["simple_python", "live_relevance"])
def test_official_metric_rejects_partial_evaluation(missing: str) -> None:
    values = _accuracies()
    del values[missing]
    with pytest.raises(BfclV4MetricError, match="missing="):
        bfcl_v4_official_overall(values)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "1"])
def test_official_metric_rejects_invalid_category_scores(value: object) -> None:
    values: dict[str, object] = _accuracies()
    values["simple_python"] = value
    with pytest.raises(BfclV4MetricError):
        bfcl_v4_official_overall(values)  # type: ignore[arg-type]


def test_file_set_identity_binds_paths_bytes_and_extra_files(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.json").write_bytes(b"one\n")
    (tmp_path / "a" / "two.json").write_bytes(b"two\n")
    paths = (PurePosixPath("a/one.json"), PurePosixPath("a/two.json"))

    original = _file_set_identity(tmp_path, paths)
    reversed_order = _file_set_identity(tmp_path, tuple(reversed(paths)))
    assert original == reversed_order

    (tmp_path / "a" / "one.json").write_bytes(b"changed\n")
    assert _file_set_identity(tmp_path, paths) != original

    with pytest.raises(BfclV4SnapshotIntegrityError, match="non-empty and unique"):
        _file_set_identity(tmp_path, (paths[0], paths[0]))
    with pytest.raises(BfclV4SnapshotIntegrityError, match="unsafe"):
        _file_set_identity(tmp_path, (PurePosixPath("../escape"),))


def test_package_tree_ignores_only_conventional_python_bytecode_cache(tmp_path: Path) -> None:
    package_root = tmp_path.joinpath(*BFCL_V4_PACKAGE_ROOT.parts)
    cache_root = package_root / "constants" / "__pycache__"
    cache_root.mkdir(parents=True)
    source = package_root / "constants" / "category_mapping.py"
    source.write_text("ALL_SCORING_CATEGORIES = ()\n", encoding="utf-8")
    (cache_root / "category_mapping.cpython-312.pyc").write_bytes(b"runtime cache")
    (cache_root / "category_mapping.cpython-312.pyo").write_bytes(b"optimized cache")

    assert _tree_paths(tmp_path, BFCL_V4_PACKAGE_ROOT) == (
        BFCL_V4_PACKAGE_ROOT / "constants" / "category_mapping.py",
    )

    meaningful_extra = cache_root / "category_mapping.json"
    meaningful_extra.write_text("{}\n", encoding="utf-8")
    assert _tree_paths(tmp_path, BFCL_V4_PACKAGE_ROOT) == (
        BFCL_V4_PACKAGE_ROOT / "constants" / "__pycache__" / "category_mapping.json",
        BFCL_V4_PACKAGE_ROOT / "constants" / "category_mapping.py",
    )


def test_snapshot_type_cannot_claim_public_answers_are_hidden() -> None:
    identity = FileSetIdentity(file_count=1, total_bytes=1, sha256="1" * 64)
    roster = {
        "categories": tuple(item.model_dump() for item in BFCL_V4_CATEGORY_COUNTS),
        "roster_sha256": "2" * 64,
    }
    common = {
        "package_tree": identity,
        "scoring_question_files": identity,
        "public_answer_files": identity,
        "release_metadata_files": identity,
        "roster": roster,
        "official_metric_fingerprint": "3" * 64,
    }
    checked = BfclV4SnapshotVerification.model_validate(common, strict=True)
    assert checked.hidden_test_evidence is False
    assert checked.runtime_execution_attested is False
    assert checked.reportable_result is False

    with pytest.raises(ValidationError):
        BfclV4SnapshotVerification.model_validate(
            {**common, "hidden_test_evidence": True},
            strict=True,
        )
