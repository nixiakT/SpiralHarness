"""Pinned BFCL V4 suite identity and official-metric helpers.

This module does not run a model and does not turn BFCL into a hidden test.
BFCL V4 publishes both its questions and its possible-answer files in the
official repository.  The verifier below only proves that a local checkout
matches the pinned public snapshot; runtime and provider attestations remain
separate obligations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_UPSTREAM_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
BFCL_V4_UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_V4_RELEASE_VERSION = "2026.3.23"
# ``MAXIMUM_STEP_LIMIT`` is 20 upstream, but the loop checks ``count > 20``
# only after issuing and executing the current call.  A non-terminating turn
# can therefore issue 21 model requests (count values 0 through 20).
BFCL_V4_EFFECTIVE_MAX_MODEL_STEPS_PER_TURN = 21
BFCL_V4_PACKAGE_ROOT = PurePosixPath("berkeley-function-call-leaderboard/bfcl_eval")
BFCL_V4_DATA_ROOT = BFCL_V4_PACKAGE_ROOT / "data"
_IGNORED_PYTHON_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})


class BfclV4SnapshotIntegrityError(ValueError):
    """A checkout differs from the complete pinned BFCL V4 snapshot."""


class BfclV4MetricError(ValueError):
    """Inputs cannot produce an official full-suite BFCL V4 overall score."""


class FileSetIdentity(ImmutableModel):
    """Digest of a sorted, path-bound set of regular files."""

    schema_version: Literal["1"] = "1"
    file_count: Annotated[int, Field(gt=0, strict=True)]
    total_bytes: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256


class BfclV4CategoryCount(ImmutableModel):
    """Exact effective evaluation-unit count for one official scoring category."""

    category: NonEmptyStr
    count: Annotated[int, Field(gt=0, strict=True)]


class BfclV4MetricTerm(ImmutableModel):
    """One category-accuracy coefficient in the official V4 overall metric."""

    category: NonEmptyStr
    numerator: Annotated[int, Field(gt=0, strict=True)]
    denominator: Annotated[int, Field(gt=0, strict=True)]

    @property
    def weight(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class BfclV4RosterIdentity(ImmutableModel):
    """Content identity of the expanded 22-category all-scoring roster."""

    schema_version: Literal["1"] = "1"
    categories: Annotated[tuple[BfclV4CategoryCount, ...], Field(min_length=22, max_length=22)]
    evaluation_unit_count: Literal[5106] = 5106
    roster_sha256: Sha256

    @field_validator("categories")
    @classmethod
    def _canonicalize_categories(
        cls,
        values: tuple[BfclV4CategoryCount, ...],
    ) -> tuple[BfclV4CategoryCount, ...]:
        by_name = {item.category: item for item in values}
        if len(by_name) != len(values):
            raise ValueError("BFCL category counts must not repeat")
        if frozenset(by_name) != frozenset(BFCL_V4_ALL_SCORING_CATEGORIES):
            raise ValueError("BFCL category-count roster differs from all_scoring")
        return tuple(by_name[name] for name in BFCL_V4_ALL_SCORING_CATEGORIES)

    @model_validator(mode="after")
    def _counts_close(self) -> BfclV4RosterIdentity:
        if sum(item.count for item in self.categories) != self.evaluation_unit_count:
            raise ValueError("BFCL category counts do not sum to 5106")
        return self


class BfclV4SnapshotVerification(ImmutableModel):
    """Successful local content audit, deliberately not an execution receipt."""

    schema_version: Literal["1"] = "1"
    suite_id: Literal["bfcl-v4@6ea57973"] = "bfcl-v4@6ea57973"
    upstream_repository: Literal[BFCL_V4_UPSTREAM_REPOSITORY] = BFCL_V4_UPSTREAM_REPOSITORY
    upstream_commit: Literal[BFCL_V4_UPSTREAM_COMMIT] = BFCL_V4_UPSTREAM_COMMIT
    release_version: Literal[BFCL_V4_RELEASE_VERSION] = BFCL_V4_RELEASE_VERSION
    license_spdx: Literal["Apache-2.0"] = "Apache-2.0"
    package_tree: FileSetIdentity
    scoring_question_files: FileSetIdentity
    public_answer_files: FileSetIdentity
    release_metadata_files: FileSetIdentity
    roster: BfclV4RosterIdentity
    official_metric_fingerprint: Sha256
    questions_public: Literal[True] = True
    possible_answers_public: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    partial_evaluation: Literal[False] = False
    source_snapshot_bound: Literal[True] = True
    official_grader_source_bound: Literal[True] = True
    dependency_environment_attested: Literal[False] = False
    runtime_execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    reportable_result: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


BFCL_V4_ALL_SCORING_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
    "web_search_base",
    "web_search_no_snippet",
)

BFCL_V4_CATEGORY_COUNTS = (
    BfclV4CategoryCount(category="simple_python", count=400),
    BfclV4CategoryCount(category="simple_java", count=100),
    BfclV4CategoryCount(category="simple_javascript", count=50),
    BfclV4CategoryCount(category="multiple", count=200),
    BfclV4CategoryCount(category="parallel", count=200),
    BfclV4CategoryCount(category="parallel_multiple", count=200),
    BfclV4CategoryCount(category="irrelevance", count=240),
    BfclV4CategoryCount(category="live_simple", count=258),
    BfclV4CategoryCount(category="live_multiple", count=1053),
    BfclV4CategoryCount(category="live_parallel", count=16),
    BfclV4CategoryCount(category="live_parallel_multiple", count=24),
    BfclV4CategoryCount(category="live_irrelevance", count=884),
    BfclV4CategoryCount(category="live_relevance", count=16),
    BfclV4CategoryCount(category="multi_turn_base", count=200),
    BfclV4CategoryCount(category="multi_turn_miss_func", count=200),
    BfclV4CategoryCount(category="multi_turn_miss_param", count=200),
    BfclV4CategoryCount(category="multi_turn_long_context", count=200),
    BfclV4CategoryCount(category="memory_kv", count=155),
    BfclV4CategoryCount(category="memory_vector", count=155),
    BfclV4CategoryCount(category="memory_rec_sum", count=155),
    BfclV4CategoryCount(category="web_search_base", count=100),
    BfclV4CategoryCount(category="web_search_no_snippet", count=100),
)

_DIRECT_QUESTION_CATEGORIES = BFCL_V4_ALL_SCORING_CATEGORIES[:17]
_SHARED_QUESTION_SOURCES = {
    "memory_kv": "memory",
    "memory_vector": "memory",
    "memory_rec_sum": "memory",
    "web_search_base": "web_search",
    "web_search_no_snippet": "web_search",
}
_NO_ANSWER_CATEGORIES = frozenset({"irrelevance", "live_irrelevance", "live_relevance"})


def _question_file(source_category: str) -> PurePosixPath:
    return BFCL_V4_DATA_ROOT / f"BFCL_v4_{source_category}.json"


def _answer_file(source_category: str) -> PurePosixPath:
    return BFCL_V4_DATA_ROOT / "possible_answer" / f"BFCL_v4_{source_category}.json"


BFCL_V4_SCORING_QUESTION_FILES = tuple(
    sorted(
        {
            *(_question_file(category) for category in _DIRECT_QUESTION_CATEGORIES),
            _question_file("memory"),
            _question_file("web_search"),
        },
        key=str,
    )
)
BFCL_V4_PUBLIC_ANSWER_FILES = tuple(
    sorted(
        {
            *(
                _answer_file(category)
                for category in _DIRECT_QUESTION_CATEGORIES
                if category not in _NO_ANSWER_CATEGORIES
            ),
            _answer_file("memory"),
            _answer_file("web_search"),
        },
        key=str,
    )
)
BFCL_V4_RELEASE_METADATA_FILES = (
    PurePosixPath("LICENSE"),
    PurePosixPath("berkeley-function-call-leaderboard/pyproject.toml"),
)

# The official evaluator implements this hierarchy:
# non-live 10%, live 10%, irrelevance 10%, multi-turn 30%, agentic 40%.
# Non-live and agentic contain additional official macro-averaging layers.
_LIVE_AST_TOTAL = 258 + 1053 + 16 + 24
BFCL_V4_OFFICIAL_METRIC_TERMS = (
    BfclV4MetricTerm(category="simple_python", numerator=1, denominator=120),
    BfclV4MetricTerm(category="simple_java", numerator=1, denominator=120),
    BfclV4MetricTerm(category="simple_javascript", numerator=1, denominator=120),
    BfclV4MetricTerm(category="multiple", numerator=1, denominator=40),
    BfclV4MetricTerm(category="parallel", numerator=1, denominator=40),
    BfclV4MetricTerm(category="parallel_multiple", numerator=1, denominator=40),
    BfclV4MetricTerm(category="irrelevance", numerator=1, denominator=20),
    BfclV4MetricTerm(category="live_simple", numerator=258, denominator=10 * _LIVE_AST_TOTAL),
    BfclV4MetricTerm(category="live_multiple", numerator=1053, denominator=10 * _LIVE_AST_TOTAL),
    BfclV4MetricTerm(category="live_parallel", numerator=16, denominator=10 * _LIVE_AST_TOTAL),
    BfclV4MetricTerm(
        category="live_parallel_multiple", numerator=24, denominator=10 * _LIVE_AST_TOTAL
    ),
    BfclV4MetricTerm(category="live_irrelevance", numerator=1, denominator=20),
    BfclV4MetricTerm(category="multi_turn_base", numerator=3, denominator=40),
    BfclV4MetricTerm(category="multi_turn_miss_func", numerator=3, denominator=40),
    BfclV4MetricTerm(category="multi_turn_miss_param", numerator=3, denominator=40),
    BfclV4MetricTerm(category="multi_turn_long_context", numerator=3, denominator=40),
    BfclV4MetricTerm(category="memory_kv", numerator=1, denominator=15),
    BfclV4MetricTerm(category="memory_vector", numerator=1, denominator=15),
    BfclV4MetricTerm(category="memory_rec_sum", numerator=1, denominator=15),
    BfclV4MetricTerm(category="web_search_base", numerator=1, denominator=10),
    BfclV4MetricTerm(category="web_search_no_snippet", numerator=1, denominator=10),
)
BFCL_V4_OFFICIAL_METRIC_FINGERPRINT = canonical_sha256(
    {
        "metric": "BFCL V4 Overall Acc",
        "terms": BFCL_V4_OFFICIAL_METRIC_TERMS,
        "unweighted_display_only_categories": ("live_relevance",),
    }
)

# Filled from a clean ``git archive`` of the byte-exact upstream commit named
# above.  These bind every package source/data file, not merely the files
# currently imported by a smoke.  Interpreter-generated bytecode below a
# ``__pycache__`` directory is deliberately outside this source identity.
_EXPECTED_PACKAGE_TREE = FileSetIdentity(
    file_count=183,
    total_bytes=13_513_439,
    sha256="3753addd78c10a6e59e3488ffdc5fb38cb46929380925ccfaecfcdb1d8b533b2",
)
_EXPECTED_QUESTION_FILES = FileSetIdentity(
    file_count=19,
    total_bytes=9_575_509,
    sha256="171c3674654690745df5aa8b30cd7a95165fe912d4b346dabb9b7c350d0bd073",
)
_EXPECTED_ANSWER_FILES = FileSetIdentity(
    file_count=16,
    total_bytes=978_870,
    sha256="4a1b6bd8c4e6b5d9a219b8c9a2fb2e458d6f028d0571b0fc0b0e84e6d83a69ab",
)
_EXPECTED_RELEASE_METADATA = FileSetIdentity(
    file_count=2,
    total_bytes=13_370,
    sha256="c95e4aecf67fe4c720f1b1a089ecf6864c22588f5e534d31a6fe26c9a82c0666",
)
_EXPECTED_ROSTER_SHA256 = "58b01b959c039d50a062b304ee824bbde64c432feaf4932b026c76087f040366"


def _validate_relative_path(path: PurePosixPath) -> None:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BfclV4SnapshotIntegrityError(f"unsafe BFCL snapshot path: {path}")


def _regular_file(checkout_root: Path, relative_path: PurePosixPath) -> Path:
    _validate_relative_path(relative_path)
    root = checkout_root.resolve(strict=True)
    candidate = root.joinpath(*relative_path.parts)
    if candidate.is_symlink():
        raise BfclV4SnapshotIntegrityError(f"BFCL snapshot path is a symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defense after path validation
        raise BfclV4SnapshotIntegrityError(
            f"BFCL snapshot path escapes checkout: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise BfclV4SnapshotIntegrityError(f"BFCL snapshot path is not a file: {relative_path}")
    return resolved


def _file_set_identity(
    checkout_root: Path,
    relative_paths: Iterable[PurePosixPath],
) -> FileSetIdentity:
    paths = tuple(sorted(relative_paths, key=str))
    if not paths or len(paths) != len(set(paths)):
        raise BfclV4SnapshotIntegrityError("BFCL file set must be non-empty and unique")
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path in paths:
        content = _regular_file(checkout_root, relative_path).read_bytes()
        total_bytes += len(content)
        leaf = hashlib.sha256(content).hexdigest()
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(leaf.encode("ascii"))
        digest.update(b"\n")
    return FileSetIdentity(
        file_count=len(paths),
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def _is_ignored_python_cache_file(relative_path: PurePosixPath) -> bool:
    """Recognize only interpreter bytecode in a conventional cache directory."""

    return (
        "__pycache__" in relative_path.parts
        and relative_path.suffix in _IGNORED_PYTHON_CACHE_SUFFIXES
    )


def _tree_paths(checkout_root: Path, relative_root: PurePosixPath) -> tuple[PurePosixPath, ...]:
    _validate_relative_path(relative_root)
    root = checkout_root.resolve(strict=True)
    tree = root.joinpath(*relative_root.parts)
    if tree.is_symlink() or not tree.is_dir():
        raise BfclV4SnapshotIntegrityError("BFCL package root is missing or is a symlink")
    paths: list[PurePosixPath] = []
    for candidate in tree.rglob("*"):
        relative = PurePosixPath(candidate.relative_to(root).as_posix())
        if candidate.is_symlink():
            raise BfclV4SnapshotIntegrityError(f"BFCL package tree contains symlink: {relative}")
        if candidate.is_file() and not _is_ignored_python_cache_file(relative):
            paths.append(relative)
    return tuple(sorted(paths, key=str))


def _load_jsonl_ids(checkout_root: Path, path: PurePosixPath) -> tuple[str, ...]:
    ids: list[str] = []
    with _regular_file(checkout_root, path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise BfclV4SnapshotIntegrityError(f"blank JSONL line in {path}:{line_number}")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise BfclV4SnapshotIntegrityError(
                    f"invalid JSONL in {path}:{line_number}"
                ) from exc
            row_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(row_id, str) or not row_id or row_id != row_id.strip():
                raise BfclV4SnapshotIntegrityError(f"invalid BFCL ID in {path}:{line_number}")
            ids.append(row_id)
    if len(ids) != len(set(ids)):
        raise BfclV4SnapshotIntegrityError(f"duplicate physical BFCL IDs in {path}")
    return tuple(ids)


def _source_category(category: str) -> str:
    return _SHARED_QUESTION_SOURCES.get(category, category)


def _effective_id(category: str, source_id: str) -> str:
    source = _source_category(category)
    if not source_id.startswith(source + "_"):
        raise BfclV4SnapshotIntegrityError(
            f"BFCL source ID {source_id!r} is outside category {source!r}"
        )
    return category + source_id[len(source) :]


def _roster_identity(checkout_root: Path) -> BfclV4RosterIdentity:
    effective_rows: list[dict[str, str]] = []
    actual_counts: list[BfclV4CategoryCount] = []
    for expected in BFCL_V4_CATEGORY_COUNTS:
        category = expected.category
        source = _source_category(category)
        question_ids = _load_jsonl_ids(checkout_root, _question_file(source))
        effective_ids = tuple(_effective_id(category, item) for item in question_ids)
        if len(effective_ids) != expected.count:
            raise BfclV4SnapshotIntegrityError(
                f"BFCL {category} count is {len(effective_ids)}, expected {expected.count}"
            )
        if category not in _NO_ANSWER_CATEGORIES:
            answer_ids = _load_jsonl_ids(checkout_root, _answer_file(source))
            if answer_ids != question_ids:
                raise BfclV4SnapshotIntegrityError(
                    f"BFCL {category} question and answer IDs are not byte-order aligned"
                )
        actual_counts.append(BfclV4CategoryCount(category=category, count=len(effective_ids)))
        effective_rows.extend(
            {"category": category, "evaluation_unit_id": item} for item in effective_ids
        )
    effective_keys = tuple((row["category"], row["evaluation_unit_id"]) for row in effective_rows)
    if len(effective_keys) != len(set(effective_keys)):
        raise BfclV4SnapshotIntegrityError("expanded BFCL all_scoring IDs are not unique")
    return BfclV4RosterIdentity(
        categories=tuple(actual_counts),
        roster_sha256=canonical_sha256(tuple(effective_rows)),
    )


def bfcl_v4_official_overall(category_accuracies: Mapping[str, float]) -> float:
    """Reproduce the official full-suite Overall Acc aggregation.

    All 22 scoring categories are required even though ``live_relevance`` is a
    displayed diagnostic with zero coefficient in Overall Acc.  Requiring it
    prevents a partial evaluation from being mislabeled as an official run.
    """

    supplied = frozenset(category_accuracies)
    expected = frozenset(BFCL_V4_ALL_SCORING_CATEGORIES)
    if supplied != expected:
        missing = ", ".join(sorted(expected - supplied)) or "none"
        extra = ", ".join(sorted(supplied - expected)) or "none"
        raise BfclV4MetricError(
            f"BFCL full-suite categories differ; missing={missing}; extra={extra}"
        )
    checked: dict[str, float] = {}
    for category, raw_value in category_accuracies.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise BfclV4MetricError(f"BFCL category accuracy is not numeric: {category}")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise BfclV4MetricError(f"BFCL category accuracy is outside [0, 1]: {category}")
        checked[category] = value
    return math.fsum(
        float(term.weight) * checked[term.category] for term in BFCL_V4_OFFICIAL_METRIC_TERMS
    )


def verify_bfcl_v4_checkout(checkout_root: Path) -> BfclV4SnapshotVerification:
    """Fail closed unless a checkout matches the complete pinned public suite."""

    if not isinstance(checkout_root, Path):
        raise TypeError("checkout_root must be a pathlib.Path")
    package_tree = _file_set_identity(
        checkout_root,
        _tree_paths(checkout_root, BFCL_V4_PACKAGE_ROOT),
    )
    questions = _file_set_identity(checkout_root, BFCL_V4_SCORING_QUESTION_FILES)
    answers = _file_set_identity(checkout_root, BFCL_V4_PUBLIC_ANSWER_FILES)
    metadata = _file_set_identity(checkout_root, BFCL_V4_RELEASE_METADATA_FILES)
    roster = _roster_identity(checkout_root)
    observed = {
        "package tree": (package_tree, _EXPECTED_PACKAGE_TREE),
        "scoring questions": (questions, _EXPECTED_QUESTION_FILES),
        "public answers": (answers, _EXPECTED_ANSWER_FILES),
        "release metadata": (metadata, _EXPECTED_RELEASE_METADATA),
    }
    mismatches = tuple(name for name, (actual, expected) in observed.items() if actual != expected)
    if roster.roster_sha256 != _EXPECTED_ROSTER_SHA256:
        mismatches = (*mismatches, "expanded all_scoring roster")
    if mismatches:
        raise BfclV4SnapshotIntegrityError(
            "BFCL V4 checkout differs from pinned 6ea57973 snapshot: " + ", ".join(mismatches)
        )
    return BfclV4SnapshotVerification(
        package_tree=package_tree,
        scoring_question_files=questions,
        public_answer_files=answers,
        release_metadata_files=metadata,
        roster=roster,
        official_metric_fingerprint=BFCL_V4_OFFICIAL_METRIC_FINGERPRINT,
    )


__all__ = [
    "BFCL_V4_ALL_SCORING_CATEGORIES",
    "BFCL_V4_CATEGORY_COUNTS",
    "BFCL_V4_EFFECTIVE_MAX_MODEL_STEPS_PER_TURN",
    "BFCL_V4_OFFICIAL_METRIC_FINGERPRINT",
    "BFCL_V4_OFFICIAL_METRIC_TERMS",
    "BFCL_V4_RELEASE_VERSION",
    "BFCL_V4_UPSTREAM_COMMIT",
    "BFCL_V4_UPSTREAM_REPOSITORY",
    "BfclV4MetricError",
    "BfclV4SnapshotIntegrityError",
    "BfclV4SnapshotVerification",
    "bfcl_v4_official_overall",
    "verify_bfcl_v4_checkout",
]
