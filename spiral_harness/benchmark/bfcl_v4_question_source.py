"""Answer-free, network-disabled reads of pinned BFCL question Git objects."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from pydantic import Field, field_validator

from spiral_harness.benchmark.bfcl_v4_identity import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT = BFCL_V4_UPSTREAM_COMMIT
_QUESTION_DATA_ROOT = PurePosixPath("berkeley-function-call-leaderboard/bfcl_eval/data")
_SOURCE_FAMILY_DOMAIN = "spiral-bfcl-v4-public-development-v2-source-family/v1"
_SEMANTIC_TEMPLATE_DOMAIN = "spiral-bfcl-v4-public-development-v2-semantic-template/v1"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C",
    "LC_ALL": "C",
}


class BfclV4QuestionSourceError(RuntimeError):
    """A local pinned-question source operation failed closed."""


class BfclV4QuestionSourceBinding(ImmutableModel):
    """Observed identity of one direct BFCL question JSONL Git blob."""

    git_path: NonEmptyStr
    size: Annotated[int, Field(gt=0, strict=True)]
    sha256: Sha256

    @field_validator("git_path")
    @classmethod
    def _path_is_direct_question_json(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.parent != _QUESTION_DATA_ROOT
            or not parsed.name.startswith("BFCL_v4_")
            or parsed.suffix != ".json"
            or parsed.as_posix() != value
        ):
            raise ValueError("question source path must be a direct canonical BFCL data JSON file")
        return value


@dataclass(frozen=True, slots=True)
class BfclV4QuestionCheckout:
    """Verified standalone checkout and absolute system Git executable."""

    root: Path
    git_executable: Path


@dataclass(frozen=True, slots=True)
class BfclV4QuestionCandidate:
    """One parsed question/schema row with question-only family coordinates."""

    category: str
    task_id: str
    value: dict[str, Any]
    row: bytes
    official_function_names: tuple[str, ...]
    semantic_tokens: tuple[str, ...]
    source_family_sha256: str
    semantic_template_sha256: str

    @property
    def row_sha256(self) -> str:
        return sha256_bytes(self.row)

    @property
    def candidate_payload_sha256(self) -> str:
        return canonical_sha256(self.value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must not contain surrogate code points")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)


def strict_bfcl_v4_question_json(content: bytes, label: str) -> Any:
    """Decode strict UTF-8 JSON with duplicate and non-finite rejection."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_constant,
        )
        _reject_surrogates(value)
        canonical_json_bytes(value)
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise BfclV4QuestionSourceError(f"{label} is not strict UTF-8 JSON") from error
    return value


def _user_text(question: object) -> str:
    contents: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("role") == "user" and isinstance(value.get("content"), str):
                contents.append(value["content"])
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(question)
    text = " ".join(contents)
    if not text:
        raise BfclV4QuestionSourceError("question has no auditable user text")
    return text


def _semantic_tokens(question: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", _user_text(question)).casefold()
    tokens = tuple(
        "<num>" if any(character.isdigit() for character in token) else token
        for token in _TOKEN_PATTERN.findall(normalized)
    )
    if not tokens:
        raise BfclV4QuestionSourceError("question has no normalized semantic tokens")
    return tokens


def parse_bfcl_v4_question_blob(
    category: str,
    content: bytes,
    *,
    expected_row_count: int,
) -> tuple[BfclV4QuestionCandidate, ...]:
    """Parse one exact-schema LF-delimited question blob."""

    candidates: list[BfclV4QuestionCandidate] = []
    seen: set[str] = set()
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.endswith(b"\r\n") or (not line.endswith(b"\n") and index != len(lines) - 1):
            raise BfclV4QuestionSourceError("question source is not canonical LF JSONL")
        payload = line[:-1] if line.endswith(b"\n") else line
        value = strict_bfcl_v4_question_json(payload, f"BFCL v2 {category} question source")
        if not isinstance(value, dict) or set(value) != {"id", "question", "function"}:
            raise BfclV4QuestionSourceError("question row schema changed")
        task_id, functions = value["id"], value["function"]
        if (
            not isinstance(task_id, str)
            or not task_id.startswith(f"{category}_")
            or task_id in seen
            or not isinstance(functions, list)
            or not functions
        ):
            raise BfclV4QuestionSourceError("question row identity or functions are invalid")
        names = tuple(item.get("name") for item in functions if isinstance(item, dict))
        if (
            len(names) != len(functions)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise BfclV4QuestionSourceError("question row function names are invalid")
        tokens = _semantic_tokens(value["question"])
        seen.add(task_id)
        candidates.append(
            BfclV4QuestionCandidate(
                category=category,
                task_id=task_id,
                value=value,
                row=line,
                official_function_names=names,
                semantic_tokens=tokens,
                source_family_sha256=canonical_sha256(
                    {
                        "domain": _SOURCE_FAMILY_DOMAIN,
                        "official_function_names": tuple(sorted(names)),
                    }
                ),
                semantic_template_sha256=canonical_sha256(
                    {
                        "domain": _SEMANTIC_TEMPLATE_DOMAIN,
                        "normalized_question_tokens": tokens,
                    }
                ),
            )
        )
    if len(candidates) != expected_row_count:
        raise BfclV4QuestionSourceError("question source row count changed")
    return tuple(candidates)


def _resolve_system_git() -> Path:
    candidate = shutil.which("git", path=os.defpath)
    if candidate is None:
        raise BfclV4QuestionSourceError("system Git executable unavailable")
    try:
        resolved = Path(candidate).resolve(strict=True)
    except OSError as error:
        raise BfclV4QuestionSourceError("system Git executable unavailable") from error
    if not resolved.is_file():
        raise BfclV4QuestionSourceError("system Git executable is not a file")
    return resolved


def _run_git(git: Path, checkout: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            (str(git), "-C", str(checkout), *arguments),
            env=dict(_GIT_ENVIRONMENT),
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BfclV4QuestionSourceError("pinned question Git object read failed") from error
    if completed.returncode != 0:
        raise BfclV4QuestionSourceError(
            "pinned question Git object read failed "
            f"(stderr_sha256={sha256_bytes(completed.stderr)})"
        )
    return completed.stdout


def open_bfcl_v4_question_checkout(checkout: str | Path) -> BfclV4QuestionCheckout:
    """Verify an exact standalone repository root at the pinned commit."""

    raw = Path(checkout)
    absolute = Path(os.path.abspath(raw))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise BfclV4QuestionSourceError("BFCL question checkout is unavailable") from error
    if resolved != absolute or not resolved.is_dir():
        raise BfclV4QuestionSourceError("BFCL question checkout must be a real directory")
    git_directory = resolved / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise BfclV4QuestionSourceError("BFCL question checkout must be a standalone checkout")

    git = _resolve_system_git()
    try:
        top_level_text = _run_git(git, resolved, "rev-parse", "--show-toplevel").decode("utf-8")
        head = _run_git(git, resolved, "rev-parse", "HEAD").decode("ascii").strip()
        top_level = Path(top_level_text.strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as error:
        raise BfclV4QuestionSourceError("BFCL question checkout metadata is invalid") from error
    if top_level != resolved or head != BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT:
        raise BfclV4QuestionSourceError("BFCL question checkout differs from the pinned root")
    return BfclV4QuestionCheckout(root=resolved, git_executable=git)


def read_bfcl_v4_question_blob(
    checkout: BfclV4QuestionCheckout,
    git_path: str,
) -> tuple[bytes, BfclV4QuestionSourceBinding]:
    """Read one direct question path from the pinned commit, never the worktree."""

    path_probe = BfclV4QuestionSourceBinding(git_path=git_path, size=1, sha256="0" * 64)
    content = _run_git(
        checkout.git_executable,
        checkout.root,
        "cat-file",
        "blob",
        f"{BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT}:{path_probe.git_path}",
    )
    return content, BfclV4QuestionSourceBinding(
        git_path=path_probe.git_path,
        size=len(content),
        sha256=sha256_bytes(content),
    )


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
