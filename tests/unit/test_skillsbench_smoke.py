from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spiral_harness.benchmark.skillsbench_smoke import _git_blob

ROOT = Path(__file__).parents[2]


def test_git_blob_reads_exact_pinned_bytes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (tmp_path / "task.txt").write_bytes(b"exact\nbytes\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "task.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()

    assert _git_blob(tmp_path, revision, "task.txt") == b"exact\nbytes\n"


def test_git_blob_rejects_missing_revision(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _git_blob(tmp_path, "0" * 40, "missing")


def test_dialogue_parser_candidate_skill_is_task_general_and_verifier_free() -> None:
    text = (ROOT / "benchmarks/skillsbench/dialogue-parser-v1.skill.txt").read_text(
        encoding="utf-8"
    )

    assert "Never create an extra node" in text
    assert "parse_script(text)" in text
    assert "test_outputs" not in text
    assert "TavernChoice" not in text
