from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from spiral_harness.benchmark.skillsbench_sandbox import (
    SkillsBenchSandboxError,
    extract_python_candidate,
)


def _bubblewrap_can_create_required_namespaces() -> bool:
    bwrap = Path("/usr/bin/bwrap")
    if not bwrap.exists():
        return False
    probe = subprocess.run(
        [
            str(bwrap),
            "--unshare-user",
            "--unshare-net",
            "--die-with-parent",
            "--ro-bind",
            "/usr",
            "/usr",
            "/usr/bin/true",
        ],
        check=False,
        capture_output=True,
        timeout=5,
    )
    return probe.returncode == 0


def test_extract_python_candidate_accepts_one_fenced_block() -> None:
    assert extract_python_candidate("```python\nprint('ok')\n```") == "print('ok')\n"


@pytest.mark.parametrize(
    "response",
    ["print('no fence')", "```python\n\n```", "```python\na=1\n```\n```python\nb=2\n```"],
)
def test_extract_python_candidate_rejects_ambiguous_responses(response: str) -> None:
    with pytest.raises(SkillsBenchSandboxError):
        extract_python_candidate(response)


def test_module_does_not_accept_a_relative_python_path(tmp_path: Path) -> None:
    from spiral_harness.benchmark.skillsbench_sandbox import run_skillsbench_task

    with pytest.raises(FileNotFoundError):
        run_skillsbench_task(tmp_path, python_executable=Path("missing-python"))


@pytest.mark.skipif(
    not _bubblewrap_can_create_required_namespaces(),
    reason="bubblewrap namespaces are unavailable",
)
def test_bubblewrap_runs_candidate_and_verifier_in_isolated_app(tmp_path: Path) -> None:
    from spiral_harness.benchmark.skillsbench_sandbox import run_skillsbench_task

    app = tmp_path / "app"
    tests = app / "tests"
    tests.mkdir(parents=True)
    (app / "solution.py").write_text(
        "from pathlib import Path\n"
        "Path('/app/result.txt').write_text('ok')\n"
        "Path('/tmp/allowed.txt').write_text('ok')\n",
        encoding="utf-8",
    )
    (tests / "test_outputs.py").write_text(
        "from pathlib import Path\n"
        "def test_result():\n"
        "    assert Path('/app/result.txt').read_text() == 'ok'\n"
        "    assert not Path('/tmp/allowed.txt').exists()\n",
        encoding="utf-8",
    )

    result = run_skillsbench_task(app, python_executable=Path(sys.prefix) / "bin" / "python")

    assert result.passed is True
    assert (app / "result.txt").read_text(encoding="utf-8") == "ok"
