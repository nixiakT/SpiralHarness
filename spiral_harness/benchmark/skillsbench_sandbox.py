"""Rootless bubblewrap execution for pinned SkillsBench task smoke runs."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PYTHON_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class SkillsBenchSandboxError(RuntimeError):
    """A candidate or verifier could not run inside the frozen sandbox."""


@dataclass(frozen=True, slots=True)
class SkillsBenchSandboxResult:
    candidate_exit_code: int
    verifier_exit_code: int
    verifier_output: str

    @property
    def passed(self) -> bool:
        return self.candidate_exit_code == 0 and self.verifier_exit_code == 0


def extract_python_candidate(text: str) -> str:
    """Require exactly one non-empty Python fenced block from an untrusted model."""

    if type(text) is not str:
        raise TypeError("candidate response must be a string")
    matches = _PYTHON_FENCE_RE.findall(text)
    if len(matches) != 1 or not matches[0].strip():
        raise SkillsBenchSandboxError("candidate response must contain exactly one Python block")
    return matches[0].strip() + "\n"


def run_skillsbench_task(
    app_dir: Path,
    *,
    python_executable: Path,
    timeout_seconds: float = 120.0,
) -> SkillsBenchSandboxResult:
    """Execute one prepared `/app` with no network and a read-only host view."""

    app_dir = Path(app_dir).resolve(strict=True)
    python_executable = Path(python_executable).absolute()
    if not python_executable.exists():
        raise FileNotFoundError(python_executable)
    resolved_python = python_executable.resolve(strict=True)
    virtualenv_root = python_executable.parent.parent
    if not (virtualenv_root / "pyvenv.cfg").is_file():
        raise SkillsBenchSandboxError("python_executable must be a virtualenv entry point")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SkillsBenchSandboxError("bubblewrap is required for SkillsBench execution")
    common = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--die-with-parent",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        str(resolved_python.parents[2]),
        str(resolved_python.parents[2]),
        "--ro-bind",
        str(virtualenv_root),
        "/runtime-venv",
        "--bind",
        str(app_dir),
        "/app",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/app",
    ]

    def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [*common, str(resolved_python), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONHASHSEED": "0",
                    "PYTHONPATH": "/runtime-venv/lib/python3.12/site-packages",
                },
            )
        except subprocess.TimeoutExpired as error:
            raise SkillsBenchSandboxError("SkillsBench sandbox timed out") from error

    candidate = invoke(["/app/solution.py"])
    verifier = invoke(["-m", "pytest", "-q", "/app/tests/test_outputs.py"])
    output = (verifier.stdout + "\n" + verifier.stderr).strip()
    return SkillsBenchSandboxResult(
        candidate_exit_code=candidate.returncode,
        verifier_exit_code=verifier.returncode,
        verifier_output=output[-16_000:],
    )


__all__ = [
    "SkillsBenchSandboxError",
    "SkillsBenchSandboxResult",
    "extract_python_candidate",
    "run_skillsbench_task",
]
