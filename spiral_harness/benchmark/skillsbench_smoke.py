"""Evidence-bound, native-compatible SkillsBench smoke orchestration."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from spiral_harness.benchmark.gsm8k_smoke import build_live_gsm8k_spec
from spiral_harness.benchmark.skillsbench_sandbox import (
    extract_python_candidate,
    run_skillsbench_task,
)
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE
from spiral_harness.execution.contracts import ModelRequest, ResolvedHarness
from spiral_harness.execution.model import ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

SKILLSBENCH_REVISION = "828bb921fb94dc065bfefd6bac4e8938be3f71e0"
SKILLSBENCH_SMOKE_MEDIA_TYPE = "application/vnd.spiral-harness.skillsbench-native-smoke.v1+json"
DEFAULT_SKILLSBENCH_SYSTEM_PROMPT = (
    "Implement the requested task completely. You may use Python's standard library and files "
    "described in the task. Return exactly one fenced Python code block containing solution.py, "
    "with no other fenced blocks. The program must create all requested outputs when executed."
)


@dataclass(frozen=True, slots=True)
class SkillsBenchSmokeResult:
    payload: dict[str, object]
    artifact_sha256: str


def _git_blob(repo: Path, revision: str, path: str) -> bytes:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def run_dialogue_parser_smoke(
    *,
    skillsbench_repo: Path,
    output: Path,
    backend: ModelBackend,
    model: str,
    skill_text: str | None = None,
    max_output_tokens: int = 8_192,
    timeout_seconds: float = 180.0,
) -> SkillsBenchSmokeResult:
    """Generate and verify the pinned dialogue-parser task without exposing its verifier."""

    repo = Path(skillsbench_repo).resolve(strict=True)
    output = Path(output).resolve()
    store = ArtifactStore(output / "artifacts")
    prefix = "tasks-no-skills/dialogue-parser"
    sources = {
        "instruction.md": _git_blob(repo, SKILLSBENCH_REVISION, f"{prefix}/instruction.md"),
        "script.txt": _git_blob(repo, SKILLSBENCH_REVISION, f"{prefix}/environment/script.txt"),
        "tests/test_outputs.py": _git_blob(
            repo, SKILLSBENCH_REVISION, f"{prefix}/tests/test_outputs.py"
        ),
        "dialogue_graph.py": _git_blob(
            repo,
            SKILLSBENCH_REVISION,
            f"{prefix}/environment/skills/dialogue_graph/scripts/dialogue_graph.py",
        ),
    }
    instruction = sources["instruction.md"].decode("utf-8")
    script = sources["script.txt"].decode("utf-8")
    user_prompt = (
        instruction + "\n\nThe exact contents of /app/script.txt are:\n```text\n" + script + "\n```"
    )
    system_prompt = DEFAULT_SKILLSBENCH_SYSTEM_PROMPT
    if skill_text is not None:
        system_prompt += "\n\nCandidate harness guidance:\n" + skill_text

    harness_ref = store.put_json(
        {
            "schema_version": "1",
            "kind": "skillsbench-smoke-prompt",
            "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
        },
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    harness = ResolvedHarness.from_prompt(harness_ref=harness_ref, system_prompt=system_prompt)

    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    response = backend.invoke(
        spec=spec,
        request=ModelRequest(
            task_id="skillsbench/dialogue-parser",
            harness_ref=harness.harness_ref,
            base_system_prompt=harness.base_system_prompt,
            base_system_prompt_sha256=harness.base_system_prompt_sha256,
            system_prompt=harness.system_prompt,
            resolved_prompt_sha256=harness.resolved_prompt_sha256,
            user_prompt=user_prompt,
            seed=0,
        ),
    )
    raw_ref = store.put_bytes(response.output.encode(), media_type="text/plain; charset=utf-8")
    candidate = extract_python_candidate(response.output)
    app = output / "app"
    if app.exists():
        shutil.rmtree(app)
    (app / "tests").mkdir(parents=True)
    (app / "instruction.md").write_bytes(sources["instruction.md"])
    (app / "script.txt").write_bytes(sources["script.txt"])
    (app / "tests" / "test_outputs.py").write_bytes(sources["tests/test_outputs.py"])
    (app / "dialogue_graph.py").write_bytes(sources["dialogue_graph.py"])
    (app / "solution.py").write_text(candidate, encoding="utf-8")
    sandbox = run_skillsbench_task(
        app, python_executable=Path(sys.prefix) / "bin" / "python", timeout_seconds=120
    )
    payload: dict[str, object] = {
        "schema_version": "1",
        "benchmark": "SkillsBench",
        "task": "dialogue-parser",
        "revision": SKILLSBENCH_REVISION,
        "protocol": "rootless-bubblewrap-native-compatible-smoke",
        "reportable_as_official": False,
        "model": model,
        "harness_variant": "baseline" if skill_text is None else "candidate",
        "harness_sha256": sha256_bytes(system_prompt.encode()),
        "source_sha256": {name: sha256_bytes(data) for name, data in sources.items()},
        "raw_response_ref": raw_ref.model_dump(mode="json"),
        "candidate_sha256": sha256_bytes(candidate.encode()),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "candidate_exit_code": sandbox.candidate_exit_code,
        "verifier_exit_code": sandbox.verifier_exit_code,
        "passed": sandbox.passed,
        "verifier_output": sandbox.verifier_output,
        "disclaimer": (
            "Single-task smoke under a native-compatible rootless protocol; "
            "not an official leaderboard score."
        ),
    }
    ref = store.put_json(payload, media_type=SKILLSBENCH_SMOKE_MEDIA_TYPE)
    return SkillsBenchSmokeResult(payload=payload, artifact_sha256=ref.sha256)


__all__ = [
    "DEFAULT_SKILLSBENCH_SYSTEM_PROMPT",
    "SKILLSBENCH_REVISION",
    "SkillsBenchSmokeResult",
    "run_dialogue_parser_smoke",
]
