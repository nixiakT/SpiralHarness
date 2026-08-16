"""Development-only SWE-bench Verified runner.

This module intentionally targets a small, local, reproducible slice rather
than claiming official containerized leaderboard execution.  It is designed to
turn the repository's prior SWE audit into an executable development harness.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{", re.S)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_LABELED_PATCH_PLAN_RE = re.compile(
    r"`?file_path`?\s*:\s*`(?P<file_path>[^`\n]+)`"
    r".*?"
    r"`?before`?\s*:\s*```(?:[A-Za-z0-9_+-]+)?\n(?P<before>.*?)```"
    r".*?"
    r"`?after`?\s*:\s*```(?:[A-Za-z0-9_+-]+)?\n(?P<after>.*?)```",
    re.S,
)
_EXCERPT_CONTEXT_LINES = 12
_EXCERPT_MAX_LINES = 48
_EXCERPT_MAX_SEGMENTS_PER_FILE = 2
_EXCERPT_MAX_SEGMENTS_TOTAL = 4


class SwebenchVerifiedDevError(RuntimeError):
    """A local SWE-bench Verified setup, generation, or evaluation step failed."""


@dataclass(frozen=True, slots=True)
class SwebenchVerifiedTask:
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    difficulty: str


@dataclass(frozen=True, slots=True)
class SwebenchPatchPlan:
    file_path: str
    before: str
    after: str
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class _RepoProfile:
    python_path: str
    requirements_files: tuple[str, ...]
    editable_install: bool
    extra_constraints: tuple[str, ...]
    pass_to_pass_sample_limit: int


_LOCAL_PROFILES = {
    "pallets/flask": _RepoProfile(
        python_path="/usr/bin/python3",
        requirements_files=("requirements/tests.txt",),
        editable_install=True,
        extra_constraints=("Werkzeug<3", "Jinja2<4", "itsdangerous<3", "click<9"),
        pass_to_pass_sample_limit=8,
    )
}


def _datasets_dataset_type() -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - exercised in CLI integration only
        raise SwebenchVerifiedDevError(
            "SWE-bench dev runner requires the optional datasets package"
        ) from exc
    return Dataset


def _load_cached_verified_dataset() -> Any:
    dataset_type = _datasets_dataset_type()
    path = Path.home() / ".cache" / "huggingface" / "datasets" / (
        "princeton-nlp___swe-bench_verified/default/0.0.0/"
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/swe-bench_verified-test.arrow"
    )
    if path.is_file():
        return dataset_type.from_file(str(path))
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SwebenchVerifiedDevError("datasets package unavailable and no cached arrow found") from exc
    try:
        return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    except Exception as exc:  # pragma: no cover - network dependent
        raise SwebenchVerifiedDevError("failed to load SWE-bench Verified dataset") from exc


def load_verified_task(instance_id: str) -> SwebenchVerifiedTask:
    dataset = _load_cached_verified_dataset()
    try:
        row = next(row for row in dataset if row["instance_id"] == instance_id)
    except StopIteration as exc:
        raise SwebenchVerifiedDevError(f"unknown SWE-bench Verified instance: {instance_id}") from exc
    fail_to_pass = tuple(json.loads(row["FAIL_TO_PASS"]))
    pass_to_pass = tuple(json.loads(row["PASS_TO_PASS"]))
    return SwebenchVerifiedTask(
        repo=row["repo"],
        instance_id=row["instance_id"],
        base_commit=row["base_commit"],
        patch=row["patch"],
        test_patch=row["test_patch"],
        problem_statement=row["problem_statement"],
        hints_text=row["hints_text"],
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        difficulty=row["difficulty"],
    )


def extract_patch_plan(text: str) -> SwebenchPatchPlan:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    decoder = json.JSONDecoder()
    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        file_path = payload.get("file_path")
        before = payload.get("before")
        after = payload.get("after")
        explanation = payload.get("explanation")
        if (
            isinstance(file_path, str)
            and isinstance(before, str)
            and isinstance(after, str)
            and (explanation is None or isinstance(explanation, str))
        ):
            return SwebenchPatchPlan(
                file_path=file_path,
                before=before,
                after=after,
                explanation=explanation,
            )
    labeled_match = _LABELED_PATCH_PLAN_RE.search(text)
    if labeled_match is not None:
        return SwebenchPatchPlan(
            file_path=labeled_match.group("file_path").strip(),
            before=labeled_match.group("before"),
            after=labeled_match.group("after"),
            explanation=None,
        )
    raise SwebenchVerifiedDevError("model output did not contain a valid patch-plan JSON object")


def _ensure_uv() -> str:
    for candidate in (shutil.which("uv"), str(Path.home() / ".local" / "bin" / "uv")):
        if candidate and Path(candidate).exists():
            return candidate
    raise SwebenchVerifiedDevError("uv is required for the local SWE development environment")


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        cwd=None if cwd is None else str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _ensure_repo_cache(task: SwebenchVerifiedTask, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / task.repo.replace("/", "__")
    if cache_dir.exists():
        _git(cache_dir, "fetch", "--all", "--tags")
    else:
        _run(["git", "clone", f"https://github.com/{task.repo}.git", str(cache_dir)])
    return cache_dir


def _materialize_worktree(task: SwebenchVerifiedTask, output: Path) -> Path:
    cache_dir = _ensure_repo_cache(task, output / "repo_cache")
    worktree = output / "worktrees" / task.instance_id
    if worktree.exists():
        shutil.rmtree(worktree)
    _run(["git", "clone", str(cache_dir), str(worktree)])
    _git(worktree, "checkout", task.base_commit)
    return worktree


def _apply_git_patch(repo_dir: Path, patch_text: str, *, label: str) -> None:
    patch_path = repo_dir / f".spiral-{label}.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    _run(["git", "apply", str(patch_path.resolve())], cwd=repo_dir)


def _prepare_env(repo_dir: Path, output: Path, profile: _RepoProfile) -> Path:
    uv = _ensure_uv()
    env_dir = output / "envs" / repo_dir.name
    python_path = Path(profile.python_path)
    if not python_path.exists():
        raise SwebenchVerifiedDevError(f"missing Python interpreter for local profile: {python_path}")
    if env_dir.exists():
        shutil.rmtree(env_dir)
    _run([uv, "venv", "--python", str(python_path), str(env_dir)])
    env_python = env_dir / "bin" / "python"
    for requirement in profile.requirements_files:
        requirement_path = repo_dir / requirement
        if requirement_path.is_file():
            _run([uv, "pip", "install", "--python", str(env_python), "-r", str(requirement_path)])
    if profile.editable_install:
        _run([uv, "pip", "install", "--python", str(env_python), "-e", str(repo_dir)])
    for specifier in profile.extra_constraints:
        _run([uv, "pip", "install", "--python", str(env_python), specifier])
    return Path(os.path.abspath(env_python))


def _pytest(
    env_python: Path,
    repo_dir: Path,
    tests: tuple[str, ...],
) -> dict[str, object]:
    result = subprocess.run(
        [str(env_python), "-m", "pytest", "-q", *tests],
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
    )
    return {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout[-20_000:],
        "stderr": result.stderr[-20_000:],
        "tests": list(tests),
    }


def _keywords(task: SwebenchVerifiedTask) -> tuple[str, ...]:
    words = {
        word
        for word in _WORD_RE.findall(task.problem_statement + "\n" + task.hints_text)
        if len(word) >= 4
    }
    preferred = sorted(words, key=lambda item: (item[0].islower(), -len(item), item))
    return tuple(preferred[:8])


def _excerpt_windows(line_numbers: list[int], total_lines: int) -> tuple[tuple[int, int], ...]:
    windows: list[tuple[int, int]] = []
    for line_no in sorted(set(line_numbers)):
        start = max(line_no - _EXCERPT_CONTEXT_LINES, 1)
        end = min(line_no + _EXCERPT_CONTEXT_LINES, total_lines)
        if windows and start <= windows[-1][1] + 4:
            prior_start, prior_end = windows[-1]
            merged_end = min(max(prior_end, end), prior_start + _EXCERPT_MAX_LINES - 1)
            windows[-1] = (prior_start, merged_end)
            continue
        if len(windows) >= _EXCERPT_MAX_SEGMENTS_PER_FILE:
            break
        windows.append((start, min(end, start + _EXCERPT_MAX_LINES - 1)))
    return tuple(windows)


def _path_priority(path_text: str, line_numbers: list[int], keywords: tuple[str, ...]) -> tuple[int, int, int, str]:
    lower_path = path_text.lower()
    keyword_hits = sum(1 for keyword in keywords if keyword.lower() in lower_path)
    return (-keyword_hits, -len(set(line_numbers)), len(path_text), path_text)


def _collect_excerpts(repo_dir: Path, task: SwebenchVerifiedTask) -> tuple[dict[str, object], ...]:
    candidates: dict[str, list[int]] = {}
    keywords = _keywords(task)
    for keyword in keywords:
        result = subprocess.run(
            ["rg", "-n", "-m", "2", keyword, "src", "tests"],
            cwd=str(repo_dir),
            text=True,
            capture_output=True,
        )
        if result.returncode not in {0, 1}:
            continue
        for line in result.stdout.splitlines():
            path_text, line_no_text, *_ = line.split(":", 2)
            try:
                line_no = int(line_no_text)
            except ValueError:
                continue
            candidates.setdefault(path_text, []).append(line_no)
    excerpts: list[dict[str, object]] = []
    for path_text, line_numbers in sorted(
        candidates.items(),
        key=lambda item: _path_priority(item[0], item[1], keywords),
    ):
        if len(excerpts) >= _EXCERPT_MAX_SEGMENTS_TOTAL:
            break
        path = repo_dir / path_text
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for start, end in _excerpt_windows(line_numbers, len(lines)):
            excerpts.append(
                {
                    "path": path_text,
                    "start_line": start,
                    "end_line": end,
                    "text": "\n".join(lines[start - 1 : end]),
                }
            )
            if len(excerpts) >= _EXCERPT_MAX_SEGMENTS_TOTAL:
                break
    if not excerpts:
        raise SwebenchVerifiedDevError("failed to find any relevant repository excerpts")
    return tuple(excerpts)


def _patch_plan_prompt(task: SwebenchVerifiedTask, excerpts: tuple[dict[str, object], ...]) -> str:
    rendered_excerpts = "\n\n".join(
        f"[FILE {item['path']}:{item['start_line']}-{item['end_line']}]\n{item['text']}"
        for item in excerpts
    )
    return (
        "You are fixing a SWE-bench Verified instance.\n"
        "Return exactly one JSON object with keys file_path, before, after, explanation.\n"
        "The before snippet must be copied exactly from one provided excerpt, and the after snippet "
        "must be the replacement text. Prefer the smallest self-contained replacement inside a "
        "function body. Do not replace a partial function signature, unmatched parentheses, or any "
        "truncated block. If an existing validation block and following assignment are present, edit "
        "that local block instead. Do not output a unified diff. Do not use markdown fences.\n\n"
        f"Issue:\n{task.problem_statement}\n\nHints:\n{task.hints_text or '(none)'}\n\n"
        f"Relevant repository excerpts:\n{rendered_excerpts}\n"
    )


def _call_patch_plan_model(
    *,
    model: str,
    prompt: str,
    timeout_seconds: float,
    max_output_tokens: int,
    seed: int,
    base_url_env: str,
    api_key_env: str,
) -> tuple[SwebenchPatchPlan, dict[str, object]]:
    import urllib.request

    base_url = os.environ.get(base_url_env)
    api_key = os.environ.get(api_key_env)
    if not base_url or not api_key:
        raise SwebenchVerifiedDevError(
            f"missing model gateway configuration in {base_url_env} / {api_key_env}"
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert Python bug-fixing assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_output_tokens,
        "seed": seed,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = json.load(response)
    message = raw["choices"][0]["message"]
    text = (message.get("content") or "") + "\n" + (message.get("reasoning_content") or "")
    return extract_patch_plan(text), raw


def _apply_patch_plan(repo_dir: Path, plan: SwebenchPatchPlan) -> None:
    path = repo_dir / plan.file_path
    if not path.is_file():
        raise SwebenchVerifiedDevError(f"patch-plan file does not exist: {plan.file_path}")
    source = path.read_text(encoding="utf-8")
    occurrences = source.count(plan.before)
    if occurrences != 1:
        raise SwebenchVerifiedDevError(
            f"patch-plan before-snippet matched {occurrences} times in {plan.file_path}"
        )
    path.write_text(source.replace(plan.before, plan.after, 1), encoding="utf-8")


def _git_diff(repo_dir: Path) -> str:
    result = _git(repo_dir, "diff", "--no-ext-diff")
    return result.stdout


def run_swebench_verified_dev(
    *,
    output: Path,
    instance_id: str,
    model: str | None,
    source: str,
    base_url_env: str = "LITELLM_BASE_URL",
    api_key_env: str = "LITELLM_API_KEY",
    timeout_seconds: float = 240.0,
    max_output_tokens: int = 2048,
) -> dict[str, object]:
    task = load_verified_task(instance_id)
    profile = _LOCAL_PROFILES.get(task.repo)
    if profile is None:
        raise SwebenchVerifiedDevError(
            f"no local development execution profile is defined for repo {task.repo}"
        )
    worktree = _materialize_worktree(task, Path(output))
    _apply_git_patch(worktree, task.test_patch, label="test")
    env_python = _prepare_env(worktree, Path(output), profile)
    fail_tests = task.fail_to_pass
    pass_tests = task.pass_to_pass[: profile.pass_to_pass_sample_limit]
    baseline = {
        "fail_to_pass": _pytest(env_python, worktree, fail_tests),
        "pass_to_pass_sample": _pytest(env_python, worktree, pass_tests),
    }
    plan_payload: dict[str, object] | None = None
    plan: SwebenchPatchPlan | None = None
    if source == "gold":
        _apply_git_patch(worktree, task.patch, label="gold")
    elif source == "model":
        if not model:
            raise SwebenchVerifiedDevError("model source requires --model")
        excerpts = _collect_excerpts(worktree, task)
        plan, raw = _call_patch_plan_model(
            model=model,
            prompt=_patch_plan_prompt(task, excerpts),
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            seed=0,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
        )
        plan_payload = {
            "file_path": plan.file_path,
            "before": plan.before,
            "after": plan.after,
            "explanation": plan.explanation,
            "provider_raw": raw,
        }
        _apply_patch_plan(worktree, plan)
    else:
        raise SwebenchVerifiedDevError("source must be gold or model")
    candidate = {
        "fail_to_pass": _pytest(env_python, worktree, fail_tests),
        "pass_to_pass_sample": _pytest(env_python, worktree, pass_tests),
    }
    payload: dict[str, object] = {
        "schema_version": "1",
        "benchmark": "SWE-bench-Verified-dev",
        "instance_id": task.instance_id,
        "repo": task.repo,
        "difficulty": task.difficulty,
        "source": source,
        "model": model,
        "baseline": baseline,
        "candidate": candidate,
        "patch_plan": plan_payload,
        "candidate_patch": _git_diff(worktree),
        "reportable": False,
        "disclaimer": (
            "Development-only local SWE-bench Verified runner. This is not the official "
            "containerized evaluation harness and currently ships a narrow local execution profile."
        ),
    }
    return payload


__all__ = [
    "SwebenchPatchPlan",
    "SwebenchVerifiedDevError",
    "SwebenchVerifiedTask",
    "extract_patch_plan",
    "load_verified_task",
    "run_swebench_verified_dev",
]
