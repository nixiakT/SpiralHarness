"""Development-only GAIA validation runner.

This runner is intentionally modest: it targets GAIA validation once the user
has lawful access to the gated dataset, supports a conservative text-first
attachment surface, and records clearly when an item is skipped because its
attachment type is unsupported in this development harness.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import string
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spiral_harness.benchmark.gsm8k_smoke import build_live_gsm8k_spec
from spiral_harness.execution.contracts import ModelRequest, ResolvedHarness
from spiral_harness.execution.model import ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.I | re.S)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


class GaiaValidationDevError(RuntimeError):
    """A local GAIA validation setup or execution step failed."""


@dataclass(frozen=True, slots=True)
class GaiaValidationTask:
    task_id: str
    level: int
    question: str
    final_answer: str
    file_name: str | None
    split: str


def _import_dataset_helpers() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:  # pragma: no cover - exercised in CLI integration only
        raise GaiaValidationDevError(
            "GAIA runner requires optional packages: datasets and huggingface_hub"
        ) from exc
    return load_dataset, GatedRepoError


def extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    match = _FINAL_ANSWER_RE.search(text)
    if match is None:
        return text.strip()
    return match.group(1).strip()


def _normalize_text(value: str) -> str:
    lowered = value.casefold().strip()
    lowered = lowered.translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", lowered)


def _maybe_number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def score_gaia_answer(prediction: str, reference: str) -> bool:
    """Conservative GAIA answer scorer for development use.

    This follows the public scorer's broad intent: exact normalized match for
    strings and exact numeric equality after canonicalization for number-like
    answers.
    """

    predicted = extract_final_answer(prediction)
    expected = reference.strip()
    predicted_number = _maybe_number(predicted)
    expected_number = _maybe_number(expected)
    if predicted_number is not None and expected_number is not None:
        return abs(predicted_number - expected_number) < 1e-9
    if "," in expected and "," in predicted:
        predicted_items = tuple(
            item for item in (_normalize_text(part) for part in predicted.split(",")) if item
        )
        expected_items = tuple(
            item for item in (_normalize_text(part) for part in expected.split(",")) if item
        )
        return predicted_items == expected_items
    return _normalize_text(predicted) == _normalize_text(expected)


def load_gaia_validation_tasks(
    *,
    split: str = "validation",
    subset: str = "2023_all",
    hf_token_env: str = "HF_TOKEN",
) -> tuple[GaiaValidationTask, ...]:
    load_dataset, gated_error_type = _import_dataset_helpers()
    token = os.environ.get(hf_token_env)
    if not token:
        raise GaiaValidationDevError(
            f"GAIA dataset is gated; set {hf_token_env} after accepting access on Hugging Face"
        )
    try:
        dataset = load_dataset("gaia-benchmark/GAIA", name=subset, split=split, token=token)
    except gated_error_type as exc:  # pragma: no cover - depends on remote auth state
        raise GaiaValidationDevError(
            f"GAIA dataset access denied for split {split!r}; verify {hf_token_env}"
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on remote auth state
        raise GaiaValidationDevError(f"failed to load GAIA {subset}/{split}") from exc
    tasks: list[GaiaValidationTask] = []
    for row in dataset:
        task_id = row.get("task_id")
        question = row.get("Question")
        final_answer = row.get("Final answer")
        level = row.get("Level")
        if not isinstance(task_id, str) or not isinstance(question, str) or not isinstance(
            final_answer, str
        ):
            raise GaiaValidationDevError("GAIA row schema changed")
        if not isinstance(level, int):
            raise GaiaValidationDevError("GAIA level is not an integer")
        file_name = row.get("file_name")
        if file_name is not None and not isinstance(file_name, str):
            raise GaiaValidationDevError("GAIA file_name is not text or null")
        tasks.append(
            GaiaValidationTask(
                task_id=task_id,
                level=level,
                question=question,
                final_answer=final_answer,
                file_name=file_name,
                split=split,
            )
        )
    return tuple(tasks)


def _attachment_text(root: Path, task: GaiaValidationTask) -> str | None:
    if task.file_name is None:
        return None
    path = root / task.split / task.file_name
    if not path.is_file():
        raise GaiaValidationDevError(f"GAIA attachment missing on disk: {task.file_name}")
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md", ".py", ".json", ".html", ".htm", ".csv"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".csv":
            rows = list(csv.reader(io.StringIO(raw)))
            raw = "\n".join(",".join(row) for row in rows[:200])
        return raw[:40_000]
    if suffix == ".zip":
        lines: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist())[:20]:
                if name.endswith("/"):
                    continue
                lines.append(f"# FILE {name}")
                with archive.open(name) as handle:
                    text = handle.read().decode("utf-8", errors="replace")
                lines.append(text[:2_000])
        return "\n".join(lines)[:40_000]
    return None


def _gaia_prompt(task: GaiaValidationTask, attachment_text: str | None) -> str:
    attachment_block = ""
    if attachment_text is not None:
        attachment_block = (
            "\n\nAttached file contents follow. Treat them as trusted benchmark input.\n\n"
            f"{attachment_text}"
        )
    return (
        "Solve the GAIA task. Reason privately if needed, but end with exactly one line of the form "
        "'FINAL ANSWER: <answer>'.\n\nQuestion:\n"
        + task.question
        + attachment_block
    )


def run_gaia_validation_dev(
    *,
    output: Path,
    dataset_root: Path,
    backend: ModelBackend,
    model: str,
    split: str = "validation",
    subset: str = "2023_all",
    level: int | None = None,
    limit: int = 8,
    max_output_tokens: int = 512,
    timeout_seconds: float = 180.0,
    hf_token_env: str = "HF_TOKEN",
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("limit must be positive")
    tasks = load_gaia_validation_tasks(split=split, subset=subset, hf_token_env=hf_token_env)
    if level is not None:
        tasks = tuple(task for task in tasks if task.level == level)
    tasks = tasks[:limit]
    store = ArtifactStore(Path(output) / "artifacts")
    harness_ref = store.put_json(
        {
            "schema_version": "1",
            "benchmark": "GAIA-validation-dev",
            "subset": subset,
            "split": split,
            "level": level,
        },
        media_type="application/vnd.spiral-harness.gaia-validation-dev.harness.v1+json",
    )
    harness = ResolvedHarness.from_prompt(
        harness_ref=harness_ref,
        system_prompt="You are a careful benchmark assistant. Follow the final-answer contract exactly.",
    )
    spec = build_live_gsm8k_spec(
        backend_fingerprint=backend.fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    records: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        attachment_text = _attachment_text(dataset_root, task)
        if task.file_name is not None and attachment_text is None:
            records.append(
                {
                    "task_id": task.task_id,
                    "level": task.level,
                    "skipped": True,
                    "skip_reason": f"unsupported attachment type: {task.file_name}",
                }
            )
            continue
        response = backend.invoke(
            spec=spec,
            request=ModelRequest(
                task_id=f"gaia-validation-dev/{split}/{task.task_id}",
                harness_ref=harness_ref,
                base_system_prompt=harness.base_system_prompt,
                base_system_prompt_sha256=harness.base_system_prompt_sha256,
                system_prompt=harness.system_prompt,
                resolved_prompt_sha256=harness.resolved_prompt_sha256,
                user_prompt=_gaia_prompt(task, attachment_text),
                seed=index,
            ),
        )
        final_answer = extract_final_answer(response.output)
        correct = score_gaia_answer(final_answer, task.final_answer)
        records.append(
            {
                "task_id": task.task_id,
                "level": task.level,
                "skipped": False,
                "attachment_name": task.file_name,
                "predicted_final_answer": final_answer,
                "correct": correct,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )
    attempted = [record for record in records if not record["skipped"]]
    correct_count = sum(bool(record["correct"]) for record in attempted)
    payload: dict[str, object] = {
        "schema_version": "1",
        "benchmark": "GAIA-validation-dev",
        "subset": subset,
        "split": split,
        "model": model,
        "limit": limit,
        "level": level,
        "attempted": len(attempted),
        "skipped": len(records) - len(attempted),
        "correct": correct_count,
        "accuracy": (correct_count / len(attempted)) if attempted else None,
        "records": records,
        "reportable": False,
        "disclaimer": (
            "Development-only GAIA validation runner. It requires lawful GAIA dataset access and "
            "currently supports only text-like attachments."
        ),
    }
    artifact_ref = store.put_json(
        payload, media_type="application/vnd.spiral-harness.gaia-validation-dev.v1+json"
    )
    payload["artifact_sha256"] = artifact_ref.sha256
    return payload


__all__ = [
    "GaiaValidationDevError",
    "GaiaValidationTask",
    "extract_final_answer",
    "load_gaia_validation_tasks",
    "run_gaia_validation_dev",
    "score_gaia_answer",
]
