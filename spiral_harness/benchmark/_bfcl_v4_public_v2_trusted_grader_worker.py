"""Isolated exact-source worker for the BFCL V4 public v2 trusted grader.

This script deliberately imports only the standard library.  It reuses the
already byte-pinned v1 worker's generic JSON, Git-object, and AST-loader
primitives, but owns an independent v2 roster and never uses the v1 answer or
split bindings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PROTOCOL = "spiral-bfcl-v4-public-development-v2-trusted-grader-worker/v1"
FAILURE_PROTOCOL = "spiral-bfcl-v4-public-development-v2-trusted-grader-worker-failure/v1"
CHECKER_SOURCE_BUNDLE_SHA256 = "cf538a0dc09f515bd0cefee3f7b81f8dcc2c904a386de6d1da5013e2f5e6300d"
SELECTED_TASK_IDS = frozenset(
    {
        "simple_python_198",
        "simple_python_162",
        "multiple_30",
        "parallel_24",
        "parallel_multiple_2",
        "simple_python_179",
        "multiple_71",
        "parallel_26",
        "parallel_multiple_52",
        "simple_python_218",
        "simple_python_286",
        "simple_python_385",
        "simple_python_292",
        "multiple_16",
        "multiple_92",
        "multiple_19",
        "multiple_84",
        "parallel_13",
        "parallel_23",
        "parallel_41",
        "parallel_39",
        "parallel_multiple_13",
        "parallel_multiple_38",
        "parallel_multiple_50",
        "parallel_multiple_16",
    }
)
FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


class WorkerFailure(RuntimeError):
    pass


def _category(task_id: str) -> str:
    for category in ("parallel_multiple", "simple_python", "multiple", "parallel"):
        if task_id.startswith(f"{category}_"):
            return category
    raise WorkerFailure


def _legacy() -> dict[str, Any]:
    path = Path(__file__).with_name("_bfcl_v4_public_grader_worker.py").resolve(strict=True)
    return runpy.run_path(str(path), run_name="_bfcl_v4_public_v2_worker_primitives")


def _evaluate(
    request: dict[str, Any],
    *,
    checkout: Path,
    git: Path,
    primitives: dict[str, Any],
) -> dict[str, Any]:
    if (
        set(request)
        != {
            "calls",
            "candidate_payload_sha256",
            "official_function_names",
            "protocol",
            "task_id",
        }
        or request["protocol"] != PROTOCOL
    ):
        raise WorkerFailure
    if dict(os.environ) != ENVIRONMENT:
        raise WorkerFailure
    system_git = shutil.which("git", path=os.defpath)
    if system_git is None or Path(system_git).resolve(strict=True) != git:
        raise WorkerFailure
    head = primitives["_git"](git, checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if head != UPSTREAM_COMMIT:
        raise WorkerFailure

    task_id = request["task_id"]
    payload_sha256 = request["candidate_payload_sha256"]
    names = request["official_function_names"]
    calls = request["calls"]
    if (
        not isinstance(task_id, str)
        or task_id not in SELECTED_TASK_IDS
        or not isinstance(payload_sha256, str)
        or SHA256.fullmatch(payload_sha256) is None
        or not isinstance(names, list)
        or any(not isinstance(name, str) or FUNCTION_NAME.fullmatch(name) is None for name in names)
        or not isinstance(calls, list)
        or len(calls) > 64
    ):
        raise WorkerFailure
    decoded: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"arguments", "function_name"}:
            raise WorkerFailure
        function_name, arguments = call["function_name"], call["arguments"]
        if (
            not isinstance(function_name, str)
            or FUNCTION_NAME.fullmatch(function_name) is None
            or not isinstance(arguments, dict)
        ):
            raise WorkerFailure
        primitives["_validate_json_tree"](arguments, "prediction-invalid")
        decoded.append({function_name: arguments})

    category = _category(task_id)
    root = "berkeley-function-call-leaderboard/bfcl_eval/data"
    question_path = f"{root}/BFCL_v4_{category}.json"
    answer_path = f"{root}/possible_answer/BFCL_v4_{category}.json"
    question_blob = primitives["_blob"](git, checkout, question_path)
    expected_q_size, expected_q_sha, expected_a_size, expected_a_sha = primitives["CATEGORY_BLOBS"][
        category
    ]
    if len(question_blob) != expected_q_size or primitives["_sha256"](question_blob) != (
        expected_q_sha
    ):
        raise WorkerFailure
    task, _ = primitives["_jsonl_entry"](
        question_blob,
        task_id,
        "question-row-invalid",
    )
    if set(task) != {"function", "id", "question"} or not isinstance(task["function"], list):
        raise WorkerFailure
    observed_names = [
        item.get("name") if isinstance(item, dict) else None for item in task["function"]
    ]
    if observed_names != names or len(set(observed_names)) != len(observed_names):
        raise WorkerFailure
    observed_payload_sha256 = primitives["_sha256"](primitives["_canonical_bytes"](task))
    if observed_payload_sha256 != payload_sha256:
        raise WorkerFailure

    # The possible answer is opened only after all caller, checkout, question,
    # and candidate-payload validation.  No answer-derived identity is emitted.
    answer_blob = primitives["_blob"](git, checkout, answer_path)
    if len(answer_blob) != expected_a_size or primitives["_sha256"](answer_blob) != (
        expected_a_sha
    ):
        raise WorkerFailure
    answer, _ = primitives["_jsonl_entry"](answer_blob, task_id, "answer-row-invalid")
    if set(answer) != {"ground_truth", "id"}:
        raise WorkerFailure
    checker, language, source_bundle_sha256 = primitives["_load_ast_checker"](git, checkout)
    if source_bundle_sha256 != CHECKER_SOURCE_BUNDLE_SHA256:
        raise WorkerFailure
    result = checker.ast_checker(
        task["function"],
        decoded,
        answer["ground_truth"],
        language.PYTHON,
        category,
        primitives["DECODER_MODEL"],
    )
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        raise WorkerFailure
    return {
        "answer_data_present": False,
        "candidate_payload_sha256": observed_payload_sha256,
        "checker_diagnostics_present": False,
        "checker_source_bundle_sha256": source_bundle_sha256,
        "correct": result["valid"],
        "protocol": PROTOCOL,
    }


def _emit(value: dict[str, Any]) -> None:
    content = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stdout.buffer.write(content + b"\n")
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--git", required=True)
    arguments = parser.parse_args()
    try:
        primitives = _legacy()
        content = sys.stdin.buffer.read()
        request = primitives["_json_object"](content, "request-json-invalid")
        if primitives["_canonical_bytes"](request) != content:
            raise WorkerFailure
        result = _evaluate(
            request,
            checkout=Path(arguments.checkout).resolve(strict=True),
            git=Path(arguments.git).resolve(strict=True),
            primitives=primitives,
        )
    except Exception:
        _emit({"failure_code": "trusted-worker-rejected", "protocol": FAILURE_PROTOCOL})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
