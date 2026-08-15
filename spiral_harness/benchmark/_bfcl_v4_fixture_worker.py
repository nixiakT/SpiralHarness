"""Credential-free worker for one pinned BFCL V4 AST fixture.

This worker deliberately does not import the installed ``bfcl_eval`` package.
It reads byte-exact source and data blobs from one pinned Git object, executes
the upstream Python AST checker in a subprocess, and emits a binary validity
result.  The upstream model registry is replaced with one explicit decoder
coordinate because importing that registry activates every provider handler.
Consequently this is a source-isolated fixture rehearsal, not the official
BFCL CLI or an official benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TASK_ID = "simple_python_0"
CATEGORY = "simple_python"
DECODER_MODEL = "spiral-fixture-no-provider"
QUESTION_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json"
ANSWER_PATH = (
    "berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json"
)
QUESTION_BLOB_SHA256 = "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991"
ANSWER_BLOB_SHA256 = "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc"
TASK_ROW_SHA256 = "9208f93fb0939c43255773e90d5f15a6362f5701891cbcb917a505195d89c5e4"
ANSWER_ROW_SHA256 = "a5499305963e3c5d0c4c67e75f9c8e9fc9cdf0e535de7d36361177fef6f7b8fb"

SOURCE_SHA256 = {
    "berkeley-function-call-leaderboard/bfcl_eval/constants/enums.py": (
        "2182becfa2a1d071ee1db30db593b4758c6bf866aa12d2d4b8daf09175ea518a"
    ),
    "berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py": (
        "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd"
    ),
    (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/java_type_converter.py"
    ): "2fd4f4b0443b3dd974a1723bb4e45c086d7b352631062da7807ad1ad40706604",
    (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/js_type_converter.py"
    ): "a114e9ff75c025cb52787ac33d6c2fbaa390905c6125a2b3c6afebab232bb5e4",
    (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py"
    ): "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538",
}


class WorkerFailure(RuntimeError):
    """A safe, code-only worker failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _task_binding_sha256(
    task: dict[str, Any],
    question_blob: bytes,
    task_row: bytes,
) -> str:
    schemas_json = _canonical_bytes(task["function"]).decode("utf-8")
    function_names = [item["name"] for item in task["function"]]
    payload = {
        "category": CATEGORY,
        "evidence_scope": "public-partial-source-isolated-fixture-rehearsal-only",
        "function_names": function_names,
        "function_schemas_json": schemas_json,
        "function_schemas_sha256": _sha256(_canonical_bytes(task["function"])),
        "hidden_test_evidence": False,
        "model_invoked": False,
        "official_score_produced": False,
        "partial_evaluation": True,
        "possible_answers_public": True,
        "question_blob_sha256": _sha256(question_blob),
        "question_blob_size": len(question_blob),
        "question_git_path": QUESTION_PATH,
        "question_sha256": _sha256(_canonical_bytes(task["question"])),
        "questions_public": True,
        "reportable_result": False,
        "schema_version": "1",
        "score_free_adapter": True,
        "sealed_evidence": False,
        "suite_id": "bfcl-v4@6ea57973",
        "task_id": TASK_ID,
        "task_row_sha256": _sha256(task_row),
        "task_row_size": len(task_row),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_release": "2026.3.23",
        "upstream_repository": "https://github.com/ShishirPatil/gorilla",
    }
    return _sha256(_canonical_bytes(payload))


def _grader_binding_sha256(
    answer: dict[str, Any],
    answer_blob: bytes,
    answer_row: bytes,
) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "answer_blob_sha256": _sha256(answer_blob),
                "answer_blob_size": len(answer_blob),
                "answer_git_path": ANSWER_PATH,
                "answer_row_sha256": _sha256(answer_row),
                "answer_row_size": len(answer_row),
                "candidate_visible": False,
                "contains_answer_derived_identity": True,
                "ground_truth_sha256": _sha256(_canonical_bytes(answer["ground_truth"])),
                "reportable_result": False,
                "schema_version": "1",
                "task_id": TASK_ID,
                "upstream_commit": UPSTREAM_COMMIT,
                "visibility": "grader-auditor-only",
            }
        )
    )


def _git(
    git_executable: Path,
    checkout: Path,
    *arguments: str,
) -> bytes:
    completed = subprocess.run(
        [str(git_executable), "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        timeout=20,
    )
    if completed.returncode != 0:
        raise WorkerFailure("git-command-failed")
    return completed.stdout


def _blob(git_executable: Path, checkout: Path, git_path: str) -> bytes:
    return _git(
        git_executable,
        checkout,
        "cat-file",
        "blob",
        f"{UPSTREAM_COMMIT}:{git_path}",
    )


def _json_object(content: bytes, failure_code: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerFailure(failure_code)
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerFailure(failure_code) from error
    if not isinstance(value, dict):
        raise WorkerFailure(failure_code)
    return value


def _jsonl_entry(blob: bytes, entry_id: str, failure_code: str) -> tuple[dict[str, Any], bytes]:
    matches: list[tuple[dict[str, Any], bytes]] = []
    lines = blob.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.endswith(b"\r\n") or (not line.endswith(b"\n") and index != len(lines) - 1):
            raise WorkerFailure(failure_code)
        payload = line[:-1] if line.endswith(b"\n") else line
        value = _json_object(payload, failure_code)
        if value.get("id") == entry_id:
            matches.append((value, line))
    if len(matches) != 1:
        raise WorkerFailure(failure_code)
    return matches[0]


def _package(name: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module


def _source_module(name: str, source: bytes, source_label: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = source_label
    sys.modules[name] = module
    try:
        code = compile(source.decode("utf-8"), source_label, "exec", dont_inherit=True)
        # Executing this byte-pinned upstream checker source is the worker's sole purpose.
        exec(code, module.__dict__)
    except Exception as error:
        raise WorkerFailure("upstream-source-load-failed") from error
    return module


def _load_ast_checker(
    git_executable: Path,
    checkout: Path,
) -> tuple[types.ModuleType, type[Enum], str]:
    sources: dict[str, bytes] = {}
    source_records: list[dict[str, Any]] = []
    for git_path, expected_sha256 in sorted(SOURCE_SHA256.items()):
        source = _blob(git_executable, checkout, git_path)
        if _sha256(source) != expected_sha256:
            raise WorkerFailure("upstream-source-hash-mismatch")
        sources[git_path] = source
        source_records.append({"path": git_path, "sha256": expected_sha256, "size": len(source)})

    for package in (
        "bfcl_eval",
        "bfcl_eval.constants",
        "bfcl_eval.eval_checker",
        "bfcl_eval.eval_checker.ast_eval",
        "bfcl_eval.eval_checker.ast_eval.type_convertor",
    ):
        _package(package)

    enums_path = "berkeley-function-call-leaderboard/bfcl_eval/constants/enums.py"
    enums_module = _source_module(
        "bfcl_eval.constants.enums",
        sources[enums_path],
        f"{UPSTREAM_COMMIT}:{enums_path}",
    )
    mappings_path = "berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py"
    _source_module(
        "bfcl_eval.constants.type_mappings",
        sources[mappings_path],
        f"{UPSTREAM_COMMIT}:{mappings_path}",
    )

    model_config = types.ModuleType("bfcl_eval.constants.model_config")
    model_config.MODEL_CONFIG_MAPPING = {DECODER_MODEL: SimpleNamespace(underscore_to_dot=False)}
    sys.modules[model_config.__name__] = model_config

    java_path = (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/java_type_converter.py"
    )
    _source_module(
        "bfcl_eval.eval_checker.ast_eval.type_convertor.java_type_converter",
        sources[java_path],
        f"{UPSTREAM_COMMIT}:{java_path}",
    )
    js_path = (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
        "type_convertor/js_type_converter.py"
    )
    _source_module(
        "bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter",
        sources[js_path],
        f"{UPSTREAM_COMMIT}:{js_path}",
    )
    checker_path = (
        "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py"
    )
    checker_module = _source_module(
        "bfcl_eval.eval_checker.ast_eval.ast_checker",
        sources[checker_path],
        f"{UPSTREAM_COMMIT}:{checker_path}",
    )
    return (
        checker_module,
        enums_module.Language,
        _sha256(_canonical_bytes(source_records)),
    )


def _evaluate(request: dict[str, Any], git_executable: Path, checkout: Path) -> dict[str, Any]:
    if set(request) != {
        "protocol",
        "git_executable",
        "grader_answer_binding_sha256",
        "response_row_jsonl",
        "response_row_sha256",
        "source_bundle_sha256",
        "task_binding_sha256",
    }:
        raise WorkerFailure("request-schema-mismatch")
    if request["protocol"] != "spiral-bfcl-v4-source-isolated-fixture/v1":
        raise WorkerFailure("request-protocol-mismatch")
    allowed_environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    if dict(os.environ) != allowed_environment:
        raise WorkerFailure("worker-environment-not-allowlisted")

    system_git = shutil.which("git", path=os.defpath)
    if system_git is None or Path(system_git).resolve(strict=True) != git_executable:
        raise WorkerFailure("git-executable-not-system-resolved")
    git_binding = request["git_executable"]
    if not isinstance(git_binding, dict) or set(git_binding) != {
        "path",
        "sha256",
        "size",
        "version",
    }:
        raise WorkerFailure("git-executable-binding-invalid")
    try:
        git_content = git_executable.read_bytes()
    except OSError as error:
        raise WorkerFailure("git-executable-binding-invalid") from error
    git_version = _git(git_executable, checkout, "--version").decode("utf-8").strip()
    if git_binding != {
        "path": str(git_executable),
        "sha256": _sha256(git_content),
        "size": len(git_content),
        "version": git_version,
    }:
        raise WorkerFailure("git-executable-binding-invalid")

    head = _git(git_executable, checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if head != UPSTREAM_COMMIT:
        raise WorkerFailure("upstream-head-mismatch")

    question_blob = _blob(git_executable, checkout, QUESTION_PATH)
    answer_blob = _blob(git_executable, checkout, ANSWER_PATH)
    if _sha256(question_blob) != QUESTION_BLOB_SHA256:
        raise WorkerFailure("question-blob-hash-mismatch")
    if _sha256(answer_blob) != ANSWER_BLOB_SHA256:
        raise WorkerFailure("answer-blob-hash-mismatch")
    task, task_row = _jsonl_entry(question_blob, TASK_ID, "task-row-invalid")
    answer, answer_row = _jsonl_entry(answer_blob, TASK_ID, "answer-row-invalid")
    if _sha256(task_row) != TASK_ROW_SHA256:
        raise WorkerFailure("task-row-hash-mismatch")
    if _sha256(answer_row) != ANSWER_ROW_SHA256:
        raise WorkerFailure("answer-row-hash-mismatch")
    if _task_binding_sha256(task, question_blob, task_row) != request["task_binding_sha256"]:
        raise WorkerFailure("task-binding-hash-mismatch")
    if (
        _grader_binding_sha256(answer, answer_blob, answer_row)
        != request["grader_answer_binding_sha256"]
    ):
        raise WorkerFailure("grader-answer-binding-hash-mismatch")

    row_text = request["response_row_jsonl"]
    if not isinstance(row_text, str):
        raise WorkerFailure("response-row-invalid")
    row_bytes = row_text.encode("utf-8")
    if not row_bytes.endswith(b"\n") or row_bytes[:-1].find(b"\n") != -1:
        raise WorkerFailure("response-row-invalid")
    if _sha256(row_bytes) != request["response_row_sha256"]:
        raise WorkerFailure("response-row-hash-mismatch")
    row = _json_object(row_bytes[:-1], "response-row-invalid")
    if set(row) != {"id", "result"} or row["id"] != TASK_ID:
        raise WorkerFailure("response-row-invalid")
    if not isinstance(row["result"], list) or not row["result"]:
        raise WorkerFailure("response-row-invalid")

    decoded: list[dict[str, Any]] = []
    for call in row["result"]:
        if not isinstance(call, dict) or len(call) != 1:
            raise WorkerFailure("response-row-invalid")
        function_name, arguments_json = next(iter(call.items()))
        if not isinstance(function_name, str) or not isinstance(arguments_json, str):
            raise WorkerFailure("response-row-invalid")
        arguments = _json_object(arguments_json.encode("utf-8"), "response-row-invalid")
        decoded.append({function_name: arguments})

    checker_module, language, bundle_sha256 = _load_ast_checker(git_executable, checkout)
    if bundle_sha256 != request["source_bundle_sha256"]:
        raise WorkerFailure("source-bundle-hash-mismatch")
    try:
        checker_result = checker_module.ast_checker(
            task["function"],
            decoded,
            answer["ground_truth"],
            language.PYTHON,
            CATEGORY,
            DECODER_MODEL,
        )
    except Exception as error:
        raise WorkerFailure("upstream-checker-execution-failed") from error
    if not isinstance(checker_result, dict) or not isinstance(checker_result.get("valid"), bool):
        raise WorkerFailure("upstream-checker-output-invalid")
    valid = checker_result["valid"]
    error_type = None if valid else checker_result.get("error_type")
    if not valid and (not isinstance(error_type, str) or not error_type):
        raise WorkerFailure("upstream-checker-output-invalid")

    return {
        "error_type": error_type,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "credential_environment_inherited": False,
        "git_executable_sha256": git_binding["sha256"],
        "grader_answer_binding_sha256": request["grader_answer_binding_sha256"],
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "protocol": request["protocol"],
        "response_row_sha256": request["response_row_sha256"],
        "source_bundle_sha256": bundle_sha256,
        "task_binding_sha256": request["task_binding_sha256"],
        "proxy_environment_inherited": False,
        "upstream_ast_checker_valid": valid,
    }


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(_canonical_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--git", required=True)
    arguments = parser.parse_args()
    try:
        request = _json_object(sys.stdin.buffer.read(), "request-json-invalid")
        result = _evaluate(
            request,
            Path(arguments.git).resolve(strict=True),
            Path(arguments.checkout).resolve(strict=True),
        )
    except (OSError, subprocess.TimeoutExpired, WorkerFailure) as error:
        code = error.code if isinstance(error, WorkerFailure) else "worker-io-failed"
        _emit({"failure_code": code, "protocol": "spiral-bfcl-v4-worker-failure/v1"})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
