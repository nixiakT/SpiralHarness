"""Credential-free exact-source worker for the 15-row BFCL V4 public pilot.

The worker is intentionally standalone: ``python -I`` executes it without
importing SpiralHarness or the installed ``bfcl_eval`` package.  It reads the
question, public possible-answer, and checker source only from the pinned Git
object, executes only the byte-pinned upstream AST checker, and emits one
grader/auditor receipt payload.  This remains public development evidence, not
a hidden test, official BFCL CLI run, or reportable full-suite score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PROTOCOL = "spiral-bfcl-v4-public-development-grader/v1"
FAILURE_PROTOCOL = "spiral-bfcl-v4-public-grader-failure/v1"
DECODER_MODEL = "spiral-public-pilot-no-provider"

TASKS = {
    "simple_python_0": (
        "simple_python",
        "fit",
        "2a7518e3feb766d5ab17c4dc5acce930297d8b3fe2f131d5b9f99a87030883e0",
        "60e78af6fe7e520e8c066d46388bf3a2a2c520700122046b5ef8d04273d7efae",
    ),
    "simple_python_211": (
        "simple_python",
        "fit",
        "46e4ef4f2d59d201b4753a953e44a5dcb57ebfc996d2a2839b76e77d06a2d6f6",
        "77ae90a2a6a3c8e2e4d649b2de5d12ee261bab8bf0d404516225aa25806fcbf2",
    ),
    "multiple_5": (
        "multiple",
        "fit",
        "e48c9d511ae591ea429c318f2b20b4d08cdbb481f232ab16983148741a9f6d90",
        "33a3cfbbfbaf62c8dd28372bebea7190f56c25a24505ece824c99538bf5bc146",
    ),
    "parallel_0": (
        "parallel",
        "fit",
        "061c73b5dece7e1a3dc4797288c41e3bbe543f00ac12b7913b8886b86d4f7c7f",
        "f4d106f78499ea9a2610d289a4fc2269b57feb438b8a89c7f907ade4539cf4f4",
    ),
    "parallel_multiple_9": (
        "parallel_multiple",
        "fit",
        "33abb475f87249bc7dbb16977ea99460b8143ce2479347f15008d98c5ed3125b",
        "d27465d36b5bf5da1a6b38b098eca5cd4157bec6fbc6a20fa48ece00eedf5fef",
    ),
    "multiple_10": (
        "multiple",
        "gate",
        "13436b9bf5d50afadd351de6aaed3ae253a6d9b940112e9ccb577d53e6636735",
        "9a59ac86a7eb2de013de804e3bd1d5f43f5d7f80e10622133b699c9f1e2f2dce",
    ),
    "parallel_multiple_11": (
        "parallel_multiple",
        "gate",
        "9f2d9d271588dbad71a9cc6bd6cfc2d6dc8f617edf2e42273e93a2754108cbfe",
        "4cd1a3a605e5c6b262b56c41f339bc14a693517324317961221aa0c5e3488549",
    ),
    "simple_python_87": (
        "simple_python",
        "holdout",
        "20ebbe4fd6d02f666873e1b947a1a08258ff4dbd59b2115d9e3df569854b6225",
        "e05e0aab910da9d444d98701b793f37bb3b9fdcdb5ce7bc9c01072191919c110",
    ),
    "simple_python_128": (
        "simple_python",
        "holdout",
        "e2fdb262172b9ffa365ec53d8ee5e84e80149c4a18b425cd297619feb7c25359",
        "e34aa2c8c5c8466ac855ff2b5edbc919fac3a2cb3df132d0d047cea2e872d999",
    ),
    "multiple_7": (
        "multiple",
        "holdout",
        "7438097a31830593c0cddedf811e8513c121089c8e1264065748bb494620c300",
        "6dfe2b0f62880dc1bc1e53979d1950d1971dde5fdefbf5224552ad64fb00ecae",
    ),
    "multiple_8": (
        "multiple",
        "holdout",
        "5fef25c54a6fad73bc87e0306d011641bf2d5d36e89daaee6e12f6bffee3f201",
        "94a2ba01864383c7a6214b5e49e749cdbbf802441cd32cc348033e95786b6895",
    ),
    "parallel_3": (
        "parallel",
        "holdout",
        "9d12708371f3d02a45505da617edac896885913dd4aa15f8c3570db26d112bfa",
        "8eba1f85733490f95848fadacf99c0d90b686e19e6613992b99de8ec21c5773d",
    ),
    "parallel_4": (
        "parallel",
        "holdout",
        "6ed4035e9a8223dbd75be5138ef23a03f61df7abb6a8b4a20d750658d37af82f",
        "5b9b076d97155588d6be9e8a50daf98fff6caed5b069b490c1675c30eee284c3",
    ),
    "parallel_multiple_5": (
        "parallel_multiple",
        "holdout",
        "27ac53eccc5c0327080d4d6c0b61783aaaa4d2e66349f53405f305dcc48409a0",
        "77712fa46b98ff20241a07b5959aba5e8dfbedfa7c07aabb82f6f09c9202895e",
    ),
    "parallel_multiple_55": (
        "parallel_multiple",
        "holdout",
        "d5a1729a92f9e46ae02fe06f19c15594a3ad47a8acd10dea745c0faebf07e06d",
        "85dface8b8eb8586f71765ccbe6b9ab73fb85be44632edc9b98533c94def00e1",
    ),
}
CATEGORY_BLOBS = {
    "multiple": (
        316_583,
        "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
        32_254,
        "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    ),
    "parallel": (
        171_896,
        "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
        66_005,
        "8a6aa19c1adddc6a5a2f7e40f9dbf30cc7e95815e7b830c90589ab318229e0f0",
    ),
    "parallel_multiple": (
        347_080,
        "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
        74_602,
        "5ebf24f458c1f16300c05505d83d6f0a1b68b79be273a033febd0d4f840507e3",
    ),
    "simple_python": (
        283_274,
        "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
        63_627,
        "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
    ),
}

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

_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkerFailure(RuntimeError):
    """A code-only failure safe to return across the process boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_surrogates(value: Any, failure_code: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise WorkerFailure(failure_code)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, failure_code)
            _reject_surrogates(item, failure_code)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item, failure_code)


def _validate_json_tree(value: Any, failure_code: str) -> None:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 10_000 or depth > 32:
            raise WorkerFailure(failure_code)
        if isinstance(item, float) and not math.isfinite(item):
            raise WorkerFailure(failure_code)
        if isinstance(item, str):
            _reject_surrogates(item, failure_code)
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise WorkerFailure(failure_code)
                _reject_surrogates(key, failure_code)
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise WorkerFailure(failure_code)

    visit(value, 0)


def _canonical_bytes(value: Any, failure_code: str = "canonical-json-failed") -> bytes:
    _validate_json_tree(value, failure_code)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise WorkerFailure(failure_code) from error


def _json_value(content: bytes, failure_code: str) -> Any:
    if len(content) > 2_097_152:
        raise WorkerFailure(failure_code)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerFailure(failure_code)
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise WorkerFailure(failure_code)

    try:
        text = content.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerFailure(failure_code) from error
    _validate_json_tree(value, failure_code)
    return value


def _json_object(content: bytes, failure_code: str) -> dict[str, Any]:
    value = _json_value(content, failure_code)
    if not isinstance(value, dict):
        raise WorkerFailure(failure_code)
    return value


def _git(git_executable: Path, checkout: Path, *arguments: str) -> bytes:
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


def _question_binding(
    task_id: str,
    category: str,
    split: str,
    question_path: str,
    question_blob: bytes,
    task: dict[str, Any],
    task_row: bytes,
) -> dict[str, Any]:
    functions = task.get("function")
    if not isinstance(functions, list) or not functions:
        raise WorkerFailure("question-row-invalid")
    names: list[str] = []
    for item in functions:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise WorkerFailure("question-row-invalid")
        names.append(item["name"])
    if len(set(names)) != len(names):
        raise WorkerFailure("question-row-invalid")
    return {
        "category": category,
        "function_schemas_sha256": _sha256(_canonical_bytes(functions)),
        "official_function_names": names,
        "question_blob_sha256": _sha256(question_blob),
        "question_blob_size": len(question_blob),
        "question_git_path": question_path,
        "question_row_sha256": _sha256(task_row),
        "question_row_size": len(task_row),
        "question_sha256": _sha256(_canonical_bytes(task.get("question"))),
        "split": split,
        "task_id": task_id,
    }


def _answer_binding(
    task_id: str,
    answer_path: str,
    answer_blob: bytes,
    answer: dict[str, Any],
    answer_row: bytes,
) -> dict[str, Any]:
    if "ground_truth" not in answer:
        raise WorkerFailure("answer-row-invalid")
    return {
        "answer_blob_sha256": _sha256(answer_blob),
        "answer_blob_size": len(answer_blob),
        "answer_git_path": answer_path,
        "answer_row_sha256": _sha256(answer_row),
        "answer_row_size": len(answer_row),
        "candidate_visible": False,
        "contains_answer_derived_identity": True,
        "ground_truth_sha256": _sha256(_canonical_bytes(answer["ground_truth"])),
        "task_id": task_id,
        "visibility": "grader-auditor-only",
    }


def _decode_prediction(
    prediction_text: Any,
    expected_task_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(prediction_text, str):
        raise WorkerFailure("prediction-invalid")
    try:
        prediction_bytes = prediction_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise WorkerFailure("prediction-invalid") from error
    prediction = _json_object(prediction_bytes, "prediction-invalid")
    if _canonical_bytes(prediction, "prediction-invalid") != prediction_bytes:
        raise WorkerFailure("prediction-not-canonical")
    if set(prediction) != {"calls", "task_id"} or prediction["task_id"] != expected_task_id:
        raise WorkerFailure("prediction-invalid")
    calls = prediction["calls"]
    if not isinstance(calls, list) or len(calls) > 64:
        raise WorkerFailure("prediction-invalid")
    decoded: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"arguments_json", "function_name"}:
            raise WorkerFailure("prediction-invalid")
        function_name = call["function_name"]
        arguments_json = call["arguments_json"]
        if (
            not isinstance(function_name, str)
            or _FUNCTION_NAME.fullmatch(function_name) is None
            or not isinstance(arguments_json, str)
        ):
            raise WorkerFailure("prediction-invalid")
        try:
            arguments_bytes = arguments_json.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise WorkerFailure("prediction-invalid") from error
        arguments = _json_object(arguments_bytes, "prediction-invalid")
        if _canonical_bytes(arguments, "prediction-invalid") != arguments_bytes:
            raise WorkerFailure("prediction-not-canonical")
        decoded.append({function_name: arguments})
    return _sha256(prediction_bytes), decoded


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
        # Executing this byte-pinned checker source is the worker's sole purpose.
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
    return checker_module, enums_module.Language, _sha256(_canonical_bytes(source_records))


def _coarse_failure(valid: bool, error_type: str | None) -> str:
    if valid:
        return "none"
    if isinstance(error_type, str) and "wrong_count" in error_type:
        return "call-count"
    return "function-or-arguments"


def _evaluate(request: dict[str, Any], git_executable: Path, checkout: Path) -> dict[str, Any]:
    if set(request) != {
        "git_executable",
        "holdout_unlock_sha256",
        "prediction_json",
        "prediction_sha256",
        "protocol",
        "question_binding_sha256",
        "slot_binding_sha256",
        "source_bundle_sha256",
        "task_id",
    }:
        raise WorkerFailure("request-schema-mismatch")
    if request["protocol"] != PROTOCOL:
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
        git_version = _git(git_executable, checkout, "--version").decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise WorkerFailure("git-executable-binding-invalid") from error
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
    task_id = request["task_id"]
    if not isinstance(task_id, str) or task_id not in TASKS:
        raise WorkerFailure("task-not-in-frozen-roster")
    category, split, expected_question_binding, expected_answer_binding = TASKS[task_id]
    holdout_unlock = request["holdout_unlock_sha256"]
    if split == "holdout":
        if not isinstance(holdout_unlock, str) or _SHA256.fullmatch(holdout_unlock) is None:
            raise WorkerFailure("holdout-selection-artifacts-not-bound")
    elif holdout_unlock is not None:
        raise WorkerFailure("non-holdout-unlock-forbidden")
    for digest_field in (
        "prediction_sha256",
        "question_binding_sha256",
        "slot_binding_sha256",
        "source_bundle_sha256",
    ):
        if (
            not isinstance(request[digest_field], str)
            or _SHA256.fullmatch(request[digest_field]) is None
        ):
            raise WorkerFailure("request-digest-invalid")

    prediction_sha256, decoded = _decode_prediction(request["prediction_json"], task_id)
    if prediction_sha256 != request["prediction_sha256"]:
        raise WorkerFailure("prediction-hash-mismatch")

    question_path = f"berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{category}.json"
    answer_path = (
        f"berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_{category}.json"
    )
    question_blob = _blob(git_executable, checkout, question_path)
    expected_q_size, expected_q_sha, expected_a_size, expected_a_sha = CATEGORY_BLOBS[category]
    if len(question_blob) != expected_q_size or _sha256(question_blob) != expected_q_sha:
        raise WorkerFailure("question-blob-hash-mismatch")
    task, task_row = _jsonl_entry(question_blob, task_id, "question-row-invalid")
    question_binding = _question_binding(
        task_id,
        category,
        split,
        question_path,
        question_blob,
        task,
        task_row,
    )
    question_binding_sha256 = _sha256(_canonical_bytes(question_binding))
    if question_binding_sha256 != expected_question_binding:
        raise WorkerFailure("question-binding-pinned-hash-mismatch")
    if question_binding_sha256 != request["question_binding_sha256"]:
        raise WorkerFailure("question-binding-request-mismatch")

    # This is the only code path in the grader implementation that opens a
    # possible-answer blob.  It runs after all candidate/request validation.
    answer_blob = _blob(git_executable, checkout, answer_path)
    if len(answer_blob) != expected_a_size or _sha256(answer_blob) != expected_a_sha:
        raise WorkerFailure("answer-blob-hash-mismatch")
    answer, answer_row = _jsonl_entry(answer_blob, task_id, "answer-row-invalid")
    answer_binding = _answer_binding(task_id, answer_path, answer_blob, answer, answer_row)
    if _sha256(_canonical_bytes(answer_binding)) != expected_answer_binding:
        raise WorkerFailure("answer-binding-pinned-hash-mismatch")

    checker_module, language, source_bundle_sha256 = _load_ast_checker(
        git_executable,
        checkout,
    )
    if source_bundle_sha256 != request["source_bundle_sha256"]:
        raise WorkerFailure("source-bundle-hash-mismatch")
    try:
        checker_result = checker_module.ast_checker(
            task["function"],
            decoded,
            answer["ground_truth"],
            language.PYTHON,
            category,
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
        "answer_binding": answer_binding,
        "coarse_failure_class": _coarse_failure(valid, error_type),
        "credential_environment_inherited": False,
        "error_type": error_type,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "git_executable_sha256": git_binding["sha256"],
        "holdout_unlock_sha256": holdout_unlock,
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "prediction_sha256": prediction_sha256,
        "protocol": PROTOCOL,
        "proxy_environment_inherited": False,
        "question_binding": question_binding,
        "slot_binding_sha256": request["slot_binding_sha256"],
        "source_bundle_sha256": source_bundle_sha256,
        "task_id": task_id,
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
        _emit({"failure_code": code, "protocol": FAILURE_PROTOCOL})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
