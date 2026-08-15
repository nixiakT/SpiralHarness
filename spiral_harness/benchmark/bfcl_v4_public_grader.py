"""Source-isolated grader bridge for the 15-task BFCL V4 public pilot."""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spiral_harness.benchmark._bfcl_v4_public_grader_worker import SOURCE_SHA256
from spiral_harness.benchmark.bfcl_v4 import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import (
    BfclV4EnvironmentEntry,
    BfclV4ExecutableBinding,
    BfclV4SourceFileBinding,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    PUBLIC_GRADER_PROTOCOL,
    BfclV4GraderAnswerBinding,
    BfclV4GradingSlotBinding,
    BfclV4HoldoutUnlock,
    BfclV4OfficialPredictionCall,
    BfclV4PublicGraderReceipt,
    BfclV4PublicPrediction,
    BfclV4PublicQuestionBinding,
    checked,
    current_public_grader_worker_binding,
    prediction_content,
    runtime_coordinates,
    source_bundle_sha256,
    strict_json,
    worker_environment,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_projections import (
    FIT_TASK_IDS,
    BfclV4FullFitFeedback,
    BfclV4ScoreFitAggregate,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)

_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C",
    "LC_ALL": "C",
}
_WORKER_ENVIRONMENT = worker_environment()
_ROSTER_BY_ID = {item.task_id: item for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster}


class BfclV4PublicGraderError(RuntimeError):
    """The public-development grader failed closed."""


def _resolve_system_git() -> Path:
    candidate = shutil.which("git", path=os.defpath)
    if candidate is None:
        raise BfclV4PublicGraderError("system Git executable unavailable")
    try:
        resolved = Path(candidate).resolve(strict=True)
    except OSError as error:
        raise BfclV4PublicGraderError("system Git executable unavailable") from error
    if not resolved.is_file():
        raise BfclV4PublicGraderError("system Git executable is not a file")
    return resolved


def _run_git(git: Path, checkout: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            [str(git), "-C", str(checkout), *arguments],
            env=dict(_GIT_ENVIRONMENT),
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BfclV4PublicGraderError("pinned Git object read failed") from error
    if completed.returncode != 0:
        raise BfclV4PublicGraderError(
            f"pinned Git object read failed (stderr_sha256={sha256_bytes(completed.stderr)})"
        )
    return completed.stdout


def _checkout_and_git(checkout: str | Path) -> tuple[Path, Path]:
    try:
        resolved = Path(checkout).resolve(strict=True)
    except OSError as error:
        raise BfclV4PublicGraderError("BFCL checkout is unavailable") from error
    if not resolved.is_dir():
        raise BfclV4PublicGraderError("BFCL checkout is not a directory")
    git = _resolve_system_git()
    try:
        head = _run_git(git, resolved, "rev-parse", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise BfclV4PublicGraderError("BFCL checkout HEAD is not ASCII") from error
    if head != BFCL_V4_UPSTREAM_COMMIT:
        raise BfclV4PublicGraderError("BFCL checkout is not at the pinned commit")
    return resolved, git


def _git_blob(git: Path, checkout: Path, git_path: str) -> bytes:
    return _run_git(
        git,
        checkout,
        "cat-file",
        "blob",
        f"{BFCL_V4_UPSTREAM_COMMIT}:{git_path}",
    )


def _jsonl_entry(blob: bytes, entry_id: str, label: str) -> tuple[dict[str, Any], bytes]:
    matches: list[tuple[dict[str, Any], bytes]] = []
    lines = blob.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.endswith(b"\r\n") or (not line.endswith(b"\n") and index != len(lines) - 1):
            raise BfclV4PublicGraderError(f"{label} is not canonical LF-delimited JSONL")
        payload = line[:-1] if line.endswith(b"\n") else line
        try:
            value = strict_json(payload.decode("utf-8"), label)
        except (UnicodeDecodeError, ValueError) as error:
            raise BfclV4PublicGraderError(f"{label} contains invalid JSONL") from error
        if isinstance(value, dict) and value.get("id") == entry_id:
            matches.append((value, line))
    if len(matches) != 1:
        raise BfclV4PublicGraderError(f"{label} does not contain exactly one pilot row")
    return matches[0]


def _load_public_question_binding(
    git: Path,
    checkout: Path,
    task_id: str,
) -> BfclV4PublicQuestionBinding:
    """Read only public question/schema bytes; never open possible answers."""

    try:
        roster = _ROSTER_BY_ID[task_id]
    except KeyError as error:
        raise BfclV4PublicGraderError("task is outside the frozen pilot roster") from error
    question_blob = _git_blob(git, checkout, roster.question_git_path)
    task, row = _jsonl_entry(question_blob, task_id, "BFCL public question blob")
    functions = task.get("function")
    if not isinstance(functions, list) or not functions:
        raise BfclV4PublicGraderError("BFCL public task has no function schemas")
    names = tuple(item.get("name") for item in functions if isinstance(item, dict))
    if len(names) != len(functions) or any(not isinstance(name, str) for name in names):
        raise BfclV4PublicGraderError("BFCL public task has invalid function names")
    try:
        return BfclV4PublicQuestionBinding(
            task_id=task_id,
            category=roster.category,
            split=roster.split,
            question_git_path=roster.question_git_path,
            question_blob_size=len(question_blob),
            question_blob_sha256=sha256_bytes(question_blob),
            question_row_size=len(row),
            question_row_sha256=sha256_bytes(row),
            question_sha256=canonical_sha256(task.get("question")),
            function_schemas_sha256=canonical_sha256(functions),
            official_function_names=names,
        )
    except ValueError as error:
        raise BfclV4PublicGraderError("public question differs from its pinned binding") from error


def make_bfcl_v4_public_prediction(
    task_id: str,
    calls: Sequence[tuple[str, Mapping[str, Any]]],
) -> BfclV4PublicPrediction:
    """Freeze only official-name plus argument-object candidate output."""

    if isinstance(calls, (str, bytes, bytearray)) or not isinstance(calls, Sequence):
        raise TypeError("prediction calls must be an ordered sequence")
    frozen: list[BfclV4OfficialPredictionCall] = []
    for item in calls:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each prediction call must be a (name, arguments) tuple")
        function_name, arguments = item
        if not isinstance(function_name, str) or not isinstance(arguments, Mapping):
            raise TypeError("prediction calls require a string name and argument mapping")
        frozen.append(
            BfclV4OfficialPredictionCall(
                function_name=function_name,
                arguments_json=canonical_json(arguments),
            )
        )
    return BfclV4PublicPrediction(task_id=task_id, calls=tuple(frozen))


def _source_bindings(
    git: Path,
    checkout: Path,
) -> tuple[BfclV4SourceFileBinding, ...]:
    bindings: list[BfclV4SourceFileBinding] = []
    for git_path, expected_sha256 in sorted(SOURCE_SHA256.items()):
        content = _git_blob(git, checkout, git_path)
        if sha256_bytes(content) != expected_sha256:
            raise BfclV4PublicGraderError(f"pinned grader source differs: {git_path}")
        bindings.append(
            BfclV4SourceFileBinding(
                git_path=git_path,
                size=len(content),
                sha256=expected_sha256,
            )
        )
    return tuple(bindings)


def _executable_binding(path: Path, version: str) -> BfclV4ExecutableBinding:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise BfclV4PublicGraderError("grader executable binding failed") from error
    return BfclV4ExecutableBinding(
        path=str(path),
        size_observation=len(content),
        sha256_observation=sha256_bytes(content),
        version_observation=version,
    )


def _worker_output(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
        if not text.endswith("\n") or "\n" in text[:-1]:
            raise ValueError("worker output is not exactly one JSONL row")
        output = strict_json(text[:-1], "public grader worker output")
    except (UnicodeDecodeError, ValueError) as error:
        raise BfclV4PublicGraderError("public grader worker output is invalid") from error
    if not isinstance(output, dict):
        raise BfclV4PublicGraderError("public grader worker output is not an object")
    return output


def grade_bfcl_v4_public_prediction(
    prediction: BfclV4PublicPrediction,
    slot: BfclV4GradingSlotBinding,
    checkout: str | Path,
    *,
    holdout_unlock: BfclV4HoldoutUnlock | None = None,
    timeout_seconds: float = 30.0,
) -> BfclV4PublicGraderReceipt:
    """Grade one prediction without exposing answer bytes outside the worker."""

    checked_prediction = checked(prediction, BfclV4PublicPrediction)
    checked_slot = checked(slot, BfclV4GradingSlotBinding)
    if checked_slot.task_id != checked_prediction.task_id:
        raise ValueError("grading slot and prediction task differ")
    if checked_slot.prediction_sha256 != checked_prediction.fingerprint:
        raise ValueError("grading slot points to another prediction")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be numeric")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    checked_unlock = (
        checked(holdout_unlock, BfclV4HoldoutUnlock) if holdout_unlock is not None else None
    )

    resolved, git = _checkout_and_git(checkout)
    question = _load_public_question_binding(git, resolved, checked_prediction.task_id)
    if question.split.value == "holdout":
        if checked_unlock is None:
            raise ValueError("HOLDOUT grading requires both frozen selection artifacts")
        if checked_unlock.plan_fingerprint != checked_slot.plan_fingerprint:
            raise ValueError("HOLDOUT unlock belongs to another plan")
    elif checked_unlock is not None:
        raise ValueError("non-HOLDOUT grade must not carry a HOLDOUT unlock")

    sources = _source_bindings(git, resolved)
    bundle_sha256 = source_bundle_sha256(sources)
    worker_source = current_public_grader_worker_binding()
    worker = Path(worker_source.path)
    python = Path(sys.executable).resolve(strict=True)
    try:
        git_version = _run_git(git, resolved, "--version").decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise BfclV4PublicGraderError("Git version output is not UTF-8") from error
    python_binding = _executable_binding(
        python,
        f"{platform.python_implementation()} {platform.python_version()}",
    )
    git_binding = _executable_binding(git, git_version)
    argv = (
        str(python),
        "-I",
        "-B",
        str(worker),
        "--checkout",
        str(resolved),
        "--git",
        str(git),
    )
    environment = tuple(
        BfclV4EnvironmentEntry(name=name, value=value)
        for name, value in sorted(_WORKER_ENVIRONMENT.items())
    )
    runtime = runtime_coordinates()
    environment_sha256 = canonical_sha256(
        {
            "entries": environment,
            "git": git_binding,
            "platform": runtime,
            "python": python_binding,
        }
    )
    prediction_json = canonical_json(prediction_content(checked_prediction))
    unlock_sha256 = checked_unlock.fingerprint if checked_unlock is not None else None
    request = {
        "git_executable": {
            "path": git_binding.path,
            "sha256": git_binding.sha256_observation,
            "size": git_binding.size_observation,
            "version": git_binding.version_observation,
        },
        "holdout_unlock_sha256": unlock_sha256,
        "prediction_json": prediction_json,
        "prediction_sha256": checked_prediction.fingerprint,
        "protocol": PUBLIC_GRADER_PROTOCOL,
        "question_binding_sha256": question.fingerprint,
        "slot_binding_sha256": checked_slot.fingerprint,
        "source_bundle_sha256": bundle_sha256,
        "task_id": checked_prediction.task_id,
    }
    try:
        completed = subprocess.run(
            argv,
            input=canonical_json_bytes(request),
            env=dict(_WORKER_ENVIRONMENT),
            check=False,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BfclV4PublicGraderError("public grader worker failed") from error
    stdout_sha256 = sha256_bytes(completed.stdout)
    stderr_sha256 = sha256_bytes(completed.stderr)
    if completed.returncode != 0:
        raise BfclV4PublicGraderError(
            "public grader worker rejected the invocation "
            f"(returncode={completed.returncode}, stdout_sha256={stdout_sha256}, "
            f"stderr_sha256={stderr_sha256})"
        )
    if completed.stderr:
        raise BfclV4PublicGraderError(
            f"public grader worker emitted stderr on success (stderr_sha256={stderr_sha256})"
        )
    output = _worker_output(completed.stdout)
    expected_keys = {
        "answer_binding",
        "coarse_failure_class",
        "credential_environment_inherited",
        "error_type",
        "exact_upstream_ast_checker_executed",
        "full_upstream_dependency_graph_loaded",
        "git_executable_sha256",
        "holdout_unlock_sha256",
        "model_invoked",
        "network_calls_requested",
        "network_isolation_attested",
        "official_cli_executed",
        "official_score_produced",
        "prediction_sha256",
        "protocol",
        "proxy_environment_inherited",
        "question_binding",
        "slot_binding_sha256",
        "source_bundle_sha256",
        "task_id",
        "upstream_ast_checker_valid",
    }
    if set(output) != expected_keys:
        raise BfclV4PublicGraderError("public grader worker output schema changed")
    expected_fixed = {
        "credential_environment_inherited": False,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "git_executable_sha256": git_binding.sha256_observation,
        "holdout_unlock_sha256": unlock_sha256,
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "prediction_sha256": checked_prediction.fingerprint,
        "protocol": PUBLIC_GRADER_PROTOCOL,
        "proxy_environment_inherited": False,
        "slot_binding_sha256": checked_slot.fingerprint,
        "source_bundle_sha256": bundle_sha256,
        "task_id": checked_prediction.task_id,
    }
    if any(output.get(key) != value for key, value in expected_fixed.items()):
        raise BfclV4PublicGraderError("public grader worker binding changed")
    if not isinstance(output["question_binding"], dict) or not isinstance(
        output["answer_binding"], dict
    ):
        raise BfclV4PublicGraderError("public grader worker binding payload is invalid")
    raw_question = dict(output["question_binding"])
    try:
        raw_question["split"] = BfclV4PilotSplit(raw_question["split"])
        raw_question["official_function_names"] = tuple(raw_question["official_function_names"])
    except (KeyError, TypeError, ValueError) as error:
        raise BfclV4PublicGraderError("worker question binding types are invalid") from error
    worker_question = BfclV4PublicQuestionBinding.model_validate(raw_question, strict=True)
    if worker_question != question:
        raise BfclV4PublicGraderError("worker recomputed a different public question binding")
    answer = BfclV4GraderAnswerBinding.model_validate(output["answer_binding"], strict=True)
    valid = output["upstream_ast_checker_valid"]
    error_type = output["error_type"]
    coarse = output["coarse_failure_class"]
    if not isinstance(valid, bool):
        raise BfclV4PublicGraderError("public grader validity is not boolean")
    if valid and error_type is not None:
        raise BfclV4PublicGraderError("valid public grader output retained an error type")
    if not valid and (not isinstance(error_type, str) or not error_type):
        raise BfclV4PublicGraderError("invalid public grader output omitted its error type")
    if coarse not in {"none", "call-count", "function-or-arguments"}:
        raise BfclV4PublicGraderError("public grader coarse failure class is invalid")

    return BfclV4PublicGraderReceipt(
        prediction=checked_prediction,
        prediction_sha256=checked_prediction.fingerprint,
        slot=checked_slot,
        slot_binding_sha256=checked_slot.fingerprint,
        question_binding=question,
        answer_binding=answer,
        holdout_unlock=checked_unlock,
        holdout_unlock_sha256=unlock_sha256,
        executed_sources=sources,
        source_bundle_sha256=bundle_sha256,
        worker_source=worker_source,
        checkout_path=str(resolved),
        argv=argv,
        argv_sha256=canonical_sha256(argv),
        environment=environment,
        environment_sha256=environment_sha256,
        runtime=runtime,
        python_executable=python_binding,
        git_executable=git_binding,
        worker_stdout_size=len(completed.stdout),
        worker_stdout_sha256=stdout_sha256,
        upstream_ast_checker_valid=valid,
        checker_error_type=error_type,
        coarse_failure_class=coarse,
    )


def project_bfcl_v4_full_fit_feedback(
    prediction: BfclV4PublicPrediction,
    receipt: BfclV4PublicGraderReceipt,
) -> BfclV4FullFitFeedback:
    """Release only a FULL arm's own FIT binary/coarse result."""

    checked_prediction = checked(prediction, BfclV4PublicPrediction)
    checked_receipt = checked(receipt, BfclV4PublicGraderReceipt)
    if checked_receipt.prediction_sha256 != checked_prediction.fingerprint:
        raise ValueError("FULL receipt belongs to another prediction")
    if checked_receipt.slot.arm != "full":
        raise ValueError("FULL feedback requires a FULL-arm receipt")
    if checked_receipt.slot.grade_role not in {"parent-fit", "candidate-fit"}:
        raise ValueError("FULL feedback requires a FIT receipt")
    failure = checked_receipt.coarse_failure_class
    if failure == "execution-failure":
        raise ValueError("successful grader receipt cannot contain execution failure")
    return BfclV4FullFitFeedback(
        task_id=checked_prediction.task_id,
        own_prediction_reference_sha256=checked_prediction.fingerprint,
        accepted=checked_receipt.upstream_ast_checker_valid,
        failure_class=failure,
    )


def project_bfcl_v4_score_fit_aggregate(
    receipts: tuple[BfclV4PublicGraderReceipt, ...],
    expected_slot_references: tuple[str, ...],
) -> BfclV4ScoreFitAggregate:
    """Release one complete SCORE 5-FIT aggregate and no task-wise labels."""

    if not isinstance(receipts, tuple) or not isinstance(expected_slot_references, tuple):
        raise TypeError("SCORE aggregation inputs must be tuples")
    if len(receipts) != 5 or len(expected_slot_references) != 5:
        raise ValueError("SCORE aggregation requires the complete five-FIT batch")
    checked_receipts = tuple(checked(receipt, BfclV4PublicGraderReceipt) for receipt in receipts)
    if tuple(receipt.prediction.task_id for receipt in checked_receipts) != FIT_TASK_IDS:
        raise ValueError("SCORE receipts differ from the frozen five-FIT order")
    if (
        tuple(receipt.slot.call_slot_reference_sha256 for receipt in checked_receipts)
        != expected_slot_references
    ):
        raise ValueError("SCORE receipts differ from the expected call slots")
    if len({receipt.fingerprint for receipt in checked_receipts}) != 5:
        raise ValueError("SCORE receipts must be unique")
    plans = {receipt.slot.plan_fingerprint for receipt in checked_receipts}
    roles = {receipt.slot.grade_role for receipt in checked_receipts}
    if len(plans) != 1 or any(receipt.slot.arm != "score" for receipt in checked_receipts):
        raise ValueError("SCORE receipts must belong to one SCORE-arm plan")
    if len(roles) != 1 or not roles <= {"parent-fit", "candidate-fit"}:
        raise ValueError("SCORE receipts must represent one complete FIT phase")
    receipt_fingerprints = tuple(receipt.fingerprint for receipt in checked_receipts)
    batch_reference = canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-score-five-fit-batch/v1",
            "plan_fingerprint": next(iter(plans)),
            "receipt_fingerprints": receipt_fingerprints,
        }
    )
    correct = sum(receipt.upstream_ast_checker_valid for receipt in checked_receipts)
    return BfclV4ScoreFitAggregate(
        plan_fingerprint=next(iter(plans)),
        batch_reference_sha256=batch_reference,
        aggregate_accuracy_basis_points=correct * 2_000,
    )


__all__ = [
    "BfclV4PublicGraderError",
    "grade_bfcl_v4_public_prediction",
    "make_bfcl_v4_public_prediction",
    "project_bfcl_v4_full_fit_feedback",
    "project_bfcl_v4_score_fit_aggregate",
]
