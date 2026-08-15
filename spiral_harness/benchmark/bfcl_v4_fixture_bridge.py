"""Provider-free adapter and isolated execution boundary for one BFCL V4 fixture."""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spiral_harness.benchmark._bfcl_v4_fixture_worker import (
    ANSWER_BLOB_SHA256,
    ANSWER_PATH,
    ANSWER_ROW_SHA256,
    QUESTION_BLOB_SHA256,
    QUESTION_PATH,
    SOURCE_SHA256,
    TASK_ID,
    TASK_ROW_SHA256,
    UPSTREAM_COMMIT,
)
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import (
    ANSWER_BLOB_SIZE,
    ANSWER_ROW_SIZE,
    CLI_PATH,
    CLI_SHA256,
    EVAL_RUNNER_PATH,
    EVAL_RUNNER_SHA256,
    FIXTURE_PROTOCOL,
    FULL_CLI_ARGV_TEMPLATE,
    FULL_CLI_ENVIRONMENT_REQUIREMENTS,
    FUNCTION_SCHEMAS_SHA256,
    GROUND_TRUTH_SHA256,
    OPENAI_DECODER_PATH,
    OPENAI_DECODER_SHA256,
    PYPROJECT_PATH,
    PYPROJECT_SHA256,
    QUESTION_BLOB_SIZE,
    QUESTION_SHA256,
    TASK_ROW_SIZE,
    BfclV4CandidateSafeFeedback,
    BfclV4EnvironmentEntry,
    BfclV4ExecutableBinding,
    BfclV4FullCliInvocationContract,
    BfclV4GraderOnlyAnswerBinding,
    BfclV4NativeToolCall,
    BfclV4OfficialResultExport,
    BfclV4PublicAstFixture,
    BfclV4SourceFileBinding,
    BfclV4SourceIsolatedReceipt,
    checked,
    current_worker_source_binding,
    fixture_contract_source_sha256,
    runtime_coordinates,
    serialize_official_result_row,
    source_bundle_sha256,
    strict_json,
    worker_environment,
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


class BfclV4FixtureBridgeError(RuntimeError):
    """The pinned fixture bridge failed closed."""


def _resolve_system_git() -> Path:
    candidate = shutil.which("git", path=os.defpath)
    if candidate is None:
        raise BfclV4FixtureBridgeError("system Git executable unavailable")
    try:
        resolved = Path(candidate).resolve(strict=True)
    except OSError as error:
        raise BfclV4FixtureBridgeError("system Git executable unavailable") from error
    if not resolved.is_file():
        raise BfclV4FixtureBridgeError("system Git executable is not a file")
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
        raise BfclV4FixtureBridgeError("pinned Git object read failed") from error
    if completed.returncode != 0:
        raise BfclV4FixtureBridgeError(
            f"pinned Git object read failed (stderr_sha256={sha256_bytes(completed.stderr)})"
        )
    return completed.stdout


def _checkout_and_git(checkout: str | Path) -> tuple[Path, Path]:
    try:
        resolved = Path(checkout).resolve(strict=True)
    except OSError as error:
        raise BfclV4FixtureBridgeError("BFCL checkout is unavailable") from error
    if not resolved.is_dir():
        raise BfclV4FixtureBridgeError("BFCL checkout is not a directory")
    git = _resolve_system_git()
    try:
        head = _run_git(git, resolved, "rev-parse", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise BfclV4FixtureBridgeError("BFCL checkout HEAD is not ASCII") from error
    if head != UPSTREAM_COMMIT:
        raise BfclV4FixtureBridgeError("BFCL checkout is not at the pinned commit")
    return resolved, git


def _git_blob(git: Path, checkout: Path, git_path: str) -> bytes:
    return _run_git(
        git,
        checkout,
        "cat-file",
        "blob",
        f"{UPSTREAM_COMMIT}:{git_path}",
    )


def _jsonl_entry(blob: bytes, entry_id: str, label: str) -> tuple[dict[str, Any], bytes]:
    matches: list[tuple[dict[str, Any], bytes]] = []
    lines = blob.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.endswith(b"\r\n") or (not line.endswith(b"\n") and index != len(lines) - 1):
            raise BfclV4FixtureBridgeError(f"{label} is not canonical LF-delimited JSONL")
        payload = line[:-1] if line.endswith(b"\n") else line
        try:
            value = strict_json(payload.decode("utf-8"), label)
        except (UnicodeDecodeError, ValueError) as error:
            raise BfclV4FixtureBridgeError(f"{label} contains invalid JSONL") from error
        if isinstance(value, dict) and value.get("id") == entry_id:
            matches.append((value, line))
    if len(matches) != 1:
        raise BfclV4FixtureBridgeError(f"{label} does not contain exactly one fixture row")
    return matches[0]


def load_bfcl_v4_public_ast_fixture(checkout: str | Path) -> BfclV4PublicAstFixture:
    """Load candidate-visible bytes without reading the possible-answer blob."""

    resolved, git = _checkout_and_git(checkout)
    question_blob = _git_blob(git, resolved, QUESTION_PATH)
    if (
        len(question_blob) != QUESTION_BLOB_SIZE
        or sha256_bytes(question_blob) != QUESTION_BLOB_SHA256
    ):
        raise BfclV4FixtureBridgeError("pinned BFCL question blob differs from the fixture")
    task, task_row = _jsonl_entry(question_blob, TASK_ID, "BFCL question blob")
    if len(task_row) != TASK_ROW_SIZE or sha256_bytes(task_row) != TASK_ROW_SHA256:
        raise BfclV4FixtureBridgeError("pinned BFCL task row differs from the fixture")
    if canonical_sha256(task.get("question")) != QUESTION_SHA256:
        raise BfclV4FixtureBridgeError("pinned BFCL question payload differs from the fixture")
    schemas = task.get("function")
    if canonical_sha256(schemas) != FUNCTION_SCHEMAS_SHA256 or not isinstance(schemas, list):
        raise BfclV4FixtureBridgeError("pinned BFCL function schemas differ from the fixture")
    names = tuple(item.get("name") for item in schemas if isinstance(item, dict))
    if len(names) != len(schemas) or any(not isinstance(name, str) for name in names):
        raise BfclV4FixtureBridgeError("pinned BFCL function-name roster is invalid")
    return BfclV4PublicAstFixture(
        function_schemas_json=canonical_json(schemas),
        function_names=names,
    )


def _load_grader_only_answer_binding(
    git: Path,
    checkout: Path,
) -> BfclV4GraderOnlyAnswerBinding:
    answer_blob = _git_blob(git, checkout, ANSWER_PATH)
    if len(answer_blob) != ANSWER_BLOB_SIZE or sha256_bytes(answer_blob) != ANSWER_BLOB_SHA256:
        raise BfclV4FixtureBridgeError("pinned BFCL answer blob differs from the grader binding")
    answer, answer_row = _jsonl_entry(answer_blob, TASK_ID, "BFCL grader-only answer blob")
    if len(answer_row) != ANSWER_ROW_SIZE or sha256_bytes(answer_row) != ANSWER_ROW_SHA256:
        raise BfclV4FixtureBridgeError("pinned BFCL answer row differs from the grader binding")
    if canonical_sha256(answer.get("ground_truth")) != GROUND_TRUTH_SHA256:
        raise BfclV4FixtureBridgeError("pinned BFCL ground truth differs from the grader binding")
    return BfclV4GraderOnlyAnswerBinding()


def make_bfcl_v4_native_tool_call(
    function_name: str,
    arguments: Mapping[str, Any],
) -> BfclV4NativeToolCall:
    """Freeze one provider-neutral call without consulting grader-only data."""

    if not isinstance(arguments, Mapping):
        raise TypeError("tool-call arguments must be a mapping")
    return BfclV4NativeToolCall(
        function_name=function_name,
        arguments_json=canonical_json(arguments),
    )


def export_bfcl_v4_official_result_row(
    fixture: BfclV4PublicAstFixture,
    calls: tuple[BfclV4NativeToolCall, ...],
) -> BfclV4OfficialResultExport:
    """Export one BFCL-format JSONL row with no answer or grader access."""

    checked_fixture = checked(fixture, BfclV4PublicAstFixture)
    if not isinstance(calls, tuple) or len(calls) != 1:
        raise ValueError("the simple AST fixture requires exactly one native tool call")
    checked_calls = tuple(checked(call, BfclV4NativeToolCall) for call in calls)
    if any(call.function_name not in checked_fixture.function_names for call in checked_calls):
        raise ValueError("native response invokes a function outside the task schemas")
    row_bytes = serialize_official_result_row(checked_fixture, checked_calls)
    return BfclV4OfficialResultExport(
        fixture=checked_fixture,
        calls=checked_calls,
        native_response_sha256=canonical_sha256(checked_calls),
        serializer_source_sha256=fixture_contract_source_sha256(),
        response_row_jsonl=row_bytes,
        response_row_size=len(row_bytes),
        response_row_sha256=sha256_bytes(row_bytes),
    )


def _source_bindings(
    git: Path,
    checkout: Path,
    expected: Mapping[str, str],
) -> tuple[BfclV4SourceFileBinding, ...]:
    bindings: list[BfclV4SourceFileBinding] = []
    for git_path, expected_sha256 in sorted(expected.items()):
        content = _git_blob(git, checkout, git_path)
        if sha256_bytes(content) != expected_sha256:
            raise BfclV4FixtureBridgeError(f"pinned source differs: {git_path}")
        bindings.append(
            BfclV4SourceFileBinding(
                git_path=git_path,
                size=len(content),
                sha256=expected_sha256,
            )
        )
    return tuple(bindings)


def _executable_binding(path: Path, version: str) -> BfclV4ExecutableBinding:
    content = path.read_bytes()
    return BfclV4ExecutableBinding(
        path=str(path),
        size_observation=len(content),
        sha256_observation=sha256_bytes(content),
        version_observation=version,
    )


def _worker_output(stdout: bytes) -> dict[str, Any]:
    try:
        output_text = stdout.decode("utf-8")
        if not output_text.endswith("\n") or "\n" in output_text[:-1]:
            raise ValueError("worker output is not exactly one JSONL row")
        output = strict_json(output_text[:-1], "fixture worker output")
    except (UnicodeDecodeError, ValueError) as error:
        raise BfclV4FixtureBridgeError(
            "source-isolated fixture worker output is invalid"
        ) from error
    if not isinstance(output, dict):
        raise BfclV4FixtureBridgeError("source-isolated fixture worker output is not an object")
    return output


def invoke_bfcl_v4_source_isolated_fixture(
    export: BfclV4OfficialResultExport,
    checkout: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> BfclV4SourceIsolatedReceipt:
    """Execute exact upstream AST-checker source under a credential-free env."""

    checked_export = checked(export, BfclV4OfficialResultExport)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be numeric")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    resolved, git = _checkout_and_git(checkout)
    if load_bfcl_v4_public_ast_fixture(resolved) != checked_export.fixture:
        raise BfclV4FixtureBridgeError("export fixture differs from the pinned checkout")
    grader_binding = _load_grader_only_answer_binding(git, resolved)
    sources = _source_bindings(git, resolved, SOURCE_SHA256)
    bundle_sha256 = source_bundle_sha256(sources)
    worker = Path(__file__).with_name("_bfcl_v4_fixture_worker.py").resolve(strict=True)
    worker_source = current_worker_source_binding()
    if worker_source.path != str(worker):
        raise BfclV4FixtureBridgeError("current worker source path changed")
    python = Path(sys.executable).resolve(strict=True)
    try:
        git_version = _run_git(git, resolved, "--version").decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise BfclV4FixtureBridgeError("Git version output is not UTF-8") from error
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
    request = {
        "git_executable": {
            "path": git_binding.path,
            "sha256": git_binding.sha256_observation,
            "size": git_binding.size_observation,
            "version": git_binding.version_observation,
        },
        "grader_answer_binding_sha256": grader_binding.fingerprint,
        "protocol": FIXTURE_PROTOCOL,
        "response_row_jsonl": checked_export.response_row_jsonl.decode("utf-8"),
        "response_row_sha256": checked_export.response_row_sha256,
        "source_bundle_sha256": bundle_sha256,
        "task_binding_sha256": checked_export.fixture.fingerprint,
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
        raise BfclV4FixtureBridgeError("source-isolated fixture worker failed") from error
    stdout_sha256 = sha256_bytes(completed.stdout)
    stderr_sha256 = sha256_bytes(completed.stderr)
    if completed.returncode != 0:
        raise BfclV4FixtureBridgeError(
            "source-isolated fixture worker rejected the invocation "
            f"(returncode={completed.returncode}, stdout_sha256={stdout_sha256}, "
            f"stderr_sha256={stderr_sha256})"
        )
    if completed.stderr:
        raise BfclV4FixtureBridgeError(
            "source-isolated fixture worker emitted stderr on success "
            f"(stderr_observation_sha256={stderr_sha256})"
        )
    output = _worker_output(completed.stdout)
    expected_keys = {
        "credential_environment_inherited",
        "error_type",
        "exact_upstream_ast_checker_executed",
        "full_upstream_dependency_graph_loaded",
        "git_executable_sha256",
        "grader_answer_binding_sha256",
        "model_invoked",
        "network_calls_requested",
        "network_isolation_attested",
        "official_cli_executed",
        "official_score_produced",
        "protocol",
        "proxy_environment_inherited",
        "response_row_sha256",
        "source_bundle_sha256",
        "task_binding_sha256",
        "upstream_ast_checker_valid",
    }
    if set(output) != expected_keys:
        raise BfclV4FixtureBridgeError("source-isolated fixture worker schema changed")
    expected_fixed = {
        "credential_environment_inherited": False,
        "exact_upstream_ast_checker_executed": True,
        "full_upstream_dependency_graph_loaded": False,
        "git_executable_sha256": git_binding.sha256_observation,
        "grader_answer_binding_sha256": grader_binding.fingerprint,
        "model_invoked": False,
        "network_calls_requested": False,
        "network_isolation_attested": False,
        "official_cli_executed": False,
        "official_score_produced": False,
        "protocol": FIXTURE_PROTOCOL,
        "proxy_environment_inherited": False,
        "response_row_sha256": checked_export.response_row_sha256,
        "source_bundle_sha256": bundle_sha256,
        "task_binding_sha256": checked_export.fixture.fingerprint,
    }
    if any(output.get(key) != value for key, value in expected_fixed.items()):
        raise BfclV4FixtureBridgeError("source-isolated fixture worker binding changed")
    if not isinstance(output["upstream_ast_checker_valid"], bool):
        raise BfclV4FixtureBridgeError("source-isolated fixture validity is not boolean")
    error_type = output["error_type"]
    valid = output["upstream_ast_checker_valid"]
    if valid and error_type is not None:
        raise BfclV4FixtureBridgeError("successful fixture output retained an error type")
    if not valid and (not isinstance(error_type, str) or not error_type):
        raise BfclV4FixtureBridgeError("source-isolated fixture error type is invalid")
    return BfclV4SourceIsolatedReceipt(
        fixture_export_observation_sha256=checked_export.fingerprint,
        worker_recomputed_task_binding_sha256=checked_export.fixture.fingerprint,
        grader_answer_binding=grader_binding,
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
        response_row_observation_sha256=checked_export.response_row_sha256,
        worker_stdout_size=len(completed.stdout),
        worker_stdout_sha256=stdout_sha256,
        upstream_ast_checker_valid=valid,
        checker_error_type=error_type,
    )


def project_bfcl_v4_candidate_safe_feedback(
    export: BfclV4OfficialResultExport,
    receipt: BfclV4SourceIsolatedReceipt,
) -> BfclV4CandidateSafeFeedback:
    """Release only binary public-fixture feedback; retain trusted identities."""

    checked_export = checked(export, BfclV4OfficialResultExport)
    checked_receipt = checked(receipt, BfclV4SourceIsolatedReceipt)
    if checked_receipt.fixture_export_observation_sha256 != checked_export.fingerprint:
        raise ValueError("auditor receipt does not belong to the exported fixture response")
    if checked_receipt.worker_recomputed_task_binding_sha256 != checked_export.fixture.fingerprint:
        raise ValueError("auditor receipt task binding differs from the candidate fixture")
    if checked_receipt.response_row_observation_sha256 != checked_export.response_row_sha256:
        raise ValueError("auditor receipt response row differs from the candidate response")
    accepted = checked_receipt.upstream_ast_checker_valid
    return BfclV4CandidateSafeFeedback(
        response_row_reference_sha256=checked_export.response_row_sha256,
        accepted=accepted,
        failure_class="none" if accepted else "ast-mismatch",
    )


def build_bfcl_v4_full_cli_invocation_contract(
    checkout: str | Path,
) -> BfclV4FullCliInvocationContract:
    """Bind the separate full CLI path without claiming it is runnable or run."""

    resolved, git = _checkout_and_git(checkout)
    expected = {
        CLI_PATH: CLI_SHA256,
        EVAL_RUNNER_PATH: EVAL_RUNNER_SHA256,
        OPENAI_DECODER_PATH: OPENAI_DECODER_SHA256,
        PYPROJECT_PATH: PYPROJECT_SHA256,
    }
    sources = _source_bindings(git, resolved, expected)
    return BfclV4FullCliInvocationContract(
        source_files=sources,
        source_bundle_sha256=source_bundle_sha256(sources),
        argv_template_sha256=canonical_sha256(FULL_CLI_ARGV_TEMPLATE),
        environment_contract_sha256=canonical_sha256(FULL_CLI_ENVIRONMENT_REQUIREMENTS),
    )


__all__ = [
    "BfclV4FixtureBridgeError",
    "build_bfcl_v4_full_cli_invocation_contract",
    "export_bfcl_v4_official_result_row",
    "invoke_bfcl_v4_source_isolated_fixture",
    "load_bfcl_v4_public_ast_fixture",
    "make_bfcl_v4_native_tool_call",
    "project_bfcl_v4_candidate_safe_feedback",
]
