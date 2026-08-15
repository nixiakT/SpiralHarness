from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark._bfcl_v4_fixture_worker as worker_subject
import spiral_harness.benchmark.bfcl_v4_fixture_bridge as subject
import spiral_harness.benchmark.bfcl_v4_fixture_contracts as contracts_subject
from spiral_harness.benchmark._bfcl_v4_fixture_worker import (
    ANSWER_PATH,
    SOURCE_SHA256,
)
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import (
    FIXTURE_PROTOCOL,
    BfclV4FullCliInvocationContract,
    BfclV4NativeToolCall,
    BfclV4OfficialResultExport,
    BfclV4PublicAstFixture,
    BfclV4SourceIsolatedReceipt,
)
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
)

_FUNCTION_SCHEMAS_JSON = (
    '[{"description":"Calculate the area of a triangle given its base and height.",'
    '"name":"calculate_triangle_area","parameters":{"properties":{"base":'
    '{"description":"The base of the triangle.","type":"integer"},"height":'
    '{"description":"The height of the triangle.","type":"integer"},"unit":'
    '{"description":"The unit of measure (defaults to \'units\' if not specified)",'
    '"type":"string"}},"required":["base","height"],"type":"dict"}}]'
)
_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


def _fixture() -> BfclV4PublicAstFixture:
    return BfclV4PublicAstFixture(
        function_schemas_json=_FUNCTION_SCHEMAS_JSON,
        function_names=("calculate_triangle_area",),
    )


def _export(*, base: int = 10) -> BfclV4OfficialResultExport:
    call = subject.make_bfcl_v4_native_tool_call(
        "calculate_triangle_area",
        {"height": 5, "base": base},
    )
    return subject.export_bfcl_v4_official_result_row(_fixture(), (call,))


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact-source integration")
    try:
        subject.load_bfcl_v4_public_ast_fixture(checkout)
    except subject.BfclV4FixtureBridgeError as error:
        pytest.skip(f"pinned BFCL checkout is not usable: {error}")
    return checkout


@pytest.fixture(scope="module")
def accepted_export_and_receipt(
    pinned_checkout: Path,
) -> tuple[BfclV4OfficialResultExport, BfclV4SourceIsolatedReceipt]:
    exported = _export()
    return exported, subject.invoke_bfcl_v4_source_isolated_fixture(
        exported,
        pinned_checkout,
    )


def _environment_payload_sha256(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "entries": payload["environment"],
            "git": payload["git_executable"],
            "platform": payload["runtime"],
            "python": payload["python_executable"],
        }
    )


def _source_payload_bundle_sha256(
    sources: tuple[dict[str, Any], ...],
) -> str:
    return canonical_sha256(
        [
            {
                "path": item["git_path"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            for item in sources
        ]
    )


def _recursive_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(
            token for key, item in value.items() for token in (str(key), *_recursive_strings(item))
        )
    if isinstance(value, (list, tuple)):
        return tuple(token for item in value for token in _recursive_strings(item))
    return (str(value),)


def test_candidate_fixture_contains_no_answer_derived_identity() -> None:
    payload = _fixture().model_dump(mode="json")

    assert "answer_git_path" not in payload
    assert "answer_blob_sha256" not in payload
    assert "answer_row_sha256" not in payload
    assert "ground_truth_sha256" not in payload
    assert "ground_truth" not in canonical_json(payload)


def test_score_free_adapter_exports_exact_openai_fc_result_row() -> None:
    exported = _export()

    assert exported.response_row_jsonl == (
        b'{"id":"simple_python_0","result":'
        b'[{"calculate_triangle_area":"{\\"base\\":10,\\"height\\":5}"}]}\n'
    )
    assert exported.partial_evaluation is True
    assert exported.hidden_test_evidence is False
    assert exported.reportable_result is False
    serialized = canonical_json(exported)
    assert "a5499305963e3c5d0c4c67e75f9c8e9fc9cdf0e535de7d36361177fef6f7b8fb" not in serialized
    assert "48730d0bdcf0db0216edfce9f256f18f96c1b0312ffd6754f00702a2d2cd1501" not in serialized


@pytest.mark.parametrize(
    "arguments_json",
    ('{"height":5,"base":10}', "[10,5]", '{"base":10,"base":10}'),
)
def test_native_tool_call_rejects_noncanonical_or_nonobject_arguments(
    arguments_json: str,
) -> None:
    with pytest.raises(ValidationError):
        BfclV4NativeToolCall(
            function_name="calculate_triangle_area",
            arguments_json=arguments_json,
        )


def test_export_rejects_functions_outside_candidate_visible_schema() -> None:
    call = subject.make_bfcl_v4_native_tool_call("unknown", {"base": 10, "height": 5})

    with pytest.raises(ValueError, match="outside the task schemas"):
        subject.export_bfcl_v4_official_result_row(_fixture(), (call,))


def test_unchecked_construction_cannot_upgrade_nonreportable_flags() -> None:
    exported = _export()
    payload = exported.model_dump(mode="python")
    payload["reportable_result"] = True
    unchecked = BfclV4OfficialResultExport.model_construct(**payload)

    with pytest.raises(ValidationError):
        unchecked.model_dump(mode="json")
    with pytest.raises(ValidationError):
        canonical_json(unchecked)


def test_response_row_tampering_fails_revalidation() -> None:
    exported = _export()
    payload = exported.model_dump(mode="python")
    payload["response_row_jsonl"] = exported.response_row_jsonl.replace(b"10", b"11")

    with pytest.raises(ValidationError, match="response row differs"):
        BfclV4OfficialResultExport.model_validate(payload, strict=True)


def test_candidate_loader_never_reads_possible_answers(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_paths: list[str] = []
    original = subject._git_blob

    def recording_blob(git: Path, checkout: Path, git_path: str) -> bytes:
        observed_paths.append(git_path)
        return original(git, checkout, git_path)

    monkeypatch.setattr(subject, "_git_blob", recording_blob)
    loaded = subject.load_bfcl_v4_public_ast_fixture(pinned_checkout)

    assert loaded == _fixture()
    assert ANSWER_PATH not in observed_paths


def test_exact_upstream_ast_checker_positive_and_negative(
    pinned_checkout: Path,
) -> None:
    accepted = subject.invoke_bfcl_v4_source_isolated_fixture(_export(), pinned_checkout)
    rejected = subject.invoke_bfcl_v4_source_isolated_fixture(
        _export(base=9),
        pinned_checkout,
    )

    assert accepted.upstream_ast_checker_valid is True
    assert rejected.upstream_ast_checker_valid is False
    assert rejected.checker_error_type == "value_error:others"
    assert (
        subject.project_bfcl_v4_candidate_safe_feedback(_export(), accepted).failure_class == "none"
    )
    assert (
        subject.project_bfcl_v4_candidate_safe_feedback(_export(base=9), rejected).failure_class
        == "ast-mismatch"
    )
    for receipt in (accepted, rejected):
        assert receipt.visibility == "grader-auditor-only"
        assert receipt.candidate_visible is False
        assert receipt.grader_answer_binding.visibility == "grader-auditor-only"
        assert receipt.exact_upstream_ast_checker_executed is True
        assert receipt.full_upstream_dependency_graph_loaded is False
        assert receipt.provider_model_registry_stubbed is True
        assert receipt.official_cli_executed is False
        assert receipt.official_score_produced is False
        assert receipt.network_isolation_attested is False
        assert receipt.network_calls_requested is False
        assert receipt.reportable_result is False


def test_worker_subprocess_inherits_no_credentials_tokens_or_proxies(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "must-not-cross-process-boundary"
    sensitive_names = {
        "ALL_PROXY",
        "HF_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
    }
    for name in sensitive_names:
        monkeypatch.setenv(name, sentinel)
    observed_environments: list[dict[str, str]] = []
    real_run = subprocess.run

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environments.append(environment.copy())
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subject.subprocess, "run", recording_run)
    receipt = subject.invoke_bfcl_v4_source_isolated_fixture(_export(), pinned_checkout)

    assert observed_environments
    assert all(sensitive_names.isdisjoint(environment) for environment in observed_environments)
    assert sentinel not in canonical_json(receipt)
    assert receipt.credential_environment_inherited is False
    assert receipt.proxy_environment_inherited is False


@pytest.mark.parametrize(
    ("tampered_field", "failure_code"),
    (
        ("task_binding_sha256", "task-binding-hash-mismatch"),
        ("grader_answer_binding_sha256", "grader-answer-binding-hash-mismatch"),
    ),
)
def test_worker_recomputes_task_and_answer_bindings_instead_of_echoing(
    pinned_checkout: Path,
    tampered_field: str,
    failure_code: str,
) -> None:
    resolved, git = subject._checkout_and_git(pinned_checkout)
    exported = _export()
    sources = subject._source_bindings(git, resolved, SOURCE_SHA256)
    grader = subject._load_grader_only_answer_binding(git, resolved)
    git_version = subject._run_git(git, resolved, "--version").decode().strip()
    git_binding = subject._executable_binding(git, git_version)
    request = {
        "git_executable": {
            "path": git_binding.path,
            "sha256": git_binding.sha256_observation,
            "size": git_binding.size_observation,
            "version": git_binding.version_observation,
        },
        "grader_answer_binding_sha256": grader.fingerprint,
        "protocol": FIXTURE_PROTOCOL,
        "response_row_jsonl": exported.response_row_jsonl.decode(),
        "response_row_sha256": exported.response_row_sha256,
        "source_bundle_sha256": subject.source_bundle_sha256(sources),
        "task_binding_sha256": exported.fixture.fingerprint,
    }
    request[tampered_field] = "0" * 64
    worker = Path(subject.__file__).with_name("_bfcl_v4_fixture_worker.py").resolve()
    command = (
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        str(worker),
        "--checkout",
        str(resolved),
        "--git",
        str(git),
    )

    completed = subprocess.run(
        command,
        input=canonical_json_bytes(request),
        env=dict(subject._WORKER_ENVIRONMENT),
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "failure_code": failure_code,
        "protocol": "spiral-bfcl-v4-worker-failure/v1",
    }


def test_worker_rejects_arbitrary_git_executable(pinned_checkout: Path) -> None:
    arbitrary = Path("/usr/bin/true")
    if not arbitrary.is_file():
        pytest.skip("adversarial non-Git executable unavailable")
    worker = Path(subject.__file__).with_name("_bfcl_v4_fixture_worker.py").resolve()
    request = {
        "git_executable": {"path": str(arbitrary), "sha256": "0" * 64, "size": 1, "version": "x"},
        "grader_answer_binding_sha256": "0" * 64,
        "protocol": FIXTURE_PROTOCOL,
        "response_row_jsonl": "{}\n",
        "response_row_sha256": "0" * 64,
        "source_bundle_sha256": "0" * 64,
        "task_binding_sha256": "0" * 64,
    }
    completed = subprocess.run(
        (
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            str(worker),
            "--checkout",
            str(pinned_checkout.resolve()),
            "--git",
            str(arbitrary.resolve()),
        ),
        input=canonical_json_bytes(request),
        env=dict(subject._WORKER_ENVIRONMENT),
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["failure_code"] == "git-executable-not-system-resolved"


def test_full_cli_contract_is_separate_and_explicitly_unexecuted(
    pinned_checkout: Path,
) -> None:
    contract = subject.build_bfcl_v4_full_cli_invocation_contract(pinned_checkout)

    assert contract.argv_template[5] == "evaluate"
    assert contract.argv_template[-1] == "--partial-eval"
    assert contract.official_cli_executed is False
    assert contract.invocation_ready is False
    assert contract.dependency_environment_attested is False
    assert contract.provider_free_handler_construction_attested is False
    assert "instantiates-a-registered-handler" in contract.blocker
    full_paths = {item.git_path for item in contract.source_files}
    assert subject.EVAL_RUNNER_PATH in full_paths
    assert subject.OPENAI_DECODER_PATH in full_paths


def test_receipt_rejects_mutated_embedded_grader_binding(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    payload["grader_answer_binding"]["ground_truth_sha256"] = "0" * 64

    with pytest.raises(ValidationError):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "tampering",
    ("reorder", "duplicate", "credential-injection", "proxy-injection"),
)
def test_receipt_rejects_environment_tampering_after_rehash(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    tampering: str,
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    assert payload["environment_sha256"] == _environment_payload_sha256(payload)
    entries = payload["environment"]
    if tampering == "reorder":
        payload["environment"] = tuple(reversed(entries))
    elif tampering == "duplicate":
        payload["environment"] = (*entries, entries[0])
    elif tampering == "credential-injection":
        payload["environment"] = (
            *entries,
            {"name": "OPENAI_API_KEY", "value": "must-not-cross-boundary"},
        )
    else:
        payload["environment"] = (
            *entries,
            {"name": "HTTPS_PROXY", "value": "http://proxy.invalid"},
        )
    payload["environment_sha256"] = _environment_payload_sha256(payload)

    with pytest.raises(ValidationError, match="exact allowlist"):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize("tampering", ("roster", "path", "size", "sha256"))
def test_receipt_rejects_executed_source_tampering_after_rehash(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    tampering: str,
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    sources = list(payload["executed_sources"])
    if tampering == "roster":
        sources.reverse()
    else:
        changed = dict(sources[0])
        if tampering == "path":
            changed["git_path"] += ".tampered"
        elif tampering == "size":
            changed["size"] += 1
        else:
            changed["sha256"] = "0" * 64
        sources[0] = changed
    payload["executed_sources"] = tuple(sources)
    payload["source_bundle_sha256"] = _source_payload_bundle_sha256(payload["executed_sources"])

    with pytest.raises(ValidationError, match="source"):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "tampering",
    ("argv", "python-path", "git-path", "worker-path", "checkout-path"),
)
def test_receipt_rejects_execution_coordinate_tampering(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    tampering: str,
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    if tampering == "argv":
        argv = list(payload["argv"])
        argv[1] = "-E"
        payload["argv"] = tuple(argv)
        payload["argv_sha256"] = canonical_sha256(payload["argv"])
    elif tampering == "python-path":
        payload["python_executable"]["path"] += ".tampered"
        payload["environment_sha256"] = _environment_payload_sha256(payload)
    elif tampering == "git-path":
        payload["git_executable"]["path"] += ".tampered"
        payload["environment_sha256"] = _environment_payload_sha256(payload)
    elif tampering == "worker-path":
        payload["worker_source"]["path"] += ".tampered"
    else:
        payload["checkout_path"] += ".tampered"

    with pytest.raises(ValidationError):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


def test_receipt_rejects_worker_source_hash_tampering(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    payload["worker_source"]["sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="current pinned worker"):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize("tampering", ("size", "sha256"))
def test_receipt_rejects_worker_stdout_tampering(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    tampering: str,
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    if tampering == "size":
        payload["worker_stdout_size"] += 1
    else:
        payload["worker_stdout_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="worker stdout"):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    ("valid", "error_type"),
    ((True, "value_error:others"), (False, None)),
)
def test_receipt_rejects_inconsistent_validity_and_error_type(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    valid: bool,
    error_type: str | None,
) -> None:
    _, receipt = accepted_export_and_receipt
    payload = receipt.model_dump(mode="python")
    payload["upstream_ast_checker_valid"] = valid
    payload["checker_error_type"] = error_type

    with pytest.raises(ValidationError, match="validity and normalized error"):
        BfclV4SourceIsolatedReceipt.model_validate(payload, strict=True)


def test_saved_runtime_coordinates_revalidate_after_verifier_runtime_changes(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt = accepted_export_and_receipt
    changed_machine = "red-team-verifier-machine"
    monkeypatch.setattr(contracts_subject.platform, "machine", lambda: changed_machine)
    assert contracts_subject.runtime_coordinates().machine == changed_machine

    validated = BfclV4SourceIsolatedReceipt.model_validate(
        receipt.model_dump(mode="python"),
        strict=True,
    )

    assert validated.runtime == receipt.runtime
    assert validated.runtime.machine != contracts_subject.runtime_coordinates().machine


def test_full_cli_contract_rejects_fake_roster_after_bundle_rehash(
    pinned_checkout: Path,
) -> None:
    contract = subject.build_bfcl_v4_full_cli_invocation_contract(pinned_checkout)
    payload = contract.model_dump(mode="python")
    sources = list(payload["source_files"])
    changed = dict(sources[0])
    changed["git_path"] += ".fake"
    sources[0] = changed
    payload["source_files"] = tuple(sources)
    payload["source_bundle_sha256"] = _source_payload_bundle_sha256(payload["source_files"])

    with pytest.raises(ValidationError, match="pinned ordered roster"):
        BfclV4FullCliInvocationContract.model_validate(payload, strict=True)


def test_candidate_safe_projection_recursively_excludes_trusted_plane_terms(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
) -> None:
    exported, receipt = accepted_export_and_receipt
    feedback = subject.project_bfcl_v4_candidate_safe_feedback(exported, receipt)
    payload = feedback.model_dump(mode="json")
    forbidden = ("answer", "ground_truth", "grader", "binding", "receipt")

    assert set(payload) == {
        "accepted",
        "candidate_visible",
        "failure_class",
        "hidden_test_evidence",
        "information_scope",
        "partial_evaluation",
        "reportable_result",
        "response_row_reference_sha256",
        "schema_version",
        "task_id",
    }
    for token in _recursive_strings(payload):
        assert all(term not in token.casefold() for term in forbidden)
    assert feedback.accepted is True
    assert feedback.failure_class == "none"
    assert feedback.response_row_reference_sha256 == exported.response_row_sha256


def test_candidate_safe_projection_rejects_mismatched_export_and_receipt(
    accepted_export_and_receipt: tuple[
        BfclV4OfficialResultExport,
        BfclV4SourceIsolatedReceipt,
    ],
) -> None:
    _, receipt = accepted_export_and_receipt

    with pytest.raises(ValueError, match="does not belong"):
        subject.project_bfcl_v4_candidate_safe_feedback(_export(base=9), receipt)


@pytest.mark.parametrize(
    ("timeout_seconds", "exception_type"),
    (
        (True, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
    ),
)
def test_fixture_invocation_rejects_nonfinite_or_boolean_timeout(
    pinned_checkout: Path,
    timeout_seconds: float | bool,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        subject.invoke_bfcl_v4_source_isolated_fixture(
            _export(),
            pinned_checkout,
            timeout_seconds=timeout_seconds,
        )


def test_worker_answer_binding_digest_uses_actual_blob_row_and_ground_truth(
    pinned_checkout: Path,
) -> None:
    resolved, git = subject._checkout_and_git(pinned_checkout)
    answer_blob = subject._git_blob(git, resolved, ANSWER_PATH)
    answer, answer_row = subject._jsonl_entry(
        answer_blob,
        worker_subject.TASK_ID,
        "BFCL grader-only answer blob",
    )
    expected = worker_subject._grader_binding_sha256(answer, answer_blob, answer_row)
    assert expected == subject._load_grader_only_answer_binding(git, resolved).fingerprint
    changed_answer = dict(answer)
    changed_answer["ground_truth"] = {"tampered": True}

    changed_digests = {
        worker_subject._grader_binding_sha256(answer, answer_blob + b"x", answer_row),
        worker_subject._grader_binding_sha256(answer, answer_blob, answer_row + b"x"),
        worker_subject._grader_binding_sha256(changed_answer, answer_blob, answer_row),
    }

    assert expected not in changed_digests
    assert len(changed_digests) == 3
