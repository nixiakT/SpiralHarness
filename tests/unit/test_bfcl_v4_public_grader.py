from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_grader as subject
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    PUBLIC_GRADER_FAILURE_PROTOCOL,
    PUBLIC_GRADER_PROTOCOL,
    BfclV4GradingSlotBinding,
    BfclV4HoldoutUnlock,
    BfclV4PublicGraderReceipt,
    BfclV4PublicPrediction,
    prediction_content,
    source_bundle_sha256,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_projections import FIT_TASK_IDS
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PilotSplit,
)
from spiral_harness.core.canonical import canonical_json, canonical_json_bytes

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_PLAN = "1" * 64


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact-source integration")
    try:
        subject._checkout_and_git(checkout)
    except subject.BfclV4PublicGraderError as error:
        pytest.skip(f"pinned BFCL checkout is not usable: {error}")
    return checkout


def _prediction(task_id: str, *, accepted_simple_fixture: bool = False) -> BfclV4PublicPrediction:
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    if accepted_simple_fixture:
        calls = (("calculate_triangle_area", {"base": 10, "height": 5}),)
    return subject.make_bfcl_v4_public_prediction(task_id, calls)


def _slot(
    prediction: BfclV4PublicPrediction,
    *,
    arm: str = "full",
    role: str | None = None,
    index: int = 0,
    plan: str = _PLAN,
) -> BfclV4GradingSlotBinding:
    roster = next(
        item for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster if item.task_id == prediction.task_id
    )
    if role is None:
        role = {
            BfclV4PilotSplit.FIT: "parent-fit",
            BfclV4PilotSplit.GATE: "gate-parent",
            BfclV4PilotSplit.HOLDOUT: "holdout",
        }[roster.split]
    return BfclV4GradingSlotBinding(
        plan_fingerprint=plan,
        call_slot_reference_sha256=f"{index + 2:064x}",
        call_id=f"{arm}/{role}/{prediction.task_id}/{index}",
        arm=arm,
        grade_role=role,
        intended_harness_variant="parent",
        executed_harness_variant="parent",
        task_id=prediction.task_id,
        prediction_sha256=prediction.fingerprint,
    )


def _unlock(*, plan: str = _PLAN) -> BfclV4HoldoutUnlock:
    return BfclV4HoldoutUnlock(
        plan_fingerprint=plan,
        score_selection_artifact_sha256="a" * 64,
        full_selection_artifact_sha256="b" * 64,
    )


@pytest.fixture(scope="module")
def all_task_receipts(
    pinned_checkout: Path,
) -> dict[str, tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt]]:
    results: dict[str, tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt]] = {}
    for index, roster in enumerate(BFCL_V4_PUBLIC_PILOT_MANIFEST.roster):
        prediction = _prediction(
            roster.task_id,
            accepted_simple_fixture=roster.task_id == "simple_python_0",
        )
        slot = _slot(prediction, index=index)
        unlock = _unlock() if roster.split == BfclV4PilotSplit.HOLDOUT else None
        receipt = subject.grade_bfcl_v4_public_prediction(
            prediction,
            slot,
            pinned_checkout,
            holdout_unlock=unlock,
        )
        results[roster.task_id] = (prediction, receipt)
    return results


@pytest.fixture(scope="module")
def score_fit_receipts(
    pinned_checkout: Path,
) -> tuple[BfclV4PublicGraderReceipt, ...]:
    receipts: list[BfclV4PublicGraderReceipt] = []
    for index, task_id in enumerate(FIT_TASK_IDS):
        prediction = _prediction(task_id)
        receipts.append(
            subject.grade_bfcl_v4_public_prediction(
                prediction,
                _slot(prediction, arm="score", role="parent-fit", index=100 + index),
                pinned_checkout,
            )
        )
    return tuple(receipts)


def _recursive_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(
            token for key, item in value.items() for token in (str(key), *_recursive_strings(item))
        )
    if isinstance(value, (list, tuple)):
        return tuple(token for item in value for token in _recursive_strings(item))
    return (str(value),)


def test_generalized_worker_executes_all_frozen_15_public_tasks(
    all_task_receipts: dict[
        str,
        tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt],
    ],
) -> None:
    assert tuple(all_task_receipts) == tuple(
        item.task_id for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    )
    for task_id, (_, receipt) in all_task_receipts.items():
        assert receipt.prediction.task_id == task_id
        assert receipt.question_binding.task_id == task_id
        assert receipt.answer_binding.task_id == task_id
        assert receipt.exact_upstream_ast_checker_executed is True
        assert receipt.visibility == "grader-auditor-only"
        assert receipt.candidate_visible is False
        assert receipt.hidden_test_evidence is False
        assert receipt.reportable_result is False
        assert receipt.official_full_suite is False
        assert receipt.network_isolation_attested is False
    assert all_task_receipts["simple_python_0"][1].upstream_ast_checker_valid is True
    for task_id in tuple(all_task_receipts)[1:]:
        receipt = all_task_receipts[task_id][1]
        assert receipt.upstream_ast_checker_valid is False
        assert receipt.coarse_failure_class == "call-count"


def test_candidate_plane_contains_only_official_calls_not_answer_identity() -> None:
    prediction = _prediction("simple_python_0", accepted_simple_fixture=True)
    payload = prediction.model_dump(mode="json")
    serialized = canonical_json(payload)

    assert set(payload) == {
        "calls",
        "schema_version",
        "task_id",
    }
    assert payload["calls"][0] == {
        "arguments_json": '{"base":10,"height":5}',
        "function_name": "calculate_triangle_area",
    }
    assert "ground_truth" not in serialized
    assert "answer_blob" not in serialized
    assert "grader_answer" not in serialized


def test_bridge_public_loader_never_reads_possible_answer_blobs(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved, git = subject._checkout_and_git(pinned_checkout)
    observed: list[str] = []
    original = subject._git_blob

    def recording_blob(git_path: Path, checkout: Path, path: str) -> bytes:
        observed.append(path)
        return original(git_path, checkout, path)

    monkeypatch.setattr(subject, "_git_blob", recording_blob)
    bindings = tuple(
        subject._load_public_question_binding(git, resolved, roster.task_id)
        for roster in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    )

    assert len(bindings) == 15
    assert all("possible_answer" not in path for path in observed)
    assert set(observed) == {
        roster.question_git_path for roster in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    }


def test_holdout_cannot_be_graded_before_both_selection_artifacts(
    pinned_checkout: Path,
) -> None:
    prediction = _prediction("simple_python_87")
    slot = _slot(prediction)

    with pytest.raises(ValueError, match="both frozen selection artifacts"):
        subject.grade_bfcl_v4_public_prediction(prediction, slot, pinned_checkout)
    with pytest.raises(ValueError, match="another plan"):
        subject.grade_bfcl_v4_public_prediction(
            prediction,
            slot,
            pinned_checkout,
            holdout_unlock=_unlock(plan="9" * 64),
        )


def test_slot_prediction_substitution_fails_before_worker(pinned_checkout: Path) -> None:
    prediction = _prediction("simple_python_0")
    other = _prediction("simple_python_0", accepted_simple_fixture=True)

    with pytest.raises(ValueError, match="another prediction"):
        subject.grade_bfcl_v4_public_prediction(
            prediction,
            _slot(other),
            pinned_checkout,
        )


def test_full_projection_is_own_fit_binary_and_coarse_only(
    all_task_receipts: dict[
        str,
        tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt],
    ],
) -> None:
    prediction, receipt = all_task_receipts["simple_python_0"]
    feedback = subject.project_bfcl_v4_full_fit_feedback(prediction, receipt)
    payload = feedback.model_dump(mode="json")

    assert feedback.accepted is True
    assert feedback.failure_class == "none"
    assert set(payload) == {
        "accepted",
        "candidate_visible",
        "failure_class",
        "hidden_test_evidence",
        "information_scope",
        "own_prediction_reference_sha256",
        "partial_evaluation",
        "reportable_result",
        "schema_version",
        "task_id",
    }
    forbidden = ("answer", "ground_truth", "grader", "binding", "receipt", "error_type")
    assert all(
        all(term not in token.casefold() for term in forbidden)
        for token in _recursive_strings(payload)
    )


def test_score_projection_requires_complete_same_plan_five_fit_batch(
    score_fit_receipts: tuple[BfclV4PublicGraderReceipt, ...],
) -> None:
    expected_slots = tuple(
        receipt.slot.call_slot_reference_sha256 for receipt in score_fit_receipts
    )
    aggregate = subject.project_bfcl_v4_score_fit_aggregate(
        score_fit_receipts,
        expected_slots,
    )
    payload = aggregate.model_dump(mode="json")

    assert aggregate.aggregate_accuracy_basis_points == 0
    assert set(payload) == {
        "aggregate_accuracy_basis_points",
        "batch_reference_sha256",
        "candidate_visible",
        "fit_task_count",
        "hidden_test_evidence",
        "information_scope",
        "partial_evaluation",
        "plan_fingerprint",
        "reportable_result",
        "schema_version",
    }
    serialized = canonical_json(payload)
    assert all(task_id not in serialized for task_id in FIT_TASK_IDS)
    with pytest.raises(ValueError, match="complete five-FIT"):
        subject.project_bfcl_v4_score_fit_aggregate(
            score_fit_receipts[:-1],
            expected_slots[:-1],
        )
    with pytest.raises(ValueError, match="expected call slots"):
        subject.project_bfcl_v4_score_fit_aggregate(
            score_fit_receipts,
            (*expected_slots[:-1], "f" * 64),
        )


def test_worker_subprocess_inherits_no_credentials_or_proxies(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = {
        "ALL_PROXY",
        "HF_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
    }
    for name in sensitive:
        monkeypatch.setenv(name, "must-not-cross-process-boundary")
    observed: list[dict[str, str]] = []
    real_run = subprocess.run

    def recording_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed.append(environment.copy())
        return real_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subject.subprocess, "run", recording_run)
    prediction = _prediction("simple_python_0")
    receipt = subject.grade_bfcl_v4_public_prediction(
        prediction,
        _slot(prediction),
        pinned_checkout,
    )

    assert observed
    assert all(sensitive.isdisjoint(environment) for environment in observed)
    assert receipt.credential_environment_inherited is False
    assert receipt.proxy_environment_inherited is False


@pytest.mark.parametrize("tampering", ("answer", "source", "runtime", "stdout"))
def test_receipt_rejects_bound_plane_tampering(
    all_task_receipts: dict[
        str,
        tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt],
    ],
    tampering: str,
) -> None:
    _, receipt = all_task_receipts["simple_python_0"]
    payload = receipt.model_dump(mode="python")
    if tampering == "answer":
        payload["answer_binding"]["ground_truth_sha256"] = "0" * 64
    elif tampering == "source":
        sources = list(payload["executed_sources"])
        sources[0] = {**sources[0], "size": sources[0]["size"] + 1}
        payload["executed_sources"] = tuple(sources)
        typed_sources = tuple(
            subject.BfclV4SourceFileBinding.model_validate(item, strict=True)
            for item in payload["executed_sources"]
        )
        payload["source_bundle_sha256"] = source_bundle_sha256(typed_sources)
    elif tampering == "runtime":
        payload["runtime"]["machine"] = "tampered-machine"
    else:
        payload["worker_stdout_sha256"] = "0" * 64

    with pytest.raises(ValidationError):
        BfclV4PublicGraderReceipt.model_validate(payload, strict=True)


def _direct_worker_request(
    prediction: BfclV4PublicPrediction,
    slot: BfclV4GradingSlotBinding,
    checkout: Path,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    resolved, git = subject._checkout_and_git(checkout)
    question = subject._load_public_question_binding(git, resolved, prediction.task_id)
    sources = subject._source_bindings(git, resolved)
    git_version = subject._run_git(git, resolved, "--version").decode().strip()
    git_binding = subject._executable_binding(git, git_version)
    worker = subject.current_public_grader_worker_binding().path
    command = (
        str(Path(subject.sys.executable).resolve()),
        "-I",
        "-B",
        worker,
        "--checkout",
        str(resolved),
        "--git",
        str(git),
    )
    request = {
        "git_executable": {
            "path": git_binding.path,
            "sha256": git_binding.sha256_observation,
            "size": git_binding.size_observation,
            "version": git_binding.version_observation,
        },
        "holdout_unlock_sha256": None,
        "prediction_json": canonical_json(prediction_content(prediction)),
        "prediction_sha256": prediction.fingerprint,
        "protocol": PUBLIC_GRADER_PROTOCOL,
        "question_binding_sha256": question.fingerprint,
        "slot_binding_sha256": slot.fingerprint,
        "source_bundle_sha256": source_bundle_sha256(sources),
        "task_id": prediction.task_id,
    }
    return command, request


@pytest.mark.parametrize(
    ("field", "failure_code"),
    (
        ("prediction_sha256", "prediction-hash-mismatch"),
        ("question_binding_sha256", "question-binding-request-mismatch"),
        ("source_bundle_sha256", "source-bundle-hash-mismatch"),
    ),
)
def test_worker_recomputes_candidate_question_and_source_bindings(
    pinned_checkout: Path,
    field: str,
    failure_code: str,
) -> None:
    prediction = _prediction("simple_python_0")
    command, request = _direct_worker_request(prediction, _slot(prediction), pinned_checkout)
    request[field] = "0" * 64

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
        "protocol": PUBLIC_GRADER_FAILURE_PROTOCOL,
    }


@pytest.mark.parametrize(
    "arguments_json",
    ('{"x":1e400}', '{"x":"\\ud800"}', '{"x":1,"x":2}', '{"b":2,"a":1}'),
)
def test_prediction_rejects_nonfinite_surrogate_duplicate_or_noncanonical_json(
    arguments_json: str,
) -> None:
    call = {
        "function_name": "calculate_triangle_area",
        "arguments_json": arguments_json,
    }
    with pytest.raises(ValidationError):
        subject.BfclV4OfficialPredictionCall.model_validate(call, strict=True)


def test_unchecked_construction_cannot_upgrade_public_development_claims(
    all_task_receipts: dict[
        str,
        tuple[BfclV4PublicPrediction, BfclV4PublicGraderReceipt],
    ],
) -> None:
    _, receipt = all_task_receipts["simple_python_0"]
    payload = receipt.model_dump(mode="python")
    payload["reportable_result"] = True
    unchecked = BfclV4PublicGraderReceipt.model_construct(**payload)

    with pytest.raises(ValidationError):
        unchecked.model_dump(mode="json")
    with pytest.raises(ValidationError):
        canonical_json(unchecked)
