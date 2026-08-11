from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark import (
    MECHANISM_CHECK_NAMES,
    CandidateOutputTrace,
    CaptureAttestation,
    CapturedOutputTrace,
    CaptureVerificationCapability,
    ControlledAdversarialVerifier,
    FixedGraderConfig,
    FixedOutputGrader,
    GradingTask,
    SealedTaskRoster,
    TrustedCaptureService,
    TrustedGradingError,
)
from spiral_harness.verification import Decision, GateConfig, TrialStatus


def grading_tasks() -> tuple[GradingTask, ...]:
    return tuple(
        GradingTask(
            task_id=f"task-{index:02d}",
            input_text=f"classify public sample {index}",
            expected_output="MATCH",
            slice_tags=("protected",) if index >= 6 else ("main",),
        )
        for index in range(8)
    )


def fixed_grader(capture_service: TrustedCaptureService) -> FixedOutputGrader:
    return FixedOutputGrader(
        FixedGraderConfig(
            required_target_component="system_prompt",
            max_tokens=64,
            max_latency_ms=10.0,
            max_tool_calls=0,
        ),
        capture_verifier=capture_service.verification_capability,
    )


def adversarial_gate() -> GateConfig:
    return GateConfig(
        version="controlled-adversarial-gate-v1",
        min_tasks=8,
        min_effect=0.25,
        bootstrap_samples=1_000,
        bootstrap_seed=2026,
        required_mechanism_checks=MECHANISM_CHECK_NAMES,
        protected_slice_floors={"protected": -0.01},
        min_slice_tasks=2,
        max_single_task_regression=0.0,
        max_regression_rate=0.0,
        regression_tolerance=0.0,
    )


def verifier(
    tasks: Sequence[GradingTask],
    capture_service: TrustedCaptureService,
) -> ControlledAdversarialVerifier:
    return ControlledAdversarialVerifier(
        fixed_grader(capture_service),
        adversarial_gate(),
        tasks=tasks,
    )


def captured_trace(
    capture_service: TrustedCaptureService,
    task: GradingTask,
    *,
    harness_id: str,
    output: str,
) -> CapturedOutputTrace:
    trace = CandidateOutputTrace(
        task_id=task.task_id,
        task_sha256=task.fingerprint,
        seed=17,
        harness_id=harness_id,
        execution_fingerprint=f"fixed-context:{task.task_id}:17",
        status=TrialStatus.COMPLETED,
        output=output,
        tokens=16,
        latency_ms=1.0,
        tool_calls=0,
        mutation_applied=True,
        target_component="system_prompt",
        activated=True,
        adhered=True,
        intervention="treatment",
    )
    return capture_service.capture(trace)


def honest_captures(
    capture_service: TrustedCaptureService,
    tasks: tuple[GradingTask, ...],
) -> tuple[tuple[CapturedOutputTrace, ...], tuple[CapturedOutputTrace, ...]]:
    parent = tuple(
        captured_trace(
            capture_service,
            task,
            harness_id="parent-harness",
            output="MATCH" if "protected" in task.slice_tags else "DIFFERENT",
        )
        for task in tasks
    )
    candidate = tuple(
        captured_trace(
            capture_service,
            task,
            harness_id="candidate-harness",
            output="MATCH",
        )
        for task in tasks
    )
    return parent, candidate


def replace_trace(
    capture_service: TrustedCaptureService,
    captures: Sequence[CapturedOutputTrace | object],
    index: int,
    **updates: object,
) -> tuple[CapturedOutputTrace | object, ...]:
    replaced = list(captures)
    captured = replaced[index]
    assert isinstance(captured, CapturedOutputTrace)
    replaced[index] = capture_service.capture(captured.trace.model_copy(update=updates))
    return tuple(replaced)


def adversarial_captures(
    case: str,
    capture_service: TrustedCaptureService,
    tasks: tuple[GradingTask, ...],
    parent: tuple[CapturedOutputTrace, ...],
    candidate: tuple[CapturedOutputTrace, ...],
) -> tuple[CapturedOutputTrace | object, ...]:
    if case == "no-op":
        return tuple(
            capture_service.capture(
                candidate_capture.trace.model_copy(
                    update={
                        "mutation_applied": False,
                        "output": parent_capture.trace.output,
                    }
                )
            )
            for parent_capture, candidate_capture in zip(parent, candidate, strict=True)
        )
    if case == "wrong-target":
        return replace_trace(capture_service, candidate, 0, target_component="memory")
    if case == "inactive":
        return replace_trace(capture_service, candidate, 0, activated=False)
    if case == "non-adherent":
        return replace_trace(capture_service, candidate, 0, adhered=False)
    if case == "placebo":
        return replace_trace(capture_service, candidate, 0, intervention="placebo")
    if case == "regression":
        captures: tuple[CapturedOutputTrace | object, ...] = candidate
        for index, task in enumerate(tasks):
            if "protected" in task.slice_tags:
                captures = replace_trace(capture_service, captures, index, output="DIFFERENT")
        return captures
    if case == "timeout":
        return replace_trace(
            capture_service,
            candidate,
            0,
            status=TrialStatus.TIMEOUT,
            output=None,
        )
    if case == "malformed":
        malformed = candidate[0].model_dump(mode="python")
        malformed["trace"]["score"] = 1.0
        return (malformed, *candidate[1:])
    if case == "tamper":
        tampered_trace = candidate[0].trace.model_copy(update={"output": "DIFFERENT"})
        tampered_capture = candidate[0].model_copy(update={"trace": tampered_trace})
        return (tampered_capture, *candidate[1:])
    if case == "budget":
        return replace_trace(capture_service, candidate, 0, tokens=65)
    if case == "leakage":
        return replace_trace(
            capture_service,
            candidate,
            0,
            accessed_resources=("sealed_answers",),
        )
    raise AssertionError(f"unknown adversarial case: {case}")


def test_candidate_trace_cannot_supply_a_score_and_trusted_grader_owns_scoring() -> None:
    task = grading_tasks()[0]
    capture_service = TrustedCaptureService()
    grader = fixed_grader(capture_service)
    captured = captured_trace(
        capture_service,
        task,
        harness_id="candidate-harness",
        output="MATCH",
    )
    forged = captured.trace.model_dump(mode="python")
    forged["score"] = 1.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CandidateOutputTrace.model_validate(forged)

    graded = grader.grade(task, captured, candidate=True)
    assert "score" not in CandidateOutputTrace.model_fields
    assert "expected_output" not in CandidateOutputTrace.model_fields
    alternate_answer = task.model_copy(update={"expected_output": "DIFFERENT"})
    assert task.fingerprint == alternate_answer.fingerprint
    assert task.sealed_fingerprint != alternate_answer.sealed_fingerprint
    assert graded.observation.score == 1.0
    assert graded.grader_fingerprint == grader.fingerprint
    assert graded.capture_attestation.attestation_sha256 == captured.attestation_sha256


def test_capture_signing_is_not_exposed_on_the_captured_value_or_grader() -> None:
    capture_service = TrustedCaptureService()

    assert not hasattr(CapturedOutputTrace, "capture")
    assert isinstance(
        capture_service.verification_capability,
        CaptureVerificationCapability,
    )
    with pytest.raises(TypeError, match="capture_verifier"):
        FixedOutputGrader()


def test_capture_from_another_service_is_rejected_even_for_the_same_trace() -> None:
    task = grading_tasks()[0]
    trusted_service = TrustedCaptureService()
    other_service = TrustedCaptureService()
    trusted_capture = captured_trace(
        trusted_service,
        task,
        harness_id="candidate-harness",
        output="MATCH",
    )
    other_capture = other_service.capture(trusted_capture.trace)
    grader = fixed_grader(trusted_service)

    assert trusted_capture.trace == other_capture.trace
    assert trusted_capture.attestor_id != other_capture.attestor_id
    assert grader.fingerprint != fixed_grader(other_service).fingerprint
    with pytest.raises(TrustedGradingError, match="untrusted attestor"):
        grader.grade(task, other_capture, candidate=True)


def test_signed_trace_tampering_and_hmac_copying_fail_closed() -> None:
    task = grading_tasks()[0]
    capture_service = TrustedCaptureService()
    grader = fixed_grader(capture_service)
    captured = captured_trace(
        capture_service,
        task,
        harness_id="candidate-harness",
        output="MATCH",
    )
    tampered_trace = captured.trace.model_copy(update={"output": "DIFFERENT"})

    digest_tamper = captured.model_copy(update={"trace": tampered_trace})
    with pytest.raises(TrustedGradingError, match="digest does not match"):
        grader.grade(task, digest_tamper, candidate=True)

    newly_signed = capture_service.capture(tampered_trace)
    copied_hmac = newly_signed.model_copy(
        update={"attestation_sha256": captured.attestation_sha256}
    )
    with pytest.raises(TrustedGradingError, match="HMAC attestation is invalid"):
        grader.grade(task, copied_hmac, candidate=True)


def test_sealed_roster_is_canonical_and_rejects_duplicate_task_ids() -> None:
    tasks = grading_tasks()

    canonical = SealedTaskRoster(tasks=tasks)
    reversed_roster = SealedTaskRoster(tasks=tuple(reversed(tasks)))

    assert canonical.tasks == tasks
    assert canonical.fingerprint == reversed_roster.fingerprint
    duplicate = tasks[0].model_copy(update={"input_text": "different public input"})
    with pytest.raises(ValidationError, match="unique task IDs"):
        SealedTaskRoster(tasks=(*tasks, duplicate))


def test_constructor_frozen_answers_and_slices_cannot_be_substituted() -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    parent, candidate = honest_captures(capture_service, tasks)
    caller_owned_tasks = list(tasks)
    controlled = verifier(caller_owned_tasks, capture_service)
    alternate_tasks = tuple(
        task.model_copy(
            update={
                "expected_output": "DIFFERENT",
                "slice_tags": ("substituted",),
            }
        )
        for task in tasks
    )

    assert tuple(task.fingerprint for task in tasks) == tuple(
        task.fingerprint for task in alternate_tasks
    )
    assert (
        SealedTaskRoster(tasks=tasks).fingerprint
        != SealedTaskRoster(tasks=alternate_tasks).fingerprint
    )

    caller_owned_tasks[:] = alternate_tasks
    result = controlled.verify(parent, candidate)
    expected_slices = {task.task_id: task.slice_tags for task in tasks}

    assert result.decision is Decision.PROMOTE
    assert result.task_set_fingerprint == SealedTaskRoster(tasks=tasks).fingerprint
    assert all(observation.score == 1.0 for observation in result.candidate_observations)
    assert all(
        observation.slice_tags == expected_slices[observation.task_id]
        for observation in result.candidate_observations
    )


def test_verifier_api_does_not_accept_caller_supplied_tasks() -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    parent, candidate = honest_captures(capture_service, tasks)
    controlled = verifier(tasks, capture_service)

    with pytest.raises(TypeError, match="positional arguments"):
        controlled.verify(tasks, parent, candidate)


def test_graded_batch_retains_task_grader_and_capture_provenance() -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    _, candidate = honest_captures(capture_service, tasks)
    grader = fixed_grader(capture_service)

    batch = grader.grade_batch(tasks, candidate, candidate=True)

    assert batch.task_set_fingerprint == SealedTaskRoster(tasks=tasks).fingerprint
    assert batch.grader_fingerprint == grader.fingerprint
    assert len(batch.capture_attestations) == len(batch.observations) == len(candidate)
    for observation, captured, provenance in zip(
        batch.observations,
        candidate,
        batch.capture_attestations,
        strict=True,
    ):
        assert isinstance(provenance, CaptureAttestation)
        assert (provenance.task_id, provenance.seed, provenance.harness_id) == (
            observation.task_id,
            observation.seed,
            observation.harness_id,
        )
        assert provenance.trace_sha256 == captured.captured_sha256
        assert provenance.attestor_id == captured.attestor_id
        assert provenance.attestation_sha256 == captured.attestation_sha256

    payload = batch.model_dump(mode="python")
    payload["capture_attestations"][0]["harness_id"] = "substituted-harness"
    with pytest.raises(ValidationError, match="provenance does not match"):
        type(batch).model_validate(payload)


def test_honest_held_out_improvement_is_a_positive_control() -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    parent, candidate = honest_captures(capture_service, tasks)
    controlled = verifier(tasks, capture_service)

    result = controlled.verify(parent, candidate)

    assert result.decision is Decision.PROMOTE
    assert result.gate_decision is not None
    assert result.grader_fingerprint == controlled.grader_fingerprint
    assert result.task_set_fingerprint == controlled.task_set_fingerprint
    assert len(result.parent_capture_attestations) == len(tasks)
    assert len(result.candidate_capture_attestations) == len(tasks)
    assert all(observation.score is not None for observation in result.candidate_observations)


@pytest.mark.parametrize("roster_case", ["missing", "extra", "duplicate"])
def test_capture_roster_mismatch_is_rejected_before_the_gate(roster_case: str) -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    parent, candidate = honest_captures(capture_service, tasks)
    controlled = verifier(tasks, capture_service)

    if roster_case == "missing":
        malformed_roster = candidate[:-1]
    elif roster_case == "extra":
        extra_task = GradingTask(
            task_id="task-extra",
            input_text="extra public sample",
            expected_output="MATCH",
            slice_tags=("main",),
        )
        extra_capture = captured_trace(
            capture_service,
            extra_task,
            harness_id="candidate-harness",
            output="MATCH",
        )
        malformed_roster = (*candidate, extra_capture)
    else:
        malformed_roster = (*candidate, candidate[0])

    result = controlled.verify(parent, malformed_roster)

    assert result.decision is Decision.REJECT
    assert result.gate_decision is None
    assert result.failure_reasons
    assert not result.parent_observations
    assert not result.candidate_observations


@pytest.mark.parametrize(
    "case",
    [
        "no-op",
        "wrong-target",
        "inactive",
        "non-adherent",
        "placebo",
        "regression",
        "timeout",
        "malformed",
        "tamper",
        "budget",
        "leakage",
    ],
)
def test_adversarial_failures_cannot_promote(case: str) -> None:
    tasks = grading_tasks()
    capture_service = TrustedCaptureService()
    parent, candidate = honest_captures(capture_service, tasks)
    adversarial = adversarial_captures(
        case,
        capture_service,
        tasks,
        parent,
        candidate,
    )

    result = verifier(tasks, capture_service).verify(parent, adversarial)

    expected = Decision.INCONCLUSIVE if case == "timeout" else Decision.REJECT
    assert result.decision is expected
    assert result.decision is not Decision.PROMOTE
    if case in {"malformed", "tamper"}:
        assert result.gate_decision is None
        assert result.failure_reasons
    else:
        assert result.gate_decision is not None


def test_adversarial_verifier_refuses_a_gate_that_omits_trusted_mechanism_checks() -> None:
    capture_service = TrustedCaptureService()

    with pytest.raises(ValueError, match="must require trusted mechanism checks"):
        ControlledAdversarialVerifier(
            fixed_grader(capture_service),
            GateConfig(),
            tasks=grading_tasks(),
        )
