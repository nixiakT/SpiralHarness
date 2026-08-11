from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from spiral_harness.execution.attempts import (
    AttemptBudget,
    AttemptBudgetExceeded,
    AttemptDisposition,
    AttemptLedger,
    AttemptOutcome,
    AttemptReservation,
)
from spiral_harness.execution.model import (
    BackendFingerprintMismatchError,
    BackendResponse,
    BackendTokenUsage,
    CandidateTask,
    ExecutionErrorClass,
    ExecutionStatus,
    FixedModelRunner,
    FrozenModelSpec,
    InferenceConfig,
    ModelExecution,
    ModelExecutionRecord,
    PromptHarness,
    ReplayBackend,
    materialize_request,
    paired_execution_fingerprint,
    replay_key,
)
from spiral_harness.storage import ArtifactStore

BACKEND_FINGERPRINT = "replay-backend@sha256:fixed-v1"


def fixed_spec(**updates: object) -> FrozenModelSpec:
    values: dict[str, object] = {
        "backend": "deterministic-replay",
        "backend_fingerprint": BACKEND_FINGERPRINT,
        "model": "hosted/example-model",
        "revision": "snapshot-2026-08-01",
        "tokenizer": "provider-managed/example-tokenizer",
        "tokenizer_revision": "snapshot-2026-08-01",
        "runtime": "spiral-worker-py3.12@sha256:runtime-v1",
        "inference": InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
            stop_sequences=("</answer>",),
        ),
    }
    values.update(updates)
    return FrozenModelSpec(**values)


def task(task_id: str = "gsm8k:test:0001", question: str = "What is 2 + 3?") -> CandidateTask:
    return CandidateTask(task_id=task_id, question=question)


def harness(
    harness_id: str = "static",
    prompt: str = "Solve carefully and end with a number.",
) -> PromptHarness:
    return PromptHarness.from_prompt(harness_id=harness_id, system_prompt=prompt)


def response(
    output: str = "5",
    *,
    input_tokens: int = 4,
    output_tokens: int = 2,
    cost_usd: float | None = None,
) -> BackendResponse:
    return BackendResponse(
        output=output,
        usage=BackendTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        cost_usd=cost_usd,
    )


def attempt_ledger(
    store: ArtifactStore,
    *,
    ledger_id: str = "experiment-1",
    attempts: int = 3,
    total_tokens: int = 36,
    per_attempt: int = 12,
) -> AttemptLedger:
    return AttemptLedger(
        store,
        ledger_id=ledger_id,
        budget=AttemptBudget(
            max_attempts=attempts,
            max_total_tokens=total_tokens,
            max_tokens_per_attempt=per_attempt,
        ),
    )


class InspectingBackend:
    def __init__(self, ledger: AttemptLedger, result: BackendResponse) -> None:
        self._ledger = ledger
        self._result = result
        self.calls = 0
        self.saw_pending_reservation = False

    @property
    def fingerprint(self) -> str:
        return BACKEND_FINGERPRINT

    def invoke(self, *, spec: FrozenModelSpec, request: object) -> BackendResponse:
        del spec, request
        self.calls += 1
        self.saw_pending_reservation = self._ledger.state().pending_reservation_ref is not None
        return self._result


class FailingBackend:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return BACKEND_FINGERPRINT

    def invoke(self, *, spec: FrozenModelSpec, request: object) -> BackendResponse:
        del spec, request
        self.calls += 1
        raise self.error


class InvalidResponseBackend:
    fingerprint = BACKEND_FINGERPRINT

    def invoke(self, *, spec: FrozenModelSpec, request: object) -> object:
        del spec, request
        return {
            "output": "attacker output",
            "usage": {"input_tokens": -1, "output_tokens": 1},
            "score": 1.0,
        }


class MutableFingerprintBackend:
    def __init__(self) -> None:
        self.current = BACKEND_FINGERPRINT

    @property
    def fingerprint(self) -> str:
        return self.current

    def invoke(self, *, spec: FrozenModelSpec, request: object) -> BackendResponse:
        del spec, request
        self.current = "changed-after-reservation"
        return response()


def test_frozen_spec_is_strict_pinned_immutable_and_canonically_fingerprinted() -> None:
    spec = fixed_spec()
    same = fixed_spec()

    assert spec.fingerprint == same.fingerprint
    assert spec.model_fingerprint == same.model_fingerprint
    assert spec.inference_fingerprint == spec.inference.fingerprint
    assert len(spec.runtime_fingerprint) == 64
    assert spec.revision == "snapshot-2026-08-01"  # hosted snapshot, not a fake weight hash

    changed = fixed_spec(runtime="spiral-worker-py3.13@sha256:runtime-v2")
    assert changed.fingerprint != spec.fingerprint
    assert changed.runtime_fingerprint != spec.runtime_fingerprint
    assert changed.model_fingerprint == spec.model_fingerprint

    with pytest.raises(ValidationError, match="immutable snapshot"):
        fixed_spec(revision="latest")
    with pytest.raises(ValidationError, match="immutable snapshot"):
        fixed_spec(tokenizer_revision="refs/heads/main")
    with pytest.raises(ValidationError):
        spec.runtime = "mutable"
    with pytest.raises(ValidationError):
        InferenceConfig(
            temperature="0",
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
        )


def test_prompt_hash_binds_exact_whitespace_and_utf8() -> None:
    exact = harness(prompt="  先思考。\nReturn π.  ")
    assert exact.system_prompt.startswith("  ")

    with pytest.raises(ValidationError, match="exact system_prompt bytes"):
        PromptHarness(
            harness_id="forged",
            system_prompt=exact.system_prompt + "!",
            prompt_sha256=exact.prompt_sha256,
        )


def test_paired_fingerprint_excludes_arm_and_prompt_but_request_hash_binds_them() -> None:
    spec = fixed_spec()
    item = task()
    left_harness = harness("parent", "Use arithmetic.")
    right_harness = harness("candidate", "Use algebra and verify.")
    left_request = materialize_request(item, left_harness, seed=17)
    right_request = materialize_request(item, right_harness, seed=17)

    left = paired_execution_fingerprint(
        spec,
        item,
        seed=17,
        backend_fingerprint=BACKEND_FINGERPRINT,
    )
    right = paired_execution_fingerprint(
        spec,
        item,
        seed=17,
        backend_fingerprint=BACKEND_FINGERPRINT,
    )

    assert left == right
    assert left_request.fingerprint != right_request.fingerprint
    assert (
        paired_execution_fingerprint(
            spec,
            item,
            seed=18,
            backend_fingerprint=BACKEND_FINGERPRINT,
        )
        != left
    )
    assert (
        paired_execution_fingerprint(
            spec,
            task(question="What is 3 + 3?"),
            seed=17,
            backend_fingerprint=BACKEND_FINGERPRINT,
        )
        != left
    )


def test_runner_reserves_before_call_then_persists_and_settles(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    backend = InspectingBackend(ledger, response(cost_usd=0.002))
    clock_values = iter((10.0, 10.25))
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend,
        attempt_ledger=ledger,
        clock=lambda: next(clock_values),
    )

    execution = runner.execute(task(), harness=harness(), seed=7)

    assert backend.calls == 1
    assert backend.saw_pending_reservation
    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.output_text == "5"
    assert execution.task_id == "gsm8k:test:0001"
    assert execution.harness_id == "static"
    assert execution.seed == 7
    assert execution.tokens == 6
    assert execution.latency_ms == 250.0
    assert execution.tool_calls == 0
    assert execution.cost_usd == 0.002
    assert runner.last_execution_ref is not None
    assert runner.last_outcome_ref == ledger.tail_ref
    assert store.get_json(runner.last_execution_ref, ModelExecution) == execution

    outcome = store.get_json(runner.last_outcome_ref, AttemptOutcome)
    assert outcome.disposition is AttemptDisposition.SETTLED
    assert outcome.reported_tokens == outcome.charged_tokens == 6
    assert outcome.execution_ref == runner.last_execution_ref
    assert ledger.state().pending_reservation_ref is None


def test_runner_accepts_independent_structural_task_and_copies_safe_view(tmp_path) -> None:
    class BenchmarkTask(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

        task_id: str
        question: str

    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=ReplayBackend(
            fingerprint=BACKEND_FINGERPRINT,
            default_response=response(),
        ),
        attempt_ledger=ledger,
    )
    external = BenchmarkTask(task_id="external-1", question="1 + 1?")

    result = runner.execute(external, harness=harness(), seed=1)

    assert result.task == CandidateTask(task_id="external-1", question="1 + 1?")
    assert type(result.task) is CandidateTask


def test_structural_task_with_trusted_answer_capability_is_rejected_before_reserve(
    tmp_path,
) -> None:
    @dataclass(frozen=True)
    class UnsafeTask:
        task_id: str
        question: str
        answer: str

    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=response(),
    )
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )

    with pytest.raises(ValueError, match="trusted fields: answer"):
        runner.execute(
            UnsafeTask(task_id="unsafe", question="leaked?", answer="42"),
            harness=harness(),
            seed=1,
        )

    assert backend.calls == ()
    assert ledger.state().attempts_used == 0


@pytest.mark.parametrize(
    ("error", "expected_class"),
    [
        (TimeoutError("provider deadline"), ExecutionErrorClass.BACKEND_TIMEOUT),
        (RuntimeError("provider unavailable"), ExecutionErrorClass.BACKEND_EXCEPTION),
    ],
)
def test_backend_exception_or_timeout_returns_failure_and_burns(
    tmp_path,
    error: Exception,
    expected_class: ExecutionErrorClass,
) -> None:
    store = ArtifactStore(tmp_path / expected_class.value)
    ledger = attempt_ledger(store)
    backend = FailingBackend(error)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )

    execution = runner.execute(task(), harness=harness(), seed=3)
    outcome = store.get_json(ledger.tail_ref, AttemptOutcome)

    assert execution.status is ExecutionStatus.FAILED
    assert execution.output is None
    assert execution.error is not None
    assert execution.error.error_class is expected_class
    assert "provider" not in execution.error.detail
    assert backend.calls == 1
    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.charged_tokens == 12
    assert ledger.state().attempts_used == 1


def test_usage_overrun_is_not_gradable_and_burns_full_reservation(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=response("would-be-output", input_tokens=10, output_tokens=9),
    )
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )

    execution = runner.execute(task(), harness=harness(), seed=3)
    outcome = store.get_json(ledger.tail_ref, AttemptOutcome)

    assert execution.status is ExecutionStatus.FAILED
    assert execution.output is None
    assert execution.usage.total_tokens == 19
    assert execution.error is not None
    assert execution.error.error_class is ExecutionErrorClass.USAGE_EXCEEDED
    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == 19
    assert outcome.charged_tokens == 19
    assert ledger.state().poisoned is True
    assert ledger.state().remaining_tokens == 0

    calls_after_overrun = backend.calls
    with pytest.raises(AttemptBudgetExceeded, match="poisoned"):
        runner.execute(task("after-overrun"), harness=harness(), seed=4)
    assert backend.calls == calls_after_overrun


def test_budget_exhaustion_never_calls_backend_again(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store, attempts=1, total_tokens=12, per_attempt=12)
    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=response(),
    )
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )
    runner.execute(task(), harness=harness(), seed=1)
    calls_after_first = backend.calls

    with pytest.raises(AttemptBudgetExceeded, match="attempt budget"):
        runner.execute(task("second"), harness=harness(), seed=2)

    assert backend.calls == calls_after_first
    assert len(backend.calls) == 1


def test_backend_fingerprint_must_match_exactly_before_reservation(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    backend = ReplayBackend(
        fingerprint="other-backend",
        default_response=response(),
    )

    with pytest.raises(BackendFingerprintMismatchError, match="exactly match"):
        FixedModelRunner(
            spec=fixed_spec(),
            backend=backend,
            attempt_ledger=ledger,
        )

    assert ledger.state().attempts_used == 0
    assert backend.calls == ()


def test_backend_fingerprint_change_during_call_burns_attempt(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=MutableFingerprintBackend(),
        attempt_ledger=ledger,
    )

    execution = runner.execute(task(), harness=harness(), seed=4)
    outcome = store.get_json(ledger.tail_ref, AttemptOutcome)

    assert execution.status is ExecutionStatus.FAILED
    assert execution.error is not None
    assert execution.error.error_class is ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH
    assert outcome.disposition is AttemptDisposition.BURNED


def test_invalid_backend_result_is_failed_and_burned(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=InvalidResponseBackend(),
        attempt_ledger=ledger,
    )

    execution = runner.execute(task(), harness=harness(), seed=4)
    outcome = store.get_json(ledger.tail_ref, AttemptOutcome)

    assert execution.status is ExecutionStatus.FAILED
    assert execution.error is not None
    assert execution.error.error_class is ExecutionErrorClass.INVALID_BACKEND_RESPONSE
    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.charged_tokens == 12


def test_execution_schema_is_score_free_and_rejects_injected_grading_fields(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=ReplayBackend(
            fingerprint=BACKEND_FINGERPRINT,
            default_response=response(),
        ),
        attempt_ledger=ledger,
    )
    execution = runner.execute(task(), harness=harness(), seed=5)
    payload = execution.model_dump(mode="json")

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(all_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(all_keys(item) for item in value))
        return set()

    assert all_keys(payload).isdisjoint(
        {"answer", "gold", "grade", "grader", "reference", "reference_answer", "score"}
    )
    with pytest.raises(ValidationError):
        ModelExecution.model_validate({**payload, "score": 1.0}, strict=True)
    with pytest.raises(ValidationError):
        CandidateTask.model_validate(
            {"task_id": "leak", "question": "?", "answer": "secret"},
            strict=True,
        )


def test_clock_changes_latency_only_not_any_fingerprint(tmp_path) -> None:
    store_a = ArtifactStore(tmp_path / "a")
    store_b = ArtifactStore(tmp_path / "b")
    clock_a = iter((1.0, 1.1))
    clock_b = iter((100.0, 102.0))
    backend_a = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=response(),
    )
    backend_b = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        default_response=response(),
    )
    runner_a = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend_a,
        attempt_ledger=attempt_ledger(store_a, ledger_id="a"),
        clock=lambda: next(clock_a),
    )
    runner_b = FixedModelRunner(
        spec=fixed_spec(),
        backend=backend_b,
        attempt_ledger=attempt_ledger(store_b, ledger_id="b"),
        clock=lambda: next(clock_b),
    )

    left = runner_a.execute(task(), harness=harness(), seed=9)
    right = runner_b.execute(task(), harness=harness(), seed=9)

    assert left.latency_ms != right.latency_ms
    assert runner_a.fingerprint == runner_b.fingerprint
    assert left.execution_fingerprint == right.execution_fingerprint
    assert left.request_sha256 == right.request_sha256
    assert left.model_fingerprint == right.model_fingerprint
    assert left.inference_fingerprint == right.inference_fingerprint
    assert left.runtime_fingerprint == right.runtime_fingerprint


def test_replay_backend_selects_by_exact_spec_and_request_not_call_order() -> None:
    spec = fixed_spec()
    item = task()
    first_request = materialize_request(item, harness("first", "Prompt A"), seed=1)
    second_request = materialize_request(item, harness("second", "Prompt B"), seed=1)
    backend = ReplayBackend(
        fingerprint=BACKEND_FINGERPRINT,
        responses={
            replay_key(spec, first_request): response("A"),
            replay_key(spec, second_request): response("B"),
        },
    )

    assert backend.invoke(spec=spec, request=second_request).output == "B"
    assert backend.invoke(spec=spec, request=first_request).output == "A"
    assert backend.invoke(spec=spec, request=second_request).output == "B"

    changed_spec = fixed_spec(revision="snapshot-2026-08-02")
    with pytest.raises(LookupError, match="frozen spec/request"):
        backend.invoke(spec=changed_spec, request=first_request)


def test_execute_record_returns_atomic_refs_under_concurrent_calls(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    ledger = attempt_ledger(store, attempts=2, total_tokens=24, per_attempt=12)
    runner = FixedModelRunner(
        spec=fixed_spec(),
        backend=ReplayBackend(
            fingerprint=BACKEND_FINGERPRINT,
            default_response=response(),
        ),
        attempt_ledger=ledger,
    )
    calls = (
        (task("task-a", "Question A?"), harness("arm-a", "Prompt A"), 31),
        (task("task-b", "Question B?"), harness("arm-b", "Prompt B"), 37),
    )

    def run(call: tuple[CandidateTask, PromptHarness, int]) -> ModelExecutionRecord:
        item, prompt_harness, seed = call
        return runner.execute_record(item, harness=prompt_harness, seed=seed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = tuple(pool.map(run, calls))

    assert {record.execution.task_id for record in records} == {"task-a", "task-b"}
    assert len({record.execution_ref.sha256 for record in records}) == 2
    assert len({record.outcome_ref.sha256 for record in records}) == 2
    for record in records:
        assert store.get_json(record.execution_ref, ModelExecution) == record.execution
        outcome = store.get_json(record.outcome_ref, AttemptOutcome)
        reservation = store.get_json(outcome.reservation_ref, AttemptReservation)
        assert outcome.execution_ref == record.execution_ref
        assert reservation.request_sha256 == record.execution.request_sha256
        assert reservation.execution_fingerprint == record.execution.execution_fingerprint

    # Compatibility diagnostics may point at whichever call completed last,
    # but the two refs themselves are updated under the same runner lock.
    assert (runner.last_execution_ref, runner.last_outcome_ref) in {
        (record.execution_ref, record.outcome_ref) for record in records
    }
    with pytest.raises(ValidationError):
        records[0].outcome_ref = records[1].outcome_ref
