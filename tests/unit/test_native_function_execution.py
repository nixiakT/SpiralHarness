from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.accounted_execution import load_accounted_execution
from spiral_harness.execution.attempts import (
    AttemptAccountingError,
    AttemptBudgetExceeded,
    AttemptLedger,
    AttemptLedgerIntegrityError,
)
from spiral_harness.execution.backend_errors import BackendResponseRejectedError
from spiral_harness.execution.contracts import (
    AttemptBudget,
    AttemptDisposition,
    AttemptOutcome,
    BackendTokenUsage,
    ExecutionErrorClass,
    ExecutionStatus,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.native_function_contracts import (
    NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE,
    NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
    NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE,
    NativeFunctionExecution,
)
from spiral_harness.execution.native_function_execution import (
    NativeFunctionBindingError,
    NativeFunctionRunner,
    NativeFunctionSlotConsumedError,
)
from spiral_harness.execution.native_function_replay import (
    NativeFunctionReplayBackend,
    native_function_replay_key,
)
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BACKEND = "a" * 64
SERIALIZER = "b" * 64
PARSER = "c" * 64
TRANSPORT = "d" * 64
TASK = "e" * 64
SLOT = "f" * 64


def _inference(*, max_output_tokens: int = 20) -> InferenceConfig:
    return InferenceConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=max_output_tokens,
        timeout_seconds=5.0,
    )


def _spec(*, inference: InferenceConfig | None = None) -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="openai-compatible-native-fixture",
        backend_fingerprint=BACKEND,
        model="fixture/native-model",
        revision="snapshot-2026-08-15",
        tokenizer="fixture/native-tokenizer",
        tokenizer_revision="snapshot-2026-08-15",
        runtime="fixture-runtime@sha256:native-v1",
        inference=_inference() if inference is None else inference,
    )


def _tool() -> FrozenNativeFunctionTool:
    return FrozenNativeFunctionTool.from_schema(
        {
            "name": "official.lookup",
            "description": "Return a fixture value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        wire_name="official_lookup",
    )


def _request(*, inference: InferenceConfig | None = None) -> NativeFunctionCallRequest:
    return NativeFunctionCallRequest(
        backend_fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
        requested_model="fixture/native-model",
        messages=(
            FrozenNativeChatMessage(role="system", content="Use the supplied function."),
            FrozenNativeChatMessage(role="user", content="Look up value 3."),
        ),
        task_required_tools=(_tool(),),
        seed=17,
        inference=_inference() if inference is None else inference,
    )


def _response(
    request: NativeFunctionCallRequest,
    *,
    input_tokens: int = 8,
    output_tokens: int = 3,
    request_fingerprint: str | None = None,
) -> NativeFunctionCallResponse:
    return NativeFunctionCallResponse(
        request_fingerprint=(
            request.fingerprint if request_fingerprint is None else request_fingerprint
        ),
        serializer_fingerprint=request.serializer_fingerprint,
        parser_fingerprint=request.parser_fingerprint,
        transport_fingerprint=request.transport_fingerprint,
        tools_fingerprint=request.tools_fingerprint,
        tool_calls=(
            NativeAssistantToolCall(
                call_id="call-1",
                official_name="official.lookup",
                wire_name="official_lookup",
                arguments_json=canonical_json({"value": 3}),
            ),
        ),
        assistant_text=None,
        finish_reason="tool_calls",
        usage=BackendTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _context(
    tmp_path: Path,
    *,
    request: NativeFunctionCallRequest | None = None,
    response: NativeFunctionCallResponse | None = None,
    backend: object | None = None,
    attempts: int = 3,
    total_tokens: int = 90,
    per_attempt: int = 30,
) -> tuple[ArtifactStore, AttemptLedger, object, NativeFunctionRunner]:
    checked_request = _request() if request is None else request
    checked_response = _response(checked_request) if response is None else response
    checked_backend = backend or NativeFunctionReplayBackend(
        fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
        responses={native_function_replay_key(checked_request): checked_response},
    )
    store = ArtifactStore(tmp_path / "cas")
    ledger = AttemptLedger(
        store,
        ledger_id="native-fixture",
        budget=AttemptBudget(
            max_attempts=attempts,
            max_total_tokens=total_tokens,
            max_tokens_per_attempt=per_attempt,
        ),
    )
    runner = NativeFunctionRunner(
        spec=_spec(inference=checked_request.inference),
        backend=checked_backend,  # type: ignore[arg-type]
        attempt_ledger=ledger,
        clock=iter((1.0, 1.25)).__next__,
    )
    return store, ledger, checked_backend, runner


def _execute(
    runner: NativeFunctionRunner,
    request: NativeFunctionCallRequest,
    *,
    slot: str = SLOT,
    ceiling: int | None = None,
):
    return runner.execute_record(
        task_fingerprint=TASK,
        slot_fingerprint=slot,
        request=request,
        reservation_token_ceiling=ceiling,
    )


def test_replay_backend_executes_once_and_publishes_complete_accounting(tmp_path: Path) -> None:
    request = _request()
    store, ledger, backend, runner = _context(tmp_path, request=request)

    record = _execute(runner, request)
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)
    loaded = load_accounted_execution(store, record.execution_ref)

    assert backend.calls == (native_function_replay_key(request),)  # type: ignore[attr-defined]
    assert record.execution.status is ExecutionStatus.COMPLETED
    assert record.execution.task_fingerprint == TASK
    assert record.execution.slot_fingerprint == SLOT
    assert record.execution.request == request
    assert record.execution.response is not None
    assert record.execution.usage.total_tokens == 11
    assert record.execution.usage.tool_calls == 1
    assert record.request_ref.media_type == NATIVE_FUNCTION_REQUEST_MEDIA_TYPE
    assert record.response_ref is not None
    assert record.response_ref.media_type == NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE
    assert record.execution_ref.media_type == NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE
    assert record.execution.provider_identity_attested is False
    assert record.execution.runtime_execution_attested is False
    assert record.execution.atomicity_attested is False
    assert record.atomicity_attested is False
    assert loaded == record.execution
    assert outcome.disposition is AttemptDisposition.SETTLED
    assert outcome.execution_ref == record.execution_ref
    assert outcome.reported_tokens == outcome.charged_tokens == 11
    assert ledger.state().attempts_used == ledger.state().completed_attempts == 1


def test_one_runner_rejects_a_consumed_slot_without_second_attempt_or_invoke(
    tmp_path: Path,
) -> None:
    request = _request()
    _, ledger, backend, runner = _context(tmp_path, request=request)
    _execute(runner, request)

    with pytest.raises(NativeFunctionSlotConsumedError, match="already consumed"):
        _execute(runner, request)

    assert len(backend.calls) == 1  # type: ignore[attr-defined]
    assert ledger.state().attempts_used == 1


@pytest.mark.parametrize(
    "tamper",
    [
        lambda request: request.model_copy(update={"requested_model": "fixture/other-model"}),
        lambda request: request.model_copy(update={"inference": _inference(max_output_tokens=19)}),
        lambda request: request.model_copy(update={"backend_fingerprint": "1" * 64}),
        lambda request: request.model_copy(update={"serializer_fingerprint": "2" * 64}),
        lambda request: request.model_copy(update={"parser_fingerprint": "3" * 64}),
        lambda request: request.model_copy(update={"transport_fingerprint": "4" * 64}),
    ],
    ids=["model", "inference", "backend", "serializer", "parser", "transport"],
)
def test_all_frozen_native_bindings_fail_before_reservation(
    tmp_path: Path,
    tamper: Callable[[NativeFunctionCallRequest], NativeFunctionCallRequest],
) -> None:
    request = _request()
    _, ledger, backend, runner = _context(tmp_path, request=request)

    with pytest.raises(NativeFunctionBindingError, match="frozen execution boundary"):
        _execute(runner, tamper(request))

    assert backend.calls == ()  # type: ignore[attr-defined]
    assert ledger.state().attempts_used == 0


class _RejectingBackend:
    fingerprint = BACKEND
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(self, *, secret: str, usage: BackendTokenUsage) -> None:
        self._secret = secret
        self._usage = usage
        self.calls = 0

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        del request
        self.calls += 1
        raise BackendResponseRejectedError(self._secret, usage=self._usage)


class _ExplodingBackend:
    fingerprint = BACKEND
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self.calls = 0

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        del request
        self.calls += 1
        raise RuntimeError(self._secret)


class _ResponseFailingStore(ArtifactStore):
    def put_json(self, value, *, media_type: str = "application/json"):
        if media_type == NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE:
            raise RuntimeError("simulated response persistence failure")
        return super().put_json(value, media_type=media_type)


def test_rejected_provider_response_preserves_verified_usage_and_burns_once(
    tmp_path: Path,
) -> None:
    request = _request()
    secret = "sk-invalid-response-must-not-persist"
    backend = _RejectingBackend(
        secret=secret,
        usage=BackendTokenUsage(input_tokens=6, output_tokens=2),
    )
    store, ledger, _, runner = _context(tmp_path, request=request, backend=backend)

    record = _execute(runner, request)
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert backend.calls == 1
    assert record.execution.status is ExecutionStatus.FAILED
    assert record.execution.response is None
    assert record.response_ref is None
    assert record.execution.usage.total_tokens == 8
    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.INVALID_BACKEND_RESPONSE
    assert secret not in record.execution.error.detail
    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.reported_tokens == 8
    assert outcome.charged_tokens == 30
    assert ledger.state().attempts_used == 1


def test_backend_exception_text_and_api_key_like_secret_never_enter_artifacts_or_repr(
    tmp_path: Path,
) -> None:
    request = _request()
    secret = "sk-exception-payload-should-never-appear"
    backend = _ExplodingBackend(secret)
    store, _, _, runner = _context(tmp_path, request=request, backend=backend)

    record = _execute(runner, request)
    rendered = "\n".join((repr(runner), repr(record), record.model_dump_json()))
    artifact_bytes = b"".join(
        path.read_bytes() for path in store.objects_dir.rglob("*") if path.is_file()
    )

    assert backend.calls == 1
    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.BACKEND_EXCEPTION
    assert record.execution.error.detail.endswith("._ExplodingBackend") is False
    assert record.execution.error.detail.endswith("RuntimeError")
    assert secret not in rendered
    assert secret.encode() not in artifact_bytes


def test_response_binding_tamper_is_a_typed_failure_with_known_usage(tmp_path: Path) -> None:
    request = _request()
    response = _response(request, request_fingerprint="9" * 64)
    store, _, backend, runner = _context(
        tmp_path,
        request=request,
        response=response,
    )

    record = _execute(runner, request)
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert len(backend.calls) == 1  # type: ignore[attr-defined]
    assert record.execution.status is ExecutionStatus.FAILED
    assert record.execution.response is None
    assert record.execution.usage.total_tokens == 11
    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.INVALID_BACKEND_RESPONSE
    assert outcome.disposition is AttemptDisposition.BURNED


def test_budget_reservation_drift_fails_before_invoke_and_does_not_consume_slot(
    tmp_path: Path,
) -> None:
    request = _request()
    _, ledger, backend, runner = _context(tmp_path, request=request)

    with pytest.raises(AttemptBudgetExceeded, match="per-attempt ceiling"):
        _execute(runner, request, ceiling=31)
    record = _execute(runner, request, ceiling=30)

    assert record.execution.status is ExecutionStatus.COMPLETED
    assert len(backend.calls) == 1  # type: ignore[attr-defined]
    assert ledger.state().attempts_used == 1


def test_reported_token_overrun_poisons_without_retry(tmp_path: Path) -> None:
    request = _request()
    response = _response(request, input_tokens=21, output_tokens=10)
    store, ledger, backend, runner = _context(
        tmp_path,
        request=request,
        response=response,
        total_tokens=30,
    )

    record = _execute(runner, request, ceiling=30)
    outcome = store.get_json(record.outcome_ref, AttemptOutcome)

    assert len(backend.calls) == 1  # type: ignore[attr-defined]
    assert record.execution.status is ExecutionStatus.FAILED
    assert record.execution.error is not None
    assert record.execution.error.error_class is ExecutionErrorClass.USAGE_EXCEEDED
    assert record.execution.usage.total_tokens == 31
    assert outcome.disposition is AttemptDisposition.POISONED
    assert outcome.reported_tokens == outcome.charged_tokens == 31
    assert ledger.state().poisoned is True


def test_response_persistence_failure_terminally_burns_the_reserved_attempt(
    tmp_path: Path,
) -> None:
    request = _request()
    response = _response(request)
    backend = NativeFunctionReplayBackend(
        fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
        responses={native_function_replay_key(request): response},
    )
    store = _ResponseFailingStore(tmp_path / "cas")
    ledger = AttemptLedger(
        store,
        ledger_id="response-failure",
        budget=AttemptBudget(
            max_attempts=1,
            max_total_tokens=30,
            max_tokens_per_attempt=30,
        ),
    )
    runner = NativeFunctionRunner(
        spec=_spec(),
        backend=backend,
        attempt_ledger=ledger,
    )

    with pytest.raises(AttemptAccountingError, match="response could not be persisted"):
        _execute(runner, request)

    state = ledger.state()
    assert state.pending_reservation_ref is None
    assert state.tail_ref is not None
    outcome = store.get_json(state.tail_ref, AttemptOutcome)
    assert outcome.disposition is AttemptDisposition.BURNED
    assert outcome.error_class == "response_persistence_failure"
    assert outcome.reported_tokens == 11
    assert outcome.charged_tokens == 30
    assert outcome.execution_ref is None
    assert len(backend.calls) == 1


def test_execution_contract_and_content_store_reject_tampering(tmp_path: Path) -> None:
    request = _request()
    store, _, _, runner = _context(tmp_path, request=request)
    record = _execute(runner, request)
    raw = record.execution.model_dump(mode="python", round_trip=True)
    raw["request_ref"]["sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="request_ref does not bind"):
        NativeFunctionExecution.model_validate(raw, strict=True)

    path = store.path_for(record.execution_ref)
    payload = bytearray(path.read_bytes())
    payload[-2] = ord(" ")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="not a canonical native function execution"):
        load_accounted_execution(store, record.execution_ref)


def test_accounted_loader_requires_linked_request_and_response_artifacts(tmp_path: Path) -> None:
    request = _request()
    store, ledger, _, runner = _context(tmp_path, request=request)
    record = _execute(runner, request)
    store.path_for(record.request_ref).unlink()

    with pytest.raises(ValueError, match="not a canonical native function execution"):
        load_accounted_execution(store, record.execution_ref)
    with pytest.raises(AttemptLedgerIntegrityError, match="cannot be verified"):
        ledger.state()
