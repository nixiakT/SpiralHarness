"""One-shot native function calls under conservative attempt accounting."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, runtime_checkable

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptAccountingError, AttemptLedger
from spiral_harness.execution.backend_errors import BackendResponseRejectedError
from spiral_harness.execution.contracts import (
    AttemptLedgerState,
    BackendTokenUsage,
    ExecutionError,
    ExecutionErrorClass,
    ExecutionStatus,
    FrozenModelSpec,
)
from spiral_harness.execution.model import BackendFingerprintMismatchError
from spiral_harness.execution.native_function_contracts import (
    NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE,
    NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
    NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE,
    NativeFunctionExecution,
    NativeFunctionExecutionRecord,
    NativeFunctionSlotBinding,
    NativeFunctionUsage,
    load_canonical_native_artifact,
    native_function_execution_fingerprint,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.protocol import ArtifactRepository

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _UnspecifiedLedgerTail:
    __slots__ = ()


_UNSPECIFIED_LEDGER_TAIL = _UnspecifiedLedgerTail()


class NativeFunctionBindingError(ValueError):
    """Raised before reservation when frozen native identities differ."""


class NativeFunctionSlotConsumedError(RuntimeError):
    """Raised before invocation when this runner has already consumed a slot."""


@runtime_checkable
class NativeFunctionBackend(Protocol):
    """Score-free native provider boundary; no property is an attestation."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def serializer_fingerprint(self) -> str: ...

    @property
    def parser_fingerprint(self) -> str: ...

    @property
    def transport_fingerprint(self) -> str: ...

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse: ...


@dataclass(frozen=True, slots=True)
class _BackendBinding:
    backend: str
    serializer: str
    parser: str
    transport: str


class NativeFunctionRunner:
    """Reserve one slot, invoke once, and terminally account without retry."""

    _IMPLEMENTATION_ID = "spiral-harness/native-function-runner/v1"

    def __init__(
        self,
        *,
        spec: FrozenModelSpec,
        backend: NativeFunctionBackend,
        attempt_ledger: AttemptLedger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._spec = FrozenModelSpec.model_validate(spec, strict=True)
        if not isinstance(backend, NativeFunctionBackend):
            raise TypeError("backend must implement NativeFunctionBackend")
        if type(attempt_ledger) is not AttemptLedger:
            raise TypeError("attempt_ledger must be an exact AttemptLedger")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._backend = backend
        self._ledger = attempt_ledger
        self._clock = clock
        self._lock = RLock()
        self._consumed_slots: set[str] = set()
        self._initial_backend_binding = self._read_backend_binding()
        if self._initial_backend_binding.backend != self._spec.backend_fingerprint:
            raise NativeFunctionBindingError(
                "backend fingerprint differs from the frozen model spec"
            )

    @property
    def spec(self) -> FrozenModelSpec:
        return self._spec

    @property
    def repository(self) -> ArtifactRepository:
        return self._ledger.repository

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "implementation": self._IMPLEMENTATION_ID,
                "spec_fingerprint": self._spec.fingerprint,
                "backend": {
                    "fingerprint": self._initial_backend_binding.backend,
                    "serializer": self._initial_backend_binding.serializer,
                    "parser": self._initial_backend_binding.parser,
                    "transport": self._initial_backend_binding.transport,
                },
            }
        )

    def attempt_state(self) -> AttemptLedgerState:
        return self._ledger.state()

    def execute_record(
        self,
        *,
        task_fingerprint: str,
        slot_fingerprint: str,
        request: NativeFunctionCallRequest,
        reservation_token_ceiling: int | None = None,
        expected_previous_ledger_tail_ref: (
            ArtifactRef | _UnspecifiedLedgerTail | None
        ) = _UNSPECIFIED_LEDGER_TAIL,
    ) -> NativeFunctionExecutionRecord:
        """Execute a process-local single-use slot with no automatic retry."""

        with self._lock:
            return self._execute_record(
                task_fingerprint=task_fingerprint,
                slot_fingerprint=slot_fingerprint,
                request=request,
                reservation_token_ceiling=reservation_token_ceiling,
                expected_previous_ledger_tail_ref=expected_previous_ledger_tail_ref,
            )

    def _execute_record(
        self,
        *,
        task_fingerprint: str,
        slot_fingerprint: str,
        request: NativeFunctionCallRequest,
        reservation_token_ceiling: int | None,
        expected_previous_ledger_tail_ref: ArtifactRef | _UnspecifiedLedgerTail | None,
    ) -> NativeFunctionExecutionRecord:
        task = NativeFunctionSlotBinding(
            task_fingerprint=task_fingerprint,
            slot_fingerprint=slot_fingerprint,
        )
        checked_request = NativeFunctionCallRequest.model_validate(request, strict=True)
        binding = self._require_request_binding(checked_request)
        if task.slot_fingerprint in self._consumed_slots:
            raise NativeFunctionSlotConsumedError(
                "native function slot was already consumed by this runner"
            )
        request_ref = self._persist_model(
            checked_request,
            model_type=NativeFunctionCallRequest,
            media_type=NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
            label="native request",
        )
        execution_fingerprint = native_function_execution_fingerprint(
            spec=self._spec,
            task=task,
            request=checked_request,
        )
        ceiling = (
            self._ledger.budget.max_tokens_per_attempt
            if reservation_token_ceiling is None
            else reservation_token_ceiling
        )
        if isinstance(expected_previous_ledger_tail_ref, _UnspecifiedLedgerTail):
            reservation_ref = self._ledger.reserve(
                task_fingerprint=task.task_fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=checked_request.fingerprint,
                token_ceiling=ceiling,
            )
        else:
            reservation_ref = self._ledger.reserve_at_tail(
                expected_previous_outcome_ref=expected_previous_ledger_tail_ref,
                task_fingerprint=task.task_fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=checked_request.fingerprint,
                token_ceiling=ceiling,
            )
        self._consumed_slots.add(task.slot_fingerprint)

        started = self._read_clock()
        try:
            raw_response = self._backend.invoke(request=checked_request)
        except BackendResponseRejectedError as exc:
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                response=None,
                response_ref=None,
                execution_fingerprint=execution_fingerprint,
                usage=exc.usage or BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=self._elapsed_ms(started),
                error_class=ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail=self._exception_detail(exc),
                cost_usd=exc.cost_usd,
            )
        except TimeoutError as exc:
            return self._exception_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                execution_fingerprint=execution_fingerprint,
                started=started,
                error_class=ExecutionErrorClass.BACKEND_TIMEOUT,
                exc=exc,
            )
        except Exception as exc:
            return self._exception_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                execution_fingerprint=execution_fingerprint,
                started=started,
                error_class=ExecutionErrorClass.BACKEND_EXCEPTION,
                exc=exc,
            )

        latency_ms = self._elapsed_ms(started)
        try:
            response = NativeFunctionCallResponse.model_validate(raw_response, strict=True)
        except Exception as exc:
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                response=None,
                response_ref=None,
                execution_fingerprint=execution_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )
        response_error = self._response_binding_error(checked_request, response)
        if response_error is not None:
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                response=None,
                response_ref=None,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail=response_error,
                cost_usd=None,
            )
        try:
            response_ref = self._persist_model(
                response,
                model_type=NativeFunctionCallResponse,
                media_type=NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE,
                label="native response",
            )
        except Exception as exc:
            with suppress(Exception):
                self._ledger.burn(
                    reservation_ref,
                    error_class="response_persistence_failure",
                    reported_tokens=response.usage.total_tokens,
                )
            raise AttemptAccountingError("native response could not be persisted") from exc
        try:
            if self._read_backend_binding() != binding:
                raise BackendFingerprintMismatchError(
                    "native backend identities changed during execution"
                )
        except Exception as exc:
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                response=response,
                response_ref=response_ref,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )
        if (
            response.usage.total_tokens > ceiling
            or response.usage.output_tokens > self._spec.inference.max_output_tokens
        ):
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=task,
                request=checked_request,
                request_ref=request_ref,
                response=response,
                response_ref=response_ref,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.USAGE_EXCEEDED,
                detail="backend-reported usage exceeded a frozen token ceiling",
                cost_usd=None,
            )

        execution = self._make_execution(
            task=task,
            request=checked_request,
            request_ref=request_ref,
            response=response,
            response_ref=response_ref,
            status=ExecutionStatus.COMPLETED,
            usage=response.usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            error=None,
            cost_usd=None,
        )
        execution_ref = self._persist_execution(execution)
        try:
            outcome_ref = self._ledger.settle(reservation_ref, execution_ref=execution_ref)
        except Exception as exc:
            with suppress(Exception):
                self._ledger.burn(
                    reservation_ref,
                    error_class="accounting_settlement_failure",
                    execution_ref=execution_ref,
                )
            raise AttemptAccountingError(
                "successful native execution could not be settled"
            ) from exc
        return NativeFunctionExecutionRecord(
            execution=execution,
            request_ref=request_ref,
            response_ref=response_ref,
            reservation_ref=reservation_ref,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _exception_failure(
        self,
        *,
        reservation_ref: ArtifactRef,
        task: NativeFunctionSlotBinding,
        request: NativeFunctionCallRequest,
        request_ref: ArtifactRef,
        execution_fingerprint: str,
        started: float | None,
        error_class: ExecutionErrorClass,
        exc: Exception,
    ) -> NativeFunctionExecutionRecord:
        return self._finish_failure(
            reservation_ref=reservation_ref,
            task=task,
            request=request,
            request_ref=request_ref,
            response=None,
            response_ref=None,
            execution_fingerprint=execution_fingerprint,
            usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
            latency_ms=self._elapsed_ms(started),
            error_class=error_class,
            detail=self._exception_detail(exc),
            cost_usd=None,
        )

    def _finish_failure(
        self,
        *,
        reservation_ref: ArtifactRef,
        task: NativeFunctionSlotBinding,
        request: NativeFunctionCallRequest,
        request_ref: ArtifactRef,
        response: NativeFunctionCallResponse | None,
        response_ref: ArtifactRef | None,
        execution_fingerprint: str,
        usage: BackendTokenUsage,
        latency_ms: float,
        error_class: ExecutionErrorClass,
        detail: str,
        cost_usd: float | None,
    ) -> NativeFunctionExecutionRecord:
        execution = self._make_execution(
            task=task,
            request=request,
            request_ref=request_ref,
            response=response,
            response_ref=response_ref,
            status=ExecutionStatus.FAILED,
            usage=usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            error=ExecutionError(error_class=error_class, detail=detail),
            cost_usd=cost_usd,
        )
        try:
            execution_ref = self._persist_execution(execution)
        except Exception as exc:
            with suppress(Exception):
                self._ledger.burn(
                    reservation_ref,
                    error_class="execution_persistence_failure",
                    reported_tokens=usage.total_tokens,
                )
            raise AttemptAccountingError("failed native execution could not be persisted") from exc
        outcome_ref = self._ledger.burn(
            reservation_ref,
            error_class=error_class.value,
            execution_ref=execution_ref,
        )
        return NativeFunctionExecutionRecord(
            execution=execution,
            request_ref=request_ref,
            response_ref=response_ref,
            reservation_ref=reservation_ref,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _make_execution(
        self,
        *,
        task: NativeFunctionSlotBinding,
        request: NativeFunctionCallRequest,
        request_ref: ArtifactRef,
        response: NativeFunctionCallResponse | None,
        response_ref: ArtifactRef | None,
        status: ExecutionStatus,
        usage: BackendTokenUsage,
        latency_ms: float,
        execution_fingerprint: str,
        error: ExecutionError | None,
        cost_usd: float | None,
    ) -> NativeFunctionExecution:
        return NativeFunctionExecution(
            task=task,
            spec=self._spec,
            request=request,
            request_ref=request_ref,
            response=response,
            response_ref=response_ref,
            status=status,
            usage=NativeFunctionUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                tool_calls=0 if response is None else response.tool_call_count,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            ),
            execution_fingerprint=execution_fingerprint,
            request_sha256=request.fingerprint,
            error=error,
        )

    def _persist_execution(self, execution: NativeFunctionExecution) -> ArtifactRef:
        return self._persist_model(
            execution,
            model_type=NativeFunctionExecution,
            media_type=NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE,
            label="native execution",
        )

    def _persist_model[ModelT](
        self,
        value: ModelT,
        *,
        model_type: type[ModelT],
        media_type: str,
        label: str,
    ) -> ArtifactRef:
        raw_ref = self._ledger.repository.put_json(value, media_type=media_type)
        try:
            ref = ArtifactRef.model_validate(raw_ref, strict=True)
        except Exception as exc:
            raise AttemptAccountingError(f"repository returned a malformed {label} ref") from exc
        if ref.media_type != media_type:
            raise AttemptAccountingError(f"repository returned the wrong {label} media type")
        try:
            checked = load_canonical_native_artifact(
                self._ledger.repository,
                ref,
                model_type,  # type: ignore[type-var]
            )
        except Exception as exc:
            raise AttemptAccountingError(f"persisted {label} cannot be verified") from exc
        if checked != value:
            raise AttemptAccountingError(f"persisted {label} content changed")
        return ref

    def _require_request_binding(self, request: NativeFunctionCallRequest) -> _BackendBinding:
        binding = self._read_backend_binding()
        if binding != self._initial_backend_binding:
            raise NativeFunctionBindingError(
                "native backend identities drifted from runner construction"
            )
        expected = (
            (request.requested_model, self._spec.model, "model"),
            (request.inference, self._spec.inference, "inference"),
            (request.backend_fingerprint, self._spec.backend_fingerprint, "backend"),
            (request.backend_fingerprint, binding.backend, "backend"),
            (request.serializer_fingerprint, binding.serializer, "serializer"),
            (request.parser_fingerprint, binding.parser, "parser"),
            (request.transport_fingerprint, binding.transport, "transport"),
        )
        for actual, frozen, label in expected:
            if actual != frozen:
                raise NativeFunctionBindingError(
                    f"native request {label} differs from its frozen execution boundary"
                )
        return binding

    def _read_backend_binding(self) -> _BackendBinding:
        try:
            binding = _BackendBinding(
                backend=self._backend.fingerprint,
                serializer=self._backend.serializer_fingerprint,
                parser=self._backend.parser_fingerprint,
                transport=self._backend.transport_fingerprint,
            )
        except Exception as exc:
            raise BackendFingerprintMismatchError(
                "native backend identities are unavailable"
            ) from exc
        values = (binding.backend, binding.serializer, binding.parser, binding.transport)
        if any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in values
        ):
            raise BackendFingerprintMismatchError(
                "native backend identities must be canonical SHA-256 values"
            )
        return binding

    def _response_binding_error(
        self,
        request: NativeFunctionCallRequest,
        response: NativeFunctionCallResponse,
    ) -> str | None:
        expected = (
            (response.request_fingerprint, request.fingerprint),
            (response.serializer_fingerprint, request.serializer_fingerprint),
            (response.parser_fingerprint, request.parser_fingerprint),
            (response.transport_fingerprint, request.transport_fingerprint),
            (response.tools_fingerprint, request.tools_fingerprint),
        )
        if any(actual != frozen for actual, frozen in expected):
            return "native response does not bind the exact frozen request"
        identity = response.provider_identity_observation
        if identity is not None and (
            identity.requested_model != self._spec.model
            or identity.backend_fingerprint != self._spec.backend_fingerprint
        ):
            return "provider identity observation does not bind the frozen native request"
        return None

    def _read_clock(self) -> float | None:
        try:
            value = self._clock()
        except Exception:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def _elapsed_ms(self, started: float | None) -> float:
        ended = self._read_clock()
        if started is None or ended is None:
            return 0.0
        return max(0.0, (ended - started) * 1_000.0)

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        # Exception text can contain credentials, URLs, or provider payloads.
        return f"{type(exc).__module__}.{type(exc).__qualname__}"[:1_024]


__all__ = [
    "NativeFunctionBackend",
    "NativeFunctionBindingError",
    "NativeFunctionRunner",
    "NativeFunctionSlotConsumedError",
]
