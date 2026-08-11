"""Provider-neutral fixed-model execution with score-free results."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from threading import RLock
from typing import Protocol, runtime_checkable

import spiral_harness.execution.contracts as _contracts
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import (
    AttemptAccountingError,
    AttemptLedger,
)
from spiral_harness.storage.protocol import ArtifactRepository

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _UnspecifiedLedgerTail:
    __slots__ = ()


_UNSPECIFIED_LEDGER_TAIL = _UnspecifiedLedgerTail()


@runtime_checkable
class ModelBackend(Protocol):
    """Provider adapter boundary; implementations have no grading authority."""

    @property
    def fingerprint(self) -> str: ...

    def invoke(
        self,
        *,
        spec: _contracts.FrozenModelSpec,
        request: _contracts.ModelRequest,
    ) -> _contracts.BackendResponse: ...


class BackendFingerprintMismatchError(ValueError):
    """Raised before reservation when a backend differs from the frozen spec."""


class ReplayMissError(LookupError):
    """Raised when a deterministic replay has no exact request record."""


class ReplayBackend:
    """Provider-free backend keyed by the complete frozen spec and request."""

    def __init__(
        self,
        *,
        fingerprint: str,
        responses: Mapping[str, _contracts.BackendResponse] | None = None,
        default_response: _contracts.BackendResponse | None = None,
    ) -> None:
        if not isinstance(fingerprint, str):
            raise TypeError("fingerprint must be a string")
        if not fingerprint or fingerprint != fingerprint.strip():
            raise ValueError("fingerprint must be an exact non-empty string")
        checked: dict[str, _contracts.BackendResponse] = {}
        for key, response in (responses or {}).items():
            if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
                raise ValueError("replay response keys must be canonical SHA-256 values")
            checked[key] = _contracts.BackendResponse.model_validate(response, strict=True)
        self._fingerprint = fingerprint
        self._responses = checked
        self._default_response = (
            None
            if default_response is None
            else _contracts.BackendResponse.model_validate(default_response, strict=True)
        )
        self._calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def invoke(
        self,
        *,
        spec: _contracts.FrozenModelSpec,
        request: _contracts.ModelRequest,
    ) -> _contracts.BackendResponse:
        checked_spec = _contracts.FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = _contracts.ModelRequest.model_validate(request, strict=True)
        if checked_spec.backend_fingerprint != self._fingerprint:
            raise BackendFingerprintMismatchError(
                "replay backend fingerprint does not match the frozen model spec"
            )
        key = replay_key(checked_spec, checked_request)
        self._calls.append(key)
        response = self._responses.get(key, self._default_response)
        if response is None:
            raise ReplayMissError(f"no replay response for frozen spec/request {key}")
        return _contracts.BackendResponse.model_validate(response, strict=True)


def replay_key(spec: _contracts.FrozenModelSpec, request: _contracts.ModelRequest) -> str:
    """Bind a replay record to both the exact model spec and exact request."""

    checked_spec = _contracts.FrozenModelSpec.model_validate(spec, strict=True)
    checked_request = _contracts.ModelRequest.model_validate(request, strict=True)
    return canonical_sha256(
        {
            "schema": "spiral-harness/model-replay-key/v2",
            "spec_fingerprint": checked_spec.fingerprint,
            "request_sha256": checked_request.fingerprint,
        }
    )


def materialize_request(
    task: object,
    harness: _contracts.ResolvedHarness,
    *,
    seed: int,
) -> _contracts.ModelRequest:
    """Build the exact structured backend request from candidate-visible inputs."""

    checked_task = _contracts.CandidateTask.from_task_view(task)
    checked_harness = _contracts.ResolvedHarness.model_validate(harness, strict=True)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must not be negative")
    return _contracts.ModelRequest(
        task_id=checked_task.task_id,
        harness_ref=checked_harness.harness_ref,
        base_system_prompt=checked_harness.base_system_prompt,
        base_system_prompt_sha256=checked_harness.base_system_prompt_sha256,
        skill_disclosure=checked_harness.skill_disclosure,
        system_prompt=checked_harness.system_prompt,
        resolved_prompt_sha256=checked_harness.resolved_prompt_sha256,
        user_prompt=checked_task.question,
        seed=seed,
    )


def paired_execution_fingerprint(
    spec: _contracts.FrozenModelSpec,
    task: object,
    *,
    seed: int,
    backend_fingerprint: str,
) -> str:
    """Hash paired context while deliberately excluding harness arm and prompt."""

    checked_spec = _contracts.FrozenModelSpec.model_validate(spec, strict=True)
    checked_task = _contracts.CandidateTask.from_task_view(task)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must not be negative")
    if not isinstance(backend_fingerprint, str) or not backend_fingerprint:
        raise ValueError("backend_fingerprint must be a non-empty string")
    return canonical_sha256(
        {
            "schema": "spiral-harness/paired-model-execution/v1",
            "backend_fingerprint": backend_fingerprint,
            "model_fingerprint": checked_spec.model_fingerprint,
            "inference_fingerprint": checked_spec.inference_fingerprint,
            "runtime_fingerprint": checked_spec.runtime_fingerprint,
            "spec_fingerprint": checked_spec.fingerprint,
            "task": checked_task,
            "seed": seed,
        }
    )


class FixedModelRunner:
    """Execute a frozen model only after reserving one auditable attempt."""

    _IMPLEMENTATION_ID = "spiral-harness/fixed-model-runner/v2"

    def __init__(
        self,
        *,
        spec: _contracts.FrozenModelSpec,
        backend: ModelBackend,
        attempt_ledger: AttemptLedger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._spec = _contracts.FrozenModelSpec.model_validate(spec, strict=True)
        if not isinstance(backend, ModelBackend):
            raise TypeError("backend must implement ModelBackend")
        if not isinstance(attempt_ledger, AttemptLedger):
            raise TypeError("attempt_ledger must be an AttemptLedger")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._backend = backend
        self._attempt_ledger = attempt_ledger
        self._clock = clock
        self._lock = RLock()
        self._last_execution_ref: ArtifactRef | None = None
        self._last_outcome_ref: ArtifactRef | None = None
        self._require_backend_match()

    @property
    def spec(self) -> _contracts.FrozenModelSpec:
        return self._spec

    @property
    def fingerprint(self) -> str:
        """Stable runner identity; injected clocks and mutable ledger state are excluded."""

        return canonical_sha256(
            {
                "implementation": self._IMPLEMENTATION_ID,
                "spec_fingerprint": self._spec.fingerprint,
                "backend_fingerprint": self._spec.backend_fingerprint,
            }
        )

    @property
    def last_execution_ref(self) -> ArtifactRef | None:
        """Most recently persisted execution ref, for local orchestration diagnostics."""

        with self._lock:
            return self._last_execution_ref

    @property
    def last_outcome_ref(self) -> ArtifactRef | None:
        """Most recent immutable attempt outcome ref."""

        with self._lock:
            return self._last_outcome_ref

    @property
    def repository(self) -> ArtifactRepository:
        """Trusted artifact repository used by execution and attempt accounting."""

        return self._attempt_ledger.repository

    def attempt_state(self) -> _contracts.AttemptLedgerState:
        """Return a replay-verified, read-only view of the current attempt ledger."""

        return self._attempt_ledger.state()

    def execute(
        self,
        task: object,
        *,
        harness: _contracts.ResolvedHarness,
        seed: int,
    ) -> _contracts.ModelExecution:
        """Reserve, invoke once, then settle or conservatively burn the attempt."""

        with self._lock:
            return self._execute_record(task, harness=harness, seed=seed).execution

    def execute_record(
        self,
        task: object,
        *,
        harness: _contracts.ResolvedHarness,
        seed: int,
        reservation_token_ceiling: int | None = None,
        expected_previous_ledger_tail_ref: (
            ArtifactRef | _UnspecifiedLedgerTail | None
        ) = _UNSPECIFIED_LEDGER_TAIL,
    ) -> _contracts.ModelExecutionRecord:
        """Return execution and accounting refs from one runner-atomic call."""

        with self._lock:
            if not isinstance(expected_previous_ledger_tail_ref, _UnspecifiedLedgerTail):
                expected_tail = (
                    None
                    if expected_previous_ledger_tail_ref is None
                    else ArtifactRef.model_validate(
                        expected_previous_ledger_tail_ref,
                        strict=True,
                    )
                )
                if self._attempt_ledger.state().tail_ref != expected_tail:
                    raise AttemptAccountingError(
                        "attempt ledger advanced beyond the expected scheduled boundary"
                    )
            return self._execute_record(
                task,
                harness=harness,
                seed=seed,
                reservation_token_ceiling=reservation_token_ceiling,
                expected_previous_ledger_tail_ref=expected_previous_ledger_tail_ref,
            )

    def _execute_record(
        self,
        task: object,
        *,
        harness: _contracts.ResolvedHarness,
        seed: int,
        reservation_token_ceiling: int | None = None,
        expected_previous_ledger_tail_ref: (
            ArtifactRef | _UnspecifiedLedgerTail | None
        ) = _UNSPECIFIED_LEDGER_TAIL,
    ) -> _contracts.ModelExecutionRecord:
        """Execute under ``_lock``; callers must never invoke this helper directly."""

        checked_task = _contracts.CandidateTask.from_task_view(task)
        checked_harness = _contracts.ResolvedHarness.model_validate(harness, strict=True)
        request = materialize_request(checked_task, checked_harness, seed=seed)
        backend_fingerprint = self._require_backend_match()
        execution_fingerprint = paired_execution_fingerprint(
            self._spec,
            checked_task,
            seed=seed,
            backend_fingerprint=backend_fingerprint,
        )
        request_sha256 = request.fingerprint
        effective_token_ceiling = (
            self._attempt_ledger.budget.max_tokens_per_attempt
            if reservation_token_ceiling is None
            else reservation_token_ceiling
        )
        if isinstance(expected_previous_ledger_tail_ref, _UnspecifiedLedgerTail):
            reservation_ref = self._attempt_ledger.reserve(
                task_fingerprint=checked_task.fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                token_ceiling=effective_token_ceiling,
            )
        else:
            reservation_ref = self._attempt_ledger.reserve_at_tail(
                expected_previous_outcome_ref=expected_previous_ledger_tail_ref,
                task_fingerprint=checked_task.fingerprint,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                token_ceiling=effective_token_ceiling,
            )

        started = self._read_clock()
        try:
            raw_response = self._backend.invoke(spec=self._spec, request=request)
        except TimeoutError as exc:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=_contracts.BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=self._elapsed_ms(started),
                error_class=_contracts.ExecutionErrorClass.BACKEND_TIMEOUT,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )
        except Exception as exc:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=_contracts.BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=self._elapsed_ms(started),
                error_class=_contracts.ExecutionErrorClass.BACKEND_EXCEPTION,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )

        latency_ms = self._elapsed_ms(started)
        try:
            current_backend_fingerprint = self._backend.fingerprint
        except Exception as exc:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=_contracts.BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=_contracts.ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )
        if current_backend_fingerprint != backend_fingerprint:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=_contracts.BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=_contracts.ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
                detail="backend fingerprint changed during execution",
                cost_usd=None,
            )
        try:
            response = _contracts.BackendResponse.model_validate(raw_response, strict=True)
        except Exception as exc:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=_contracts.BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=_contracts.ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )

        reserved_tokens = effective_token_ceiling
        if (
            response.usage.total_tokens > reserved_tokens
            or response.usage.output_tokens > self._spec.inference.max_output_tokens
        ):
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=_contracts.ExecutionErrorClass.USAGE_EXCEEDED,
                detail="backend-reported usage exceeded a frozen token ceiling",
                cost_usd=response.cost_usd,
            )

        execution = self._make_execution(
            task=checked_task,
            request=request,
            output=response.output,
            status=_contracts.ExecutionStatus.COMPLETED,
            usage=response.usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            error=None,
            cost_usd=response.cost_usd,
        )
        execution_ref = self._persist_execution(execution)
        try:
            outcome_ref = self._attempt_ledger.settle(
                reservation_ref,
                execution_ref=execution_ref,
            )
        except Exception as exc:
            # No successful result may escape without terminal accounting.  If
            # the normal settlement path failed while the reservation remains
            # current, attempt a conservative burn and then surface the error.
            with suppress(Exception):
                self._attempt_ledger.burn(
                    reservation_ref,
                    error_class="accounting_settlement_failure",
                    execution_ref=execution_ref,
                )
            raise AttemptAccountingError("successful execution could not be settled") from exc
        self._last_execution_ref = execution_ref
        self._last_outcome_ref = outcome_ref
        return _contracts.ModelExecutionRecord(
            execution=execution,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _record_failure(
        self,
        *,
        reservation_ref: ArtifactRef,
        task: _contracts.CandidateTask,
        request: _contracts.ModelRequest,
        execution_fingerprint: str,
        request_sha256: str,
        usage: _contracts.BackendTokenUsage,
        latency_ms: float,
        error_class: _contracts.ExecutionErrorClass,
        detail: str,
        cost_usd: float | None,
    ) -> _contracts.ModelExecutionRecord:
        execution = self._make_execution(
            task=task,
            request=request,
            output=None,
            status=_contracts.ExecutionStatus.FAILED,
            usage=usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            error=_contracts.ExecutionError(error_class=error_class, detail=detail),
            cost_usd=cost_usd,
        )
        try:
            execution_ref = self._persist_execution(execution)
        except Exception as exc:
            with suppress(Exception):
                self._attempt_ledger.burn(
                    reservation_ref,
                    error_class="execution_persistence_failure",
                    reported_tokens=usage.total_tokens,
                )
            raise AttemptAccountingError("failed execution could not be persisted") from exc
        outcome_ref = self._attempt_ledger.burn(
            reservation_ref,
            error_class=error_class.value,
            execution_ref=execution_ref,
        )
        self._last_execution_ref = execution_ref
        self._last_outcome_ref = outcome_ref
        return _contracts.ModelExecutionRecord(
            execution=execution,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _make_execution(
        self,
        *,
        task: _contracts.CandidateTask,
        request: _contracts.ModelRequest,
        output: str | None,
        status: _contracts.ExecutionStatus,
        usage: _contracts.BackendTokenUsage,
        latency_ms: float,
        execution_fingerprint: str,
        request_sha256: str,
        error: _contracts.ExecutionError | None,
        cost_usd: float | None,
    ) -> _contracts.ModelExecution:
        return _contracts.ModelExecution(
            task=task,
            request=request,
            output=output,
            status=status,
            usage=_contracts.ModelUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            ),
            spec=self._spec,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            error=error,
        )

    def _persist_execution(self, execution: _contracts.ModelExecution) -> ArtifactRef:
        raw_ref = self._attempt_ledger.repository.put_json(
            execution,
            media_type=_contracts.MODEL_EXECUTION_MEDIA_TYPE,
        )
        ref = ArtifactRef.model_validate(raw_ref, strict=True)
        if ref.media_type != _contracts.MODEL_EXECUTION_MEDIA_TYPE:
            raise AttemptAccountingError("repository returned the wrong execution media type")
        loaded = self._attempt_ledger.repository.get_json(ref, _contracts.ModelExecution)
        checked = _contracts.ModelExecution.model_validate(loaded, strict=True)
        if checked != execution:
            raise AttemptAccountingError("persisted execution content changed")
        return ref

    def _require_backend_match(self) -> str:
        try:
            fingerprint = self._backend.fingerprint
        except Exception as exc:
            raise BackendFingerprintMismatchError("backend fingerprint is unavailable") from exc
        if not isinstance(fingerprint, str) or fingerprint != self._spec.backend_fingerprint:
            raise BackendFingerprintMismatchError(
                "backend fingerprint does not exactly match FrozenModelSpec.backend_fingerprint"
            )
        return fingerprint

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
        # Backend exception strings can contain request URLs, credentials, or
        # provider payloads.  Persist a stable class only; provider adapters may
        # emit separately redacted diagnostics through a trusted channel.
        return f"{type(exc).__module__}.{type(exc).__qualname__}"[:1_024]


__all__ = [
    "BackendFingerprintMismatchError",
    "FixedModelRunner",
    "ModelBackend",
    "ReplayBackend",
    "ReplayMissError",
    "materialize_request",
    "paired_execution_fingerprint",
    "replay_key",
]
