"""Auditable execution for the harness-free PURE reference condition."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from threading import RLock
from typing import Protocol, runtime_checkable

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptAccountingError, AttemptLedger
from spiral_harness.execution.contracts import (
    PURE_REFERENCE_EXECUTION_MEDIA_TYPE,
    AttemptLedgerState,
    BackendResponse,
    BackendTokenUsage,
    CandidateTask,
    ExecutionError,
    ExecutionErrorClass,
    ExecutionStatus,
    FrozenModelSpec,
    ModelUsage,
    ProviderIdentityObservation,
)
from spiral_harness.execution.model import (
    BackendFingerprintMismatchError,
    ReplayMissError,
    paired_execution_fingerprint,
)
from spiral_harness.execution.pure_contracts import (
    PureReferenceExecution,
    PureReferenceExecutionRecord,
    PureReferenceRequest,
    materialize_pure_request,
)
from spiral_harness.storage.protocol import ArtifactRepository

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class PureModelBackend(Protocol):
    """Provider boundary that can only receive a one-user-message PURE request."""

    @property
    def fingerprint(self) -> str: ...

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse: ...


def pure_replay_key(spec: FrozenModelSpec, request: PureReferenceRequest) -> str:
    """Bind a replay response to the complete model spec and exact PURE request."""

    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_request = PureReferenceRequest.model_validate(request, strict=True)
    return canonical_sha256(
        {
            "schema": "spiral-harness/pure-model-replay-key/v1",
            "spec_fingerprint": checked_spec.fingerprint,
            "request_sha256": checked_request.fingerprint,
        }
    )


class PureReplayBackend:
    """Provider-free PURE backend keyed by exact spec and request."""

    def __init__(
        self,
        *,
        fingerprint: str,
        responses: Mapping[str, BackendResponse] | None = None,
        default_response: BackendResponse | None = None,
    ) -> None:
        if not isinstance(fingerprint, str):
            raise TypeError("fingerprint must be a string")
        if not fingerprint or fingerprint != fingerprint.strip():
            raise ValueError("fingerprint must be an exact non-empty string")
        checked: dict[str, BackendResponse] = {}
        for key, response in (responses or {}).items():
            if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
                raise ValueError("replay response keys must be canonical SHA-256 values")
            checked[key] = BackendResponse.model_validate(response, strict=True)
        self._fingerprint = fingerprint
        self._responses = checked
        self._default_response = (
            None
            if default_response is None
            else BackendResponse.model_validate(default_response, strict=True)
        )
        self._calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse:
        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = PureReferenceRequest.model_validate(request, strict=True)
        if checked_spec.backend_fingerprint != self._fingerprint:
            raise BackendFingerprintMismatchError(
                "replay backend fingerprint does not match the frozen model spec"
            )
        key = pure_replay_key(checked_spec, checked_request)
        self._calls.append(key)
        response = self._responses.get(key, self._default_response)
        if response is None:
            raise ReplayMissError(f"no replay response for frozen PURE spec/request {key}")
        return BackendResponse.model_validate(response, strict=True)


class PureModelRunner:
    """Execute one bare-model call under the shared append-only attempt ledger."""

    _IMPLEMENTATION_ID = "spiral-harness/pure-model-runner/v1"

    def __init__(
        self,
        *,
        spec: FrozenModelSpec,
        backend: PureModelBackend,
        attempt_ledger: AttemptLedger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._spec = FrozenModelSpec.model_validate(spec, strict=True)
        if not isinstance(backend, PureModelBackend):
            raise TypeError("backend must implement PureModelBackend")
        if type(attempt_ledger) is not AttemptLedger:
            raise TypeError("attempt_ledger must be an exact AttemptLedger")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._backend = backend
        self._ledger = attempt_ledger
        self._clock = clock
        self._lock = RLock()
        self._require_backend_match()

    @property
    def spec(self) -> FrozenModelSpec:
        return self._spec

    @property
    def repository(self) -> ArtifactRepository:
        return self._ledger.repository

    @property
    def ledger_id(self) -> str:
        return self._ledger.ledger_id

    @property
    def budget_fingerprint(self) -> str:
        return self._ledger.budget.fingerprint

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "implementation": self._IMPLEMENTATION_ID,
                "spec_fingerprint": self._spec.fingerprint,
                "backend_fingerprint": self._spec.backend_fingerprint,
            }
        )

    def attempt_state(self) -> AttemptLedgerState:
        return self._ledger.state()

    def execute_record(
        self,
        task: object,
        *,
        reference_id: str,
        rollout_seed: int,
        reservation_token_ceiling: int | None = None,
    ) -> PureReferenceExecutionRecord:
        """Reserve, invoke exactly once, then settle or conservatively burn."""

        with self._lock:
            return self._execute_record(
                task,
                reference_id=reference_id,
                rollout_seed=rollout_seed,
                reservation_token_ceiling=reservation_token_ceiling,
            )

    def _execute_record(
        self,
        task: object,
        *,
        reference_id: str,
        rollout_seed: int,
        reservation_token_ceiling: int | None,
    ) -> PureReferenceExecutionRecord:
        checked_task = CandidateTask.from_task_view(task)
        request = materialize_pure_request(
            checked_task,
            reference_id=reference_id,
            rollout_seed=rollout_seed,
        )
        backend_fingerprint = self._require_backend_match()
        execution_fingerprint = paired_execution_fingerprint(
            self._spec,
            checked_task,
            seed=rollout_seed,
            backend_fingerprint=backend_fingerprint,
        )
        ceiling = (
            self._ledger.budget.max_tokens_per_attempt
            if reservation_token_ceiling is None
            else reservation_token_ceiling
        )
        reservation_ref = self._ledger.reserve(
            task_fingerprint=checked_task.fingerprint,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request.fingerprint,
            token_ceiling=ceiling,
        )
        started = self._read_clock()
        try:
            raw_response = self._backend.invoke_pure(spec=self._spec, request=request)
        except TimeoutError as exc:
            return self._record_failure(
                reservation_ref,
                checked_task,
                request,
                execution_fingerprint,
                started,
                ExecutionErrorClass.BACKEND_TIMEOUT,
                exc,
            )
        except Exception as exc:
            return self._record_failure(
                reservation_ref,
                checked_task,
                request,
                execution_fingerprint,
                started,
                ExecutionErrorClass.BACKEND_EXCEPTION,
                exc,
            )

        latency_ms = self._elapsed_ms(started)
        try:
            response = BackendResponse.model_validate(raw_response, strict=True)
        except Exception as exc:
            return self._record_failure_at_latency(
                reservation_ref,
                checked_task,
                request,
                execution_fingerprint,
                latency_ms,
                ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                exc,
            )
        identity = response.provider_identity_observation
        if identity is not None and (
            identity.requested_model != self._spec.model
            or identity.backend_fingerprint != self._spec.backend_fingerprint
        ):
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail="provider identity observation does not bind the frozen PURE request",
                cost_usd=response.cost_usd,
            )

        try:
            if self._require_backend_match() != backend_fingerprint:
                raise BackendFingerprintMismatchError(
                    "backend fingerprint changed during execution"
                )
        except Exception as exc:
            # The response has already crossed the provider boundary.  Preserve
            # its reported usage even when backend identity becomes unavailable
            # or drifts, so a simultaneous token overrun still poisons rather
            # than merely burns the reservation.
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
                detail=self._exception_detail(exc),
                cost_usd=response.cost_usd,
                provider_identity_observation=response.provider_identity_observation,
            )

        if (
            response.usage.total_tokens > ceiling
            or response.usage.output_tokens > self._spec.inference.max_output_tokens
        ):
            return self._finish_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.USAGE_EXCEEDED,
                detail="backend-reported usage exceeded a frozen token ceiling",
                cost_usd=response.cost_usd,
                provider_identity_observation=response.provider_identity_observation,
            )

        execution = self._make_execution(
            task=checked_task,
            request=request,
            output=response.output,
            status=ExecutionStatus.COMPLETED,
            usage=response.usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            error=None,
            cost_usd=response.cost_usd,
            provider_identity_observation=response.provider_identity_observation,
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
            raise AttemptAccountingError("successful PURE execution could not be settled") from exc
        return PureReferenceExecutionRecord(
            execution=execution,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _record_failure(
        self,
        reservation_ref: ArtifactRef,
        task: CandidateTask,
        request: PureReferenceRequest,
        execution_fingerprint: str,
        started: float | None,
        error_class: ExecutionErrorClass,
        exc: Exception,
    ) -> PureReferenceExecutionRecord:
        return self._record_failure_at_latency(
            reservation_ref,
            task,
            request,
            execution_fingerprint,
            self._elapsed_ms(started),
            error_class,
            exc,
        )

    def _record_failure_at_latency(
        self,
        reservation_ref: ArtifactRef,
        task: CandidateTask,
        request: PureReferenceRequest,
        execution_fingerprint: str,
        latency_ms: float,
        error_class: ExecutionErrorClass,
        exc: Exception,
    ) -> PureReferenceExecutionRecord:
        return self._finish_failure(
            reservation_ref=reservation_ref,
            task=task,
            request=request,
            execution_fingerprint=execution_fingerprint,
            usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
            latency_ms=latency_ms,
            error_class=error_class,
            detail=self._exception_detail(exc),
            cost_usd=None,
        )

    def _finish_failure(
        self,
        *,
        reservation_ref: ArtifactRef,
        task: CandidateTask,
        request: PureReferenceRequest,
        execution_fingerprint: str,
        usage: BackendTokenUsage,
        latency_ms: float,
        error_class: ExecutionErrorClass,
        detail: str,
        cost_usd: float | None,
        provider_identity_observation: ProviderIdentityObservation | None = None,
    ) -> PureReferenceExecutionRecord:
        execution = self._make_execution(
            task=task,
            request=request,
            output=None,
            status=ExecutionStatus.FAILED,
            usage=usage,
            latency_ms=latency_ms,
            execution_fingerprint=execution_fingerprint,
            error=ExecutionError(error_class=error_class, detail=detail),
            cost_usd=cost_usd,
            provider_identity_observation=provider_identity_observation,
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
            raise AttemptAccountingError("failed PURE execution could not be persisted") from exc
        outcome_ref = self._ledger.burn(
            reservation_ref,
            error_class=error_class.value,
            execution_ref=execution_ref,
        )
        return PureReferenceExecutionRecord(
            execution=execution,
            execution_ref=execution_ref,
            outcome_ref=outcome_ref,
        )

    def _make_execution(
        self,
        *,
        task: CandidateTask,
        request: PureReferenceRequest,
        output: str | None,
        status: ExecutionStatus,
        usage: BackendTokenUsage,
        latency_ms: float,
        execution_fingerprint: str,
        error: ExecutionError | None,
        cost_usd: float | None,
        provider_identity_observation: ProviderIdentityObservation | None,
    ) -> PureReferenceExecution:
        return PureReferenceExecution(
            task=task,
            request=request,
            output=output,
            status=status,
            usage=ModelUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            ),
            spec=self._spec,
            provider_identity_observation=provider_identity_observation,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request.fingerprint,
            error=error,
        )

    def _persist_execution(self, execution: PureReferenceExecution) -> ArtifactRef:
        raw_ref = self._ledger.repository.put_json(
            execution,
            media_type=PURE_REFERENCE_EXECUTION_MEDIA_TYPE,
        )
        ref = ArtifactRef.model_validate(raw_ref, strict=True)
        if ref.media_type != PURE_REFERENCE_EXECUTION_MEDIA_TYPE:
            raise AttemptAccountingError("repository returned the wrong PURE execution media type")
        loaded = self._ledger.repository.get_json(ref, PureReferenceExecution)
        if PureReferenceExecution.model_validate(loaded, strict=True) != execution:
            raise AttemptAccountingError("persisted PURE execution content changed")
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
        return f"{type(exc).__module__}.{type(exc).__qualname__}"[:1_024]


__all__ = [
    "PureModelBackend",
    "PureModelRunner",
    "PureReplayBackend",
    "pure_replay_key",
]
