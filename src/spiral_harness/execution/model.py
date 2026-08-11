"""Provider-neutral fixed-model execution with score-free results."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import (
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptAccountingError,
    AttemptLedger,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNPINNED_REVISIONS = frozenset(
    {
        "auto",
        "current",
        "default",
        "head",
        "latest",
        "main",
        "master",
    }
)


class _FrozenExecutionModel(BaseModel):
    """Strict immutable model that preserves prompt whitespace byte-for-byte."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


ExactNonEmptyText = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class InferenceConfig(_FrozenExecutionModel):
    """Complete provider-neutral inference settings for a fixed run."""

    schema_version: Literal["1"] = "1"
    temperature: Annotated[float, Field(ge=0, strict=True)]
    top_p: Annotated[float, Field(gt=0, le=1, strict=True)]
    max_output_tokens: Annotated[int, Field(ge=1, strict=True)]
    timeout_seconds: Annotated[float, Field(gt=0, strict=True)]
    stop_sequences: tuple[ExactNonEmptyText, ...] = ()

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_are_exact_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("stop_sequences must not contain duplicates")
        return values

    @property
    def fingerprint(self) -> str:
        """Canonical identity of every frozen inference setting."""

        return canonical_sha256(self)


class FrozenModelSpec(_FrozenExecutionModel):
    """Pinned provider/model/tokenizer/runtime contract.

    ``revision`` and ``tokenizer_revision`` are explicit immutable provider
    snapshots.  Hosted services may use a dated snapshot or provider revision;
    they are not required to invent a local weights hash.  Moving aliases such
    as ``latest`` and ``main`` are rejected.
    """

    schema_version: Literal["1"] = "1"
    backend: ExactNonEmptyText
    backend_fingerprint: ExactNonEmptyText
    model: ExactNonEmptyText
    revision: ExactNonEmptyText
    tokenizer: ExactNonEmptyText
    tokenizer_revision: ExactNonEmptyText
    runtime: ExactNonEmptyText
    inference: InferenceConfig

    @field_validator(
        "backend",
        "backend_fingerprint",
        "model",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "runtime",
    )
    @classmethod
    def identities_are_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("model identity fields must not have surrounding whitespace")
        return value

    @field_validator("revision", "tokenizer_revision")
    @classmethod
    def revisions_are_pinned(cls, value: str) -> str:
        normalized = value.casefold()
        if normalized in _UNPINNED_REVISIONS or normalized.rsplit("/", maxsplit=1)[-1] in {
            "main",
            "master",
            "latest",
        }:
            raise ValueError("revision must be an explicit immutable snapshot, not a moving alias")
        return value

    @property
    def model_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/model-identity/v1",
                "model": self.model,
                "revision": self.revision,
                "tokenizer": self.tokenizer,
                "tokenizer_revision": self.tokenizer_revision,
            }
        )

    @property
    def inference_fingerprint(self) -> str:
        return self.inference.fingerprint

    @property
    def runtime_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/runtime-identity/v1",
                "runtime": self.runtime,
            }
        )

    @property
    def fingerprint(self) -> str:
        """Canonical identity covering backend, model, tokenizer, runtime, and inference."""

        return canonical_sha256(self)


class CandidateTask(_FrozenExecutionModel):
    """The complete candidate-facing task view; no answer is representable."""

    schema_version: Literal["1"] = "1"
    task_id: ExactNonEmptyText
    question: ExactNonEmptyText

    @field_validator("task_id")
    @classmethod
    def task_id_is_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("task_id must not have surrounding whitespace")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @classmethod
    def from_task_view(cls, task: object) -> CandidateTask:
        """Copy a structural candidate view without importing its benchmark type.

        Only ``task_id`` and ``question`` cross the boundary.  Objects exposing
        obvious trusted-answer or grading attributes are rejected even though
        those values would not otherwise be copied.
        """

        if isinstance(task, cls):
            return cls.model_validate(task, strict=True)
        if isinstance(task, Mapping):
            return cls.model_validate(task, strict=True)
        forbidden = ("answer", "gold", "grade", "reference", "reference_answer", "score")
        exposed = tuple(name for name in forbidden if hasattr(task, name))
        if exposed:
            joined = ", ".join(exposed)
            raise ValueError(f"candidate task view exposes trusted fields: {joined}")
        try:
            task_id = task.task_id  # type: ignore[attr-defined]
            question = task.question  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError("task must expose task_id and question attributes") from exc
        return cls(task_id=task_id, question=question)


class PromptHarness(_FrozenExecutionModel):
    """One prompt arm whose declared hash must match its exact UTF-8 bytes."""

    schema_version: Literal["1"] = "1"
    harness_id: ExactNonEmptyText
    system_prompt: ExactNonEmptyText
    prompt_sha256: Sha256

    @field_validator("harness_id")
    @classmethod
    def harness_id_is_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("harness_id must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def declared_prompt_hash_matches_bytes(self) -> Self:
        actual = sha256_bytes(self.system_prompt.encode("utf-8"))
        if self.prompt_sha256 != actual:
            raise ValueError("prompt_sha256 does not match the exact system_prompt bytes")
        return self

    @classmethod
    def from_prompt(cls, *, harness_id: str, system_prompt: str) -> PromptHarness:
        """Construct a harness while making the exact prompt hash visible."""

        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        return cls(
            harness_id=harness_id,
            system_prompt=system_prompt,
            prompt_sha256=sha256_bytes(system_prompt.encode("utf-8")),
        )


class ModelRequest(_FrozenExecutionModel):
    """Exact structured request sent to a backend."""

    schema_version: Literal["1"] = "1"
    task_id: ExactNonEmptyText
    harness_id: ExactNonEmptyText
    system_prompt: ExactNonEmptyText
    user_prompt: ExactNonEmptyText
    seed: Annotated[int, Field(ge=0, strict=True)]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BackendTokenUsage(_FrozenExecutionModel):
    """Provider-reported token counts before trusted latency capture."""

    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelUsage(_FrozenExecutionModel):
    """Score-independent resources captured for one execution."""

    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    latency_ms: Annotated[float, Field(ge=0, strict=True)]
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens(self) -> int:
        return self.total_tokens

    @property
    def tool_calls(self) -> int:
        return 0


class BackendResponse(_FrozenExecutionModel):
    """Minimal score-free value returned by a provider adapter."""

    output: str
    usage: BackendTokenUsage
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None


class ExecutionStatus(StrEnum):
    """Whether an output is admissible for trusted grading."""

    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionErrorClass(StrEnum):
    """Stable infrastructure failure classes; none imply a task score."""

    BACKEND_TIMEOUT = "backend_timeout"
    BACKEND_EXCEPTION = "backend_exception"
    BACKEND_FINGERPRINT_MISMATCH = "backend_fingerprint_mismatch"
    INVALID_BACKEND_RESPONSE = "invalid_backend_response"
    USAGE_EXCEEDED = "usage_exceeded"


class ExecutionError(_FrozenExecutionModel):
    """Captured infrastructure classification without grading authority."""

    error_class: ExecutionErrorClass
    detail: ExactNonEmptyText


class ModelExecution(_FrozenExecutionModel):
    """Strict score-free execution artifact.

    The schema intentionally cannot represent a score, gold answer, reference
    answer, or grader verdict.  Those remain capabilities of a trusted
    benchmark adapter.
    """

    schema_version: Literal["1"] = "1"
    task: CandidateTask
    request: ModelRequest
    output: str | None
    status: ExecutionStatus
    usage: ModelUsage
    backend_fingerprint: ExactNonEmptyText
    model_fingerprint: Sha256
    inference_fingerprint: Sha256
    runtime_fingerprint: Sha256
    spec_fingerprint: Sha256
    execution_fingerprint: Sha256
    request_sha256: Sha256
    error: ExecutionError | None

    @model_validator(mode="after")
    def provenance_and_status_are_consistent(self) -> Self:
        if self.request.task_id != self.task.task_id:
            raise ValueError("request task_id does not match task")
        if self.request.user_prompt != self.task.question:
            raise ValueError("request user_prompt does not match candidate-facing question")
        if self.request_sha256 != self.request.fingerprint:
            raise ValueError("request_sha256 does not match the exact request")
        if self.status is ExecutionStatus.COMPLETED:
            if self.output is None:
                raise ValueError("completed execution requires an output")
            if self.error is not None:
                raise ValueError("completed execution must not contain an error")
        else:
            if self.output is not None:
                raise ValueError("failed execution must not expose an output")
            if self.error is None:
                raise ValueError("failed execution requires an error classification")
        return self

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def seed(self) -> int:
        return self.request.seed

    @property
    def harness_id(self) -> str:
        return self.request.harness_id

    @property
    def output_text(self) -> str | None:
        return self.output

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def latency_ms(self) -> float:
        return self.usage.latency_ms

    @property
    def tool_calls(self) -> int:
        return 0

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd


@runtime_checkable
class ModelBackend(Protocol):
    """Provider adapter boundary; implementations have no grading authority."""

    @property
    def fingerprint(self) -> str: ...

    def invoke(self, *, spec: FrozenModelSpec, request: ModelRequest) -> BackendResponse: ...


class BackendFingerprintMismatchError(ValueError):
    """Raised before reservation when a backend differs from the frozen spec."""


class ReplayMissError(LookupError):
    """Raised when a deterministic replay has no exact request record."""


class ReplayBackend:
    """Deterministic provider-free backend for tests and recorded experiments.

    Responses are selected by a canonical key over the complete frozen model
    spec and request, never by call order.  An optional immutable default
    response is useful for fixtures that do not care about prompt-specific
    output.  No provider SDK or network is used.
    """

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

    def invoke(self, *, spec: FrozenModelSpec, request: ModelRequest) -> BackendResponse:
        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = ModelRequest.model_validate(request, strict=True)
        if checked_spec.backend_fingerprint != self._fingerprint:
            raise BackendFingerprintMismatchError(
                "replay backend fingerprint does not match the frozen model spec"
            )
        key = replay_key(checked_spec, checked_request)
        self._calls.append(key)
        response = self._responses.get(key, self._default_response)
        if response is None:
            raise ReplayMissError(f"no replay response for frozen spec/request {key}")
        return BackendResponse.model_validate(response, strict=True)


def replay_key(spec: FrozenModelSpec, request: ModelRequest) -> str:
    """Bind a replay record to both the exact model spec and exact request."""

    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_request = ModelRequest.model_validate(request, strict=True)
    return canonical_sha256(
        {
            "schema": "spiral-harness/model-replay-key/v1",
            "spec_fingerprint": checked_spec.fingerprint,
            "request_sha256": checked_request.fingerprint,
        }
    )


def materialize_request(
    task: object,
    harness: PromptHarness,
    *,
    seed: int,
) -> ModelRequest:
    """Build the exact structured backend request from candidate-visible inputs."""

    checked_task = CandidateTask.from_task_view(task)
    checked_harness = PromptHarness.model_validate(harness, strict=True)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must not be negative")
    return ModelRequest(
        task_id=checked_task.task_id,
        harness_id=checked_harness.harness_id,
        system_prompt=checked_harness.system_prompt,
        user_prompt=checked_task.question,
        seed=seed,
    )


def paired_execution_fingerprint(
    spec: FrozenModelSpec,
    task: object,
    *,
    seed: int,
    backend_fingerprint: str,
) -> str:
    """Hash paired context while deliberately excluding harness arm and prompt."""

    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_task = CandidateTask.from_task_view(task)
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

    _IMPLEMENTATION_ID = "spiral-harness/fixed-model-runner/v1"

    def __init__(
        self,
        *,
        spec: FrozenModelSpec,
        backend: ModelBackend,
        attempt_ledger: AttemptLedger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._spec = FrozenModelSpec.model_validate(spec, strict=True)
        if not isinstance(backend, ModelBackend):
            raise TypeError("backend must implement ModelBackend")
        if not isinstance(attempt_ledger, AttemptLedger):
            raise TypeError("attempt_ledger must be an AttemptLedger")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._backend = backend
        self._attempt_ledger = attempt_ledger
        self._clock = clock
        self._last_execution_ref: ArtifactRef | None = None
        self._last_outcome_ref: ArtifactRef | None = None
        self._require_backend_match()

    @property
    def spec(self) -> FrozenModelSpec:
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

        return self._last_execution_ref

    @property
    def last_outcome_ref(self) -> ArtifactRef | None:
        """Most recent immutable attempt outcome ref."""

        return self._last_outcome_ref

    def execute(
        self,
        task: object,
        *,
        harness: PromptHarness,
        seed: int,
    ) -> ModelExecution:
        """Reserve, invoke once, then settle or conservatively burn the attempt."""

        checked_task = CandidateTask.from_task_view(task)
        checked_harness = PromptHarness.model_validate(harness, strict=True)
        request = materialize_request(checked_task, checked_harness, seed=seed)
        backend_fingerprint = self._require_backend_match()
        execution_fingerprint = paired_execution_fingerprint(
            self._spec,
            checked_task,
            seed=seed,
            backend_fingerprint=backend_fingerprint,
        )
        request_sha256 = request.fingerprint
        reservation_ref = self._attempt_ledger.reserve(
            task_fingerprint=checked_task.fingerprint,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            token_ceiling=self._attempt_ledger.budget.max_tokens_per_attempt,
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
                backend_fingerprint=backend_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=self._elapsed_ms(started),
                error_class=ExecutionErrorClass.BACKEND_TIMEOUT,
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
                backend_fingerprint=backend_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=self._elapsed_ms(started),
                error_class=ExecutionErrorClass.BACKEND_EXCEPTION,
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
                backend_fingerprint=backend_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
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
                backend_fingerprint=backend_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.BACKEND_FINGERPRINT_MISMATCH,
                detail="backend fingerprint changed during execution",
                cost_usd=None,
            )
        try:
            response = BackendResponse.model_validate(raw_response, strict=True)
        except Exception as exc:
            return self._record_failure(
                reservation_ref=reservation_ref,
                task=checked_task,
                request=request,
                execution_fingerprint=execution_fingerprint,
                request_sha256=request_sha256,
                backend_fingerprint=backend_fingerprint,
                usage=BackendTokenUsage(input_tokens=0, output_tokens=0),
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.INVALID_BACKEND_RESPONSE,
                detail=self._exception_detail(exc),
                cost_usd=None,
            )

        reserved_tokens = self._attempt_ledger.budget.max_tokens_per_attempt
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
                backend_fingerprint=backend_fingerprint,
                usage=response.usage,
                latency_ms=latency_ms,
                error_class=ExecutionErrorClass.USAGE_EXCEEDED,
                detail="backend-reported usage exceeded a frozen token ceiling",
                cost_usd=response.cost_usd,
            )

        execution = self._make_execution(
            task=checked_task,
            request=request,
            output=response.output,
            status=ExecutionStatus.COMPLETED,
            usage=response.usage,
            latency_ms=latency_ms,
            backend_fingerprint=backend_fingerprint,
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
        return execution

    def _record_failure(
        self,
        *,
        reservation_ref: ArtifactRef,
        task: CandidateTask,
        request: ModelRequest,
        execution_fingerprint: str,
        request_sha256: str,
        backend_fingerprint: str,
        usage: BackendTokenUsage,
        latency_ms: float,
        error_class: ExecutionErrorClass,
        detail: str,
        cost_usd: float | None,
    ) -> ModelExecution:
        execution = self._make_execution(
            task=task,
            request=request,
            output=None,
            status=ExecutionStatus.FAILED,
            usage=usage,
            latency_ms=latency_ms,
            backend_fingerprint=backend_fingerprint,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            error=ExecutionError(error_class=error_class, detail=detail),
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
        return execution

    def _make_execution(
        self,
        *,
        task: CandidateTask,
        request: ModelRequest,
        output: str | None,
        status: ExecutionStatus,
        usage: BackendTokenUsage,
        latency_ms: float,
        backend_fingerprint: str,
        execution_fingerprint: str,
        request_sha256: str,
        error: ExecutionError | None,
        cost_usd: float | None,
    ) -> ModelExecution:
        return ModelExecution(
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
            backend_fingerprint=backend_fingerprint,
            model_fingerprint=self._spec.model_fingerprint,
            inference_fingerprint=self._spec.inference_fingerprint,
            runtime_fingerprint=self._spec.runtime_fingerprint,
            spec_fingerprint=self._spec.fingerprint,
            execution_fingerprint=execution_fingerprint,
            request_sha256=request_sha256,
            error=error,
        )

    def _persist_execution(self, execution: ModelExecution) -> ArtifactRef:
        raw_ref = self._attempt_ledger.repository.put_json(
            execution,
            media_type=MODEL_EXECUTION_MEDIA_TYPE,
        )
        ref = ArtifactRef.model_validate(raw_ref, strict=True)
        if ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise AttemptAccountingError("repository returned the wrong execution media type")
        loaded = self._attempt_ledger.repository.get_json(ref, ModelExecution)
        checked = ModelExecution.model_validate(loaded, strict=True)
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
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
    "MODEL_EXECUTION_MEDIA_TYPE",
    "BackendFingerprintMismatchError",
    "BackendResponse",
    "BackendTokenUsage",
    "CandidateTask",
    "ExecutionError",
    "ExecutionErrorClass",
    "ExecutionStatus",
    "FixedModelRunner",
    "FrozenModelSpec",
    "InferenceConfig",
    "ModelBackend",
    "ModelExecution",
    "ModelRequest",
    "ModelUsage",
    "PromptHarness",
    "ReplayBackend",
    "ReplayMissError",
    "materialize_request",
    "paired_execution_fingerprint",
    "replay_key",
]
