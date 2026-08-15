"""Immutable artifacts for one accounted native function-call attempt."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    ExecutionError,
    ExecutionStatus,
    FrozenModelSpec,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.protocol import ArtifactRepository

NATIVE_FUNCTION_REQUEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.native-function-request.v1+json"
)
NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.native-function-response.v1+json"
)
NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE = "application/vnd.spiral-harness.native-execution.v1+json"

_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _FrozenNativeExecutionModel(BaseModel):
    """Strict immutable base for score-free native execution evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class NativeFunctionSlotBinding(_FrozenNativeExecutionModel):
    """Opaque public coordinates for exactly one planned provider call slot."""

    schema_version: Literal["1"] = "1"
    task_fingerprint: _Sha256
    slot_fingerprint: _Sha256

    @property
    def fingerprint(self) -> str:
        """Expose the task identity expected by the shared attempt ledger."""

        return self.task_fingerprint

    @property
    def binding_fingerprint(self) -> str:
        return canonical_sha256(self)


class NativeFunctionUsage(_FrozenNativeExecutionModel):
    """Provider-reported score-independent resources for one native call."""

    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    tool_calls: Annotated[int, Field(ge=0, strict=True)]
    latency_ms: Annotated[float, Field(ge=0, strict=True)]
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens(self) -> int:
        return self.total_tokens


def native_function_execution_fingerprint(
    *,
    spec: FrozenModelSpec,
    task: NativeFunctionSlotBinding,
    request: NativeFunctionCallRequest,
) -> str:
    """Bind a slot to the exact spec, request, and native implementation identities."""

    checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    checked_task = NativeFunctionSlotBinding.model_validate(task, strict=True)
    checked_request = NativeFunctionCallRequest.model_validate(request, strict=True)
    return canonical_sha256(
        {
            "schema": "spiral-harness/native-function-execution-identity/v1",
            "spec_fingerprint": checked_spec.fingerprint,
            "task_fingerprint": checked_task.task_fingerprint,
            "slot_fingerprint": checked_task.slot_fingerprint,
            "request_sha256": checked_request.fingerprint,
            "backend_fingerprint": checked_request.backend_fingerprint,
            "serializer_fingerprint": checked_request.serializer_fingerprint,
            "parser_fingerprint": checked_request.parser_fingerprint,
            "transport_fingerprint": checked_request.transport_fingerprint,
        }
    )


def _ref_matches_model(
    ref: ArtifactRef,
    value: BaseModel,
    *,
    media_type: str,
    label: str,
) -> None:
    if ref.media_type != media_type:
        raise ValueError(f"{label}_ref declares the wrong media type")
    payload = canonical_json_bytes(value)
    if ref.sha256 != canonical_sha256(value) or ref.size != len(payload):
        raise ValueError(f"{label}_ref does not bind the exact canonical artifact")


def load_canonical_native_artifact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
) -> ModelT:
    """Load strict native models while preserving JSON-array tuple semantics.

    The provider contracts validate local Python inputs strictly and use a
    before-validator for Unicode scalar checks. Pydantic consequently treats
    nested JSON arrays as Python lists. This loader permits JSON
    representation conversion only, then requires an exact canonical-byte
    round trip so all other coercions fail closed.
    """

    if not isinstance(repository, ArtifactRepository):
        raise TypeError("repository must implement ArtifactRepository")
    checked_ref = ArtifactRef.model_validate(ref, strict=True)
    payload = repository.get_bytes(checked_ref)
    raw = repository.get_json(checked_ref)
    checked = model_type.model_validate(raw, strict=False)
    if canonical_json_bytes(checked) != payload:
        raise ValueError("native artifact is not canonical for its declared model")
    return checked


class NativeFunctionExecution(_FrozenNativeExecutionModel):
    """Score-free native result; source binding is not execution attestation."""

    schema_version: Literal["1"] = "1"
    provider_identity_attested: Literal[False] = False
    runtime_execution_attested: Literal[False] = False
    atomicity_attested: Literal[False] = False
    task: NativeFunctionSlotBinding
    spec: FrozenModelSpec
    request: NativeFunctionCallRequest
    request_ref: ArtifactRef
    response: NativeFunctionCallResponse | None
    response_ref: ArtifactRef | None
    status: ExecutionStatus
    usage: NativeFunctionUsage
    execution_fingerprint: _Sha256
    request_sha256: _Sha256
    error: ExecutionError | None

    @model_validator(mode="after")
    def declared_provenance_is_exact(self) -> Self:
        if self.request.requested_model != self.spec.model:
            raise ValueError("native request model differs from the frozen model spec")
        if self.request.inference != self.spec.inference:
            raise ValueError("native request inference differs from the frozen model spec")
        if self.request.backend_fingerprint != self.spec.backend_fingerprint:
            raise ValueError("native request backend differs from the frozen model spec")
        if self.request_sha256 != self.request.fingerprint:
            raise ValueError("request_sha256 does not match the exact native request")
        _ref_matches_model(
            self.request_ref,
            self.request,
            media_type=NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
            label="request",
        )
        expected_execution = native_function_execution_fingerprint(
            spec=self.spec,
            task=self.task,
            request=self.request,
        )
        if self.execution_fingerprint != expected_execution:
            raise ValueError("execution_fingerprint does not bind the exact native slot")

        if (self.response is None) != (self.response_ref is None):
            raise ValueError("native response and response_ref must be present together")
        if self.response is not None:
            assert self.response_ref is not None
            _ref_matches_model(
                self.response_ref,
                self.response,
                media_type=NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE,
                label="response",
            )
            bindings = (
                (self.response.request_fingerprint, self.request.fingerprint),
                (self.response.serializer_fingerprint, self.request.serializer_fingerprint),
                (self.response.parser_fingerprint, self.request.parser_fingerprint),
                (self.response.transport_fingerprint, self.request.transport_fingerprint),
                (self.response.tools_fingerprint, self.request.tools_fingerprint),
            )
            if any(actual != expected for actual, expected in bindings):
                raise ValueError("native response differs from its exact request binding")
            if (
                self.usage.input_tokens != self.response.usage.input_tokens
                or self.usage.output_tokens != self.response.usage.output_tokens
                or self.usage.tool_calls != self.response.tool_call_count
            ):
                raise ValueError("native execution usage differs from its response")
            identity = self.response.provider_identity_observation
            if identity is not None and (
                identity.requested_model != self.spec.model
                or identity.backend_fingerprint != self.spec.backend_fingerprint
            ):
                raise ValueError("provider identity observation differs from the frozen request")
        elif self.usage.tool_calls != 0:
            raise ValueError("an unverified native response cannot report accepted tool calls")

        if self.status is ExecutionStatus.COMPLETED:
            if self.response is None or self.error is not None:
                raise ValueError("completed native execution requires only a verified response")
        elif self.error is None:
            raise ValueError("failed native execution requires a sanitized typed error")
        return self

    @property
    def task_fingerprint(self) -> str:
        return self.task.task_fingerprint

    @property
    def slot_fingerprint(self) -> str:
        return self.task.slot_fingerprint

    @property
    def seed(self) -> int:
        return self.request.seed

    @property
    def output_text(self) -> str | None:
        return None if self.response is None else self.response.assistant_text

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def latency_ms(self) -> float:
        return self.usage.latency_ms

    @property
    def tool_calls(self) -> int:
        return self.usage.tool_calls

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd


class NativeFunctionExecutionRecord(_FrozenNativeExecutionModel):
    """References published in sequence, without a cross-store atomicity claim."""

    schema_version: Literal["1"] = "1"
    atomicity_attested: Literal[False] = False
    execution: NativeFunctionExecution
    request_ref: ArtifactRef
    response_ref: ArtifactRef | None
    reservation_ref: ArtifactRef
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef

    @model_validator(mode="after")
    def references_bind_the_execution(self) -> Self:
        expected_types = (
            (self.request_ref, NATIVE_FUNCTION_REQUEST_MEDIA_TYPE, "request_ref"),
            (self.reservation_ref, ATTEMPT_RESERVATION_MEDIA_TYPE, "reservation_ref"),
            (self.execution_ref, NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE, "execution_ref"),
            (self.outcome_ref, ATTEMPT_OUTCOME_MEDIA_TYPE, "outcome_ref"),
        )
        for ref, media_type, label in expected_types:
            if ref.media_type != media_type:
                raise ValueError(f"{label} declares the wrong media type")
        if self.response_ref is not None and (
            self.response_ref.media_type != NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE
        ):
            raise ValueError("response_ref declares the wrong media type")
        _ref_matches_model(
            self.execution_ref,
            self.execution,
            media_type=NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE,
            label="execution",
        )
        if self.request_ref != self.execution.request_ref:
            raise ValueError("record request_ref differs from its execution")
        if self.response_ref != self.execution.response_ref:
            raise ValueError("record response_ref differs from its execution")
        return self


__all__ = [
    "NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE",
    "NATIVE_FUNCTION_REQUEST_MEDIA_TYPE",
    "NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE",
    "NativeFunctionExecution",
    "NativeFunctionExecutionRecord",
    "NativeFunctionSlotBinding",
    "NativeFunctionUsage",
    "load_canonical_native_artifact",
    "native_function_execution_fingerprint",
]
