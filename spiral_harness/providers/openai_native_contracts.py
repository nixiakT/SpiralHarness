"""Immutable contracts and serializer for native function calling.

Official and OpenAI wire names are both explicit. A pinned benchmark adapter
must supply any mapping; this provider layer never invents one.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json, canonical_sha256, module_source_sha256
from spiral_harness.core.models import Sha256
from spiral_harness.execution.contracts import (
    BackendTokenUsage,
    InferenceConfig,
    ProviderIdentityObservation,
)

_SERIALIZER_ID = "spiral-harness/openai-native-function-serializer/v2"
_MAX_JSON_NESTING = 64
_WIRE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SERIALIZER_POLICY = {
    "protocol": "normalized-openai-compatible-litellm/v1",
    "n": 1,
    "parallel_tool_calls": True,
    "stream": False,
    "tool_choice": "auto",
}


def _reject_surrogate_text(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("Unicode surrogate code points are forbidden")
    return value


def _reject_surrogates(value: Any) -> None:
    """Reject surrogates throughout local model input without encoding it first."""

    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            _reject_surrogate_text(item)
        elif isinstance(item, BaseModel):
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(getattr(item, name) for name in type(item).model_fields)
        elif isinstance(item, Mapping):
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(item.keys())
                pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)) and id(item) not in seen:
            seen.add(id(item))
            pending.extend(item)


class _FrozenNativeModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _all_strings_are_unicode_scalars(cls, value: Any) -> Any:
        _reject_surrogates(value)
        return value


ExactNonEmptyText = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object keys are forbidden")
        result[key] = value
    return result


def _validate_json_tree(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            _reject_surrogate_text(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON numbers are forbidden")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _check_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise ValueError("JSON nesting exceeds the frozen parser limit")
        elif character in "]}":
            depth -= 1


def _strict_json(text: str, *, label: str) -> Any:
    _check_json_nesting(text)
    failed = False
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        _validate_json_tree(parsed)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        failed = True
    if failed:
        raise ValueError(f"{label} must be unambiguous strict JSON") from None
    return parsed


def _exact_identifier(value: str, *, label: str) -> str:
    _reject_surrogate_text(value)
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must be exact and contain no control characters")
    return value


def _wire_identifier(value: str) -> str:
    if _WIRE_NAME.fullmatch(value) is None:
        raise ValueError(
            "wire function name must match [A-Za-z0-9_-]{1,64}; dotted official names "
            "require an explicit pinned adapter mapping"
        )
    return value


class NativeAssistantToolCall(_FrozenNativeModel):
    """One call binding provider wire identity back to official identity."""

    schema_version: Literal["2"] = "2"
    call_id: ExactNonEmptyText
    type: Literal["function"] = "function"
    official_name: ExactNonEmptyText
    wire_name: ExactNonEmptyText
    arguments_json: str

    @field_validator("call_id", "official_name")
    @classmethod
    def _identifiers_are_exact(cls, value: str) -> str:
        return _exact_identifier(value, label="tool-call identifier")

    @field_validator("wire_name")
    @classmethod
    def _wire_name_is_openai_compatible(cls, value: str) -> str:
        return _wire_identifier(value)

    @field_validator("arguments_json")
    @classmethod
    def _arguments_are_an_object(cls, value: str) -> str:
        arguments = _strict_json(value, label="tool-call arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool-call arguments must decode to a JSON object")
        return value

    @property
    def function_name(self) -> str:
        """Official name consumed by downstream score-free serializers."""

        return self.official_name

    @property
    def canonical_arguments_json(self) -> str:
        return canonical_json(_strict_json(self.arguments_json, label="tool-call arguments"))


class FrozenNativeChatMessage(_FrozenNativeModel):
    schema_version: Literal["2"] = "2"
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    name: ExactNonEmptyText | None = None
    tool_call_id: ExactNonEmptyText | None = None
    tool_calls: tuple[NativeAssistantToolCall, ...] = ()

    @field_validator("name", "tool_call_id")
    @classmethod
    def _optional_identifiers_are_exact(cls, value: str | None) -> str | None:
        return None if value is None else _exact_identifier(value, label="message identifier")

    @model_validator(mode="after")
    def _role_shape_is_unambiguous(self) -> Self:
        if self.role in {"system", "user"}:
            if self.content is None or self.tool_call_id is not None or self.tool_calls:
                raise ValueError("system/user messages require text and no tool-call fields")
        elif self.role == "assistant":
            if self.tool_call_id is not None or (self.content is None and not self.tool_calls):
                raise ValueError("assistant messages require text or calls and cannot be results")
        elif self.content is None or self.tool_call_id is None or self.tool_calls:
            raise ValueError("tool messages require text and one ID; they cannot initiate calls")
        ids = tuple(call.call_id for call in self.tool_calls)
        if len(ids) != len(set(ids)):
            raise ValueError("assistant message contains duplicate tool-call IDs")
        return self


class FrozenNativeFunctionTool(_FrozenNativeModel):
    """Pinned-adapter-normalized official schema plus an explicit wire name.

    ``function_schema_json`` is the exact JSON handed off by a pinned benchmark
    adapter and retains the official function name. It is not untouched BFCL
    source JSON: BFCL-specific types must be normalized before construction.
    Serialization changes only ``name`` to ``wire_name``.
    """

    schema_version: Literal["2"] = "2"
    schema_input_contract: Literal["pinned-adapter-normalized-official-schema/v1"] = (
        "pinned-adapter-normalized-official-schema/v1"
    )
    adapter_normalization_runtime_attested: Literal[False] = False
    official_name: ExactNonEmptyText
    wire_name: ExactNonEmptyText
    function_schema_json: ExactNonEmptyText

    @field_validator("official_name")
    @classmethod
    def _official_name_is_exact(cls, value: str) -> str:
        return _exact_identifier(value, label="official function name")

    @field_validator("wire_name")
    @classmethod
    def _wire_name_is_valid(cls, value: str) -> str:
        return _wire_identifier(value)

    @model_validator(mode="after")
    def _schema_is_bound(self) -> Self:
        schema = _strict_json(self.function_schema_json, label="function schema")
        if not isinstance(schema, dict) or canonical_json(schema) != self.function_schema_json:
            raise ValueError("function schema must be one canonical JSON object")
        if schema.get("name") != self.official_name:
            raise ValueError("official function name differs from the frozen schema")
        if not isinstance(schema.get("parameters"), dict):
            raise ValueError("function schema requires object-valued parameters")
        return self

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, Any],
        *,
        wire_name: str | None = None,
    ) -> FrozenNativeFunctionTool:
        """Freeze an already benchmark-adapter-normalized official schema."""

        if not isinstance(schema, Mapping):
            raise TypeError("function schema must be a mapping")
        name = schema.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("function schema requires a non-empty string name")
        return cls(
            official_name=name,
            wire_name=name if wire_name is None else wire_name,
            function_schema_json=canonical_json(schema),
        )

    @property
    def wire_schema(self) -> dict[str, Any]:
        """Return provider schema after changing only official name to wire name."""

        schema = dict(_strict_json(self.function_schema_json, label="function schema"))
        schema["name"] = self.wire_name
        return schema


class NativeFunctionCallRequest(_FrozenNativeModel):
    """Exact model input; this provider does not decide either tool bundle."""

    schema_version: Literal["2"] = "2"
    backend_fingerprint: Sha256
    serializer_fingerprint: Sha256
    parser_fingerprint: Sha256
    transport_fingerprint: Sha256
    requested_model: ExactNonEmptyText
    messages: Annotated[tuple[FrozenNativeChatMessage, ...], Field(min_length=1)]
    task_required_tools: Annotated[tuple[FrozenNativeFunctionTool, ...], Field(min_length=1)]
    harness_added_tools: tuple[FrozenNativeFunctionTool, ...] = ()
    seed: NonNegativeInt
    inference: InferenceConfig

    @field_validator("requested_model")
    @classmethod
    def _model_is_exact(cls, value: str) -> str:
        return _exact_identifier(value, label="requested model")

    @model_validator(mode="after")
    def _tool_roster_and_transcript_are_closed(self) -> Self:
        tools = self.task_required_tools + self.harness_added_tools
        official = tuple(tool.official_name for tool in tools)
        wires = tuple(tool.wire_name for tool in tools)
        if len(official) != len(set(official)) or len(wires) != len(set(wires)):
            raise ValueError("official and wire tool names must each be globally unique")
        by_wire = {tool.wire_name: tool for tool in tools}
        pending: dict[str, FrozenNativeFunctionTool] = {}
        seen: set[str] = set()
        resolved: set[str] = set()
        for message in self.messages:
            if pending and message.role != "tool":
                raise ValueError("unresolved tool calls must be followed by their tool results")
            if message.role == "tool":
                call_id = message.tool_call_id
                if call_id in resolved:
                    raise ValueError("transcript contains a duplicate tool result")
                if call_id not in pending:
                    raise ValueError("transcript contains an orphan tool result")
                expected = pending.pop(call_id)
                if message.name is not None and message.name != expected.wire_name:
                    raise ValueError("tool-result name differs from its pending wire name")
                resolved.add(call_id)
            elif message.role == "assistant":
                for call in message.tool_calls:
                    tool = by_wire.get(call.wire_name)
                    if tool is None or tool.official_name != call.official_name:
                        raise ValueError("message history invokes outside the frozen tool mapping")
                    if call.call_id in seen:
                        raise ValueError("message history contains a duplicate tool-call ID")
                    seen.add(call.call_id)
                    pending[call.call_id] = tool
        if pending:
            raise ValueError("transcript ends with unresolved tool calls")
        return self

    @property
    def historical_call_ids(self) -> frozenset[str]:
        return frozenset(call.call_id for message in self.messages for call in message.tool_calls)

    @property
    def tools_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/native-function-tools/v2",
                "task_required_tools": self.task_required_tools,
                "harness_added_tools": self.harness_added_tools,
            }
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class NativeFunctionCallResponse(_FrozenNativeModel):
    """Score-free output with source binding but no runtime attestation."""

    schema_version: Literal["2"] = "2"
    protocol_scope: Literal["normalized-openai-compatible-litellm/v1"] = (
        "normalized-openai-compatible-litellm/v1"
    )
    upstream_handler_equivalence_attested: Literal[False] = False
    source_bound: Literal[True] = True
    runtime_execution_attested: Literal[False] = False
    request_fingerprint: Sha256
    serializer_fingerprint: Sha256
    parser_fingerprint: Sha256
    transport_fingerprint: Sha256
    tools_fingerprint: Sha256
    tool_calls: tuple[NativeAssistantToolCall, ...]
    assistant_text: str | None
    finish_reason: Literal["stop", "tool_calls"]
    usage: BackendTokenUsage
    provider_identity_observation: ProviderIdentityObservation | None = None

    @model_validator(mode="after")
    def _shape_matches_finish_reason(self) -> Self:
        if self.tool_calls and self.finish_reason != "tool_calls":
            raise ValueError("native calls require finish_reason=tool_calls")
        if not self.tool_calls and (self.finish_reason != "stop" or self.assistant_text is None):
            raise ValueError("no-call completion requires text and finish_reason=stop")
        return self

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _tool_call_payload(call: NativeAssistantToolCall) -> dict[str, Any]:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.wire_name, "arguments": call.arguments_json},
    }


def _message_payload(message: FrozenNativeChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_payload(call) for call in message.tool_calls]
    return payload


def _serialize_native_request(request: NativeFunctionCallRequest) -> dict[str, Any]:
    inference = request.inference
    payload: dict[str, Any] = {
        "model": request.requested_model,
        "messages": [_message_payload(message) for message in request.messages],
        "tools": [
            {"type": "function", "function": tool.wire_schema}
            for tool in request.task_required_tools + request.harness_added_tools
        ],
        "tool_choice": "auto",
        "n": 1,
        "parallel_tool_calls": True,
        "stream": False,
        "temperature": inference.temperature,
        "top_p": inference.top_p,
        "max_tokens": inference.max_output_tokens,
        "seed": request.seed,
    }
    if inference.stop_sequences:
        payload["stop"] = list(inference.stop_sequences)
    return payload


def _contracts_source_fingerprint() -> str:
    return module_source_sha256(sys.modules[__name__])


def native_function_serializer_fingerprint() -> str:
    """Bind serializer policy and source bytes, not runtime execution."""

    return canonical_sha256(
        {
            "id": _SERIALIZER_ID,
            "policy": _SERIALIZER_POLICY,
            "source": _contracts_source_fingerprint(),
        }
    )


__all__ = [
    "FrozenNativeChatMessage",
    "FrozenNativeFunctionTool",
    "NativeAssistantToolCall",
    "NativeFunctionCallRequest",
    "NativeFunctionCallResponse",
    "native_function_serializer_fingerprint",
]
