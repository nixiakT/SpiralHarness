"""Direct-only, score-free native function calls via normalized LiteLLM JSON."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import ValidationError

from spiral_harness.core.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    module_source_sha256,
)
from spiral_harness.execution.backend_errors import BackendResponseRejectedError
from spiral_harness.execution.contracts import BackendTokenUsage, ProviderIdentityObservation
from spiral_harness.providers.openai_compatible import (
    OpenAICompatibleBackendError,
    normalize_openai_base_url,
)
from spiral_harness.providers.openai_native_contracts import (
    _MAX_JSON_NESTING,
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
    _contracts_source_fingerprint,
    _reject_surrogate_text,
    _serialize_native_request,
    _strict_json,
    native_function_serializer_fingerprint,
)

_PARSER_ID = "spiral-harness/openai-native-function-parser/v2"
_TRANSPORT_ID = "spiral-harness/openai-native-function-transport/v2"
_BACKEND_ID = "spiral-harness/openai-native-function-backend/v2"
_MAX_RESPONSE_BYTES = 1_048_576
_PARSER_POLICY = {
    "strict_utf8": True,
    "duplicate_json_keys": "reject",
    "non_finite_json": "reject",
    "max_json_nesting": _MAX_JSON_NESTING,
    "tool_calls_null_policy": "reject/v1",
    "explicit_assistant_role": True,
}
_TRANSPORT_POLICY = {
    "proxy_policy": "direct-only/ProxyHandler-empty/v1",
    "redirect_policy": "deny-all/v1",
    "max_response_bytes": _MAX_RESPONSE_BYTES,
    "method": "POST",
    "endpoint": "/chat/completions",
}


class OpenAICompatibleNativeFunctionError(OpenAICompatibleBackendError):
    """Sanitized native function-calling failure."""


class OpenAICompatibleNativeInvalidResponseError(
    BackendResponseRejectedError,
    OpenAICompatibleNativeFunctionError,
):
    """Malformed response retaining only independently unambiguous usage."""


class _ProviderParseError(RuntimeError):
    """Internal static-detail parse error, never attached to a public error."""


def _safe_base_url(value: str) -> str:
    if isinstance(value, str):
        _reject_surrogate_text(value)
        if any(ord(character) < 32 for character in value):
            raise ValueError("base_url must not contain control characters")
    normalized = normalize_openai_base_url(value)
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) origin/path without credentials or a query")
    return normalized


def _module_source_fingerprint() -> str:
    return module_source_sha256(sys.modules[__name__])


def native_function_parser_fingerprint() -> str:
    """Bind parser policy and source bytes, without claiming runtime execution."""

    return canonical_sha256(
        {
            "id": _PARSER_ID,
            "policy": _PARSER_POLICY,
            "parser_source": _module_source_fingerprint(),
            "contracts_source": _contracts_source_fingerprint(),
        }
    )


def native_function_transport_fingerprint(*, base_url: str, user_agent: str) -> str:
    """Bind endpoint, user agent, policy, and source; this is not attestation."""

    _reject_surrogate_text(base_url)
    _reject_surrogate_text(user_agent)
    return canonical_sha256(
        {
            "id": _TRANSPORT_ID,
            "base_url": base_url,
            "user_agent": user_agent,
            "policy": _TRANSPORT_POLICY,
            "source": _module_source_fingerprint(),
        }
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


def _direct_only_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )


def _post_raw_http(
    *,
    url: str,
    api_key: str,
    user_agent: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=canonical_json_bytes(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    detail: str | None = None
    try:
        with _direct_only_opener().open(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = f"OpenAI-compatible backend HTTP {exc.code}" if type(exc.code) is int else None
    except Exception:
        detail = None
    if detail is not None:
        raise OpenAICompatibleNativeFunctionError(detail) from None
    if "raw" not in locals():
        raise OpenAICompatibleNativeFunctionError(
            "OpenAI-compatible native transport failed"
        ) from None
    return raw


def _decode_raw_response(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > _MAX_RESPONSE_BYTES:
        raise _ProviderParseError("response body violates the frozen byte limit")
    failed = False
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        failed = True
    if failed:
        raise _ProviderParseError("response body is not strict UTF-8") from None
    failed = False
    try:
        decoded = _strict_json(text, label="response body")
    except ValueError:
        failed = True
    if failed:
        raise _ProviderParseError("response body is not unambiguous strict JSON") from None
    if type(decoded) is not dict:
        raise _ProviderParseError("response body must be one JSON object")
    return decoded


def _usage_alias(usage: dict[str, Any], primary: str, alias: str) -> int:
    present = [name for name in (primary, alias) if name in usage]
    if not present:
        raise _ProviderParseError("chat completion usage is incomplete")
    values = [usage[name] for name in present]
    if any(type(value) is not int or value < 0 for value in values):
        raise _ProviderParseError("chat completion token usage is invalid")
    if len(values) == 2 and values[0] != values[1]:
        raise _ProviderParseError("chat completion usage aliases conflict")
    return values[0]


def _parse_usage(response: dict[str, Any]) -> BackendTokenUsage:
    usage = response.get("usage")
    if type(usage) is not dict:
        raise _ProviderParseError("chat completion response is missing token usage")
    input_tokens = _usage_alias(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_alias(usage, "completion_tokens", "output_tokens")
    if "total_tokens" in usage:
        total = usage["total_tokens"]
        if type(total) is not int or total < 0 or total != input_tokens + output_tokens:
            raise _ProviderParseError("chat completion total token usage is inconsistent")
    return BackendTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _provider_identity(
    response: dict[str, Any], request: NativeFunctionCallRequest
) -> ProviderIdentityObservation | None:
    values: list[str | None] = []
    for field_name in ("model", "system_fingerprint"):
        value = response.get(field_name)
        if value is not None and (
            not isinstance(value, str)
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise _ProviderParseError("chat completion provider identity is invalid")
        values.append(value)
    if values == [None, None]:
        return None
    failed = False
    try:
        identity = ProviderIdentityObservation(
            requested_model=request.requested_model,
            response_model=values[0],
            system_fingerprint=values[1],
            backend_fingerprint=request.backend_fingerprint,
        )
    except (TypeError, ValueError, ValidationError):
        failed = True
    if failed:
        raise _ProviderParseError("chat completion provider identity is invalid") from None
    return identity


def _tool_call_from_wire(
    raw: Any,
    by_wire: dict[str, FrozenNativeFunctionTool],
) -> NativeAssistantToolCall:
    if type(raw) is not dict or set(raw) != {"id", "type", "function"}:
        raise _ProviderParseError("assistant tool call has malformed fields")
    function = raw.get("function")
    if type(function) is not dict or set(function) != {"name", "arguments"}:
        raise _ProviderParseError("assistant tool-call function has malformed fields")
    wire_name = function.get("name")
    tool = by_wire.get(wire_name) if isinstance(wire_name, str) else None
    if tool is None:
        raise _ProviderParseError("assistant invoked outside the frozen wire-tool roster")
    failed = False
    try:
        call = NativeAssistantToolCall(
            call_id=raw.get("id"),
            type=raw.get("type"),
            official_name=tool.official_name,
            wire_name=tool.wire_name,
            arguments_json=function.get("arguments"),
        )
    except (TypeError, ValueError, ValidationError):
        failed = True
    if failed:
        raise _ProviderParseError("assistant tool call is malformed") from None
    return call


def _parse_completion(
    response: dict[str, Any], request: NativeFunctionCallRequest
) -> tuple[tuple[NativeAssistantToolCall, ...], str | None, Literal["stop", "tool_calls"]]:
    choices = response.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise _ProviderParseError("native completion requires exactly one object choice")
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish == "length":
        raise _ProviderParseError("native chat completion was truncated")
    if finish not in {"stop", "tool_calls"}:
        raise _ProviderParseError("native completion has no supported finish reason")
    message = choice.get("message")
    if type(message) is not dict or message.get("role") != "assistant":
        raise _ProviderParseError("response requires an explicit assistant message role")
    if "function_call" in message:
        raise _ProviderParseError("legacy function_call is outside the native protocol")
    if "tool_calls" in message:
        raw_calls = message["tool_calls"]
        if raw_calls is None:
            raise _ProviderParseError("tool_calls:null is rejected by parser policy v1")
        if finish != "tool_calls" or type(raw_calls) is not list or not raw_calls:
            raise _ProviderParseError("assistant tool_calls disagree with finish reason")
        by_wire = {
            tool.wire_name: tool
            for tool in request.task_required_tools + request.harness_added_tools
        }
        calls = tuple(_tool_call_from_wire(raw, by_wire) for raw in raw_calls)
        ids = tuple(call.call_id for call in calls)
        if len(ids) != len(set(ids)):
            raise _ProviderParseError("assistant response contains duplicate tool-call IDs")
        if request.historical_call_ids.intersection(ids):
            raise _ProviderParseError("response tool-call IDs overlap transcript history")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise _ProviderParseError("assistant call content must be text or null")
        return calls, content, "tool_calls"
    if finish != "stop" or not isinstance(message.get("content"), str):
        raise _ProviderParseError("no-call response requires text and finish_reason=stop")
    return (), message["content"], "stop"


def _parse_completion_and_identity(
    response: dict[str, Any], request: NativeFunctionCallRequest
) -> tuple[
    tuple[NativeAssistantToolCall, ...],
    str | None,
    Literal["stop", "tool_calls"],
    ProviderIdentityObservation | None,
]:
    calls, text, finish = _parse_completion(response, request)
    return calls, text, finish, _provider_identity(response, request)


def _capture_parse[T](operation: Callable[[], T]) -> tuple[T | None, str | None]:
    try:
        return operation(), None
    except _ProviderParseError as exc:
        return None, str(exc)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleNativeFunctionBackend:
    """Direct-only, no-redirect ``/chat/completions`` native transport."""

    base_url: str
    api_key: str = field(repr=False)
    user_agent: str = "spiral-harness/openai-native-function-backend"
    serializer_fingerprint: str = field(init=False)
    parser_fingerprint: str = field(init=False)
    transport_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        base_url = _safe_base_url(self.base_url)
        if (
            not isinstance(self.api_key, str)
            or not self.api_key
            or any(ord(character) < 32 for character in self.api_key)
        ):
            raise ValueError("api_key must be a non-empty string")
        _reject_surrogate_text(self.api_key)
        valid_agent = isinstance(self.user_agent, str) and bool(self.user_agent)
        if (
            not valid_agent
            or self.user_agent.strip() != self.user_agent
            or any(ord(character) < 32 for character in self.user_agent)
        ):
            raise ValueError("user_agent must be an exact non-empty string")
        _reject_surrogate_text(self.user_agent)
        serializer = native_function_serializer_fingerprint()
        parser = native_function_parser_fingerprint()
        transport = native_function_transport_fingerprint(
            base_url=base_url, user_agent=self.user_agent
        )
        backend = canonical_sha256(
            {
                "id": _BACKEND_ID,
                "base_url": base_url,
                "serializer": serializer,
                "parser": parser,
                "transport": transport,
            }
        )
        for name, value in (
            ("base_url", base_url),
            ("serializer_fingerprint", serializer),
            ("parser_fingerprint", parser),
            ("transport_fingerprint", transport),
            ("fingerprint", backend),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_endpoint(cls, *, base_url: str, api_key: str) -> OpenAICompatibleNativeFunctionBackend:
        return cls(base_url=base_url, api_key=api_key)

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        checked = NativeFunctionCallRequest.model_validate(request, strict=True)
        expected = (
            (checked.backend_fingerprint, self.fingerprint, "backend"),
            (checked.serializer_fingerprint, self.serializer_fingerprint, "serializer"),
            (checked.parser_fingerprint, self.parser_fingerprint, "parser"),
            (checked.transport_fingerprint, self.transport_fingerprint, "transport"),
        )
        for declared, actual, label in expected:
            if declared != actual:
                raise OpenAICompatibleNativeFunctionError(
                    f"{label} fingerprint differs from the frozen native request"
                )
        raw = self._post_raw(
            "/chat/completions",
            _serialize_native_request(checked),
            timeout_seconds=checked.inference.timeout_seconds,
        )
        response, error = _capture_parse(lambda: _decode_raw_response(raw))
        if error is not None:
            raise OpenAICompatibleNativeInvalidResponseError(error, usage=None) from None
        assert response is not None
        usage, error = _capture_parse(lambda: _parse_usage(response))
        if error is not None:
            raise OpenAICompatibleNativeInvalidResponseError(error, usage=None) from None
        assert usage is not None
        parsed, error = _capture_parse(lambda: _parse_completion_and_identity(response, checked))
        if error is not None:
            raise OpenAICompatibleNativeInvalidResponseError(error, usage=usage) from None
        assert parsed is not None
        calls, text, finish, identity = parsed
        return NativeFunctionCallResponse(
            request_fingerprint=checked.fingerprint,
            serializer_fingerprint=self.serializer_fingerprint,
            parser_fingerprint=self.parser_fingerprint,
            transport_fingerprint=self.transport_fingerprint,
            tools_fingerprint=checked.tools_fingerprint,
            tool_calls=calls,
            assistant_text=text,
            finish_reason=finish,
            usage=usage,
            provider_identity_observation=identity,
        )

    def _post_raw(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> bytes:
        return _post_raw_http(
            url=self.base_url + path,
            api_key=self.api_key,
            user_agent=self.user_agent,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )


__all__ = [
    "FrozenNativeChatMessage",
    "FrozenNativeFunctionTool",
    "NativeAssistantToolCall",
    "NativeFunctionCallRequest",
    "NativeFunctionCallResponse",
    "OpenAICompatibleNativeFunctionBackend",
    "OpenAICompatibleNativeFunctionError",
    "OpenAICompatibleNativeInvalidResponseError",
    "native_function_parser_fingerprint",
    "native_function_serializer_fingerprint",
    "native_function_transport_fingerprint",
]
