"""Native function-call projection for one BFCL public-v2 visible request.

This module is deliberately a transport adapter, not an execution authority.  It
accepts only the already-minimized model-visible request produced by the trusted
v2 resolver and binds those exact bytes to the frozen native backend.  It does
not receive task IDs, split coordinates, answers, grades, or controller state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    BfclV4PublicPilotError,
    adapt_bfcl_v4_openai_completions_tools,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2ModelVisibleRequest,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeFunctionCallRequest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BfclV4PublicV2NativeMaterializationError(ValueError):
    """A safe visible request could not be bound to the frozen native wire."""


@runtime_checkable
class BfclV4PublicV2NativeBackendIdentity(Protocol):
    """Credential-free implementation identities exposed by the live backend."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def serializer_fingerprint(self) -> str: ...

    @property
    def parser_fingerprint(self) -> str: ...

    @property
    def transport_fingerprint(self) -> str: ...


def _reject(message: str) -> None:
    raise BfclV4PublicV2NativeMaterializationError(message) from None


def _checked_visible(
    value: BfclV4PublicV2ModelVisibleRequest,
) -> BfclV4PublicV2ModelVisibleRequest:
    if type(value) is not BfclV4PublicV2ModelVisibleRequest:
        _reject("visible request must use the exact BFCL v2 contract")
    try:
        return BfclV4PublicV2ModelVisibleRequest.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("visible request differs from the frozen BFCL v2 contract")


def _checked_spec(value: FrozenModelSpec) -> FrozenModelSpec:
    if type(value) is not FrozenModelSpec:
        _reject("model spec must use the exact frozen execution contract")
    try:
        return FrozenModelSpec.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("model spec differs from the frozen execution contract")


def _backend_fingerprints(
    backend: BfclV4PublicV2NativeBackendIdentity,
) -> tuple[str, str, str, str]:
    if not isinstance(backend, BfclV4PublicV2NativeBackendIdentity):
        _reject("backend does not expose the four frozen native identities")
    try:
        values = (
            backend.fingerprint,
            backend.serializer_fingerprint,
            backend.parser_fingerprint,
            backend.transport_fingerprint,
        )
    except Exception:
        _reject("backend native identities are unavailable")
    if any(type(value) is not str or _SHA256.fullmatch(value) is None for value in values):
        _reject("backend native identities must be lowercase SHA-256 values")
    return values


def _strict_canonical_json(text: str, *, label: str) -> Any:
    try:
        value = json.loads(text)
        if canonical_json(value) != text:
            raise ValueError("non-canonical JSON")
        return value
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        _reject(f"{label} must be canonical strict JSON")


def _messages(
    visible: BfclV4PublicV2ModelVisibleRequest,
) -> tuple[FrozenNativeChatMessage, ...]:
    question = _strict_canonical_json(visible.question_json, label="BFCL v2 question")
    if type(question) is not list or len(question) != 1:
        _reject("BFCL v2 native request requires exactly one conversation turn")
    turn = question[0]
    if type(turn) is not list or not turn:
        _reject("BFCL v2 conversation turn must be a non-empty message list")

    task_messages: list[FrozenNativeChatMessage] = []
    try:
        for item in turn:
            if type(item) is not dict or set(item) != {"role", "content"}:
                raise ValueError("message shape")
            if item["role"] not in {"system", "user"} or type(item["content"]) is not str:
                raise ValueError("message values")
            task_messages.append(
                FrozenNativeChatMessage(role=item["role"], content=item["content"])
            )
    except (KeyError, TypeError, ValidationError, ValueError):
        _reject("BFCL v2 question cannot be represented as native messages")

    if visible.system_prompt is None:
        if any(message.role == "system" for message in task_messages):
            _reject("bare BFCL v2 request unexpectedly contains a task system message")
        return tuple(task_messages)
    try:
        harness_message = FrozenNativeChatMessage(
            role="system",
            content=visible.system_prompt,
        )
    except (TypeError, ValidationError, ValueError):
        _reject("BFCL v2 harness prompt cannot be represented as a native message")
    return (harness_message, *task_messages)


def _official_schema(
    wire_tool: object,
    *,
    official_name: str,
    wire_name: str,
) -> Mapping[str, Any]:
    if (
        type(wire_tool) is not dict
        or set(wire_tool) != {"type", "function"}
        or wire_tool.get("type") != "function"
    ):
        _reject("pinned BFCL adapter emitted an unexpected tool envelope")
    schema = wire_tool.get("function")
    if type(schema) is not dict or schema.get("name") != wire_name:
        _reject("pinned BFCL adapter tool differs from its wire-name binding")
    return {**schema, "name": official_name}


def _tools(
    visible: BfclV4PublicV2ModelVisibleRequest,
) -> tuple[FrozenNativeFunctionTool, ...]:
    functions = _strict_canonical_json(
        visible.function_schemas_json,
        label="BFCL v2 function schemas",
    )
    if type(functions) is not list or not functions:
        _reject("BFCL v2 function schemas must be one non-empty list")
    try:
        adapted = adapt_bfcl_v4_openai_completions_tools(functions)
        wire_tools = _strict_canonical_json(
            adapted.tools_json,
            label="adapted BFCL v2 tools",
        )
        if type(wire_tools) is not list or len(wire_tools) != len(adapted.name_bindings):
            raise ValueError("adapter cardinality")
        result = tuple(
            FrozenNativeFunctionTool.from_schema(
                _official_schema(
                    wire_tool,
                    official_name=binding.official_name,
                    wire_name=binding.wire_name,
                ),
                wire_name=binding.wire_name,
            )
            for wire_tool, binding in zip(wire_tools, adapted.name_bindings, strict=True)
        )
    except (
        BfclV4PublicPilotError,
        BfclV4PublicV2NativeMaterializationError,
        TypeError,
        ValidationError,
        ValueError,
    ):
        _reject("BFCL v2 schemas failed the pinned native tool adapter")
    if tuple(tool.official_name for tool in result) != tuple(
        item.get("name") for item in functions
    ):
        _reject("native BFCL v2 tool roster differs from the visible schemas")
    return result


def materialize_bfcl_v4_public_v2_native_request(
    *,
    visible_request: BfclV4PublicV2ModelVisibleRequest,
    expected_visible_request_sha256: str,
    spec: FrozenModelSpec,
    backend: BfclV4PublicV2NativeBackendIdentity,
) -> NativeFunctionCallRequest:
    """Bind one minimal visible request to an exact native provider request.

    ``expected_visible_request_sha256`` must come from the surrounding trusted
    request wrapper.  This transport layer checks it but cannot issue semantic,
    evaluation, or scoring authority.
    """

    visible = _checked_visible(visible_request)
    checked_spec = _checked_spec(spec)
    if (
        type(expected_visible_request_sha256) is not str
        or _SHA256.fullmatch(expected_visible_request_sha256) is None
    ):
        _reject("expected visible request identity must be a lowercase SHA-256")
    if visible.fingerprint != expected_visible_request_sha256:
        _reject("visible request differs from its trusted wrapper identity")

    backend_fingerprint, serializer, parser, transport = _backend_fingerprints(backend)
    if backend_fingerprint != checked_spec.backend_fingerprint:
        _reject("native backend fingerprint differs from the frozen model spec")
    if visible.model_route != checked_spec.model or visible.inference != checked_spec.inference:
        _reject("visible request model or inference differs from the frozen model spec")

    try:
        request = NativeFunctionCallRequest(
            backend_fingerprint=backend_fingerprint,
            serializer_fingerprint=serializer,
            parser_fingerprint=parser,
            transport_fingerprint=transport,
            requested_model=visible.model_route,
            messages=_messages(visible),
            task_required_tools=_tools(visible),
            harness_added_tools=(),
            seed=visible.provider_seed_u63,
            inference=visible.inference,
        )
    except (
        BfclV4PublicV2NativeMaterializationError,
        TypeError,
        ValidationError,
        ValueError,
    ):
        _reject("BFCL v2 visible request cannot be represented on the frozen native wire")

    observed = (
        request.backend_fingerprint,
        request.serializer_fingerprint,
        request.parser_fingerprint,
        request.transport_fingerprint,
        request.requested_model,
        request.seed,
        request.inference,
        request.harness_added_tools,
    )
    expected = (
        backend_fingerprint,
        serializer,
        parser,
        transport,
        visible.model_route,
        visible.provider_seed_u63,
        visible.inference,
        (),
    )
    if observed != expected:
        _reject("native BFCL v2 request drifted from the visible execution binding")
    return request


__all__ = [name for name in globals() if name.startswith("Bfcl") or name.startswith("materialize")]
