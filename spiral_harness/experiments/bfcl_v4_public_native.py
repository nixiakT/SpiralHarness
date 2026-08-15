"""Pure request materialization for the public/development BFCL V4 pilot.

This module stops at immutable provider input.  It neither invokes a model nor
loads benchmark files.  The caller must supply a task already bound to the
frozen public roster and the exact output of the pinned OpenAI-completions
adapter.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    BfclV4PublicPilotError,
    adapt_bfcl_v4_public_pilot_task,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4OpenAiToolAdapterResult,
    BfclV4PublicPilotTask,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeFunctionCallRequest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_PROVIDER_SEED = 2**63 - 1


class BfclV4PublicNativeMaterializationError(ValueError):
    """A solver-side public-pilot request could not be frozen exactly."""


class BfclV4PublicNativePromptKind(StrEnum):
    """Prompt provenance for a task-bound solver request."""

    PURE = "pure"
    STATIC = "static"
    PARENT = "parent"
    CANDIDATE = "candidate"


@runtime_checkable
class BfclV4PublicNativeBackendIdentity(Protocol):
    """Secret-free implementation identities exposed by the native backend."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def serializer_fingerprint(self) -> str: ...

    @property
    def parser_fingerprint(self) -> str: ...

    @property
    def transport_fingerprint(self) -> str: ...


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object keys are forbidden")
        output[key] = value
    return output


def _reject_non_json_values(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in item):
                raise ValueError("Unicode surrogate code points are forbidden")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON numbers are forbidden")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _strict_json(text: str, *, label: str) -> Any:
    if not isinstance(text, str):
        raise BfclV4PublicNativeMaterializationError(f"{label} must be text")
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
        _reject_non_json_values(parsed)
        if canonical_json(parsed) != text:
            raise ValueError("JSON is not canonical")
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            f"{label} must be canonical strict JSON"
        ) from error
    return parsed


def _checked_task(task: BfclV4PublicPilotTask) -> BfclV4PublicPilotTask:
    try:
        return BfclV4PublicPilotTask.model_validate(task, strict=True)
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            "BFCL task differs from the frozen public-pilot contract"
        ) from error


def _checked_adapter(
    task: BfclV4PublicPilotTask,
    adapter: BfclV4OpenAiToolAdapterResult,
) -> BfclV4OpenAiToolAdapterResult:
    try:
        checked = BfclV4OpenAiToolAdapterResult.model_validate(adapter, strict=True)
        expected = adapt_bfcl_v4_public_pilot_task(task)
    except (BfclV4PublicPilotError, TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            "BFCL adapter output is not valid for the frozen task"
        ) from error
    if checked != expected:
        raise BfclV4PublicNativeMaterializationError(
            "BFCL adapter output differs from the pinned task adapter"
        )
    return checked


def _official_schema(
    wire_tool: Any,
    *,
    official_name: str,
    wire_name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(wire_tool, dict)
        or set(wire_tool) != {"type", "function"}
        or wire_tool.get("type") != "function"
    ):
        raise BfclV4PublicNativeMaterializationError(
            "pinned BFCL adapter emitted an unexpected tool envelope"
        )
    wire_schema = wire_tool.get("function")
    if not isinstance(wire_schema, dict) or wire_schema.get("name") != wire_name:
        raise BfclV4PublicNativeMaterializationError(
            "pinned BFCL adapter tool differs from its wire-name binding"
        )
    return {**wire_schema, "name": official_name}


def materialize_bfcl_v4_public_native_tools(
    task: BfclV4PublicPilotTask,
    adapter: BfclV4OpenAiToolAdapterResult,
) -> tuple[FrozenNativeFunctionTool, ...]:
    """Freeze adapter-normalized schemas with explicit official/wire names."""

    checked_task = _checked_task(task)
    checked_adapter = _checked_adapter(checked_task, adapter)
    wire_tools = _strict_json(checked_adapter.tools_json, label="BFCL adapted tools")
    if not isinstance(wire_tools, list) or len(wire_tools) != len(checked_adapter.name_bindings):
        raise BfclV4PublicNativeMaterializationError(
            "BFCL adapted tools differ from their explicit name bindings"
        )

    frozen: list[FrozenNativeFunctionTool] = []
    for wire_tool, binding in zip(
        wire_tools,
        checked_adapter.name_bindings,
        strict=True,
    ):
        schema = _official_schema(
            wire_tool,
            official_name=binding.official_name,
            wire_name=binding.wire_name,
        )
        try:
            tool = FrozenNativeFunctionTool.from_schema(
                schema,
                wire_name=binding.wire_name,
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise BfclV4PublicNativeMaterializationError(
                "BFCL adapted tool cannot be represented by the native provider contract"
            ) from error
        if tool.wire_schema != wire_tool["function"]:
            raise BfclV4PublicNativeMaterializationError(
                "native tool serialization differs from the pinned BFCL adapter output"
            )
        frozen.append(tool)

    if tuple(tool.official_name for tool in frozen) != checked_task.official_function_names:
        raise BfclV4PublicNativeMaterializationError(
            "native tool roster differs from the frozen BFCL task"
        )
    return tuple(frozen)


def _question_messages(task: BfclV4PublicPilotTask) -> tuple[FrozenNativeChatMessage, ...]:
    question = _strict_json(task.question_json, label="BFCL question")
    if not isinstance(question, list) or len(question) != 1:
        raise BfclV4PublicNativeMaterializationError(
            "public-pilot solver requests require exactly one BFCL conversation turn"
        )
    turn = question[0]
    if not isinstance(turn, list) or not turn:
        raise BfclV4PublicNativeMaterializationError(
            "BFCL conversation turn must be a non-empty message list"
        )

    messages: list[FrozenNativeChatMessage] = []
    for raw_message in turn:
        if not isinstance(raw_message, dict) or set(raw_message) != {"role", "content"}:
            raise BfclV4PublicNativeMaterializationError(
                "BFCL question message has an unsupported shape"
            )
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise BfclV4PublicNativeMaterializationError(
                "BFCL question message requires a system/user role and text content"
            )
        try:
            messages.append(FrozenNativeChatMessage(role=role, content=content))
        except (TypeError, ValidationError, ValueError) as error:
            raise BfclV4PublicNativeMaterializationError(
                "BFCL question message violates the native provider contract"
            ) from error
    return tuple(messages)


def _prompt_messages(
    task_messages: tuple[FrozenNativeChatMessage, ...],
    *,
    prompt_kind: BfclV4PublicNativePromptKind,
    system_prompt: str | None,
) -> tuple[FrozenNativeChatMessage, ...]:
    if type(prompt_kind) is not BfclV4PublicNativePromptKind:
        raise BfclV4PublicNativeMaterializationError(
            "prompt_kind must be an exact BFCL native prompt kind"
        )
    if prompt_kind is BfclV4PublicNativePromptKind.PURE:
        if system_prompt is not None:
            raise BfclV4PublicNativeMaterializationError(
                "PURE request must not contain a harness system prompt"
            )
        if any(message.role == "system" for message in task_messages):
            raise BfclV4PublicNativeMaterializationError(
                "frozen public-pilot PURE task unexpectedly contains a system message"
            )
        return task_messages

    if not isinstance(system_prompt, str) or not system_prompt:
        raise BfclV4PublicNativeMaterializationError(
            "STATIC, parent, and candidate requests require an exact system prompt"
        )
    try:
        harness_message = FrozenNativeChatMessage(role="system", content=system_prompt)
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            "harness system prompt violates the native provider contract"
        ) from error
    return (harness_message, *task_messages)


def _backend_fingerprints(
    backend: BfclV4PublicNativeBackendIdentity,
) -> tuple[str, str, str, str]:
    if not isinstance(backend, BfclV4PublicNativeBackendIdentity):
        raise BfclV4PublicNativeMaterializationError(
            "backend does not expose the four frozen native identities"
        )
    try:
        values = (
            backend.fingerprint,
            backend.serializer_fingerprint,
            backend.parser_fingerprint,
            backend.transport_fingerprint,
        )
    except Exception as error:
        raise BfclV4PublicNativeMaterializationError(
            "backend native identities are unavailable"
        ) from error
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in values):
        raise BfclV4PublicNativeMaterializationError(
            "backend native identities must be lowercase SHA-256 values"
        )
    return values


def materialize_bfcl_v4_public_native_request(
    *,
    task: BfclV4PublicPilotTask,
    adapter: BfclV4OpenAiToolAdapterResult,
    spec: FrozenModelSpec,
    backend: BfclV4PublicNativeBackendIdentity,
    seed: int,
    prompt_kind: BfclV4PublicNativePromptKind,
    system_prompt: str | None = None,
) -> NativeFunctionCallRequest:
    """Build one score-free request bound to a frozen backend, spec, and seed."""

    checked_task = _checked_task(task)
    try:
        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            "model spec differs from the frozen execution contract"
        ) from error
    if type(seed) is not int or not 0 <= seed <= _MAX_PROVIDER_SEED:
        raise BfclV4PublicNativeMaterializationError(
            "provider seed must be an unsigned 63-bit integer"
        )

    backend_fingerprint, serializer, parser, transport = _backend_fingerprints(backend)
    if backend_fingerprint != checked_spec.backend_fingerprint:
        raise BfclV4PublicNativeMaterializationError(
            "native backend fingerprint differs from the frozen model spec"
        )
    tools = materialize_bfcl_v4_public_native_tools(checked_task, adapter)
    messages = _prompt_messages(
        _question_messages(checked_task),
        prompt_kind=prompt_kind,
        system_prompt=system_prompt,
    )
    try:
        request = NativeFunctionCallRequest(
            backend_fingerprint=backend_fingerprint,
            serializer_fingerprint=serializer,
            parser_fingerprint=parser,
            transport_fingerprint=transport,
            requested_model=checked_spec.model,
            messages=messages,
            task_required_tools=tools,
            harness_added_tools=(),
            seed=seed,
            inference=checked_spec.inference,
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicNativeMaterializationError(
            "BFCL task cannot be represented by the frozen native request contract"
        ) from error

    if (
        request.backend_fingerprint != checked_spec.backend_fingerprint
        or request.requested_model != checked_spec.model
        or request.inference != checked_spec.inference
        or request.seed != seed
        or request.harness_added_tools
    ):
        raise BfclV4PublicNativeMaterializationError(
            "materialized BFCL request drifted from its frozen execution binding"
        )
    return request


__all__ = [
    "BfclV4PublicNativeBackendIdentity",
    "BfclV4PublicNativeMaterializationError",
    "BfclV4PublicNativePromptKind",
    "materialize_bfcl_v4_public_native_request",
    "materialize_bfcl_v4_public_native_tools",
]
