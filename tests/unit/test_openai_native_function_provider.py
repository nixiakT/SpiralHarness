from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.execution.contracts import InferenceConfig
from spiral_harness.providers.openai_native_function import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    OpenAICompatibleNativeFunctionBackend,
    OpenAICompatibleNativeFunctionError,
    OpenAICompatibleNativeInvalidResponseError,
)


class CapturingBackend(OpenAICompatibleNativeFunctionBackend):
    def __init__(self, response: dict[str, Any] | bytes) -> None:
        super().__init__(base_url="http://litellm.example/v1/", api_key="fixture-secret")
        raw = response if isinstance(response, bytes) else _raw_json(response)
        object.__setattr__(self, "response", raw)
        object.__setattr__(self, "calls", [])

    def _post_raw(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> bytes:
        self.calls.append((path, payload, timeout_seconds))
        return self.response


def _raw_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _exception_graph_text(error: BaseException) -> str:
    pending = [error]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        rendered.extend((str(item), repr(item)))
        for linked in (item.__cause__, item.__context__):
            if linked is not None:
                pending.append(linked)
    return "\n".join(rendered)


def _history_call(call_id: str = "call_prior") -> NativeAssistantToolCall:
    return NativeAssistantToolCall(
        call_id=call_id,
        official_name="official_lookup",
        wire_name="official_lookup",
        arguments_json='{"value":1}',
    )


def _tool(name: str, *, wire_name: str | None = None) -> FrozenNativeFunctionTool:
    return FrozenNativeFunctionTool.from_schema(
        {
            "name": name,
            "description": f"Call {name}.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        wire_name=wire_name,
    )


def _request(
    backend: OpenAICompatibleNativeFunctionBackend,
    *,
    task_tools: tuple[FrozenNativeFunctionTool, ...] | None = None,
    added_tools: tuple[FrozenNativeFunctionTool, ...] = (),
    messages: tuple[FrozenNativeChatMessage, ...] | None = None,
) -> NativeFunctionCallRequest:
    return NativeFunctionCallRequest(
        backend_fingerprint=backend.fingerprint,
        serializer_fingerprint=backend.serializer_fingerprint,
        parser_fingerprint=backend.parser_fingerprint,
        transport_fingerprint=backend.transport_fingerprint,
        requested_model="dashscope/qwen36-35b-a3b",
        messages=messages
        or (
            FrozenNativeChatMessage(
                role="user",
                content="  Keep this whitespace exactly.\n",
            ),
        ),
        task_required_tools=task_tools or (_tool("official_lookup"),),
        harness_added_tools=added_tools,
        seed=17,
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=128,
            timeout_seconds=9.0,
            stop_sequences=("<END>",),
        ),
    )


def _tool_call_response(
    *calls: tuple[str, str, str],
    finish_reason: str = "tool_calls",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for call_id, name, arguments in calls
                    ],
                },
            }
        ],
        "model": "qwen36-35b-a3b-served-snapshot",
        "system_fingerprint": "fp_provider_snapshot",
        "usage": {"prompt_tokens": 31, "completion_tokens": 9},
    }


def test_native_backend_posts_frozen_request_and_binds_score_free_response() -> None:
    backend = CapturingBackend(
        _tool_call_response(
            ("call_1", "official_lookup", '{"value": 3}'),
            ("call_2", "harness_helper", '{\n  "value": 4\n}'),
        )
    )
    request = _request(backend, added_tools=(_tool("harness_helper"),))

    response = backend.invoke(request=request)

    assert response.request_fingerprint == request.fingerprint
    assert response.serializer_fingerprint == backend.serializer_fingerprint
    assert response.parser_fingerprint == backend.parser_fingerprint
    assert response.transport_fingerprint == backend.transport_fingerprint
    assert response.tools_fingerprint == request.tools_fingerprint
    assert response.protocol_scope == "normalized-openai-compatible-litellm/v1"
    assert response.upstream_handler_equivalence_attested is False
    assert response.source_bound is True
    assert response.runtime_execution_attested is False
    assert response.finish_reason == "tool_calls"
    assert response.assistant_text is None
    assert response.tool_call_count == 2
    assert response.usage.input_tokens == 31
    assert response.usage.output_tokens == 9
    assert response.tool_calls[0].arguments_json == '{"value": 3}'
    assert response.tool_calls[0].official_name == "official_lookup"
    assert response.tool_calls[0].wire_name == "official_lookup"
    assert response.tool_calls[0].canonical_arguments_json == '{"value":3}'
    assert response.tool_calls[1].canonical_arguments_json == '{"value":4}'
    assert not hasattr(response, "score")
    identity = response.provider_identity_observation
    assert identity is not None
    assert identity.trust_level == "provider-declared"
    assert identity.requested_model == "dashscope/qwen36-35b-a3b"
    assert identity.response_model == "qwen36-35b-a3b-served-snapshot"
    assert identity.system_fingerprint == "fp_provider_snapshot"
    assert identity.backend_fingerprint == backend.fingerprint

    path, payload, timeout = backend.calls[0]
    assert path == "/chat/completions"
    assert timeout == 9.0
    assert payload["model"] == "dashscope/qwen36-35b-a3b"
    assert payload["messages"] == [{"role": "user", "content": "  Keep this whitespace exactly.\n"}]
    assert payload["tools"] == [
        {"type": "function", "function": _schema("official_lookup")},
        {"type": "function", "function": _schema("harness_helper")},
    ]
    assert payload["tool_choice"] == "auto"
    assert payload["n"] == 1
    assert payload["parallel_tool_calls"] is True
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 128
    assert payload["seed"] == 17
    assert payload["stop"] == ["<END>"]


def _schema(name: str) -> dict[str, Any]:
    return {
        "description": f"Call {name}.",
        "name": name,
        "parameters": {
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "type": "object",
        },
    }


def test_task_required_and_harness_added_provenance_changes_tools_fingerprint() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "No call needed."},
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        }
    )
    official = _tool("official_lookup")
    helper = _tool("harness_helper")
    mixed = _request(backend, task_tools=(official,), added_tools=(helper,))
    all_required = _request(backend, task_tools=(official, helper))

    assert mixed.tools_fingerprint != all_required.tools_fingerprint
    response = backend.invoke(request=mixed)
    assert response.tool_calls == ()
    assert response.assistant_text == "No call needed."
    assert response.finish_reason == "stop"
    assert response.provider_identity_observation is None


def test_official_schema_drift_changes_tools_and_request_fingerprints() -> None:
    backend = CapturingBackend({})
    original = _request(backend)
    changed_tool = FrozenNativeFunctionTool.from_schema(
        {
            "name": "official_lookup",
            "description": "A changed official schema.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        }
    )
    drifted = _request(backend, task_tools=(changed_tool,))

    assert drifted.tools_fingerprint != original.tools_fingerprint
    assert drifted.fingerprint != original.fingerprint


@pytest.mark.parametrize("content", ["\ud800", "prefix\udfff suffix"])
def test_local_request_message_content_rejects_lone_surrogates(content: str) -> None:
    with pytest.raises(ValidationError, match="surrogate"):
        FrozenNativeChatMessage(role="user", content=content)


def test_local_message_validation_error_never_echoes_caller_input() -> None:
    secret = "message-content-unique-secret"
    with pytest.raises(ValidationError, match="surrogate") as exc_info:
        FrozenNativeChatMessage(role="user", content=f"{secret}\ud800")

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert secret not in _exception_graph_text(exc_info.value)


@pytest.mark.parametrize(
    ("field", "secret", "surrogate"),
    [
        ("official_name", "official-name-unique-secret", "\ud800"),
        ("description", "description-unique-secret", "\udfff"),
    ],
)
def test_local_schema_validation_error_never_echoes_caller_input(
    field: str,
    secret: str,
    surrogate: str,
) -> None:
    schema = {
        "name": "safe_name",
        "description": "safe description",
        "parameters": {"type": "object", "properties": {}},
    }
    schema["name" if field == "official_name" else field] = f"{secret}{surrogate}"

    with pytest.raises(ValidationError, match="surrogate") as exc_info:
        FrozenNativeFunctionTool.from_schema(schema)

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert secret not in _exception_graph_text(exc_info.value)


def test_local_identifiers_and_argument_json_reject_lone_surrogates() -> None:
    with pytest.raises(ValidationError, match="surrogate"):
        NativeAssistantToolCall(
            call_id="call_\ud800",
            official_name="official_lookup",
            wire_name="official_lookup",
            arguments_json="{}",
        )
    with pytest.raises(ValidationError, match="surrogate"):
        NativeAssistantToolCall(
            call_id="call_safe",
            official_name="official_lookup",
            wire_name="official_lookup",
            arguments_json='{"value":"\udfff"}',
        )


@pytest.mark.parametrize(
    "schema",
    [
        {
            "name": "unsafe_description",
            "description": "\ud800",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "unsafe_key",
            "parameters": {
                "type": "object",
                "properties": {"\udfff": {"type": "string"}},
            },
        },
    ],
)
def test_local_schema_tree_rejects_surrogate_keys_and_values(schema: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="surrogate"):
        FrozenNativeFunctionTool.from_schema(schema)


def test_local_request_stop_sequence_rejects_surrogate_before_fingerprint() -> None:
    backend = CapturingBackend({})
    clean = _request(backend)
    unchecked_inference = clean.inference.model_copy(update={"stop_sequences": ("\ud800",)})
    payload = clean.model_dump(mode="python")
    payload["inference"] = unchecked_inference

    with pytest.raises(ValidationError, match="surrogate"):
        NativeFunctionCallRequest.model_validate(payload, strict=True)


def test_local_response_artifact_rejects_surrogate_and_false_source_binding() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    response = backend.invoke(request=_request(backend))
    payload = response.model_dump(mode="python")
    payload["assistant_text"] = "artifact\ud800"
    with pytest.raises(ValidationError, match="surrogate"):
        type(response).model_validate(payload, strict=True)
    payload["assistant_text"] = "done"
    payload["source_bound"] = False
    with pytest.raises(ValidationError, match="source_bound"):
        type(response).model_validate(payload, strict=True)


def test_component_fingerprints_are_split_and_transport_binds_user_agent() -> None:
    first = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="fixture-one",
        user_agent="spiral-harness/test-agent-one",
    )
    second = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="fixture-two",
        user_agent="spiral-harness/test-agent-two",
    )

    assert first.serializer_fingerprint == second.serializer_fingerprint
    assert first.parser_fingerprint == second.parser_fingerprint
    assert first.transport_fingerprint != second.transport_fingerprint
    assert first.fingerprint != second.fingerprint
    assert (
        len(
            {
                first.serializer_fingerprint,
                first.parser_fingerprint,
                first.transport_fingerprint,
            }
        )
        == 3
    )
    for fingerprint in (
        first.serializer_fingerprint,
        first.parser_fingerprint,
        first.transport_fingerprint,
        first.fingerprint,
    ):
        assert len(fingerprint) == 64
        int(fingerprint, 16)


def test_same_declared_function_may_be_called_twice_with_unique_ids() -> None:
    backend = CapturingBackend(
        _tool_call_response(
            ("call_1", "official_lookup", '{"value":1}'),
            ("call_2", "official_lookup", '{"value":2}'),
        )
    )

    response = backend.invoke(request=_request(backend))

    assert tuple(call.function_name for call in response.tool_calls) == (
        "official_lookup",
        "official_lookup",
    )


def test_dotted_official_name_requires_explicit_pinned_wire_mapping() -> None:
    with pytest.raises(ValidationError, match="explicit pinned adapter mapping"):
        _tool("math.add")

    mapped = _tool("math.add", wire_name="math_add")
    backend = CapturingBackend(_tool_call_response(("call_1", "math_add", '{"value":1}')))
    request = _request(backend, task_tools=(mapped,))

    response = backend.invoke(request=request)

    assert response.tool_calls[0].official_name == "math.add"
    assert response.tool_calls[0].wire_name == "math_add"
    assert response.tool_calls[0].function_name == "math.add"
    wire_schema = backend.calls[0][1]["tools"][0]["function"]
    assert wire_schema["name"] == "math_add"
    assert mapped.official_name == "math.add"
    assert '"name":"math.add"' in mapped.function_schema_json
    assert mapped.schema_input_contract == "pinned-adapter-normalized-official-schema/v1"
    assert mapped.adapter_normalization_runtime_attested is False
    assert wire_schema["parameters"] == _schema("math.add")["parameters"]


def test_duplicate_tool_schema_names_fail_before_transport() -> None:
    backend = CapturingBackend({})

    with pytest.raises(ValidationError, match="globally unique"):
        _request(
            backend,
            task_tools=(_tool("same_name"),),
            added_tools=(_tool("same_name"),),
        )
    assert backend.calls == []


def test_unknown_response_tool_name_is_rejected_with_known_usage() -> None:
    backend = CapturingBackend(_tool_call_response(("call_1", "not_in_the_task", '{"value":1}')))

    with pytest.raises(
        OpenAICompatibleNativeInvalidResponseError,
        match="outside the frozen wire-tool roster",
    ) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None
    assert exc_info.value.usage.total_tokens == 40


@pytest.mark.parametrize(
    "raw_call",
    [
        {"type": "function", "function": {"name": "official_lookup", "arguments": "{}"}},
        {
            "id": "call_1",
            "type": "not-function",
            "function": {"name": "official_lookup", "arguments": "{}"},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup"},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup", "arguments": "[]"},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup", "arguments": '{"x":1,"x":2}'},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup", "arguments": '{"x":NaN}'},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup", "arguments": '{"x":1e400}'},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "official_lookup", "arguments": '{"x":"\\ud800"}'},
        },
    ],
)
def test_malformed_native_tool_calls_are_rejected_without_echo(raw_call: dict[str, Any]) -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [raw_call]},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        }
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None
    assert exc_info.value.usage.total_tokens == 11
    assert "NaN" not in str(exc_info.value)


def test_duplicate_tool_call_ids_are_rejected() -> None:
    backend = CapturingBackend(
        _tool_call_response(
            ("call_duplicate", "official_lookup", '{"value":1}'),
            ("call_duplicate", "official_lookup", '{"value":2}'),
        )
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match="duplicate"):
        backend.invoke(request=_request(backend))


@pytest.mark.parametrize(
    ("finish_reason", "message", "error"),
    [
        (
            "length",
            {"role": "assistant", "content": "partial"},
            "truncated",
        ),
        (
            "tool_calls",
            {"role": "assistant", "content": "no calls"},
            "requires text and finish_reason=stop",
        ),
        (
            "stop",
            {"role": "assistant", "content": None},
            "requires text and finish_reason=stop",
        ),
        (
            "stop",
            {
                "role": "assistant",
                "content": "mismatch",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "official_lookup", "arguments": "{}"},
                    }
                ],
            },
            "disagree",
        ),
    ],
)
def test_finish_reason_and_completion_shape_must_agree(
    finish_reason: str,
    message: dict[str, Any],
    error: str,
) -> None:
    backend = CapturingBackend(
        {
            "choices": [{"finish_reason": finish_reason, "message": message}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        }
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match=error) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None


@pytest.mark.parametrize(
    "message",
    [
        {"content": "implicit roles are forbidden"},
        {"role": "user", "content": "wrong role"},
        {"role": None, "content": "null role"},
    ],
)
def test_response_requires_explicit_assistant_role(message: dict[str, Any]) -> None:
    backend = CapturingBackend(
        {
            "choices": [{"finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    with pytest.raises(
        OpenAICompatibleNativeInvalidResponseError,
        match="explicit assistant",
    ) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None


def test_tool_calls_null_is_rejected_by_versioned_strict_policy() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "provider emitted an ambiguous null",
                        "tool_calls": None,
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    with pytest.raises(
        OpenAICompatibleNativeInvalidResponseError,
        match=r"tool_calls:null.*policy v1",
    ) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 1},
        {"completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": 1},
    ],
)
def test_missing_or_invalid_usage_is_rejected(usage: dict[str, int] | None) -> None:
    response: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }
        ]
    }
    if usage is not None:
        response["usage"] = usage
    backend = CapturingBackend(response)

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match=r"usage|token usage"):
        backend.invoke(request=_request(backend))


@pytest.mark.parametrize(
    ("usage", "error"),
    [
        (
            {
                "prompt_tokens": 1,
                "input_tokens": 2,
                "completion_tokens": 3,
            },
            "aliases conflict",
        ),
        (
            {
                "prompt_tokens": 1,
                "completion_tokens": 3,
                "output_tokens": 4,
            },
            "aliases conflict",
        ),
        (
            {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 99},
            "total token usage is inconsistent",
        ),
    ],
)
def test_usage_alias_conflicts_and_total_mismatch_are_ambiguous(
    usage: dict[str, int], error: str
) -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": usage,
        }
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match=error) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is None


def test_equal_usage_aliases_and_matching_total_are_accepted() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "input_tokens": 2,
                "completion_tokens": 3,
                "output_tokens": 3,
                "total_tokens": 5,
            },
        }
    )

    assert backend.invoke(request=_request(backend)).usage.total_tokens == 5


def test_non_finite_provider_metadata_is_rejected() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "unsafe_metadata": float("nan"),
    }
    backend = CapturingBackend(response)

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match="strict JSON"):
        backend.invoke(request=_request(backend))


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (
            b'{"usage":{"prompt_tokens":1,"completion_tokens":1},'
            b'"usage":{"prompt_tokens":2,"completion_tokens":2},'
            b'"secret":"wire-duplicate-secret"}',
            "strict JSON",
        ),
        (
            b'{"usage":{"prompt_tokens":1,"completion_tokens":1},'
            b'"unsafe":NaN,"secret":"wire-nan-secret"}',
            "strict JSON",
        ),
        (
            b'{"usage":{"prompt_tokens":1,"completion_tokens":1},'
            b'"unsafe":1e400,"secret":"wire-overflow-secret"}',
            "strict JSON",
        ),
        (
            b'{"usage":{"prompt_tokens":1,"completion_tokens":1},'
            b'"unsafe":"\\ud800","secret":"wire-value-surrogate-secret"}',
            "strict JSON",
        ),
        (
            b'{"usage":{"prompt_tokens":1,"completion_tokens":1},'
            b'"\\udfff":0,"secret":"wire-key-surrogate-secret"}',
            "strict JSON",
        ),
        (b'{"secret":"wire-utf8-secret","bad":"\xff"}', "UTF-8"),
        (b"[]", "one JSON object"),
        (b"{" + b'"x":{' * 65 + b"0" + b"}" * 66, "strict JSON"),
        (b"x" * 1_048_577, "byte limit"),
    ],
)
def test_ambiguous_raw_wire_is_rejected_without_usage_or_secret_graph(
    raw: bytes,
    error: str,
) -> None:
    backend = CapturingBackend(raw)

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError, match=error) as exc_info:
        backend.invoke(request=_request(backend))

    assert exc_info.value.usage is None
    graph = _exception_graph_text(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "wire-" not in graph
    assert "Authorization" not in graph


def test_provider_tool_argument_parse_failure_keeps_usage_but_erases_secret_graph() -> None:
    secret = "provider-argument-secret"
    backend = CapturingBackend(
        _tool_call_response(("call_1", "official_lookup", f'{{"secret":"{secret}","x":NaN}}'))
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError) as exc_info:
        backend.invoke(request=_request(backend))

    assert exc_info.value.usage is not None
    assert exc_info.value.usage.total_tokens == 40
    graph = _exception_graph_text(exc_info.value)
    assert secret not in graph
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_provider_identity_parse_failure_erases_provider_value_graph() -> None:
    secret = "provider-identity-secret"
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "model": [secret],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    with pytest.raises(OpenAICompatibleNativeInvalidResponseError) as exc_info:
        backend.invoke(request=_request(backend))

    assert exc_info.value.usage is not None
    graph = _exception_graph_text(exc_info.value)
    assert secret not in graph
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_provider_identity_requires_exact_strings() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "model": " provider-model-with-padding ",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    with pytest.raises(
        OpenAICompatibleNativeInvalidResponseError,
        match="provider identity is invalid",
    ) as exc_info:
        backend.invoke(request=_request(backend))
    assert exc_info.value.usage is not None


def test_frozen_backend_and_serializer_fingerprints_are_checked_before_transport() -> None:
    backend = CapturingBackend({})
    request = _request(backend)

    with pytest.raises(OpenAICompatibleNativeFunctionError, match="backend fingerprint"):
        backend.invoke(request=request.model_copy(update={"backend_fingerprint": "0" * 64}))
    with pytest.raises(OpenAICompatibleNativeFunctionError, match="serializer fingerprint"):
        backend.invoke(request=request.model_copy(update={"serializer_fingerprint": "1" * 64}))
    with pytest.raises(OpenAICompatibleNativeFunctionError, match="parser fingerprint"):
        backend.invoke(request=request.model_copy(update={"parser_fingerprint": "2" * 64}))
    with pytest.raises(OpenAICompatibleNativeFunctionError, match="transport fingerprint"):
        backend.invoke(request=request.model_copy(update={"transport_fingerprint": "3" * 64}))
    assert backend.calls == []


def test_message_history_is_serialized_exactly_and_rejects_unknown_functions() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "final"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
    )
    history_call = NativeAssistantToolCall(
        call_id="call_prior",
        official_name="official_lookup",
        wire_name="official_lookup",
        arguments_json='{ "value" : 8 }',
    )
    messages = (
        FrozenNativeChatMessage(role="user", content=" lookup "),
        FrozenNativeChatMessage(role="assistant", content=None, tool_calls=(history_call,)),
        FrozenNativeChatMessage(
            role="tool",
            content=' {"result": 9} ',
            tool_call_id="call_prior",
        ),
    )
    request = _request(backend, messages=messages)

    backend.invoke(request=request)

    assert backend.calls[0][1]["messages"] == [
        {"role": "user", "content": " lookup "},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_prior",
                    "type": "function",
                    "function": {
                        "name": "official_lookup",
                        "arguments": '{ "value" : 8 }',
                    },
                }
            ],
        },
        {"role": "tool", "content": ' {"result": 9} ', "tool_call_id": "call_prior"},
    ]

    unknown = NativeAssistantToolCall(
        call_id="bad_prior",
        official_name="unknown",
        wire_name="unknown",
        arguments_json="{}",
    )
    with pytest.raises(ValidationError, match="outside the frozen tool mapping"):
        _request(
            backend,
            messages=(
                FrozenNativeChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=(unknown,),
                ),
            ),
        )


def _assistant_call_message(call_id: str = "call_prior") -> FrozenNativeChatMessage:
    return FrozenNativeChatMessage(
        role="assistant",
        content=None,
        tool_calls=(_history_call(call_id),),
    )


def _tool_result_message(
    call_id: str = "call_prior", *, name: str | None = None
) -> FrozenNativeChatMessage:
    return FrozenNativeChatMessage(
        role="tool",
        content='{"result":2}',
        tool_call_id=call_id,
        name=name,
    )


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ((_tool_result_message("orphan"),), "orphan tool result"),
        (
            (
                FrozenNativeChatMessage(role="user", content="start"),
                _assistant_call_message(),
            ),
            "ends with unresolved",
        ),
        (
            (
                _assistant_call_message(),
                FrozenNativeChatMessage(role="user", content="out of order"),
            ),
            "unresolved tool calls must be followed",
        ),
        (
            (
                _assistant_call_message(),
                _tool_result_message(),
                _tool_result_message(),
            ),
            "duplicate tool result",
        ),
        (
            (
                _assistant_call_message(),
                _tool_result_message(),
                _assistant_call_message(),
                _tool_result_message(),
            ),
            "duplicate tool-call ID",
        ),
        (
            (
                _assistant_call_message(),
                _tool_result_message(name="wrong_wire_name"),
            ),
            "differs from its pending wire name",
        ),
    ],
)
def test_transcript_call_result_state_machine_rejects_invalid_history(
    messages: tuple[FrozenNativeChatMessage, ...], error: str
) -> None:
    backend = CapturingBackend({})

    with pytest.raises(ValidationError, match=error):
        _request(backend, messages=messages)


def test_response_tool_call_ids_must_be_disjoint_from_resolved_history() -> None:
    backend = CapturingBackend(
        _tool_call_response(("call_prior", "official_lookup", '{"value":3}'))
    )
    request = _request(
        backend,
        messages=(
            _assistant_call_message(),
            _tool_result_message(),
            FrozenNativeChatMessage(role="user", content="continue"),
        ),
    )

    with pytest.raises(
        OpenAICompatibleNativeInvalidResponseError,
        match="overlap transcript history",
    ) as exc_info:
        backend.invoke(request=request)
    assert exc_info.value.usage is not None


def test_raw_transport_is_direct_only_no_redirect_and_bounded(monkeypatch) -> None:
    raw_response = _raw_json(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
    )
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, limit: int) -> bytes:
            captured["read_limit"] = limit
            return raw_response

    class FakeOpener:
        def open(self, request: object, *, timeout: float) -> FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

    def build_opener(*handlers: object) -> FakeOpener:
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-must-not-be-used.example:8888")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy-must-not-be-used.example:8888")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("default urlopen must never be used"),
    )
    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="transport-fixture-secret",
        user_agent="spiral-harness/raw-wire-test",
    )

    response = backend.invoke(request=_request(backend))

    assert response.assistant_text == "done"
    handlers = captured["handlers"]
    proxy = next(item for item in handlers if isinstance(item, urllib.request.ProxyHandler))
    redirect = next(
        item for item in handlers if isinstance(item, urllib.request.HTTPRedirectHandler)
    )
    assert proxy.proxies == {}
    assert redirect.redirect_request(None, None, 302, "secret", {}, "http://redirect") is None
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "http://litellm.example/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer transport-fixture-secret"
    assert request.get_header("User-agent") == "spiral-harness/raw-wire-test"
    assert request.data is not None
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["n"] == 1
    assert payload["parallel_tool_calls"] is True
    assert payload["stream"] is False
    assert captured["timeout"] == 9.0
    assert captured["read_limit"] == 1_048_577


def test_model_catalog_uses_direct_only_get_and_returns_exact_ids(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    raw_response = _raw_json(
        {
            "data": [
                {"id": "dashscope/qwen36-35b-a3b", "owned_by": "lab"},
                {"id": "MiniMax-M2.5"},
                {"id": "dashscope/qwen36-35b-a3b"},
            ]
        }
    )

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, limit: int) -> bytes:
            captured["read_limit"] = limit
            return raw_response

    class FakeOpener:
        def open(self, request: object, *, timeout: float) -> FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

    def build_opener(*handlers: object) -> FakeOpener:
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-must-not-be-used.example:8888")
    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1/",
        api_key="catalog-fixture-secret",
        user_agent="spiral-harness/catalog-test",
    )

    assert backend.list_models(timeout_seconds=4.5) == (
        "MiniMax-M2.5",
        "dashscope/qwen36-35b-a3b",
    )
    handlers = captured["handlers"]
    proxy = next(item for item in handlers if isinstance(item, urllib.request.ProxyHandler))
    redirect = next(
        item for item in handlers if isinstance(item, urllib.request.HTTPRedirectHandler)
    )
    assert proxy.proxies == {}
    assert redirect.redirect_request(None, None, 302, "secret", {}, "http://redirect") is None
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "http://litellm.example/v1/models"
    assert request.method == "GET"
    assert request.data is None
    assert request.get_header("Authorization") == "Bearer catalog-fixture-secret"
    assert request.get_header("User-agent") == "spiral-harness/catalog-test"
    assert captured["timeout"] == 4.5
    assert captured["read_limit"] == 1_048_577


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_model_catalog_rejects_invalid_timeouts(timeout: object) -> None:
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="fixture-secret",
    )
    with pytest.raises((TypeError, ValueError)):
        backend.list_models(timeout_seconds=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {}},
        {"data": ["bad"]},
        {"data": [{"id": " spaced "}]},
        {"data": [{"id": "bad\nmodel"}]},
    ],
)
def test_model_catalog_rejects_malformed_entries(monkeypatch, payload: object) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, _limit: int) -> bytes:
            return _raw_json(payload)

    class FakeOpener:
        def open(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: FakeOpener())
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="fixture-secret",
    )
    with pytest.raises(OpenAICompatibleNativeFunctionError):
        backend.list_models()


def test_http_errors_and_backend_repr_never_disclose_secrets(monkeypatch) -> None:
    backend = OpenAICompatibleNativeFunctionBackend.from_endpoint(
        base_url="http://litellm.example/v1",
        api_key="sk-request-secret",
    )
    request = _request(backend)

    class FailingOpener:
        def open(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise urllib.error.HTTPError(
                "http://litellm.example/v1/chat/completions?echo=sk-url-secret",
                401,
                "Authorization: Bearer sk-reason-secret",
                {"Authorization": "Bearer sk-header-secret"},
                BytesIO(b"Authorization: Bearer sk-body-secret"),
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: FailingOpener())

    with pytest.raises(OpenAICompatibleNativeFunctionError) as exc_info:
        backend.invoke(request=request)
    rendered = str(exc_info.value)
    assert "HTTP 401" in rendered
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    graph = _exception_graph_text(exc_info.value)
    for secret in (
        "sk-request-secret",
        "sk-url-secret",
        "sk-reason-secret",
        "sk-header-secret",
        "sk-body-secret",
    ):
        assert secret not in rendered
        assert secret not in graph
        assert secret not in repr(backend)


def test_redirect_http_response_is_rejected_without_a_followup(monkeypatch) -> None:
    calls: list[str] = []

    class RedirectingOpener:
        def open(self, request: urllib.request.Request, *, timeout: float) -> object:
            del timeout
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Location: http://redirect-secret.example",
                {"Location": "http://redirect-secret.example"},
                BytesIO(b"redirect-body-secret"),
            )

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *handlers: RedirectingOpener(),
    )
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="redirect-request-secret",
    )

    with pytest.raises(OpenAICompatibleNativeFunctionError, match="HTTP 302") as exc_info:
        backend.invoke(request=_request(backend))

    assert calls == ["http://litellm.example/v1/chat/completions"]
    graph = _exception_graph_text(exc_info.value)
    assert "redirect-secret" not in graph
    assert "redirect-body-secret" not in graph
    assert "redirect-request-secret" not in graph
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None


def test_generic_transport_exception_drops_provider_controlled_graph(monkeypatch) -> None:
    secret = "generic-transport-provider-secret"

    class FailingOpener:
        def open(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *handlers: FailingOpener(),
    )
    backend = OpenAICompatibleNativeFunctionBackend(
        base_url="http://litellm.example/v1",
        api_key="generic-request-secret",
    )

    with pytest.raises(OpenAICompatibleNativeFunctionError, match="transport failed") as exc_info:
        backend.invoke(request=_request(backend))

    graph = _exception_graph_text(exc_info.value)
    assert secret not in graph
    assert "generic-request-secret" not in graph
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None


def test_endpoint_credentials_and_queries_are_rejected_without_echo() -> None:
    secret = "sk-url-secret"
    with pytest.raises(ValueError) as exc_info:
        OpenAICompatibleNativeFunctionBackend.from_endpoint(
            base_url=f"http://user:{secret}@litellm.example/v1?api_key={secret}",
            api_key="safe-placeholder",
        )
    assert secret not in str(exc_info.value)


def test_header_control_characters_are_rejected_before_transport_without_echo() -> None:
    secret = "header-secret"
    with pytest.raises(ValueError) as key_error:
        OpenAICompatibleNativeFunctionBackend(
            base_url="http://litellm.example/v1",
            api_key=f"{secret}\nInjected: value",
        )
    with pytest.raises(ValueError) as agent_error:
        OpenAICompatibleNativeFunctionBackend(
            base_url="http://litellm.example/v1",
            api_key="placeholder",
            user_agent=f"agent\r\nAuthorization: {secret}",
        )
    assert secret not in str(key_error.value)
    assert secret not in str(agent_error.value)
