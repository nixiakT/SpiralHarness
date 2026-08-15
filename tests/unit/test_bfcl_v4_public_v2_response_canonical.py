from __future__ import annotations

import pytest

from spiral_harness.execution.contracts import BackendTokenUsage
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    BfclV4PublicV2CanonicalResponseError,
    canonicalize_bfcl_v4_public_v2_native_response,
    parse_bfcl_v4_public_v2_canonical_response,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallResponse,
)


def _response(*, call_id: str = "provider-call-1", wire_name: str = "wire_0"):
    return NativeFunctionCallResponse(
        request_fingerprint="1" * 64,
        serializer_fingerprint="2" * 64,
        parser_fingerprint="3" * 64,
        transport_fingerprint="4" * 64,
        tools_fingerprint="5" * 64,
        tool_calls=(
            NativeAssistantToolCall(
                call_id=call_id,
                official_name="weather.lookup",
                wire_name=wire_name,
                arguments_json='{"unit":"C","city":"Hangzhou"}',
            ),
            NativeAssistantToolCall(
                call_id=f"{call_id}-second",
                official_name="notify",
                wire_name="wire_1",
                arguments_json='{"urgent":true}',
            ),
        ),
        assistant_text=None,
        finish_reason="tool_calls",
        usage=BackendTokenUsage(input_tokens=100, output_tokens=20),
    )


def test_projection_is_ordered_canonical_and_ignores_provider_local_ids() -> None:
    left = canonicalize_bfcl_v4_public_v2_native_response(_response())
    right = canonicalize_bfcl_v4_public_v2_native_response(
        _response(call_id="other-provider-id", wire_name="another_wire")
    )

    assert left == right
    assert left == (
        '[{"arguments":{"city":"Hangzhou","unit":"C"},"function_name":"weather.lookup"},'
        '{"arguments":{"urgent":true},"function_name":"notify"}]'
    )
    parsed = parse_bfcl_v4_public_v2_canonical_response(left)
    assert tuple(call.function_name for call in parsed.calls) == (
        "weather.lookup",
        "notify",
    )
    assert parsed.canonical_json == left


def test_no_call_response_is_one_canonical_empty_vote() -> None:
    response = NativeFunctionCallResponse(
        request_fingerprint="1" * 64,
        serializer_fingerprint="2" * 64,
        parser_fingerprint="3" * 64,
        transport_fingerprint="4" * 64,
        tools_fingerprint="5" * 64,
        tool_calls=(),
        assistant_text="I cannot call a function.",
        finish_reason="stop",
        usage=BackendTokenUsage(input_tokens=100, output_tokens=20),
    )
    assert canonicalize_bfcl_v4_public_v2_native_response(response) == "[]"


@pytest.mark.parametrize(
    "value",
    [
        ' [{"arguments":{},"function_name":"x"}]',
        '[{"function_name":"x","arguments":{}}]',
        '[{"arguments":[],"function_name":"x"}]',
        '[{"arguments":{},"function_name":"x","extra":1}]',
        '{"arguments":{},"function_name":"x"}',
        "not-json",
    ],
)
def test_alternate_or_malformed_encodings_are_rejected(value: str) -> None:
    with pytest.raises(BfclV4PublicV2CanonicalResponseError):
        parse_bfcl_v4_public_v2_canonical_response(value)
