from __future__ import annotations

from dataclasses import dataclass

import pytest

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2ModelVisibleRequest,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.experiments.bfcl_v4_public_v2_native import (
    BfclV4PublicV2NativeMaterializationError,
    materialize_bfcl_v4_public_v2_native_request,
)


@dataclass(frozen=True)
class _Backend:
    fingerprint: str = "1" * 64
    serializer_fingerprint: str = "2" * 64
    parser_fingerprint: str = "3" * 64
    transport_fingerprint: str = "4" * 64


def _visible(*, prompt: str | None = None) -> BfclV4PublicV2ModelVisibleRequest:
    question = [[{"role": "user", "content": "Find weather, then notify me."}]]
    functions = [
        {
            "name": "weather.lookup",
            "description": "Look up weather.",
            "parameters": {
                "properties": {
                    "latitude": {
                        "type": "float",
                        "description": "Latitude.",
                    }
                },
                "required": ["latitude"],
            },
        },
        {
            "name": "notify",
            "description": "Send a notification.",
            "parameters": {"properties": {}},
        },
    ]
    return BfclV4PublicV2ModelVisibleRequest(
        model_route=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
        inference=BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
        provider_seed_u63=123,
        system_prompt=prompt,
        question_json=canonical_json(question),
        function_schemas_json=canonical_json(functions),
    )


def _spec(backend: _Backend | None = None) -> FrozenModelSpec:
    bound = backend or _Backend()
    return FrozenModelSpec(
        backend="litellm-openai-compatible-native",
        backend_fingerprint=bound.fingerprint,
        model=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
        revision="gateway-opaque-revision-2026-08-15",
        tokenizer="gateway-opaque-tokenizer",
        tokenizer_revision="gateway-opaque-tokenizer-revision-2026-08-15",
        runtime="spiral-native-openai-compatible@v2",
        inference=BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    )


@pytest.mark.parametrize("prompt", (None, "Use the frozen static strategy."))
def test_visible_request_projects_to_exact_native_messages_tools_and_seed(prompt) -> None:
    visible = _visible(prompt=prompt)
    backend = _Backend()
    request = materialize_bfcl_v4_public_v2_native_request(
        visible_request=visible,
        expected_visible_request_sha256=visible.fingerprint,
        spec=_spec(backend),
        backend=backend,
    )

    assert request.requested_model == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
    assert request.seed == 123
    assert request.inference == BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE
    assert request.harness_added_tools == ()
    assert tuple(tool.official_name for tool in request.task_required_tools) == (
        "weather.lookup",
        "notify",
    )
    assert tuple(tool.wire_name for tool in request.task_required_tools) == (
        "weather_lookup",
        "notify",
    )
    weather_schema = request.task_required_tools[0].wire_schema
    latitude = weather_schema["parameters"]["properties"]["latitude"]
    assert latitude["type"] == "number"
    assert latitude["format"] == "float"
    assert tuple(message.role for message in request.messages) == (
        ("user",) if prompt is None else ("system", "user")
    )
    assert tuple(message.content for message in request.messages)[-1] == (
        "Find weather, then notify me."
    )


def test_visible_identity_model_and_backend_mismatches_fail_closed() -> None:
    visible = _visible()
    backend = _Backend()
    with pytest.raises(BfclV4PublicV2NativeMaterializationError, match="wrapper identity"):
        materialize_bfcl_v4_public_v2_native_request(
            visible_request=visible,
            expected_visible_request_sha256="f" * 64,
            spec=_spec(backend),
            backend=backend,
        )

    other_backend = _Backend(fingerprint="9" * 64)
    with pytest.raises(BfclV4PublicV2NativeMaterializationError, match="backend fingerprint"):
        materialize_bfcl_v4_public_v2_native_request(
            visible_request=visible,
            expected_visible_request_sha256=visible.fingerprint,
            spec=_spec(backend),
            backend=other_backend,
        )


def test_bare_request_rejects_task_authored_system_message() -> None:
    base = _visible()
    question = [[{"role": "system", "content": "task prompt"}, {"role": "user", "content": "x"}]]
    visible = base.model_copy(update={"question_json": canonical_json(question)})
    with pytest.raises(BfclV4PublicV2NativeMaterializationError, match="native wire"):
        materialize_bfcl_v4_public_v2_native_request(
            visible_request=visible,
            expected_visible_request_sha256=visible.fingerprint,
            spec=_spec(),
            backend=_Backend(),
        )


def test_dot_substitution_collision_is_rejected_without_network() -> None:
    base = _visible()
    functions = [
        {
            "name": "a.b",
            "description": "first",
            "parameters": {"properties": {}},
        },
        {
            "name": "a_b",
            "description": "second",
            "parameters": {"properties": {}},
        },
    ]
    visible = base.model_copy(update={"function_schemas_json": canonical_json(functions)})
    with pytest.raises(BfclV4PublicV2NativeMaterializationError, match="native wire"):
        materialize_bfcl_v4_public_v2_native_request(
            visible_request=visible,
            expected_visible_request_sha256=visible.fingerprint,
            spec=_spec(),
            backend=_Backend(),
        )
