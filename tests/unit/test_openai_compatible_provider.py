from __future__ import annotations

import urllib.error
from io import BytesIO

import pytest

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE, ArtifactRef
from spiral_harness.execution.contracts import (
    CandidateTask,
    FrozenModelSpec,
    InferenceConfig,
    ModelRequest,
    ResolvedHarness,
)
from spiral_harness.execution.pure_contracts import materialize_pure_request
from spiral_harness.providers.openai_compatible import (
    OpenAICompatibleBackendError,
    OpenAICompatibleChatBackend,
    OpenAICompatibleInvalidResponseError,
    normalize_openai_base_url,
)


def spec_for(backend: OpenAICompatibleChatBackend) -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="openai-compatible-chat",
        backend_fingerprint=backend.fingerprint,
        model="dashscope/qwen36-35b-a3b",
        revision="hosted-snapshot-2026-08-13",
        tokenizer="provider-reported",
        tokenizer_revision="hosted-snapshot-2026-08-13",
        runtime="spiral-harness-live-smoke-py3.12@2026-08-13",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
            timeout_seconds=5.0,
            stop_sequences=("END",),
        ),
    )


def request() -> ModelRequest:
    harness = ResolvedHarness.from_prompt(
        harness_ref=ArtifactRef(
            sha256="a" * 64,
            size=0,
            media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        ),
        system_prompt="Solve carefully.",
    )
    return ModelRequest(
        task_id="gsm8k-example",
        harness_ref=harness.harness_ref,
        base_system_prompt=harness.base_system_prompt,
        base_system_prompt_sha256=harness.base_system_prompt_sha256,
        skill_disclosure=None,
        system_prompt=harness.system_prompt,
        resolved_prompt_sha256=harness.resolved_prompt_sha256,
        user_prompt="What is 2+3?",
        seed=0,
    )


class CapturingBackend(OpenAICompatibleChatBackend):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__(
            base_url="http://litellm.example/v1",
            api_key="secret",
            fingerprint=canonical_sha256(
                {
                    "schema": "spiral-harness/openai-compatible-backend/v1",
                    "base_url": "http://litellm.example/v1",
                    "endpoint": "/chat/completions",
                }
            ),
        )
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "calls", [])

    def _post_json(
        self,
        path: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append((path, payload, timeout_seconds))
        return self.response

    def _get_json(self, path: str, *, timeout_seconds: float) -> dict[str, object]:
        self.calls.append((path, None, timeout_seconds))
        return self.response


def test_base_url_normalization_is_credential_free_and_stable() -> None:
    assert normalize_openai_base_url(" http://10.0.0.1:8010/v1/ ") == "http://10.0.0.1:8010/v1"

    backend = OpenAICompatibleChatBackend.from_endpoint(
        base_url="http://10.0.0.1:8010/v1/",
        api_key="sk-secret",
    )
    assert backend.base_url == "http://10.0.0.1:8010/v1"
    assert "sk-secret" not in backend.fingerprint


def test_chat_backend_posts_frozen_chat_completion_payload_and_usage() -> None:
    backend = CapturingBackend(
        {
            "choices": [{"message": {"content": "Reasoning.\n#### 5"}}],
            "model": "qwen36-35b-a3b-served-revision",
            "system_fingerprint": "fp_provider_snapshot_123",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
    )

    response = backend.invoke(spec=spec_for(backend), request=request())

    assert response.output == "Reasoning.\n#### 5"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    identity = response.provider_identity_observation
    assert identity is not None
    assert identity.trust_level == "provider-declared"
    assert identity.requested_model == "dashscope/qwen36-35b-a3b"
    assert identity.response_model == "qwen36-35b-a3b-served-revision"
    assert identity.system_fingerprint == "fp_provider_snapshot_123"
    assert identity.backend_fingerprint == backend.fingerprint
    assert identity.fingerprint == canonical_sha256(identity)
    path, payload, timeout = backend.calls[0]
    assert path == "/chat/completions"
    assert timeout == 5.0
    assert payload["model"] == "dashscope/qwen36-35b-a3b"
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 64
    assert payload["seed"] == 0
    assert payload["stop"] == ["END"]
    assert payload["messages"] == [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 2+3?"},
    ]


def test_pure_backend_posts_only_one_user_message_and_no_harness_fields() -> None:
    backend = CapturingBackend(
        {
            "choices": [{"message": {"content": "#### 5"}}],
            "model": None,
            "system_fingerprint": "fp_pure_provider_snapshot_456",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
    )
    pure_request = materialize_pure_request(
        CandidateTask(task_id="gsm8k-example", question="What is 2+3?"),
        reference_id="a" * 64,
        rollout_seed=19,
    )

    response = backend.invoke_pure(spec=spec_for(backend), request=pure_request)

    assert response.output == "#### 5"
    identity = response.provider_identity_observation
    assert identity is not None
    assert identity.requested_model == "dashscope/qwen36-35b-a3b"
    assert identity.response_model is None
    assert identity.system_fingerprint == "fp_pure_provider_snapshot_456"
    assert identity.backend_fingerprint == backend.fingerprint
    _, payload, _ = backend.calls[0]
    assert payload["messages"] == [{"role": "user", "content": "What is 2+3?"}]
    assert payload["seed"] == 19
    assert set(payload) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop",
    }


def test_chat_backend_accepts_openai_responses_style_text_parts() -> None:
    backend = CapturingBackend(
        {
            "choices": [{"message": {"content": [{"type": "text", "text": "#### 5"}]}}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )

    response = backend.invoke(spec=spec_for(backend), request=request())

    assert response.output == "#### 5"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert response.provider_identity_observation is None


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("model", False),
        ("model", 7),
        ("model", ""),
        ("system_fingerprint", []),
        ("system_fingerprint", "   "),
    ],
)
def test_chat_and_pure_backends_fail_closed_on_malformed_identity_metadata(
    pure: bool,
    field_name: str,
    invalid_value: object,
) -> None:
    backend = CapturingBackend(
        {
            "choices": [{"message": {"content": "#### 5"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            field_name: invalid_value,
        }
    )
    with pytest.raises(OpenAICompatibleInvalidResponseError, match=field_name) as exc_info:
        if pure:
            backend.invoke_pure(
                spec=spec_for(backend),
                request=materialize_pure_request(
                    CandidateTask(task_id="gsm8k-example", question="What is 2+3?"),
                    reference_id="a" * 64,
                    rollout_seed=19,
                ),
            )
        else:
            backend.invoke(spec=spec_for(backend), request=request())
    assert isinstance(exc_info.value, OpenAICompatibleBackendError)
    assert exc_info.value.usage is not None
    assert exc_info.value.usage.total_tokens == 5


@pytest.mark.parametrize("pure", [False, True])
def test_missing_or_null_provider_identity_fields_remain_backward_compatible(
    pure: bool,
) -> None:
    backend = CapturingBackend(
        {
            "choices": [{"message": {"content": "#### 5"}}],
            "model": None,
            "system_fingerprint": None,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )

    if pure:
        provider_response = backend.invoke_pure(
            spec=spec_for(backend),
            request=materialize_pure_request(
                CandidateTask(task_id="gsm8k-example", question="What is 2+3?"),
                reference_id="a" * 64,
                rollout_seed=19,
            ),
        )
    else:
        provider_response = backend.invoke(spec=spec_for(backend), request=request())

    assert provider_response.provider_identity_observation is None


def test_chat_backend_rejects_malformed_provider_responses() -> None:
    backend = CapturingBackend({"choices": []})

    with pytest.raises(OpenAICompatibleBackendError, match="no choices"):
        backend.invoke(spec=spec_for(backend), request=request())


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 1},
        {"completion_tokens": 1},
    ],
)
def test_chat_backend_rejects_missing_or_incomplete_usage(
    usage: dict[str, int] | None,
) -> None:
    response: dict[str, object] = {"choices": [{"message": {"content": "#### 5"}}]}
    if usage is not None:
        response["usage"] = usage
    backend = CapturingBackend(response)

    with pytest.raises(OpenAICompatibleBackendError, match=r"usage|token usage"):
        backend.invoke(spec=spec_for(backend), request=request())


def test_chat_backend_rejects_token_limit_truncation_even_when_text_exists() -> None:
    backend = CapturingBackend(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "unfinished reasoning that mentions (A)"},
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 64},
        }
    )

    with pytest.raises(OpenAICompatibleBackendError, match="truncated"):
        backend.invoke(spec=spec_for(backend), request=request())


def test_model_catalog_is_an_availability_only_sorted_unique_id_list() -> None:
    backend = CapturingBackend(
        {
            "data": [
                {"id": "openai/gpt-test", "owned_by": "ignored"},
                {"id": "dashscope/qwen-test"},
                {"id": "openai/gpt-test"},
            ]
        }
    )

    assert backend.list_models(timeout_seconds=7) == (
        "dashscope/qwen-test",
        "openai/gpt-test",
    )
    assert backend.calls == [("/models", None, 7.0)]


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "no data list"),
        ({"data": ["bad"]}, "entry 0 must be an object"),
        ({"data": [{}]}, "entry 0 has no non-empty id"),
    ],
)
def test_model_catalog_rejects_malformed_responses(
    response: dict[str, object], message: str
) -> None:
    backend = CapturingBackend(response)

    with pytest.raises(OpenAICompatibleBackendError, match=message):
        backend.list_models()


def test_http_error_detail_does_not_include_request_headers(monkeypatch) -> None:
    backend = OpenAICompatibleChatBackend.from_endpoint(
        base_url="http://litellm.example/v1",
        api_key="sk-secret",
    )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise urllib.error.HTTPError(
            "http://litellm.example/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            BytesIO(b"gateway echoed Authorization: Bearer sk-do-not-log-this"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)

    with pytest.raises(OpenAICompatibleBackendError) as exc_info:
        backend.invoke(spec=spec_for(backend), request=request())
    assert "sk-secret" not in str(exc_info.value)
    assert "sk-do-not-log-this" not in str(exc_info.value)
    assert "HTTP 401" in str(exc_info.value)
