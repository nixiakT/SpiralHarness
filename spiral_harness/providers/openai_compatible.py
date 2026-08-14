"""OpenAI-compatible live chat backend for trusted fixed-model runs.

The backend intentionally depends only on the Python standard library.  It
targets OpenAI-compatible gateways such as LiteLLM while preserving the
score-free :class:`~spiral_harness.execution.model.FixedModelRunner` boundary:
the provider can return text and usage, but it never receives grader authority.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.execution.contracts import (
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    ModelRequest,
)
from spiral_harness.execution.pure_contracts import PureReferenceRequest


class OpenAICompatibleBackendError(RuntimeError):
    """Raised when an OpenAI-compatible gateway response is unusable."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleChatBackend:
    """Minimal ``/chat/completions`` adapter for OpenAI-compatible gateways."""

    base_url: str
    api_key: str
    fingerprint: str
    user_agent: str = "spiral-harness/openai-compatible-backend"

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.api_key, str) or not self.api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ValueError("fingerprint must be a non-empty string")
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("user_agent must be a non-empty string")

    @classmethod
    def from_endpoint(cls, *, base_url: str, api_key: str) -> OpenAICompatibleChatBackend:
        """Build a backend with a credential-free endpoint fingerprint."""

        normalized_base_url = normalize_openai_base_url(base_url)
        return cls(
            base_url=normalized_base_url,
            api_key=api_key,
            fingerprint=canonical_sha256(
                {
                    "schema": "spiral-harness/openai-compatible-backend/v1",
                    "base_url": normalized_base_url,
                    "endpoint": "/chat/completions",
                }
            ),
        )

    def invoke(self, *, spec: FrozenModelSpec, request: ModelRequest) -> BackendResponse:
        """Call ``/chat/completions`` and return score-free text plus usage."""

        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = ModelRequest.model_validate(request, strict=True)
        if checked_spec.backend_fingerprint != self.fingerprint:
            raise OpenAICompatibleBackendError("backend fingerprint differs from frozen spec")

        payload = {
            "model": checked_spec.model,
            "messages": [
                {"role": "system", "content": checked_request.system_prompt},
                {"role": "user", "content": checked_request.user_prompt},
            ],
            "temperature": checked_spec.inference.temperature,
            "top_p": checked_spec.inference.top_p,
            "max_tokens": checked_spec.inference.max_output_tokens,
            "seed": checked_request.seed,
        }
        if checked_spec.inference.stop_sequences:
            payload["stop"] = list(checked_spec.inference.stop_sequences)

        response = self._post_json(
            "/chat/completions",
            payload,
            timeout_seconds=checked_spec.inference.timeout_seconds,
        )
        _require_complete_chat_choice(response)
        output = _extract_chat_output(response)
        usage = _extract_usage(response)
        return BackendResponse(output=output, usage=usage, cost_usd=None)

    def invoke_pure(
        self,
        *,
        spec: FrozenModelSpec,
        request: PureReferenceRequest,
    ) -> BackendResponse:
        """Call the model with exactly one user message and no harness surface."""

        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = PureReferenceRequest.model_validate(request, strict=True)
        if checked_spec.backend_fingerprint != self.fingerprint:
            raise OpenAICompatibleBackendError("backend fingerprint differs from frozen spec")

        payload = {
            "model": checked_spec.model,
            "messages": [
                {
                    "role": checked_request.messages[0].role,
                    "content": checked_request.messages[0].content,
                }
            ],
            "temperature": checked_spec.inference.temperature,
            "top_p": checked_spec.inference.top_p,
            "max_tokens": checked_spec.inference.max_output_tokens,
            "seed": checked_request.rollout_seed,
        }
        if checked_spec.inference.stop_sequences:
            payload["stop"] = list(checked_spec.inference.stop_sequences)

        response = self._post_json(
            "/chat/completions",
            payload,
            timeout_seconds=checked_spec.inference.timeout_seconds,
        )
        _require_complete_chat_choice(response)
        return BackendResponse(
            output=_extract_chat_output(response),
            usage=_extract_usage(response),
            cost_usd=None,
        )

    def list_models(self, *, timeout_seconds: float = 15.0) -> tuple[str, ...]:
        """Return the gateway's advertised model IDs without invoking a model.

        This is an availability probe only.  It sends one authenticated
        ``GET /models`` request and deliberately exposes neither provider
        metadata nor credentials in the returned catalog.
        """

        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a number")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        response = self._get_json("/models", timeout_seconds=float(timeout_seconds))
        data = response.get("data")
        if not isinstance(data, list):
            raise OpenAICompatibleBackendError("model catalog response has no data list")
        model_ids: list[str] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise OpenAICompatibleBackendError(f"model catalog entry {index} must be an object")
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise OpenAICompatibleBackendError(
                    f"model catalog entry {index} has no non-empty id"
                )
            model_ids.append(model_id.strip())
        return tuple(sorted(set(model_ids)))

    def _get_json(self, path: str, *, timeout_seconds: float) -> dict[str, Any]:
        request = urllib.request.Request(
            normalize_openai_base_url(self.base_url) + path,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        return self._open_json(request, timeout_seconds=timeout_seconds)

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            normalize_openai_base_url(self.base_url) + path,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        return self._open_json(request, timeout_seconds=timeout_seconds)

    @staticmethod
    def _open_json(
        request: urllib.request.Request,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
        except TimeoutError:
            raise
        except urllib.error.HTTPError as exc:
            detail = _redacted_http_error_detail(exc)
            raise OpenAICompatibleBackendError(detail) from exc
        except urllib.error.URLError as exc:
            raise OpenAICompatibleBackendError("OpenAI-compatible backend URL error") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenAICompatibleBackendError(
                "OpenAI-compatible backend returned non-JSON content"
            ) from exc
        if type(decoded) is not dict:
            raise OpenAICompatibleBackendError(
                "OpenAI-compatible backend response must be a JSON object"
            )
        return decoded


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize one OpenAI-compatible ``/v1`` base URL without adding credentials."""

    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    value = base_url.strip()
    if not value:
        raise ValueError("base_url must not be empty")
    return value.rstrip("/")


def _extract_chat_output(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatibleBackendError("chat completion response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenAICompatibleBackendError("chat completion choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenAICompatibleBackendError("chat completion choice has no message object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
    raise OpenAICompatibleBackendError("chat completion message content is not text")


def _require_complete_chat_choice(response: dict[str, Any]) -> None:
    """Reject provider-declared truncation before text can be graded as completed."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        raise OpenAICompatibleBackendError("chat completion was truncated at the token limit")


def _extract_usage(response: dict[str, Any]) -> BackendTokenUsage:
    usage = response.get("usage")
    if usage is None:
        raise OpenAICompatibleBackendError("chat completion response is missing token usage")
    if not isinstance(usage, dict):
        raise OpenAICompatibleBackendError("chat completion usage must be an object")
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
        raise OpenAICompatibleBackendError("chat completion prompt token usage must be an integer")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
        raise OpenAICompatibleBackendError(
            "chat completion completion token usage must be an integer"
        )
    return BackendTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _redacted_http_error_detail(exc: urllib.error.HTTPError) -> str:
    # Gateways and reverse proxies occasionally echo request material in error
    # bodies.  Do not include any response body, URL query, or headers in an
    # exception that may be persisted in experiment logs.
    return f"OpenAI-compatible backend HTTP {exc.code}"


__all__ = [
    "OpenAICompatibleBackendError",
    "OpenAICompatibleChatBackend",
    "normalize_openai_base_url",
]
