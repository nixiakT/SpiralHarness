"""Deterministic provider-free replay for native function-call requests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.execution.model import BackendFingerprintMismatchError, ReplayMissError
from spiral_harness.providers.openai_native_contracts import (
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _ReplayBinding:
    backend: str
    serializer: str
    parser: str
    transport: str


def native_function_replay_key(request: NativeFunctionCallRequest) -> str:
    """Key replay data by the exact provider-visible native request."""

    checked = NativeFunctionCallRequest.model_validate(request, strict=True)
    return canonical_sha256(
        {
            "schema": "spiral-harness/native-function-replay-key/v1",
            "request_sha256": checked.fingerprint,
        }
    )


class NativeFunctionReplayBackend:
    """Provider-free native backend with all four implementation identities."""

    def __init__(
        self,
        *,
        fingerprint: str,
        serializer_fingerprint: str,
        parser_fingerprint: str,
        transport_fingerprint: str,
        responses: Mapping[str, NativeFunctionCallResponse] | None = None,
        default_response: NativeFunctionCallResponse | None = None,
    ) -> None:
        values = {
            "fingerprint": fingerprint,
            "serializer_fingerprint": serializer_fingerprint,
            "parser_fingerprint": parser_fingerprint,
            "transport_fingerprint": transport_fingerprint,
        }
        for label, value in values.items():
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{label} must be a canonical SHA-256 value")
        checked: dict[str, NativeFunctionCallResponse] = {}
        for key, response in (responses or {}).items():
            if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
                raise ValueError("replay response keys must be canonical SHA-256 values")
            checked[key] = NativeFunctionCallResponse.model_validate(response, strict=True)
        self._binding = _ReplayBinding(
            backend=fingerprint,
            serializer=serializer_fingerprint,
            parser=parser_fingerprint,
            transport=transport_fingerprint,
        )
        self._responses = checked
        self._default_response = (
            None
            if default_response is None
            else NativeFunctionCallResponse.model_validate(default_response, strict=True)
        )
        self._calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return self._binding.backend

    @property
    def serializer_fingerprint(self) -> str:
        return self._binding.serializer

    @property
    def parser_fingerprint(self) -> str:
        return self._binding.parser

    @property
    def transport_fingerprint(self) -> str:
        return self._binding.transport

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        checked = NativeFunctionCallRequest.model_validate(request, strict=True)
        expected = _ReplayBinding(
            backend=checked.backend_fingerprint,
            serializer=checked.serializer_fingerprint,
            parser=checked.parser_fingerprint,
            transport=checked.transport_fingerprint,
        )
        if expected != self._binding:
            raise BackendFingerprintMismatchError(
                "replay backend identities differ from the exact native request"
            )
        key = native_function_replay_key(checked)
        self._calls.append(key)
        response = self._responses.get(key, self._default_response)
        if response is None:
            raise ReplayMissError(f"no replay response for frozen native request {key}")
        return NativeFunctionCallResponse.model_validate(response, strict=True)


__all__ = ["NativeFunctionReplayBackend", "native_function_replay_key"]
