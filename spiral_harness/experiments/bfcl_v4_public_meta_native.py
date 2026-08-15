"""Pure native request materialization for BFCL V4 meta-controller calls.

The materializer stops at immutable provider input.  It accepts only a
strictly revalidated diagnosis or proposal prompt, pins that prompt to its
single controller submit tool, and never invokes the supplied backend.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_CANDIDATE_SUBMIT_TOOL,
    BFCL_V4_DIAGNOSER_SYSTEM_PROMPT,
    BFCL_V4_DIAGNOSIS_SUBMIT_TOOL,
    BFCL_V4_PROPOSER_SYSTEM_PROMPT,
)
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeFunctionCallRequest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_PROVIDER_SEED = 2**63 - 1

BfclV4PublicMetaPrompt = BfclV4DiagnosisPrompt | BfclV4ProposalPrompt


class BfclV4PublicMetaNativeMaterializationError(ValueError):
    """A BFCL meta-controller request could not be frozen exactly."""


@runtime_checkable
class BfclV4PublicMetaNativeBackendIdentity(Protocol):
    """Secret-free implementation identities exposed by a native backend."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def serializer_fingerprint(self) -> str: ...

    @property
    def parser_fingerprint(self) -> str: ...

    @property
    def transport_fingerprint(self) -> str: ...


def _prompt_contract(
    prompt: BfclV4PublicMetaPrompt,
) -> tuple[type[BfclV4DiagnosisPrompt] | type[BfclV4ProposalPrompt], FrozenNativeFunctionTool, str]:
    if type(prompt) is BfclV4DiagnosisPrompt:
        return (
            BfclV4DiagnosisPrompt,
            BFCL_V4_DIAGNOSIS_SUBMIT_TOOL,
            BFCL_V4_DIAGNOSER_SYSTEM_PROMPT,
        )
    if type(prompt) is BfclV4ProposalPrompt:
        return (
            BfclV4ProposalPrompt,
            BFCL_V4_CANDIDATE_SUBMIT_TOOL,
            BFCL_V4_PROPOSER_SYSTEM_PROMPT,
        )
    raise BfclV4PublicMetaNativeMaterializationError(
        "meta prompt must be an exact BFCL diagnosis or proposal prompt"
    )


def _checked_prompt(prompt: BfclV4PublicMetaPrompt) -> BfclV4PublicMetaPrompt:
    prompt_type, expected_tool, expected_system_prompt = _prompt_contract(prompt)
    try:
        source_fingerprint = canonical_sha256(prompt)
        checked = prompt_type.model_validate(
            prompt.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        checked_fingerprint = canonical_sha256(checked)
        checked_tool_fingerprint = canonical_sha256(checked.submit_tool)
    except (RecursionError, TypeError, UnicodeError, ValidationError, ValueError) as error:
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta prompt differs from its strict frozen contract"
        ) from error

    if source_fingerprint != checked_fingerprint or checked.fingerprint != checked_fingerprint:
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta prompt fingerprint changed during strict revalidation"
        )
    if (
        checked.submit_tool != expected_tool
        or checked.submit_tool_fingerprint != checked_tool_fingerprint
        or checked_tool_fingerprint != canonical_sha256(expected_tool)
    ):
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta prompt is not bound to its exact controller submit tool"
        )
    if checked.system_prompt != expected_system_prompt:
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta prompt differs from its fixed controller system prompt"
        )
    return checked


def _checked_spec(spec: FrozenModelSpec) -> FrozenModelSpec:
    if type(spec) is not FrozenModelSpec:
        raise BfclV4PublicMetaNativeMaterializationError(
            "model spec must be an exact FrozenModelSpec"
        )
    try:
        checked = FrozenModelSpec.model_validate(
            spec.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
    except (RecursionError, TypeError, UnicodeError, ValidationError, ValueError) as error:
        raise BfclV4PublicMetaNativeMaterializationError(
            "model spec differs from the frozen execution contract"
        ) from error
    if checked.fingerprint != spec.fingerprint:
        raise BfclV4PublicMetaNativeMaterializationError(
            "model spec fingerprint changed during strict revalidation"
        )
    return checked


def _backend_fingerprints(
    backend: BfclV4PublicMetaNativeBackendIdentity,
) -> tuple[str, str, str, str]:
    if not isinstance(backend, BfclV4PublicMetaNativeBackendIdentity):
        raise BfclV4PublicMetaNativeMaterializationError(
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
        raise BfclV4PublicMetaNativeMaterializationError(
            "backend native identities are unavailable"
        ) from error
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in values):
        raise BfclV4PublicMetaNativeMaterializationError(
            "backend native identities must be lowercase SHA-256 values"
        )
    return values


def _messages(prompt: BfclV4PublicMetaPrompt) -> tuple[FrozenNativeChatMessage, ...]:
    try:
        messages = (
            FrozenNativeChatMessage(role="system", content=prompt.system_prompt),
            FrozenNativeChatMessage(role="user", content=prompt.user_prompt),
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta prompt cannot be represented by strict system/user messages"
        ) from error
    if (
        tuple(message.role for message in messages) != ("system", "user")
        or tuple(message.content for message in messages)
        != (prompt.system_prompt, prompt.user_prompt)
        or any(
            message.name is not None or message.tool_call_id is not None or message.tool_calls
            for message in messages
        )
    ):
        raise BfclV4PublicMetaNativeMaterializationError(
            "meta message materialization drifted from the frozen prompt"
        )
    return messages


def materialize_bfcl_v4_public_meta_native_request(
    *,
    prompt: BfclV4PublicMetaPrompt,
    spec: FrozenModelSpec,
    backend: BfclV4PublicMetaNativeBackendIdentity,
    seed: int,
) -> NativeFunctionCallRequest:
    """Build one immutable diagnosis/proposal request without invoking a model."""

    checked_prompt = _checked_prompt(prompt)
    checked_spec = _checked_spec(spec)
    if type(seed) is not int or not 0 <= seed <= _MAX_PROVIDER_SEED:
        raise BfclV4PublicMetaNativeMaterializationError(
            "provider seed must be an unsigned 63-bit integer"
        )
    backend_fingerprint, serializer, parser, transport = _backend_fingerprints(backend)
    if backend_fingerprint != checked_spec.backend_fingerprint:
        raise BfclV4PublicMetaNativeMaterializationError(
            "native backend fingerprint differs from the frozen model spec"
        )

    try:
        request = NativeFunctionCallRequest(
            backend_fingerprint=backend_fingerprint,
            serializer_fingerprint=serializer,
            parser_fingerprint=parser,
            transport_fingerprint=transport,
            requested_model=checked_spec.model,
            messages=_messages(checked_prompt),
            task_required_tools=(checked_prompt.submit_tool,),
            harness_added_tools=(),
            seed=seed,
            inference=checked_spec.inference,
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicMetaNativeMaterializationError(
            "BFCL meta call cannot be represented by the frozen native request contract"
        ) from error

    expected_binding = (
        backend_fingerprint,
        serializer,
        parser,
        transport,
        checked_spec.model,
        checked_spec.inference,
        seed,
        (checked_prompt.submit_tool,),
        (),
    )
    actual_binding = (
        request.backend_fingerprint,
        request.serializer_fingerprint,
        request.parser_fingerprint,
        request.transport_fingerprint,
        request.requested_model,
        request.inference,
        request.seed,
        request.task_required_tools,
        request.harness_added_tools,
    )
    if actual_binding != expected_binding:
        raise BfclV4PublicMetaNativeMaterializationError(
            "materialized BFCL meta request drifted from its frozen execution binding"
        )
    return request


__all__ = [
    "BfclV4PublicMetaNativeBackendIdentity",
    "BfclV4PublicMetaNativeMaterializationError",
    "BfclV4PublicMetaPrompt",
    "materialize_bfcl_v4_public_meta_native_request",
]
