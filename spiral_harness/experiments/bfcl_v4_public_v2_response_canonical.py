"""Label-free canonical response projection for BFCL v2 tool calls."""

from __future__ import annotations

import json
from typing import Annotated, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr
from spiral_harness.providers.openai_native_contracts import NativeFunctionCallResponse


class BfclV4PublicV2CanonicalResponseError(ValueError):
    """A native result cannot enter modal aggregation or trusted grading."""


class BfclV4PublicV2CanonicalCall(ImmutableModel):
    """One call after removing provider call IDs and reversible wire aliases."""

    function_name: NonEmptyStr
    arguments_json: NonEmptyStr

    @model_validator(mode="after")
    def _canonical_arguments(self) -> Self:
        try:
            value = json.loads(self.arguments_json)
        except json.JSONDecodeError as error:
            raise ValueError("canonical call arguments are not JSON") from error
        if type(value) is not dict or canonical_json(value) != self.arguments_json:
            raise ValueError("canonical call arguments must be one canonical JSON object")
        return self


class BfclV4PublicV2CanonicalResponse(ImmutableModel):
    """Ordered official calls used as the exact PURE@B vote value."""

    calls: Annotated[tuple[BfclV4PublicV2CanonicalCall, ...], Field(max_length=64)]

    @property
    def canonical_json(self) -> str:
        return canonical_json(
            [
                {
                    "arguments": json.loads(call.arguments_json),
                    "function_name": call.function_name,
                }
                for call in self.calls
            ]
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def canonicalize_bfcl_v4_public_v2_native_response(
    response: NativeFunctionCallResponse,
) -> str:
    """Project one strict native response to ordered official calls only."""

    try:
        checked = NativeFunctionCallResponse.model_validate(response, strict=True)
        projected = BfclV4PublicV2CanonicalResponse(
            calls=tuple(
                BfclV4PublicV2CanonicalCall(
                    function_name=call.official_name,
                    arguments_json=call.canonical_arguments_json,
                )
                for call in checked.tool_calls
            )
        )
    except Exception:
        raise BfclV4PublicV2CanonicalResponseError(
            "native response failed canonical BFCL v2 projection"
        ) from None
    return projected.canonical_json


def parse_bfcl_v4_public_v2_canonical_response(
    value: str,
) -> BfclV4PublicV2CanonicalResponse:
    """Strictly invert the canonical vote without accepting alternate encodings."""

    if type(value) is not str:
        raise BfclV4PublicV2CanonicalResponseError("canonical response must be exact text")
    try:
        decoded = json.loads(value)
        if type(decoded) is not list or canonical_json(decoded) != value:
            raise ValueError("non-canonical response")
        calls = []
        for item in decoded:
            if type(item) is not dict or set(item) != {"arguments", "function_name"}:
                raise ValueError("wrong call schema")
            if type(item["arguments"]) is not dict or type(item["function_name"]) is not str:
                raise ValueError("wrong call values")
            calls.append(
                BfclV4PublicV2CanonicalCall(
                    function_name=item["function_name"],
                    arguments_json=canonical_json(item["arguments"]),
                )
            )
        parsed = BfclV4PublicV2CanonicalResponse(calls=tuple(calls))
    except Exception:
        raise BfclV4PublicV2CanonicalResponseError(
            "canonical BFCL v2 response is malformed"
        ) from None
    if parsed.canonical_json != value:
        raise BfclV4PublicV2CanonicalResponseError(
            "canonical BFCL v2 response changed during round trip"
        )
    return parsed


__all__ = [
    name
    for name in globals()
    if name.startswith("Bfcl")
    or name.startswith("canonicalize_bfcl")
    or name.startswith("parse_bfcl")
]
