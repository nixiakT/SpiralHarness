"""Single-attempt LiteLLM transport for opaque BFCL v2 semantic reviews."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    validate_bfcl_v4_public_v2_semantic_reviewer_response,
    verify_bfcl_v4_public_v2_semantic_distribution_integrity,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts import (
    BfclV4PublicV2SemanticReviewerAssignment,
    BfclV4PublicV2SemanticReviewerDeclaration,
    BfclV4PublicV2SemanticReviewerKind,
    BfclV4PublicV2SemanticReviewerPacket,
    BfclV4PublicV2SemanticReviewerResponse,
    BfclV4PublicV2SemanticTrustedMapping,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticReviewExecutionAttestation,
    BfclV4PublicV2SemanticReviewSessionEvidence,
)
from spiral_harness.core.canonical import canonical_json, canonical_json_bytes, canonical_sha256

BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MODEL_REVISION = "gateway-opaque-revision-unavailable@2026-08-15"
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MAX_OUTPUT_TOKENS = 12_000
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MAX_RESPONSE_BYTES = 4_194_304
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_USER_AGENT = "spiral-harness/bfcl-v2-semantic-review-v1"
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT = (
    "You are an independent semantic-isolation reviewer for a public function-calling "
    "benchmark.\n\n"
    "You receive one secret-permuted packet of exactly 40 public items. Each item has only "
    "an opaque ID, a canonical user question, and its function schemas. You receive no task "
    "IDs, source paths, split labels, possible answers, scores, benchmark-model outputs, or "
    "harness outputs.\n\n"
    "Assign a semantic-family label to every item. Two items must have the same label if, "
    "after abstracting incidental names, entities, constants, and wording, they require "
    "materially the same user intent, tool-selection logic, argument structure, "
    "multiplicity/order behavior, and composition pattern. Do not merge items merely because "
    "they share a broad domain or API vocabulary. Do merge plausible paraphrases or template "
    "variants whose solution strategy and required function behavior transfer directly. "
    "Examine all pairs conservatively; rationale text must state the decisive intent and "
    "tool-structure evidence.\n\n"
    "Return one JSON object and no markdown or prose outside it. It must have exactly one key, "
    '"assignments". Its value must be an array in the exact packet order. Every array element '
    'must have exactly these string keys: "opaque_item_id", "semantic_family_label", and '
    '"rationale". Copy every opaque ID exactly once. Labels are local arbitrary cluster names; '
    "equal labels mean the same family and different labels mean different families."
)
BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256 = hashlib.sha256(
    BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT.encode("utf-8")
).hexdigest()

_RawTransport = Callable[[str, str, str, dict[str, Any], float], bytes]


class BfclV4PublicV2SemanticReviewRole(StrEnum):
    """Frozen role within double review and optional arbitration."""

    PRIMARY_ONE = "primary-one"
    PRIMARY_TWO = "primary-two"
    ADJUDICATOR = "adjudicator"


class BfclV4PublicV2SemanticReviewLiveError(RuntimeError):
    """Sanitized single-attempt review failure."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


def _normalized_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("semantic review base URL must be non-empty")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("semantic review base URL must be a credential-free HTTP(S) URL")
    return normalized


def _post_raw_http(
    url: str,
    api_key: str,
    user_agent: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=canonical_json_bytes(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read(BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise BfclV4PublicV2SemanticReviewLiveError(
            f"semantic review provider returned HTTP {error.code}"
        ) from None
    except Exception:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review provider transport failed"
        ) from None
    if len(raw) > BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MAX_RESPONSE_BYTES:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review provider response exceeded byte limit"
        )
    return raw


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    if type(raw) is not bytes:
        raise BfclV4PublicV2SemanticReviewLiveError(f"{label} is not bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        del value
        raise ValueError("non-finite number")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise BfclV4PublicV2SemanticReviewLiveError(f"{label} is not strict JSON") from None


def build_bfcl_v4_public_v2_semantic_review_request(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    *,
    model: str,
    generation_seed_u64: int,
) -> dict[str, Any]:
    """Build the exact label-free request body without a credential or network call."""

    packet = BfclV4PublicV2SemanticReviewerPacket.model_validate(
        reviewer_packet,
        strict=True,
    )
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("semantic reviewer model must be a normalized non-empty string")
    if (
        not isinstance(generation_seed_u64, int)
        or isinstance(generation_seed_u64, bool)
        or not 0 <= generation_seed_u64 <= 2**64 - 1
    ):
        raise ValueError("semantic reviewer seed must be an unsigned 64-bit integer")
    packet_payload = {
        "protocol": "spiral-bfcl-v4-public-v2-semantic-review/v1",
        "reviewer_packet": packet.model_dump(mode="json"),
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT},
            {"role": "user", "content": canonical_json(packet_payload)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MAX_OUTPUT_TOKENS,
        "seed": generation_seed_u64,
        "response_format": {"type": "json_object"},
    }


def _parse_usage(value: object) -> tuple[int, int, int]:
    if type(value) is not dict:
        raise BfclV4PublicV2SemanticReviewLiveError("semantic review response has no token usage")
    input_tokens = value.get("prompt_tokens", value.get("input_tokens"))
    output_tokens = value.get("completion_tokens", value.get("output_tokens"))
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review response has invalid token usage"
        )
    total = value.get("total_tokens", input_tokens + output_tokens)
    if type(total) is not int or total != input_tokens + output_tokens:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review response has inconsistent token usage"
        )
    return input_tokens, output_tokens, total


def _parse_assignments(
    content: str,
    packet: BfclV4PublicV2SemanticReviewerPacket,
) -> tuple[BfclV4PublicV2SemanticReviewerAssignment, ...]:
    decoded = _strict_json_bytes(content.encode("utf-8"), label="semantic reviewer content")
    if type(decoded) is not dict or set(decoded) != {"assignments"}:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic reviewer content has the wrong top-level schema"
        )
    rows = decoded["assignments"]
    if type(rows) is not list or len(rows) != 40:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic reviewer content lacks exact 40-item coverage"
        )
    assignments: list[BfclV4PublicV2SemanticReviewerAssignment] = []
    try:
        for row in rows:
            if type(row) is not dict or set(row) != {
                "opaque_item_id",
                "semantic_family_label",
                "rationale",
            }:
                raise ValueError("assignment schema")
            assignments.append(BfclV4PublicV2SemanticReviewerAssignment(**row))
    except (TypeError, ValueError, ValidationError):
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic reviewer content has an invalid assignment"
        ) from None
    expected_ids = tuple(item.opaque_item_id for item in packet.items)
    if tuple(item.opaque_item_id for item in assignments) != expected_ids:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic reviewer content changed packet order or opaque IDs"
        )
    return tuple(assignments)


def _parse_provider_response(
    raw: bytes,
    packet: BfclV4PublicV2SemanticReviewerPacket,
    *,
    requested_model: str,
) -> tuple[
    str,
    str,
    str | None,
    tuple[int, int, int],
    tuple[BfclV4PublicV2SemanticReviewerAssignment, ...],
]:
    decoded = _strict_json_bytes(raw, label="semantic review provider response")
    if type(decoded) is not dict:
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review provider response is not an object"
        )
    response_id = decoded.get("id")
    response_model = decoded.get("model")
    system_fingerprint = decoded.get("system_fingerprint")
    choices = decoded.get("choices")
    if (
        not isinstance(response_id, str)
        or not response_id.strip()
        or response_id != response_id.strip()
        or response_model != requested_model
        or (system_fingerprint is not None and not isinstance(system_fingerprint, str))
        or type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review provider identity or completion status is invalid"
        )
    message = choices[0].get("message")
    if type(message) is not dict or not isinstance(message.get("content"), str):
        raise BfclV4PublicV2SemanticReviewLiveError(
            "semantic review provider returned no text content"
        )
    usage = _parse_usage(decoded.get("usage"))
    assignments = _parse_assignments(message["content"], packet)
    return response_id, response_model, system_fingerprint, usage, assignments


def _declared_route_identity(
    *,
    provider_origin_sha256: str,
    catalog_observation_sha256: str,
    model: str,
) -> str:
    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-opaque-review-model-route/v1",
            "provider_origin_sha256": provider_origin_sha256,
            "catalog_observation_sha256": catalog_observation_sha256,
            "model": model,
            "revision": BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MODEL_REVISION,
            "actual_weights_cryptographically_verified": False,
        }
    )


def run_bfcl_v4_public_v2_semantic_review(
    reviewer_packet: BfclV4PublicV2SemanticReviewerPacket,
    trusted_mapping: BfclV4PublicV2SemanticTrustedMapping,
    *,
    runtime_hmac_secret: bytes,
    base_url: str,
    api_key: str,
    model: str,
    role: BfclV4PublicV2SemanticReviewRole,
    generation_seed_u64: int,
    catalog_observation_sha256: str,
    timeout_seconds: float = 300.0,
    transport: _RawTransport = _post_raw_http,
) -> BfclV4PublicV2SemanticReviewSessionEvidence:
    """Perform exactly one direct provider attempt and return credential-free evidence."""

    packet = BfclV4PublicV2SemanticReviewerPacket.model_validate(
        reviewer_packet,
        strict=True,
    )
    mapping = BfclV4PublicV2SemanticTrustedMapping.model_validate(
        trusted_mapping,
        strict=True,
    )
    checked_role = BfclV4PublicV2SemanticReviewRole(role)
    normalized_url = _normalized_base_url(base_url)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("semantic review API key must be non-empty")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise TypeError("semantic review timeout must be numeric")
    if timeout_seconds <= 0:
        raise ValueError("semantic review timeout must be positive")
    provider_origin_sha256 = canonical_sha256(normalized_url)
    declaration = BfclV4PublicV2SemanticReviewerDeclaration(
        declared_reviewer_kind=BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED,
        declared_principal_id=f"litellm:{checked_role.value}:{model}",
        model_provider="litellm-openai-compatible",
        model_name=model,
        model_revision=BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_MODEL_REVISION,
        served_model_weights_identity_sha256=_declared_route_identity(
            provider_origin_sha256=provider_origin_sha256,
            catalog_observation_sha256=catalog_observation_sha256,
            model=model,
        ),
        model_prompt_sha256=BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256,
        model_generation_seed_u64=generation_seed_u64,
    )
    receipt = verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        packet,
        mapping,
        declaration,
        runtime_hmac_secret=runtime_hmac_secret,
    )
    request_payload = build_bfcl_v4_public_v2_semantic_review_request(
        packet,
        model=model,
        generation_seed_u64=generation_seed_u64,
    )
    raw = transport(
        normalized_url + "/chat/completions",
        api_key,
        BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_USER_AGENT,
        request_payload,
        float(timeout_seconds),
    )
    response_id, response_model, system_fingerprint, usage, assignments = _parse_provider_response(
        raw, packet, requested_model=model
    )
    response = BfclV4PublicV2SemanticReviewerResponse(
        reviewer_packet_ref=packet.ref,
        reviewer_declaration_ref=declaration.ref,
        assignments=assignments,
    )
    validate_bfcl_v4_public_v2_semantic_reviewer_response(packet, declaration, response)
    input_tokens, output_tokens, total_tokens = usage
    attestation = BfclV4PublicV2SemanticReviewExecutionAttestation(
        reviewer_packet_ref=packet.ref,
        trusted_mapping_ref=mapping.ref,
        reviewer_declaration_ref=declaration.ref,
        distribution_receipt_ref=receipt.ref,
        reviewer_response_ref=response.ref,
        provider_origin_sha256=provider_origin_sha256,
        request_body_sha256=hashlib.sha256(canonical_json_bytes(request_payload)).hexdigest(),
        raw_response_body_sha256=hashlib.sha256(raw).hexdigest(),
        provider_response_id=response_id,
        requested_model=model,
        provider_reported_model=response_model,
        provider_system_fingerprint=system_fingerprint,
        provider_seed_u64=generation_seed_u64,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    return BfclV4PublicV2SemanticReviewSessionEvidence(
        reviewer_declaration=declaration,
        distribution_receipt=receipt,
        reviewer_response=response,
        execution_attestation=attestation,
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("build_bfcl")
    or name.startswith("run_bfcl")
]
