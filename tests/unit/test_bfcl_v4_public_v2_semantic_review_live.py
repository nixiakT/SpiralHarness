from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import load_bfcl_v4_public_pilot
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection import (
    build_bfcl_v4_public_v2_semantic_reviewer_projection,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_review_live import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT,
    BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256,
    BfclV4PublicV2SemanticReviewLiveError,
    BfclV4PublicV2SemanticReviewRole,
    build_bfcl_v4_public_v2_semantic_review_request,
    run_bfcl_v4_public_v2_semantic_review,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256

_DEFAULT_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SECRET = b"live-semantic-review-test-secret" * 2
_CATALOG_SHA256 = hashlib.sha256(b"sanitized model catalog").hexdigest()


@pytest.fixture(scope="module")
def projection():
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for semantic live-review test")
    return build_bfcl_v4_public_v2_semantic_reviewer_projection(
        load_bfcl_v4_public_pilot(checkout),
        load_bfcl_v4_public_development_v2(checkout),
        runtime_hmac_secret=_SECRET,
    )


def _provider_body(packet, *, model: str = "open-review-model") -> bytes:
    assignments = [
        {
            "opaque_item_id": item.opaque_item_id,
            "semantic_family_label": f"family-{index}",
            "rationale": "The visible intent and required function structure are distinct.",
        }
        for index, item in enumerate(packet.items)
    ]
    return json.dumps(
        {
            "id": "review-completion-id",
            "model": model,
            "system_fingerprint": "opaque-system-fingerprint",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"assignments": assignments}),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 4_000,
                "completion_tokens": 2_000,
                "total_tokens": 6_000,
            },
        }
    ).encode()


def test_prompt_and_request_are_frozen_and_do_not_expose_private_mapping(projection) -> None:
    request = build_bfcl_v4_public_v2_semantic_review_request(
        projection.reviewer_packet,
        model="open-review-model",
        generation_seed_u64=101,
    )
    serialized = canonical_json_bytes(request).decode()

    assert BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256 == (
        "b13e8f8da48230e8fa6f8180b1a03e762b22aac9add4838d73d9513015b6085c"
    )
    assert hashlib.sha256(BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT.encode()).hexdigest() == (
        BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256
    )
    assert request["messages"][0] == {
        "role": "system",
        "content": BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT,
    }
    assert request["temperature"] == 0.0
    assert request["seed"] == 101
    assert request["response_format"] == {"type": "json_object"}
    assert all(item.public_task_id not in serialized for item in projection.trusted_mapping.entries)
    assert "private_boundary_label" not in serialized
    assert "candidate_payload_sha256" not in serialized
    assert "question_git_path" not in serialized


def test_single_attempt_run_returns_credential_free_bound_session(projection) -> None:
    captured = []

    def transport(url, api_key, user_agent, payload, timeout):
        captured.append((url, api_key, user_agent, payload, timeout))
        return _provider_body(projection.reviewer_packet)

    session = run_bfcl_v4_public_v2_semantic_review(
        projection.reviewer_packet,
        projection.trusted_mapping,
        runtime_hmac_secret=_SECRET,
        base_url="http://internal-gateway.invalid/v1/",
        api_key="credential-SENTINEL",
        model="open-review-model",
        role=BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE,
        generation_seed_u64=101,
        catalog_observation_sha256=_CATALOG_SHA256,
        transport=transport,
    )

    assert len(captured) == 1
    assert captured[0][0] == "http://internal-gateway.invalid/v1/chat/completions"
    assert captured[0][1] == "credential-SENTINEL"
    assert session.execution_attestation.provider_attempts == 1
    assert session.execution_attestation.input_tokens == 4_000
    assert session.execution_attestation.output_tokens == 2_000
    assert session.execution_attestation.total_tokens == 6_000
    assert session.execution_attestation.provider_reported_model == "open-review-model"
    assert session.execution_attestation.served_weights_cryptographically_verified is False
    assert session.reviewer_declaration.model_prompt_sha256 == (
        BFCL_V4_PUBLIC_V2_SEMANTIC_REVIEW_PROMPT_SHA256
    )
    assert session.reviewer_response.reviewer_packet_ref == projection.reviewer_packet.ref
    assert "credential-SENTINEL" not in session.model_dump_json()
    assert session.execution_attestation.provider_origin_sha256 == canonical_sha256(
        "http://internal-gateway.invalid/v1"
    )
    assert (
        session.execution_attestation.request_body_sha256
        == hashlib.sha256(canonical_json_bytes(captured[0][3])).hexdigest()
    )


@pytest.mark.parametrize("tamper", ["model", "finish", "coverage", "usage"])
def test_malformed_provider_completion_fails_without_retry(projection, tamper: str) -> None:
    decoded = json.loads(_provider_body(projection.reviewer_packet))
    if tamper == "model":
        decoded["model"] = "different-model"
    elif tamper == "finish":
        decoded["choices"][0]["finish_reason"] = "length"
    elif tamper == "coverage":
        content = json.loads(decoded["choices"][0]["message"]["content"])
        content["assignments"].pop()
        decoded["choices"][0]["message"]["content"] = json.dumps(content)
    else:
        decoded["usage"]["total_tokens"] += 1
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        return json.dumps(decoded).encode()

    with pytest.raises(BfclV4PublicV2SemanticReviewLiveError):
        run_bfcl_v4_public_v2_semantic_review(
            projection.reviewer_packet,
            projection.trusted_mapping,
            runtime_hmac_secret=_SECRET,
            base_url="http://internal-gateway.invalid/v1",
            api_key="credential-SENTINEL",
            model="open-review-model",
            role=BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE,
            generation_seed_u64=101,
            catalog_observation_sha256=_CATALOG_SHA256,
            transport=transport,
        )
    assert calls == 1


def test_transport_failure_consumes_the_only_attempt(projection) -> None:
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        raise BfclV4PublicV2SemanticReviewLiveError("sanitized failure")

    with pytest.raises(BfclV4PublicV2SemanticReviewLiveError, match="sanitized failure"):
        run_bfcl_v4_public_v2_semantic_review(
            projection.reviewer_packet,
            projection.trusted_mapping,
            runtime_hmac_secret=_SECRET,
            base_url="http://internal-gateway.invalid/v1",
            api_key="credential-SENTINEL",
            model="open-review-model",
            role=BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE,
            generation_seed_u64=101,
            catalog_observation_sha256=_CATALOG_SHA256,
            transport=transport,
        )
    assert calls == 1
