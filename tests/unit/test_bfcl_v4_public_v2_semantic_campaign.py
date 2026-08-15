from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_review_live import (
    BfclV4PublicV2SemanticReviewRole,
)
from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.experiments.bfcl_v4_public_v2_semantic_campaign import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,
    BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS,
    run_bfcl_v4_public_v2_semantic_campaign,
)

_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SECRET = b"semantic-campaign-test-secret-2026-08-15" * 2


def _require_checkout() -> None:
    if not (_CHECKOUT / ".git").exists():
        pytest.skip("pinned BFCL checkout unavailable for semantic campaign test")


def _catalog(_base_url: str, _api_key: str, _timeout: float) -> tuple[str, ...]:
    return (
        "unrelated-open-model",
        *BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS,
        BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,
    )


class _Transport:
    def __init__(self, *, disagree: bool) -> None:
        self.disagree = disagree
        self.models: list[str] = []

    def __call__(
        self,
        _url: str,
        _api_key: str,
        _user_agent: str,
        payload: dict,
        _timeout: float,
    ) -> bytes:
        model = payload["model"]
        self.models.append(model)
        packet_payload = json.loads(payload["messages"][1]["content"])
        items = packet_payload["reviewer_packet"]["items"]
        assignments = []
        for index, item in enumerate(items):
            label = f"family-{index}"
            if (
                self.disagree
                and model == BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS[1]
                and index == 1
            ):
                label = "family-0"
            assignments.append(
                {
                    "opaque_item_id": item["opaque_item_id"],
                    "semantic_family_label": label,
                    "rationale": "Distinct intent and tool structure in this synthetic review.",
                }
            )
        content = json.dumps({"assignments": assignments}, ensure_ascii=False)
        return canonical_json_bytes(
            {
                "id": f"response-{len(self.models)}",
                "model": model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                },
            }
        )


@pytest.mark.parametrize(("disagree", "attempts"), ((False, 2), (True, 3)))
def test_semantic_campaign_runs_two_reviews_and_only_adjudicates_disagreement(
    disagree: bool,
    attempts: int,
) -> None:
    _require_checkout()
    transport = _Transport(disagree=disagree)
    persisted_roles = []
    result = run_bfcl_v4_public_v2_semantic_campaign(
        checkout=_CHECKOUT,
        runtime_hmac_secret=_SECRET,
        base_url="http://litellm.invalid/v1",
        api_key="not-recorded-test-key",
        catalog_loader=_catalog,
        transport=transport,
        session_sink=lambda role, _session: persisted_roles.append(role.value),
    )

    expected_models = [*BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS]
    if disagree:
        expected_models.append(BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL)
    assert transport.models == expected_models
    assert persisted_roles == [
        "primary-one",
        "primary-two",
        *(("adjudicator",) if disagree else ()),
    ]
    assert result.review_attempt_count == attempts
    assert result.release.reviewer_count == attempts
    assert result.total_review_tokens == attempts * 300
    assert result.release.final_semantic_family_count == 40
    assert result.credentials_recorded is False
    assert result.hmac_secret_recorded is False
    assert result.solver_campaign_started is False

    replay_transport = _Transport(disagree=disagree)
    existing = {
        BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE: result.primary_sessions[0],
        BfclV4PublicV2SemanticReviewRole.PRIMARY_TWO: result.primary_sessions[1],
    }
    if result.adjudicator_session is not None:
        existing[BfclV4PublicV2SemanticReviewRole.ADJUDICATOR] = result.adjudicator_session
    replayed = run_bfcl_v4_public_v2_semantic_campaign(
        checkout=_CHECKOUT,
        runtime_hmac_secret=_SECRET,
        base_url="http://litellm.invalid/v1",
        api_key="not-recorded-test-key",
        catalog_loader=_catalog,
        transport=replay_transport,
        existing_sessions=existing,
    )
    assert replay_transport.models == []
    assert replayed == result

    partial_transport = _Transport(disagree=disagree)
    partial = run_bfcl_v4_public_v2_semantic_campaign(
        checkout=_CHECKOUT,
        runtime_hmac_secret=_SECRET,
        base_url="http://litellm.invalid/v1",
        api_key="not-recorded-test-key",
        catalog_loader=_catalog,
        transport=partial_transport,
        existing_sessions={
            BfclV4PublicV2SemanticReviewRole.PRIMARY_ONE: result.primary_sessions[0]
        },
    )
    assert partial_transport.models == [
        BFCL_V4_PUBLIC_V2_SEMANTIC_PRIMARY_MODELS[1],
        *((BFCL_V4_PUBLIC_V2_SEMANTIC_ADJUDICATOR_MODEL,) if disagree else ()),
    ]
    assert partial.release.reviewer_count == attempts
