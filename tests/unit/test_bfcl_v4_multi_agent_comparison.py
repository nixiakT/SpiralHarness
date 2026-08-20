from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spiral_harness.experiments.bfcl_v4_multi_agent_comparison import (
    BFCL_MULTI_AGENT_TASK_REFS,
    canonicalize_bfcl_candidate_text,
    comparison_protocol_summary,
    paired_comparison_seed,
)
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    BfclV4PublicV2CanonicalResponseError,
)


@pytest.fixture()
def pinned_bfcl_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    candidates = (
        *((Path(configured),) if configured else ()),
        Path("data/benchmarks/bfcl-v4/gorilla"),
        Path("/tmp/spiral-bfcl-upstream"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("pinned BFCL checkout unavailable")


def test_fresh_protocol_uses_only_fit_and_gate(pinned_bfcl_checkout) -> None:
    summary = comparison_protocol_summary(pinned_bfcl_checkout)

    assert summary["task_refs"] == list(BFCL_MULTI_AGENT_TASK_REFS)
    assert summary["model_calls"] == 54
    assert summary["baseline_model_calls"] == summary["harness_model_calls"] == 27
    assert summary["holdout_opened"] is False
    assert summary["hidden_test_evidence"] is False
    assert summary["reportable_result"] is False


def test_paired_seeds_are_stable_and_turn_specific() -> None:
    observed = tuple(
        paired_comparison_seed(
            root_seed=2026082001,
            task_id="simple_python_198",
            turn_id=turn_id,
        )
        for turn_id in ("analysis", "critique", "final")
    )

    assert observed == tuple(
        paired_comparison_seed(
            root_seed=2026082001,
            task_id="simple_python_198",
            turn_id=turn_id,
        )
        for turn_id in ("analysis", "critique", "final")
    )
    assert len(set(observed)) == 3
    assert all(0 <= seed < 1 << 31 for seed in observed)


def test_candidate_text_parser_accepts_one_array_with_optional_fence() -> None:
    value = [
        {
            "arguments": {"city": "杭州"},
            "function_name": "weather.get",
        }
    ]
    serialized = json.dumps(value, ensure_ascii=False)

    parsed = canonicalize_bfcl_candidate_text(f"```json\n{serialized}\n```")

    assert json.loads(parsed) == value


@pytest.mark.parametrize(
    "output",
    (
        "not json",
        '{"function_name":"x","arguments":{}}',
        '[{"name":"x","arguments":{}}]',
        "[] and also []",
    ),
)
def test_candidate_text_parser_fails_closed(output: str) -> None:
    with pytest.raises(BfclV4PublicV2CanonicalResponseError):
        canonicalize_bfcl_candidate_text(output)
