from __future__ import annotations

import os
from pathlib import Path

import pytest

from spiral_harness.execution.contracts import (
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.model import ReplayBackend
from spiral_harness.experiments.bfcl_v4_multi_agent_comparison import (
    BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE,
    run_bfcl_v4_multi_agent_comparison,
)
from spiral_harness.experiments.bfcl_v4_multi_agent_verification import (
    verify_bfcl_v4_multi_agent_comparison,
)
from spiral_harness.storage.artifact_store import ArtifactStore


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


def test_complete_provider_free_comparison_closes_all_evidence(
    tmp_path: Path,
    pinned_bfcl_checkout: Path,
) -> None:
    backend_fingerprint = "bfcl-multi-agent-integration-replay-v1"
    spec = FrozenModelSpec(
        backend="replay",
        backend_fingerprint=backend_fingerprint,
        model="fixture/model",
        revision="snapshot-2026-08-20",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-20",
        runtime="fixture-runtime-v1",
        inference=InferenceConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2_048,
            timeout_seconds=5.0,
        ),
    )
    backend = ReplayBackend(
        fingerprint=backend_fingerprint,
        default_response=BackendResponse(
            output="[]",
            usage=BackendTokenUsage(input_tokens=10, output_tokens=1),
        ),
    )
    output = tmp_path / "comparison"

    result, result_ref = run_bfcl_v4_multi_agent_comparison(
        checkout=pinned_bfcl_checkout,
        output=output,
        spec=spec,
        backend=backend,
    )
    verified = verify_bfcl_v4_multi_agent_comparison(
        store=ArtifactStore(output / "artifacts"),
        result_ref=result_ref,
    )

    assert len(backend.calls) == 54
    assert result == verified
    assert result.consumed_model_calls == 54
    assert result.total_reported_tokens == 54 * 11
    assert (
        sum(
            len(outcome.baseline_execution_refs) + len(outcome.harness_execution_refs)
            for outcome in result.outcomes
        )
        == 54
    )
    assert (result.baseline_single_correct, result.baseline_budget_matched_correct) == (0, 0)
    assert result.harness_correct == 0

    first = result.outcomes[0]
    forged_first = first.model_copy(
        update={
            "baseline_canonical_responses": (
                '[{"arguments":{},"function_name":"forged"}]',
                *first.baseline_canonical_responses[1:],
            )
        }
    )
    forged = result.model_copy(update={"outcomes": (forged_first, *result.outcomes[1:])})
    forged_ref = ArtifactStore(output / "artifacts").put_json(
        forged,
        media_type=BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE,
    )

    with pytest.raises(ValueError, match="canonical responses differ"):
        verify_bfcl_v4_multi_agent_comparison(
            store=ArtifactStore(output / "artifacts"),
            result_ref=forged_ref,
        )
