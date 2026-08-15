from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import spiral_harness.experiments.bfcl_v4_public_campaign_executor as executor_module
from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import BackendTokenUsage, ProviderIdentityObservation
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis_contracts import (
    BfclV4PublicCampaignAnalysisInput,
    BfclV4PublicCampaignDescriptiveAnalysis,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor import (
    run_bfcl_v4_public_campaign,
    verify_bfcl_v4_public_campaign_execution,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor_contracts import (
    BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
    BfclV4CampaignExecutionStatus,
    BfclV4CampaignFailureStage,
)
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
    freeze_bfcl_v4_public_live_execution_config,
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BfclV4PublicPilotRunResult,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BACKEND = "5" * 64
SERIALIZER = "6" * 64
PARSER = "7" * 64
TRANSPORT = "8" * 64


@dataclass(slots=True)
class _ScriptedCampaignBackend:
    """One provider-free object reused across all three registered replicates."""

    fingerprint: str = BACKEND
    serializer_fingerprint: str = SERIALIZER
    parser_fingerprint: str = PARSER
    transport_fingerprint: str = TRANSPORT
    requests: list[NativeFunctionCallRequest] = field(default_factory=list, init=False)

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        checked = NativeFunctionCallRequest.model_validate(request, strict=True)
        self.requests.append(checked)
        tool = checked.task_required_tools[0]
        calls: tuple[NativeAssistantToolCall, ...]
        if tool.official_name == "submit_bfcl_diagnosis":
            calls = (
                NativeAssistantToolCall(
                    call_id=f"campaign-script-{len(self.requests):03d}",
                    official_name=tool.official_name,
                    wire_name=tool.wire_name,
                    arguments_json=canonical_json(
                        {
                            "diagnosis": (
                                "Audit semantic tool choice, exact arguments, and independent "
                                "call multiplicity without inspecting HOLDOUT targets."
                            )
                        }
                    ),
                ),
            )
        elif tool.official_name == "submit_bfcl_candidate":
            calls = (
                NativeAssistantToolCall(
                    call_id=f"campaign-script-{len(self.requests):03d}",
                    official_name=tool.official_name,
                    wire_name=tool.wire_name,
                    arguments_json=canonical_json(
                        {
                            "strategy_appendix": (
                                "Decompose independent operations, select tools by documented "
                                "semantics, and validate all required arguments before calls."
                            )
                        }
                    ),
                ),
            )
        else:
            calls = ()
        return NativeFunctionCallResponse(
            request_fingerprint=checked.fingerprint,
            serializer_fingerprint=checked.serializer_fingerprint,
            parser_fingerprint=checked.parser_fingerprint,
            transport_fingerprint=checked.transport_fingerprint,
            tools_fingerprint=checked.tools_fingerprint,
            tool_calls=calls,
            assistant_text=None if calls else "Provider-free scripted solver abstained.",
            finish_reason="tool_calls" if calls else "stop",
            usage=BackendTokenUsage(input_tokens=1, output_tokens=1),
            provider_identity_observation=ProviderIdentityObservation(
                requested_model=checked.requested_model,
                response_model="scripted-campaign/same-declared-model",
                system_fingerprint="scripted_campaign_system_fingerprint",
                backend_fingerprint=checked.backend_fingerprint,
            ),
        )


def _config():
    observation = observe_bfcl_v4_public_live_model_catalog(
        (BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,),
        observed_at_utc="2026-08-15T18:00:00Z",
    )
    return freeze_bfcl_v4_public_live_execution_config(
        catalog_observation=observation,
        backend="provider-free-scripted-campaign",
        backend_fingerprint=BACKEND,
        serializer_fingerprint=SERIALIZER,
        parser_fingerprint=PARSER,
        transport_fingerprint=TRANSPORT,
    )


def _pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else Path("/tmp/spiral-bfcl-upstream")
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for offline integration")
    return checkout


def test_scripted_campaign_executes_300_calls_and_publishes_exact_analysis(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    backend = _ScriptedCampaignBackend()
    config = _config()
    campaign = build_bfcl_v4_public_pilot_campaign()

    record = run_bfcl_v4_public_campaign(
        store,
        checkout=_pinned_checkout(),
        live_config=config,
        backend=backend,
    )

    assert record.result.status is BfclV4CampaignExecutionStatus.COMPLETE
    assert record.result.campaign_fingerprint == BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT
    assert len(record.result.completed_replicates) == 3
    assert len(record.result.checkpoint_refs) == 3
    assert record.result.verified_closed_model_calls == 300
    assert record.result.failure is None
    assert record.result.analysis_input_ref is not None
    assert record.result.analysis_ref is not None
    assert len(backend.requests) == 300
    assert tuple(item.seed for item in backend.requests) == tuple(
        slot.seed_u63 for replicate in campaign.replicates for slot in replicate.call_plan.calls
    )
    assert all(
        item.requested_model == config.model_spec.model
        and item.backend_fingerprint == config.backend_fingerprint
        and item.inference == config.model_spec.inference
        for item in backend.requests
    )

    for replicate, binding in zip(
        campaign.replicates,
        record.result.completed_replicates,
        strict=True,
    ):
        stored = store.get_json(binding.run_result_ref, BfclV4PublicPilotRunResult)
        assert stored.outer_seed_u64 == replicate.outer_seed_u64
        assert stored.plan_fingerprint == replicate.call_plan.fingerprint
        assert stored.model_spec == config.model_spec
        assert stored.attempt_budget == config.attempt_budget
        assert stored.total_model_calls == 100
        assert stored.automatic_retries_used is False
        assert binding.run_result_fingerprint == stored.fingerprint
        assert binding.joint_selection_decision_ref == stored.joint_selection_decision_ref
        assert binding.descriptive_metrics_ref == stored.descriptive_metrics_ref

    analysis_input = store.get_json(
        record.result.analysis_input_ref,
        BfclV4PublicCampaignAnalysisInput,
    )
    analysis = store.get_json(
        record.result.analysis_ref,
        BfclV4PublicCampaignDescriptiveAnalysis,
    )
    assert record.result.analysis_input_ref.media_type == (
        BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE
    )
    assert record.result.analysis_ref.media_type == BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE
    assert tuple(item.closure_ref for item in analysis_input.replicates) == tuple(
        item.closure_ref for item in record.result.completed_replicates
    )
    assert all(item.correct_task_seed_cells == 0 for item in analysis.arms)
    assert all(item.accuracy.numerator == 0 for item in analysis.arms)
    assert all(item.exact_delta.numerator == 0 for item in analysis.paired_deltas)
    assert all(
        item.strictly_exceeds_positive_ten_percentage_points is False
        for item in analysis.paired_deltas
    )

    audit = verify_bfcl_v4_public_campaign_execution(store, record.result_ref)
    assert audit.status is BfclV4CampaignExecutionStatus.COMPLETE
    assert audit.verified_replicate_count == 3
    assert audit.verified_model_calls == 300
    assert audit.verified_checkpoint_count == 3
    assert audit.analysis_recomputed_and_matched is True


def test_second_replicate_infrastructure_failure_publishes_explicit_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    backend = _ScriptedCampaignBackend()
    real_runner = executor_module.run_bfcl_v4_public_pilot_replicate
    invocations = 0

    def fail_before_second_replicate(*args, **kwargs):
        nonlocal invocations
        invocations += 1
        if invocations == 2:
            raise RuntimeError("test-only infrastructure stop")
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(
        executor_module,
        "run_bfcl_v4_public_pilot_replicate",
        fail_before_second_replicate,
    )
    record = run_bfcl_v4_public_campaign(
        store,
        checkout=_pinned_checkout(),
        live_config=_config(),
        backend=backend,
    )

    assert record.result.status is BfclV4CampaignExecutionStatus.INCOMPLETE
    assert len(record.result.completed_replicates) == 1
    assert len(record.result.checkpoint_refs) == 1
    assert record.result.verified_closed_model_calls == 100
    assert len(backend.requests) == 100
    assert record.result.analysis_input_ref is None
    assert record.result.analysis_ref is None
    assert record.result.failure is not None
    assert record.result.failure.stage is BfclV4CampaignFailureStage.REPLICATE_EXECUTION
    assert record.result.failure.active_replicate_ordinal == 1
    assert record.result.failure.exception_text_persisted is False
    assert record.result.failure.automatic_retry_attempted is False
    assert record.result.automatic_retries_used is False
    assert record.result.seed_skipping_used is False
    audit = verify_bfcl_v4_public_campaign_execution(store, record.result_ref)
    assert audit.status is BfclV4CampaignExecutionStatus.INCOMPLETE
    assert audit.verified_replicate_count == 1
    assert audit.verified_model_calls == 100
    assert audit.analysis_recomputed_and_matched is False
