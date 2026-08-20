from __future__ import annotations

from dataclasses import dataclass

import pytest

import spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts as request_contracts  # noqa: E501
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2NodeKind,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_barrier_capability import (
    BfclV4PublicV2VerifiedDecisionBarrierCapability,
    BfclV4PublicV2VerifiedDecisionBarrierReceipt,
    _mint_bfcl_v4_public_v2_verified_decision_barrier,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationId,
    BfclV4PublicV2MutationProposal,
    materialize_bfcl_v4_public_v2_mutation,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts import (
    BfclV4PublicV2PureAtBBatchGradeReceipt,
    BfclV4PublicV2PureAtBBatchGradeRequest,
    BfclV4PublicV2PureAtBCellGradeReceipt,
    bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint,
    bfcl_v4_public_v2_pure_at_b_source_set_fingerprint,
    build_bfcl_v4_public_v2_pure_at_b_batch_grade_request,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2FrozenArmTreatment,
    BfclV4PublicV2ModelVisibleRequest,
    BfclV4PublicV2RequestMaterialization,
    BfclV4PublicV2ResolvedTaskReceipt,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
    BfclV4PublicV2TrustedGraderReceipt,
)
from spiral_harness.core.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import BackendTokenUsage, ExecutionStatus
from spiral_harness.execution.native_function_contracts import (
    NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
    NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE,
    NativeFunctionExecution,
    NativeFunctionSlotBinding,
    NativeFunctionUsage,
    native_function_execution_fingerprint,
)
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_v2_dispatch_contracts import (
    BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    BfclV4PublicV2DispatchReceipt,
    bfcl_v4_public_v2_journal_prefix_fingerprint,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2ProviderRequest,
    BfclV4PublicV2PureAtBAggregationRecord,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2LiveExecutionConfig,
    freeze_bfcl_v4_public_v2_live_execution_config,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationTreatmentRole,
    materialize_bfcl_v4_public_v2_mutation_runtime_batch,
)
from spiral_harness.experiments.bfcl_v4_public_v2_native import (
    materialize_bfcl_v4_public_v2_native_request,
)
from spiral_harness.experiments.bfcl_v4_public_v2_trusted_adapter_contracts import (
    BfclV4PublicV2IssuedTrustedCall,
    BfclV4PublicV2TrustedCallProducer,
    BfclV4PublicV2TrustedCallProducerError,
    BfclV4PublicV2TrustedCallRecord,
    bfcl_v4_public_v2_native_slot_fingerprint,
)
from spiral_harness.experiments.bfcl_v4_public_v2_trusted_adapters import (
    BfclV4PublicV2PureAtBBatchGraderAdapter,
    BfclV4PublicV2TrustedAdapterError,
    BfclV4PublicV2TrustedCallRegistry,
    BfclV4PublicV2TrustedGraderAdapter,
    project_bfcl_v4_public_v2_pure_at_b_batch_grade,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)

RUNTIME = "b" * 64
LOADED = "c" * 64
GRADER_SOURCE = "d" * 64
PURE_GRADER_SOURCE = "e" * 64


def _dispatch_receipt(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node,
    *,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    request_payload_sha256: str,
    previous_event_sha256: str | None,
) -> BfclV4PublicV2DispatchReceipt:
    return BfclV4PublicV2DispatchReceipt(
        node=node,
        node_reference_sha256=canonical_sha256(node),
        journal_prefix_fingerprint=bfcl_v4_public_v2_journal_prefix_fingerprint(
            campaign_plan_fingerprint=campaign.fingerprint,
            node_schedule_content_sha256=campaign.node_schedule_content_sha256,
            runtime_fingerprint=runtime_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
            event_count=node.node_slot,
            tail_event_sha256=previous_event_sha256,
        ),
        journal_prefix_event_count=node.node_slot,
        journal_prefix_tail_event_sha256=previous_event_sha256,
        campaign_plan_fingerprint=campaign.fingerprint,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        request_materialization_fingerprint=canonical_sha256(
            {
                "domain": "bfcl-v2-test-materialization/v1",
                "node_id": node.node_id,
                "payload": request_payload_sha256,
            }
        ),
        native_request_fingerprint=request_payload_sha256,
        request_payload_sha256=request_payload_sha256,
        proposal_batch_set_fingerprint=BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    )


@pytest.fixture(scope="module")
def campaign() -> BfclV4PublicDevelopmentV2CampaignPlan:
    return build_bfcl_v4_public_development_v2_campaign_plan()


@pytest.fixture(scope="module")
def live_config(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    bfcl_v2_verified_legacy_authority,
) -> BfclV4PublicV2LiveExecutionConfig:
    return freeze_bfcl_v4_public_v2_live_execution_config(
        catalog_observation=observe_bfcl_v4_public_live_model_catalog(
            (campaign.execution_profile.model_route,),
            observed_at_utc="2026-08-15T12:00:00Z",
        ),
        campaign=campaign,
        semantic_authority=bfcl_v2_verified_legacy_authority.capability,
        backend_name="openai-compatible-native-function",
        backend_fingerprint=canonical_sha256("backend"),
        serializer_fingerprint=canonical_sha256("serializer"),
        parser_fingerprint=canonical_sha256("parser"),
        transport_fingerprint=canonical_sha256("transport"),
    )


@pytest.fixture(scope="module")
def prompts():
    materialization = materialize_bfcl_v4_public_v2_mutation(
        BfclV4PublicV2MutationProposal(
            catalogue_id=BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER
        )
    )
    batch = materialize_bfcl_v4_public_v2_mutation_runtime_batch(materialization)
    return {prompt.role: prompt for prompt in batch.prompts}


def _decision_evidence(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    *,
    semantic_release: str,
) -> BfclV4PublicV2DecisionBarrierEvidence:
    decision_nodes = tuple(
        node
        for node in campaign.nodes
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
    )
    event_fingerprints = tuple(f"{index + 1:064x}" for index in range(6))
    return BfclV4PublicV2DecisionBarrierEvidence(
        semantic_release_fingerprint=semantic_release,
        decision_node_references=tuple(canonical_sha256(node) for node in decision_nodes),
        decision_event_fingerprints=event_fingerprints,
        final_decision_event_fingerprint=event_fingerprints[-1],
    )


def _unlock(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    *,
    semantic_release: str,
    verified_barrier: BfclV4PublicV2VerifiedDecisionBarrierCapability | None = None,
) -> BfclV4PublicV2EvaluationUnlock:
    evidence = _decision_evidence(campaign, semantic_release=semantic_release)
    capability = (
        _mint_test_verified_barrier(campaign, semantic_release=semantic_release)
        if verified_barrier is None
        else verified_barrier
    )
    return BfclV4PublicV2EvaluationUnlock(
        barrier_evidence=evidence,
        barrier_evidence_fingerprint=evidence.fingerprint,
        verified_barrier_receipt_fingerprint=capability.receipt.fingerprint,
        authority_key_id="1" * 64,
        authentication_tag_hmac_sha256="2" * 64,
    )


def _mint_test_verified_barrier(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    *,
    semantic_release: str,
) -> BfclV4PublicV2VerifiedDecisionBarrierCapability:
    """Test-only mint for adapter tests that do not own a durable executor journal."""

    evidence = _decision_evidence(campaign, semantic_release=semantic_release)
    receipt = BfclV4PublicV2VerifiedDecisionBarrierReceipt(
        evidence=evidence,
        evidence_fingerprint=evidence.fingerprint,
        journal_snapshot_fingerprint=canonical_sha256("test-only-journal-snapshot"),
        journal_prefix_event_count=1_000,
        journal_tail_event_fingerprint=evidence.final_decision_event_fingerprint,
        runtime_fingerprint=RUNTIME,
        semantic_release_fingerprint=semantic_release,
        replay_state_fingerprint=canonical_sha256("test-only-independent-replay"),
    )
    return _mint_bfcl_v4_public_v2_verified_decision_barrier(receipt)


@dataclass(frozen=True)
class _Backend:
    live: BfclV4PublicV2LiveExecutionConfig

    @property
    def fingerprint(self) -> str:
        return self.live.backend_fingerprint

    @property
    def serializer_fingerprint(self) -> str:
        return self.live.serializer_fingerprint

    @property
    def parser_fingerprint(self) -> str:
        return self.live.parser_fingerprint

    @property
    def transport_fingerprint(self) -> str:
        return self.live.transport_fingerprint


@dataclass(frozen=True)
class _ProducedCall:
    producer: BfclV4PublicV2TrustedCallProducer
    issued: BfclV4PublicV2IssuedTrustedCall

    @property
    def record(self) -> BfclV4PublicV2TrustedCallRecord:
        return self.issued.record


def _task_coordinate(task_ref: str) -> tuple[BfclV4PublicDevelopmentV2Split, int, int]:
    split_name, raw_index = task_ref.split("-", maxsplit=1)
    split = BfclV4PublicDevelopmentV2Split(split_name)
    split_index = int(raw_index)
    offset = {
        BfclV4PublicDevelopmentV2Split.FIT: 0,
        BfclV4PublicDevelopmentV2Split.GATE: 5,
        BfclV4PublicDevelopmentV2Split.HOLDOUT: 9,
    }[split]
    return split, split_index, offset + split_index


def _request_materialization(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    live: BfclV4PublicV2LiveExecutionConfig,
    prompts,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: BfclV4PublicDevelopmentV2NodeKind,
    task_ref: str,
) -> BfclV4PublicV2RequestMaterialization:
    node = next(
        node
        for node in campaign.nodes
        if node.arm is BfclV4PublicDevelopmentV2Arm.FULL
        and node.kind is kind
        and node.task_ref == task_ref
    )
    split, split_index, ordinal = _task_coordinate(task_ref)
    task_id = BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS[ordinal]
    frozen = list(BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[task_id])
    names = frozen[3]
    question = [[{"role": "user", "content": f"trusted synthetic question {ordinal}"}]]
    functions = [
        {
            "name": name,
            "description": "trusted synthetic schema",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        }
        for name in names
    ]
    candidate_sha256 = canonical_sha256(
        {"id": task_id, "question": question, "function": functions}
    )
    frozen[2] = candidate_sha256
    identities = dict(BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES)
    identities[task_id] = tuple(frozen)
    monkeypatch.setattr(
        request_contracts,
        "BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES",
        identities,
    )
    category = next(
        value
        for value in ("parallel_multiple", "simple_python", "multiple", "parallel")
        if task_id.startswith(f"{value}_")
    )
    question_json = canonical_json(question)
    functions_json = canonical_json(functions)
    resolved = BfclV4PublicV2ResolvedTaskReceipt(
        task_ref=task_ref,
        split=split,
        split_index=split_index,
        manifest_ordinal=ordinal,
        task_id=task_id,
        category=category,
        official_function_names=names,
        candidate_payload_sha256=candidate_sha256,
        question_json_sha256=sha256_bytes(question_json.encode()),
        function_schemas_json_sha256=sha256_bytes(functions_json.encode()),
    )
    prompt_role = (
        BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
        if node.harness_variant == "parent"
        else BfclV4PublicV2MutationTreatmentRole.CANDIDATE
    )
    treatment = BfclV4PublicV2FrozenArmTreatment(
        arm=node.arm,
        harness_variant=node.harness_variant,
        prompt=prompts[prompt_role],
    )
    visible = BfclV4PublicV2ModelVisibleRequest(
        model_route=campaign.execution_profile.model_route,
        inference=campaign.execution_profile.inference,
        provider_seed_u63=node.provider_seed_u63,
        system_prompt=treatment.prompt.system_prompt,
        question_json=question_json,
        function_schemas_json=functions_json,
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    return BfclV4PublicV2RequestMaterialization(
        semantic_release_ref=live.semantic_release_ref,
        semantic_release_fingerprint=live.semantic_release_fingerprint,
        semantic_release_evidence_shape=live.semantic_release_evidence_shape,
        semantic_authority_verification_input_fingerprint=(
            live.semantic_authority_verification_input_fingerprint
        ),
        lineage=lineage,
        node=node,
        resolved_task=resolved,
        treatment=treatment,
        model_visible_request=visible,
        request_payload_sha256=visible.fingerprint,
    )


def _model_ref(value, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=canonical_sha256(value),
        size=len(payload),
        media_type=media_type,
    )


def _native_execution(
    materialization: BfclV4PublicV2RequestMaterialization,
    live: BfclV4PublicV2LiveExecutionConfig,
    *,
    request: NativeFunctionCallRequest | None = None,
    spec=None,
) -> NativeFunctionExecution:
    bound_spec = live.model_spec if spec is None else spec
    native_request = request or materialize_bfcl_v4_public_v2_native_request(
        visible_request=materialization.model_visible_request,
        expected_visible_request_sha256=materialization.request_payload_sha256,
        spec=bound_spec,
        backend=_Backend(live),
    )
    tool = native_request.task_required_tools[0]
    response = NativeFunctionCallResponse(
        request_fingerprint=native_request.fingerprint,
        serializer_fingerprint=native_request.serializer_fingerprint,
        parser_fingerprint=native_request.parser_fingerprint,
        transport_fingerprint=native_request.transport_fingerprint,
        tools_fingerprint=native_request.tools_fingerprint,
        tool_calls=(
            NativeAssistantToolCall(
                call_id="provider-call-0",
                official_name=tool.official_name,
                wire_name=tool.wire_name,
                arguments_json=canonical_json({}),
            ),
        ),
        assistant_text=None,
        finish_reason="tool_calls",
        usage=BackendTokenUsage(input_tokens=10, output_tokens=5),
    )
    task = NativeFunctionSlotBinding(
        task_fingerprint=materialization.resolved_task.fingerprint,
        slot_fingerprint=bfcl_v4_public_v2_native_slot_fingerprint(
            live_config=live,
            materialization=materialization,
        ),
    )
    usage = NativeFunctionUsage(
        input_tokens=10,
        output_tokens=5,
        tool_calls=1,
        latency_ms=1.0,
    )
    return NativeFunctionExecution(
        task=task,
        spec=bound_spec,
        request=native_request,
        request_ref=_model_ref(native_request, NATIVE_FUNCTION_REQUEST_MEDIA_TYPE),
        response=response,
        response_ref=_model_ref(response, NATIVE_FUNCTION_RESPONSE_MEDIA_TYPE),
        status=ExecutionStatus.COMPLETED,
        usage=usage,
        execution_fingerprint=native_function_execution_fingerprint(
            spec=bound_spec,
            task=task,
            request=native_request,
        ),
        request_sha256=native_request.fingerprint,
        error=None,
    )


def _ordinary_call(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    live: BfclV4PublicV2LiveExecutionConfig,
    prompts,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: BfclV4PublicDevelopmentV2NodeKind = BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
    task_ref: str = "fit-00",
) -> _ProducedCall:
    materialization = _request_materialization(
        campaign,
        live,
        prompts,
        monkeypatch,
        kind=kind,
        task_ref=task_ref,
    )
    producer = BfclV4PublicV2TrustedCallProducer(campaign=campaign, live_config=live)
    issued = producer.produce(
        request_materialization=materialization,
        native_execution=_native_execution(materialization, live),
    )
    return _ProducedCall(producer=producer, issued=issued)


def _ordinary_receipt(
    call: BfclV4PublicV2TrustedCallRecord,
    *,
    evaluation_unlock: BfclV4PublicV2EvaluationUnlock | None = None,
    correct: bool = True,
) -> BfclV4PublicV2TrustedGraderReceipt:
    node = call.grade_request.node
    split = {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT: BfclV4PublicDevelopmentV2Split.FIT,
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT: BfclV4PublicDevelopmentV2Split.HOLDOUT,
    }[node.kind]
    assert node.campaign_call_slot is not None
    assert node.task_ref is not None
    return BfclV4PublicV2TrustedGraderReceipt(
        node_reference_sha256=canonical_sha256(node),
        campaign_call_slot=node.campaign_call_slot,
        task_ref=node.task_ref,
        task_reference_sha256=canonical_sha256(
            {
                "domain": ("spiral-bfcl-v4-public-development-v2-structural-task-reference/v1"),
                "campaign_plan_fingerprint": (
                    call.grade_request.request_lineage.campaign_plan_fingerprint
                ),
                "task_ref": node.task_ref,
                "task_payload_sha256": call.grade_request.task_payload_sha256,
            }
        ),
        split_role=split,
        request_fingerprint=call.grade_request.request_fingerprint,
        response_fingerprint=call.grade_request.response_fingerprint,
        evaluation_unlock_fingerprint=(
            None if evaluation_unlock is None else evaluation_unlock.fingerprint
        ),
        loaded_question_bundle_fingerprint=LOADED,
        grader_source_sha256=GRADER_SOURCE,
        correct=correct,
    )


@dataclass
class _FullGrader:
    receipt: BfclV4PublicV2TrustedGraderReceipt
    unlock: BfclV4PublicV2EvaluationUnlock | None = None
    grade_calls: int = 0
    authorization_calls: int = 0
    observed_unlock: BfclV4PublicV2EvaluationUnlock | None = None
    observed_capability: BfclV4PublicV2VerifiedDecisionBarrierCapability | None = None

    def issue_evaluation_unlock(self, capability):
        self.authorization_calls += 1
        self.observed_capability = capability
        if self.unlock is None:
            raise AssertionError("test grader has no unlock")
        return self.unlock

    def grade(self, request, *, evaluation_unlock=None):
        self.grade_calls += 1
        self.observed_unlock = evaluation_unlock
        return self.receipt


def _ordinary_adapter(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    live: BfclV4PublicV2LiveExecutionConfig,
    produced: _ProducedCall,
    receipt: BfclV4PublicV2TrustedGraderReceipt,
    *,
    unlock: BfclV4PublicV2EvaluationUnlock | None = None,
) -> tuple[
    BfclV4PublicV2TrustedGraderAdapter,
    BfclV4PublicV2TrustedCallRegistry,
    _FullGrader,
]:
    registry = BfclV4PublicV2TrustedCallRegistry(producer=produced.producer)
    registry.register(produced.issued)
    grader = _FullGrader(receipt=receipt, unlock=unlock)
    adapter = BfclV4PublicV2TrustedGraderAdapter(
        grader=grader,
        campaign=campaign,
        live_config=live,
        call_registry=registry,
    )
    return adapter, registry, grader


def _grade_with_executor_surface(
    adapter: BfclV4PublicV2TrustedGraderAdapter,
    call: BfclV4PublicV2TrustedCallRecord,
    *,
    canonical_response: str | None = None,
    request_payload_sha256: str | None = None,
    provider_response_fingerprint: str | None = None,
    evaluation_unlock_fingerprint: str | None = None,
):
    return adapter.grade(
        call.grade_request.node,
        call.canonical_response if canonical_response is None else canonical_response,
        request_payload_sha256=(
            call.request_payload_sha256
            if request_payload_sha256 is None
            else request_payload_sha256
        ),
        provider_response_fingerprint=(
            call.provider_response_fingerprint
            if provider_response_fingerprint is None
            else provider_response_fingerprint
        ),
        evaluation_unlock_fingerprint=evaluation_unlock_fingerprint,
    )


def test_ordinary_adapter_projects_full_receipt_and_consumes_call_once(
    campaign,
    live_config,
    prompts,
    monkeypatch,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    call = produced.record
    receipt = _ordinary_receipt(call)
    adapter, registry, grader = _ordinary_adapter(campaign, live_config, produced, receipt)

    projection = _grade_with_executor_surface(adapter, call)

    assert projection.correct is True
    assert projection.request_payload_sha256 == call.request_payload_sha256
    assert projection.provider_response_fingerprint == call.provider_response_fingerprint
    assert projection.trusted_grade_request_fingerprint == call.grade_request.fingerprint
    assert projection.trusted_grader_receipt_fingerprint == receipt.fingerprint
    assert projection.evaluation_unlock_fingerprint is None
    assert registry.pending_count == 0
    assert grader.grade_calls == 1
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="lacks"):
        _grade_with_executor_surface(adapter, call)
    assert grader.grade_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_response", "[]"),
        ("request_payload_sha256", "9" * 64),
        ("provider_response_fingerprint", "9" * 64),
    ],
)
def test_ordinary_adapter_rejects_executor_lineage_mismatch_before_grading(
    campaign,
    live_config,
    prompts,
    monkeypatch,
    field,
    value,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    call = produced.record
    adapter, registry, grader = _ordinary_adapter(
        campaign,
        live_config,
        produced,
        _ordinary_receipt(call),
    )

    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="lineage"):
        _grade_with_executor_surface(adapter, call, **{field: value})

    assert registry.pending_count == 1
    assert grader.grade_calls == 0


@pytest.mark.parametrize(
    "receipt_update",
    [
        {"node_reference_sha256": "9" * 64},
        {"task_reference_sha256": "9" * 64},
        {"request_fingerprint": "9" * 64},
        {"response_fingerprint": "9" * 64},
    ],
)
def test_ordinary_adapter_rejects_full_receipt_mismatch(
    campaign,
    live_config,
    prompts,
    monkeypatch,
    receipt_update,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    call = produced.record
    receipt = _ordinary_receipt(call).model_copy(update=receipt_update)
    adapter, registry, grader = _ordinary_adapter(campaign, live_config, produced, receipt)

    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="full grader receipt"):
        _grade_with_executor_surface(adapter, call)

    assert registry.pending_count == 0
    assert grader.grade_calls == 1


def test_registry_rejects_duplicate_and_foreign_producer(
    campaign,
    live_config,
    prompts,
    monkeypatch,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    registry = BfclV4PublicV2TrustedCallRegistry(producer=produced.producer)
    registry.register(produced.issued)
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="already"):
        registry.register(produced.issued)

    foreign = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    other_registry = BfclV4PublicV2TrustedCallRegistry(producer=foreign.producer)
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="unverified producer"):
        other_registry.register(produced.issued)


def test_producer_rejects_prompt_materialization_and_release_tampering(
    campaign,
    live_config,
    prompts,
    monkeypatch,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    materialization = produced.record.request_materialization
    bad_prompt = materialization.treatment.prompt.model_copy(
        update={"system_prompt": "attacker prompt"}
    )
    bad_treatment = materialization.treatment.model_copy(update={"prompt": bad_prompt})
    bad_visible = materialization.model_visible_request.model_copy(
        update={"system_prompt": "attacker prompt"}
    )
    bad_materialization = materialization.model_copy(
        update={
            "treatment": bad_treatment,
            "model_visible_request": bad_visible,
            "request_payload_sha256": bad_visible.fingerprint,
        }
    )
    with pytest.raises(BfclV4PublicV2TrustedCallProducerError, match="materialization"):
        produced.producer.produce(
            request_materialization=bad_materialization,
            native_execution=produced.record.native_execution,
        )

    wrong_ref = ArtifactRef(
        sha256="9" * 64,
        size=1,
        media_type=live_config.semantic_release_ref.media_type,
    )
    wrong_release = materialization.model_copy(
        update={
            "semantic_release_ref": wrong_ref,
            "semantic_release_fingerprint": wrong_ref.sha256,
        }
    )
    with pytest.raises(BfclV4PublicV2TrustedCallProducerError, match="production"):
        produced.producer.produce(
            request_materialization=wrong_release,
            native_execution=_native_execution(wrong_release, live_config),
        )


@pytest.mark.parametrize("surface", ["messages", "tools", "backend"])
def test_producer_rejects_native_surface_or_spec_tampering(
    campaign,
    live_config,
    prompts,
    monkeypatch,
    surface,
) -> None:
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    materialization = produced.record.request_materialization
    request = produced.record.native_execution.request
    spec = live_config.model_spec
    if surface == "messages":
        first = request.messages[0].model_copy(update={"content": "attacker message"})
        request = request.model_copy(update={"messages": (first, *request.messages[1:])})
    elif surface == "tools":
        first = request.task_required_tools[0]
        schema = first.wire_schema
        schema["description"] = "attacker tool description"
        changed = type(first).from_schema(schema, wire_name=first.wire_name)
        request = request.model_copy(
            update={"task_required_tools": (changed, *request.task_required_tools[1:])}
        )
    else:
        spec = spec.model_copy(update={"backend_fingerprint": "9" * 64})
        request = request.model_copy(update={"backend_fingerprint": "9" * 64})
    execution = _native_execution(materialization, live_config, request=request, spec=spec)
    with pytest.raises(BfclV4PublicV2TrustedCallProducerError, match="production"):
        produced.producer.produce(
            request_materialization=materialization,
            native_execution=execution,
        )


def test_holdout_adapter_requires_exact_issued_unlock(
    campaign,
    live_config,
    prompts,
    monkeypatch,
) -> None:
    semantic_release = live_config.semantic_release_fingerprint
    capability = _mint_test_verified_barrier(
        campaign,
        semantic_release=semantic_release,
    )
    unlock = _unlock(
        campaign,
        semantic_release=semantic_release,
        verified_barrier=capability,
    )
    produced = _ordinary_call(
        campaign,
        live_config,
        prompts,
        monkeypatch,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        task_ref="holdout-00",
    )
    call = produced.record
    receipt = _ordinary_receipt(call, evaluation_unlock=unlock)
    adapter, registry, grader = _ordinary_adapter(
        campaign,
        live_config,
        produced,
        receipt,
        unlock=unlock,
    )

    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="unknown"):
        _grade_with_executor_surface(
            adapter,
            call,
            evaluation_unlock_fingerprint="9" * 64,
        )
    assert registry.pending_count == 1
    issued = adapter.issue_evaluation_unlock(capability)
    projection = _grade_with_executor_surface(
        adapter,
        call,
        evaluation_unlock_fingerprint=issued.fingerprint,
    )
    assert projection.evaluation_unlock_fingerprint == issued.fingerprint
    assert grader.authorization_calls == grader.grade_calls == 1
    assert grader.observed_unlock == unlock
    assert grader.observed_capability is capability


def test_unlock_release_mismatch_fails_before_trusted_authority(
    campaign,
    live_config,
    prompts,
    monkeypatch,
) -> None:
    unlock = _unlock(
        campaign,
        semantic_release=live_config.semantic_release_fingerprint,
    )
    produced = _ordinary_call(campaign, live_config, prompts, monkeypatch)
    call = produced.record
    adapter, _, grader = _ordinary_adapter(
        campaign,
        live_config,
        produced,
        _ordinary_receipt(call),
        unlock=unlock,
    )
    wrong = _mint_test_verified_barrier(campaign, semantic_release="9" * 64)
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="decision barrier"):
        adapter.issue_evaluation_unlock(wrong)
    assert grader.authorization_calls == 0

    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="verified replay capability"):
        adapter.issue_evaluation_unlock(  # type: ignore[arg-type]
            _decision_evidence(
                campaign,
                semantic_release=live_config.semantic_release_fingerprint,
            )
        )
    assert grader.authorization_calls == 0


def _source_events(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    unlock: BfclV4PublicV2EvaluationUnlock,
) -> tuple[BfclV4PublicV2JournalEvent, ...]:
    events: list[BfclV4PublicV2JournalEvent] = []
    semantic_release = unlock.barrier_evidence.semantic_release_fingerprint
    for node in campaign.nodes:
        if node.kind is not BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE:
            continue
        assert node.campaign_call_slot is not None
        assert node.provider_seed_u63 is not None
        payload_sha256 = canonical_sha256({"node_id": node.node_id})
        previous_event_sha256 = canonical_sha256(
            {"domain": "bfcl-v2-test-prefix-tail/v1", "node_slot": node.node_slot}
        )
        provider_request = BfclV4PublicV2ProviderRequest(
            campaign_plan_fingerprint=campaign.fingerprint,
            node_schedule_content_sha256=campaign.node_schedule_content_sha256,
            mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
            runtime_fingerprint=RUNTIME,
            semantic_release_fingerprint=semantic_release,
            node_id=node.node_id,
            node_reference_sha256=canonical_sha256(node),
            campaign_call_slot=node.campaign_call_slot,
            provider_seed_u63=node.provider_seed_u63,
            request_payload_sha256=payload_sha256,
            decision_barrier_evidence_fingerprint=unlock.barrier_evidence_fingerprint,
            evaluation_unlock_fingerprint=unlock.fingerprint,
        )
        dispatch = _dispatch_receipt(
            campaign,
            node,
            runtime_fingerprint=RUNTIME,
            semantic_release_fingerprint=semantic_release,
            request_payload_sha256=payload_sha256,
            previous_event_sha256=previous_event_sha256,
        )
        events.append(
            BfclV4PublicV2JournalEvent(
                sequence=node.node_slot,
                previous_event_sha256=previous_event_sha256,
                campaign_plan_fingerprint=campaign.fingerprint,
                node_schedule_content_sha256=campaign.node_schedule_content_sha256,
                mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
                runtime_fingerprint=RUNTIME,
                semantic_release_fingerprint=semantic_release,
                node_id=node.node_id,
                node_slot=node.node_slot,
                node_reference_sha256=canonical_sha256(node),
                event_kind=BfclV4PublicV2EventKind.CALL,
                request_fingerprint=provider_request.fingerprint,
                request_payload_sha256=payload_sha256,
                dispatch_fingerprint=dispatch.fingerprint,
                journal_prefix_fingerprint=dispatch.journal_prefix_fingerprint,
                request_materialization_fingerprint=dispatch.request_materialization_fingerprint,
                native_request_fingerprint=dispatch.native_request_fingerprint,
                proposal_batch_set_fingerprint=dispatch.proposal_batch_set_fingerprint,
                provider_attempt_disposition=(BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE),
                provider_attempts_consumed=1,
                executed_harness_variant="bare",
                decision_barrier_evidence_fingerprint=(unlock.barrier_evidence_fingerprint),
                evaluation_unlock_fingerprint=unlock.fingerprint,
            )
        )
    assert len(events) == 330
    return tuple(events)


def _aggregations(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> tuple[BfclV4PublicV2PureAtBAggregationRecord, ...]:
    by_node = {event.node_id: event for event in events}
    records: list[BfclV4PublicV2PureAtBAggregationRecord] = []
    for outer_seed in campaign.outer_seeds_u64:
        for allocation in campaign.pure_at_b_allocation:
            nodes = tuple(
                node
                for node in campaign.nodes
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
                and node.outer_seed_u64 == outer_seed
                and node.task_ref == allocation.task_ref
            )
            sources = tuple(by_node[node.node_id] for node in nodes)
            records.append(
                BfclV4PublicV2PureAtBAggregationRecord(
                    outer_seed_u64=outer_seed,
                    task_ref=allocation.task_ref,
                    source_event_sha256=tuple(event.fingerprint for event in sources),
                    result=aggregate_bfcl_v4_public_development_v2_pure_at_b(
                        tuple(event.canonical_response for event in sources)
                    ),
                )
            )
    assert len(records) == 48
    return tuple(records)


def _batch_request(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    *,
    semantic_release: str,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    unlock = _unlock(campaign, semantic_release=semantic_release)
    events = _source_events(campaign, unlock)
    return build_bfcl_v4_public_v2_pure_at_b_batch_grade_request(
        campaign=campaign,
        events=events,
        aggregations=_aggregations(campaign, events),
        evaluation_unlock=unlock,
        semantic_release_fingerprint=semantic_release,
    )


def _batch_receipt(
    request: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> BfclV4PublicV2PureAtBBatchGradeReceipt:
    cells = tuple(
        BfclV4PublicV2PureAtBCellGradeReceipt(
            semantic_release_fingerprint=request.semantic_release_fingerprint,
            outer_seed_u64=cell.outer_seed_u64,
            task_ref=cell.task_ref,
            cell_grade_request_fingerprint=cell.fingerprint,
            allocation=cell.allocation,
            allocation_fingerprint=canonical_sha256(cell.allocation),
            source_event_sha256=cell.source_event_sha256,
            source_set_fingerprint=bfcl_v4_public_v2_pure_at_b_source_set_fingerprint(
                cell.source_event_sha256
            ),
            aggregation_spec_fingerprint=cell.aggregation_spec_fingerprint,
            aggregation_result=cell.aggregation_result,
            aggregation_result_fingerprint=canonical_sha256(cell.aggregation_result),
            selected_canonical_response=(cell.aggregation_result.selected_canonical_response),
            selected_canonical_response_fingerprint=(
                bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
                    cell.aggregation_result.selected_canonical_response
                )
            ),
            decision_barrier_evidence_fingerprint=(request.decision_barrier_evidence_fingerprint),
            evaluation_unlock_fingerprint=request.evaluation_unlock_fingerprint,
            loaded_question_bundle_fingerprint=LOADED,
            grader_source_sha256=GRADER_SOURCE,
            pure_at_b_grader_source_sha256=PURE_GRADER_SOURCE,
            correct=False,
            isolated_worker_executed=False,
            exact_upstream_ast_checker_executed=False,
            possible_answers_read_in_isolated_trusted_worker=False,
        )
        for cell in request.cells
    )
    return BfclV4PublicV2PureAtBBatchGradeReceipt(
        semantic_release_fingerprint=request.semantic_release_fingerprint,
        batch_grade_request_fingerprint=request.fingerprint,
        evaluation_unlock_fingerprint=request.evaluation_unlock_fingerprint,
        decision_barrier_evidence_fingerprint=(request.decision_barrier_evidence_fingerprint),
        cell_receipts=cells,
        cell_receipt_fingerprints=tuple(cell.fingerprint for cell in cells),
        correct_count=0,
        isolated_worker_execution_count=0,
        loaded_question_bundle_fingerprint=LOADED,
        grader_source_sha256=GRADER_SOURCE,
        pure_at_b_grader_source_sha256=PURE_GRADER_SOURCE,
    )


@dataclass
class _FullBatchGrader:
    receipt: BfclV4PublicV2PureAtBBatchGradeReceipt
    calls: int = 0
    observed_request: BfclV4PublicV2PureAtBBatchGradeRequest | None = None

    def __call__(self, grader, request):
        self.calls += 1
        self.observed_request = request
        return self.receipt


@pytest.fixture(scope="module")
def batch_artifacts(campaign, live_config):
    request = _batch_request(
        campaign,
        semantic_release=live_config.semantic_release_fingerprint,
    )
    return request, _batch_receipt(request)


def test_batch_adapter_projects_48_full_receipts_without_sample_grades(
    campaign,
    batch_artifacts,
) -> None:
    request, receipt = batch_artifacts
    full_grader = _FullBatchGrader(receipt)
    adapter = BfclV4PublicV2PureAtBBatchGraderAdapter(
        grader=object(),
        campaign=campaign,
        semantic_release_fingerprint=request.semantic_release_fingerprint,
        full_batch_grader=full_grader,
    )

    projection = adapter.grade_pure_at_b_batch(request)

    assert full_grader.calls == 1
    assert full_grader.observed_request == request
    assert projection.batch_grade_request_fingerprint == request.fingerprint
    assert projection.batch_grade_receipt_fingerprint == receipt.fingerprint
    assert len(projection.cells) == 48
    assert projection.source_event_count == 330
    assert projection.trusted_grade_attempt_count == 48
    assert projection.individual_sample_grade_count == 0
    first_record = BfclV4PublicV2PureAtBAggregationRecord(
        outer_seed_u64=request.cells[0].outer_seed_u64,
        task_ref=request.cells[0].task_ref,
        source_event_sha256=request.cells[0].source_event_sha256,
        result=request.cells[0].aggregation_result,
    )
    assert projection.cells[0].aggregation_record_fingerprint == first_record.fingerprint
    assert projection.cells[0].cell_grade_receipt_fingerprint == (
        receipt.cell_receipts[0].fingerprint
    )


def test_batch_release_mismatch_fails_before_full_grader(campaign, batch_artifacts) -> None:
    request, receipt = batch_artifacts
    full_grader = _FullBatchGrader(receipt)
    adapter = BfclV4PublicV2PureAtBBatchGraderAdapter(
        grader=object(),
        campaign=campaign,
        semantic_release_fingerprint="9" * 64,
        full_batch_grader=full_grader,
    )
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="campaign, release, or unlock"):
        adapter.grade_pure_at_b_batch(request)
    assert full_grader.calls == 0


def test_batch_projection_rejects_request_and_cell_receipt_mismatch(
    campaign,
    batch_artifacts,
) -> None:
    request, receipt = batch_artifacts
    wrong_request = receipt.model_copy(update={"batch_grade_request_fingerprint": "9" * 64})
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="batch request"):
        project_bfcl_v4_public_v2_pure_at_b_batch_grade(
            campaign=campaign,
            semantic_release_fingerprint=request.semantic_release_fingerprint,
            request=request,
            receipt=wrong_request,
        )

    changed_cell = receipt.cell_receipts[0].model_copy(
        update={"cell_grade_request_fingerprint": "9" * 64}
    )
    changed_cells = (changed_cell, *receipt.cell_receipts[1:])
    wrong_cell = receipt.model_copy(
        update={
            "cell_receipts": changed_cells,
            "cell_receipt_fingerprints": tuple(cell.fingerprint for cell in changed_cells),
        }
    )
    with pytest.raises(BfclV4PublicV2TrustedAdapterError, match="cell receipt"):
        project_bfcl_v4_public_v2_pure_at_b_batch_grade(
            campaign=campaign,
            semantic_release_fingerprint=request.semantic_release_fingerprint,
            request=request,
            receipt=wrong_cell,
        )
