from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2FeedbackView,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationId,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BfclV4PublicV2SemanticDevelopmentRelease,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import BackendTokenUsage
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    observe_bfcl_v4_public_live_model_catalog,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2ProposalDisposition,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2LiveExecutionConfig,
    freeze_bfcl_v4_public_v2_live_execution_config,
)
from spiral_harness.experiments.bfcl_v4_public_v2_meta_runtime import (
    BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,
    BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL,
    BfclV4PublicV2MetaRuntimeError,
    materialize_bfcl_v4_public_v2_diagnosis_request,
    materialize_bfcl_v4_public_v2_proposal_request,
    parse_bfcl_v4_public_v2_diagnosis_response,
    parse_bfcl_v4_public_v2_proposal_response,
    resolve_bfcl_v4_public_v2_proposal_batch,
)
from spiral_harness.experiments.bfcl_v4_public_v2_meta_runtime_contracts import (
    BfclV4PublicV2DiagnosisFailure,
    BfclV4PublicV2FullDiagnosisPayload,
    BfclV4PublicV2MetaFitTaskProjection,
    BfclV4PublicV2MetaPrompt,
    BfclV4PublicV2MetaRequestMaterialization,
    BfclV4PublicV2ProposalFailure,
    BfclV4PublicV2ProposalParseResult,
    BfclV4PublicV2ProposalPayload,
    BfclV4PublicV2ScoreDiagnosisPayload,
)
from spiral_harness.experiments.bfcl_v4_public_v2_meta_runtime_native import (
    expected_provider_request,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallResponse,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(label: str, media_type: str = "application/x-test") -> ArtifactRef:
    return ArtifactRef(sha256=_digest(label), size=len(label), media_type=media_type)


def _release() -> BfclV4PublicV2SemanticDevelopmentRelease:
    return BfclV4PublicV2SemanticDevelopmentRelease(
        source_universe_fingerprint=_digest("universe"),
        reviewer_packet_ref=_ref("packet"),
        trusted_mapping_ref=_ref("mapping"),
        primary_review_refs=(_ref("review-one"), _ref("review-two")),
        primary_execution_attestation_refs=(_ref("attestation-one"), _ref("attestation-two")),
        final_partition_sha256=_digest("partition"),
        final_semantic_family_count=40,
        primary_pairwise_disagreement_count=0,
        reviewer_count=2,
        release_authority_hmac_sha256=_digest("release-hmac"),
    )


@pytest.fixture(scope="module")
def campaign():
    return build_bfcl_v4_public_development_v2_campaign_plan()


@pytest.fixture(scope="module")
def runtime(campaign) -> BfclV4PublicV2LiveExecutionConfig:
    return freeze_bfcl_v4_public_v2_live_execution_config(
        catalog_observation=observe_bfcl_v4_public_live_model_catalog(
            (BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,),
            observed_at_utc="2026-08-15T12:00:00Z",
        ),
        campaign=campaign,
        verified_semantic_release=_release(),
        backend_name="synthetic-native-backend",
        backend_fingerprint=_digest("backend"),
        serializer_fingerprint=_digest("serializer"),
        parser_fingerprint=_digest("parser"),
        transport_fingerprint=_digest("transport"),
    )


@dataclass
class _BackendTrap:
    fingerprint: str
    serializer_fingerprint: str
    parser_fingerprint: str
    transport_fingerprint: str
    invocations: int = 0

    def invoke(self, *_args: object, **_kwargs: object) -> None:
        self.invocations += 1
        raise AssertionError("provider-free meta runtime invoked a backend")


@pytest.fixture
def backend(runtime) -> _BackendTrap:
    return _BackendTrap(
        fingerprint=runtime.backend_fingerprint,
        serializer_fingerprint=runtime.serializer_fingerprint,
        parser_fingerprint=runtime.parser_fingerprint,
        transport_fingerprint=runtime.transport_fingerprint,
    )


def _node(campaign, *, arm, kind, pipeline=0, replicate_index=0):
    replicate = campaign.replicate_ids[replicate_index]
    return next(
        item
        for item in campaign.nodes
        if item.replicate_id == replicate
        and item.arm is arm
        and item.kind is kind
        and item.pipeline_index == pipeline
    )


def _parent_events(campaign, runtime, diagnosis_node):
    by_id = {item.node_id: item for item in campaign.nodes}
    events = []
    for index, node_id in enumerate(diagnosis_node.allowed_evidence_from):
        source = by_id[node_id]
        payload = _digest(f"parent-payload-{source.node_id}")
        request = expected_provider_request(campaign, runtime, source, payload)
        response = canonical_json({"calls": [{"name": "synthetic", "index": index}]})
        response_fingerprint = _digest(f"parent-response-{source.node_id}")
        events.append(
            BfclV4PublicV2JournalEvent(
                sequence=source.node_slot,
                previous_event_sha256=(
                    None if source.node_slot == 0 else _digest(f"previous-{source.node_slot}")
                ),
                campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
                node_schedule_content_sha256=campaign.node_schedule_content_sha256,
                mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
                runtime_fingerprint=runtime.fingerprint,
                semantic_release_fingerprint=runtime.semantic_release_fingerprint,
                node_id=source.node_id,
                node_slot=source.node_slot,
                node_reference_sha256=canonical_sha256(source),
                event_kind=BfclV4PublicV2EventKind.CALL,
                request_fingerprint=request.fingerprint,
                request_payload_sha256=payload,
                provider_attempt_disposition=BfclV4PublicV2AttemptDisposition.SUCCEEDED,
                provider_attempts_consumed=1,
                executed_harness_variant="parent",
                canonical_response=response,
                provider_response_fingerprint=response_fingerprint,
                binary_grade=index % 3 == 0,
                trusted_grade_request_fingerprint=_digest(f"grade-request-{source.node_id}"),
                trusted_grader_receipt_fingerprint=_digest(f"grade-receipt-{source.node_id}"),
                trusted_grade_attempts_consumed=1,
            )
        )
    return tuple(events)


def _projections(campaign, diagnosis_node):
    by_id = {item.node_id: item for item in campaign.nodes}
    return tuple(
        BfclV4PublicV2MetaFitTaskProjection(
            source_node_id=node_id,
            source_node_reference_sha256=canonical_sha256(by_id[node_id]),
            source_request_payload_sha256=_digest(f"parent-payload-{node_id}"),
            question_json=canonical_json(
                [[{"role": "user", "content": f"synthetic-question-{index}"}]]
            ),
            function_schemas_json=canonical_json(
                [
                    {
                        "name": f"synthetic_tool_{index}",
                        "description": "synthetic schema only",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                    }
                ]
            ),
            question_sha256=sha256_bytes(
                canonical_json(
                    [[{"role": "user", "content": f"synthetic-question-{index}"}]]
                ).encode()
            ),
            function_schemas_sha256=sha256_bytes(
                canonical_json(
                    [
                        {
                            "name": f"synthetic_tool_{index}",
                            "description": "synthetic schema only",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                                "required": ["value"],
                            },
                        }
                    ]
                ).encode()
            ),
        )
        for index, node_id in enumerate(diagnosis_node.allowed_evidence_from)
    )


def _native_response(request, tool, arguments, *, assistant_text=None, calls=1, wrong=False):
    tool_calls = tuple(
        NativeAssistantToolCall(
            call_id=f"call-{index}",
            official_name="wrong_submit_tool" if wrong else tool.official_name,
            wire_name="wrong_submit_tool" if wrong else tool.wire_name,
            arguments_json=canonical_json(arguments),
        )
        for index in range(calls)
    )
    return NativeFunctionCallResponse(
        request_fingerprint=request.fingerprint,
        serializer_fingerprint=request.serializer_fingerprint,
        parser_fingerprint=request.parser_fingerprint,
        transport_fingerprint=request.transport_fingerprint,
        tools_fingerprint=request.tools_fingerprint,
        tool_calls=tool_calls,
        assistant_text=assistant_text,
        finish_reason="tool_calls",
        usage=BackendTokenUsage(input_tokens=11, output_tokens=7),
    )


def _text_response(request, text="plain text"):
    return NativeFunctionCallResponse(
        request_fingerprint=request.fingerprint,
        serializer_fingerprint=request.serializer_fingerprint,
        parser_fingerprint=request.parser_fingerprint,
        transport_fingerprint=request.transport_fingerprint,
        tools_fingerprint=request.tools_fingerprint,
        tool_calls=(),
        assistant_text=text,
        finish_reason="stop",
        usage=BackendTokenUsage(input_tokens=11, output_tokens=7),
    )


def _diagnosis_event(campaign, runtime, node, call, result):
    provider = expected_provider_request(campaign, runtime, node, call.native_request.fingerprint)
    succeeded = result.native_response_fingerprint is not None
    return BfclV4PublicV2JournalEvent(
        sequence=node.node_slot,
        previous_event_sha256=_digest(f"previous-{node.node_slot}"),
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
        runtime_fingerprint=runtime.fingerprint,
        semantic_release_fingerprint=runtime.semantic_release_fingerprint,
        node_id=node.node_id,
        node_slot=node.node_slot,
        node_reference_sha256=canonical_sha256(node),
        event_kind=BfclV4PublicV2EventKind.CALL,
        request_fingerprint=provider.fingerprint,
        request_payload_sha256=call.native_request.fingerprint,
        provider_attempt_disposition=(
            BfclV4PublicV2AttemptDisposition.SUCCEEDED
            if succeeded
            else BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE
        ),
        provider_attempts_consumed=1,
        executed_harness_variant=node.harness_variant,
        canonical_response=result.journal_canonical_response,
        provider_response_fingerprint=result.native_response_fingerprint,
    )


def _diagnosis_flow(campaign, runtime, backend, *, arm, pipeline=0, valid=True):
    node = _node(
        campaign,
        arm=arm,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
        pipeline=pipeline,
    )
    events = _parent_events(campaign, runtime, node)
    projections = _projections(campaign, node) if arm is BfclV4PublicDevelopmentV2Arm.FULL else ()
    call = materialize_bfcl_v4_public_v2_diagnosis_request(
        campaign=campaign,
        runtime=runtime,
        node=node,
        evidence_events=events,
        fit_task_projections=projections,
        backend=backend,
    )
    response = (
        _native_response(
            call.native_request,
            BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,
            {"diagnosis": "general reusable cause"},
        )
        if valid
        else _text_response(call.native_request)
    )
    result = parse_bfcl_v4_public_v2_diagnosis_response(
        campaign=campaign,
        runtime=runtime,
        request=call,
        response=response,
    )
    return node, events, projections, call, result


def _proposal_flow(campaign, runtime, backend, *, pipeline, mutation_id, valid_diagnosis=True):
    diagnosis_node, _, _, diagnosis_call, diagnosis = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        pipeline=pipeline,
        valid=valid_diagnosis,
    )
    proposal_node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        pipeline=pipeline,
    )
    proposal_call = materialize_bfcl_v4_public_v2_proposal_request(
        campaign=campaign,
        runtime=runtime,
        node=proposal_node,
        diagnosis_request=diagnosis_call,
        diagnosis_result=diagnosis,
        diagnosis_event=_diagnosis_event(
            campaign,
            runtime,
            diagnosis_node,
            diagnosis_call,
            diagnosis,
        ),
        backend=backend,
    )
    response = _native_response(
        proposal_call.native_request,
        BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL,
        {"catalogue_id": mutation_id.value},
    )
    result = parse_bfcl_v4_public_v2_proposal_response(
        campaign=campaign,
        runtime=runtime,
        request=proposal_call,
        diagnosis_result=diagnosis,
        response=response,
    )
    return proposal_node, proposal_call, diagnosis, result


def test_score_payload_is_aggregate_only_and_backend_is_never_invoked(
    campaign, runtime, backend
) -> None:
    node, events, _, call, _ = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
    )
    payload = call.prompt.model_visible_payload

    assert type(payload) is BfclV4PublicV2ScoreDiagnosisPayload
    assert payload.binary_summary.observation_count == 10
    assert payload.binary_summary.binary_correct_count == 4
    assert set(payload.binary_summary.model_dump()) == {
        "observation_count",
        "binary_correct_count",
        "binary_incorrect_count",
    }
    assert "question" not in call.prompt.user_prompt
    assert "function_schemas" not in call.prompt.user_prompt
    assert all(event.canonical_response not in call.prompt.user_prompt for event in events)
    assert all(source_id not in call.prompt.user_prompt for source_id in node.allowed_evidence_from)
    assert tuple(message.role for message in call.native_request.messages) == ("system", "user")
    assert call.native_request.task_required_tools == (BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,)
    assert call.native_request.seed == node.provider_seed_u63
    assert backend.invocations == 0


def test_full_payload_contains_only_candidate_safe_observations(campaign, runtime, backend) -> None:
    node, events, projections, call, _ = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
    )
    payload = call.prompt.model_visible_payload
    prompt = call.prompt.user_prompt

    assert type(payload) is BfclV4PublicV2FullDiagnosisPayload
    assert tuple(item.observation_index for item in payload.fit_observations) == tuple(range(10))
    assert tuple(item.question_json for item in payload.fit_observations) == tuple(
        item.question_json for item in projections
    )
    assert tuple(item.own_canonical_response for item in payload.fit_observations) == tuple(
        item.canonical_response for item in events
    )
    assert all(source_id not in prompt for source_id in node.allowed_evidence_from)
    for forbidden in ("task_id", "possible_answer", "checker_diagnostics", '"roster"'):
        assert forbidden not in prompt
    assert set(payload.model_dump()) == {"controller", "feedback_view", "fit_observations"}


def test_feedback_views_reject_wrong_projection_coverage(campaign, runtime, backend) -> None:
    score = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="SCORE"):
        materialize_bfcl_v4_public_v2_diagnosis_request(
            campaign=campaign,
            runtime=runtime,
            node=score,
            evidence_events=_parent_events(campaign, runtime, score),
            fit_task_projections=_projections(campaign, score),
            backend=backend,
        )

    full = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="coverage"):
        materialize_bfcl_v4_public_v2_diagnosis_request(
            campaign=campaign,
            runtime=runtime,
            node=full,
            evidence_events=_parent_events(campaign, runtime, full),
            fit_task_projections=tuple(reversed(_projections(campaign, full))),
            backend=backend,
        )


def test_cross_arm_seed_and_gate_evidence_substitution_fail_closed(
    campaign, runtime, backend
) -> None:
    score = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    full = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    wrong = list(_parent_events(campaign, runtime, score))
    wrong[0] = _parent_events(campaign, runtime, full)[0]
    with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="coverage"):
        materialize_bfcl_v4_public_v2_diagnosis_request(
            campaign=campaign,
            runtime=runtime,
            node=score,
            evidence_events=tuple(wrong),
            backend=backend,
        )

    other_seed = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
        replicate_index=1,
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="coverage"):
        materialize_bfcl_v4_public_v2_diagnosis_request(
            campaign=campaign,
            runtime=runtime,
            node=score,
            evidence_events=_parent_events(campaign, runtime, other_seed),
            backend=backend,
        )

    gate = next(
        item for item in campaign.nodes if item.kind is BfclV4PublicDevelopmentV2NodeKind.GATE
    )
    forged = BfclV4PublicV2JournalEvent.model_construct(
        **{
            **wrong[0].model_dump(mode="python"),
            "node_id": gate.node_id,
            "node_slot": gate.node_slot,
            "sequence": gate.node_slot,
            "node_reference_sha256": canonical_sha256(gate),
        }
    )
    gate_sources = list(_parent_events(campaign, runtime, score))
    gate_sources[0] = forged
    with pytest.raises(BfclV4PublicV2MetaRuntimeError):
        materialize_bfcl_v4_public_v2_diagnosis_request(
            campaign=campaign,
            runtime=runtime,
            node=score,
            evidence_events=tuple(gate_sources),
            backend=backend,
        )


@pytest.mark.parametrize(
    ("response_factory", "failure"),
    [
        (lambda request: None, BfclV4PublicV2DiagnosisFailure.NO_VERIFIED_RESPONSE),
        (lambda request: object(), BfclV4PublicV2DiagnosisFailure.INVALID_RESPONSE_CONTRACT),
        (lambda request: _text_response(request), BfclV4PublicV2DiagnosisFailure.TEXT_ONLY),
        (
            lambda request: _native_response(
                request, BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL, {"diagnosis": "x"}, calls=2
            ),
            BfclV4PublicV2DiagnosisFailure.WRONG_CALL_COUNT,
        ),
        (
            lambda request: _native_response(
                request,
                BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,
                {"diagnosis": "x"},
                assistant_text="also text",
            ),
            BfclV4PublicV2DiagnosisFailure.ASSISTANT_TEXT_PRESENT,
        ),
        (
            lambda request: _native_response(
                request, BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL, {"diagnosis": "x"}, wrong=True
            ),
            BfclV4PublicV2DiagnosisFailure.WRONG_TOOL,
        ),
        (
            lambda request: _native_response(
                request,
                BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,
                {"diagnosis": "x", "extra": True},
            ),
            BfclV4PublicV2DiagnosisFailure.EXTRA_ARGUMENT_FIELDS,
        ),
    ],
)
def test_diagnosis_parser_is_total_and_strict(
    campaign, runtime, backend, response_factory, failure
) -> None:
    _, _, _, call, _ = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
    )
    result = parse_bfcl_v4_public_v2_diagnosis_response(
        campaign=campaign,
        runtime=runtime,
        request=call,
        response=response_factory(call.native_request),
    )
    assert result.valid is False
    assert result.failure is failure
    assert result.automatic_retry_used is False
    assert result.output_repair_used is False


def test_response_request_binding_is_checked(campaign, runtime, backend) -> None:
    _, _, _, call, _ = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
    )
    response = _native_response(
        call.native_request,
        BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL,
        {"diagnosis": "bound diagnosis"},
    ).model_copy(update={"request_fingerprint": _digest("foreign-request")})
    result = parse_bfcl_v4_public_v2_diagnosis_response(
        campaign=campaign, runtime=runtime, request=call, response=response
    )
    assert result.failure is BfclV4PublicV2DiagnosisFailure.RESPONSE_BINDING_MISMATCH


def test_proposal_sees_only_own_diagnosis_and_materializes_closed_id(
    campaign, runtime, backend
) -> None:
    node, call, diagnosis, result = _proposal_flow(
        campaign,
        runtime,
        backend,
        pipeline=0,
        mutation_id=BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR,
    )
    payload = call.prompt.model_visible_payload

    assert type(payload) is BfclV4PublicV2ProposalPayload
    assert payload.diagnosis.diagnosis == diagnosis.diagnosis_text
    assert payload.mutation_catalogue_ids == tuple(BfclV4PublicV2MutationId)
    assert set(payload.model_dump()) == {
        "controller",
        "feedback_view",
        "diagnosis",
        "mutation_catalogue_ids",
    }
    assert result.valid is True
    assert result.mutation_id is BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR
    assert result.materialization.candidate_ref != result.materialization.parent_ref
    assert result.runtime_batch.candidate_prompt_sha256 != (
        result.runtime_batch.static_parent_prompt_sha256
    )
    assert call.native_request.seed == node.provider_seed_u63


@pytest.mark.parametrize(
    ("arguments", "failure"),
    [
        ({"catalogue_id": "not-in-catalogue"}, BfclV4PublicV2ProposalFailure.UNKNOWN_CATALOGUE_ID),
        (
            {"catalogue_id": BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER.value, "x": 1},
            BfclV4PublicV2ProposalFailure.EXTRA_ARGUMENT_FIELDS,
        ),
        ({"catalogue_id": 3}, BfclV4PublicV2ProposalFailure.ARGUMENT_NOT_TEXT),
    ],
)
def test_proposal_parser_rejects_non_closed_or_extra_values(
    campaign, runtime, backend, arguments, failure
) -> None:
    _, call, diagnosis, _ = _proposal_flow(
        campaign,
        runtime,
        backend,
        pipeline=0,
        mutation_id=BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER,
    )
    result = parse_bfcl_v4_public_v2_proposal_response(
        campaign=campaign,
        runtime=runtime,
        request=call,
        diagnosis_result=diagnosis,
        response=_native_response(call.native_request, BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL, arguments),
    )
    assert result.valid is False
    assert result.failure is failure
    assert result.materialization is None


def test_three_pipeline_duplicate_closure_is_deterministic(campaign, runtime, backend) -> None:
    ids = (
        BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER,
        BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER,
        BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER,
    )
    results = tuple(
        _proposal_flow(
            campaign,
            runtime,
            backend,
            pipeline=index,
            mutation_id=mutation_id,
        )[3]
        for index, mutation_id in enumerate(ids)
    )
    batch = resolve_bfcl_v4_public_v2_proposal_batch(
        campaign=campaign,
        runtime=runtime,
        proposal_results=results,
    )
    replay = resolve_bfcl_v4_public_v2_proposal_batch(
        campaign=campaign,
        runtime=runtime,
        proposal_results=results,
    )

    assert tuple(item.disposition for item in batch.proposals) == (
        BfclV4PublicV2ProposalDisposition.VALID,
        BfclV4PublicV2ProposalDisposition.DUPLICATE,
        BfclV4PublicV2ProposalDisposition.VALID,
    )
    assert batch.proposals[1].duplicate_of_pipeline_index == 0
    assert batch.proposals[1].admitted_runtime_batch is None
    assert batch.fingerprint == replay.fingerprint


def test_invalid_diagnosis_cannot_become_admissible(campaign, runtime, backend) -> None:
    results = tuple(
        _proposal_flow(
            campaign,
            runtime,
            backend,
            pipeline=index,
            mutation_id=tuple(BfclV4PublicV2MutationId)[index],
            valid_diagnosis=index != 0,
        )[3]
        for index in range(3)
    )
    assert results[0].valid is True
    assert results[0].diagnosis_valid is False
    batch = resolve_bfcl_v4_public_v2_proposal_batch(
        campaign=campaign,
        runtime=runtime,
        proposal_results=results,
    )
    assert batch.proposals[0].disposition is BfclV4PublicV2ProposalDisposition.INVALID
    assert batch.proposals[0].admitted_runtime_batch is None


def test_noop_invalid_and_provider_failure_have_typed_dispositions(
    campaign, runtime, backend
) -> None:
    templates = tuple(
        _proposal_flow(
            campaign,
            runtime,
            backend,
            pipeline=index,
            mutation_id=tuple(BfclV4PublicV2MutationId)[index],
        )[3]
        for index in range(3)
    )
    failures = (
        BfclV4PublicV2ProposalFailure.NO_OP,
        BfclV4PublicV2ProposalFailure.WRONG_TOOL,
        BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE,
    )
    invalid = tuple(
        BfclV4PublicV2ProposalParseResult(
            target_node_id=template.target_node_id,
            pipeline_index=index,
            request_materialization_fingerprint=template.request_materialization_fingerprint,
            diagnosis_result_fingerprint=template.diagnosis_result_fingerprint,
            diagnosis_valid=True,
            native_response_fingerprint=(
                None
                if failure is BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE
                else _digest(f"response-{index}")
            ),
            journal_canonical_response=(
                None if failure is BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE else "{}"
            ),
            valid=False,
            failure=failure,
        )
        for index, (template, failure) in enumerate(zip(templates, failures, strict=True))
    )
    batch = resolve_bfcl_v4_public_v2_proposal_batch(
        campaign=campaign, runtime=runtime, proposal_results=invalid
    )
    assert tuple(item.disposition for item in batch.proposals) == (
        BfclV4PublicV2ProposalDisposition.NO_OP,
        BfclV4PublicV2ProposalDisposition.INVALID,
        BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE,
    )


def test_prompt_tool_seed_source_and_runtime_tampering_fail_closed(
    campaign, runtime, backend
) -> None:
    _, _, _, call, _ = _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
    )
    forged_prompt = BfclV4PublicV2MetaPrompt.model_construct(
        **{**call.prompt.model_dump(mode="python"), "system_prompt": "forged prompt"}
    )
    forged_call = BfclV4PublicV2MetaRequestMaterialization.model_construct(
        **{**call.model_dump(mode="python"), "prompt": forged_prompt}
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError):
        parse_bfcl_v4_public_v2_diagnosis_response(
            campaign=campaign, runtime=runtime, request=forged_call, response=None
        )

    foreign_tool = BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL
    tool_prompt = BfclV4PublicV2MetaPrompt.model_construct(
        **{
            **call.prompt.model_dump(mode="python"),
            "submit_tool": foreign_tool,
            "submit_tool_fingerprint": canonical_sha256(foreign_tool),
        }
    )
    tool_call = BfclV4PublicV2MetaRequestMaterialization.model_construct(
        **{**call.model_dump(mode="python"), "prompt": tool_prompt}
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError):
        parse_bfcl_v4_public_v2_diagnosis_response(
            campaign=campaign, runtime=runtime, request=tool_call, response=None
        )

    seeded_native = call.native_request.model_copy(update={"seed": call.native_request.seed + 1})
    seed_call = BfclV4PublicV2MetaRequestMaterialization.model_construct(
        **{
            **call.model_dump(mode="python"),
            "native_request": seeded_native,
            "native_request_fingerprint": seeded_native.fingerprint,
            "provider_request_payload_sha256": seeded_native.fingerprint,
        }
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError):
        parse_bfcl_v4_public_v2_diagnosis_response(
            campaign=campaign, runtime=runtime, request=seed_call, response=None
        )

    source_call = BfclV4PublicV2MetaRequestMaterialization.model_construct(
        **{**call.model_dump(mode="python"), "meta_runtime_source_fingerprint": "0" * 64}
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="changed"):
        parse_bfcl_v4_public_v2_diagnosis_response(
            campaign=campaign, runtime=runtime, request=source_call, response=None
        )

    forged_runtime = BfclV4PublicV2LiveExecutionConfig.model_construct(
        **{**runtime.model_dump(mode="python"), "parser_fingerprint": "0" * 64}
    )
    with pytest.raises(BfclV4PublicV2MetaRuntimeError):
        parse_bfcl_v4_public_v2_diagnosis_response(
            campaign=campaign, runtime=forged_runtime, request=call, response=None
        )


def test_private_node_and_evidence_lineage_tampering_fails_closed(
    campaign, runtime, backend
) -> None:
    _, call, diagnosis, _ = _proposal_flow(
        campaign,
        runtime,
        backend,
        pipeline=0,
        mutation_id=BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR,
    )
    cases = (
        ({"arm": BfclV4PublicDevelopmentV2Arm.FULL}, {"arm": BfclV4PublicDevelopmentV2Arm.FULL}),
        ({"allowed_evidence_node_ids": ("forged-evidence-node",)}, {}),
        ({"campaign_plan_fingerprint": "0" * 64}, {}),
        ({"runtime_fingerprint": "0" * 64}, {}),
        ({"semantic_release_fingerprint": "0" * 64}, {}),
        ({}, {"pipeline_index": 1}),
    )
    for evidence_updates, prompt_updates in cases:
        evidence = call.evidence_binding.model_copy(update=evidence_updates)
        prompt = call.prompt.model_copy(
            update={
                **prompt_updates,
                "evidence_binding_fingerprint": evidence.fingerprint,
            }
        )
        forged = call.model_copy(update={"evidence_binding": evidence, "prompt": prompt})
        with pytest.raises(BfclV4PublicV2MetaRuntimeError, match="changed"):
            parse_bfcl_v4_public_v2_proposal_response(
                campaign=campaign,
                runtime=runtime,
                request=forged,
                diagnosis_result=diagnosis,
                response=None,
            )


def test_proposal_payload_feedback_view_is_bound_to_prompt(campaign, runtime, backend) -> None:
    _, call, _, _ = _proposal_flow(
        campaign,
        runtime,
        backend,
        pipeline=0,
        mutation_id=BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR,
    )
    payload = call.prompt.model_visible_payload.model_copy(
        update={"feedback_view": (BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL.value)}
    )
    with pytest.raises(ValidationError, match="authorized feedback view"):
        BfclV4PublicV2MetaPrompt.model_validate(
            {
                **call.prompt.model_dump(mode="python"),
                "model_visible_payload": payload,
                "model_visible_payload_fingerprint": canonical_sha256(payload),
            },
            strict=True,
        )


def test_projection_extra_private_fields_are_unrepresentable(campaign) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    projection = _projections(campaign, node)[0]
    with pytest.raises(ValidationError):
        BfclV4PublicV2MetaFitTaskProjection.model_validate(
            {**projection.model_dump(mode="python"), "task_id": "private-task-id"},
            strict=True,
        )


def test_network_and_backend_invocation_traps_remain_idle(
    campaign, runtime, backend, monkeypatch
) -> None:
    def blocked_socket(*_args: object, **_kwargs: object):
        raise AssertionError("provider-free meta runtime attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    _diagnosis_flow(
        campaign,
        runtime,
        backend,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
    )
    assert backend.invocations == 0


def test_meta_production_modules_obey_size_and_import_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "spiral_harness/experiments/bfcl_v4_public_v2_meta_runtime.py",
        root / "spiral_harness/experiments/bfcl_v4_public_v2_meta_runtime_contracts.py",
        root / "spiral_harness/experiments/bfcl_v4_public_v2_meta_runtime_native.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 700
        assert "importlib" not in source
