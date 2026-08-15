"""Provider-free diagnosis/proposal runtime for the BFCL public-v2 DAG.

It validates supplied journal evidence, materializes immutable native requests,
parses supplied responses, and closes duplicate proposals.  It has no provider,
grader, benchmark loader, retry loop, or score authority.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic import BaseModel, ValidationError

import spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts as plan_contracts
import spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts as campaign_contracts  # noqa: E501
import spiral_harness.benchmark.bfcl_v4_public_v2_mutations as mutations
import spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts as executor_contracts
import spiral_harness.experiments.bfcl_v4_public_v2_meta_runtime_contracts as contracts
import spiral_harness.experiments.bfcl_v4_public_v2_meta_runtime_native as meta_native
import spiral_harness.providers.openai_native_contracts as native
from spiral_harness.core.canonical import (
    canonical_sha256,
    module_source_sha256,
    sha256_bytes,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2LiveExecutionConfig,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    materialize_bfcl_v4_public_v2_mutation_runtime_batch,
)
from spiral_harness.skills.package import RESERVED_SKILL_CONTEXT_DELIMITERS

BfclV4PublicV2MetaRuntimeError = meta_native.BfclV4PublicV2MetaRuntimeError
BfclV4PublicV2MetaBackendIdentity = meta_native.BfclV4PublicV2MetaBackendIdentity
BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL = meta_native.BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL
BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL = meta_native.BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL
BFCL_V4_PUBLIC_V2_DIAGNOSER_SYSTEM_PROMPT = meta_native.BFCL_V4_PUBLIC_V2_DIAGNOSER_SYSTEM_PROMPT
BFCL_V4_PUBLIC_V2_PROPOSER_SYSTEM_PROMPT = meta_native.BFCL_V4_PUBLIC_V2_PROPOSER_SYSTEM_PROMPT


def _reject(message: str) -> None:
    raise BfclV4PublicV2MetaRuntimeError(message) from None


def _strict[ModelT: BaseModel](model_type: type[ModelT], value: object, label: str) -> ModelT:
    if type(value) is not model_type:
        _reject(f"{label} must use the exact frozen contract")
    try:
        before = canonical_sha256(value)
        checked = model_type.model_validate(
            value.model_dump(mode="python", round_trip=True, warnings="none"),  # type: ignore[union-attr]
            strict=True,
        )
        if before != canonical_sha256(checked):
            raise ValueError("source hash changed")
        return checked
    except (RecursionError, TypeError, UnicodeError, ValidationError, ValueError):
        _reject(f"{label} differs from its strict frozen contract")


def bfcl_v4_public_v2_meta_runtime_source_fingerprint() -> str:
    """Bind both production sources without a dynamic import."""

    return canonical_sha256(
        {
            "protocol": "spiral-bfcl-v4-public-v2-meta-runtime/v1",
            "contracts": module_source_sha256(contracts),
            "native": module_source_sha256(meta_native),
            "runtime": module_source_sha256(sys.modules[__name__]),
        }
    )


@lru_cache(maxsize=8)
def _checked_campaign(
    value: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
) -> campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan:
    checked = _strict(
        campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
        value,
        "campaign",
    )
    if checked.fingerprint != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT:
        _reject("campaign fingerprint differs from the repository-frozen plan")
    return checked


@lru_cache(maxsize=8)
def _checked_live_runtime(
    value: BfclV4PublicV2LiveExecutionConfig,
) -> BfclV4PublicV2LiveExecutionConfig:
    return _strict(BfclV4PublicV2LiveExecutionConfig, value, "live runtime")


def _context(
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    node: plan_contracts.BfclV4PublicDevelopmentV2DagNode,
    kind: plan_contracts.BfclV4PublicDevelopmentV2NodeKind,
) -> tuple[
    campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    BfclV4PublicV2LiveExecutionConfig,
    plan_contracts.BfclV4PublicDevelopmentV2DagNode,
]:
    try:
        plan = _checked_campaign(campaign)
    except TypeError:
        plan = _strict(
            campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan, campaign, "campaign"
        )
        if plan.fingerprint != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT:
            _reject("campaign fingerprint differs from the repository-frozen plan")
    try:
        live = _checked_live_runtime(runtime)
    except TypeError:
        live = _strict(BfclV4PublicV2LiveExecutionConfig, runtime, "live runtime")
    target = _strict(plan_contracts.BfclV4PublicDevelopmentV2DagNode, node, "target node")
    matches = tuple(item for item in plan.nodes if item.node_id == target.node_id)
    if (
        plan.node_schedule_content_sha256
        != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
        or plan.mutation_catalog_fingerprint
        != mutations.BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.fingerprint
        or live.campaign_plan_fingerprint
        != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or live.node_schedule_content_sha256 != plan.node_schedule_content_sha256
        or live.semantic_release_ref.sha256 != live.semantic_release_fingerprint
        or live.model_spec_fingerprint != live.model_spec.fingerprint
        or len(matches) != 1
        or matches[0] != target
        or target.kind is not kind
        or target.arm
        not in {
            plan_contracts.BfclV4PublicDevelopmentV2Arm.SCORE,
            plan_contracts.BfclV4PublicDevelopmentV2Arm.FULL,
        }
        or target.task_ref is not None
        or target.pipeline_index is None
        or target.provider_seed_u63 is None
    ):
        _reject("campaign, runtime, release, model spec, or controller node binding changed")
    return plan, live, target


def _materialize(plan, live, node, backend_ids, evidence, payload, kind, diagnosis_sha=None):  # type: ignore[no-untyped-def]
    system, tool, grammar = meta_native.prompt_values(kind)
    user = meta_native.model_visible_user_prompt(payload, tool)
    prompt = contracts.BfclV4PublicV2MetaPrompt(
        controller_kind=kind,
        arm=node.arm,
        feedback_view=node.feedback_view,
        pipeline_index=node.pipeline_index,
        evidence_binding_fingerprint=evidence.fingerprint,
        model_visible_payload=payload,
        model_visible_payload_fingerprint=canonical_sha256(payload),
        system_prompt=system,
        user_prompt=user,
        user_prompt_sha256=sha256_bytes(user.encode()),
        submit_tool=tool,
        submit_tool_fingerprint=canonical_sha256(tool),
        output_grammar=grammar,
    )
    request = native.NativeFunctionCallRequest(
        backend_fingerprint=backend_ids[0],
        serializer_fingerprint=backend_ids[1],
        parser_fingerprint=backend_ids[2],
        transport_fingerprint=backend_ids[3],
        requested_model=live.model_spec.model,
        messages=(
            native.FrozenNativeChatMessage(role="system", content=system),
            native.FrozenNativeChatMessage(role="user", content=user),
        ),
        task_required_tools=(tool,),
        seed=node.provider_seed_u63,
        inference=live.model_spec.inference,
    )
    return contracts.BfclV4PublicV2MetaRequestMaterialization(
        campaign_plan_fingerprint=(
            plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        ),
        node_schedule_content_sha256=plan.node_schedule_content_sha256,
        mutation_catalog_fingerprint=plan.mutation_catalog_fingerprint,
        target_node_id=node.node_id,
        target_node_reference_sha256=canonical_sha256(node),
        runtime_fingerprint=live.fingerprint,
        semantic_release_ref=live.semantic_release_ref,
        semantic_release_fingerprint=live.semantic_release_fingerprint,
        model_spec=live.model_spec,
        model_spec_fingerprint=live.model_spec.fingerprint,
        meta_runtime_source_fingerprint=bfcl_v4_public_v2_meta_runtime_source_fingerprint(),
        source_diagnosis_result_fingerprint=diagnosis_sha,
        evidence_binding=evidence,
        prompt=prompt,
        native_request=request,
        native_request_fingerprint=request.fingerprint,
        provider_request_payload_sha256=request.fingerprint,
    )


def materialize_bfcl_v4_public_v2_diagnosis_request(
    *,
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    node: plan_contracts.BfclV4PublicDevelopmentV2DagNode,
    evidence_events: tuple[executor_contracts.BfclV4PublicV2JournalEvent, ...],
    backend: BfclV4PublicV2MetaBackendIdentity,
    fit_task_projections: tuple[contracts.BfclV4PublicV2MetaFitTaskProjection, ...] = (),
) -> contracts.BfclV4PublicV2MetaRequestMaterialization:
    """Build one diagnosis request from exact own-arm parent-FIT evidence."""

    plan, live, target = _context(
        campaign,
        runtime,
        node,
        plan_contracts.BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
    )
    if type(evidence_events) is not tuple or type(fit_task_projections) is not tuple:
        _reject("diagnosis sources must be exact tuples")
    if tuple(item.node_id for item in evidence_events) != target.allowed_evidence_from:
        _reject("diagnosis evidence coverage or order changed")
    by_id = {item.node_id: item for item in plan.nodes}
    events = []
    for raw, source_id in zip(evidence_events, target.allowed_evidence_from, strict=True):
        source = by_id[source_id]
        if (
            source.kind is not plan_contracts.BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT
            or source.arm is not target.arm
            or source.replicate_id != target.replicate_id
        ):
            _reject("diagnosis evidence crossed arm, seed, GATE, or HOLDOUT boundary")
        item = meta_native.validate_source_event(raw, source, plan, live)
        succeeded = (
            item.provider_attempt_disposition
            is executor_contracts.BfclV4PublicV2AttemptDisposition.SUCCEEDED
        )
        if (
            type(item.binary_grade) is not bool
            or item.proposal_disposition is not None
            or item.candidate_artifact_sha256 is not None
            or item.executed_harness_variant != "parent"
            or succeeded is not (item.canonical_response is not None)
            or (not succeeded and item.binary_grade)
            or succeeded
            is not (
                item.trusted_grade_request_fingerprint is not None
                and item.trusted_grader_receipt_fingerprint is not None
                and item.trusted_grade_attempts_consumed == 1
            )
        ):
            _reject("parent-FIT event lacks its exact terminal binary shape")
        events.append(item)

    projection_hashes: tuple[str, ...] = ()
    if target.arm is plan_contracts.BfclV4PublicDevelopmentV2Arm.SCORE:
        if fit_task_projections:
            _reject("SCORE cannot receive questions or schemas")
        payload = contracts.BfclV4PublicV2ScoreDiagnosisPayload(
            binary_summary=contracts.BfclV4PublicV2ScoreBinarySummary(
                binary_correct_count=sum(item.binary_grade is True for item in events),
                binary_incorrect_count=sum(item.binary_grade is False for item in events),
            )
        )
    else:
        if (
            tuple(item.source_node_id for item in fit_task_projections)
            != target.allowed_evidence_from
        ):
            _reject("FULL projection coverage or order changed")
        projections = tuple(
            _strict(contracts.BfclV4PublicV2MetaFitTaskProjection, item, "FIT projection")
            for item in fit_task_projections
        )
        for item, event, source_id in zip(
            projections, events, target.allowed_evidence_from, strict=True
        ):
            if (
                item.source_node_reference_sha256 != canonical_sha256(by_id[source_id])
                or item.source_request_payload_sha256 != event.request_payload_sha256
            ):
                _reject("FULL projection belongs to another event")
        projection_hashes = tuple(item.fingerprint for item in projections)
        payload = contracts.BfclV4PublicV2FullDiagnosisPayload(
            fit_observations=tuple(
                contracts.BfclV4PublicV2FullVisibleObservation(
                    observation_index=index,
                    question_json=projection.question_json,
                    function_schemas_json=projection.function_schemas_json,
                    own_canonical_response=event.canonical_response,
                    binary_grade=event.binary_grade,
                )
                for index, (projection, event) in enumerate(zip(projections, events, strict=True))
            )
        )
    evidence = contracts.BfclV4PublicV2MetaEvidenceBinding(
        controller_kind=contracts.BfclV4PublicV2MetaControllerKind.DIAGNOSIS,
        arm=target.arm,
        feedback_view=target.feedback_view,
        target_node_id=target.node_id,
        target_node_reference_sha256=canonical_sha256(target),
        allowed_evidence_node_ids=target.allowed_evidence_from,
        source_event_fingerprints=tuple(item.fingerprint for item in events),
        source_projection_fingerprints=projection_hashes,
        campaign_plan_fingerprint=(
            plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        ),
        runtime_fingerprint=live.fingerprint,
        semantic_release_fingerprint=live.semantic_release_fingerprint,
    )
    return _materialize(
        plan,
        live,
        target,
        meta_native.backend_identities(backend, live),
        evidence,
        payload,
        contracts.BfclV4PublicV2MetaControllerKind.DIAGNOSIS,
    )


def _checked_request(value, campaign, runtime, kind):  # type: ignore[no-untyped-def]
    call = _strict(contracts.BfclV4PublicV2MetaRequestMaterialization, value, "meta request")
    expected_kind = {
        contracts.BfclV4PublicV2MetaControllerKind.DIAGNOSIS: (
            plan_contracts.BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS
        ),
        contracts.BfclV4PublicV2MetaControllerKind.PROPOSAL: (
            plan_contracts.BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
        ),
    }[kind]
    selected = next((item for item in campaign.nodes if item.node_id == call.target_node_id), None)
    plan, live, node = _context(campaign, runtime, selected, expected_kind)
    system, tool, grammar = meta_native.prompt_values(kind)
    user = meta_native.model_visible_user_prompt(call.prompt.model_visible_payload, tool)
    expected = native.NativeFunctionCallRequest(
        backend_fingerprint=live.backend_fingerprint,
        serializer_fingerprint=live.serializer_fingerprint,
        parser_fingerprint=live.parser_fingerprint,
        transport_fingerprint=live.transport_fingerprint,
        requested_model=live.model_spec.model,
        messages=(
            native.FrozenNativeChatMessage(role="system", content=system),
            native.FrozenNativeChatMessage(role="user", content=user),
        ),
        task_required_tools=(tool,),
        seed=node.provider_seed_u63,
        inference=live.model_spec.inference,
    )
    if (
        call.campaign_plan_fingerprint
        != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or call.node_schedule_content_sha256 != plan.node_schedule_content_sha256
        or call.mutation_catalog_fingerprint != plan.mutation_catalog_fingerprint
        or call.target_node_reference_sha256 != canonical_sha256(node)
        or call.runtime_fingerprint != live.fingerprint
        or call.semantic_release_ref != live.semantic_release_ref
        or call.semantic_release_fingerprint != live.semantic_release_fingerprint
        or call.model_spec != live.model_spec
        or call.evidence_binding.arm is not node.arm
        or call.evidence_binding.feedback_view is not node.feedback_view
        or call.evidence_binding.allowed_evidence_node_ids != node.allowed_evidence_from
        or call.evidence_binding.campaign_plan_fingerprint
        != plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or call.evidence_binding.runtime_fingerprint != live.fingerprint
        or call.evidence_binding.semantic_release_fingerprint != live.semantic_release_fingerprint
        or call.meta_runtime_source_fingerprint
        != bfcl_v4_public_v2_meta_runtime_source_fingerprint()
        or call.prompt.controller_kind is not kind
        or call.prompt.arm is not node.arm
        or call.prompt.feedback_view is not node.feedback_view
        or call.prompt.pipeline_index != node.pipeline_index
        or call.prompt.system_prompt != system
        or call.prompt.user_prompt != user
        or call.prompt.submit_tool != tool
        or call.prompt.output_grammar != grammar
        or call.native_request != expected
    ):
        _reject("meta request changed after deterministic materialization")
    return call, node, plan, live


def _diagnosis_failure(call, fingerprint, journal, failure):  # type: ignore[no-untyped-def]
    return contracts.BfclV4PublicV2DiagnosisParseResult(
        target_node_id=call.target_node_id,
        request_materialization_fingerprint=call.fingerprint,
        native_response_fingerprint=fingerprint,
        journal_canonical_response=journal,
        valid=False,
        failure=failure,
    )


def parse_bfcl_v4_public_v2_diagnosis_response(
    *,
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    request: contracts.BfclV4PublicV2MetaRequestMaterialization,
    response: object,
) -> contracts.BfclV4PublicV2DiagnosisParseResult:
    """Strictly parse one supplied response without repair or retry."""

    call, _, _, _ = _checked_request(
        request,
        campaign,
        runtime,
        contracts.BfclV4PublicV2MetaControllerKind.DIAGNOSIS,
    )
    value, fingerprint, journal, error = meta_native.extract_submit_argument(
        response, call.native_request, BFCL_V4_PUBLIC_V2_DIAGNOSIS_TOOL, "diagnosis"
    )
    if error is not None:
        return _diagnosis_failure(
            call, fingerprint, journal, contracts.BfclV4PublicV2DiagnosisFailure(error)
        )
    if not isinstance(value, str):
        failure = contracts.BfclV4PublicV2DiagnosisFailure.ARGUMENT_NOT_TEXT
    elif not value.strip(" \t\n"):
        failure = contracts.BfclV4PublicV2DiagnosisFailure.EMPTY_DIAGNOSIS
    elif len(value.encode()) > 12_000:
        failure = contracts.BfclV4PublicV2DiagnosisFailure.DIAGNOSIS_TOO_LARGE
    elif any((ord(char) < 32 and char not in {"\t", "\n"}) or char == "\x7f" for char in value):
        failure = contracts.BfclV4PublicV2DiagnosisFailure.INVALID_CONTROL_CHARACTER
    elif any(delimiter in value for delimiter in RESERVED_SKILL_CONTEXT_DELIMITERS):
        failure = contracts.BfclV4PublicV2DiagnosisFailure.FORBIDDEN_DELIMITER
    else:
        return contracts.BfclV4PublicV2DiagnosisParseResult(
            target_node_id=call.target_node_id,
            request_materialization_fingerprint=call.fingerprint,
            native_response_fingerprint=fingerprint,
            journal_canonical_response=journal,
            valid=True,
            failure=contracts.BfclV4PublicV2DiagnosisFailure.NONE,
            diagnosis_text=value,
            diagnosis_text_sha256=sha256_bytes(value.encode()),
        )
    return _diagnosis_failure(call, fingerprint, journal, failure)


def _diagnosis_result(value, call):  # type: ignore[no-untyped-def]
    result = _strict(contracts.BfclV4PublicV2DiagnosisParseResult, value, "diagnosis result")
    if result.target_node_id != call.target_node_id or (
        result.request_materialization_fingerprint != call.fingerprint
    ):
        _reject("diagnosis result belongs to another request")
    return result


def materialize_bfcl_v4_public_v2_proposal_request(
    *,
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    node: plan_contracts.BfclV4PublicDevelopmentV2DagNode,
    diagnosis_request: contracts.BfclV4PublicV2MetaRequestMaterialization,
    diagnosis_result: contracts.BfclV4PublicV2DiagnosisParseResult,
    diagnosis_event: executor_contracts.BfclV4PublicV2JournalEvent,
    backend: BfclV4PublicV2MetaBackendIdentity,
) -> contracts.BfclV4PublicV2MetaRequestMaterialization:
    """Build one proposal from only its own frozen diagnosis event."""

    plan, live, target = _context(
        campaign,
        runtime,
        node,
        plan_contracts.BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
    )
    diagnosis_call, source, _, _ = _checked_request(
        diagnosis_request,
        plan,
        live,
        contracts.BfclV4PublicV2MetaControllerKind.DIAGNOSIS,
    )
    result = _diagnosis_result(diagnosis_result, diagnosis_call)
    if target.allowed_evidence_from != (source.node_id,) or (
        target.arm is not source.arm
        or target.replicate_id != source.replicate_id
        or target.pipeline_index != source.pipeline_index
    ):
        _reject("proposal crossed arm, seed, or pipeline diagnosis boundary")
    event = meta_native.validate_source_event(diagnosis_event, source, plan, live)
    provider = meta_native.expected_provider_request(
        plan, live, source, diagnosis_call.native_request.fingerprint
    )
    succeeded = (
        event.provider_attempt_disposition
        is executor_contracts.BfclV4PublicV2AttemptDisposition.SUCCEEDED
    )
    if (
        event.request_payload_sha256 != diagnosis_call.native_request.fingerprint
        or event.request_fingerprint != provider.fingerprint
        or event.binary_grade is not None
        or event.proposal_disposition is not None
        or event.candidate_artifact_sha256 is not None
        or event.canonical_response != result.journal_canonical_response
        or event.provider_response_fingerprint != result.native_response_fingerprint
        or event.executed_harness_variant != source.harness_variant
        or succeeded is not (result.native_response_fingerprint is not None)
    ):
        _reject("proposal diagnosis event differs from its parsed response")
    payload = contracts.BfclV4PublicV2ProposalPayload(
        feedback_view=target.feedback_view.value,
        diagnosis=contracts.BfclV4PublicV2VisibleDiagnosisResult(
            valid=result.valid,
            failure=result.failure,
            diagnosis=result.diagnosis_text,
        ),
    )
    evidence = contracts.BfclV4PublicV2MetaEvidenceBinding(
        controller_kind=contracts.BfclV4PublicV2MetaControllerKind.PROPOSAL,
        arm=target.arm,
        feedback_view=target.feedback_view,
        target_node_id=target.node_id,
        target_node_reference_sha256=canonical_sha256(target),
        allowed_evidence_node_ids=target.allowed_evidence_from,
        source_event_fingerprints=(event.fingerprint,),
        source_projection_fingerprints=(),
        campaign_plan_fingerprint=(
            plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        ),
        runtime_fingerprint=live.fingerprint,
        semantic_release_fingerprint=live.semantic_release_fingerprint,
    )
    return _materialize(
        plan,
        live,
        target,
        meta_native.backend_identities(backend, live),
        evidence,
        payload,
        contracts.BfclV4PublicV2MetaControllerKind.PROPOSAL,
        result.fingerprint,
    )


def _proposal_failure(call, diagnosis, pipeline, fingerprint, journal, failure):  # type: ignore[no-untyped-def]
    return contracts.BfclV4PublicV2ProposalParseResult(
        target_node_id=call.target_node_id,
        pipeline_index=pipeline,
        request_materialization_fingerprint=call.fingerprint,
        diagnosis_result_fingerprint=diagnosis.fingerprint,
        diagnosis_valid=diagnosis.valid,
        native_response_fingerprint=fingerprint,
        journal_canonical_response=journal,
        valid=False,
        failure=failure,
    )


def parse_bfcl_v4_public_v2_proposal_response(
    *,
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    request: contracts.BfclV4PublicV2MetaRequestMaterialization,
    diagnosis_result: contracts.BfclV4PublicV2DiagnosisParseResult,
    response: object,
) -> contracts.BfclV4PublicV2ProposalParseResult:
    """Parse one closed ID and materialize its deterministic atomic runtime."""

    call, node, _, _ = _checked_request(
        request,
        campaign,
        runtime,
        contracts.BfclV4PublicV2MetaControllerKind.PROPOSAL,
    )
    diagnosis = _strict(contracts.BfclV4PublicV2DiagnosisParseResult, diagnosis_result, "diagnosis")
    if call.source_diagnosis_result_fingerprint != diagnosis.fingerprint:
        _reject("proposal request belongs to another diagnosis")
    value, fingerprint, journal, error = meta_native.extract_submit_argument(
        response, call.native_request, BFCL_V4_PUBLIC_V2_PROPOSAL_TOOL, "catalogue_id"
    )
    if error is not None:
        failure = contracts.BfclV4PublicV2ProposalFailure(error)
    elif not isinstance(value, str):
        failure = contracts.BfclV4PublicV2ProposalFailure.ARGUMENT_NOT_TEXT
    else:
        try:
            mutation_id = mutations.BfclV4PublicV2MutationId(value)
        except ValueError:
            failure = contracts.BfclV4PublicV2ProposalFailure.UNKNOWN_CATALOGUE_ID
        else:
            try:
                proposal = mutations.BfclV4PublicV2MutationProposal(catalogue_id=mutation_id)
                materialization = mutations.materialize_bfcl_v4_public_v2_mutation(proposal)
                batch = materialize_bfcl_v4_public_v2_mutation_runtime_batch(materialization)
                if materialization.parent_ref == materialization.candidate_ref or (
                    batch.static_parent_prompt_sha256 == batch.candidate_prompt_sha256
                ):
                    failure = contracts.BfclV4PublicV2ProposalFailure.NO_OP
                else:
                    return contracts.BfclV4PublicV2ProposalParseResult(
                        target_node_id=call.target_node_id,
                        pipeline_index=node.pipeline_index,
                        request_materialization_fingerprint=call.fingerprint,
                        diagnosis_result_fingerprint=diagnosis.fingerprint,
                        diagnosis_valid=diagnosis.valid,
                        native_response_fingerprint=fingerprint,
                        journal_canonical_response=journal,
                        valid=True,
                        failure=contracts.BfclV4PublicV2ProposalFailure.NONE,
                        mutation_id=mutation_id,
                        proposal=proposal,
                        materialization=materialization,
                        runtime_batch=batch,
                    )
            except (RecursionError, TypeError, ValidationError, ValueError):
                failure = contracts.BfclV4PublicV2ProposalFailure.MATERIALIZATION_INVALID
    return _proposal_failure(call, diagnosis, node.pipeline_index, fingerprint, journal, failure)


def resolve_bfcl_v4_public_v2_proposal_batch(
    *,
    campaign: campaign_contracts.BfclV4PublicDevelopmentV2CampaignPlan,
    runtime: BfclV4PublicV2LiveExecutionConfig,
    proposal_results: tuple[contracts.BfclV4PublicV2ProposalParseResult, ...],
) -> contracts.BfclV4PublicV2ResolvedProposalBatch:
    """Close duplicates only after all three parallel proposals freeze."""

    if type(proposal_results) is not tuple or len(proposal_results) != 3:
        _reject("duplicate closure requires exactly three results")
    results = tuple(
        _strict(contracts.BfclV4PublicV2ProposalParseResult, item, "proposal result")
        for item in proposal_results
    )
    selected = tuple(
        next((node for node in campaign.nodes if node.node_id == item.target_node_id), None)
        for item in results
    )
    contexts = tuple(
        _context(
            campaign,
            runtime,
            node,
            plan_contracts.BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        )
        for node in selected
    )
    _, live, first = contexts[0]
    nodes = tuple(item[2] for item in contexts)
    if tuple(node.pipeline_index for node in nodes) != (0, 1, 2) or any(
        node.arm is not first.arm or node.replicate_id != first.replicate_id for node in nodes
    ):
        _reject("proposal batch must be one arm/seed in pipeline order")
    seen: dict[mutations.BfclV4PublicV2MutationId, int] = {}
    resolved = []
    for result in results:
        admissible = result.valid and result.diagnosis_valid
        duplicate_of = seen.get(result.mutation_id) if admissible else None
        if duplicate_of is not None:
            disposition = executor_contracts.BfclV4PublicV2ProposalDisposition.DUPLICATE
        elif admissible:
            disposition = executor_contracts.BfclV4PublicV2ProposalDisposition.VALID
            seen[result.mutation_id] = result.pipeline_index
        elif result.failure is contracts.BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE:
            disposition = executor_contracts.BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
        elif result.failure is contracts.BfclV4PublicV2ProposalFailure.NO_OP:
            disposition = executor_contracts.BfclV4PublicV2ProposalDisposition.NO_OP
        else:
            disposition = executor_contracts.BfclV4PublicV2ProposalDisposition.INVALID
        admitted = disposition is executor_contracts.BfclV4PublicV2ProposalDisposition.VALID
        materialization = result.materialization if admitted else None
        resolved.append(
            contracts.BfclV4PublicV2ResolvedAtomicMutation(
                pipeline_index=result.pipeline_index,
                proposal_node_id=result.target_node_id,
                parse_result=result,
                parse_result_fingerprint=result.fingerprint,
                disposition=disposition,
                mutation_id=result.mutation_id,
                duplicate_of_pipeline_index=duplicate_of,
                admitted_materialization=materialization,
                admitted_runtime_batch=result.runtime_batch if admitted else None,
                candidate_artifact_sha256=(
                    materialization.candidate_ref.sha256 if materialization is not None else None
                ),
            )
        )
    return contracts.BfclV4PublicV2ResolvedProposalBatch(
        campaign_plan_fingerprint=(
            plan_contracts.BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        ),
        runtime_fingerprint=live.fingerprint,
        semantic_release_fingerprint=live.semantic_release_fingerprint,
        replicate_id=first.replicate_id,
        arm=first.arm,
        proposals=tuple(resolved),
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("bfcl_v4")
    or name.startswith("materialize_bfcl")
    or name.startswith("parse_bfcl")
    or name.startswith("resolve_bfcl")
]
