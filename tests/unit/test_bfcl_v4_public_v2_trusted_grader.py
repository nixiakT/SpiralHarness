from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2Split,
    BfclV4PublicDevelopmentV2Task,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader import (
    BfclV4PublicV2EvaluationAuthorizationError,
    BfclV4PublicV2TrustedGrader,
    grade_bfcl_v4_public_v2_response,
    open_bfcl_v4_public_v2_trusted_grader,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
    BfclV4PublicV2TrustedGradeRequest,
    BfclV4PublicV2TrustedGraderReceipt,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.execution.contracts import BackendTokenUsage
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeChatMessage,
    FrozenNativeFunctionTool,
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SEMANTIC_RELEASE = "a" * 64
_AUTHORITY_SECRET = b"bfcl-v2-test-evaluation-authority-secret"


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact-source integration")
    return checkout


@pytest.fixture(scope="module")
def campaign() -> BfclV4PublicDevelopmentV2CampaignPlan:
    return build_bfcl_v4_public_development_v2_campaign_plan()


@pytest.fixture(scope="module")
def loaded(pinned_checkout: Path) -> BfclV4LoadedPublicDevelopmentV2:
    return load_bfcl_v4_public_development_v2(pinned_checkout)


@pytest.fixture(scope="module")
def grader(
    pinned_checkout: Path,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> BfclV4PublicV2TrustedGrader:
    return open_bfcl_v4_public_v2_trusted_grader(pinned_checkout, campaign)


@pytest.fixture(scope="module")
def authorized_grader(
    pinned_checkout: Path,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> BfclV4PublicV2TrustedGrader:
    return open_bfcl_v4_public_v2_trusted_grader(
        pinned_checkout,
        campaign,
        evaluation_authority_secret=_AUTHORITY_SECRET,
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
    )


def _node(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    kind: BfclV4PublicDevelopmentV2NodeKind,
    task_ref: str,
) -> BfclV4PublicDevelopmentV2DagNode:
    return next(node for node in campaign.nodes if node.kind is kind and node.task_ref == task_ref)


def _task(
    loaded: BfclV4LoadedPublicDevelopmentV2,
    task_ref: str,
) -> BfclV4PublicDevelopmentV2Task:
    split_name, raw_ordinal = task_ref.split("-", maxsplit=1)
    split = BfclV4PublicDevelopmentV2Split(split_name)
    tasks = tuple(
        task
        for task, entry in zip(loaded.tasks, loaded.manifest.roster, strict=True)
        if entry.split is split
    )
    return tasks[int(raw_ordinal)]


def _native_request(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node: BfclV4PublicDevelopmentV2DagNode,
    task: BfclV4PublicDevelopmentV2Task,
    *,
    requested_model: str | None = None,
) -> NativeFunctionCallRequest:
    schemas = json.loads(task.function_schemas_json)
    tools = tuple(
        FrozenNativeFunctionTool.from_schema(schema, wire_name=f"task_tool_{index}")
        for index, schema in enumerate(schemas)
    )
    assert node.provider_seed_u63 is not None
    return NativeFunctionCallRequest(
        backend_fingerprint="1" * 64,
        serializer_fingerprint="2" * 64,
        parser_fingerprint="3" * 64,
        transport_fingerprint="4" * 64,
        requested_model=(
            campaign.execution_profile.model_route if requested_model is None else requested_model
        ),
        messages=(FrozenNativeChatMessage(role="user", content=task.question_json),),
        task_required_tools=tools,
        seed=node.provider_seed_u63,
        inference=campaign.execution_profile.inference,
    )


def _native_response(
    request: NativeFunctionCallRequest,
    calls: tuple[tuple[str, dict[str, Any]], ...],
    *,
    input_tokens: int = 10,
    output_tokens: int = 10,
) -> NativeFunctionCallResponse:
    tools = {tool.official_name: tool for tool in request.task_required_tools}
    native_calls = tuple(
        NativeAssistantToolCall(
            call_id=f"call-{index}",
            official_name=name,
            wire_name=tools[name].wire_name,
            arguments_json=canonical_json(arguments),
        )
        for index, (name, arguments) in enumerate(calls)
    )
    return NativeFunctionCallResponse(
        request_fingerprint=request.fingerprint,
        serializer_fingerprint=request.serializer_fingerprint,
        parser_fingerprint=request.parser_fingerprint,
        transport_fingerprint=request.transport_fingerprint,
        tools_fingerprint=request.tools_fingerprint,
        tool_calls=native_calls,
        assistant_text=None if native_calls else "No function call.",
        finish_reason="tool_calls" if native_calls else "stop",
        usage=BackendTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _grade_request(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
    *,
    kind: BfclV4PublicDevelopmentV2NodeKind,
    task_ref: str,
    calls: tuple[tuple[str, dict[str, Any]], ...] = (),
    requested_model: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 10,
) -> BfclV4PublicV2TrustedGradeRequest:
    node = _node(campaign, kind, task_ref)
    task = _task(loaded, task_ref)
    request = _native_request(
        campaign,
        node,
        task,
        requested_model=requested_model,
    )
    response = _native_response(
        request,
        calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return BfclV4PublicV2TrustedGradeRequest(
        node=node,
        node_reference_sha256=canonical_sha256(node),
        request_lineage=derive_bfcl_v4_public_development_v2_node_request_lineage(
            campaign=campaign,
            node_id=node.node_id,
        ),
        task_payload_sha256=task.candidate_payload_sha256,
        request=request,
        request_fingerprint=request.fingerprint,
        request_payload_sha256=request.fingerprint,
        raw_response=response,
        response_fingerprint=response.fingerprint,
    )


def _barrier_evidence(
    grader: BfclV4PublicV2TrustedGrader,
) -> BfclV4PublicV2DecisionBarrierEvidence:
    events = tuple(f"{index + 1:064x}" for index in range(6))
    return BfclV4PublicV2DecisionBarrierEvidence(
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
        decision_node_references=grader.decision_node_references,
        decision_event_fingerprints=events,
        final_decision_event_fingerprint=events[-1],
    )


def _recursive_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(
            token for key, item in value.items() for token in (str(key), *_recursive_strings(item))
        )
    if isinstance(value, (list, tuple)):
        return tuple(token for item in value for token in _recursive_strings(item))
    return (str(value),)


def test_fit_exact_upstream_checker_accepts_known_public_call(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
        calls=(("calculate_daily_water_intake", {"weight": 70}),),
    )

    receipt = grade_bfcl_v4_public_v2_response(grader, request)

    assert receipt.correct is True
    assert receipt.split_role is BfclV4PublicDevelopmentV2Split.FIT
    assert receipt.task_ref == "fit-00"
    assert receipt.evaluation_unlock_fingerprint is None
    assert receipt.exact_upstream_ast_checker_executed is True
    assert receipt.score_bearing_execution_allowed is False


def test_wrong_fit_and_gate_calls_return_only_binary_false(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    fit = grader.grade(
        _grade_request(
            campaign,
            loaded,
            kind=BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
            task_ref="fit-01",
        )
    )
    gate = grader.grade(
        _grade_request(
            campaign,
            loaded,
            kind=BfclV4PublicDevelopmentV2NodeKind.GATE,
            task_ref="gate-00",
        )
    )

    assert fit.correct is False
    assert gate.correct is False
    assert fit.split_role is BfclV4PublicDevelopmentV2Split.FIT
    assert gate.split_role is BfclV4PublicDevelopmentV2Split.GATE
    assert fit.checker_diagnostics_present is False
    assert gate.checker_diagnostics_present is False


def test_receipt_omits_task_id_answer_identity_and_checker_diagnostics(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    receipt = grader.grade(
        _grade_request(
            campaign,
            loaded,
            kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
            task_ref="fit-00",
        )
    )
    payload = receipt.model_dump(mode="json")
    serialized = canonical_json(payload)
    strings = _recursive_strings(payload)

    assert receipt.candidate_visible is False
    assert receipt.answers_present is False
    assert receipt.answer_derived_identities_present is False
    assert receipt.task_id_present is False
    assert "simple_python_198" not in strings
    assert "calculate_daily_water_intake" not in strings
    assert "ground_truth" not in serialized
    assert "answer_blob" not in serialized
    assert "error_type" not in serialized
    assert {key for key in payload if "diagnostic" in key} == {"checker_diagnostics_present"}


def test_holdout_is_blocked_without_authenticated_global_barrier(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        task_ref="holdout-00",
    )

    with pytest.raises(
        BfclV4PublicV2EvaluationAuthorizationError,
        match="authenticated global decision barrier",
    ):
        grader.grade(request)


def test_hmac_unlock_binds_six_decisions_and_authorizes_holdout(
    authorized_grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    unlock = authorized_grader.issue_evaluation_unlock(_barrier_evidence(authorized_grader))
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        task_ref="holdout-00",
    )

    receipt = authorized_grader.grade(request, evaluation_unlock=unlock)

    assert receipt.correct is False
    assert receipt.split_role is BfclV4PublicDevelopmentV2Split.HOLDOUT
    assert receipt.evaluation_unlock_fingerprint == unlock.fingerprint


def test_tampered_unlock_and_wrong_decision_barrier_fail_closed(
    authorized_grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    evidence = _barrier_evidence(authorized_grader)
    unlock = authorized_grader.issue_evaluation_unlock(evidence)
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        task_ref="holdout-00",
    )
    tampered = unlock.model_copy(update={"authentication_tag_hmac_sha256": "0" * 64})
    wrong_references = ("0" * 64, *authorized_grader.decision_node_references[1:])
    wrong_evidence = evidence.model_copy(update={"decision_node_references": wrong_references})

    with pytest.raises(
        BfclV4PublicV2EvaluationAuthorizationError,
        match="authentication failed",
    ):
        authorized_grader.grade(request, evaluation_unlock=tampered)
    with pytest.raises(
        BfclV4PublicV2EvaluationAuthorizationError,
        match="does not match",
    ):
        authorized_grader.issue_evaluation_unlock(wrong_evidence)


def test_non_holdout_rejects_even_authentic_unlock(
    authorized_grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    unlock = authorized_grader.issue_evaluation_unlock(_barrier_evidence(authorized_grader))
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
    )

    with pytest.raises(ValueError, match="non-HOLDOUT"):
        authorized_grader.grade(request, evaluation_unlock=unlock)


def test_lineage_task_and_response_tampering_are_revalidated(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
    )
    wrong_lineage = request.request_lineage.model_copy(
        update={"campaign_plan_fingerprint": "0" * 64}
    )
    wrong_response = request.raw_response.model_copy(update={"request_fingerprint": "0" * 64})

    with pytest.raises(ValueError, match="revalidation failed"):
        grader.grade(request.model_copy(update={"request_lineage": wrong_lineage}))
    with pytest.raises(ValueError, match="revalidation failed"):
        grader.grade(request.model_copy(update={"raw_response": wrong_response}))
    with pytest.raises(ValueError, match="another task payload"):
        grader.grade(request.model_copy(update={"task_payload_sha256": "0" * 64}))


def test_same_model_and_total_token_budget_are_enforced(
    grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    wrong_model = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
        requested_model="another-model",
    )
    over_total = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
        input_tokens=32_000,
        output_tokens=1_000,
    )

    with pytest.raises(ValueError, match="same-model profile"):
        grader.grade(wrong_model)
    with pytest.raises(ValueError, match="total-token ceiling"):
        grader.grade(over_total)


def test_answer_or_split_injection_is_not_representable(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
    )
    payload = request.model_dump(mode="python")
    payload["ground_truth"] = [{"forged": True}]
    payload["task_id"] = "simple_python_198"
    payload["split"] = "fit"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BfclV4PublicV2TrustedGradeRequest.model_validate(payload, strict=True)


def test_receipt_contract_rejects_diagnostic_or_answer_fields() -> None:
    payload = {
        "node_reference_sha256": "1" * 64,
        "campaign_call_slot": 0,
        "task_ref": "fit-00",
        "task_reference_sha256": "2" * 64,
        "split_role": BfclV4PublicDevelopmentV2Split.FIT,
        "request_fingerprint": "3" * 64,
        "response_fingerprint": "4" * 64,
        "loaded_question_bundle_fingerprint": "5" * 64,
        "grader_source_sha256": "6" * 64,
        "correct": False,
        "answer_sha256": "7" * 64,
        "checker_error_type": "wrong_count",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BfclV4PublicV2TrustedGraderReceipt.model_validate(payload, strict=True)


def test_unlock_contract_never_contains_secret_or_answers(
    authorized_grader: BfclV4PublicV2TrustedGrader,
) -> None:
    unlock = authorized_grader.issue_evaluation_unlock(_barrier_evidence(authorized_grader))
    payload = unlock.model_dump(mode="json")
    serialized = canonical_json(payload)

    assert unlock.authority_secret_present is False
    assert unlock.answers_present is False
    assert _AUTHORITY_SECRET.decode() not in serialized
    assert "ground_truth" not in serialized


def test_import_does_not_open_worker_or_upstream_checker_graph() -> None:
    code = """
import sys
import spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader
for name in sorted(sys.modules):
    if name.startswith('bfcl_eval') or 'trusted_grader_worker' in name:
        print(name)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_public_facade_requires_real_trusted_grader(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        task_ref="fit-00",
    )

    with pytest.raises(TypeError, match="trusted grader"):
        grade_bfcl_v4_public_v2_response(object(), request)  # type: ignore[arg-type]


def test_evaluation_authority_configuration_is_atomic(
    pinned_checkout: Path,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> None:
    with pytest.raises(ValueError, match="configured together"):
        open_bfcl_v4_public_v2_trusted_grader(
            pinned_checkout,
            campaign,
            evaluation_authority_secret=_AUTHORITY_SECRET,
        )
    with pytest.raises(ValueError, match="too short"):
        open_bfcl_v4_public_v2_trusted_grader(
            pinned_checkout,
            campaign,
            evaluation_authority_secret=b"short",
            semantic_release_fingerprint=_SEMANTIC_RELEASE,
        )


def test_unlock_instance_revalidation_rejects_barrier_fingerprint_tamper(
    authorized_grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    unlock = authorized_grader.issue_evaluation_unlock(_barrier_evidence(authorized_grader))
    malformed = BfclV4PublicV2EvaluationUnlock.model_construct(
        **{
            **unlock.model_dump(mode="python"),
            "barrier_evidence_fingerprint": "0" * 64,
        }
    )
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        task_ref="holdout-00",
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        authorized_grader.grade(request, evaluation_unlock=malformed)


def test_single_pure_at_b_sample_cannot_use_ordinary_grader(
    authorized_grader: BfclV4PublicV2TrustedGrader,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> None:
    unlock = authorized_grader.issue_evaluation_unlock(_barrier_evidence(authorized_grader))
    request = _grade_request(
        campaign,
        loaded,
        kind=BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
        task_ref="holdout-00",
    )

    with pytest.raises(ValueError, match="node kind differs"):
        authorized_grader.grade(request, evaluation_unlock=unlock)
