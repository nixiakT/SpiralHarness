from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_grader import (
    make_bfcl_v4_public_prediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_projections import (
    FIT_TASK_IDS,
    BfclV4FullFitFeedback,
    BfclV4ScoreFitAggregate,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    load_bfcl_v4_public_pilot,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.execution.contracts import BackendTokenUsage
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4CandidateParseFailure,
    BfclV4CandidateResolution,
    BfclV4CandidateResolutionFailure,
    BfclV4DiagnosisFailure,
    BfclV4FullFitDiagnosisBatch,
    BfclV4FullFitDiagnosisObservation,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_CANDIDATE_SUBMIT_TOOL,
    BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
    BFCL_V4_DIAGNOSIS_SUBMIT_TOOL,
    BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
    BFCL_V4_SEED_SYSTEM_PROMPT,
    BFCL_V4_SOLVER_CORE_PROMPT,
    BFCL_V4_STATIC_STRATEGY,
    build_bfcl_v4_full_diagnosis_prompt,
    build_bfcl_v4_proposal_prompt,
    build_bfcl_v4_score_diagnosis_prompt,
    parse_bfcl_v4_candidate,
    parse_bfcl_v4_diagnosis,
    resolve_bfcl_v4_candidate,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallResponse,
)
from spiral_harness.skills.package import SKILL_CONTEXT_START_DELIMITER

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


def _score_prompt(parent: str = BFCL_V4_SEED_SYSTEM_PROMPT):
    aggregate = BfclV4ScoreFitAggregate(
        plan_fingerprint="1" * 64,
        batch_reference_sha256="2" * 64,
        aggregate_accuracy_basis_points=4_000,
    )
    return build_bfcl_v4_score_diagnosis_prompt(
        parent_system_prompt=parent,
        aggregate=aggregate,
    )


def _call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str = "call-1",
) -> NativeAssistantToolCall:
    return NativeAssistantToolCall(
        call_id=call_id,
        official_name=tool_name,
        wire_name=tool_name,
        arguments_json=canonical_json(arguments),
    )


def _response(
    *calls: NativeAssistantToolCall,
    assistant_text: str | None = None,
) -> NativeFunctionCallResponse:
    return NativeFunctionCallResponse(
        request_fingerprint="a" * 64,
        serializer_fingerprint="b" * 64,
        parser_fingerprint="c" * 64,
        transport_fingerprint="d" * 64,
        tools_fingerprint="e" * 64,
        tool_calls=tuple(calls),
        assistant_text=assistant_text,
        finish_reason="tool_calls" if calls else "stop",
        usage=BackendTokenUsage(input_tokens=10, output_tokens=5),
    )


def _diagnosis_response(text: str) -> NativeFunctionCallResponse:
    return _response(
        _call(
            tool_name=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
            arguments={"diagnosis": text},
        )
    )


def _candidate_response(text: str) -> NativeFunctionCallResponse:
    return _response(
        _call(
            tool_name=BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
            arguments={"strategy_appendix": text},
        )
    )


def _valid_proposal(parent: str = BFCL_V4_SEED_SYSTEM_PROMPT):
    diagnosis_prompt = _score_prompt(parent)
    diagnosis = parse_bfcl_v4_diagnosis(
        diagnosis_prompt,
        _diagnosis_response(
            "The parent can conflate independent requests; preserve correct single calls and "
            "validate call multiplicity before submission."
        ),
    )
    proposal = build_bfcl_v4_proposal_prompt(diagnosis_prompt, diagnosis)
    return diagnosis_prompt, diagnosis, proposal


def test_meta_submit_tools_are_pinned_single_field_native_schemas() -> None:
    diagnosis = json.loads(BFCL_V4_DIAGNOSIS_SUBMIT_TOOL.function_schema_json)
    candidate = json.loads(BFCL_V4_CANDIDATE_SUBMIT_TOOL.function_schema_json)

    assert BFCL_V4_DIAGNOSIS_SUBMIT_TOOL.official_name == (BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME)
    assert BFCL_V4_DIAGNOSIS_SUBMIT_TOOL.wire_name == BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME
    assert BFCL_V4_CANDIDATE_SUBMIT_TOOL.official_name == BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME
    assert BFCL_V4_CANDIDATE_SUBMIT_TOOL.wire_name == BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME
    assert diagnosis["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": (
                    "Concise evidence, root cause, preserved behavior, and corrective principle."
                ),
                "minLength": 1,
                "maxLength": 12_000,
            }
        },
        "required": ["diagnosis"],
    }
    assert candidate["parameters"]["additionalProperties"] is False
    assert set(candidate["parameters"]["properties"]) == {"strategy_appendix"}
    assert candidate["parameters"]["required"] == ["strategy_appendix"]


def test_score_diagnosis_contains_only_aggregate_feedback_and_native_contract() -> None:
    prompt = _score_prompt()

    assert prompt.feedback_view == "score-only"
    assert prompt.submit_tool == BFCL_V4_DIAGNOSIS_SUBMIT_TOOL
    assert prompt.submit_tool_fingerprint == canonical_sha256(prompt.submit_tool)
    assert prompt.output_grammar == "one-submit-bfcl-diagnosis-native-call-v1"
    assert "aggregate_accuracy_basis_points" in prompt.user_prompt
    assert "4000" in prompt.user_prompt
    assert all(task_id not in prompt.user_prompt for task_id in FIT_TASK_IDS)
    assert "possible_answer" not in prompt.user_prompt
    assert "ground_truth" not in prompt.user_prompt


def test_diagnosis_accepts_exactly_one_expected_native_submission() -> None:
    prompt = _score_prompt()
    response = _diagnosis_response("Evidence; root cause; invariant; corrective principle.")

    result = parse_bfcl_v4_diagnosis(prompt, response)

    assert result.valid is True
    assert result.failure is BfclV4DiagnosisFailure.NONE
    assert result.diagnosis_text == "Evidence; root cause; invariant; corrective principle."
    assert result.native_response_fingerprint == response.fingerprint
    assert result.automatic_retry_used is False
    assert result.output_repair_used is False


def test_diagnosis_native_extractor_rejects_text_wrong_tool_multi_call_and_fields() -> None:
    prompt = _score_prompt()
    wrong = _call(tool_name="wrong_tool", arguments={"diagnosis": "x"})
    right = _call(
        tool_name=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
        arguments={"diagnosis": "x"},
    )
    cases = (
        (None, BfclV4DiagnosisFailure.NO_VERIFIED_RESPONSE),
        ({}, BfclV4DiagnosisFailure.INVALID_RESPONSE_CONTRACT),
        (_response(assistant_text="plain diagnosis"), BfclV4DiagnosisFailure.TEXT_ONLY),
        (_response(wrong), BfclV4DiagnosisFailure.WRONG_TOOL),
        (
            _response(right, right.model_copy(update={"call_id": "call-2"})),
            BfclV4DiagnosisFailure.WRONG_CALL_COUNT,
        ),
        (
            _response(right, assistant_text="also text"),
            BfclV4DiagnosisFailure.ASSISTANT_TEXT_PRESENT,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
                    arguments={"diagnosis": "x", "extra": True},
                )
            ),
            BfclV4DiagnosisFailure.EXTRA_ARGUMENT_FIELDS,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
                    arguments={"other": "x"},
                )
            ),
            BfclV4DiagnosisFailure.MISSING_ARGUMENT,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL_NAME,
                    arguments={"diagnosis": 7},
                )
            ),
            BfclV4DiagnosisFailure.ARGUMENT_NOT_TEXT,
        ),
    )

    for response, expected in cases:
        result = parse_bfcl_v4_diagnosis(prompt, response)
        assert result.valid is False
        assert result.failure is expected
        assert result.diagnosis_text is None


def test_diagnosis_content_limits_are_fail_closed_without_repair() -> None:
    prompt = _score_prompt()
    cases = (
        (" \n\t ", BfclV4DiagnosisFailure.EMPTY_DIAGNOSIS),
        ("x" * 12_001, BfclV4DiagnosisFailure.DIAGNOSIS_TOO_LARGE),
        (
            "safe" + SKILL_CONTEXT_START_DELIMITER,
            BfclV4DiagnosisFailure.FORBIDDEN_DELIMITER,
        ),
        ("unsafe\x00", BfclV4DiagnosisFailure.INVALID_CONTROL_CHARACTER),
    )

    for content, expected in cases:
        result = parse_bfcl_v4_diagnosis(prompt, _diagnosis_response(content))
        assert result.valid is False
        assert result.failure is expected


def test_candidate_parser_composes_strategy_under_immutable_native_core() -> None:
    _, diagnosis, proposal = _valid_proposal()
    strategy = (
        "First decompose every requested operation. Then choose the semantically exact function, "
        "build schema-valid arguments, and audit multiplicity before emitting native calls."
    )
    response = _candidate_response(strategy)

    candidate = parse_bfcl_v4_candidate(proposal, response)
    resolution = resolve_bfcl_v4_candidate(
        diagnosis_result=diagnosis,
        proposal_prompt=proposal,
        candidate_parse_result=candidate,
    )

    assert proposal.submit_tool == BFCL_V4_CANDIDATE_SUBMIT_TOOL
    assert proposal.output_grammar == "one-submit-bfcl-candidate-native-call-v1"
    assert candidate.valid is True
    assert candidate.failure is BfclV4CandidateParseFailure.NONE
    assert candidate.strategy_text == strategy
    assert candidate.native_response_fingerprint == response.fingerprint
    assert candidate.candidate_system_prompt is not None
    assert candidate.candidate_system_prompt.startswith(BFCL_V4_SOLVER_CORE_PROMPT)
    assert candidate.candidate_system_prompt.endswith(strategy)
    assert resolution.candidate_admissible is True
    assert resolution.resolution_failure is BfclV4CandidateResolutionFailure.NONE
    assert resolution.executed_harness_variant == "candidate"
    assert resolution.exact_parent_fallback_used is False
    assert resolution.forced_rollback is False
    assert resolution.selection_eligibility == "gate-pending"


def test_candidate_native_extractor_and_content_parser_reject_every_ambiguous_shape() -> None:
    _, _, proposal = _valid_proposal()
    right = _call(
        tool_name=BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
        arguments={"strategy_appendix": "valid"},
    )
    cases = (
        (None, BfclV4CandidateParseFailure.NO_VERIFIED_RESPONSE),
        (_response(assistant_text="plain strategy"), BfclV4CandidateParseFailure.TEXT_ONLY),
        (
            _response(_call(tool_name="wrong_tool", arguments={"strategy_appendix": "x"})),
            BfclV4CandidateParseFailure.WRONG_TOOL,
        ),
        (
            _response(right, right.model_copy(update={"call_id": "call-2"})),
            BfclV4CandidateParseFailure.WRONG_CALL_COUNT,
        ),
        (
            _response(right, assistant_text="extra prose"),
            BfclV4CandidateParseFailure.ASSISTANT_TEXT_PRESENT,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
                    arguments={"strategy_appendix": "x", "extra": "y"},
                )
            ),
            BfclV4CandidateParseFailure.EXTRA_ARGUMENT_FIELDS,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
                    arguments={"wrong": "x"},
                )
            ),
            BfclV4CandidateParseFailure.MISSING_ARGUMENT,
        ),
        (
            _response(
                _call(
                    tool_name=BFCL_V4_CANDIDATE_SUBMIT_TOOL_NAME,
                    arguments={"strategy_appendix": ["x"]},
                )
            ),
            BfclV4CandidateParseFailure.ARGUMENT_NOT_TEXT,
        ),
        (_candidate_response(" \n "), BfclV4CandidateParseFailure.EMPTY_STRATEGY),
        (_candidate_response("x" * 8_001), BfclV4CandidateParseFailure.STRATEGY_TOO_LARGE),
        (
            _candidate_response("safe" + SKILL_CONTEXT_START_DELIMITER),
            BfclV4CandidateParseFailure.FORBIDDEN_DELIMITER,
        ),
        (_candidate_response("unsafe\x00"), BfclV4CandidateParseFailure.INVALID_CONTROL_CHARACTER),
        (_candidate_response(BFCL_V4_STATIC_STRATEGY), BfclV4CandidateParseFailure.NO_OP),
    )

    for response, expected in cases:
        result = parse_bfcl_v4_candidate(proposal, response)
        assert result.valid is False
        assert result.failure is expected
        assert result.candidate_system_prompt is None
        assert result.output_repair_used is False


def test_invalid_candidate_uses_exact_parent_for_all_slots_and_forces_rollback() -> None:
    _, diagnosis, proposal = _valid_proposal()
    invalid = parse_bfcl_v4_candidate(proposal, None)

    resolution = resolve_bfcl_v4_candidate(
        diagnosis_result=diagnosis,
        proposal_prompt=proposal,
        candidate_parse_result=invalid,
    )

    assert resolution.candidate_admissible is False
    assert resolution.resolution_failure is (
        BfclV4CandidateResolutionFailure.CANDIDATE_PARSE_INVALID
    )
    assert resolution.evaluation_system_prompt == resolution.parent_system_prompt
    assert resolution.evaluation_system_prompt_sha256 == resolution.parent_system_prompt_sha256
    assert resolution.executed_harness_variant == "parent"
    assert resolution.exact_parent_fallback_used is True
    assert resolution.forced_rollback is True
    assert resolution.selection_eligibility == "forced-rollback"
    assert resolution.all_frozen_candidate_and_gate_slots_must_execute is True
    assert resolution.invalid_candidate_slot_policy == ("parent-fallback-consumes-all-frozen-slots")

    tampered = resolution.model_dump(mode="python", round_trip=True)
    tampered["evaluation_system_prompt"] += "\nnot the parent"
    with pytest.raises(ValidationError, match=r"hash differs|exact parent"):
        BfclV4CandidateResolution.model_validate(tampered, strict=True)


def test_invalid_diagnosis_has_precedence_even_if_proposal_call_is_well_formed() -> None:
    diagnosis_prompt = _score_prompt()
    diagnosis = parse_bfcl_v4_diagnosis(diagnosis_prompt, None)
    proposal = build_bfcl_v4_proposal_prompt(diagnosis_prompt, diagnosis)
    parsed = parse_bfcl_v4_candidate(
        proposal,
        _candidate_response("Validate function semantics and every schema field."),
    )

    resolution = resolve_bfcl_v4_candidate(
        diagnosis_result=diagnosis,
        proposal_prompt=proposal,
        candidate_parse_result=parsed,
    )

    assert parsed.valid is True
    assert resolution.candidate_admissible is False
    assert resolution.resolution_failure is BfclV4CandidateResolutionFailure.DIAGNOSIS_INVALID
    assert resolution.evaluation_system_prompt == BFCL_V4_SEED_SYSTEM_PROMPT
    assert resolution.forced_rollback is True
    assert "cannot become admissible" in proposal.user_prompt


@pytest.fixture(scope="module")
def full_fit_batch() -> BfclV4FullFitDiagnosisBatch:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact FULL prompt test")
    loaded = load_bfcl_v4_public_pilot(checkout)
    by_id = {task.task_id: task for task in loaded.tasks}
    observations = []
    for task_id in FIT_TASK_IDS:
        prediction = make_bfcl_v4_public_prediction(task_id, ())
        observations.append(
            BfclV4FullFitDiagnosisObservation(
                task=by_id[task_id],
                own_prediction=prediction,
                feedback=BfclV4FullFitFeedback(
                    task_id=task_id,
                    own_prediction_reference_sha256=prediction.fingerprint,
                    accepted=False,
                    failure_class="call-count",
                ),
            )
        )
    return BfclV4FullFitDiagnosisBatch(observations=tuple(observations))


def test_full_prompt_joins_only_candidate_safe_source_bound_fit_views(
    full_fit_batch: BfclV4FullFitDiagnosisBatch,
) -> None:
    score = _score_prompt()
    full = build_bfcl_v4_full_diagnosis_prompt(
        parent_system_prompt=BFCL_V4_SEED_SYSTEM_PROMPT,
        batch=full_fit_batch,
    )

    assert full.system_prompt == score.system_prompt
    assert full.feedback_view == "candidate-safe-full"
    assert full.authorized_input_sha256 == full_fit_batch.fingerprint
    assert all(task_id in full.user_prompt for task_id in FIT_TASK_IDS)
    assert '"failure_class":"call-count"' in full.user_prompt
    assert '"own_prediction"' in full.user_prompt
    assert '"function_schemas"' in full.user_prompt
    assert "possible_answer" not in full.user_prompt
    assert "ground_truth" not in full.user_prompt
    assert "question_blob_sha256" not in full.user_prompt

    reversed_payload = full_fit_batch.model_dump(mode="python", round_trip=True)
    reversed_payload["observations"] = tuple(reversed(reversed_payload["observations"]))
    with pytest.raises(ValidationError, match="exact frozen FIT order"):
        BfclV4FullFitDiagnosisBatch.model_validate(reversed_payload, strict=True)
