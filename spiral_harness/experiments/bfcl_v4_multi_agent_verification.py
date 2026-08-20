"""Independent artifact-closure verification for the BFCL multi-agent comparison."""

from __future__ import annotations

from spiral_harness.agents.contracts import MULTI_AGENT_RUN_MEDIA_TYPE, MultiAgentRun
from spiral_harness.agents.receipt_backend import render_agent_turn_user_prompt
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_development_grader import (
    BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE,
    BfclV4PublicV2DevelopmentGradeReceipt,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import (
    MODEL_EXECUTION_MEDIA_TYPE,
    ExecutionStatus,
    ModelExecution,
)
from spiral_harness.experiments.bfcl_v4_multi_agent_comparison import (
    _BASELINE_SYSTEM_PROMPT,
    BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE,
    BFCL_MULTI_AGENT_TASK_REFS,
    BFCL_MULTI_AGENT_TURN_IDS,
    BfclMultiAgentComparisonResult,
    _team,
    _workflow,
    canonicalize_bfcl_candidate_text,
    paired_comparison_seed,
)
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    BfclV4PublicV2CanonicalResponseError,
    parse_bfcl_v4_public_v2_canonical_response,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def verify_bfcl_v4_multi_agent_comparison(
    *,
    store: ArtifactStore,
    result_ref: ArtifactRef,
) -> BfclMultiAgentComparisonResult:
    """Reload and close every execution, transcript, grade, and aggregate count."""

    checked_ref = ArtifactRef.model_validate(result_ref, strict=True)
    if checked_ref.media_type != BFCL_MULTI_AGENT_COMPARISON_MEDIA_TYPE:
        raise ValueError("comparison result ref declares the wrong media type")
    result = store.get_json(checked_ref, BfclMultiAgentComparisonResult)
    if result.fingerprint != checked_ref.sha256:
        raise ValueError("comparison result fingerprint differs from its artifact ref")
    if (
        result.task_refs != BFCL_MULTI_AGENT_TASK_REFS
        or tuple(outcome.task_ref for outcome in result.outcomes) != BFCL_MULTI_AGENT_TASK_REFS
    ):
        raise ValueError("comparison result differs from the frozen task roster")
    if result.task_roster_fingerprint != canonical_sha256(
        tuple(outcome.task_id for outcome in result.outcomes)
    ):
        raise ValueError("comparison task roster fingerprint changed")
    team = _team(result.model_spec)
    workflow = _workflow(team)
    if (
        result.team_fingerprint != team.fingerprint
        or result.workflow_fingerprint != workflow.fingerprint
    ):
        raise ValueError("comparison team or workflow differs from the frozen protocol")

    executions: list[ModelExecution] = []
    for outcome in result.outcomes:
        expected_seeds = tuple(
            paired_comparison_seed(
                root_seed=result.root_seed,
                task_id=outcome.task_id,
                turn_id=turn_id,
            )
            for turn_id in BFCL_MULTI_AGENT_TURN_IDS
        )
        if outcome.paired_seeds != expected_seeds:
            raise ValueError("comparison paired seeds changed")
        baseline_executions = tuple(
            _load_execution(
                store=store,
                ref=ref,
                result=result,
                task_id=outcome.task_id,
                seed=seed,
            )
            for ref, seed in zip(outcome.baseline_execution_refs, expected_seeds, strict=True)
        )
        harness_executions = tuple(
            _load_execution(
                store=store,
                ref=ref,
                result=result,
                task_id=f"{outcome.task_id}/{turn_id}",
                seed=seed,
            )
            for ref, seed, turn_id in zip(
                outcome.harness_execution_refs,
                expected_seeds,
                BFCL_MULTI_AGENT_TURN_IDS,
                strict=True,
            )
        )
        executions.extend((*baseline_executions, *harness_executions))
        baseline_task_payload = _verify_baseline_projection(
            outcome=outcome,
            executions=baseline_executions,
        )
        _verify_run(
            store=store,
            result=result,
            outcome=outcome,
            executions=harness_executions,
            task_payload=baseline_task_payload,
        )
        _verify_grades(store=store, outcome=outcome)

    refs = {
        ref.sha256
        for outcome in result.outcomes
        for ref in (*outcome.baseline_execution_refs, *outcome.harness_execution_refs)
    }
    if len(refs) != 54:
        raise ValueError("comparison execution references are not unique")
    if len(executions) != 54 or sum(execution.tokens for execution in executions) != (
        result.total_reported_tokens
    ):
        raise ValueError("comparison execution accounting does not close")
    if (
        sum(outcome.baseline_single_correct for outcome in result.outcomes)
        != result.baseline_single_correct
        or sum(outcome.baseline_budget_matched_correct for outcome in result.outcomes)
        != result.baseline_budget_matched_correct
        or sum(outcome.harness_correct for outcome in result.outcomes) != result.harness_correct
    ):
        raise ValueError("comparison aggregate counts do not match task outcomes")
    return result


def _load_execution(
    *,
    store: ArtifactStore,
    ref: ArtifactRef,
    result: BfclMultiAgentComparisonResult,
    task_id: str,
    seed: int,
) -> ModelExecution:
    if ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
        raise ValueError("execution ref declares the wrong media type")
    execution = store.get_json(ref, ModelExecution)
    if (
        execution.status is not ExecutionStatus.COMPLETED
        or execution.task_id != task_id
        or execution.seed != seed
        or execution.spec != result.model_spec
    ):
        raise ValueError("execution differs from the frozen comparison")
    return execution


def _canonical_execution_output(execution: ModelExecution) -> str | None:
    if execution.output is None:
        return None
    try:
        return canonicalize_bfcl_candidate_text(execution.output)
    except BfclV4PublicV2CanonicalResponseError:
        return None


def _verify_baseline_projection(*, outcome, executions: tuple[ModelExecution, ...]) -> str:
    task_payload = executions[0].request.user_prompt
    if any(
        execution.request.user_prompt != task_payload
        or execution.request.system_prompt != _BASELINE_SYSTEM_PROMPT
        for execution in executions
    ):
        raise ValueError("baseline executions differ from the frozen task or prompt")
    responses = tuple(_canonical_execution_output(execution) for execution in executions)
    if responses != outcome.baseline_canonical_responses:
        raise ValueError("baseline canonical responses differ from their executions")
    aggregate = aggregate_bfcl_v4_public_development_v2_pure_at_b(responses)
    if outcome.baseline_budget_matched_response != aggregate.selected_canonical_response:
        raise ValueError("baseline modal response differs from the frozen aggregation")
    return task_payload


def _verify_run(
    *,
    store: ArtifactStore,
    result,
    outcome,
    executions: tuple[ModelExecution, ...],
    task_payload: str,
) -> None:
    if outcome.harness_run_ref.media_type != MULTI_AGENT_RUN_MEDIA_TYPE:
        raise ValueError("harness run ref declares the wrong media type")
    run = store.get_json(outcome.harness_run_ref, MultiAgentRun)
    if (
        run.team_fingerprint != result.team_fingerprint
        or run.workflow_fingerprint != result.workflow_fingerprint
        or tuple(record.request.task_id for record in run.records) != (outcome.task_id,) * 3
        or tuple(record.request.turn_id for record in run.records) != BFCL_MULTI_AGENT_TURN_IDS
        or len(run.records) != len(executions)
    ):
        raise ValueError("harness transcript differs from the frozen comparison")
    team = _team(result.model_spec)
    workflow = _workflow(team)
    agents = {agent.agent_id: agent for agent in team.agents}
    for record, execution, expected_turn in zip(
        run.records,
        executions,
        workflow.turns,
        strict=True,
    ):
        expected_agent = agents[expected_turn.agent_id]
        if (
            record.request.task_payload != task_payload
            or record.request.turn_id != expected_turn.turn_id
            or record.request.agent_id != expected_turn.agent_id
            or record.request.objective != expected_turn.objective
            or tuple(dependency.turn_id for dependency in record.request.dependency_outputs)
            != expected_turn.depends_on
            or record.request.role != expected_agent.role
            or record.request.instruction != expected_agent.instruction
            or execution.request.user_prompt != render_agent_turn_user_prompt(record.request)
            or execution.request.system_prompt != record.request.instruction
            or execution.output != record.response.content
            or execution.usage.input_tokens != record.response.input_tokens
            or execution.usage.output_tokens != record.response.output_tokens
            or record.response.model_fingerprint != result.model_spec.model_fingerprint
            or record.response.runtime_fingerprint != result.model_spec.runtime_fingerprint
        ):
            raise ValueError("harness transcript is not bound to its model executions")
    if _canonical_execution_output(executions[-1]) != outcome.harness_canonical_response:
        raise ValueError("harness canonical response differs from its final execution")


def _verify_grades(*, store: ArtifactStore, outcome) -> None:
    coordinates = (
        (
            outcome.baseline_single_grade_ref,
            outcome.baseline_canonical_responses[0],
            outcome.baseline_single_correct,
        ),
        (
            outcome.baseline_budget_matched_grade_ref,
            outcome.baseline_budget_matched_response,
            outcome.baseline_budget_matched_correct,
        ),
        (outcome.harness_grade_ref, outcome.harness_canonical_response, outcome.harness_correct),
    )
    for grade_ref, response, correct in coordinates:
        if response is None:
            if grade_ref is not None or correct:
                raise ValueError("missing response has an invalid grade projection")
            continue
        if (
            grade_ref is None
            or grade_ref.media_type != BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE
        ):
            raise ValueError("gradable response lacks one exact grade receipt")
        grade = store.get_json(grade_ref, BfclV4PublicV2DevelopmentGradeReceipt)
        parsed = parse_bfcl_v4_public_v2_canonical_response(response)
        if (
            grade.task_ref != outcome.task_ref
            or grade.task_id != outcome.task_id
            or grade.split != outcome.split
            or grade.canonical_response_sha256 != canonical_sha256(parsed)
            or grade.correct is not correct
        ):
            raise ValueError("grade receipt differs from the result projection")


__all__ = ["verify_bfcl_v4_multi_agent_comparison"]
