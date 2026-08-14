from __future__ import annotations

import inspect
import json

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark._harness_fault_cases import (
    RepairRuleId,
    branch_for_rule,
    evaluate_branch,
    parse_public_task_input,
)
from spiral_harness.benchmark.harness_fault_authority import HarnessFaultScenarioAuthority
from spiral_harness.benchmark.harness_fault_compiler import HarnessRole
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.execution.contracts import (
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
    ModelExecution,
)
from spiral_harness.experiments.confirmatory_arms import (
    FaultFactorialCell,
    OptimizerFeedbackMode,
    PromotionRule,
)
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE,
    DevelopmentInvocationMode,
    FeedbackEvidenceLinkage,
    HarnessFaultDevelopmentError,
    HarnessFaultDevelopmentPlan,
    HarnessFaultModelDrivenDevelopmentResult,
    PromotionDecision,
    ProposerVisibleFeedbackMode,
)
from spiral_harness.experiments.harness_fault_development_runner import (
    run_model_driven_harness_fault_development,
)
from spiral_harness.experiments.harness_fault_development_support import (
    PROPOSER_TASK_ID,
    build_feedback_projection,
    proposer_task,
    proposer_visible_feedback,
)
from spiral_harness.experiments.harness_fault_development_verification import (
    verify_model_driven_harness_fault_development,
)
from spiral_harness.storage.artifact_store import ArtifactStore

_FINGERPRINT = "model-output-driven-harness-fault-fixture"


def _authority(store: ArtifactStore) -> HarnessFaultScenarioAuthority:
    return HarnessFaultScenarioAuthority(
        store,
        exploration_salt=b"model-driven-exploration-salt-00001",
        gate_salt=b"model-driven-gate-salt-00000000001",
        sealed_salt=b"model-driven-sealed-salt-0000001",
    )


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="model-output-driven-fixture",
        backend_fingerprint=_FINGERPRINT,
        model="fixture/same-solver-and-proposer",
        revision="snapshot-2026-08-14",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-14",
        runtime="python-3.12/direct-model-output",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
            timeout_seconds=5.0,
        ),
    )


def _plan(
    cell: FaultFactorialCell = FaultFactorialCell.MM,
    linkage: FeedbackEvidenceLinkage = FeedbackEvidenceLinkage.SOURCE_LINKED,
) -> HarnessFaultDevelopmentPlan:
    task_count = 28
    max_calls = 4 * task_count + 1
    return HarnessFaultDevelopmentPlan(
        cell=cell,
        feedback_linkage=linkage,
        invocation_mode=DevelopmentInvocationMode.FIXTURE_REPLAY,
        model_spec=_spec(),
        task_count=task_count,
        proposal_seed=17,
        solver_master_seed=20270814,
        max_tokens_per_call=64,
        max_model_calls=max_calls,
        max_total_tokens=max_calls * 64,
    )


class _OutputDrivenFixture:
    def __init__(self, proposal: str) -> None:
        self._proposal = proposal
        self.outputs: dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        return _FINGERPRINT

    def invoke(self, *, spec, request) -> BackendResponse:
        assert spec == _spec()
        if request.task_id == PROPOSER_TASK_ID:
            output = self._proposal
        else:
            rule_line = next(
                line for line in request.system_prompt.splitlines() if line.startswith("RULE=")
            )
            rule = RepairRuleId(rule_line.removeprefix("RULE="))
            task = parse_public_task_input(request.user_prompt)
            branch = branch_for_rule(rule, task.family, task.context)
            answer, observable = evaluate_branch(
                task.family,
                branch,
                primary=task.primary,
                secondary=task.secondary,
            )
            output = json.dumps(
                {"answer": answer, "observable": observable},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        self.outputs[request.fingerprint] = output
        return BackendResponse(
            output=output,
            usage=BackendTokenUsage(input_tokens=3, output_tokens=4),
            cost_usd=0.0,
        )


def test_factorial_profiles_and_shuffled_placebo_remain_expressible() -> None:
    expected = {
        FaultFactorialCell.SS: (
            FeedbackEvidenceLinkage.SCORE_ONLY,
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.MS: (
            FeedbackEvidenceLinkage.SOURCE_LINKED,
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.PERFORMANCE_ONLY,
        ),
        FaultFactorialCell.SM: (
            FeedbackEvidenceLinkage.SCORE_ONLY,
            OptimizerFeedbackMode.SCORE_ONLY,
            PromotionRule.MECHANISM_GATED,
        ),
        FaultFactorialCell.MM: (
            FeedbackEvidenceLinkage.SOURCE_LINKED,
            OptimizerFeedbackMode.MECHANISM_VISIBLE,
            PromotionRule.MECHANISM_GATED,
        ),
    }
    for cell, (linkage, feedback, gate) in expected.items():
        plan = _plan(cell, linkage)
        assert plan.optimizer_feedback is feedback
        assert plan.promotion_rule is gate
        assert plan.max_model_calls == 113
    shuffled = _plan(FaultFactorialCell.MM, FeedbackEvidenceLinkage.SHUFFLED_PLACEBO)
    assert shuffled.feedback_linkage is FeedbackEvidenceLinkage.SHUFFLED_PLACEBO
    with pytest.raises(ValidationError, match="SS/SM require"):
        _plan(FaultFactorialCell.SS, FeedbackEvidenceLinkage.SHUFFLED_PLACEBO)
    invalid_budget = _plan().model_dump(mode="python", round_trip=True)
    invalid_budget["max_model_calls"] = 112
    with pytest.raises(ValidationError, match=r"4N\+1"):
        HarnessFaultDevelopmentPlan.model_validate(invalid_budget, strict=True)


def test_raw_model_outputs_choose_candidate_and_drive_behavior_without_middleware(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    proposal = json.dumps({"rule_id": RepairRuleId.ROUTED_POLICY.value}, separators=(",", ":"))
    backend = _OutputDrivenFixture(proposal)
    record = run_model_driven_harness_fault_development(
        store=store,
        backend=backend,
        partition_grant=authority.issue_exploration_grant(),
        plan=_plan(),
    )
    result = record.result

    assert len(backend.outputs) == result.model_call_count == 113
    assert result.proposal.parsed_action is not None
    assert result.proposal.parsed_action.rule_id is RepairRuleId.ROUTED_POLICY
    assert result.proposal.applied_rule_id is RepairRuleId.ROUTED_POLICY
    assert result.gate.decision is PromotionDecision.PROMOTE
    assert result.gate.candidate_correct > result.gate.parent_correct
    assert result.gate.candidate_correct > result.gate.revert_correct
    assert result.gate.candidate_correct > result.gate.placebo_correct
    assert result.model_output_drives_candidate
    assert result.raw_model_output_drives_scored_behavior
    assert not result.deterministic_middleware_behavior_rewrite_used
    assert result.permanently_nonreportable_development_artifact
    assert not result.reportable_benchmark_result
    assert not result.confirmatory_inference
    assert not result.sealed_evidence
    assert not result.provider_identity_attested

    candidate = next(
        item for item in result.observations if item.harness_role is HarnessRole.CANDIDATE
    )
    execution = store.get_json(candidate.execution_ref, ModelExecution)
    assert execution.output_text == backend.outputs[execution.request_sha256]
    assert candidate.raw_model_output_sha256 == sha256_bytes(execution.output_text.encode("utf-8"))
    assert verify_model_driven_harness_fault_development(store, record.result_ref) == result

    parent = tuple(
        item for item in result.observations if item.harness_role is HarnessRole.FAULTY_PARENT
    )
    placebo_projection = build_feedback_projection(
        _plan(FaultFactorialCell.MM, FeedbackEvidenceLinkage.SHUFFLED_PLACEBO),
        parent,
    )
    assert placebo_projection.authority_labeled_placebo
    visible = proposer_visible_feedback(placebo_projection)
    assert visible.feedback_mode is ProposerVisibleFeedbackMode.MECHANISM_VISIBLE
    assert "plan_fingerprint" not in type(visible).model_fields
    proposer_question = proposer_task(placebo_projection).question
    assert placebo_projection.plan_fingerprint not in proposer_question
    for authority_only_label in ("shuffled", "placebo", "source_task_id", '"linkage"'):
        assert authority_only_label not in proposer_question

    payload = result.model_dump(mode="python", round_trip=True)
    payload["observations"][0]["behavior_correct"] = not payload["observations"][0][
        "behavior_correct"
    ]
    forged = HarnessFaultModelDrivenDevelopmentResult.model_validate(payload, strict=True)
    forged_ref = store.put_json(forged, media_type=HARNESS_FAULT_DEVELOPMENT_RESULT_MEDIA_TYPE)
    with pytest.raises(HarnessFaultDevelopmentError, match="raw model output"):
        verify_model_driven_harness_fault_development(store, forged_ref)

    parameters = inspect.signature(run_model_driven_harness_fault_development).parameters
    assert "candidate" not in parameters
    assert "action" not in parameters
    assert "ranker" not in parameters


def test_invalid_proposal_fails_closed_without_retry_or_human_selection(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    backend = _OutputDrivenFixture("not-json")
    record = run_model_driven_harness_fault_development(
        store=store,
        backend=backend,
        partition_grant=authority.issue_exploration_grant(),
        plan=_plan(FaultFactorialCell.SS, FeedbackEvidenceLinkage.SCORE_ONLY),
    )
    result = record.result
    assert not result.proposal.proposal_valid
    assert result.proposal.applied_rule_id is RepairRuleId.CONSTANT_LEGACY
    assert result.proposal.retries_used == 0
    assert not result.proposal.human_selection_used
    assert result.gate.decision is PromotionDecision.ROLLBACK
    assert len(backend.outputs) == 113
    assert verify_model_driven_harness_fault_development(store, record.result_ref) == result


def test_runner_rejects_nonexploration_grant_before_any_model_call(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    authority = _authority(store)
    backend = _OutputDrivenFixture(
        json.dumps({"rule_id": RepairRuleId.ROUTED_POLICY.value}, separators=(",", ":"))
    )
    with pytest.raises(HarnessFaultDevelopmentError, match="exploration"):
        run_model_driven_harness_fault_development(
            store=store,
            backend=backend,
            partition_grant=authority.issue_gate_grant(),
            plan=_plan(),
        )
    assert not backend.outputs
