from __future__ import annotations

import hashlib
import inspect
import json

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark._harness_fault_cases import (
    RepairRuleId,
    branch_for_rule,
    evaluate_branch,
    parse_public_task_input,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault_authority import HarnessFaultScenarioAuthority
from spiral_harness.benchmark.harness_fault_compiler import HarnessRole
from spiral_harness.execution.contracts import (
    AttemptOutcome,
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
    ModelExecution,
)
from spiral_harness.experiments.confirmatory_arms import FaultFactorialCell
from spiral_harness.experiments.harness_fault_development_contracts import (
    DevelopmentInvocationMode,
    HarnessFaultDevelopmentError,
    PromotionDecision,
)
from spiral_harness.experiments.harness_fault_development_support import PROPOSER_TASK_ID
from spiral_harness.experiments.harness_fault_four_context_contracts import (
    FAULT_CONTEXT_ORDER,
    HarnessFaultFourContextPlan,
    JointFaultContextCandidateFreeze,
    make_harness_fault_four_context_plan,
)
from spiral_harness.experiments.harness_fault_four_context_runner import (
    _require_fit_gate_disjoint,
    run_model_driven_harness_fault_four_context_development,
)
from spiral_harness.experiments.harness_fault_four_context_verification import (
    verify_model_driven_harness_fault_four_context_development,
)
from spiral_harness.storage.artifact_store import ArtifactStore

_FINGERPRINT = "four-context-model-output-fixture"


def _salt(context_index: int, partition: str) -> bytes:
    return hashlib.sha256(f"four-context/{context_index}/{partition}".encode()).digest()


def _authority(store: ArtifactStore, context_index: int) -> HarnessFaultScenarioAuthority:
    return HarnessFaultScenarioAuthority(
        store,
        exploration_salt=_salt(context_index, "exploration"),
        gate_salt=_salt(context_index, "gate"),
        sealed_salt=_salt(context_index, "sealed"),
    )


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="four-context-fixture",
        backend_fingerprint=_FINGERPRINT,
        model="fixture/same-model-four-isolated-clients",
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


def _plan(store: ArtifactStore) -> HarnessFaultFourContextPlan:
    authority = _authority(store, 0)
    return make_harness_fault_four_context_plan(
        fit_partition_grant=authority.issue_exploration_grant(),
        gate_partition_grant=authority.issue_gate_grant(),
        invocation_mode=DevelopmentInvocationMode.FIXTURE_REPLAY,
        model_spec=_spec(),
        fit_task_count=28,
        gate_task_count=28,
        proposal_seed=17,
        solver_master_seed=20270814,
        max_tokens_per_call=64,
    )


class _IsolatedOutputFixture:
    def __init__(self, cell: FaultFactorialCell) -> None:
        self.cell = cell
        self.outputs: dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        return _FINGERPRINT

    def invoke(self, *, spec, request) -> BackendResponse:
        assert spec == _spec()
        if request.task_id == PROPOSER_TASK_ID:
            output = json.dumps(
                {"rule_id": RepairRuleId.ROUTED_POLICY.value},
                separators=(",", ":"),
            )
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


def test_plan_pairs_one_fresh_latent_block_across_four_isolated_namespaces(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    plan = _plan(store)
    assert tuple(block.cell for block in plan.blocks) == FAULT_CONTEXT_ORDER
    assert plan.candidate_definition_call_count == 116
    assert plan.gate_attribution_call_count == 448
    assert plan.max_model_calls == 564
    assert plan.max_total_tokens == 564 * 64
    assert plan.attempt_budget.max_attempts == 564
    assert (
        len({block.fit_partition_grant.public_commitment.authority_id for block in plan.blocks})
        == 1
    )
    assert len({block.fit_latent_block_id for block in plan.blocks}) == 1
    assert len({block.gate_latent_block_id for block in plan.blocks}) == 1
    assert len({block.block_id for block in plan.blocks}) == 4

    payload = plan.model_dump(mode="python", round_trip=True)
    foreign = _authority(store, 1)
    payload["blocks"][1]["fit_partition_grant"] = foreign.issue_exploration_grant().model_dump(
        mode="python", round_trip=True
    )
    payload["blocks"][1]["gate_partition_grant"] = foreign.issue_gate_grant().model_dump(
        mode="python", round_trip=True
    )
    with pytest.raises(ValidationError, match="differ outside"):
        HarnessFaultFourContextPlan.model_validate(payload, strict=True)

    reversed_payload = plan.model_dump(mode="python", round_trip=True)
    reversed_payload["blocks"] = tuple(reversed(reversed_payload["blocks"]))
    reordered = HarnessFaultFourContextPlan.model_validate(reversed_payload, strict=True)
    assert tuple(block.cell for block in reordered.blocks) == FAULT_CONTEXT_ORDER

    authority = _authority(store, 2)
    same_grant = authority.issue_exploration_grant()
    with pytest.raises(ValidationError, match="gate partition"):
        make_harness_fault_four_context_plan(
            fit_partition_grant=same_grant,
            gate_partition_grant=same_grant,
            invocation_mode=DevelopmentInvocationMode.FIXTURE_REPLAY,
            model_spec=_spec(),
            fit_task_count=28,
            gate_task_count=28,
            proposal_seed=17,
            solver_master_seed=20270814,
            max_tokens_per_call=64,
        )

    fit_opened = verify_partition_opening(
        store,
        plan.blocks[0].fit_partition_grant,
    )
    with pytest.raises(HarnessFaultDevelopmentError, match="overlap"):
        _require_fit_gate_disjoint(fit_opened, fit_opened)


def test_one_invocation_freezes_four_candidates_then_replays_the_exact_global_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ArtifactStore, "_fsync_directory", staticmethod(lambda _path: True))
    store = ArtifactStore(tmp_path / "cas")
    plan = _plan(store)

    shared = _IsolatedOutputFixture(FaultFactorialCell.SS)
    with pytest.raises(HarnessFaultDevelopmentError, match="distinct backend client"):
        run_model_driven_harness_fault_four_context_development(
            store=store,
            context_backends={cell: shared for cell in FAULT_CONTEXT_ORDER},
            plan=plan,
        )
    assert not shared.outputs

    backends = {cell: _IsolatedOutputFixture(cell) for cell in FAULT_CONTEXT_ORDER}
    record = run_model_driven_harness_fault_four_context_development(
        store=store,
        context_backends=backends,
        plan=plan,
    )
    result = record.result

    assert result.model_call_count == plan.max_model_calls == 564
    assert sum(len(backend.outputs) for backend in backends.values()) == 564
    assert result.candidate_freeze.candidate_definition_call_count == 116
    assert result.candidate_freeze.last_candidate_definition_sequence == 115
    assert result.candidate_freeze.first_attribution_sequence == 116
    early_attribution = result.candidate_freeze.model_dump(mode="python", round_trip=True)
    early_attribution["first_attribution_sequence"] = 115
    with pytest.raises(ValidationError, match="attribution must start"):
        JointFaultContextCandidateFreeze.model_validate(early_attribution, strict=True)
    assert result.gate_batch.model_call_count_before_gate_batch == 564
    assert result.one_trusted_runner_invocation_completed
    assert not result.cross_process_atomicity_attested
    assert result.exact_nominal_model_spec_shared
    assert result.distinct_backend_client_instances_runtime_checked
    assert not result.provider_transport_or_session_isolation_attested
    assert result.raw_outputs_and_ledger_offline_replayable
    assert result.candidate_definition_before_gate_attribution_calls_offline_replayable
    assert result.trusted_runner_opened_gate_after_freeze
    assert not result.freeze_persistence_timing_independently_attested
    assert not result.gate_opening_timing_independently_attested
    assert not result.gate_block_one_time_consumption_attested
    assert not result.cross_context_feedback_disclosed
    assert result.permanently_nonreportable_development_artifact
    assert not result.reportable_benchmark_result
    assert not result.confirmatory_inference
    assert not result.sealed_evidence
    assert not result.provider_identity_attested
    assert not result.exact_served_revision_claim_allowed
    assert all(context.gate.decision is PromotionDecision.PROMOTE for context in result.contexts)

    fit_rosters = []
    gate_rosters = []
    for block in result.plan.blocks:
        fit_opened = verify_partition_opening(store, block.fit_partition_grant)
        gate_opened = verify_partition_opening(store, block.gate_partition_grant)
        fit_roster = {scenario.task.task_id for scenario in fit_opened.scenarios}
        gate_roster = {scenario.task.task_id for scenario in gate_opened.scenarios}
        assert len(fit_roster) == block.fit_task_count
        assert len(gate_roster) == block.gate_task_count
        assert fit_roster.isdisjoint(gate_roster)
        assert all(fit_roster == previous for previous in fit_rosters)
        assert all(gate_roster == previous for previous in gate_rosters)
        fit_rosters.append(fit_roster)
        gate_rosters.append(gate_roster)

    proposal_sequences = []
    attribution_sequences = []
    for context in result.contexts:
        proposal_outcome = store.get_json(context.proposal.outcome_ref, AttemptOutcome)
        proposal_sequences.append(proposal_outcome.sequence)
        for observation in context.gate_observations:
            attribution_sequences.append(
                store.get_json(observation.outcome_ref, AttemptOutcome).sequence
            )
        candidate = next(
            observation
            for observation in context.gate_observations
            if observation.harness_role is HarnessRole.CANDIDATE
        )
        execution = store.get_json(candidate.execution_ref, ModelExecution)
        assert execution.output_text == backends[context.cell].outputs[execution.request_sha256]
    assert proposal_sequences == [28, 57, 86, 115]
    assert min(attribution_sequences) == 116
    assert max(attribution_sequences) == 563
    assert len(set(attribution_sequences)) == 4 * 4 * 28

    verified = verify_model_driven_harness_fault_four_context_development(
        store,
        record.result_ref,
    )
    assert verified == result

    parameters = inspect.signature(
        run_model_driven_harness_fault_four_context_development
    ).parameters
    assert "candidate" not in parameters
    assert "action" not in parameters
    assert "ranker" not in parameters
