from __future__ import annotations

import hmac
from hashlib import sha256
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_execution_receipts import (
    ledger_for,
    persist_preflight,
    prompt_request,
    record_attempt,
    runner_spec,
    scheduled_batch,
)

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.evolution.feedback_media_types import (
    EXPLORATION_INPUTS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
)
from spiral_harness.evolution.models import (
    PROMPT_PROPOSAL_MEDIA_TYPE,
    PromptProposal,
)
from spiral_harness.evolution.objective_evidence import (
    TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    TrustedObjectiveAggregate,
    TrustedObjectiveAggregateContent,
    TrustedObjectiveAggregateService,
    TrustedObjectiveIntervalEvidence,
)
from spiral_harness.evolution.orchestrator import (
    DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    EXPLORATION_AGGREGATES_MEDIA_TYPE,
    EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    MUTATION_HYPOTHESIS_MEDIA_TYPE,
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    SafeBenchmarkMetadata,
    SearchBenchmarkBinding,
)
from spiral_harness.evolution.score_feedback_attestation import (
    ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE,
    AttestedScoreFeedback,
    AttestedScoreFeedbackEnvelope,
    ScoreFeedbackAttestationError,
    ScoreFeedbackAttestationService,
    ScoreFeedbackProjectionRequest,
    ScoreFeedbackVerificationCapability,
    ScorePerformancePolicy,
)
from spiral_harness.evolution.score_receipt_closure import ScoreReceiptReplayCapability
from spiral_harness.execution.contracts import (
    MODEL_EXECUTION_MEDIA_TYPE,
    CandidateTask,
    ExecutionStatus,
    ModelExecution,
    ModelUsage,
)
from spiral_harness.execution.model import paired_execution_fingerprint
from spiral_harness.execution.receipts import publish_execution_receipt
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    publish_evaluation_schedule,
)
from spiral_harness.experiments.baseline_profiles import make_matched_contrast_profile
from spiral_harness.experiments.baselines import BaselineKind, FrozenMutationPolicy
from spiral_harness.experiments.matched_v2 import (
    MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
    MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    MatchedV2ExecutionCeilings,
    MatchedV2GateQueryBlock,
    MatchedV2GateTask,
    MatchedV2PlannedTopology,
    MatchedV2PolicyBindings,
    MatchedV2SharedCoordinates,
    make_matched_v2_run_manifest,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.mechanism import ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE
from spiral_harness.verification.models import Decision


def _json(store: ArtifactStore, label: str, media_type: str = "application/json") -> ArtifactRef:
    return store.put_json({"label": label}, media_type=media_type)


def _record_complete_pair(store, schedule):
    ledger = ledger_for(store, schedule)
    schedule_ref = publish_evaluation_schedule(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)
    records = tuple(
        record_attempt(
            store,
            ledger,
            schedule,
            preflight_ref,
            cell,
            attempt_index=0,
            status=ExecutionStatus.COMPLETED,
            input_tokens=2,
            output_tokens=1,
        )
        for cell in schedule.iter_cells()
    )
    return SimpleNamespace(
        ledger=ledger,
        schedule_ref=schedule_ref,
        preflight_ref=preflight_ref,
        records=records,
    )


def _record_custom_complete_pair(
    store: ArtifactStore,
    schedule: EvaluationBatchSchedule,
    harness_refs: dict[EvaluationSide, ArtifactRef],
) -> SimpleNamespace:
    ledger = ledger_for(store, schedule)
    schedule_ref = publish_evaluation_schedule(store, schedule)
    preflight_ref = persist_preflight(store, schedule, ledger)
    records = []
    spec = runner_spec()
    for cell in schedule.iter_cells():
        harness_ref = harness_refs[cell.side]
        harness = store.get_json(harness_ref, HarnessManifest)
        prompt = store.get_bytes(harness.components[0].artifact).decode()
        task = CandidateTask(task_id=cell.task_id, question=f"Question for {cell.task_id}?")
        request = prompt_request(
            task_id=cell.task_id,
            question=task.question,
            harness_ref=harness_ref,
            system_prompt=prompt,
            seed=schedule.seed_for(cell, attempt_index=0),
        )
        execution = ModelExecution(
            task=task,
            request=request,
            output="answer",
            status=ExecutionStatus.COMPLETED,
            usage=ModelUsage(
                input_tokens=2,
                output_tokens=1,
                latency_ms=1.0,
                cost_usd=None,
            ),
            spec=spec,
            execution_fingerprint=paired_execution_fingerprint(
                spec,
                task,
                seed=request.seed,
                backend_fingerprint=spec.backend_fingerprint,
            ),
            request_sha256=request.fingerprint,
            error=None,
        )
        reservation_ref = ledger.reserve(
            task_fingerprint=execution.task.fingerprint,
            execution_fingerprint=execution.execution_fingerprint,
            request_sha256=execution.request_sha256,
            token_ceiling=schedule.token_ceiling_per_attempt,
        )
        execution_ref = store.put_json(
            execution,
            media_type=MODEL_EXECUTION_MEDIA_TYPE,
        )
        outcome_ref = ledger.settle(reservation_ref, execution_ref=execution_ref)
        receipt_ref = publish_execution_receipt(
            store,
            schedule=schedule,
            cell=cell,
            attempt_index=0,
            preflight_ref=preflight_ref,
            reservation_ref=reservation_ref,
            outcome_ref=outcome_ref,
            ledger_tail_ref=outcome_ref,
        )
        records.append(SimpleNamespace(receipt_ref=receipt_ref))
    return SimpleNamespace(
        ledger=ledger,
        schedule_ref=schedule_ref,
        preflight_ref=preflight_ref,
        records=tuple(records),
    )


def _objective_content(
    fixture: SimpleNamespace,
    *,
    schedule,
    records,
    primary_score: float,
    mean_delta: float,
    confidence_lower: float,
    confidence_upper: float,
    proposal_ref: ArtifactRef | None = None,
    candidate_ref: ArtifactRef | None = None,
    parent_harness_ref: ArtifactRef | None = None,
    candidate_harness_ref: ArtifactRef | None = None,
    regression_rate: float = 0.0,
    tokens_ratio: float = 1.0,
    latency_ratio: float = 1.0,
) -> TrustedObjectiveAggregateContent:
    return TrustedObjectiveAggregateContent(
        search_run_ref=fixture.run_ref,
        proposal_ref=proposal_ref or fixture.proposal_ref,
        candidate_ref=candidate_ref or fixture.candidate_ref,
        parent_harness_ref=(parent_harness_ref or fixture.harness_refs[EvaluationSide.PARENT]),
        candidate_harness_ref=(
            candidate_harness_ref or fixture.harness_refs[EvaluationSide.CANDIDATE]
        ),
        benchmark_binding_ref=fixture.benchmark_ref,
        grader_fingerprint="trusted-grader@fixed-v1",
        schedule_fingerprint=schedule.fingerprint,
        receipt_refs=tuple(record.receipt_ref for record in records.records),
        primary_score=primary_score,
        mean_delta=mean_delta,
        confidence_interval=TrustedObjectiveIntervalEvidence(
            confidence_level=fixture.performance_policy.confidence_level,
            lower=confidence_lower,
            upper=confidence_upper,
            bootstrap_samples=fixture.performance_policy.bootstrap_samples,
            bootstrap_seed=0,
            n_tasks=len(schedule.task_ids),
            n_valid_pairs=(
                len(schedule.task_ids) * len(schedule.search_runs) * len(schedule.repeat_seeds)
            ),
        ),
        regression_rate=regression_rate,
        tokens_ratio=tokens_ratio,
        latency_ratio=latency_ratio,
    )


def _build_fixture(tmp_path) -> SimpleNamespace:
    store = ArtifactStore(tmp_path / "score-artifacts")
    spec = runner_spec()
    objective_service = TrustedObjectiveAggregateService(store, secret=b"o" * 32)
    exploration_ids = ("exploration-01",)
    safe_ref = store.put_json(
        SafeBenchmarkMetadata(
            benchmark_fingerprint="benchmark@fixed-v1",
            exploration_task_ids=exploration_ids,
        ),
        media_type=SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    inputs_ref = store.put_json(
        {"partition": "exploration", "task_ids": list(exploration_ids)},
        media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
    )
    diagnostic_ref = _json(store, "diagnostic", DIAGNOSTIC_CLUSTER_MEDIA_TYPE)
    benchmark = SearchBenchmarkBinding(
        benchmark_fingerprint="benchmark@fixed-v1",
        objective_aggregate_attestor_id=(objective_service.verification_capability.attestor_id),
        strategy_feedback_attestor_id="1" * 64,
        protocol_splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=_json(store, "exploration-split"),
            ),
            ProtocolSplit(
                partition=ProtocolPartition.GATE,
                manifest_ref=_json(store, "gate-split"),
            ),
        ),
        exploration_task_ids=exploration_ids,
        safe_benchmark_metadata_ref=safe_ref,
        exploration_inputs_ref=inputs_ref,
        exploration_aggregates_ref=_json(
            store,
            "exploration-aggregates",
            EXPLORATION_AGGREGATES_MEDIA_TYPE,
        ),
        exploration_item_feedback_ref=_json(
            store,
            "exploration-items",
            EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
        ),
        exploration_trajectories_ref=_json(
            store,
            "exploration-trajectories",
            EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
        ),
        diagnostic_evidence_ref=diagnostic_ref,
        diagnostic_closure_refs=(
            diagnostic_ref,
            _json(store, "closure-1"),
            _json(store, "closure-2"),
            _json(store, "closure-3"),
        ),
    )
    benchmark_ref = store.put_json(benchmark, media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE)

    exploration_schedule, harness_refs = scheduled_batch(
        store,
        spec,
        study="score-v2-study",
        kind=BaselineKind.SCORE_ONLY_MATCHED.value,
        phase=EvaluationPhase.EXPLORATION,
        query=1,
        task_ids=exploration_ids,
        search_runs=(101,),
        repeat_seeds=(11,),
    )
    gate_schedule, gate_harness_refs = scheduled_batch(
        store,
        spec,
        study="score-v2-study",
        kind=BaselineKind.SCORE_ONLY_MATCHED.value,
        phase=EvaluationPhase.GATE,
        query=0,
        task_ids=("gate-task-0",),
        search_runs=(101,),
        repeat_seeds=(11,),
    )
    assert harness_refs == gate_harness_refs
    exploration_records = _record_complete_pair(store, exploration_schedule)
    gate_records = _record_complete_pair(store, gate_schedule)

    blocks = tuple(
        store.put_json(
            MatchedV2GateQueryBlock(
                query_index=index,
                nomination_index=index,
                tasks=(
                    MatchedV2GateTask(
                        task_id=f"gate-task-{index}",
                        source_id=f"source-{index}",
                        family_id=f"family-{index}",
                    ),
                ),
            ),
            media_type=MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
        )
        for index in range(2)
    )
    policy = FrozenMutationPolicy(
        grammar_version="matched-atomic-replace-v2",
        allowed_component_kinds=(ComponentKind.PROMPT, ComponentKind.SKILL),
        max_artifact_size_bytes=8_192,
    )
    contrast = make_matched_contrast_profile(mutation_policy=policy)
    performance_policy = ScorePerformancePolicy(
        confidence_level=0.95,
        bootstrap_samples=10_000,
        minimum_mean_delta=0.05,
        minimum_confidence_lower=0.0,
        maximum_regression_rate=0.05,
        maximum_tokens_ratio=1.2,
        maximum_latency_ratio=1.2,
    )
    shared = MatchedV2SharedCoordinates(
        contrast=contrast,
        contrast_fingerprint=contrast.fingerprint,
        study_id="score-v2-study",
        benchmark_binding_ref=benchmark_ref,
        model_fingerprint=spec.model_fingerprint,
        inference_fingerprint=spec.inference_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        seed_harness_ref=harness_refs[EvaluationSide.PARENT],
        proposal_master_seed=17,
        rollout_master_seed=991,
        search_run_seed=101,
        repeat_seeds=(11,),
        gate_query_block_refs=blocks,
        mutation_policy_fingerprint=canonical_sha256(contrast.score.mutation_policy),
        action_capability_fingerprint=canonical_sha256(contrast.score.action_capability),
        policies=MatchedV2PolicyBindings(
            proposer_policy_fingerprint="2" * 64,
            nomination_policy_fingerprint="3" * 64,
            optimizer_config_fingerprint="4" * 64,
            solver_config_fingerprint="5" * 64,
            grader_fingerprint="trusted-grader@fixed-v1",
            gate_policy_fingerprint="6" * 64,
            performance_policy_fingerprint=performance_policy.fingerprint,
            price_table_fingerprint="7" * 64,
        ),
        planned_topology=MatchedV2PlannedTopology(
            proposer_implementation_fingerprint="8" * 64,
            proposer_call_graph_fingerprint="9" * 64,
        ),
        ceilings=MatchedV2ExecutionCeilings(
            max_rounds=2,
            max_proposals_per_round=2,
            max_total_proposals=4,
            max_total_nominations=2,
            max_optimizer_model_calls=4,
            max_solver_model_calls=16,
            max_gate_queries=2,
            max_evaluations=16,
            max_feedback_queries=2,
            max_attempts_per_evaluation=2,
            token_ceiling_per_attempt=10,
            max_tokens=10_000,
            max_wall_time_seconds=120.0,
            max_cost_usd=2.0,
        ),
    )
    run = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    run_ref = store.put_json(run, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    parent_harness = store.get_json(
        harness_refs[EvaluationSide.PARENT],
        HarnessManifest,
    )
    candidate_harness = store.get_json(
        harness_refs[EvaluationSide.CANDIDATE],
        HarnessManifest,
    )
    hypothesis = MutationHypothesis(
        evidence_refs=(diagnostic_ref,),
        where="system-prompt",
        why="the candidate instruction is more explicit",
        expected_activation="the replacement prompt is loaded",
        expected_adherence="the model follows the replacement instruction",
        expected_behavior="paired task accuracy improves",
        expected_benefit="positive normalized score delta",
        protected_slices=("all",),
        falsifier="no score improvement on the fresh gate block",
        negative_control="the placebo replacement does not improve score",
        risks=("overfitting",),
    )
    hypothesis_ref = store.put_json(
        hypothesis,
        media_type=MUTATION_HYPOTHESIS_MEDIA_TYPE,
    )
    mutation = CandidateMutation(
        target_component="system-prompt",
        before=parent_harness.components[0],
        after=candidate_harness.components[0],
        hypothesis=hypothesis,
    )
    mutation_ref = store.put_json(
        mutation,
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
    )
    candidate = CandidateManifest(
        experiment_ref=_json(store, "experiment", EXPERIMENT_MANIFEST_MEDIA_TYPE),
        parent_harness_ref=harness_refs[EvaluationSide.PARENT],
        child_harness_ref=harness_refs[EvaluationSide.CANDIDATE],
        mutation_ref=mutation_ref,
        evidence_refs=(hypothesis_ref,),
        evaluation_plan_ref=gate_records.schedule_ref,
    )
    candidate_ref = store.put_json(
        candidate,
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    proposal = PromptProposal(
        proposal_id="score-proposal-0",
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        round_index=0,
        parent_harness_ref=harness_refs[EvaluationSide.PARENT],
        target_component_name="system-prompt",
        before_prompt_ref=parent_harness.components[0].artifact,
        after_prompt_ref=candidate_harness.components[0].artifact,
        hypothesis_ref=hypothesis_ref,
        mechanism_family="instruction-clarification",
    )
    proposal_ref = store.put_json(
        proposal,
        media_type=PROMPT_PROPOSAL_MEDIA_TYPE,
    )
    fixture = SimpleNamespace(
        store=store,
        spec=spec,
        objective_service=objective_service,
        benchmark=benchmark,
        benchmark_ref=benchmark_ref,
        harness_refs=harness_refs,
        exploration_schedule=exploration_schedule,
        exploration_records=exploration_records,
        gate_schedule=gate_schedule,
        gate_records=gate_records,
        blocks=blocks,
        shared=shared,
        run=run,
        run_ref=run_ref,
        proposal_ref=proposal_ref,
        candidate_ref=candidate_ref,
        performance_policy=performance_policy,
    )
    exploration_content = _objective_content(
        fixture,
        schedule=exploration_schedule,
        records=exploration_records,
        primary_score=0.70,
        mean_delta=0.10,
        confidence_lower=0.02,
        confidence_upper=0.18,
    )
    gate_content = _objective_content(
        fixture,
        schedule=gate_schedule,
        records=gate_records,
        primary_score=0.72,
        mean_delta=0.12,
        confidence_lower=0.03,
        confidence_upper=0.20,
    )
    exploration_objective_ref = objective_service.attest(exploration_content)
    gate_objective_ref = objective_service.attest(gate_content)
    request = ScoreFeedbackProjectionRequest(
        matched_run_ref=run_ref,
        exploration_objective_aggregate_ref=exploration_objective_ref,
        gate_objective_aggregate_ref=gate_objective_ref,
        gate_query_block_ref=blocks[0],
        round_index=1,
        performance_policy=performance_policy,
    )
    replay_capability = ScoreReceiptReplayCapability(
        store,
        bindings=(
            (
                exploration_records.schedule_ref,
                exploration_records.preflight_ref,
                exploration_records.ledger,
            ),
            (
                gate_records.schedule_ref,
                gate_records.preflight_ref,
                gate_records.ledger,
            ),
        ),
    )
    score_service = ScoreFeedbackAttestationService(
        store,
        secret=b"s" * 32,
        objective_verifier=objective_service.verification_capability,
        receipt_replay_capability=replay_capability,
    )
    fixture.exploration_content = exploration_content
    fixture.gate_content = gate_content
    fixture.exploration_objective_ref = exploration_objective_ref
    fixture.gate_objective_ref = gate_objective_ref
    fixture.replay_capability = replay_capability
    fixture.request = request
    fixture.score_service = score_service
    return fixture


def _request_with(fixture: SimpleNamespace, **updates: object) -> ScoreFeedbackProjectionRequest:
    values = fixture.request.model_dump(mode="python", round_trip=True, warnings="none")
    values.update(updates)
    return ScoreFeedbackProjectionRequest(**values)


def test_attested_score_feedback_is_safe_wrapper_over_exact_sources(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    ref = fixture.score_service.attest(fixture.request)
    feedback = fixture.score_service.verification_capability.verify(ref)
    assert isinstance(feedback, AttestedScoreFeedback)
    assert feedback.source_role_binding_attested is True
    assert feedback.performance_projection_attested is True
    assert feedback.full_mechanism_gate_outcome_consumed is False
    assert feedback.view.runtime_role_binding_attested is False
    assert feedback.view.performance_projection_attested is False
    assert feedback.view.benchmark_metadata_ref == fixture.benchmark.safe_benchmark_metadata_ref
    assert feedback.view.exploration_inputs_ref == fixture.benchmark.exploration_inputs_ref
    assert feedback.view.exploration_aggregate.candidate_score_mean == 0.70
    assert feedback.view.exploration_aggregate.parent_score_mean == 0.60
    assert feedback.view.exploration_aggregate.n_tasks == 1
    assert feedback.view.exploration_aggregate.n_valid_pairs == 1
    assert feedback.view.exploration_aggregate.resources.total_model_calls == 2
    assert feedback.view.prior_gate_decision is not None
    assert feedback.view.prior_gate_decision.decision is Decision.PROMOTE
    assert feedback.view.prior_gate_decision.basis.excluded_mechanism_checks == (
        "activation",
        "adherence",
        "behavior",
    )
    assert "request" not in AttestedScoreFeedback.model_fields
    assert "receipt_refs" not in AttestedScoreFeedback.model_fields
    assert "objective_aggregate_ref" not in AttestedScoreFeedback.model_fields


@pytest.mark.parametrize("role", ("safe", "inputs"))
def test_projector_rejects_arbitrary_or_mechanism_json_relabel(role: str, tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    mechanism_ref = fixture.store.put_json(
        {"passed": True, "mechanism": "activation"},
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    spoofed = mechanism_ref.model_copy(
        update={
            "media_type": (
                SAFE_BENCHMARK_METADATA_MEDIA_TYPE
                if role == "safe"
                else EXPLORATION_INPUTS_MEDIA_TYPE
            )
        }
    )
    values = fixture.benchmark.model_dump(mode="python", round_trip=True, warnings="none")
    values["safe_benchmark_metadata_ref" if role == "safe" else "exploration_inputs_ref"] = spoofed
    benchmark = SearchBenchmarkBinding(**values)
    benchmark_ref = fixture.store.put_json(
        benchmark,
        media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    shared_values = fixture.shared.model_dump(mode="python", round_trip=True, warnings="none")
    shared_values["benchmark_binding_ref"] = benchmark_ref
    shared = MatchedV2SharedCoordinates(**shared_values)
    run = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    run_ref = fixture.store.put_json(run, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    request = _request_with(fixture, matched_run_ref=run_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.attest(request)


def test_full_gate_outcome_cannot_be_retagged_as_performance_decision(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    values = fixture.request.model_dump(mode="python", round_trip=True, warnings="none")
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreFeedbackProjectionRequest(
            **values,
            full_gate_outcome_ref=_json(fixture.store, "full-gate"),
        )

    relabeled = fixture.store.put_json(
        {"decision": "promote", "activation_passed": True},
        media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    )
    request = _request_with(fixture, gate_objective_aggregate_ref=relabeled)
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.attest(request)


def test_performance_policy_rejects_protected_failure_without_full_gate(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    content = fixture.gate_content.model_copy(update={"regression_rate": 0.5})
    aggregate_ref = fixture.objective_service.attest(content)
    request = _request_with(fixture, gate_objective_aggregate_ref=aggregate_ref)
    ref = fixture.score_service.attest(request)
    feedback = fixture.score_service.verification_capability.verify(ref)
    assert feedback.view.prior_gate_decision is not None
    assert feedback.view.prior_gate_decision.decision is Decision.REJECT
    assert feedback.full_mechanism_gate_outcome_consumed is False


def test_projector_rejects_policy_grader_and_receipt_ratio_drift(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    drifted_policy = fixture.performance_policy.model_copy(update={"maximum_tokens_ratio": 1.3})
    request = _request_with(fixture, performance_policy=drifted_policy)
    with pytest.raises(ScoreFeedbackAttestationError, match="performance policy"):
        fixture.score_service.attest(request)

    content = fixture.gate_content.model_copy(update={"grader_fingerprint": "foreign-grader"})
    aggregate_ref = fixture.objective_service.attest(content)
    request = _request_with(fixture, gate_objective_aggregate_ref=aggregate_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="another grader"):
        fixture.score_service.attest(request)

    content = fixture.gate_content.model_copy(update={"tokens_ratio": 1.1})
    aggregate_ref = fixture.objective_service.attest(content)
    request = _request_with(fixture, gate_objective_aggregate_ref=aggregate_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="resource ratios"):
        fixture.score_service.attest(request)


def test_aggregate_receipt_schedule_or_phase_mismatch_fails_closed(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    content = fixture.gate_content.model_copy(update={"schedule_fingerprint": "f" * 64})
    aggregate_ref = fixture.objective_service.attest(content)
    request = _request_with(fixture, gate_objective_aggregate_ref=aggregate_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="receipt closure"):
        fixture.score_service.attest(request)

    content = fixture.gate_content.model_copy(
        update={"receipt_refs": fixture.exploration_content.receipt_refs}
    )
    aggregate_ref = fixture.objective_service.attest(content)
    request = _request_with(fixture, gate_objective_aggregate_ref=aggregate_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="receipt closure"):
        fixture.score_service.attest(request)


def test_objective_wrong_attestor_and_noncanonical_bytes_are_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    foreign_service = TrustedObjectiveAggregateService(fixture.store, secret=b"x" * 32)
    foreign_ref = foreign_service.attest(fixture.gate_content)
    request = _request_with(fixture, gate_objective_aggregate_ref=foreign_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="authentication failed"):
        fixture.score_service.attest(request)

    envelope = fixture.store.get_json(
        fixture.gate_objective_ref,
        TrustedObjectiveAggregate,
    )
    noncanonical_ref = fixture.store.put_bytes(
        envelope.model_dump_json(indent=2).encode(),
        media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    )
    request = _request_with(fixture, gate_objective_aggregate_ref=noncanonical_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.attest(request)

    wrong_size = fixture.run_ref.model_copy(update={"size": fixture.run_ref.size + 1})
    request = _request_with(fixture, matched_run_ref=wrong_size)
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.attest(request)


def test_score_envelope_wrong_attestor_hmac_media_and_canonical_bytes_fail(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    ref = fixture.score_service.attest(fixture.request)
    foreign = ScoreFeedbackVerificationCapability(
        fixture.store,
        secret=b"t" * 32,
        objective_verifier=fixture.objective_service.verification_capability,
        receipt_replay_capability=fixture.replay_capability,
    )
    with pytest.raises(ScoreFeedbackAttestationError, match="another attestor"):
        foreign.verify(ref)

    envelope = fixture.store.get_json(ref, AttestedScoreFeedbackEnvelope)
    forged = envelope.model_copy(update={"authentication_tag": "0" * 64})
    forged_ref = fixture.store.put_json(forged, media_type=ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE)
    with pytest.raises(ScoreFeedbackAttestationError, match="authentication failed"):
        fixture.score_service.verification_capability.verify(forged_ref)

    wrong_media = ref.model_copy(update={"media_type": "application/json"})
    with pytest.raises(ScoreFeedbackAttestationError, match="wrong media type"):
        fixture.score_service.verification_capability.verify(wrong_media)

    noncanonical_ref = fixture.store.put_bytes(
        envelope.model_dump_json(indent=2).encode(),
        media_type=ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE,
    )
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.verification_capability.verify(noncanonical_ref)


def test_valid_hmac_cannot_override_recomputed_projection(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    ref = fixture.score_service.attest(fixture.request)
    envelope = fixture.store.get_json(ref, AttestedScoreFeedbackEnvelope)
    assert envelope.content.view.prior_gate_decision is not None
    forged_decision = envelope.content.view.prior_gate_decision.model_copy(
        update={"decision": Decision.REJECT}
    )
    forged_view = envelope.content.view.model_copy(update={"prior_gate_decision": forged_decision})
    forged_content = envelope.content.model_copy(update={"view": forged_view})
    authentication = hmac.new(
        b"s" * 32,
        b"spiral-harness/score-feedback/v2\x00",
        sha256,
    )
    authentication.update(envelope.attestor_id.encode("ascii") + b"\x00")
    authentication.update(canonical_json_bytes(forged_content))
    forged = AttestedScoreFeedbackEnvelope(
        content=forged_content,
        attestor_id=envelope.attestor_id,
        authentication_tag=authentication.hexdigest(),
    )
    forged_ref = fixture.store.put_json(forged, media_type=ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE)
    with pytest.raises(ScoreFeedbackAttestationError, match="trusted projection"):
        fixture.score_service.verification_capability.verify(forged_ref)


def test_interval_is_objective_attested_and_not_caller_supplied(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    values = fixture.request.model_dump(mode="python", round_trip=True, warnings="none")
    assert "gate_confidence_upper" not in ScoreFeedbackProjectionRequest.model_fields
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreFeedbackProjectionRequest(**values, gate_confidence_upper=0.99)

    drifted_interval = fixture.gate_content.confidence_interval.model_copy(
        update={"confidence_level": 0.90}
    )
    drifted = fixture.gate_content.model_copy(update={"confidence_interval": drifted_interval})
    drifted_ref = fixture.objective_service.attest(drifted)
    request = _request_with(fixture, gate_objective_aggregate_ref=drifted_ref)
    with pytest.raises(ScoreFeedbackAttestationError, match="frozen performance policy"):
        fixture.score_service.attest(request)


@pytest.mark.parametrize(
    ("field_name", "media_type"),
    (
        ("proposal_ref", PROMPT_PROPOSAL_MEDIA_TYPE),
        ("candidate_ref", CANDIDATE_MANIFEST_MEDIA_TYPE),
    ),
)
def test_typed_candidate_lineage_rejects_relabelled_json(
    tmp_path,
    field_name: str,
    media_type: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    spoofed_ref = _json(fixture.store, f"fake-{field_name}", media_type)
    exploration = fixture.exploration_content.model_copy(update={field_name: spoofed_ref})
    gate = fixture.gate_content.model_copy(update={field_name: spoofed_ref})
    request = _request_with(
        fixture,
        exploration_objective_aggregate_ref=fixture.objective_service.attest(exploration),
        gate_objective_aggregate_ref=fixture.objective_service.attest(gate),
    )
    with pytest.raises(ScoreFeedbackAttestationError, match="canonical content"):
        fixture.score_service.attest(request)


def test_live_ledger_advance_and_schedule_ref_spoof_fail_closed(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    wrong_size = fixture.gate_records.schedule_ref.model_copy(
        update={"size": fixture.gate_records.schedule_ref.size + 1}
    )
    with pytest.raises((ValueError, ScoreFeedbackAttestationError)):
        ScoreReceiptReplayCapability(
            fixture.store,
            bindings=(
                (
                    wrong_size,
                    fixture.gate_records.preflight_ref,
                    fixture.gate_records.ledger,
                ),
            ),
        )

    reservation_ref = fixture.gate_records.ledger.reserve(
        task_fingerprint="a" * 64,
        execution_fingerprint="b" * 64,
        request_sha256="c" * 64,
        token_ceiling=fixture.gate_schedule.token_ceiling_per_attempt,
    )
    fixture.gate_records.ledger.burn(
        reservation_ref,
        error_class="unreceipted-ledger-advance",
    )
    with pytest.raises(ScoreFeedbackAttestationError, match="receipt closure"):
        fixture.score_service.attest(fixture.request)


def test_pairing_fingerprint_excludes_intentionally_different_harness_prompts(
    tmp_path,
) -> None:
    fixture = _build_fixture(tmp_path)
    executions = [
        fixture.store.get_json(record.execution_ref, ModelExecution)
        for record in fixture.gate_records.records
    ]
    parent = next(
        execution
        for execution, record in zip(executions, fixture.gate_records.records, strict=True)
        if record.cell.side is EvaluationSide.PARENT
    )
    candidate = next(
        execution
        for execution, record in zip(executions, fixture.gate_records.records, strict=True)
        if record.cell.side is EvaluationSide.CANDIDATE
    )
    assert parent.request_sha256 != candidate.request_sha256
    assert parent.request.system_prompt != candidate.request.system_prompt
    assert parent.execution_fingerprint == candidate.execution_fingerprint


def test_second_round_parent_is_prior_promoted_champion(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    first_ref = fixture.score_service.attest(fixture.request)
    first = fixture.score_service.verification_capability.verify(first_ref)
    prior_parent_ref = fixture.harness_refs[EvaluationSide.CANDIDATE]
    assert first.champion_harness_sha256 == prior_parent_ref.sha256

    prior_parent = fixture.store.get_json(prior_parent_ref, HarnessManifest)
    next_prompt_ref = fixture.store.put_bytes(
        b"Prompt for promoted round two",
        media_type="text/plain",
    )
    next_component = prior_parent.components[0].model_copy(update={"artifact": next_prompt_ref})
    next_harness = prior_parent.model_copy(update={"components": (next_component,)})
    next_harness_ref = fixture.store.put_json(
        next_harness,
        media_type=prior_parent_ref.media_type,
    )
    next_hypothesis = MutationHypothesis(
        evidence_refs=(fixture.benchmark.diagnostic_evidence_ref,),
        where="system-prompt",
        why="the promoted champion still has a diagnosed ambiguity",
        expected_activation="the second replacement prompt is loaded",
        expected_adherence="the model follows the second replacement",
        expected_behavior="fresh-block score improves again",
        expected_benefit="another positive normalized score delta",
        protected_slices=("all",),
        falsifier="the second fresh gate block shows no benefit",
        negative_control="the second placebo shows no benefit",
        risks=("compounding-overfit",),
    )
    next_hypothesis_ref = fixture.store.put_json(
        next_hypothesis,
        media_type=MUTATION_HYPOTHESIS_MEDIA_TYPE,
    )
    next_mutation_ref = fixture.store.put_json(
        CandidateMutation(
            target_component="system-prompt",
            before=prior_parent.components[0],
            after=next_component,
            hypothesis=next_hypothesis,
        ),
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
    )
    next_proposal_ref = fixture.store.put_json(
        PromptProposal(
            proposal_id="score-proposal-1",
            baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
            round_index=1,
            parent_harness_ref=prior_parent_ref,
            target_component_name="system-prompt",
            before_prompt_ref=prior_parent.components[0].artifact,
            after_prompt_ref=next_prompt_ref,
            hypothesis_ref=next_hypothesis_ref,
            mechanism_family="instruction-clarification-round-two",
        ),
        media_type=PROMPT_PROPOSAL_MEDIA_TYPE,
    )

    common_schedule = {
        "study": fixture.shared.study_id,
        "kind": BaselineKind.SCORE_ONLY_MATCHED.value,
        "master_seed": fixture.shared.rollout_master_seed,
        "parent_harness_id": prior_parent_ref.sha256,
        "candidate_harness_id": next_harness_ref.sha256,
        "search_runs": (fixture.shared.search_run_seed,),
        "repeat_seeds": fixture.shared.repeat_seeds,
        "max_attempts_per_cell": fixture.shared.ceilings.max_attempts_per_evaluation,
        "token_ceiling_per_attempt": fixture.shared.ceilings.token_ceiling_per_attempt,
    }
    exploration_schedule = EvaluationBatchSchedule(
        **common_schedule,
        phase=EvaluationPhase.EXPLORATION,
        query=2,
        task_ids=fixture.benchmark.exploration_task_ids,
    )
    gate_schedule = EvaluationBatchSchedule(
        **common_schedule,
        phase=EvaluationPhase.GATE,
        query=1,
        task_ids=("gate-task-1",),
    )
    next_harness_refs = {
        EvaluationSide.PARENT: prior_parent_ref,
        EvaluationSide.CANDIDATE: next_harness_ref,
    }
    exploration_records = _record_custom_complete_pair(
        fixture.store,
        exploration_schedule,
        next_harness_refs,
    )
    gate_records = _record_custom_complete_pair(
        fixture.store,
        gate_schedule,
        next_harness_refs,
    )
    first_candidate = fixture.store.get_json(fixture.candidate_ref, CandidateManifest)
    next_candidate_ref = fixture.store.put_json(
        CandidateManifest(
            experiment_ref=first_candidate.experiment_ref,
            parent_harness_ref=prior_parent_ref,
            child_harness_ref=next_harness_ref,
            mutation_ref=next_mutation_ref,
            evidence_refs=(next_hypothesis_ref,),
            evaluation_plan_ref=gate_records.schedule_ref,
        ),
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    exploration_content = _objective_content(
        fixture,
        schedule=exploration_schedule,
        records=exploration_records,
        primary_score=0.80,
        mean_delta=0.08,
        confidence_lower=0.01,
        confidence_upper=0.15,
        proposal_ref=next_proposal_ref,
        candidate_ref=next_candidate_ref,
        parent_harness_ref=prior_parent_ref,
        candidate_harness_ref=next_harness_ref,
    )
    gate_content = _objective_content(
        fixture,
        schedule=gate_schedule,
        records=gate_records,
        primary_score=0.82,
        mean_delta=0.10,
        confidence_lower=0.02,
        confidence_upper=0.18,
        proposal_ref=next_proposal_ref,
        candidate_ref=next_candidate_ref,
        parent_harness_ref=prior_parent_ref,
        candidate_harness_ref=next_harness_ref,
    )
    exploration_ref = fixture.objective_service.attest(exploration_content)
    gate_ref = fixture.objective_service.attest(gate_content)
    replay_capability = ScoreReceiptReplayCapability(
        fixture.store,
        bindings=(
            (
                fixture.exploration_records.schedule_ref,
                fixture.exploration_records.preflight_ref,
                fixture.exploration_records.ledger,
            ),
            (
                fixture.gate_records.schedule_ref,
                fixture.gate_records.preflight_ref,
                fixture.gate_records.ledger,
            ),
            (
                exploration_records.schedule_ref,
                exploration_records.preflight_ref,
                exploration_records.ledger,
            ),
            (
                gate_records.schedule_ref,
                gate_records.preflight_ref,
                gate_records.ledger,
            ),
        ),
    )
    service = ScoreFeedbackAttestationService(
        fixture.store,
        secret=b"s" * 32,
        objective_verifier=fixture.objective_service.verification_capability,
        receipt_replay_capability=replay_capability,
    )
    request = ScoreFeedbackProjectionRequest(
        matched_run_ref=fixture.run_ref,
        exploration_objective_aggregate_ref=exploration_ref,
        gate_objective_aggregate_ref=gate_ref,
        gate_query_block_ref=fixture.blocks[1],
        round_index=2,
        prior_parent_attestation_ref=first_ref,
        performance_policy=fixture.performance_policy,
    )
    second_ref = service.attest(request)
    second = service.verification_capability.verify(second_ref)
    assert second.view.round_index == 2
    assert second.champion_harness_sha256 == next_harness_ref.sha256

    wrong_parent_request = request.model_copy(update={"prior_parent_attestation_ref": None})
    with pytest.raises(ValidationError, match="prior parent attestation"):
        ScoreFeedbackProjectionRequest.model_validate(wrong_parent_request)
