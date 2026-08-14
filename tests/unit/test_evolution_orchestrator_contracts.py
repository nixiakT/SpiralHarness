from __future__ import annotations

import hmac
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_execution_receipts import (
    ledger_for,
    put_prompt_harness,
    record_attempt,
    runner_spec,
)
from test_terminal_decision import build_graph

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    ExperimentManifest,
    ProtocolPartition,
)
from spiral_harness.core.models import (
    ArtifactRef,
    ComponentKind,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.evidence.models import (
    DiagnosticCluster,
    EvidencePacket,
    EvidenceSpanRef,
    FailureSignature,
    Trajectory,
    TrajectoryEvent,
    TrajectoryEventKind,
)
from spiral_harness.evolution.models import (
    PROMPT_PROPOSAL_MEDIA_TYPE,
    SEARCH_POLICY_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SEARCH_STOPPING_POLICY_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    CandidateScreen,
    CandidateScreenFailure,
    CandidateScreenStatus,
    PromptProposal,
    SearchPolicy,
    SearchRunManifest,
    StrategyFeedbackView,
)
from spiral_harness.evolution.objective_evidence import (
    TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    ObjectiveAggregateVerificationCapability,
    ObjectiveAggregateVerificationError,
    TrustedObjectiveAggregate,
    TrustedObjectiveAggregateContent,
    TrustedObjectiveAggregateService,
    TrustedObjectiveIntervalEvidence,
)
from spiral_harness.evolution.orchestrator import (
    DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE,
    EVIDENCE_PACKET_MEDIA_TYPE,
    EXPLORATION_AGGREGATES_MEDIA_TYPE,
    EXPLORATION_INPUTS_MEDIA_TYPE,
    EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    FAILURE_SIGNATURE_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE,
    AutomaticSearchLoop,
    AutomaticSearchLoopError,
    AutomaticSearchLoopResult,
    CandidateMaterialization,
    ExplorationTrajectoryIndex,
    SafeBenchmarkMetadata,
    SearchAnalysisPlan,
    SearchBenchmarkBinding,
    SearchRunAdmissionError,
    SearchRunAdmissionExpectation,
    SearchRunAdmissionService,
    StrategyArtifactOperation,
    StrategyArtifactView,
    StrategyFeedbackVerificationCapability,
    TrustedScreenEvaluation,
    TrustedStrategyFeedback,
    TrustedStrategyFeedbackContent,
    TrustedStrategyFeedbackService,
)
from spiral_harness.evolution.seeds import derive_strategy_seed
from spiral_harness.evolution.strategies import (
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    PromptMutationCatalogue,
    PromptMutationEntry,
    make_search_policy,
    make_strategy_plugin_manifest,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import ATTEMPT_OUTCOME_MEDIA_TYPE, ExecutionStatus
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceiptIntegrityError,
    TrustedExecutionUsage,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
    SchedulePreflightCertificate,
    preflight_attempt_budget,
    publish_schedule_preflight,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    LEGACY_BASELINE_KINDS,
    BaselineKind,
    BaselineStudyPlan,
    FeedbackType,
    FrozenMutationPolicy,
    FrozenRunContext,
    PairedEvaluationPlan,
    ResourceCeilings,
    plan_four_baselines,
)
from spiral_harness.experiments.search import (
    SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    SearchController,
    SearchControllerManifest,
    TrustedSearchEventService,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def put_json(
    store: ArtifactStore,
    label: str,
    *,
    media_type: str = "application/json",
) -> ArtifactRef:
    return store.put_json({"label": label}, media_type=media_type)


def feedback_binding_artifacts(
    store: ArtifactStore,
    *,
    benchmark_fingerprint: str,
    seed_harness_ref: ArtifactRef,
    seed_prompt_ref: ArtifactRef,
) -> SimpleNamespace:
    task_ids = ("exploration-01", "exploration-02")
    safe_metadata_ref = store.put_json(
        SafeBenchmarkMetadata(
            benchmark_fingerprint=benchmark_fingerprint,
            exploration_task_ids=task_ids,
        ),
        media_type=SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    inputs_ref = store.put_json(
        {"task_ids": list(task_ids)},
        media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
    )
    aggregates_ref = store.put_json(
        {"primary_score": 0.5},
        media_type=EXPLORATION_AGGREGATES_MEDIA_TYPE,
    )
    item_feedback_ref = store.put_json(
        {"task_id": task_ids[0], "outcome": "incorrect"},
        media_type=EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    )
    started_ref = store.put_json({"status": "started"})
    error_ref = store.put_json({"error": "missing-verification"})
    trajectory_ref = store.put_json(
        Trajectory(
            run_id="contract-trajectory-01",
            task_id=task_ids[0],
            seed=701,
            harness_ref=seed_harness_ref,
            execution_fingerprint="contract-execution@sha256:fixed-v1",
            events=(
                TrajectoryEvent(
                    sequence=0,
                    elapsed_ms=0.0,
                    kind=TrajectoryEventKind.RUN_STARTED,
                    payload_ref=started_ref,
                ),
                TrajectoryEvent(
                    sequence=1,
                    elapsed_ms=1.0,
                    kind=TrajectoryEventKind.ERROR,
                    payload_ref=error_ref,
                    causal_parent_sequences=(0,),
                ),
            ),
        ),
        media_type=DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE,
    )
    trajectories_ref = store.put_json(
        ExplorationTrajectoryIndex(
            exploration_task_ids=task_ids,
            trajectory_refs=(trajectory_ref,),
        ),
        media_type=EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    )
    signature_ref = store.put_json(
        FailureSignature(
            failure_id="contract-failure-01",
            failure_class="verification-omission",
            summary="the answer omitted an independent verification",
            symptom_codes=("no-verification",),
            affected_component_refs=(seed_prompt_ref,),
            affected_task_ids=(task_ids[0],),
        ),
        media_type=FAILURE_SIGNATURE_MEDIA_TYPE,
    )
    packet_ref = store.put_json(
        EvidencePacket(
            source_spans=(
                EvidenceSpanRef(
                    trajectory_ref=trajectory_ref,
                    start_sequence=1,
                    end_sequence=1,
                    claim="the run ended before an independent verification",
                    expected_event_kinds=(TrajectoryEventKind.ERROR,),
                ),
            ),
            failure_signature_ref=signature_ref,
        ),
        media_type=EVIDENCE_PACKET_MEDIA_TYPE,
    )
    cluster_ref = store.put_json(
        DiagnosticCluster(
            cluster_id="contract-cluster-01",
            label="missing independent verification",
            failure_signature_refs=(signature_ref,),
            representative_signature_ref=signature_ref,
            evidence_packet_refs=(packet_ref,),
            task_ids=(task_ids[0],),
        ),
        media_type=DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    )
    return SimpleNamespace(
        task_ids=task_ids,
        safe_metadata_ref=safe_metadata_ref,
        inputs_ref=inputs_ref,
        aggregates_ref=aggregates_ref,
        item_feedback_ref=item_feedback_ref,
        trajectories_ref=trajectories_ref,
        diagnostic_ref=cluster_ref,
        diagnostic_closure=(
            cluster_ref,
            signature_ref,
            packet_ref,
            trajectory_ref,
            seed_harness_ref,
            seed_prompt_ref,
            started_ref,
            error_ref,
        ),
    )


def objective_content(store: ArtifactStore) -> TrustedObjectiveAggregateContent:
    return TrustedObjectiveAggregateContent(
        search_run_ref=put_json(
            store,
            "run",
            media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
        ),
        proposal_ref=put_json(
            store,
            "proposal",
            media_type=PROMPT_PROPOSAL_MEDIA_TYPE,
        ),
        candidate_ref=put_json(store, "candidate"),
        parent_harness_ref=put_json(store, "parent"),
        candidate_harness_ref=put_json(store, "child"),
        benchmark_binding_ref=put_json(
            store,
            "benchmark",
            media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
        ),
        grader_fingerprint="trusted-grader@sha256:fixed-v1",
        schedule_fingerprint="a" * 64,
        receipt_refs=(put_json(store, "receipt", media_type=EXECUTION_RECEIPT_MEDIA_TYPE),),
        primary_score=0.8,
        mean_delta=0.1,
        confidence_interval=TrustedObjectiveIntervalEvidence(
            confidence_level=0.95,
            lower=0.02,
            upper=0.18,
            bootstrap_samples=10_000,
            bootstrap_seed=0,
            n_tasks=1,
            n_valid_pairs=1,
        ),
        regression_rate=0.0,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )


def hypothesis(evidence_ref: ArtifactRef, *, where: str = "system") -> MutationHypothesis:
    return MutationHypothesis(
        evidence_refs=(evidence_ref,),
        where=where,
        why="the current instruction omits verification",
        expected_activation="the final answer invokes verification",
        expected_adherence="the verification step is completed",
        expected_behavior="arithmetic is checked independently",
        expected_benefit="fewer incorrect answers",
        protected_slices=("already-correct",),
        falsifier="verification activates without reducing failures",
        negative_control="unrelated formatting remains unchanged",
        risks=("additional tokens",),
    )


class NeverTerminalVerifier:
    def verify_terminal_authorization(self, ref: ArtifactRef) -> object:
        del ref
        raise AssertionError("empty-round fixture must never request terminal authorization")


class LedgerRuntime:
    def __init__(self, ledger: object) -> None:
        self.ledger = ledger

    def attempt_ledger_for(self, evaluation_ref: ArtifactRef) -> object:
        del evaluation_ref
        return self.ledger


class FixedFeedbackRuntime:
    def __init__(self, feedback_ref: ArtifactRef) -> None:
        self.feedback_ref = feedback_ref
        self.call_count = 0

    def collect_feedback(self, **_: object) -> object:
        self.call_count += 1
        return self.feedback_ref


def build_admission_fixture(tmp_path: Path) -> SimpleNamespace:
    graph = build_graph(tmp_path)
    store = graph.store
    protocol = graph.protocol
    seed_harness_ref = graph.candidate.parent_harness_ref
    seed_harness = store.get_json(seed_harness_ref, HarnessManifest)
    seed_prompt = next(
        component for component in seed_harness.components if component.kind is ComponentKind.PROMPT
    )
    base_experiment = store.get_json(graph.experiment_ref, ExperimentManifest)
    experiment = base_experiment.model_copy(
        update={
            "objective": "benchmark-score",
            "baselines": tuple(kind.value for kind in LEGACY_BASELINE_KINDS),
        }
    )
    experiment_ref = store.put_json(
        experiment,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )

    objective_service = TrustedObjectiveAggregateService(store, secret=b"o" * 32)
    feedback_service = TrustedStrategyFeedbackService(store, secret=b"v" * 32)
    frozen_feedback = feedback_binding_artifacts(
        store,
        benchmark_fingerprint=protocol.benchmark_fingerprint,
        seed_harness_ref=seed_harness_ref,
        seed_prompt_ref=seed_prompt.artifact,
    )
    benchmark = SearchBenchmarkBinding(
        benchmark_fingerprint=protocol.benchmark_fingerprint,
        objective_aggregate_attestor_id=(objective_service.verification_capability.attestor_id),
        strategy_feedback_attestor_id=(feedback_service.verification_capability.attestor_id),
        protocol_splits=protocol.splits,
        exploration_task_ids=frozen_feedback.task_ids,
        safe_benchmark_metadata_ref=frozen_feedback.safe_metadata_ref,
        exploration_inputs_ref=frozen_feedback.inputs_ref,
        exploration_aggregates_ref=frozen_feedback.aggregates_ref,
        exploration_item_feedback_ref=frozen_feedback.item_feedback_ref,
        exploration_trajectories_ref=frozen_feedback.trajectories_ref,
        diagnostic_evidence_ref=frozen_feedback.diagnostic_ref,
        diagnostic_closure_refs=frozen_feedback.diagnostic_closure,
    )
    benchmark_ref = store.put_json(
        benchmark,
        media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    mutation_policy = FrozenMutationPolicy(
        grammar_version="atomic-prompt-replace-v1",
        allowed_component_kinds=(ComponentKind.PROMPT,),
        max_artifact_size_bytes=experiment.mutation_policy.max_artifact_size_bytes,
    )
    evaluation = PairedEvaluationPlan(
        search_run_seeds=(701, 702),
        repeat_seeds=(11, 12),
    )
    study = plan_four_baselines(
        context=FrozenRunContext(
            benchmark_ref=benchmark_ref,
            model_fingerprint=protocol.model_fingerprint,
            inference_fingerprint=protocol.inference_fingerprint,
            runtime_fingerprint=protocol.runtime_fingerprint,
            seed_harness_ref=seed_harness_ref,
            mutation_policy=mutation_policy,
            proposal_random_seed=991,
        ),
        evaluation=evaluation,
        ceilings=ResourceCeilings(
            max_evaluations=20,
            max_feedback_queries=2,
            max_proposals=2,
            max_optimizer_model_calls=4,
            max_tokens=20_000,
            max_wall_time_seconds=60.0,
            max_cost_usd=1.0,
        ),
    )
    study_ref = store.put_json(study, media_type=BASELINE_STUDY_PLAN_MEDIA_TYPE)
    analysis_ref = store.put_json(
        SearchAnalysisPlan(
            family_alpha=0.05,
            max_gate_queries=1,
            gate_confidence_level=0.95,
            search_run_seeds=evaluation.search_run_seeds,
            repeat_seeds=evaluation.repeat_seeds,
        ),
        media_type=SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
    )

    evidence_ref = put_json(store, "catalogue-evidence")
    after_prompt_ref = store.put_bytes(b"Verify before answering.", media_type="text/plain")
    hypothesis_ref = store.put_json(
        hypothesis(evidence_ref),
        media_type="application/vnd.spiral-harness.mutation-hypothesis.v1+json",
    )
    catalogue = PromptMutationCatalogue(
        catalogue_id="frozen-random-valid-v1",
        grammar_version=mutation_policy.grammar_version,
        parent_harness_ref=seed_harness_ref,
        entries=(
            PromptMutationEntry(
                entry_id="entry-01",
                target_component_name=seed_prompt.name,
                expected_before_prompt_ref=seed_prompt.artifact,
                after_prompt_ref=after_prompt_ref,
                hypothesis_ref=hypothesis_ref,
                mechanism_family="explicit-verification",
            ),
        ),
    )
    catalogue_ref = store.put_json(
        catalogue,
        media_type=PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    )

    runs: dict[BaselineKind, tuple[ArtifactRef, ArtifactRef]] = {}
    controllers: dict[BaselineKind, SearchController] = {}
    for kind in LEGACY_BASELINE_KINDS:
        static = kind is BaselineKind.STATIC
        policy = make_search_policy(
            baseline_kind=kind,
            mutation_policy=mutation_policy,
            max_rounds=0 if static else 1,
            max_gate_queries=0 if static else 1,
            patience_rounds=0 if static else 1,
            max_consecutive_declines=0 if static else 1,
            max_diagnoses=1 if kind is BaselineKind.EVIDENCE_TARGETED else 0,
            max_proposals=0 if static else 1,
            max_screens=0 if static else 1,
            max_proposals_per_round=0 if static else 1,
            max_candidates_screened_per_round=0 if static else 1,
            family_alpha=0.05,
        )
        policy_ref = store.put_json(policy, media_type=SEARCH_POLICY_MEDIA_TYPE)
        stopping_ref = store.put_json(
            policy.stopping_policy,
            media_type=SEARCH_STOPPING_POLICY_MEDIA_TYPE,
        )
        implementation_ref = store.put_bytes(
            f"fixture strategy: {kind.value}".encode(),
            media_type="text/x-python",
        )
        plugin_ref = store.put_json(
            make_strategy_plugin_manifest(
                plugin_id=f"fixture-{kind.value}",
                plugin_version="1",
                implementation_ref=implementation_ref,
                baseline_kind=kind,
            ),
            media_type=STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
        )
        run = SearchRunManifest(
            baseline_study_plan_ref=study_ref,
            experiment_ref=experiment_ref,
            search_policy_ref=policy_ref,
            search_policy_fingerprint=policy_ref.sha256,
            strategy_plugin_ref=plugin_ref,
            strategy_plugin_fingerprint=plugin_ref.sha256,
            analysis_plan_ref=analysis_ref,
            stopping_policy_ref=stopping_ref,
            stopping_policy_fingerprint=stopping_ref.sha256,
            seed_harness_ref=seed_harness_ref,
            baseline_kind=kind,
            baseline_plan_fingerprint=study_ref.sha256,
            proposal_master_seed=study.arm(kind).context.proposal_random_seed,
            search_run_seed=evaluation.search_run_seeds[0],
            repeat_seeds=evaluation.repeat_seeds,
            strategy_seed=derive_strategy_seed(
                proposal_master_seed=study.arm(kind).context.proposal_random_seed,
                search_run_seed=evaluation.search_run_seeds[0],
                baseline_kind=kind,
            ),
            prompt_mutation_catalogue_ref=(
                catalogue_ref if kind is BaselineKind.RANDOM_VALID else None
            ),
        )
        run_ref = store.put_json(run, media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
        event_service = TrustedSearchEventService(
            secret=f"contract-{kind.value}".encode().ljust(32, b"x")
        )
        controller_ref = store.put_json(
            SearchControllerManifest(
                search_run_ref=run_ref,
                study_ref=study_ref,
                experiment_ref=experiment_ref,
                run_id=f"{kind.value}-701",
                baseline_kind=kind,
                search_seed=run.search_run_seed,
                initial_champion_harness_ref=seed_harness_ref,
                analysis_plan_ref=analysis_ref,
                max_rounds=policy.max_rounds,
                max_gate_nominations=policy.max_gate_queries,
                patience_rounds=policy.patience_rounds,
                max_consecutive_declines=policy.max_consecutive_declines,
                journal_attestor_id=event_service.attestor_id,
            ),
            media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
        )
        runs[kind] = (run_ref, controller_ref)
        controllers[kind] = SearchController(
            store,
            controller_manifest_ref=controller_ref,
            event_service=event_service,
            terminal_authorization_verifier=(None if static else NeverTerminalVerifier()),
        )

    return SimpleNamespace(
        store=store,
        objective_service=objective_service,
        feedback_service=feedback_service,
        study_ref=study_ref,
        experiment_ref=experiment_ref,
        analysis_ref=analysis_ref,
        benchmark=benchmark,
        benchmark_ref=benchmark_ref,
        frozen_feedback=frozen_feedback,
        runs=runs,
        controllers=controllers,
        protocol=protocol,
        seed_harness=seed_harness,
        seed_harness_ref=seed_harness_ref,
        after_prompt_ref=after_prompt_ref,
        hypothesis_ref=hypothesis_ref,
    )


def legal_feedback_content(
    env: SimpleNamespace,
    *,
    baseline_kind: BaselineKind = BaselineKind.PROMPT_ONLY,
) -> TrustedStrategyFeedbackContent:
    run_ref, _ = env.runs[baseline_kind]
    run = env.store.get_json(run_ref, SearchRunManifest)
    benchmark = env.benchmark
    values: dict[str, object] = {
        "baseline_kind": baseline_kind,
        "benchmark_metadata_ref": benchmark.safe_benchmark_metadata_ref,
    }
    exposed = {FeedbackType.BENCHMARK_METADATA}
    if baseline_kind is not BaselineKind.STATIC:
        values["exploration_inputs_ref"] = benchmark.exploration_inputs_ref
        exposed.add(FeedbackType.EXPLORATION_INPUTS)
    if baseline_kind in {BaselineKind.PROMPT_ONLY, BaselineKind.EVIDENCE_TARGETED}:
        values.update(
            exploration_aggregates_ref=benchmark.exploration_aggregates_ref,
            exploration_item_feedback_ref=benchmark.exploration_item_feedback_ref,
            exploration_trajectories_ref=benchmark.exploration_trajectories_ref,
        )
        exposed.update(
            {
                FeedbackType.EXPLORATION_AGGREGATES,
                FeedbackType.EXPLORATION_ITEM_FEEDBACK,
                FeedbackType.EXPLORATION_TRAJECTORIES,
            }
        )
    if baseline_kind is BaselineKind.EVIDENCE_TARGETED:
        values["diagnostic_evidence_ref"] = benchmark.diagnostic_evidence_ref
        exposed.add(FeedbackType.DIAGNOSTIC_EVIDENCE)
    view = StrategyFeedbackView(exposed_feedback=tuple(exposed), **values)
    return TrustedStrategyFeedbackContent(
        search_run_ref=run_ref,
        experiment_ref=run.experiment_ref,
        benchmark_binding_ref=env.benchmark_ref,
        exploration_split_ref=benchmark.exploration_split_ref,
        baseline_kind=baseline_kind,
        search_run_seed=run.search_run_seed,
        round_index=0,
        champion_harness_ref=env.seed_harness_ref,
        prior_gate_aggregate=None,
        view=view,
    )


def open_feedback_context(
    env: SimpleNamespace,
    *,
    runtime: object,
    baseline_kind: BaselineKind = BaselineKind.PROMPT_ONLY,
) -> SimpleNamespace:
    run_ref, _ = env.runs[baseline_kind]
    run = env.store.get_json(run_ref, SearchRunManifest)
    controller = env.controllers[baseline_kind]
    tail = controller.freeze()
    tail = controller.start_search(previous_tail_ref=tail)
    tail = controller.open_round(previous_tail_ref=tail)
    loop = AutomaticSearchLoop(
        env.store,
        runtime=runtime,  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=env.objective_service.verification_capability,
        strategy_feedback_verifier=env.feedback_service.verification_capability,
    )
    return SimpleNamespace(
        run_ref=run_ref,
        run=run,
        controller=controller,
        tail=tail,
        loop=loop,
    )


def build_receipt_backed_screen(tmp_path: Path) -> SimpleNamespace:
    env = build_admission_fixture(tmp_path)
    store = env.store
    run_ref, _ = env.runs[BaselineKind.PROMPT_ONLY]
    run = store.get_json(run_ref, SearchRunManifest)
    seed_prompt = next(
        component
        for component in env.seed_harness.components
        if component.kind is ComponentKind.PROMPT
    )
    execution_spec = runner_spec()
    study = store.get_json(run.baseline_study_plan_ref, BaselineStudyPlan)
    frozen_context = study.arms[0].context.model_copy(
        update={
            "model_fingerprint": execution_spec.model_fingerprint,
            "inference_fingerprint": execution_spec.inference_fingerprint,
            "runtime_fingerprint": execution_spec.runtime_fingerprint,
        }
    )
    study = study.model_copy(
        update={
            "arms": tuple(arm.model_copy(update={"context": frozen_context}) for arm in study.arms)
        }
    )
    study_ref = store.put_json(study, media_type=BASELINE_STUDY_PLAN_MEDIA_TYPE)
    protocol = env.protocol.model_copy(
        update={
            "model_fingerprint": execution_spec.model_fingerprint,
            "inference_fingerprint": execution_spec.inference_fingerprint,
            "runtime_fingerprint": execution_spec.runtime_fingerprint,
            "model_spec_fingerprint": execution_spec.fingerprint,
        }
    )
    protocol_ref = store.put_json(protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = store.get_json(run.experiment_ref, ExperimentManifest)
    experiment_ref = store.put_json(
        experiment.model_copy(update={"protocol_ref": protocol_ref}),
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    run = run.model_copy(
        update={
            "baseline_study_plan_ref": study_ref,
            "experiment_ref": experiment_ref,
            "baseline_plan_fingerprint": study_ref.sha256,
        }
    )
    run_ref = store.put_json(run, media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
    parent_harness_ref = put_prompt_harness(store, execution_spec, EvaluationSide.PARENT)
    candidate_harness_ref = put_prompt_harness(store, execution_spec, EvaluationSide.CANDIDATE)
    candidate_ref = put_json(store, "screen-candidate")
    proposal = PromptProposal(
        proposal_id="receipt-backed-proposal",
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        parent_harness_ref=parent_harness_ref,
        target_component_name=seed_prompt.name,
        before_prompt_ref=seed_prompt.artifact,
        after_prompt_ref=env.after_prompt_ref,
        hypothesis_ref=env.hypothesis_ref,
        mechanism_family="explicit-verification",
        proposer_confidence=0.8,
    )
    proposal_ref = store.put_json(proposal, media_type=PROMPT_PROPOSAL_MEDIA_TYPE)
    schedule = EvaluationBatchSchedule(
        study=study_ref.sha256,
        kind=BaselineKind.PROMPT_ONLY.value,
        phase=EvaluationPhase.EXPLORATION,
        query=0,
        master_seed=run.search_run_seed,
        parent_harness_id=parent_harness_ref.sha256,
        candidate_harness_id=candidate_harness_ref.sha256,
        task_ids=("exploration-01", "exploration-02"),
        search_runs=(run.search_run_seed,),
        repeat_seeds=run.repeat_seeds,
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=10,
    )
    ledger = ledger_for(store, schedule, extra_attempts=1)
    preflight = preflight_attempt_budget(schedule, ledger, execution_spec)
    preflight_ref = publish_schedule_preflight(store, preflight)
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
    usage = replay_trusted_usage(
        store,
        schedule=schedule,
        preflight_ref=preflight_ref,
        attempt_ledger=ledger,
        receipt_refs=tuple(record.receipt_ref for record in records),
    )
    metrics = {
        "primary_score": 0.75,
        "mean_delta": 0.10,
        "confidence_lower": 0.02,
        "regression_rate": 0.0,
        "tokens_ratio": 1.0,
        "latency_ratio": 1.0,
    }
    aggregate_ref = env.objective_service.attest(
        TrustedObjectiveAggregateContent(
            search_run_ref=run_ref,
            proposal_ref=proposal_ref,
            candidate_ref=candidate_ref,
            parent_harness_ref=parent_harness_ref,
            candidate_harness_ref=candidate_harness_ref,
            benchmark_binding_ref=env.benchmark_ref,
            grader_fingerprint=env.protocol.grader_fingerprint,
            schedule_fingerprint=schedule.fingerprint,
            receipt_refs=usage.receipt_refs,
            primary_score=metrics["primary_score"],
            mean_delta=metrics["mean_delta"],
            confidence_interval=TrustedObjectiveIntervalEvidence(
                confidence_level=0.95,
                lower=metrics["confidence_lower"],
                upper=0.18,
                bootstrap_samples=10_000,
                bootstrap_seed=0,
                n_tasks=len(schedule.task_ids),
                n_valid_pairs=(
                    len(schedule.task_ids) * len(schedule.search_runs) * len(schedule.repeat_seeds)
                ),
            ),
            regression_rate=metrics["regression_rate"],
            tokens_ratio=metrics["tokens_ratio"],
            latency_ratio=metrics["latency_ratio"],
        )
    )
    evaluation = TrustedScreenEvaluation(
        search_run_ref=run_ref,
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        proposal_ref=proposal_ref,
        candidate_ref=candidate_ref,
        parent_harness_ref=parent_harness_ref,
        candidate_harness_ref=candidate_harness_ref,
        schedule=schedule,
        preflight_ref=preflight_ref,
        receipt_refs=usage.receipt_refs,
        final_ledger_tail_ref=ledger.tail_ref,
        trusted_usage=usage,
        objective_aggregate_ref=aggregate_ref,
        **metrics,
    )
    evaluation_ref = store.put_json(
        evaluation,
        media_type=TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    )
    screen = CandidateScreen(
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        proposal_ref=proposal_ref,
        candidate_ref=candidate_ref,
        candidate_harness_ref=candidate_harness_ref,
        evaluation_ref=evaluation_ref,
        status=CandidateScreenStatus.ELIGIBLE,
        **metrics,
    )
    runtime = LedgerRuntime(ledger)
    loop = AutomaticSearchLoop(
        store,
        runtime=runtime,  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=env.objective_service.verification_capability,
        strategy_feedback_verifier=env.feedback_service.verification_capability,
    )
    return SimpleNamespace(
        **vars(env),
        run=run,
        run_ref=run_ref,
        proposal_ref=proposal_ref,
        candidate_ref=candidate_ref,
        screen_parent_harness_ref=parent_harness_ref,
        candidate_harness_ref=candidate_harness_ref,
        schedule=schedule,
        preflight=preflight,
        ledger=ledger,
        usage=usage,
        aggregate_ref=aggregate_ref,
        evaluation=evaluation,
        evaluation_ref=evaluation_ref,
        screen=screen,
        loop=loop,
    )


def test_objective_aggregate_requires_independent_exact_capability(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    content = objective_content(store)
    service = TrustedObjectiveAggregateService(store, secret=b"g" * 32)
    feedback_service = TrustedStrategyFeedbackService(store, secret=b"h" * 32)
    aggregate_ref = service.attest(content)

    assert service.verification_capability.verify(aggregate_ref) == content
    loop = AutomaticSearchLoop(
        store,
        runtime=object(),  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=service.verification_capability,
        strategy_feedback_verifier=feedback_service.verification_capability,
    )
    loop._verify_objective_attestor(service.verification_capability.attestor_id)
    foreign_service = TrustedObjectiveAggregateService(store, secret=b"f" * 32)
    with pytest.raises(AutomaticSearchLoopError, match="frozen benchmark authority"):
        loop._verify_objective_attestor(foreign_service.verification_capability.attestor_id)

    class DuckVerifier:
        def verify(self, aggregate_ref: ArtifactRef) -> TrustedObjectiveAggregateContent:
            del aggregate_ref
            return content

    with pytest.raises(TypeError, match="exact trusted capability"):
        AutomaticSearchLoop(
            store,
            runtime=object(),  # type: ignore[arg-type]
            lifecycle=object(),  # type: ignore[arg-type]
            objective_aggregate_verifier=DuckVerifier(),  # type: ignore[arg-type]
            strategy_feedback_verifier=feedback_service.verification_capability,
        )

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class EvilVerifier(ObjectiveAggregateVerificationCapability):
            pass


def test_objective_aggregate_v4_binds_interval_receipts_attestor_and_hmac_domain(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    content = objective_content(store)
    secret = b"v" * 32
    service = TrustedObjectiveAggregateService(store, secret=secret)
    aggregate_ref = service.attest(content)
    envelope = store.get_json(aggregate_ref, TrustedObjectiveAggregate)

    assert TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE.endswith(".v4+json")
    assert aggregate_ref.media_type == TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE
    assert content.schema_version == "4"
    assert envelope.schema_version == "4"
    assert service.verification_capability.attestor_id == sha256_bytes(
        b"spiral-harness/objective-aggregate-attestor/v4\x00" + secret
    )
    expected = hmac.new(secret, b"spiral-harness/objective-aggregate/v4\x00", sha256)
    expected.update(envelope.attestor_id.encode("ascii") + b"\x00")
    expected.update(canonical_json_bytes(envelope.content))
    assert envelope.authentication_tag == expected.hexdigest()

    content_payload = content.model_dump(mode="python", round_trip=True, warnings="none")
    content_payload["receipt_refs"] = (
        put_json(
            store,
            "legacy-receipt",
            media_type="application/vnd.spiral-harness.execution-receipt.v1+json",
        ),
    )
    with pytest.raises(ValidationError, match="must be execution receipts"):
        TrustedObjectiveAggregateContent.model_validate(content_payload, strict=True)


def test_screen_evaluation_v3_rejects_legacy_receipt_and_outcome_refs(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    schedule = EvaluationBatchSchedule(
        study="screen-v3-contract",
        kind=BaselineKind.PROMPT_ONLY.value,
        phase=EvaluationPhase.EXPLORATION,
        query=0,
        master_seed=17,
        parent_harness_id="parent-v3",
        candidate_harness_id="candidate-v3",
        task_ids=("task-v3",),
        search_runs=(17,),
        repeat_seeds=(23,),
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=10,
    )
    receipt_ref = put_json(store, "receipt-v3", media_type=EXECUTION_RECEIPT_MEDIA_TYPE)
    outcome_ref = put_json(store, "outcome-v3", media_type=ATTEMPT_OUTCOME_MEDIA_TYPE)
    usage = TrustedExecutionUsage(
        schedule_fingerprint=schedule.fingerprint,
        receipt_refs=(receipt_ref,),
        ledger_tail_refs=(outcome_ref,),
        cell_count=1,
        attempt_count=1,
        settled_attempts=1,
        burned_attempts=0,
        poisoned_attempts=0,
        reported_tokens=3,
        charged_tokens=3,
    )
    evaluation = TrustedScreenEvaluation(
        search_run_ref=put_json(store, "run-v3", media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE),
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        proposal_ref=put_json(store, "proposal-v3", media_type=PROMPT_PROPOSAL_MEDIA_TYPE),
        candidate_ref=put_json(store, "candidate-v3"),
        parent_harness_ref=put_json(store, "parent-v3"),
        candidate_harness_ref=put_json(store, "harness-v3"),
        schedule=schedule,
        preflight_ref=put_json(store, "preflight-v3", media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE),
        receipt_refs=(receipt_ref,),
        final_ledger_tail_ref=outcome_ref,
        trusted_usage=usage,
        objective_aggregate_ref=put_json(
            store,
            "objective-v3",
            media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
        ),
        primary_score=0.8,
        mean_delta=0.1,
        confidence_lower=0.02,
        regression_rate=0.0,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )
    evaluation_ref = store.put_json(
        evaluation,
        media_type=TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    )
    payload = evaluation.model_dump(mode="python", round_trip=True, warnings="none")

    assert TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE.endswith(".v3+json")
    assert evaluation_ref.media_type == TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE
    assert evaluation.schema_version == "3"
    assert evaluation.trusted_usage.schema_version == "3"
    assert all(ref.media_type == EXECUTION_RECEIPT_MEDIA_TYPE for ref in evaluation.receipt_refs)
    assert evaluation.final_ledger_tail_ref.media_type == ATTEMPT_OUTCOME_MEDIA_TYPE

    payload["receipt_refs"] = tuple(
        ref.model_copy(
            update={"media_type": "application/vnd.spiral-harness.execution-receipt.v1+json"}
        )
        for ref in evaluation.receipt_refs
    )
    with pytest.raises(ValidationError, match="must be execution receipts"):
        TrustedScreenEvaluation.model_validate(payload, strict=True)

    payload = evaluation.model_dump(mode="python", round_trip=True, warnings="none")
    payload["final_ledger_tail_ref"] = evaluation.final_ledger_tail_ref.model_copy(
        update={"media_type": "application/vnd.spiral-harness.attempt-outcome.v1+json"}
    )
    with pytest.raises(ValidationError, match="final_ledger_tail_ref declares the wrong"):
        TrustedScreenEvaluation.model_validate(payload, strict=True)


def test_shared_analysis_plan_admits_static_and_all_three_search_arms(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    admission = SearchRunAdmissionService(env.store)

    reports = {}
    for kind, (run_ref, controller_ref) in env.runs.items():
        reports[kind] = admission.admit(
            search_run_ref=run_ref,
            controller_manifest_ref=controller_ref,
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=env.study_ref,
                experiment_ref=env.experiment_ref,
                baseline_kind=kind,
                search_run_seed=701,
            ),
        )

    assert frozenset(reports) == frozenset(LEGACY_BASELINE_KINDS)
    assert {report.analysis_plan_ref for report in reports.values()} == {env.analysis_ref}
    assert {report.benchmark_binding_ref for report in reports.values()} == {env.benchmark_ref}
    assert reports[BaselineKind.STATIC].gate_confidence_level is None
    assert all(
        report.gate_confidence_level == 0.95
        for kind, report in reports.items()
        if kind is not BaselineKind.STATIC
    )


def test_random_valid_catalogue_missing_child_is_an_admission_error(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    run_ref, controller_ref = env.runs[BaselineKind.RANDOM_VALID]
    run = env.store.get_json(run_ref, SearchRunManifest)
    assert run.prompt_mutation_catalogue_ref is not None
    catalogue = env.store.get_json(
        run.prompt_mutation_catalogue_ref,
        PromptMutationCatalogue,
    )
    missing_prompt_ref = ArtifactRef(
        sha256="f" * 64,
        size=17,
        media_type="text/plain",
    )
    broken_entry = catalogue.entries[0].model_copy(update={"after_prompt_ref": missing_prompt_ref})
    broken_catalogue_ref = env.store.put_json(
        catalogue.model_copy(update={"entries": (broken_entry,)}),
        media_type=PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    )
    broken_run_ref = env.store.put_json(
        run.model_copy(update={"prompt_mutation_catalogue_ref": broken_catalogue_ref}),
        media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    )
    controller_manifest = env.store.get_json(
        controller_ref,
        SearchControllerManifest,
    )
    broken_controller_ref = env.store.put_json(
        controller_manifest.model_copy(update={"search_run_ref": broken_run_ref}),
        media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    )

    with pytest.raises(
        SearchRunAdmissionError,
        match="random-valid catalogue artifacts could not be strictly loaded",
    ):
        SearchRunAdmissionService(env.store).admit(
            search_run_ref=broken_run_ref,
            controller_manifest_ref=broken_controller_ref,
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=env.study_ref,
                experiment_ref=env.experiment_ref,
                baseline_kind=BaselineKind.RANDOM_VALID,
                search_run_seed=701,
            ),
        )


def test_full_loop_rejects_verifier_outside_frozen_benchmark_authority(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    run_ref, controller_ref = env.runs[BaselineKind.PROMPT_ONLY]
    foreign_service = TrustedObjectiveAggregateService(env.store, secret=b"f" * 32)
    loop = AutomaticSearchLoop(
        env.store,
        runtime=object(),  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=foreign_service.verification_capability,
        strategy_feedback_verifier=env.feedback_service.verification_capability,
    )

    with pytest.raises(AutomaticSearchLoopError, match="frozen benchmark authority"):
        loop.run(
            search_run_ref=run_ref,
            controller=SimpleNamespace(controller_manifest_ref=controller_ref),  # type: ignore[arg-type]
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=env.study_ref,
                experiment_ref=env.experiment_ref,
                baseline_kind=BaselineKind.PROMPT_ONLY,
                search_run_seed=701,
            ),
        )


def test_full_loop_rejects_feedback_verifier_before_runtime_call(tmp_path: Path) -> None:
    env = build_admission_fixture(tmp_path)
    run_ref, _ = env.runs[BaselineKind.PROMPT_ONLY]
    legal_ref = env.feedback_service.attest(legal_feedback_content(env))
    runtime = FixedFeedbackRuntime(legal_ref)
    foreign_service = TrustedStrategyFeedbackService(env.store, secret=b"f" * 32)
    loop = AutomaticSearchLoop(
        env.store,
        runtime=runtime,  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=env.objective_service.verification_capability,
        strategy_feedback_verifier=foreign_service.verification_capability,
    )

    with pytest.raises(AutomaticSearchLoopError, match="frozen benchmark authority"):
        loop.run(
            search_run_ref=run_ref,
            controller=env.controllers[BaselineKind.PROMPT_ONLY],
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=env.study_ref,
                experiment_ref=env.experiment_ref,
                baseline_kind=BaselineKind.PROMPT_ONLY,
                search_run_seed=701,
            ),
        )
    assert runtime.call_count == 0


@pytest.mark.parametrize("attack", ["direct", "copied", "wrapped"])
def test_feedback_signer_rejects_gate_content_in_frozen_exploration_role(
    tmp_path: Path,
    attack: str,
) -> None:
    env = build_admission_fixture(tmp_path)
    gate_split_ref = next(
        split.manifest_ref
        for split in env.protocol.splits
        if split.partition is ProtocolPartition.GATE
    )
    if attack == "direct":
        attack_ref = gate_split_ref
    elif attack == "copied":
        attack_ref = ArtifactRef(
            sha256=gate_split_ref.sha256,
            size=gate_split_ref.size,
            media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
        )
    else:
        attack_ref = env.store.put_json(
            {"copied_gate_payload": env.store.get_json(gate_split_ref)},
            media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
        )
    legal = legal_feedback_content(env)
    forged = legal.model_copy(
        update={"view": legal.view.model_copy(update={"exploration_inputs_ref": attack_ref})}
    )

    with pytest.raises(ValueError, match="frozen role ref"):
        env.feedback_service.attest(forged)


def test_legal_feedback_uses_safe_metadata_and_frozen_optimizer_grants(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    content = legal_feedback_content(env)
    trusted_ref = env.feedback_service.attest(content)
    runtime = FixedFeedbackRuntime(trusted_ref)
    context = open_feedback_context(env, runtime=runtime)

    trusted, actual_ref = context.loop._call_feedback(
        run_ref=context.run_ref,
        run=context.run,
        round_index=0,
        champion_ref=env.seed_harness_ref,
        prior_gate=None,
        controller=context.controller,
        tail=context.tail,
    )
    policy = env.store.get_json(context.run.search_policy_ref, SearchPolicy)
    artifacts = context.loop._make_strategy_artifact_view(
        run_ref=context.run_ref,
        run=context.run,
        policy=policy,
        round_index=0,
        champion_ref=env.seed_harness_ref,
        trusted_feedback=trusted,
        feedback_ref=actual_ref,
        invocation="proposal",
        additional_read_refs=(),
    )

    assert artifacts is not None
    assert (
        artifacts.read_json(env.benchmark.safe_benchmark_metadata_ref)["redaction_schema"]
        == "exploration-only-no-partition-refs-v1"
    )
    with pytest.raises(AutomaticSearchLoopError, match="non-allowlisted"):
        artifacts.read_json(env.benchmark_ref)
    with pytest.raises(AutomaticSearchLoopError, match="non-allowlisted"):
        artifacts.read_json(env.benchmark.exploration_split_ref)


def test_feedback_signer_rejects_gate_ref_nested_in_diagnostic_graph(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    gate_split_ref = next(
        split.manifest_ref
        for split in env.protocol.splits
        if split.partition is ProtocolPartition.GATE
    )
    cluster = env.store.get_json(env.benchmark.diagnostic_evidence_ref, DiagnosticCluster)
    packet = env.store.get_json(cluster.evidence_packet_refs[0], EvidencePacket)
    nested_packet_ref = env.store.put_json(
        packet.model_copy(update={"grader_verdict_ref": gate_split_ref}),
        media_type=EVIDENCE_PACKET_MEDIA_TYPE,
    )
    nested_cluster_ref = env.store.put_json(
        cluster.model_copy(update={"evidence_packet_refs": (nested_packet_ref,)}),
        media_type=DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    )
    legal = legal_feedback_content(env, baseline_kind=BaselineKind.EVIDENCE_TARGETED)
    forged = legal.model_copy(
        update={
            "view": legal.view.model_copy(update={"diagnostic_evidence_ref": nested_cluster_ref})
        }
    )

    with pytest.raises(ValueError, match="diagnostic root"):
        env.feedback_service.attest(forged)


@pytest.mark.parametrize(
    "coordinate",
    ["search_run_ref", "experiment_ref", "round_index", "champion_harness_ref"],
)
def test_loop_rejects_validly_attested_foreign_feedback_coordinates(
    tmp_path: Path,
    coordinate: str,
) -> None:
    env = build_admission_fixture(tmp_path)
    legal = legal_feedback_content(env)
    replacements: dict[str, object] = {
        "search_run_ref": put_json(
            env.store,
            "foreign-run",
            media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
        ),
        "experiment_ref": put_json(
            env.store,
            "foreign-experiment",
            media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
        ),
        "round_index": 1,
        "champion_harness_ref": put_json(env.store, "foreign-champion"),
    }
    foreign = legal.model_copy(update={coordinate: replacements[coordinate]})
    trusted_ref = env.feedback_service.attest(foreign)
    context = open_feedback_context(env, runtime=FixedFeedbackRuntime(trusted_ref))

    with pytest.raises(AutomaticSearchLoopError, match="invalid feedback output"):
        context.loop._call_feedback(
            run_ref=context.run_ref,
            run=context.run,
            round_index=0,
            champion_ref=env.seed_harness_ref,
            prior_gate=None,
            controller=context.controller,
            tail=context.tail,
        )


def test_loop_rejects_validly_attested_foreign_binding_and_signer_rejects_split(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)
    foreign_metadata_ref = env.store.put_json(
        SafeBenchmarkMetadata(
            benchmark_fingerprint="foreign-benchmark-v1",
            exploration_task_ids=env.benchmark.exploration_task_ids,
        ),
        media_type=SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    foreign_binding = env.benchmark.model_copy(
        update={
            "benchmark_fingerprint": "foreign-benchmark-v1",
            "safe_benchmark_metadata_ref": foreign_metadata_ref,
        }
    )
    foreign_binding_ref = env.store.put_json(
        foreign_binding,
        media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    legal = legal_feedback_content(env)
    foreign_content = legal.model_copy(
        update={
            "benchmark_binding_ref": foreign_binding_ref,
            "view": legal.view.model_copy(update={"benchmark_metadata_ref": foreign_metadata_ref}),
        }
    )
    trusted_ref = env.feedback_service.attest(foreign_content)
    context = open_feedback_context(env, runtime=FixedFeedbackRuntime(trusted_ref))

    with pytest.raises(AutomaticSearchLoopError, match="invalid feedback output"):
        context.loop._call_feedback(
            run_ref=context.run_ref,
            run=context.run,
            round_index=0,
            champion_ref=env.seed_harness_ref,
            prior_gate=None,
            controller=context.controller,
            tail=context.tail,
        )

    alternate_split_ref = env.store.put_json({"partition": "foreign-exploration"})
    wrong_split = legal.model_copy(update={"exploration_split_ref": alternate_split_ref})
    with pytest.raises(ValueError, match="another exploration split"):
        env.feedback_service.attest(wrong_split)


@pytest.mark.parametrize("tamper", ["content", "tag"])
def test_feedback_hmac_tamper_fails_closed(tmp_path: Path, tamper: str) -> None:
    env = build_admission_fixture(tmp_path)
    trusted_ref = env.feedback_service.attest(legal_feedback_content(env))
    envelope = env.store.get_json(trusted_ref, TrustedStrategyFeedback)
    if tamper == "content":
        forged = envelope.model_copy(
            update={"content": envelope.content.model_copy(update={"round_index": 1})}
        )
    else:
        forged = envelope.model_copy(update={"authentication_tag": "0" * 64})
    forged_ref = env.store.put_json(
        forged,
        media_type=TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE,
    )

    with pytest.raises(AutomaticSearchLoopError, match="authentication failed"):
        env.feedback_service.verification_capability.verify(forged_ref)


def test_feedback_wrong_attestor_and_media_spoof_fail_closed(tmp_path: Path) -> None:
    env = build_admission_fixture(tmp_path)
    legal = legal_feedback_content(env)
    foreign_service = TrustedStrategyFeedbackService(env.store, secret=b"f" * 32)
    foreign_ref = foreign_service.attest(legal)
    with pytest.raises(AutomaticSearchLoopError, match="another attestor"):
        env.feedback_service.verification_capability.verify(foreign_ref)

    trusted_ref = env.feedback_service.attest(legal)
    spoofed_ref = ArtifactRef(
        sha256=trusted_ref.sha256,
        size=trusted_ref.size,
        media_type="application/json",
    )
    with pytest.raises(AutomaticSearchLoopError, match="wrong media type"):
        env.feedback_service.verification_capability.verify(spoofed_ref)


def test_feedback_verifier_requires_exact_non_subclassable_capability(
    tmp_path: Path,
) -> None:
    env = build_admission_fixture(tmp_path)

    class DuckVerifier:
        attestor_id = env.feedback_service.verification_capability.attestor_id

        def verify(self, _: ArtifactRef) -> TrustedStrategyFeedbackContent:
            return legal_feedback_content(env)

    with pytest.raises(TypeError, match="exact trusted capability"):
        AutomaticSearchLoop(
            env.store,
            runtime=object(),  # type: ignore[arg-type]
            lifecycle=object(),  # type: ignore[arg-type]
            objective_aggregate_verifier=env.objective_service.verification_capability,
            strategy_feedback_verifier=DuckVerifier(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class EvilFeedbackVerifier(StrategyFeedbackVerificationCapability):
            pass


def test_automatic_loop_rejects_resume_after_a_completed_round(tmp_path: Path) -> None:
    env = build_admission_fixture(tmp_path)
    controller = env.controllers[BaselineKind.RANDOM_VALID]
    tail = controller.freeze()
    tail = controller.start_search(previous_tail_ref=tail)
    tail = controller.open_round(previous_tail_ref=tail)
    tail = controller.record_evidence(
        previous_tail_ref=tail,
        evidence_packet_ref=put_json(env.store, "round-evidence"),
    )
    tail = controller.record_candidate_pool(
        previous_tail_ref=tail,
        candidate_pool_ref=put_json(env.store, "empty-pool"),
        candidate_refs=(),
        declined=True,
    )
    tail = controller.record_screen(
        previous_tail_ref=tail,
        screen_report_ref=put_json(env.store, "empty-screen"),
    )
    tail = controller.record_no_candidate(
        previous_tail_ref=tail,
        reason="no eligible frozen catalogue entry",
    )
    controller.complete_no_candidate_round(previous_tail_ref=tail)
    run_ref, _ = env.runs[BaselineKind.RANDOM_VALID]
    loop = AutomaticSearchLoop(
        env.store,
        runtime=object(),  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=env.objective_service.verification_capability,
        strategy_feedback_verifier=env.feedback_service.verification_capability,
    )

    with pytest.raises(AutomaticSearchLoopError, match="resume after completed rounds"):
        loop.run(
            search_run_ref=run_ref,
            controller=controller,
            expectation=SearchRunAdmissionExpectation(
                baseline_study_plan_ref=env.study_ref,
                experiment_ref=env.experiment_ref,
                baseline_kind=BaselineKind.RANDOM_VALID,
                search_run_seed=701,
            ),
        )


def test_receipt_backed_screen_replays_live_ledger_and_hmac_aggregate(
    tmp_path: Path,
) -> None:
    env = build_receipt_backed_screen(tmp_path)

    verified = env.loop._verify_screen_evaluation(
        run_ref=env.run_ref,
        run=env.run,
        round_index=0,
        champion_ref=env.screen_parent_harness_ref,
        proposal_ref=env.proposal_ref,
        screen=env.screen,
    )

    assert verified == env.evaluation
    assert verified.trusted_usage.cell_count == 8
    assert verified.trusted_usage.attempt_count == 8
    assert verified.trusted_usage.settled_attempts == 8
    assert verified.trusted_usage.charged_tokens == 24
    assert verified.schedule.repeat_seeds == (11, 12)


def test_screen_rechecks_ledger_after_objective_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_receipt_backed_screen(tmp_path)
    original_verify = ObjectiveAggregateVerificationCapability.verify

    def verify_then_advance(
        capability: ObjectiveAggregateVerificationCapability,
        aggregate_ref: ArtifactRef,
    ) -> TrustedObjectiveAggregateContent:
        content = original_verify(capability, aggregate_ref)
        reservation_ref = env.ledger.reserve(
            task_fingerprint="a" * 64,
            execution_fingerprint="b" * 64,
            request_sha256="c" * 64,
            token_ceiling=env.schedule.token_ceiling_per_attempt,
        )
        env.ledger.burn(reservation_ref, error_class="objective-verification-race")
        return content

    monkeypatch.setattr(ObjectiveAggregateVerificationCapability, "verify", verify_then_advance)

    with pytest.raises(AutomaticSearchLoopError, match="ledger changed during verification"):
        env.loop._verify_screen_evaluation(
            run_ref=env.run_ref,
            run=env.run,
            round_index=0,
            champion_ref=env.screen_parent_harness_ref,
            proposal_ref=env.proposal_ref,
            screen=env.screen,
        )


def test_screen_rejects_reconstructed_ledger_before_objective_verification(
    tmp_path: Path,
) -> None:
    env = build_receipt_backed_screen(tmp_path)
    reconstructed = AttemptLedger(
        env.store,
        ledger_id=env.ledger.ledger_id,
        budget=env.ledger.budget,
        tail_ref=env.ledger.tail_ref,
    )
    assert reconstructed.state().writer_epoch_id != env.ledger.state().writer_epoch_id
    unreachable_objective_ref = ArtifactRef(
        sha256="0" * 64,
        size=1,
        media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    )
    evaluation_ref = env.store.put_json(
        env.evaluation.model_copy(update={"objective_aggregate_ref": unreachable_objective_ref}),
        media_type=TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    )
    screen = env.screen.model_copy(update={"evaluation_ref": evaluation_ref})
    loop = AutomaticSearchLoop(
        env.store,
        runtime=LedgerRuntime(reconstructed),  # type: ignore[arg-type]
        lifecycle=object(),  # type: ignore[arg-type]
        objective_aggregate_verifier=env.objective_service.verification_capability,
        strategy_feedback_verifier=env.feedback_service.verification_capability,
    )

    with pytest.raises(ExecutionReceiptIntegrityError, match="writer epoch"):
        loop._verify_screen_evaluation(
            run_ref=env.run_ref,
            run=env.run,
            round_index=0,
            champion_ref=env.screen_parent_harness_ref,
            proposal_ref=env.proposal_ref,
            screen=screen,
        )


def test_screen_rejects_foreign_preflight_model_spec(tmp_path: Path) -> None:
    env = build_receipt_backed_screen(tmp_path)
    foreign_spec = env.preflight.model_spec.model_copy(
        update={"backend_fingerprint": "foreign-backend@sha256:fixed-v2"}
    )
    assert (
        foreign_spec.model_fingerprint,
        foreign_spec.inference_fingerprint,
        foreign_spec.runtime_fingerprint,
    ) == (
        env.preflight.model_spec.model_fingerprint,
        env.preflight.model_spec.inference_fingerprint,
        env.preflight.model_spec.runtime_fingerprint,
    )
    assert foreign_spec.fingerprint != env.preflight.model_spec.fingerprint
    foreign_preflight = SchedulePreflightCertificate.model_validate(
        env.preflight.model_copy(update={"model_spec": foreign_spec}),
        strict=True,
    )
    foreign_preflight_ref = env.store.put_json(
        foreign_preflight,
        media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    )
    foreign_evaluation = env.evaluation.model_copy(update={"preflight_ref": foreign_preflight_ref})
    foreign_evaluation_ref = env.store.put_json(
        foreign_evaluation,
        media_type=TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    )
    foreign_screen = env.screen.model_copy(update={"evaluation_ref": foreign_evaluation_ref})

    with pytest.raises(AutomaticSearchLoopError, match="frozen model boundary"):
        env.loop._verify_screen_evaluation(
            run_ref=env.run_ref,
            run=env.run,
            round_index=0,
            champion_ref=env.screen_parent_harness_ref,
            proposal_ref=env.proposal_ref,
            screen=foreign_screen,
        )


def test_screen_rejects_hmac_aggregate_tamper_and_unreceipted_ledger_tail(
    tmp_path: Path,
) -> None:
    env = build_receipt_backed_screen(tmp_path)
    envelope = env.store.get_json(env.aggregate_ref, TrustedObjectiveAggregate)
    forged_content = envelope.content.model_copy(update={"primary_score": 0.99})
    forged_aggregate_ref = env.store.put_json(
        envelope.model_copy(update={"content": forged_content}),
        media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    )
    forged_evaluation_ref = env.store.put_json(
        env.evaluation.model_copy(update={"objective_aggregate_ref": forged_aggregate_ref}),
        media_type=TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE,
    )
    forged_screen = env.screen.model_copy(update={"evaluation_ref": forged_evaluation_ref})

    with pytest.raises(AutomaticSearchLoopError, match="authentication failed"):
        env.loop._verify_screen_evaluation(
            run_ref=env.run_ref,
            run=env.run,
            round_index=0,
            champion_ref=env.screen_parent_harness_ref,
            proposal_ref=env.proposal_ref,
            screen=forged_screen,
        )

    reservation_ref = env.ledger.reserve(
        task_fingerprint="a" * 64,
        execution_fingerprint="b" * 64,
        request_sha256="c" * 64,
        token_ceiling=env.schedule.token_ceiling_per_attempt,
    )
    env.ledger.burn(reservation_ref, error_class="unreceipted-trailing-attempt")
    with pytest.raises(ExecutionReceiptIntegrityError, match="exactly cover every outcome"):
        env.loop._verify_screen_evaluation(
            run_ref=env.run_ref,
            run=env.run,
            round_index=0,
            champion_ref=env.screen_parent_harness_ref,
            proposal_ref=env.proposal_ref,
            screen=env.screen,
        )


def test_objective_aggregate_tamper_and_foreign_attestor_fail_closed(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    service = TrustedObjectiveAggregateService(store, secret=b"a" * 32)
    aggregate_ref = service.attest(objective_content(store))
    envelope = store.get_json(aggregate_ref, TrustedObjectiveAggregate)
    forged_content = envelope.content.model_copy(update={"primary_score": 0.99})
    forged_ref = store.put_json(
        envelope.model_copy(update={"content": forged_content}),
        media_type=aggregate_ref.media_type,
    )

    with pytest.raises(ObjectiveAggregateVerificationError, match="authentication failed"):
        service.verification_capability.verify(forged_ref)

    foreign = TrustedObjectiveAggregateService(store, secret=b"b" * 32)
    with pytest.raises(ObjectiveAggregateVerificationError, match="another attestor"):
        foreign.verification_capability.verify(aggregate_ref)


def test_strategy_artifact_view_has_no_grant_and_rejects_unauthorized_reads(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    allowed_ref = put_json(store, "allowed-evidence")
    secret_ref = put_json(store, "sealed-secret")
    run_ref = put_json(store, "run", media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
    view = StrategyArtifactView(
        store,
        search_run_ref=run_ref,
        baseline_kind=BaselineKind.PROMPT_ONLY,
        round_index=0,
        invocation="proposal",
        allowed_read_refs=(allowed_ref,),
        max_prompt_bytes=2_048,
    )

    assert not hasattr(view, "grant_read_refs")
    with pytest.raises(AutomaticSearchLoopError, match="non-allowlisted"):
        view.read_json(secret_ref)
    with pytest.raises(ValueError, match="must have been read"):
        view.put_hypothesis(hypothesis(allowed_ref))

    assert view.read_json(allowed_ref) == {"label": "allowed-evidence"}
    prompt_ref = view.put_prompt("Verify the answer before returning it.")
    hypothesis_ref = view.put_hypothesis(hypothesis(allowed_ref))
    access_log = view.access_log()

    assert prompt_ref in view.written_prompt_refs
    assert hypothesis_ref in view.written_hypothesis_refs
    assert tuple(access.operation for access in access_log.accesses) == (
        StrategyArtifactOperation.READ_JSON,
        StrategyArtifactOperation.WRITE_PROMPT,
        StrategyArtifactOperation.WRITE_HYPOTHESIS,
    )
    assert access_log.written_bytes == prompt_ref.size + hypothesis_ref.size


def test_diagnosis_view_cannot_write_and_hypotheses_are_byte_bounded(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    evidence_ref = put_json(store, "evidence")
    run_ref = put_json(store, "run", media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
    diagnosis_view = StrategyArtifactView(
        store,
        search_run_ref=run_ref,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
        round_index=0,
        invocation="diagnosis",
        allowed_read_refs=(evidence_ref,),
        max_prompt_bytes=128,
    )
    with pytest.raises(AutomaticSearchLoopError, match="cannot write prompts"):
        diagnosis_view.put_prompt("forbidden")

    proposal_view = StrategyArtifactView(
        store,
        search_run_ref=run_ref,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
        round_index=0,
        invocation="proposal",
        allowed_read_refs=(evidence_ref,),
        max_prompt_bytes=128,
    )
    proposal_view.read_json(evidence_ref)
    with pytest.raises(ValueError, match="artifact size limit"):
        proposal_view.put_hypothesis(hypothesis(evidence_ref, where="x" * 512))


def test_materialization_and_screen_reject_partial_candidate_shapes() -> None:
    run_ref = ArtifactRef(
        sha256="1" * 64,
        size=1,
        media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    )
    proposal_ref = ArtifactRef(
        sha256="2" * 64,
        size=1,
        media_type=PROMPT_PROPOSAL_MEDIA_TYPE,
    )
    candidate_ref = ArtifactRef(sha256="3" * 64, size=1, media_type="application/json")

    with pytest.raises(ValidationError, match="present together"):
        CandidateMaterialization(
            search_run_ref=run_ref,
            baseline_kind=BaselineKind.PROMPT_ONLY,
            round_index=0,
            proposal_ref=proposal_ref,
            candidate_ref=candidate_ref,
        )
    with pytest.raises(ValidationError, match="present together"):
        CandidateScreen(
            baseline_kind=BaselineKind.PROMPT_ONLY,
            round_index=0,
            proposal_ref=proposal_ref,
            candidate_ref=candidate_ref,
            status=CandidateScreenStatus.REJECTED,
            failure_codes=(CandidateScreenFailure.EVALUATION_INCOMPLETE,),
        )


def test_loop_result_derives_optimizer_call_count_from_exact_invocations(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    common_ref = put_json(store, "common")
    run_ref = put_json(store, "run", media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE)
    values = {
        "non_reportable_fixture": True,
        "search_run_ref": run_ref,
        "admission_report_ref": common_ref,
        "final_search_tail_ref": common_ref,
        "search_selection_closure_ref": common_ref,
        "final_experiment_tail_ref": common_ref,
        "final_champion_harness_ref": common_ref,
        "final_champion_candidate_ref": None,
        "completed_rounds": 1,
        "gate_nominations": 0,
        "diagnosis_count": 2,
        "diagnosis_invocation_count": 1,
        "proposal_count": 1,
        "proposal_invocation_count": 1,
        "screen_count": 1,
        "optimizer_model_call_count": 3,
        "trusted_screen_cell_count": 4,
        "trusted_screen_charged_tokens": 12,
        "strategy_artifact_written_bytes": 0,
        "strategy_access_log_refs": (),
        "archived_artifact_refs": (),
    }

    with pytest.raises(ValidationError, match="optimizer call count"):
        AutomaticSearchLoopResult(**values)

    checked = AutomaticSearchLoopResult(**{**values, "optimizer_model_call_count": 2})
    assert checked.optimizer_model_call_count == (
        checked.diagnosis_invocation_count + checked.proposal_invocation_count
    )
