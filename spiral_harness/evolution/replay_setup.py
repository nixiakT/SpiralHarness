"""Frozen artifacts for the non-reportable automatic-search replay study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    ExperimentManifest,
    MutationPolicy,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
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
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    SEARCH_POLICY_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SEARCH_STOPPING_POLICY_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    SearchRunManifest,
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
    MUTATION_HYPOTHESIS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    ExplorationTrajectoryIndex,
    SafeBenchmarkMetadata,
    SearchAnalysisPlan,
    SearchBenchmarkBinding,
    TrustedObjectiveAggregateService,
    TrustedStrategyFeedbackService,
)
from spiral_harness.evolution.seeds import derive_strategy_seed
from spiral_harness.evolution.strategies import (
    PromptMutationCatalogue,
    PromptMutationEntry,
    make_search_policy,
    make_search_stopping_policy,
    make_strategy_plugin_manifest,
)
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.execution.policy import CAPABILITY_POLICY_MEDIA_TYPE, CapabilityPolicy
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineKind,
    BaselineStudyPlan,
    FrozenMutationPolicy,
    FrozenRunContext,
    PairedEvaluationPlan,
    ResourceCeilings,
    plan_four_baselines,
)
from spiral_harness.experiments.study import StudyRunKey, expected_run_keys
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.artifacts import TrustedGateBatchService
from spiral_harness.verification.mechanism import TrustedMechanismEvidenceService
from spiral_harness.verification.models import GateConfig

GATE_CONFIG_MEDIA_TYPE = "application/vnd.spiral-harness.gate-config+json"
PROMPT_MEDIA_TYPE = "text/plain"
REPLAY_BENCHMARK_FINGERPRINT = "non-reportable-replay-benchmark-v1"
SEARCH_RUN_SEEDS = (701, 702)
REPEAT_SEEDS = (11, 12)
FAMILY_ALPHA = 0.05
MAX_GATE_QUERIES = 1
GATE_CONFIDENCE = 1.0 - (FAMILY_ALPHA / MAX_GATE_QUERIES)
PROMPT_COMPONENT_NAME = "system"


@dataclass(frozen=True, slots=True)
class ReplayFeedbackArtifacts:
    safe_benchmark_metadata_ref: ArtifactRef
    exploration_inputs_ref: ArtifactRef
    exploration_aggregates_ref: ArtifactRef
    exploration_item_feedback_ref: ArtifactRef
    exploration_trajectories_ref: ArtifactRef
    diagnostic_cluster_ref: ArtifactRef
    diagnostic_closure_refs: tuple[ArtifactRef, ...]
    failure_signature_ref: ArtifactRef
    evidence_packet_ref: ArtifactRef


@dataclass(frozen=True, slots=True)
class FrozenReplayFixture:
    store: ArtifactStore
    model_spec: FrozenModelSpec
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    experiment: ExperimentManifest
    experiment_ref: ArtifactRef
    seed_harness_ref: ArtifactRef
    seed_prompt_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    benchmark_binding_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    objective_service: TrustedObjectiveAggregateService
    feedback_service: TrustedStrategyFeedbackService
    gate_batch_service: TrustedGateBatchService
    mechanism_evidence_service: TrustedMechanismEvidenceService
    feedback: ReplayFeedbackArtifacts
    policy_refs: dict[BaselineKind, ArtifactRef]
    stopping_refs: dict[BaselineKind, ArtifactRef]
    plugin_refs: dict[BaselineKind, ArtifactRef]
    implementation_refs: dict[BaselineKind, ArtifactRef]
    catalogue_ref: ArtifactRef


def put_json(
    store: ArtifactStore,
    value: object,
    *,
    media_type: str = "application/json",
) -> ArtifactRef:
    return store.put_json(value, media_type=media_type)


def fixed_replay_model_spec() -> FrozenModelSpec:
    """Return the fixed model contract shared by the replay study fixtures."""

    return FrozenModelSpec(
        backend="deterministic-replay",
        backend_fingerprint="replay-study-backend@sha256:non-reportable-v1",
        model="fixture/replay-study-model",
        revision="snapshot-2026-08-12",
        tokenizer="fixture/replay-study-tokenizer",
        tokenizer_revision="snapshot-2026-08-12",
        runtime="python-3.12/replay-study-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=32,
            timeout_seconds=10.0,
        ),
    )


def _build_feedback_artifacts(
    store: ArtifactStore,
    *,
    seed_harness_ref: ArtifactRef,
    seed_prompt_ref: ArtifactRef,
    benchmark_fingerprint: str,
) -> ReplayFeedbackArtifacts:
    safe_benchmark_metadata_ref = put_json(
        store,
        SafeBenchmarkMetadata(
            benchmark_fingerprint=benchmark_fingerprint,
            exploration_task_ids=("exploration-1",),
        ),
        media_type=SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    exploration_inputs_ref = put_json(
        store,
        {"task_ids": ["exploration-1"], "partition": "exploration"},
        media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
    )
    exploration_aggregates_ref = put_json(
        store,
        {"primary_score": 0.5, "failure_count": 1},
        media_type=EXPLORATION_AGGREGATES_MEDIA_TYPE,
    )
    exploration_item_feedback_ref = put_json(
        store,
        {"task_id": "exploration-1", "outcome": "incorrect"},
        media_type=EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    )
    run_started_payload_ref = put_json(
        store,
        {"status": "started", "task_id": "exploration-1"},
    )
    error_payload_ref = put_json(
        store,
        {"error_class": "missing-independent-verification"},
    )
    trajectory_ref = put_json(
        store,
        Trajectory(
            run_id="fixture-trajectory-1",
            task_id="exploration-1",
            seed=701,
            harness_ref=seed_harness_ref,
            execution_fingerprint="fixture-execution@sha256:non-reportable-v1",
            events=(
                TrajectoryEvent(
                    sequence=0,
                    elapsed_ms=0.0,
                    kind=TrajectoryEventKind.RUN_STARTED,
                    payload_ref=run_started_payload_ref,
                ),
                TrajectoryEvent(
                    sequence=1,
                    elapsed_ms=1.0,
                    kind=TrajectoryEventKind.ERROR,
                    payload_ref=error_payload_ref,
                    causal_parent_sequences=(0,),
                ),
            ),
        ),
        media_type=DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE,
    )
    exploration_trajectories_ref = put_json(
        store,
        ExplorationTrajectoryIndex(
            exploration_task_ids=("exploration-1",),
            trajectory_refs=(trajectory_ref,),
        ),
        media_type=EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    )
    failure_signature_ref = put_json(
        store,
        FailureSignature(
            failure_id="missing-verification-v1",
            failure_class="verification-omission",
            summary="the fixture output was returned without an independent check",
            symptom_codes=("no-verification-event",),
            affected_component_refs=(seed_prompt_ref,),
            affected_task_ids=("exploration-1",),
            slice_tags=("fixture",),
        ),
        media_type=FAILURE_SIGNATURE_MEDIA_TYPE,
    )
    evidence_packet_ref = put_json(
        store,
        EvidencePacket(
            source_spans=(
                EvidenceSpanRef(
                    trajectory_ref=trajectory_ref,
                    start_sequence=1,
                    end_sequence=1,
                    claim="the recorded fixture trajectory terminates without verification",
                    expected_event_kinds=(TrajectoryEventKind.ERROR,),
                ),
            ),
            failure_signature_ref=failure_signature_ref,
        ),
        media_type=EVIDENCE_PACKET_MEDIA_TYPE,
    )
    diagnostic_cluster_ref = put_json(
        store,
        DiagnosticCluster(
            cluster_id="fixture-cluster-v1",
            label="missing independent verification",
            failure_signature_refs=(failure_signature_ref,),
            representative_signature_ref=failure_signature_ref,
            evidence_packet_refs=(evidence_packet_ref,),
            task_ids=("exploration-1",),
            slice_tags=("fixture",),
        ),
        media_type=DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    )
    diagnostic_closure_refs = (
        diagnostic_cluster_ref,
        failure_signature_ref,
        evidence_packet_ref,
        trajectory_ref,
        seed_harness_ref,
        seed_prompt_ref,
        run_started_payload_ref,
        error_payload_ref,
    )
    return ReplayFeedbackArtifacts(
        safe_benchmark_metadata_ref=safe_benchmark_metadata_ref,
        exploration_inputs_ref=exploration_inputs_ref,
        exploration_aggregates_ref=exploration_aggregates_ref,
        exploration_item_feedback_ref=exploration_item_feedback_ref,
        exploration_trajectories_ref=exploration_trajectories_ref,
        diagnostic_cluster_ref=diagnostic_cluster_ref,
        diagnostic_closure_refs=diagnostic_closure_refs,
        failure_signature_ref=failure_signature_ref,
        evidence_packet_ref=evidence_packet_ref,
    )


def build_frozen_replay_fixture(root: str | Path) -> FrozenReplayFixture:
    """Freeze every artifact shared by the eight replay-study cells."""

    store = ArtifactStore(root)
    model_spec = fixed_replay_model_spec()
    gate_batch_service = TrustedGateBatchService()
    mechanism_evidence_service = TrustedMechanismEvidenceService()
    objective_service = TrustedObjectiveAggregateService(
        store,
        secret=b"spiral-harness/replay-study/objective-v1",
    )
    feedback_service = TrustedStrategyFeedbackService(
        store,
        secret=b"spiral-harness/replay-study/feedback-v1",
    )
    split_refs = {
        partition: put_json(
            store,
            {
                "fixture": "non-reportable-replay-study-v1",
                "partition": partition.value,
                "task_ids": [f"{partition.value}-1"],
            },
        )
        for partition in ProtocolPartition
    }
    splits = tuple(
        ProtocolSplit(partition=partition, manifest_ref=split_refs[partition])
        for partition in ProtocolPartition
    )
    gate_config_ref = put_json(
        store,
        GateConfig(
            version="non-reportable-replay-study-v1",
            min_tasks=1,
            min_effect=0.0,
            confidence_level=GATE_CONFIDENCE,
            bootstrap_samples=1_000,
            bootstrap_seed=17,
            expected_task_ids=("gate-1",),
            expected_seeds=REPEAT_SEEDS,
            required_mechanism_checks=(
                "patch_validity",
                "activation",
                "adherence",
                "behavior",
            ),
            max_single_task_regression=0.0,
            max_regression_rate=0.0,
            max_tokens_ratio=1.0,
            max_latency_ratio=1.0,
            max_tool_calls_ratio=1.0,
        ),
        media_type=GATE_CONFIG_MEDIA_TYPE,
    )
    capability_policy_ref = put_json(
        store,
        CapabilityPolicy(),
        media_type=CAPABILITY_POLICY_MEDIA_TYPE,
    )
    seed_prompt_ref = store.put_bytes(
        b"Answer the task directly.",
        media_type=PROMPT_MEDIA_TYPE,
    )
    budget = BudgetPolicy(
        max_tokens=100_000,
        max_tool_calls=0,
        max_wall_time_seconds=600.0,
        max_cost_usd=0.0,
        max_evaluations=100,
    )
    seed_harness = HarnessManifest(
        model_fingerprint=model_spec.model_fingerprint,
        runtime_fingerprint=model_spec.runtime_fingerprint,
        trusted_plane_version="fixture-trusted-plane-v1",
        components=(
            HarnessComponentRef(
                name=PROMPT_COMPONENT_NAME,
                kind=ComponentKind.PROMPT,
                artifact=seed_prompt_ref,
            ),
        ),
        budget=budget,
    )
    seed_harness_ref = put_json(
        store,
        seed_harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    protocol = ProtocolManifest(
        benchmark_fingerprint=REPLAY_BENCHMARK_FINGERPRINT,
        splits=splits,
        model_fingerprint=seed_harness.model_fingerprint,
        inference_fingerprint=model_spec.inference_fingerprint,
        runtime_fingerprint=seed_harness.runtime_fingerprint,
        model_spec_fingerprint=model_spec.fingerprint,
        sandbox_fingerprint="fixture-logical-isolation:no-security-claim",
        capability_policy_ref=capability_policy_ref,
        grader_fingerprint=REPLAY_BENCHMARK_FINGERPRINT,
        gate_batch_attestor_id=gate_batch_service.attestor_id,
        mechanism_evidence_attestor_id=mechanism_evidence_service.attestor_id,
        gate_config_ref=gate_config_ref,
        trusted_plane_version=seed_harness.trusted_plane_version,
        budget=budget,
    )
    protocol_ref = put_json(
        store,
        protocol,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
    )
    mutation_policy = MutationPolicy(
        allowed_kinds=(ComponentKind.PROMPT,),
        allowed_component_names=(PROMPT_COMPONENT_NAME,),
        allowed_media_types=(PROMPT_MEDIA_TYPE,),
        max_artifact_size_bytes=4_096,
    )
    experiment = ExperimentManifest(
        protocol_ref=protocol_ref,
        seed_harness_ref=seed_harness_ref,
        mutation_policy=mutation_policy,
        objective="benchmark-score",
        baselines=tuple(kind.value for kind in BaselineKind),
        stopping=("one-round-or-no-candidate",),
        search_budget=budget,
    )
    experiment_ref = put_json(
        store,
        experiment,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    feedback = _build_feedback_artifacts(
        store,
        seed_harness_ref=seed_harness_ref,
        seed_prompt_ref=seed_prompt_ref,
        benchmark_fingerprint=protocol.benchmark_fingerprint,
    )
    benchmark_binding_ref = put_json(
        store,
        SearchBenchmarkBinding(
            benchmark_fingerprint=protocol.benchmark_fingerprint,
            objective_aggregate_attestor_id=(objective_service.verification_capability.attestor_id),
            strategy_feedback_attestor_id=(feedback_service.verification_capability.attestor_id),
            protocol_splits=protocol.splits,
            exploration_task_ids=("exploration-1",),
            safe_benchmark_metadata_ref=feedback.safe_benchmark_metadata_ref,
            exploration_inputs_ref=feedback.exploration_inputs_ref,
            exploration_aggregates_ref=feedback.exploration_aggregates_ref,
            exploration_item_feedback_ref=feedback.exploration_item_feedback_ref,
            exploration_trajectories_ref=feedback.exploration_trajectories_ref,
            diagnostic_evidence_ref=feedback.diagnostic_cluster_ref,
            diagnostic_closure_refs=feedback.diagnostic_closure_refs,
        ),
        media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    analysis_plan_ref = put_json(
        store,
        SearchAnalysisPlan(
            family_alpha=FAMILY_ALPHA,
            max_gate_queries=MAX_GATE_QUERIES,
            gate_confidence_level=GATE_CONFIDENCE,
            search_run_seeds=SEARCH_RUN_SEEDS,
            repeat_seeds=REPEAT_SEEDS,
        ),
        media_type=SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
    )
    frozen_mutation_policy = FrozenMutationPolicy(
        grammar_version="atomic-prompt-replace-v1",
        allowed_component_kinds=(ComponentKind.PROMPT,),
        max_artifact_size_bytes=mutation_policy.max_artifact_size_bytes or 4_096,
    )
    evaluation = PairedEvaluationPlan(
        search_run_seeds=SEARCH_RUN_SEEDS,
        repeat_seeds=REPEAT_SEEDS,
    )
    baseline_plan = plan_four_baselines(
        context=FrozenRunContext(
            benchmark_ref=benchmark_binding_ref,
            model_fingerprint=protocol.model_fingerprint,
            inference_fingerprint=protocol.inference_fingerprint,
            runtime_fingerprint=protocol.runtime_fingerprint,
            seed_harness_ref=seed_harness_ref,
            mutation_policy=frozen_mutation_policy,
            proposal_random_seed=991,
        ),
        evaluation=evaluation,
        ceilings=ResourceCeilings(
            max_evaluations=100,
            max_feedback_queries=2,
            max_proposals=2,
            max_optimizer_model_calls=4,
            max_tokens=100_000,
            max_wall_time_seconds=600.0,
            max_cost_usd=0.0,
        ),
    )
    baseline_study_plan_ref = put_json(
        store,
        baseline_plan,
        media_type=BASELINE_STUDY_PLAN_MEDIA_TYPE,
    )
    policy_refs: dict[BaselineKind, ArtifactRef] = {}
    stopping_refs: dict[BaselineKind, ArtifactRef] = {}
    plugin_refs: dict[BaselineKind, ArtifactRef] = {}
    implementation_refs: dict[BaselineKind, ArtifactRef] = {}
    for kind in BaselineKind:
        is_static = kind is BaselineKind.STATIC
        policy = make_search_policy(
            baseline_kind=kind,
            mutation_policy=frozen_mutation_policy,
            max_rounds=0 if is_static else 1,
            max_gate_queries=0 if is_static else MAX_GATE_QUERIES,
            patience_rounds=0 if is_static else 1,
            max_consecutive_declines=0 if is_static else 1,
            max_diagnoses=1 if kind is BaselineKind.EVIDENCE_TARGETED else 0,
            max_proposals=0 if is_static else 1,
            max_screens=0 if is_static else 1,
            max_proposals_per_round=0 if is_static else 1,
            max_candidates_screened_per_round=0 if is_static else 1,
            family_alpha=FAMILY_ALPHA,
        )
        policy_refs[kind] = put_json(
            store,
            policy,
            media_type=SEARCH_POLICY_MEDIA_TYPE,
        )
        stopping_refs[kind] = put_json(
            store,
            make_search_stopping_policy(policy=policy),
            media_type=SEARCH_STOPPING_POLICY_MEDIA_TYPE,
        )
        implementation_refs[kind] = store.put_bytes(
            f"non-reportable replay strategy: {kind.value}\n".encode(),
            media_type="text/x-python",
        )
        plugin_refs[kind] = put_json(
            store,
            make_strategy_plugin_manifest(
                plugin_id=f"replay-{kind.value}",
                plugin_version="1.0.0",
                implementation_ref=implementation_refs[kind],
                baseline_kind=kind,
            ),
            media_type=STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
        )

    random_hypothesis = MutationHypothesis(
        evidence_refs=(feedback.exploration_aggregates_ref,),
        where="the system prompt",
        why="the finite fixture catalogue adds an explicit verification instruction",
        expected_activation="the replacement prompt is active",
        expected_adherence="answers are checked before return",
        expected_behavior="fixture verification omissions decrease",
        expected_benefit="benchmark-score reliability improves",
        protected_slices=("already-correct",),
        falsifier="candidate traces omit verification",
        negative_control="unrelated formatting remains unchanged",
        risks=("additional latency",),
    )
    random_hypothesis_ref = put_json(
        store,
        random_hypothesis,
        media_type=MUTATION_HYPOTHESIS_MEDIA_TYPE,
    )
    catalogue_ref = put_json(
        store,
        PromptMutationCatalogue(
            catalogue_id="replay-finite-prompt-catalogue-v1",
            grammar_version=frozen_mutation_policy.grammar_version,
            parent_harness_ref=seed_harness_ref,
            entries=(
                PromptMutationEntry(
                    entry_id="verify-before-answer",
                    target_component_name=PROMPT_COMPONENT_NAME,
                    expected_before_prompt_ref=seed_prompt_ref,
                    after_prompt_ref=store.put_bytes(
                        b"Answer carefully and verify the result.",
                        media_type=PROMPT_MEDIA_TYPE,
                    ),
                    hypothesis_ref=random_hypothesis_ref,
                    mechanism_family="explicit-self-verification",
                ),
            ),
        ),
        media_type=PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    )
    return FrozenReplayFixture(
        store=store,
        model_spec=model_spec,
        protocol=protocol,
        protocol_ref=protocol_ref,
        experiment=experiment,
        experiment_ref=experiment_ref,
        seed_harness_ref=seed_harness_ref,
        seed_prompt_ref=seed_prompt_ref,
        gate_config_ref=gate_config_ref,
        benchmark_binding_ref=benchmark_binding_ref,
        analysis_plan_ref=analysis_plan_ref,
        baseline_study_plan_ref=baseline_study_plan_ref,
        objective_service=objective_service,
        feedback_service=feedback_service,
        gate_batch_service=gate_batch_service,
        mechanism_evidence_service=mechanism_evidence_service,
        feedback=feedback,
        policy_refs=policy_refs,
        stopping_refs=stopping_refs,
        plugin_refs=plugin_refs,
        implementation_refs=implementation_refs,
        catalogue_ref=catalogue_ref,
    )


def freeze_replay_search_runs(
    fixture: FrozenReplayFixture,
) -> tuple[tuple[StudyRunKey, SearchRunManifest, ArtifactRef], ...]:
    """Freeze the exact eight run manifests before study collection opens."""

    frozen_plan = fixture.store.get_json(
        fixture.baseline_study_plan_ref,
        BaselineStudyPlan,
    )
    runs: list[tuple[StudyRunKey, SearchRunManifest, ArtifactRef]] = []
    for key in expected_run_keys(frozen_plan):
        kind = key.baseline_kind
        search_run = SearchRunManifest(
            baseline_study_plan_ref=fixture.baseline_study_plan_ref,
            experiment_ref=fixture.experiment_ref,
            search_policy_ref=fixture.policy_refs[kind],
            search_policy_fingerprint=fixture.policy_refs[kind].sha256,
            strategy_plugin_ref=fixture.plugin_refs[kind],
            strategy_plugin_fingerprint=fixture.plugin_refs[kind].sha256,
            analysis_plan_ref=fixture.analysis_plan_ref,
            stopping_policy_ref=fixture.stopping_refs[kind],
            stopping_policy_fingerprint=fixture.stopping_refs[kind].sha256,
            seed_harness_ref=fixture.seed_harness_ref,
            baseline_kind=kind,
            baseline_plan_fingerprint=fixture.baseline_study_plan_ref.sha256,
            proposal_master_seed=991,
            search_run_seed=key.search_run_seed,
            repeat_seeds=REPEAT_SEEDS,
            strategy_seed=derive_strategy_seed(
                proposal_master_seed=991,
                search_run_seed=key.search_run_seed,
                baseline_kind=kind,
            ),
            prompt_mutation_catalogue_ref=(
                fixture.catalogue_ref if kind is BaselineKind.RANDOM_VALID else None
            ),
        )
        search_run_ref = put_json(
            fixture.store,
            search_run,
            media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
        )
        runs.append((key, search_run, search_run_ref))
    return tuple(runs)


__all__ = [
    "PROMPT_COMPONENT_NAME",
    "REPEAT_SEEDS",
    "REPLAY_BENCHMARK_FINGERPRINT",
    "SEARCH_RUN_SEEDS",
    "FrozenReplayFixture",
    "ReplayFeedbackArtifacts",
    "build_frozen_replay_fixture",
    "fixed_replay_model_spec",
    "freeze_replay_search_runs",
    "put_json",
]
