"""One deterministic M0 vertical slice over a synthetic controlled fault.
The flow exists to test evidence plumbing and promotion semantics.  It is not
a real benchmark, a learned optimizer, or evidence of general agent gains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spiral_harness.benchmark.controlled_fixture import (
    CANDIDATE_PROMPT,
    EXPLORATION_SEED,
    EXPLORATION_TASKS,
    FIXTURE_KIND,
    FIXTURE_SEED,
    GATE_TASKS,
    NORMALIZATION_SLICE,
    PROTECTED_CANONICAL_SLICE,
    SEED_PROMPT,
    BenchmarkTask,
    DeterministicExecution,
    DeterministicExecutor,
)
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    MutationPolicy,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.execution.policy import CAPABILITY_POLICY_MEDIA_TYPE, CapabilityPolicy
from spiral_harness.experiments.admission import (
    ADMISSION_REPORT_MEDIA_TYPE,
    CandidateAdmissionService,
)
from spiral_harness.experiments.controller import ExperimentController
from spiral_harness.experiments.decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    GateEvaluationManifest,
    TerminalDecisionService,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateTrialArm,
    TrustedGateBatchService,
)
from spiral_harness.verification.gate import PromotionGate
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import (
    Decision,
    GateConfig,
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    TrialObservation,
)

SYNTHETIC_DISCLAIMER = (
    "Synthetic controlled-fault fixture only; scores are pipeline checks, "
    "not real benchmark results. Exploration and gate rosters are logically "
    "disjoint but are not securely sealed. Gate batches use a process-local "
    "HMAC capability, not OS isolation."
)
_PROMPT_MEDIA_TYPE = "text/plain; charset=utf-8"


class _FrozenArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ControlledFaultEvidence(_FrozenArtifactModel):
    """Baseline diagnosis used to justify the one-component prompt mutation."""

    fixture_kind: Literal["synthetic-controlled-fault-v1"] = FIXTURE_KIND
    disclaimer: str = SYNTHETIC_DISCLAIMER
    exploration_suite_ref: ArtifactRef
    seed_prompt_ref: ArtifactRef
    seed_manifest_ref: ArtifactRef
    exploration_trials_ref: ArtifactRef
    exploration_execution_ref: ArtifactRef
    failed_normalization_task_ids: tuple[str, ...]
    passing_protected_task_ids: tuple[str, ...]
    root_cause: str


class ControlledDemoRefs(_FrozenArtifactModel):
    """Content-addressed handles for every material demo output."""

    exploration_suite_ref: ArtifactRef
    gate_suite_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    capability_policy_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    protocol_manifest_ref: ArtifactRef
    experiment_manifest_ref: ArtifactRef
    seed_prompt_ref: ArtifactRef
    candidate_prompt_ref: ArtifactRef
    seed_manifest_ref: ArtifactRef
    candidate_manifest_ref: ArtifactRef
    candidate_record_ref: ArtifactRef
    exploration_trials_ref: ArtifactRef
    exploration_execution_ref: ArtifactRef
    candidate_exploration_trials_ref: ArtifactRef
    candidate_exploration_execution_ref: ArtifactRef
    seed_gate_trials_ref: ArtifactRef
    candidate_gate_trials_ref: ArtifactRef
    seed_gate_execution_ref: ArtifactRef
    candidate_gate_execution_ref: ArtifactRef
    fault_evidence_ref: ArtifactRef
    candidate_mutation_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef
    seed_gate_batch_ref: ArtifactRef
    candidate_gate_batch_ref: ArtifactRef
    gate_evaluation_ref: ArtifactRef
    gate_decision_ref: ArtifactRef
    terminal_decision_report_ref: ArtifactRef
    experiment_usage_claim_ref: ArtifactRef
    experiment_usage_tail_ref: ArtifactRef
    terminal_authorization_ref: ArtifactRef
    candidate_journal_tail_ref: ArtifactRef
    experiment_journal_tail_ref: ArtifactRef

    @property
    def benchmark_suite_ref(self) -> ArtifactRef:
        """The score-bearing gate suite; exploration is separately named."""

        return self.gate_suite_ref

    @property
    def seed_trials_ref(self) -> ArtifactRef:
        return self.seed_gate_trials_ref

    @property
    def candidate_trials_ref(self) -> ArtifactRef:
        return self.candidate_gate_trials_ref

    @property
    def seed_execution_ref(self) -> ArtifactRef:
        return self.seed_gate_execution_ref

    @property
    def candidate_execution_ref(self) -> ArtifactRef:
        return self.candidate_gate_execution_ref


class ControlledDemoResult(_FrozenArtifactModel):
    """In-memory result plus durable references returned by the demo."""

    fixture_kind: Literal["synthetic-controlled-fault-v1"] = FIXTURE_KIND
    disclaimer: str = SYNTHETIC_DISCLAIMER
    run_manifest_ref: ArtifactRef
    refs: ControlledDemoRefs
    decision: GateDecision


def _put_models(
    store: ArtifactStore,
    models: tuple[BaseModel, ...],
    *,
    media_type: str,
) -> ArtifactRef:
    return store.put_json(
        [model.model_dump(mode="json", by_alias=False, exclude_none=False) for model in models],
        media_type=media_type,
    )


def _gate_config() -> GateConfig:
    """Freeze all evaluation expectations before either harness arm runs."""

    return GateConfig(
        version="synthetic-controlled-fault-v1",
        min_tasks=len(GATE_TASKS),
        min_effect=0.50,
        confidence_level=0.95,
        bootstrap_samples=5_000,
        bootstrap_seed=FIXTURE_SEED,
        require_complete_pairs=True,
        expected_task_ids=tuple(task.task_id for task in GATE_TASKS),
        expected_seeds=(FIXTURE_SEED,),
        required_mechanism_checks=(
            "patch_validity",
            "activation",
            "adherence",
            "behavior",
        ),
        protected_slice_floors={PROTECTED_CANONICAL_SLICE: -0.01},
        min_slice_tasks=4,
        max_single_task_regression=0.0,
        max_regression_rate=0.0,
        regression_tolerance=0.0,
        max_tokens_ratio=1.0,
        max_latency_ratio=1.0,
        max_tool_calls_ratio=1.0,
    )


def _manifest(
    prompt_component: HarnessComponentRef,
    *,
    parent: ArtifactRef | None = None,
) -> HarnessManifest:
    executor = DeterministicExecutor()
    return HarnessManifest(
        model_fingerprint="synthetic-deterministic-executor:no-model-weights",
        runtime_fingerprint=executor.version,
        trusted_plane_version="promotion-gate:m0",
        parent=parent,
        components=(prompt_component,),
        budget=BudgetPolicy(
            max_tokens=executor.tokens_per_trial * len(GATE_TASKS),
            max_tool_calls=0,
            max_wall_time_seconds=(executor.latency_ms_per_trial * len(GATE_TASKS)) / 1_000,
            max_cost_usd=0.0,
            max_evaluations=len(GATE_TASKS),
        ),
    )


def _observations(
    executions: tuple[DeterministicExecution, ...],
) -> tuple[TrialObservation, ...]:
    return tuple(execution.observation for execution in executions)


def _mechanism_evidence(
    *,
    mutation: CandidateMutation,
    mutation_ref: ArtifactRef,
    seed_manifest: HarnessManifest,
    seed_manifest_ref: ArtifactRef,
    candidate_manifest: HarnessManifest,
    candidate_manifest_ref: ArtifactRef,
    candidate_prompt_ref: ArtifactRef,
    parent_probe_trials: tuple[TrialObservation, ...],
    parent_probe_trials_ref: ArtifactRef,
    candidate_probe_trials: tuple[TrialObservation, ...],
    candidate_probe_trials_ref: ArtifactRef,
    candidate_executions: tuple[DeterministicExecution, ...],
    candidate_execution_ref: ArtifactRef,
    exploration_tasks: tuple[BenchmarkTask, ...],
) -> MechanismEvidence:
    tasks_by_id = {task.task_id: task for task in exploration_tasks}
    parent_by_id = {trial.task_id: trial for trial in parent_probe_trials}
    candidate_by_id = {trial.task_id: trial for trial in candidate_probe_trials}

    patch_validity = (
        len(seed_manifest.components) == 1
        and len(candidate_manifest.components) == 1
        and candidate_manifest.parent == seed_manifest_ref
        and candidate_manifest.model_fingerprint == seed_manifest.model_fingerprint
        and candidate_manifest.runtime_fingerprint == seed_manifest.runtime_fingerprint
        and candidate_manifest.trusted_plane_version == seed_manifest.trusted_plane_version
        and candidate_manifest.budget == seed_manifest.budget
        and mutation.before == seed_manifest.components[0]
        and mutation.after == candidate_manifest.components[0]
        and mutation.target_component == "match-policy"
    )
    activation = all(
        execution.prompt_sha256 == candidate_prompt_ref.sha256
        and execution.policy == "strip-and-casefold"
        for execution in candidate_executions
    )
    adherence = all(
        execution.compared_reference
        == tasks_by_id[execution.observation.task_id].reference_text.strip().casefold()
        and execution.compared_observed
        == tasks_by_id[execution.observation.task_id].observed_text.strip().casefold()
        for execution in candidate_executions
    )
    improved_task_ids = tuple(
        task.task_id
        for task in EXPLORATION_TASKS
        if candidate_by_id[task.task_id].score > parent_by_id[task.task_id].score
    )
    protected_task_ids = tuple(
        task.task_id for task in EXPLORATION_TASKS if PROTECTED_CANONICAL_SLICE in task.slice_tags
    )
    protected_unchanged = all(
        candidate_by_id[task_id].score == parent_by_id[task_id].score
        for task_id in protected_task_ids
    )
    expected_probe_improvements = sum(
        NORMALIZATION_SLICE in task.slice_tags for task in EXPLORATION_TASKS
    )
    behavior = len(improved_task_ids) == expected_probe_improvements and protected_unchanged

    return MechanismEvidence(
        candidate_harness_id=candidate_manifest_ref.sha256,
        checks=(
            MechanismCheck(
                name="patch_validity",
                passed=patch_validity,
                details="one prompt component changed and candidate lineage points to seed",
                evidence_refs=(
                    mutation_ref.sha256,
                    seed_manifest_ref.sha256,
                    candidate_manifest_ref.sha256,
                ),
            ),
            MechanismCheck(
                name="activation",
                passed=activation,
                details="every candidate trace activated the repaired prompt policy",
                evidence_refs=(candidate_execution_ref.sha256,),
            ),
            MechanismCheck(
                name="adherence",
                passed=adherence,
                details="every candidate trace stripped and casefolded both operands",
                evidence_refs=(candidate_execution_ref.sha256,),
            ),
            MechanismCheck(
                name="behavior",
                passed=behavior,
                details=(
                    f"{len(improved_task_ids)} exploration probes improved; "
                    f"{len(protected_task_ids)} protected canonical tasks were unchanged"
                ),
                evidence_refs=(
                    parent_probe_trials_ref.sha256,
                    candidate_probe_trials_ref.sha256,
                ),
            ),
        ),
    )


def run_controlled_demo(
    root: str | Path,
    *,
    gate_batch_service: TrustedGateBatchService | None = None,
    mechanism_evidence_service: TrustedMechanismEvidenceService | None = None,
) -> ControlledDemoResult:
    """Run and durably record the deterministic synthetic M0 vertical slice."""

    trusted_gate_batches = gate_batch_service or TrustedGateBatchService()
    trusted_mechanism_evidence = mechanism_evidence_service or TrustedMechanismEvidenceService()
    store = ArtifactStore(root)
    seed_prompt_ref = store.put_bytes(SEED_PROMPT.encode(), media_type=_PROMPT_MEDIA_TYPE)
    exploration_suite_ref = _put_models(
        store,
        EXPLORATION_TASKS,
        media_type="application/vnd.spiral-harness.synthetic-exploration-tasks+json",
    )
    gate_suite_ref = _put_models(
        store,
        GATE_TASKS,
        media_type="application/vnd.spiral-harness.synthetic-gate-tasks+json",
    )
    # The gate policy is persisted before any observations are produced.
    gate_config = _gate_config()
    gate_config_ref = store.put_json(
        gate_config,
        media_type="application/vnd.spiral-harness.gate-config+json",
    )
    capability_policy_ref = store.put_json(
        CapabilityPolicy(),
        media_type=CAPABILITY_POLICY_MEDIA_TYPE,
    )

    seed_component = HarnessComponentRef(
        name="match-policy",
        kind=ComponentKind.PROMPT,
        artifact=seed_prompt_ref,
    )
    seed_manifest = _manifest(seed_component)
    seed_manifest_ref = store.put_json(seed_manifest, media_type=HARNESS_MANIFEST_MEDIA_TYPE)

    executor = DeterministicExecutor()
    mutation_policy = MutationPolicy(
        allowed_component_names=("match-policy",),
        allowed_media_types=("text/plain",),
        max_artifact_size_bytes=4_096,
    )
    total_evaluations = 2 * len(EXPLORATION_TASKS) + 2 * len(GATE_TASKS)
    protocol_manifest = ProtocolManifest(
        benchmark_fingerprint=FIXTURE_KIND,
        splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=exploration_suite_ref,
            ),
            ProtocolSplit(
                partition=ProtocolPartition.GATE,
                manifest_ref=gate_suite_ref,
            ),
        ),
        model_fingerprint=seed_manifest.model_fingerprint,
        inference_fingerprint="synthetic-deterministic:no-inference-settings",
        runtime_fingerprint=seed_manifest.runtime_fingerprint,
        model_spec_fingerprint="0" * 64,
        sandbox_fingerprint="logical-fixture-isolation:no-security-claim",
        capability_policy_ref=capability_policy_ref,
        grader_fingerprint="synthetic-match-label-grader-v1",
        gate_batch_attestor_id=trusted_gate_batches.attestor_id,
        mechanism_evidence_attestor_id=trusted_mechanism_evidence.attestor_id,
        gate_config_ref=gate_config_ref,
        trusted_plane_version=seed_manifest.trusted_plane_version,
        budget=BudgetPolicy(
            max_tokens=executor.tokens_per_trial * total_evaluations,
            max_tool_calls=0,
            max_wall_time_seconds=(executor.latency_ms_per_trial * total_evaluations / 1_000),
            max_cost_usd=0.0,
            max_evaluations=total_evaluations,
        ),
    )
    protocol_manifest_ref = store.put_json(
        protocol_manifest,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
    )

    # Execution consumes only inputs reloaded through the persisted protocol.
    # Keeping module constants out of the execution path makes a replayed ref,
    # rather than accidental Python object identity, the source of truth.
    frozen_protocol = store.get_json(protocol_manifest_ref, ProtocolManifest)
    split_refs = {split.partition: split.manifest_ref for split in frozen_protocol.splits}
    exploration_tasks = store.get_json(
        split_refs[ProtocolPartition.EXPLORATION],
        tuple[BenchmarkTask, ...],
    )
    gate_tasks = store.get_json(
        split_refs[ProtocolPartition.GATE],
        tuple[BenchmarkTask, ...],
    )
    frozen_gate_config = store.get_json(frozen_protocol.gate_config_ref, GateConfig)
    if {task.task_id for task in exploration_tasks} & {task.task_id for task in gate_tasks}:
        raise RuntimeError("controlled protocol task IDs overlap across exploration and gate")
    exploration_content = {(task.reference_text, task.observed_text) for task in exploration_tasks}
    gate_content = {(task.reference_text, task.observed_text) for task in gate_tasks}
    if exploration_content & gate_content:
        raise RuntimeError("controlled protocol task content overlaps across partitions")
    if frozen_gate_config.expected_task_ids != tuple(task.task_id for task in gate_tasks):
        raise RuntimeError("controlled gate roster does not match its frozen gate config")

    experiment_manifest = ExperimentManifest(
        protocol_ref=protocol_manifest_ref,
        seed_harness_ref=seed_manifest_ref,
        mutation_policy=mutation_policy,
        objective="repair the controlled normalization fault under the frozen paired gate",
        baselines=("synthetic-static-seed",),
        stopping=("one-preregistered-candidate", "total-evaluation-budget"),
        search_budget=frozen_protocol.budget,
    )
    experiment_manifest_ref = store.put_json(
        experiment_manifest,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    analysis_plan_ref = store.put_json(
        {
            "schema_version": "1",
            "fixture_kind": FIXTURE_KIND,
            "primary_metric": "paired exact-match accuracy",
            "protected_slices": [PROTECTED_CANONICAL_SLICE],
            "sealed_evaluation": "not available in the controlled two-split fixture",
        },
        media_type="application/vnd.spiral-harness.synthetic-analysis-plan.v1+json",
    )

    exploration_executions = executor.execute_suite(
        exploration_tasks,
        prompt_bytes=store.get_bytes(seed_prompt_ref),
        prompt_ref=seed_prompt_ref,
        harness_id=seed_manifest_ref.sha256,
        seed=EXPLORATION_SEED,
    )
    exploration_trials = _observations(exploration_executions)
    exploration_execution_ref = _put_models(
        store,
        exploration_executions,
        media_type="application/vnd.spiral-harness.synthetic-exploration-executions+json",
    )
    exploration_trials_ref = _put_models(
        store,
        exploration_trials,
        media_type="application/vnd.spiral-harness.exploration-observations+json",
    )

    failed_normalization_task_ids = tuple(
        trial.task_id
        for trial in exploration_trials
        if NORMALIZATION_SLICE in trial.slice_tags and trial.score == 0.0
    )
    passing_protected_task_ids = tuple(
        trial.task_id
        for trial in exploration_trials
        if PROTECTED_CANONICAL_SLICE in trial.slice_tags and trial.score == 1.0
    )
    fault_evidence = ControlledFaultEvidence(
        exploration_suite_ref=exploration_suite_ref,
        seed_prompt_ref=seed_prompt_ref,
        seed_manifest_ref=seed_manifest_ref,
        exploration_trials_ref=exploration_trials_ref,
        exploration_execution_ref=exploration_execution_ref,
        failed_normalization_task_ids=failed_normalization_task_ids,
        passing_protected_task_ids=passing_protected_task_ids,
        root_cause=(
            "the seed prompt explicitly requests exact comparison and therefore omits "
            "casefolding and surrounding-whitespace normalization"
        ),
    )
    fault_evidence_ref = store.put_json(
        fault_evidence,
        media_type="application/vnd.spiral-harness.synthetic-fault-evidence+json",
    )

    candidate_prompt_ref = store.put_bytes(CANDIDATE_PROMPT.encode(), media_type=_PROMPT_MEDIA_TYPE)
    candidate_component = HarnessComponentRef(
        name="match-policy",
        kind=ComponentKind.PROMPT,
        artifact=candidate_prompt_ref,
    )
    mutation = CandidateMutation(
        target_component="match-policy",
        before=seed_component,
        after=candidate_component,
        hypothesis=MutationHypothesis(
            evidence_refs=(fault_evidence_ref,),
            where="the synthetic match-policy prompt comparison rule",
            why=fault_evidence.root_cause,
            expected_activation="the repaired prompt is active in every candidate trial",
            expected_adherence="both operands are stripped and casefolded before comparison",
            expected_behavior="normalization probes change from DIFFERENT to MATCH",
            expected_benefit="at least ten paired synthetic task scores improve",
            protected_slices=(PROTECTED_CANONICAL_SLICE,),
            falsifier="fewer than ten normalization tasks improve under paired execution",
            negative_control="already-canonical strings should retain identical scores",
            risks=("normalization may be invalid for case-sensitive real tasks",),
        ),
    )
    candidate_mutation_ref = store.put_json(
        mutation,
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
    )
    candidate_manifest = HarnessRegistry(mutation_policy).apply_mutation(
        parent=seed_manifest,
        parent_ref=seed_manifest_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(candidate_prompt_ref),
        artifact_media_type=_PROMPT_MEDIA_TYPE,
    )
    candidate_manifest_ref = store.put_json(
        candidate_manifest,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate_record = CandidateManifest(
        experiment_ref=experiment_manifest_ref,
        parent_harness_ref=seed_manifest_ref,
        child_harness_ref=candidate_manifest_ref,
        mutation_ref=candidate_mutation_ref,
        evidence_refs=(fault_evidence_ref,),
        evaluation_plan_ref=gate_config_ref,
    )
    candidate_record_ref = store.put_json(
        candidate_record,
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    admission_service = CandidateAdmissionService(store)
    admission_report = admission_service.admit(
        candidate_ref=candidate_record_ref,
        experiment_ref=experiment_manifest_ref,
    )
    admission_report_ref = store.put_json(
        admission_report,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )
    admission_service.verify_report(
        candidate_ref=candidate_record_ref,
        experiment_ref=experiment_manifest_ref,
        report_ref=admission_report_ref,
    )

    controller = ExperimentController(
        store,
        experiment_ref=experiment_manifest_ref,
        gate_batch_verifier=trusted_gate_batches.verification_capability,
        mechanism_evidence_verifier=trusted_mechanism_evidence.verification_capability,
    )
    experiment_journal_tail_ref = controller.freeze_experiment()
    experiment_journal_tail_ref = controller.start_search(
        previous_tail_ref=experiment_journal_tail_ref
    )
    candidate_journal_tail_ref = controller.register_candidate(candidate_ref=candidate_record_ref)
    candidate_journal_tail_ref = controller.admit_candidate(
        candidate_ref=candidate_record_ref,
        previous_tail_ref=candidate_journal_tail_ref,
        admission_report_ref=admission_report_ref,
    )
    candidate_journal_tail_ref = controller.start_probes(
        candidate_ref=candidate_record_ref,
        previous_tail_ref=candidate_journal_tail_ref,
    )

    candidate_exploration_executions = executor.execute_suite(
        exploration_tasks,
        prompt_bytes=store.get_bytes(candidate_prompt_ref),
        prompt_ref=candidate_prompt_ref,
        harness_id=candidate_manifest_ref.sha256,
        seed=EXPLORATION_SEED,
    )
    candidate_exploration_trials = _observations(candidate_exploration_executions)
    candidate_exploration_execution_ref = _put_models(
        store,
        candidate_exploration_executions,
        media_type="application/vnd.spiral-harness.synthetic-exploration-executions+json",
    )
    candidate_exploration_trials_ref = _put_models(
        store,
        candidate_exploration_trials,
        media_type="application/vnd.spiral-harness.exploration-observations+json",
    )
    mechanism_evidence = _mechanism_evidence(
        mutation=mutation,
        mutation_ref=candidate_mutation_ref,
        seed_manifest=seed_manifest,
        seed_manifest_ref=seed_manifest_ref,
        candidate_manifest=candidate_manifest,
        candidate_manifest_ref=candidate_manifest_ref,
        candidate_prompt_ref=candidate_prompt_ref,
        parent_probe_trials=exploration_trials,
        parent_probe_trials_ref=exploration_trials_ref,
        candidate_probe_trials=candidate_exploration_trials,
        candidate_probe_trials_ref=candidate_exploration_trials_ref,
        candidate_executions=candidate_exploration_executions,
        candidate_execution_ref=candidate_exploration_execution_ref,
        exploration_tasks=exploration_tasks,
    )
    attested_mechanism_evidence = trusted_mechanism_evidence.create(
        protocol_ref=protocol_manifest_ref,
        protocol=frozen_protocol,
        candidate_ref=candidate_record_ref,
        candidate_harness_ref=candidate_manifest_ref,
        source_refs=(
            candidate_exploration_execution_ref,
            candidate_exploration_trials_ref,
            candidate_manifest_ref,
            candidate_mutation_ref,
            exploration_trials_ref,
            seed_manifest_ref,
        ),
        evidence=mechanism_evidence,
    )
    mechanism_evidence_ref = store.put_json(
        attested_mechanism_evidence,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    candidate_journal_tail_ref = controller.start_gate(
        candidate_ref=candidate_record_ref,
        previous_tail_ref=candidate_journal_tail_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
    )

    # Gate measurements use task IDs and strings that never appeared in the
    # exploration evidence.  This is logical fixture isolation, not sealing.
    seed_gate_executions = executor.execute_suite(
        gate_tasks,
        prompt_bytes=store.get_bytes(seed_prompt_ref),
        prompt_ref=seed_prompt_ref,
        harness_id=seed_manifest_ref.sha256,
        seed=FIXTURE_SEED,
    )
    candidate_gate_executions = executor.execute_suite(
        gate_tasks,
        prompt_bytes=store.get_bytes(candidate_prompt_ref),
        prompt_ref=candidate_prompt_ref,
        harness_id=candidate_manifest_ref.sha256,
        seed=FIXTURE_SEED,
    )
    seed_gate_trials = _observations(seed_gate_executions)
    candidate_gate_trials = _observations(candidate_gate_executions)
    seed_gate_execution_ref = _put_models(
        store,
        seed_gate_executions,
        media_type="application/vnd.spiral-harness.synthetic-gate-executions+json",
    )
    candidate_gate_execution_ref = _put_models(
        store,
        candidate_gate_executions,
        media_type="application/vnd.spiral-harness.synthetic-gate-executions+json",
    )
    seed_gate_trials_ref = _put_models(
        store,
        seed_gate_trials,
        media_type="application/vnd.spiral-harness.gate-trial-observations+json",
    )
    candidate_gate_trials_ref = _put_models(
        store,
        candidate_gate_trials,
        media_type="application/vnd.spiral-harness.gate-trial-observations+json",
    )

    seed_gate_batch = trusted_gate_batches.create(
        protocol_ref=protocol_manifest_ref,
        protocol=frozen_protocol,
        candidate_ref=candidate_record_ref,
        arm=GateTrialArm.PARENT,
        harness_ref=seed_manifest_ref,
        gate_split_ref=split_refs[ProtocolPartition.GATE],
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(seed_gate_execution_ref, seed_gate_trials_ref),
        observations=seed_gate_trials,
    )
    seed_gate_batch_ref = store.put_json(
        seed_gate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    candidate_gate_batch = trusted_gate_batches.create(
        protocol_ref=protocol_manifest_ref,
        protocol=frozen_protocol,
        candidate_ref=candidate_record_ref,
        arm=GateTrialArm.CANDIDATE,
        harness_ref=candidate_manifest_ref,
        gate_split_ref=split_refs[ProtocolPartition.GATE],
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(candidate_gate_execution_ref, candidate_gate_trials_ref),
        observations=candidate_gate_trials,
    )
    candidate_gate_batch_ref = store.put_json(
        candidate_gate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )

    gate_evaluation = GateEvaluationManifest(
        candidate_ref=candidate_record_ref,
        admission_report_ref=admission_report_ref,
        gate_config_ref=frozen_protocol.gate_config_ref,
        gate_split_ref=split_refs[ProtocolPartition.GATE],
        parent_batch_ref=seed_gate_batch_ref,
        candidate_batch_ref=candidate_gate_batch_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
    )
    gate_evaluation_ref = store.put_json(
        gate_evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )

    evidence_completion = controller.complete_evidence(
        candidate_ref=candidate_record_ref,
        previous_tail_ref=candidate_journal_tail_ref,
        evaluation_ref=gate_evaluation_ref,
        previous_usage_tail_ref=None,
    )
    candidate_journal_tail_ref = evidence_completion.candidate_tail_ref
    experiment_usage_claim_ref = evidence_completion.usage_claim_ref
    experiment_usage_tail_ref = evidence_completion.usage_tail_ref
    decision = PromotionGate(frozen_gate_config).evaluate(
        seed_gate_trials,
        candidate_gate_trials,
        mechanism_evidence,
        parent_harness_id=seed_manifest_ref.sha256,
        candidate_harness_id=candidate_manifest_ref.sha256,
    )
    gate_decision_ref = store.put_json(
        decision,
        media_type="application/vnd.spiral-harness.gate-decision+json",
    )
    terminal_state = {
        Decision.PROMOTE: CandidateState.PROMOTED,
        Decision.REJECT: CandidateState.REJECTED,
        Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
    }[decision.decision]
    terminal_decision_service = TerminalDecisionService(
        store,
        gate_batch_verifier=trusted_gate_batches.verification_capability,
        mechanism_evidence_verifier=trusted_mechanism_evidence.verification_capability,
    )
    terminal_decision_report = terminal_decision_service.validate(
        candidate_ref=candidate_record_ref,
        experiment_ref=experiment_manifest_ref,
        evaluation_ref=gate_evaluation_ref,
        decision_ref=gate_decision_ref,
        terminal_state=terminal_state,
    )
    terminal_decision_report_ref = store.put_json(
        terminal_decision_report,
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )
    terminal_decision_service.verify_report(
        terminal_decision_report_ref,
        candidate_ref=candidate_record_ref,
        experiment_ref=experiment_manifest_ref,
        evaluation_ref=gate_evaluation_ref,
    )
    terminal_completion = controller.finalize_candidate(
        candidate_ref=candidate_record_ref,
        previous_tail_ref=candidate_journal_tail_ref,
        terminal_decision_report_ref=terminal_decision_report_ref,
    )
    candidate_journal_tail_ref = terminal_completion.candidate_tail_ref
    terminal_authorization_ref = terminal_completion.authorization_ref
    experiment_journal_tail_ref = controller.close_selection(
        previous_tail_ref=experiment_journal_tail_ref,
        previous_usage_tail_ref=experiment_usage_tail_ref,
        champion_candidate_ref=candidate_record_ref,
        champion_candidate_tail_ref=candidate_journal_tail_ref,
        champion_harness_ref=candidate_manifest_ref,
        analysis_plan_ref=analysis_plan_ref,
    )

    refs = ControlledDemoRefs(
        exploration_suite_ref=exploration_suite_ref,
        gate_suite_ref=gate_suite_ref,
        gate_config_ref=gate_config_ref,
        capability_policy_ref=capability_policy_ref,
        analysis_plan_ref=analysis_plan_ref,
        protocol_manifest_ref=protocol_manifest_ref,
        experiment_manifest_ref=experiment_manifest_ref,
        seed_prompt_ref=seed_prompt_ref,
        candidate_prompt_ref=candidate_prompt_ref,
        seed_manifest_ref=seed_manifest_ref,
        candidate_manifest_ref=candidate_manifest_ref,
        candidate_record_ref=candidate_record_ref,
        exploration_trials_ref=exploration_trials_ref,
        exploration_execution_ref=exploration_execution_ref,
        candidate_exploration_trials_ref=candidate_exploration_trials_ref,
        candidate_exploration_execution_ref=candidate_exploration_execution_ref,
        seed_gate_trials_ref=seed_gate_trials_ref,
        candidate_gate_trials_ref=candidate_gate_trials_ref,
        seed_gate_execution_ref=seed_gate_execution_ref,
        candidate_gate_execution_ref=candidate_gate_execution_ref,
        fault_evidence_ref=fault_evidence_ref,
        candidate_mutation_ref=candidate_mutation_ref,
        admission_report_ref=admission_report_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        seed_gate_batch_ref=seed_gate_batch_ref,
        candidate_gate_batch_ref=candidate_gate_batch_ref,
        gate_evaluation_ref=gate_evaluation_ref,
        gate_decision_ref=gate_decision_ref,
        terminal_decision_report_ref=terminal_decision_report_ref,
        experiment_usage_claim_ref=experiment_usage_claim_ref,
        experiment_usage_tail_ref=experiment_usage_tail_ref,
        terminal_authorization_ref=terminal_authorization_ref,
        candidate_journal_tail_ref=candidate_journal_tail_ref,
        experiment_journal_tail_ref=experiment_journal_tail_ref,
    )
    run_manifest_ref = store.put_json(
        refs,
        media_type="application/vnd.spiral-harness.controlled-run-manifest+json",
    )
    return ControlledDemoResult(
        run_manifest_ref=run_manifest_ref,
        refs=refs,
        decision=decision,
    )


__all__ = [
    "SYNTHETIC_DISCLAIMER",
    "ControlledDemoRefs",
    "ControlledDemoResult",
    "ControlledFaultEvidence",
    "run_controlled_demo",
]
