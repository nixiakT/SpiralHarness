"""One deterministic M0 vertical slice over a synthetic controlled fault.

The flow exists to test evidence plumbing and promotion semantics.  It is not
a real benchmark, a learned optimizer, or evidence of general agent gains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spiral_harness.benchmark import (
    CANDIDATE_PROMPT,
    EXPLORATION_SEED,
    EXPLORATION_TASKS,
    FIXTURE_KIND,
    FIXTURE_SEED,
    GATE_TASKS,
    NORMALIZATION_SLICE,
    PROTECTED_CANONICAL_SLICE,
    SEED_PROMPT,
    DeterministicExecution,
    DeterministicExecutor,
)
from spiral_harness.core import (
    ArtifactRef,
    BudgetPolicy,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.storage import ArtifactStore
from spiral_harness.verification import (
    GateConfig,
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    PromotionGate,
    TrialObservation,
)

SYNTHETIC_DISCLAIMER = (
    "Synthetic controlled-fault fixture only; scores are pipeline checks, "
    "not real benchmark results. Exploration and gate rosters are logically "
    "disjoint but are not securely sealed."
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
    seed_prompt_ref: ArtifactRef
    candidate_prompt_ref: ArtifactRef
    seed_manifest_ref: ArtifactRef
    candidate_manifest_ref: ArtifactRef
    exploration_trials_ref: ArtifactRef
    exploration_execution_ref: ArtifactRef
    seed_gate_trials_ref: ArtifactRef
    candidate_gate_trials_ref: ArtifactRef
    seed_gate_execution_ref: ArtifactRef
    candidate_gate_execution_ref: ArtifactRef
    fault_evidence_ref: ArtifactRef
    candidate_mutation_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef
    gate_decision_ref: ArtifactRef

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
    seed_trials: tuple[TrialObservation, ...],
    seed_trials_ref: ArtifactRef,
    candidate_trials: tuple[TrialObservation, ...],
    candidate_trials_ref: ArtifactRef,
    candidate_executions: tuple[DeterministicExecution, ...],
    candidate_execution_ref: ArtifactRef,
) -> MechanismEvidence:
    tasks_by_id = {task.task_id: task for task in GATE_TASKS}
    seed_by_id = {trial.task_id: trial for trial in seed_trials}
    candidate_by_id = {trial.task_id: trial for trial in candidate_trials}

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
        for task in GATE_TASKS
        if candidate_by_id[task.task_id].score > seed_by_id[task.task_id].score
    )
    protected_task_ids = tuple(
        task.task_id for task in GATE_TASKS if PROTECTED_CANONICAL_SLICE in task.slice_tags
    )
    protected_unchanged = all(
        candidate_by_id[task_id].score == seed_by_id[task_id].score
        for task_id in protected_task_ids
    )
    behavior = len(improved_task_ids) >= 10 and protected_unchanged

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
                    f"{len(improved_task_ids)} synthetic tasks improved; "
                    f"{len(protected_task_ids)} protected canonical tasks were unchanged"
                ),
                evidence_refs=(seed_trials_ref.sha256, candidate_trials_ref.sha256),
            ),
        ),
    )


def run_controlled_demo(root: str | Path) -> ControlledDemoResult:
    """Run and durably record the deterministic synthetic M0 vertical slice."""

    store = ArtifactStore(root)
    seed_prompt_ref = store.put_bytes(SEED_PROMPT.encode(), media_type=_PROMPT_MEDIA_TYPE)
    exploration_suite_ref = _put_models(
        store,
        EXPLORATION_TASKS,
        media_type="application/vnd.spiral-harness.synthetic-exploration-tasks+json",
    )
    # The gate policy is persisted before any observations are produced.
    gate_config = _gate_config()
    gate_config_ref = store.put_json(
        gate_config,
        media_type="application/vnd.spiral-harness.gate-config+json",
    )

    seed_component = HarnessComponentRef(
        name="match-policy",
        kind=ComponentKind.PROMPT,
        artifact=seed_prompt_ref,
    )
    seed_manifest = _manifest(seed_component)
    seed_manifest_ref = store.put_json(
        seed_manifest,
        media_type="application/vnd.spiral-harness.manifest+json",
    )

    executor = DeterministicExecutor()
    exploration_executions = executor.execute_suite(
        EXPLORATION_TASKS,
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
        media_type="application/vnd.spiral-harness.candidate-mutation+json",
    )
    candidate_manifest = _manifest(candidate_component, parent=seed_manifest_ref)
    candidate_manifest_ref = store.put_json(
        candidate_manifest,
        media_type="application/vnd.spiral-harness.manifest+json",
    )

    # Gate measurements use task IDs and strings that never appeared in the
    # exploration evidence.  This is logical fixture isolation, not sealing.
    gate_suite_ref = _put_models(
        store,
        GATE_TASKS,
        media_type="application/vnd.spiral-harness.synthetic-tasks+json",
    )
    seed_gate_executions = executor.execute_suite(
        GATE_TASKS,
        prompt_bytes=store.get_bytes(seed_prompt_ref),
        prompt_ref=seed_prompt_ref,
        harness_id=seed_manifest_ref.sha256,
        seed=FIXTURE_SEED,
    )
    candidate_gate_executions = executor.execute_suite(
        GATE_TASKS,
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

    mechanism_evidence = _mechanism_evidence(
        mutation=mutation,
        mutation_ref=candidate_mutation_ref,
        seed_manifest=seed_manifest,
        seed_manifest_ref=seed_manifest_ref,
        candidate_manifest=candidate_manifest,
        candidate_manifest_ref=candidate_manifest_ref,
        candidate_prompt_ref=candidate_prompt_ref,
        seed_trials=seed_gate_trials,
        seed_trials_ref=seed_gate_trials_ref,
        candidate_trials=candidate_gate_trials,
        candidate_trials_ref=candidate_gate_trials_ref,
        candidate_executions=candidate_gate_executions,
        candidate_execution_ref=candidate_gate_execution_ref,
    )
    mechanism_evidence_ref = store.put_json(
        mechanism_evidence,
        media_type="application/vnd.spiral-harness.mechanism-evidence+json",
    )
    decision = PromotionGate(gate_config).evaluate(
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

    refs = ControlledDemoRefs(
        exploration_suite_ref=exploration_suite_ref,
        gate_suite_ref=gate_suite_ref,
        gate_config_ref=gate_config_ref,
        seed_prompt_ref=seed_prompt_ref,
        candidate_prompt_ref=candidate_prompt_ref,
        seed_manifest_ref=seed_manifest_ref,
        candidate_manifest_ref=candidate_manifest_ref,
        exploration_trials_ref=exploration_trials_ref,
        exploration_execution_ref=exploration_execution_ref,
        seed_gate_trials_ref=seed_gate_trials_ref,
        candidate_gate_trials_ref=candidate_gate_trials_ref,
        seed_gate_execution_ref=seed_gate_execution_ref,
        candidate_gate_execution_ref=candidate_gate_execution_ref,
        fault_evidence_ref=fault_evidence_ref,
        candidate_mutation_ref=candidate_mutation_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        gate_decision_ref=gate_decision_ref,
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
