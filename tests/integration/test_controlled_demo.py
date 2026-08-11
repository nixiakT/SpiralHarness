from __future__ import annotations

from spiral_harness.benchmark import (
    CANDIDATE_PROMPT,
    CONTROLLED_TASKS,
    EXPLORATION_SEED,
    EXPLORATION_TASKS,
    FIXTURE_SEED,
    GATE_TASKS,
    PROTECTED_CANONICAL_SLICE,
    SEED_PROMPT,
    BenchmarkTask,
    DeterministicExecution,
)
from spiral_harness.core import (
    CandidateManifest,
    CandidateMutation,
    CandidateState,
    ExperimentManifest,
    HarnessManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.evolution import (
    ControlledDemoRefs,
    ControlledFaultEvidence,
    run_controlled_demo,
)
from spiral_harness.experiments import (
    AdmissionReport,
    GateEvaluationManifest,
    TerminalDecisionReport,
)
from spiral_harness.storage import ArtifactStore, CandidateJournal
from spiral_harness.verification import (
    Decision,
    GateConfig,
    GateDecision,
    MechanismEvidence,
    TrialObservation,
)


def test_controlled_fault_demo_promotes_with_replayable_lineage(tmp_path) -> None:
    ledger = tmp_path / "controlled-ledger"
    first = run_controlled_demo(ledger)
    store = ArtifactStore(ledger)

    assert first.decision.decision is Decision.PROMOTE
    assert "Synthetic controlled-fault fixture" in first.disclaimer
    metrics = first.decision.metrics
    assert metrics is not None
    assert metrics.n_tasks == len(CONTROLLED_TASKS) == 16
    assert metrics.wins >= 10
    assert len(metrics.task_comparisons) == len(CONTROLLED_TASKS)
    assert metrics.slices[PROTECTED_CANONICAL_SLICE].mean_delta == 0.0

    refs = first.refs
    stored_refs = store.get_json(first.run_manifest_ref, ControlledDemoRefs)
    assert stored_refs == refs
    for field_name in refs.__class__.model_fields:
        store.get_bytes(getattr(refs, field_name))

    assert store.get_bytes(refs.seed_prompt_ref).decode() == SEED_PROMPT
    assert store.get_bytes(refs.candidate_prompt_ref).decode() == CANDIDATE_PROMPT
    assert "NOT A REAL BENCHMARK" in SEED_PROMPT
    assert (
        sum(
            seed_line != candidate_line
            for seed_line, candidate_line in zip(
                SEED_PROMPT.splitlines(), CANDIDATE_PROMPT.splitlines(), strict=True
            )
        )
        == 1
    )

    exploration_tasks = store.get_json(refs.exploration_suite_ref, tuple[BenchmarkTask, ...])
    gate_tasks = store.get_json(refs.gate_suite_ref, tuple[BenchmarkTask, ...])
    gate_config = store.get_json(refs.gate_config_ref, GateConfig)
    protocol_manifest = store.get_json(refs.protocol_manifest_ref, ProtocolManifest)
    experiment_manifest = store.get_json(refs.experiment_manifest_ref, ExperimentManifest)
    seed_manifest = store.get_json(refs.seed_manifest_ref, HarnessManifest)
    candidate_manifest = store.get_json(refs.candidate_manifest_ref, HarnessManifest)
    candidate_record = store.get_json(refs.candidate_record_ref, CandidateManifest)
    mutation = store.get_json(refs.candidate_mutation_ref, CandidateMutation)
    fault_evidence = store.get_json(refs.fault_evidence_ref, ControlledFaultEvidence)
    exploration_trials = store.get_json(refs.exploration_trials_ref, tuple[TrialObservation, ...])
    exploration_executions = store.get_json(
        refs.exploration_execution_ref, tuple[DeterministicExecution, ...]
    )
    candidate_exploration_trials = store.get_json(
        refs.candidate_exploration_trials_ref,
        tuple[TrialObservation, ...],
    )
    candidate_exploration_executions = store.get_json(
        refs.candidate_exploration_execution_ref,
        tuple[DeterministicExecution, ...],
    )
    seed_trials = store.get_json(refs.seed_gate_trials_ref, tuple[TrialObservation, ...])
    candidate_trials = store.get_json(refs.candidate_gate_trials_ref, tuple[TrialObservation, ...])
    seed_executions = store.get_json(
        refs.seed_gate_execution_ref, tuple[DeterministicExecution, ...]
    )
    candidate_executions = store.get_json(
        refs.candidate_gate_execution_ref, tuple[DeterministicExecution, ...]
    )
    mechanism = store.get_json(refs.mechanism_evidence_ref, MechanismEvidence)
    gate_evaluation = store.get_json(
        refs.gate_evaluation_ref,
        GateEvaluationManifest,
    )
    admission = store.get_json(refs.admission_report_ref, AdmissionReport)
    stored_decision = store.get_json(refs.gate_decision_ref, GateDecision)
    terminal_report = store.get_json(
        refs.terminal_decision_report_ref,
        TerminalDecisionReport,
    )
    lifecycle_events = CandidateJournal(store).replay(refs.candidate_journal_tail_ref)

    assert exploration_tasks == EXPLORATION_TASKS
    assert gate_tasks == GATE_TASKS == CONTROLLED_TASKS
    assert {task.task_id for task in exploration_tasks}.isdisjoint(
        task.task_id for task in gate_tasks
    )
    assert {(task.reference_text, task.observed_text) for task in exploration_tasks}.isdisjoint(
        (task.reference_text, task.observed_text) for task in gate_tasks
    )
    assert {trial.task_id for trial in exploration_trials}.isdisjoint(
        trial.task_id for trial in seed_trials
    )
    assert {trial.seed for trial in exploration_trials} == {EXPLORATION_SEED}
    assert {trial.seed for trial in seed_trials} == {FIXTURE_SEED}
    assert gate_config.expected_task_ids == tuple(task.task_id for task in GATE_TASKS)
    assert gate_config.expected_seeds == (FIXTURE_SEED,)
    assert gate_config.required_mechanism_checks == (
        "patch_validity",
        "activation",
        "adherence",
        "behavior",
    )
    assert gate_config.max_tokens_ratio == 1.0
    assert gate_config.max_latency_ratio == 1.0
    assert gate_config.max_tool_calls_ratio == 1.0
    assert gate_config.protected_slice_floors == {PROTECTED_CANONICAL_SLICE: -0.01}

    splits = {split.partition: split.manifest_ref for split in protocol_manifest.splits}
    assert splits == {
        ProtocolPartition.EXPLORATION: refs.exploration_suite_ref,
        ProtocolPartition.GATE: refs.gate_suite_ref,
    }
    assert protocol_manifest.gate_config_ref == refs.gate_config_ref
    assert protocol_manifest.budget.max_evaluations == (
        2 * len(EXPLORATION_TASKS) + 2 * len(GATE_TASKS)
    )
    assert experiment_manifest.protocol_ref == refs.protocol_manifest_ref
    assert experiment_manifest.seed_harness_ref == refs.seed_manifest_ref
    assert experiment_manifest.mutation_policy.allowed_kinds[0].value == "prompt"
    assert experiment_manifest.mutation_policy.allowed_component_names == ("match-policy",)
    assert experiment_manifest.search_budget == protocol_manifest.budget

    assert candidate_manifest.parent == refs.seed_manifest_ref
    assert candidate_manifest.model_fingerprint == seed_manifest.model_fingerprint
    assert candidate_manifest.runtime_fingerprint == seed_manifest.runtime_fingerprint
    assert candidate_manifest.trusted_plane_version == seed_manifest.trusted_plane_version
    assert candidate_manifest.budget == seed_manifest.budget
    assert candidate_record.experiment_ref == refs.experiment_manifest_ref
    assert candidate_record.parent_harness_ref == refs.seed_manifest_ref
    assert candidate_record.child_harness_ref == refs.candidate_manifest_ref
    assert candidate_record.mutation_ref == refs.candidate_mutation_ref
    assert candidate_record.evidence_refs == (refs.fault_evidence_ref,)
    assert candidate_record.evaluation_plan_ref == refs.gate_config_ref
    assert mutation.before == seed_manifest.components[0]
    assert mutation.after == candidate_manifest.components[0]
    assert mutation.hypothesis.evidence_refs == (refs.fault_evidence_ref,)
    assert admission.admitted
    assert admission.protocol_ref == refs.protocol_manifest_ref
    assert admission.experiment_ref == refs.experiment_manifest_ref
    assert admission.mutation_ref == refs.candidate_mutation_ref
    assert admission.parent_harness_ref == refs.seed_manifest_ref
    assert admission.child_harness_ref == refs.candidate_manifest_ref
    assert admission.candidate_ref == refs.candidate_record_ref
    assert admission.gate_config_ref == refs.gate_config_ref
    assert gate_evaluation.candidate_ref == refs.candidate_record_ref
    assert gate_evaluation.admission_report_ref == refs.admission_report_ref
    assert gate_evaluation.gate_config_ref == refs.gate_config_ref
    assert gate_evaluation.gate_split_ref == refs.gate_suite_ref
    assert gate_evaluation.parent_trials_ref == refs.seed_gate_trials_ref
    assert gate_evaluation.candidate_trials_ref == refs.candidate_gate_trials_ref
    assert gate_evaluation.mechanism_evidence_ref == refs.mechanism_evidence_ref
    assert fault_evidence.exploration_suite_ref == refs.exploration_suite_ref
    assert fault_evidence.exploration_trials_ref == refs.exploration_trials_ref
    assert len(fault_evidence.failed_normalization_task_ids) == 4
    assert len(fault_evidence.passing_protected_task_ids) == 2
    assert set(fault_evidence.failed_normalization_task_ids).issubset(
        task.task_id for task in exploration_tasks
    )
    assert set(fault_evidence.failed_normalization_task_ids).isdisjoint(
        task.task_id for task in gate_tasks
    )

    exploration_parent_by_id = {trial.task_id: trial for trial in exploration_trials}
    exploration_candidate_by_id = {trial.task_id: trial for trial in candidate_exploration_trials}
    assert exploration_parent_by_id.keys() == exploration_candidate_by_id.keys()
    assert (
        sum(
            exploration_candidate_by_id[task_id].score > exploration_parent_by_id[task_id].score
            for task_id in exploration_parent_by_id
        )
        == 4
    )

    seed_by_key = {(trial.task_id, trial.seed): trial for trial in seed_trials}
    candidate_by_key = {(trial.task_id, trial.seed): trial for trial in candidate_trials}
    assert seed_by_key.keys() == candidate_by_key.keys()
    for key, seed_trial in seed_by_key.items():
        candidate_trial = candidate_by_key[key]
        assert seed_trial.execution_fingerprint == candidate_trial.execution_fingerprint
        assert seed_trial.harness_id != candidate_trial.harness_id

    assert all(execution.policy == "exact" for execution in seed_executions)
    assert all(execution.policy == "exact" for execution in exploration_executions)
    assert all(
        execution.policy == "strip-and-casefold" for execution in candidate_exploration_executions
    )
    assert all(execution.policy == "strip-and-casefold" for execution in candidate_executions)
    assert all(check.passed is True for check in mechanism.checks)
    assert stored_decision.gate_config_sha256 == refs.gate_config_ref.sha256
    assert stored_decision == first.decision
    assert terminal_report.candidate_ref == refs.candidate_record_ref
    assert terminal_report.experiment_ref == refs.experiment_manifest_ref
    assert terminal_report.admission_report_ref == refs.admission_report_ref
    assert terminal_report.evaluation_ref == refs.gate_evaluation_ref
    assert terminal_report.decision_ref == refs.gate_decision_ref
    assert terminal_report.gate_config_ref == refs.gate_config_ref
    assert terminal_report.terminal_state is CandidateState.PROMOTED
    assert tuple(event.to_state for event in lifecycle_events) == (
        CandidateState.REGISTERED,
        CandidateState.VALID,
        CandidateState.RUNNING_PROBES,
        CandidateState.RUNNING_GATE,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.PROMOTED,
    )
    assert all(event.candidate_ref == refs.candidate_record_ref for event in lifecycle_events)
    assert lifecycle_events[1].evidence_refs == (refs.admission_report_ref,)
    assert lifecycle_events[3].evidence_refs == (refs.mechanism_evidence_ref,)
    assert refs.gate_evaluation_ref in lifecycle_events[4].evidence_refs
    assert lifecycle_events[-1].evidence_refs == tuple(
        sorted(
            (refs.gate_decision_ref, refs.terminal_decision_report_ref),
            key=lambda ref: (ref.sha256, ref.size, ref.media_type),
        )
    )

    object_paths_before = tuple(
        sorted(path for path in store.objects_dir.rglob("*") if path.is_file())
    )
    second = run_controlled_demo(ledger)
    object_paths_after = tuple(
        sorted(path for path in store.objects_dir.rglob("*") if path.is_file())
    )

    assert second == first
    assert object_paths_after == object_paths_before
