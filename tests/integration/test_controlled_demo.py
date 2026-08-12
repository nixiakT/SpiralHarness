from __future__ import annotations

from spiral_harness.benchmark.controlled_fixture import (
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
from spiral_harness.core.experiment import (
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import CandidateMutation, HarnessManifest
from spiral_harness.evolution.controlled_demo import (
    ControlledDemoRefs,
    ControlledFaultEvidence,
    run_controlled_demo,
)
from spiral_harness.experiments.admission import AdmissionReport
from spiral_harness.experiments.controller import ExperimentController
from spiral_harness.experiments.controller_artifacts import (
    ExperimentUsageClaim,
    ExperimentUsageEntry,
    TerminalTransitionAuthorization,
)
from spiral_harness.experiments.decision import (
    GateEvaluationManifest,
    TerminalDecisionReport,
    TerminalDecisionService,
)
from spiral_harness.experiments.lifecycle import (
    ExperimentJournal,
    ExperimentState,
    SelectionClosure,
    SelectionReason,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.journal import CandidateJournal, JournalEntry
from spiral_harness.verification.artifacts import GateTrialBatch, TrustedGateBatchService
from spiral_harness.verification.mechanism import (
    AttestedMechanismEvidence,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import (
    Decision,
    GateConfig,
    GateDecision,
    TrialObservation,
)


def test_controlled_fault_demo_promotes_with_replayable_lineage(tmp_path) -> None:
    ledger = tmp_path / "controlled-ledger"
    gate_batch_service = TrustedGateBatchService()
    mechanism_evidence_service = TrustedMechanismEvidenceService()
    first = run_controlled_demo(
        ledger,
        gate_batch_service=gate_batch_service,
        mechanism_evidence_service=mechanism_evidence_service,
    )
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
    mechanism_envelope = store.get_json(
        refs.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    mechanism = mechanism_envelope.evidence
    gate_evaluation = store.get_json(
        refs.gate_evaluation_ref,
        GateEvaluationManifest,
    )
    seed_gate_batch = store.get_json(refs.seed_gate_batch_ref, GateTrialBatch)
    candidate_gate_batch = store.get_json(refs.candidate_gate_batch_ref, GateTrialBatch)
    admission = store.get_json(refs.admission_report_ref, AdmissionReport)
    stored_decision = store.get_json(refs.gate_decision_ref, GateDecision)
    terminal_report = store.get_json(
        refs.terminal_decision_report_ref,
        TerminalDecisionReport,
    )
    usage_claim = store.get_json(refs.experiment_usage_claim_ref, ExperimentUsageClaim)
    usage_entry = store.get_json(refs.experiment_usage_tail_ref, ExperimentUsageEntry)
    authorization = store.get_json(
        refs.terminal_authorization_ref,
        TerminalTransitionAuthorization,
    )
    final_entry = store.get_json(refs.candidate_journal_tail_ref, JournalEntry)
    lifecycle_events = CandidateJournal(store).replay(refs.candidate_journal_tail_ref)
    experiment_events = ExperimentJournal(store).replay(refs.experiment_journal_tail_ref)
    selection_closure = store.get_json(experiment_events[-1].evidence_refs[0], SelectionClosure)

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
    assert protocol_manifest.capability_policy_ref == refs.capability_policy_ref
    assert protocol_manifest.gate_batch_attestor_id == gate_batch_service.attestor_id
    assert (
        protocol_manifest.mechanism_evidence_attestor_id == mechanism_evidence_service.attestor_id
    )
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
    assert gate_evaluation.parent_batch_ref == refs.seed_gate_batch_ref
    assert gate_evaluation.candidate_batch_ref == refs.candidate_gate_batch_ref
    assert gate_evaluation.mechanism_evidence_ref == refs.mechanism_evidence_ref
    assert seed_gate_batch.observations == seed_trials
    assert candidate_gate_batch.observations == candidate_trials
    assert seed_gate_batch.gate_split_ref == refs.gate_suite_ref
    assert candidate_gate_batch.gate_split_ref == refs.gate_suite_ref
    assert seed_gate_batch.protocol_ref == refs.protocol_manifest_ref
    assert candidate_gate_batch.protocol_ref == refs.protocol_manifest_ref
    assert seed_gate_batch.task_set_fingerprint == refs.gate_suite_ref.sha256
    assert candidate_gate_batch.task_set_fingerprint == refs.gate_suite_ref.sha256
    assert seed_gate_batch.mechanism_evidence_ref == refs.mechanism_evidence_ref
    assert candidate_gate_batch.mechanism_evidence_ref == refs.mechanism_evidence_ref
    assert seed_gate_batch.execution_context.grader_fingerprint == (
        protocol_manifest.grader_fingerprint
    )
    assert candidate_gate_batch.execution_context.capability_policy_ref == (
        refs.capability_policy_ref
    )
    assert {ref.sha256 for ref in seed_gate_batch.source_refs} == {
        refs.seed_gate_execution_ref.sha256,
        refs.seed_gate_trials_ref.sha256,
    }
    assert {ref.sha256 for ref in candidate_gate_batch.source_refs} == {
        refs.candidate_gate_execution_ref.sha256,
        refs.candidate_gate_trials_ref.sha256,
    }
    assert gate_batch_service.verification_capability.verify(seed_gate_batch) == seed_gate_batch
    assert (
        gate_batch_service.verification_capability.verify(candidate_gate_batch)
        == candidate_gate_batch
    )
    assert (
        mechanism_evidence_service.verification_capability.verify(mechanism_envelope)
        == mechanism_envelope
    )
    assert mechanism_envelope.protocol_ref == refs.protocol_manifest_ref
    assert mechanism_envelope.candidate_ref == refs.candidate_record_ref
    assert mechanism_envelope.candidate_harness_ref == refs.candidate_manifest_ref
    assert mechanism_envelope.exploration_split_ref == refs.exploration_suite_ref
    assert {ref.sha256 for ref in mechanism_envelope.source_refs} == {
        refs.candidate_exploration_execution_ref.sha256,
        refs.candidate_exploration_trials_ref.sha256,
        refs.candidate_manifest_ref.sha256,
        refs.candidate_mutation_ref.sha256,
        refs.exploration_trials_ref.sha256,
        refs.seed_manifest_ref.sha256,
    }
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
    assert terminal_report.gate_batch_attestor_id == gate_batch_service.attestor_id
    assert terminal_report.mechanism_evidence_attestor_id == mechanism_evidence_service.attestor_id
    assert terminal_report.mechanism_evidence_ref == refs.mechanism_evidence_ref
    assert terminal_report.terminal_state is CandidateState.PROMOTED
    assert (
        TerminalDecisionService(
            store,
            gate_batch_verifier=gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(mechanism_evidence_service.verification_capability),
        ).verify_report(
            refs.terminal_decision_report_ref,
            candidate_ref=refs.candidate_record_ref,
            experiment_ref=refs.experiment_manifest_ref,
            evaluation_ref=refs.gate_evaluation_ref,
        )
        == terminal_report
    )
    assert usage_claim.candidate_ref == refs.candidate_record_ref
    assert usage_claim.evaluation_ref == refs.gate_evaluation_ref
    assert usage_claim.evaluation_units == 2 * len(GATE_TASKS)
    expected_tokens = sum(trial.tokens for trial in (*seed_trials, *candidate_trials))
    expected_tool_calls = sum(trial.tool_calls for trial in (*seed_trials, *candidate_trials))
    expected_wall_time_seconds = (
        sum(trial.latency_ms for trial in (*seed_trials, *candidate_trials)) / 1_000
    )
    expected_cost = sum(trial.cost_usd or 0.0 for trial in (*seed_trials, *candidate_trials))
    assert usage_claim.tokens == expected_tokens
    assert usage_claim.tool_calls == expected_tool_calls == 0
    assert usage_claim.wall_time_seconds == expected_wall_time_seconds
    assert usage_claim.cost_usd == expected_cost == 0.0
    assert usage_entry.claim_ref == refs.experiment_usage_claim_ref
    assert usage_entry.cumulative_evaluations == 2 * len(GATE_TASKS)
    assert usage_entry.cumulative_tokens == expected_tokens
    assert usage_entry.cumulative_tool_calls == expected_tool_calls
    assert usage_entry.cumulative_wall_time_seconds == expected_wall_time_seconds
    assert usage_entry.cumulative_cost_usd == expected_cost
    usage = ExperimentController(
        store,
        experiment_ref=refs.experiment_manifest_ref,
        gate_batch_verifier=gate_batch_service.verification_capability,
        mechanism_evidence_verifier=(mechanism_evidence_service.verification_capability),
        usage_tail_ref=refs.experiment_usage_tail_ref,
    ).current_usage()
    assert usage.query_count == 1
    assert usage.total_evaluations == 2 * len(GATE_TASKS)
    assert usage.total_tokens == expected_tokens
    assert usage.total_tool_calls == expected_tool_calls
    assert usage.total_wall_time_seconds == expected_wall_time_seconds
    assert usage.total_cost_usd == expected_cost
    assert final_entry.previous_entry_ref is not None
    assert authorization.evidence_complete_tail_ref == final_entry.previous_entry_ref
    assert authorization.usage_entry_ref == refs.experiment_usage_tail_ref
    assert authorization.evaluation_ref == refs.gate_evaluation_ref
    assert authorization.terminal_decision_report_ref == refs.terminal_decision_report_ref
    assert authorization.decision_ref == refs.gate_decision_ref
    assert tuple(event.to_state for event in experiment_events) == (
        ExperimentState.FROZEN,
        ExperimentState.SEARCHING,
        ExperimentState.SELECTION_CLOSED,
    )
    assert selection_closure.champion_candidate_ref == refs.candidate_record_ref
    assert selection_closure.champion_candidate_tail_ref == refs.candidate_journal_tail_ref
    assert selection_closure.champion_harness_ref == refs.candidate_manifest_ref
    assert selection_closure.analysis_plan_ref == refs.analysis_plan_ref
    assert selection_closure.usage_tail_ref == refs.experiment_usage_tail_ref
    assert selection_closure.selection_reason is SelectionReason.PROMOTED_CHAMPION
    assert selection_closure.stopping_criteria == experiment_manifest.stopping
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
    assert lifecycle_events[4].evidence_refs == tuple(
        sorted(
            (refs.gate_evaluation_ref, refs.experiment_usage_tail_ref),
            key=lambda ref: (ref.sha256, ref.size, ref.media_type),
        )
    )
    assert lifecycle_events[-1].evidence_refs == tuple(
        sorted(
            (
                refs.gate_decision_ref,
                refs.terminal_decision_report_ref,
                refs.terminal_authorization_ref,
            ),
            key=lambda ref: (ref.sha256, ref.size, ref.media_type),
        )
    )

    object_paths_before = tuple(
        sorted(path for path in store.objects_dir.rglob("*") if path.is_file())
    )
    second = run_controlled_demo(
        ledger,
        gate_batch_service=gate_batch_service,
        mechanism_evidence_service=mechanism_evidence_service,
    )
    object_paths_after = tuple(
        sorted(path for path in store.objects_dir.rglob("*") if path.is_file())
    )

    assert second == first
    assert object_paths_after == object_paths_before
