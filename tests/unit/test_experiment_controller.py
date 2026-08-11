from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_terminal_decision import (
    DecisionGraph,
    ForgedGateVerificationCapability,
    ForgedMechanismVerificationCapability,
    build_graph,
    put_json,
    reissue_batch,
    store_forged_score_attack,
)

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.lifecycle import CandidateLifecycleEvent, CandidateState
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    CandidateMutation,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.experiments.admission import (
    ADMISSION_REPORT_MEDIA_TYPE,
    CandidateAdmissionService,
)
from spiral_harness.experiments.controller import (
    ADMISSION_FAILURE_REPORT_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    PROBE_REJECTION_REPORT_MEDIA_TYPE,
    SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE,
    TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    AdmissionFailureCode,
    AdmissionFailureReport,
    ExperimentBudgetError,
    ExperimentController,
    ExperimentControllerError,
    ExperimentUsageClaim,
    ExperimentUsageEntry,
    ProbeRejectionCode,
    ProbeRejectionReport,
    StaleControllerTailError,
    SupersededCandidateReport,
    TerminalTransitionAuthorization,
)
from spiral_harness.experiments.decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    GateEvaluationManifest,
    TerminalDecisionService,
)
from spiral_harness.experiments.lifecycle import (
    EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE,
    EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE,
    SEALED_EVALUATION_REPORT_MEDIA_TYPE,
    SEALED_RUN_AUTHORIZATION_MEDIA_TYPE,
    SELECTION_CLOSURE_MEDIA_TYPE,
    ExperimentCompletionReport,
    ExperimentInvalidationReport,
    ExperimentJournal,
    ExperimentState,
    ExperimentViolationCode,
    SealedEvaluationReport,
    SealedRunAuthorization,
    SelectionClosure,
    SelectionReason,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.storage.journal import CandidateJournal
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateBatchExecutionContext,
    GateTrialArm,
    TrustedGateBatchService,
)
from spiral_harness.verification.gate import PromotionGate
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    AttestedMechanismEvidence,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import (
    Decision,
    MechanismCheck,
    MechanismEvidence,
)


def controller_for(
    graph: DecisionGraph | SimpleNamespace,
    **kwargs: object,
) -> ExperimentController:
    return ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        **kwargs,
    )


def store_attested_mechanism_evidence(
    graph: DecisionGraph | SimpleNamespace,
    *,
    candidate_ref: ArtifactRef,
    candidate_harness_ref: ArtifactRef,
    evidence: MechanismEvidence,
    protocol_ref: ArtifactRef | None = None,
    protocol: ProtocolManifest | None = None,
) -> ArtifactRef:
    """Reissue trusted probe evidence after a test fixture rebinds its lineage."""

    bound_protocol_ref = protocol_ref or graph.protocol_ref
    bound_protocol = protocol or graph.store.get_json(bound_protocol_ref, ProtocolManifest)
    original = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    envelope = graph.mechanism_evidence_service.create(
        protocol_ref=bound_protocol_ref,
        protocol=bound_protocol,
        candidate_ref=candidate_ref,
        candidate_harness_ref=candidate_harness_ref,
        source_refs=original.source_refs,
        evidence=evidence,
    )
    return put_json(
        graph.store,
        envelope,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )


def advance_to_gate(
    graph: DecisionGraph | SimpleNamespace,
    controller: ExperimentController,
):
    if controller.experiment_state is None:
        frozen = controller.freeze_experiment()
        controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=graph.admission_report_ref,
    )
    probes = controller.start_probes(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=valid,
    )
    gate = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=graph.mechanism_evidence_ref,
    )
    return registered, valid, probes, gate


def advance_to_probes(
    graph: DecisionGraph | SimpleNamespace,
    controller: ExperimentController,
) -> ArtifactRef:
    if controller.experiment_state is None:
        frozen = controller.freeze_experiment()
        controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=graph.admission_report_ref,
    )
    return controller.start_probes(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=valid,
    )


def store_terminal_report(graph: DecisionGraph):
    report = graph.service.validate(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=graph.evaluation_ref,
        decision_ref=graph.decision_ref,
        terminal_state=CandidateState.PROMOTED,
    )
    report_ref = put_json(
        graph.store,
        report,
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )
    return report, report_ref


def rebind_experiment_budget(
    graph: DecisionGraph,
    limit: int,
    *,
    search_budget: BudgetPolicy | None = None,
    observation_updates: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Rebuild refs downstream of a lower frozen experiment budget."""

    experiment = graph.store.get_json(graph.experiment_ref, ExperimentManifest)
    experiment = experiment.model_copy(
        update={"search_budget": search_budget or BudgetPolicy(max_evaluations=limit)}
    )
    experiment_ref = put_json(
        graph.store,
        experiment,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    candidate = graph.candidate.model_copy(update={"experiment_ref": experiment_ref})
    candidate_ref = put_json(
        graph.store,
        candidate,
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    admission = CandidateAdmissionService(graph.store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
    )
    admission_report_ref = put_json(
        graph.store,
        admission,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )
    mechanism_evidence_ref = store_attested_mechanism_evidence(
        graph,
        candidate_ref=candidate_ref,
        candidate_harness_ref=candidate.child_harness_ref,
        evidence=graph.mechanism_evidence,
    )
    updates = observation_updates or {}
    parent_observations = tuple(
        observation.model_copy(update=updates) for observation in graph.parent_batch.observations
    )
    parent_batch = reissue_batch(
        graph,
        graph.parent_batch,
        candidate_ref=candidate_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        observations=parent_observations,
    )
    parent_batch_ref = put_json(
        graph.store,
        parent_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    candidate_observations = tuple(
        observation.model_copy(update=updates) for observation in graph.candidate_batch.observations
    )
    candidate_batch = reissue_batch(
        graph,
        graph.candidate_batch,
        candidate_ref=candidate_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        observations=candidate_observations,
    )
    candidate_batch_ref = put_json(
        graph.store,
        candidate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    evaluation = graph.evaluation.model_copy(
        update={
            "candidate_ref": candidate_ref,
            "admission_report_ref": admission_report_ref,
            "parent_batch_ref": parent_batch_ref,
            "candidate_batch_ref": candidate_batch_ref,
            "mechanism_evidence_ref": mechanism_evidence_ref,
        }
    )
    evaluation_ref = put_json(
        graph.store,
        evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    return SimpleNamespace(
        store=graph.store,
        gate_batch_service=graph.gate_batch_service,
        mechanism_evidence_service=graph.mechanism_evidence_service,
        protocol_ref=graph.protocol_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
        admission_report_ref=admission_report_ref,
        mechanism_evidence=graph.mechanism_evidence,
        mechanism_evidence_ref=mechanism_evidence_ref,
        evaluation_ref=evaluation_ref,
    )


def add_sealed_split(
    graph: DecisionGraph,
    *,
    max_evaluations: int | None = None,
) -> SimpleNamespace:
    """Rebind downstream artifacts to a protocol with an immutable sealed split."""

    store = graph.store
    experiment = store.get_json(graph.experiment_ref, ExperimentManifest)
    protocol = store.get_json(experiment.protocol_ref, ProtocolManifest)
    sealed_split_ref = put_json(store, {"tasks": ["sealed-task"]})
    protocol_updates: dict[str, object] = {
        "splits": (
            *protocol.splits,
            ProtocolSplit(
                partition=ProtocolPartition.SEALED,
                manifest_ref=sealed_split_ref,
            ),
        )
    }
    if max_evaluations is not None:
        protocol_updates["budget"] = BudgetPolicy(max_evaluations=max_evaluations)
    protocol = protocol.model_copy(update=protocol_updates)
    protocol_ref = put_json(store, protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment_updates: dict[str, object] = {"protocol_ref": protocol_ref}
    if max_evaluations is not None:
        experiment_updates["search_budget"] = BudgetPolicy(max_evaluations=max_evaluations)
    experiment = experiment.model_copy(update=experiment_updates)
    experiment_ref = put_json(
        store,
        experiment,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    candidate = graph.candidate.model_copy(update={"experiment_ref": experiment_ref})
    candidate_ref = put_json(store, candidate, media_type=CANDIDATE_MANIFEST_MEDIA_TYPE)
    admission = CandidateAdmissionService(store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
    )
    admission_report_ref = put_json(
        store,
        admission,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )
    mechanism_evidence_ref = store_attested_mechanism_evidence(
        graph,
        candidate_ref=candidate_ref,
        candidate_harness_ref=candidate.child_harness_ref,
        evidence=graph.mechanism_evidence,
        protocol_ref=protocol_ref,
        protocol=protocol,
    )
    parent_batch = graph.gate_batch_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.PARENT,
        harness_ref=graph.parent_batch.harness_ref,
        gate_split_ref=graph.gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=graph.parent_batch.source_refs,
        observations=graph.parent_batch.observations,
    )
    parent_batch_ref = put_json(
        store,
        parent_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    candidate_batch = graph.gate_batch_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.CANDIDATE,
        harness_ref=graph.candidate_batch.harness_ref,
        gate_split_ref=graph.gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=graph.candidate_batch.source_refs,
        observations=graph.candidate_batch.observations,
    )
    candidate_batch_ref = put_json(
        store,
        candidate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    evaluation = graph.evaluation.model_copy(
        update={
            "candidate_ref": candidate_ref,
            "admission_report_ref": admission_report_ref,
            "parent_batch_ref": parent_batch_ref,
            "candidate_batch_ref": candidate_batch_ref,
            "mechanism_evidence_ref": mechanism_evidence_ref,
        }
    )
    evaluation_ref = put_json(
        store,
        evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    values = dict(graph.__dict__)
    values.update(
        protocol_ref=protocol_ref,
        experiment_ref=experiment_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        admission_report_ref=admission_report_ref,
        parent_batch=parent_batch,
        parent_batch_ref=parent_batch_ref,
        candidate_batch=candidate_batch,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        evaluation=evaluation,
        evaluation_ref=evaluation_ref,
        sealed_split_ref=sealed_split_ref,
    )
    return SimpleNamespace(**values)


def another_candidate(graph: DecisionGraph) -> SimpleNamespace:
    """Create a second valid atomic child under the same frozen experiment."""

    store = graph.store
    experiment = store.get_json(graph.experiment_ref, ExperimentManifest)
    parent = store.get_json(graph.candidate.parent_harness_ref, HarnessManifest)
    mutation = store.get_json(graph.candidate.mutation_ref, CandidateMutation)
    second_artifact_ref = store.put_bytes(b"second candidate prompt", media_type="text/plain")
    second_component = HarnessComponentRef(
        name=mutation.after.name,
        kind=mutation.after.kind,
        artifact=second_artifact_ref,
    )
    second_mutation = mutation.model_copy(update={"after": second_component})
    mutation_ref = put_json(
        store,
        second_mutation,
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
    )
    child = HarnessRegistry(experiment.mutation_policy).apply_mutation(
        parent=parent,
        parent_ref=graph.candidate.parent_harness_ref,
        mutation=second_mutation,
        artifact_bytes=store.get_bytes(second_artifact_ref),
        artifact_media_type=second_artifact_ref.media_type,
    )
    child_ref = store.put_json(child, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = CandidateManifest(
        experiment_ref=graph.experiment_ref,
        parent_harness_ref=graph.candidate.parent_harness_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
        evidence_refs=graph.candidate.evidence_refs,
        evaluation_plan_ref=graph.candidate.evaluation_plan_ref,
    )
    candidate_ref = put_json(store, candidate, media_type=CANDIDATE_MANIFEST_MEDIA_TYPE)
    admission = CandidateAdmissionService(store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=graph.experiment_ref,
    )
    admission_report_ref = put_json(
        store,
        admission,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )
    parent_observations = tuple(
        observation.model_copy(
            update={
                "execution_fingerprint": (f"second:{observation.task_id}:seed={observation.seed}")
            }
        )
        for observation in graph.parent_trials
    )
    candidate_observations = tuple(
        observation.model_copy(
            update={
                "harness_id": child_ref.sha256,
                "execution_fingerprint": (f"second:{observation.task_id}:seed={observation.seed}"),
            }
        )
        for observation in graph.candidate_trials
    )
    mechanism = MechanismEvidence(
        candidate_harness_id=child_ref.sha256,
        checks=graph.mechanism_evidence.checks,
    )
    protocol = store.get_json(experiment.protocol_ref, ProtocolManifest)
    mechanism_evidence_ref = store_attested_mechanism_evidence(
        graph,
        candidate_ref=candidate_ref,
        candidate_harness_ref=child_ref,
        evidence=mechanism,
        protocol_ref=experiment.protocol_ref,
        protocol=protocol,
    )
    parent_source_ref = put_json(
        store,
        parent_observations,
        media_type="application/vnd.spiral-harness.test-parent-trials+json",
    )
    candidate_source_ref = put_json(
        store,
        candidate_observations,
        media_type="application/vnd.spiral-harness.test-candidate-trials+json",
    )
    parent_batch = graph.gate_batch_service.create(
        protocol_ref=experiment.protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.PARENT,
        harness_ref=graph.candidate.parent_harness_ref,
        gate_split_ref=graph.gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(parent_source_ref,),
        observations=parent_observations,
    )
    parent_batch_ref = put_json(
        store,
        parent_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    candidate_batch = graph.gate_batch_service.create(
        protocol_ref=experiment.protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.CANDIDATE,
        harness_ref=child_ref,
        gate_split_ref=graph.gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(candidate_source_ref,),
        observations=candidate_observations,
    )
    candidate_batch_ref = put_json(
        store,
        candidate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    evaluation = GateEvaluationManifest(
        candidate_ref=candidate_ref,
        admission_report_ref=admission_report_ref,
        gate_config_ref=graph.gate_config_ref,
        gate_split_ref=graph.gate_split_ref,
        gate_implementation_fingerprint=graph.evaluation.gate_implementation_fingerprint,
        parent_batch_ref=parent_batch_ref,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
    )
    evaluation_ref = put_json(
        store,
        evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    return SimpleNamespace(
        store=store,
        gate_batch_service=graph.gate_batch_service,
        mechanism_evidence_service=graph.mechanism_evidence_service,
        protocol_ref=experiment.protocol_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        experiment_ref=graph.experiment_ref,
        admission_report_ref=admission_report_ref,
        parent_trials=parent_observations,
        parent_batch=parent_batch,
        parent_batch_ref=parent_batch_ref,
        candidate_trials=candidate_observations,
        candidate_batch=candidate_batch,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence=mechanism,
        mechanism_evidence_ref=mechanism_evidence_ref,
        evaluation=evaluation,
        evaluation_ref=evaluation_ref,
    )


def test_controller_owns_complete_semantic_path_usage_and_exact_terminal_branch(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    completion = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    usage = controller.current_usage()
    assert usage.query_count == 1
    assert usage.total_evaluations == (
        len(graph.parent_batch.observations) + len(graph.candidate_batch.observations)
    )
    assert usage.remaining_evaluations == 8
    assert usage.candidate_refs == (graph.candidate_ref,)
    assert usage.evaluation_refs == (graph.evaluation_ref,)
    assert controller.query_usage(completion.usage_tail_ref) == usage
    resumed = controller_for(
        graph,
        usage_tail_ref=completion.usage_tail_ref,
    )
    assert resumed.current_usage() == usage

    report, report_ref = store_terminal_report(graph)
    terminal = controller.finalize_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=completion.candidate_tail_ref,
        terminal_decision_report_ref=report_ref,
    )
    authorization = graph.store.get_json(
        terminal.authorization_ref,
        TerminalTransitionAuthorization,
    )
    assert terminal.authorization_ref.media_type == TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE
    assert authorization.evidence_complete_tail_ref == completion.candidate_tail_ref
    assert authorization.evaluation_ref == graph.evaluation_ref
    assert authorization.decision_ref == report.decision_ref
    assert authorization.terminal_state is CandidateState.PROMOTED
    assert controller.verify_terminal_authorization(terminal.authorization_ref) == authorization
    events = CandidateJournal(graph.store).replay(terminal.candidate_tail_ref)
    assert tuple(event.to_state for event in events) == (
        CandidateState.REGISTERED,
        CandidateState.VALID,
        CandidateState.RUNNING_PROBES,
        CandidateState.RUNNING_GATE,
        CandidateState.EVIDENCE_COMPLETE,
        CandidateState.PROMOTED,
    )
    assert terminal.authorization_ref in events[-1].evidence_refs

    forged = authorization.model_copy(
        update={"prior_champion_harness_ref": graph.candidate_ref},
    )
    forged_ref = graph.store.put_json(
        forged,
        media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    )
    with pytest.raises(
        ExperimentControllerError,
        match="was not published by this controller branch",
    ):
        controller.verify_terminal_authorization(forged_ref)


def test_controller_requires_the_protocol_frozen_gate_batch_capability(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)

    with pytest.raises(TypeError, match="gate_batch_verifier"):
        ExperimentController(graph.store, experiment_ref=graph.experiment_ref)
    with pytest.raises(ExperimentControllerError, match="protocol-frozen attestor"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=TrustedGateBatchService().verification_capability,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        )
    with pytest.raises(TypeError, match="mechanism_evidence_verifier"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
        )
    with pytest.raises(ExperimentControllerError, match="protocol-frozen attestor"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(TrustedMechanismEvidenceService().verification_capability),
        )


def test_controller_rejects_verifier_subclasses_that_override_trust_checks(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    forged_gate = ForgedGateVerificationCapability(b"g" * 32)
    forged_mechanism = ForgedMechanismVerificationCapability(b"m" * 32)

    with pytest.raises(TypeError, match="gate_batch_verifier"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=forged_gate,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        )
    with pytest.raises(TypeError, match="mechanism_evidence_verifier"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=forged_mechanism,
        )


def test_controller_fails_closed_on_experiment_lifecycle_resume_even_for_a_genuine_tail(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    writer = controller_for(graph)
    frozen_tail = writer.freeze_experiment()
    searching_tail = writer.start_search(previous_tail_ref=frozen_tail)

    with pytest.raises(ExperimentControllerError, match=r"does not support.*resume"):
        controller_for(graph, experiment_tail_ref=searching_tail)


@pytest.mark.parametrize("attack", ["old-signature", "rogue-signer"])
def test_controller_rejects_forged_promote_batches_before_usage_is_published(
    tmp_path: Path,
    attack: str,
) -> None:
    graph = build_graph(tmp_path, Decision.REJECT)
    rogue = TrustedGateBatchService() if attack == "rogue-signer" else None
    evaluation_ref, _ = store_forged_score_attack(graph, signer=rogue)
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(ExperimentControllerError, match=r"attestation|attestor"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=evaluation_ref,
            previous_usage_tail_ref=None,
        )

    assert controller.usage_tail_ref is None
    assert controller.current_usage().total_evaluations == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"tokens": 999},
        {"latency_ms": 999.0},
        {"tool_calls": 999},
    ],
)
def test_controller_rejects_tampered_resource_usage_before_accounting(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    graph = build_graph(tmp_path)
    observations = tuple(
        observation.model_copy(update=updates) for observation in graph.candidate_batch.observations
    )
    tampered_batch = graph.candidate_batch.model_copy(update={"observations": observations})
    tampered_batch_ref = put_json(
        graph.store,
        tampered_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    evaluation_ref = put_json(
        graph.store,
        graph.evaluation.model_copy(update={"candidate_batch_ref": tampered_batch_ref}),
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(ExperimentControllerError, match="attestation"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=evaluation_ref,
            previous_usage_tail_ref=None,
        )
    assert controller.usage_tail_ref is None


def test_admission_failure_is_controller_authored_and_irreversibly_invalid(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen = controller.freeze_experiment()
    controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    forged_report_ref = graph.admission_report_ref.model_copy(
        update={"media_type": "application/json"}
    )

    invalid_tail = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=forged_report_ref,
    )

    events = CandidateJournal(graph.store).replay(invalid_tail)
    assert events[-1].to_state is CandidateState.INVALID
    assert len(events[-1].evidence_refs) == 1
    failure_ref = events[-1].evidence_refs[0]
    assert failure_ref.media_type == ADMISSION_FAILURE_REPORT_MEDIA_TYPE
    failure = graph.store.get_json(failure_ref, AdmissionFailureReport)
    assert failure.candidate_ref == graph.candidate_ref
    assert failure.experiment_ref == graph.experiment_ref
    assert failure.attempted_admission_report_ref == forged_report_ref
    assert failure.error_code is AdmissionFailureCode.REPORT_REPLAY_FAILED
    with pytest.raises(ExperimentControllerError, match="expected 'valid'"):
        controller.start_probes(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=invalid_tail,
        )


@pytest.mark.parametrize(
    ("checks", "expected_code", "failed", "missing"),
    [
        (
            (MechanismCheck(name="activation", passed=False),),
            ProbeRejectionCode.REQUIRED_CHECK_FAILED,
            ("activation",),
            (),
        ),
        ((), ProbeRejectionCode.REQUIRED_CHECK_MISSING, (), ("activation",)),
    ],
)
def test_probe_failure_or_missing_check_generates_typed_rejection_without_caller_boolean(
    tmp_path: Path,
    checks: tuple[MechanismCheck, ...],
    expected_code: ProbeRejectionCode,
    failed: tuple[str, ...],
    missing: tuple[str, ...],
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen = controller.freeze_experiment()
    controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=graph.admission_report_ref,
    )
    probes = controller.start_probes(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=valid,
    )
    evidence_ref = store_attested_mechanism_evidence(
        graph,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.candidate.child_harness_ref,
        evidence=MechanismEvidence(
            candidate_harness_id=graph.candidate.child_harness_ref.sha256,
            checks=checks,
        ),
    )

    rejected_tail = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=evidence_ref,
    )

    event = CandidateJournal(graph.store).replay(rejected_tail)[-1]
    assert event.to_state is CandidateState.REJECTED
    report_ref = next(
        ref for ref in event.evidence_refs if ref.media_type == PROBE_REJECTION_REPORT_MEDIA_TYPE
    )
    report = graph.store.get_json(report_ref, ProbeRejectionReport)
    assert report.error_code is expected_code
    assert report.failed_checks == failed
    assert report.missing_checks == missing
    assert evidence_ref in event.evidence_refs


@pytest.mark.parametrize(
    ("media_type", "error"),
    [
        ("application/json", "wrong media type"),
        (ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE, "could not load canonical"),
    ],
)
def test_raw_caller_authored_passed_mechanism_evidence_cannot_enter_gate(
    tmp_path: Path,
    media_type: str,
    error: str,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    probes = advance_to_probes(graph, controller)
    raw_evidence_ref = put_json(
        graph.store,
        graph.mechanism_evidence,
        media_type=media_type,
    )

    with pytest.raises(ExperimentControllerError, match=error):
        controller.start_gate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=probes,
            mechanism_evidence_ref=raw_evidence_ref,
        )
    recovered = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=graph.mechanism_evidence_ref,
    )
    assert (
        CandidateJournal(graph.store).replay(recovered)[-1].to_state is CandidateState.RUNNING_GATE
    )


@pytest.mark.parametrize(
    ("attack", "error"),
    [
        ("old-signature", "attestation"),
        ("rogue-signer", "attestor"),
        ("context-substitution", "execution context"),
        ("candidate-substitution", "another candidate"),
    ],
)
def test_mechanism_attestation_and_frozen_context_fail_closed_before_gate(
    tmp_path: Path,
    attack: str,
    error: str,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    probes = advance_to_probes(graph, controller)
    original = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )

    if attack == "old-signature":
        changed_check = original.evidence.checks[0].model_copy(
            update={"details": "caller replaced trusted probe semantics"}
        )
        attacked = original.model_copy(
            update={"evidence": original.evidence.model_copy(update={"checks": (changed_check,)})}
        )
    elif attack == "rogue-signer":
        attacked = TrustedMechanismEvidenceService().issue(original.content)
    elif attack == "context-substitution":
        context = GateBatchExecutionContext.from_protocol(graph.protocol).model_copy(
            update={"runtime_fingerprint": "substituted-runtime"}
        )
        attacked = graph.mechanism_evidence_service.issue(
            original.content.model_copy(update={"execution_context": context})
        )
    else:
        foreign_candidate_ref = put_json(
            graph.store,
            {"candidate": "foreign"},
            media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
        )
        attacked = graph.mechanism_evidence_service.issue(
            original.content.model_copy(update={"candidate_ref": foreign_candidate_ref})
        )
    attacked_ref = put_json(
        graph.store,
        attacked,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )

    with pytest.raises(ExperimentControllerError, match=error):
        controller.start_gate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=probes,
            mechanism_evidence_ref=attacked_ref,
        )
    recovered = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=graph.mechanism_evidence_ref,
    )
    assert (
        CandidateJournal(graph.store).replay(recovered)[-1].to_state is CandidateState.RUNNING_GATE
    )


def test_signed_mechanism_evidence_with_missing_source_fails_before_gate(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    probes = advance_to_probes(graph, controller)
    missing_source_ref = ArtifactRef(
        sha256="0" * 64,
        size=1,
        media_type="application/json",
    )
    evidence = MechanismEvidence(
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
        checks=(
            MechanismCheck(
                name="activation",
                passed=True,
                evidence_refs=(missing_source_ref.sha256,),
            ),
        ),
    )
    envelope = graph.mechanism_evidence_service.create(
        protocol_ref=graph.protocol_ref,
        protocol=graph.protocol,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.candidate.child_harness_ref,
        source_refs=(missing_source_ref,),
        evidence=evidence,
    )
    envelope_ref = put_json(
        graph.store,
        envelope,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )

    with pytest.raises(ExperimentControllerError, match="source artifact"):
        controller.start_gate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=probes,
            mechanism_evidence_ref=envelope_ref,
        )
    recovered = controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=graph.mechanism_evidence_ref,
    )
    assert (
        CandidateJournal(graph.store).replay(recovered)[-1].to_state is CandidateState.RUNNING_GATE
    )


def test_mechanism_signer_rejects_checks_that_cite_unbound_sources(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    original = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    evidence = MechanismEvidence(
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
        checks=(
            MechanismCheck(
                name="activation",
                passed=True,
                evidence_refs=("not-a-bound-source",),
            ),
        ),
    )

    with pytest.raises(ValidationError, match="outside source_refs"):
        graph.mechanism_evidence_service.create(
            protocol_ref=graph.protocol_ref,
            protocol=graph.protocol,
            candidate_ref=graph.candidate_ref,
            candidate_harness_ref=graph.candidate.child_harness_ref,
            source_refs=original.source_refs,
            evidence=evidence,
        )


def test_wrong_candidate_evidence_and_old_candidate_or_usage_tails_fail_closed(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    _, _, probes_tail, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(StaleControllerTailError, match="candidate tail is stale"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=probes_tail,
            evaluation_ref=graph.evaluation_ref,
            previous_usage_tail_ref=None,
        )
    other_candidate_ref = put_json(
        graph.store,
        {"candidate": "foreign"},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(ExperimentControllerError, match="not registered"):
        controller.complete_evidence(
            candidate_ref=other_candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=graph.evaluation_ref,
            previous_usage_tail_ref=None,
        )

    first = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    second = another_candidate(graph)
    _, _, _, second_gate_tail = advance_to_gate(second, controller)
    with pytest.raises(StaleControllerTailError, match="usage tail is stale"):
        controller.complete_evidence(
            candidate_ref=second.candidate_ref,
            previous_tail_ref=second_gate_tail,
            evaluation_ref=second.evaluation_ref,
            previous_usage_tail_ref=None,
        )
    assert controller.usage_tail_ref == first.usage_tail_ref


def test_later_promoted_sibling_becomes_typed_inconclusive_without_champion_deadlock(
    tmp_path: Path,
) -> None:
    graph = add_sealed_split(build_graph(tmp_path), max_evaluations=40)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)
    sibling = another_candidate(graph)

    _, _, _, first_gate_tail = advance_to_gate(graph, controller)
    _, _, _, sibling_gate_tail = advance_to_gate(sibling, controller)
    first_evidence = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=first_gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    sibling_evidence = controller.complete_evidence(
        candidate_ref=sibling.candidate_ref,
        previous_tail_ref=sibling_gate_tail,
        evaluation_ref=sibling.evaluation_ref,
        previous_usage_tail_ref=first_evidence.usage_tail_ref,
    )

    _, first_report_ref = store_terminal_report(graph)
    first_terminal = controller.finalize_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=first_evidence.candidate_tail_ref,
        terminal_decision_report_ref=first_report_ref,
    )
    sibling_decision = PromotionGate(graph.gate_config).evaluate(
        sibling.parent_trials,
        sibling.candidate_trials,
        sibling.mechanism_evidence,
        parent_harness_id=sibling.candidate.parent_harness_ref.sha256,
        candidate_harness_id=sibling.candidate.child_harness_ref.sha256,
    )
    assert sibling_decision.decision is Decision.PROMOTE
    sibling_decision_ref = put_json(graph.store, sibling_decision)
    sibling_service = TerminalDecisionService(
        graph.store,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
    )
    sibling_report = sibling_service.validate(
        candidate_ref=sibling.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=sibling.evaluation_ref,
        decision_ref=sibling_decision_ref,
        terminal_state=CandidateState.PROMOTED,
    )
    sibling_report_ref = put_json(
        graph.store,
        sibling_report,
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )

    sibling_terminal = controller.finalize_candidate(
        candidate_ref=sibling.candidate_ref,
        previous_tail_ref=sibling_evidence.candidate_tail_ref,
        terminal_decision_report_ref=sibling_report_ref,
    )

    sibling_event = CandidateJournal(graph.store).replay(sibling_terminal.candidate_tail_ref)[-1]
    assert sibling_event.to_state is CandidateState.INCONCLUSIVE
    assert sibling_terminal.superseded_report_ref is not None
    assert (
        sibling_terminal.superseded_report_ref.media_type == SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE
    )
    superseded = graph.store.get_json(
        sibling_terminal.superseded_report_ref,
        SupersededCandidateReport,
    )
    assert superseded.evidence_complete_tail_ref == sibling_evidence.candidate_tail_ref
    assert superseded.terminal_decision_report_ref == sibling_report_ref
    assert superseded.decision_ref == sibling_decision_ref
    assert superseded.stale_parent_harness_ref == graph.candidate.parent_harness_ref
    assert superseded.current_champion_harness_ref == graph.candidate.child_harness_ref
    assert superseded.superseding_candidate_ref == graph.candidate_ref
    authorization = graph.store.get_json(
        sibling_terminal.authorization_ref,
        TerminalTransitionAuthorization,
    )
    assert authorization.gate_terminal_state is CandidateState.PROMOTED
    assert authorization.terminal_state is CandidateState.INCONCLUSIVE
    assert authorization.superseded_report_ref == sibling_terminal.superseded_report_ref

    selection_tail = controller.close_selection(
        previous_tail_ref=search_tail,
        previous_usage_tail_ref=sibling_evidence.usage_tail_ref,
        champion_candidate_ref=graph.candidate_ref,
        champion_candidate_tail_ref=first_terminal.candidate_tail_ref,
        champion_harness_ref=graph.candidate.child_harness_ref,
        analysis_plan_ref=put_json(graph.store, {"winner": "first-sibling"}),
    )
    closure_event = ExperimentJournal(graph.store).replay(selection_tail)[-1]
    closure = graph.store.get_json(closure_event.evidence_refs[0], SelectionClosure)
    assert closure.champion_candidate_ref == graph.candidate_ref
    assert closure.champion_harness_ref == graph.candidate.child_harness_ref


def test_budget_is_recomputed_from_both_batches_and_overrun_publishes_no_usage_head(
    tmp_path: Path,
) -> None:
    base = build_graph(tmp_path)
    graph = rebind_experiment_budget(base, limit=10)
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(ExperimentBudgetError, match=r"requested=12, limit=10"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=graph.evaluation_ref,
            previous_usage_tail_ref=None,
        )

    assert controller.usage_tail_ref is None
    assert controller.current_usage().total_evaluations == 0
    assert (
        CandidateJournal(base.store).replay(gate_tail)[-1].to_state is CandidateState.RUNNING_GATE
    )


@pytest.mark.parametrize(
    ("search_budget", "observation_updates", "error"),
    [
        (BudgetPolicy(max_evaluations=20, max_tokens=1), {"tokens": 1}, "max_tokens"),
        (
            BudgetPolicy(max_evaluations=20, max_tool_calls=1),
            {"tool_calls": 1},
            "max_tool_calls",
        ),
        (
            BudgetPolicy(max_evaluations=20, max_wall_time_seconds=0.001),
            {"latency_ms": 1.0},
            "max_wall_time_seconds",
        ),
    ],
)
def test_tokens_tools_and_wall_time_are_recomputed_from_both_batches(
    tmp_path: Path,
    search_budget: BudgetPolicy,
    observation_updates: dict[str, object],
    error: str,
) -> None:
    base = build_graph(tmp_path)
    graph = rebind_experiment_budget(
        base,
        limit=20,
        search_budget=search_budget,
        observation_updates=observation_updates,
    )
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(ExperimentBudgetError, match=error):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=graph.evaluation_ref,
            previous_usage_tail_ref=None,
        )
    assert controller.usage_tail_ref is None


def test_cost_ceiling_fails_closed_when_any_observation_omits_cost(tmp_path: Path) -> None:
    base = build_graph(tmp_path)
    graph = rebind_experiment_budget(
        base,
        limit=20,
        search_budget=BudgetPolicy(max_evaluations=20, max_cost_usd=1.0),
    )
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)

    with pytest.raises(ExperimentBudgetError, match="omit cost_usd"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=graph.evaluation_ref,
            previous_usage_tail_ref=None,
        )
    assert controller.usage_tail_ref is None


def test_evidence_complete_and_terminal_reject_forged_evidence_or_alternate_branch(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)
    wrong_evaluation_ref = put_json(
        graph.store,
        graph.evaluation.model_copy(
            update={
                "mechanism_evidence_ref": put_json(
                    graph.store,
                    {"forged": True},
                    media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
                )
            }
        ),
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(ExperimentControllerError, match="mechanism evidence"):
        controller.complete_evidence(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=gate_tail,
            evaluation_ref=wrong_evaluation_ref,
            previous_usage_tail_ref=None,
        )
    completion = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    report, report_ref = store_terminal_report(graph)
    forged_report_ref = put_json(
        graph.store,
        report.model_copy(update={"decision_ref": put_json(graph.store, {"forged": True})}),
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )
    with pytest.raises(ExperimentControllerError, match="terminal decision replay failed"):
        controller.finalize_candidate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=completion.candidate_tail_ref,
            terminal_decision_report_ref=forged_report_ref,
        )

    alternate_tail = CandidateJournal(graph.store).append(
        stream_id=f"candidate/{graph.candidate_ref.sha256}",
        previous_entry_ref=gate_tail,
        event=CandidateLifecycleEvent(
            candidate_ref=graph.candidate_ref,
            from_state=CandidateState.RUNNING_GATE,
            to_state=CandidateState.EVIDENCE_COMPLETE,
            evidence_refs=(graph.evaluation_ref, completion.usage_tail_ref),
            reason="attacker-created alternate branch",
        ),
    )
    assert alternate_tail != completion.candidate_tail_ref
    with pytest.raises(StaleControllerTailError, match="another branch"):
        controller.finalize_candidate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=alternate_tail,
            terminal_decision_report_ref=report_ref,
        )


def test_usage_replay_rejects_duplicate_candidate_and_evaluation_claim(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    _, _, _, gate_tail = advance_to_gate(graph, controller)
    completion = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    first_entry = graph.store.get_json(completion.usage_tail_ref, ExperimentUsageEntry)
    first_claim = graph.store.get_json(first_entry.claim_ref, ExperimentUsageClaim)
    repeated_claim_ref = put_json(
        graph.store,
        first_claim,
        media_type=first_entry.claim_ref.media_type,
    )
    duplicate_entry = ExperimentUsageEntry(
        experiment_ref=graph.experiment_ref,
        protocol_ref=first_entry.protocol_ref,
        sequence=1,
        claim_ref=repeated_claim_ref,
        cumulative_evaluations=24,
        cumulative_tokens=first_entry.cumulative_tokens * 2,
        cumulative_tool_calls=first_entry.cumulative_tool_calls * 2,
        cumulative_wall_time_seconds=first_entry.cumulative_wall_time_seconds * 2,
        cumulative_cost_usd=(
            None if first_entry.cumulative_cost_usd is None else first_entry.cumulative_cost_usd * 2
        ),
        previous_entry_ref=completion.usage_tail_ref,
    )
    duplicate_tail = put_json(
        graph.store,
        duplicate_entry,
        media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    )

    with pytest.raises(ExperimentControllerError, match="charged more than once"):
        ExperimentController(
            graph.store,
            experiment_ref=graph.experiment_ref,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
            usage_tail_ref=duplicate_tail,
        )


def test_complete_experiment_lifecycle_freezes_champion_usage_sealed_split_and_report(
    tmp_path: Path,
) -> None:
    graph = add_sealed_split(build_graph(tmp_path))
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)
    _, _, _, gate_tail = advance_to_gate(graph, controller)
    evidence = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    report, report_ref = store_terminal_report(graph)
    candidate_terminal = controller.finalize_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=evidence.candidate_tail_ref,
        terminal_decision_report_ref=report_ref,
    )
    assert report.terminal_state is CandidateState.PROMOTED
    analysis_plan_ref = put_json(
        graph.store,
        {"metric": "paired-score", "sealed_queries": 1},
    )

    selection_tail = controller.close_current_selection(
        previous_tail_ref=search_tail,
        analysis_plan_ref=analysis_plan_ref,
    )
    selection_event = ExperimentJournal(graph.store).replay(selection_tail)[-1]
    closure_ref = selection_event.evidence_refs[0]
    assert closure_ref.media_type == SELECTION_CLOSURE_MEDIA_TYPE
    closure = graph.store.get_json(closure_ref, SelectionClosure)
    assert closure.champion_candidate_ref == graph.candidate_ref
    assert closure.champion_harness_ref == graph.candidate.child_harness_ref
    assert closure.champion_candidate_tail_ref == candidate_terminal.candidate_tail_ref
    assert closure.analysis_plan_ref == analysis_plan_ref
    assert closure.usage_tail_ref == evidence.usage_tail_ref
    assert controller.verify_experiment_selection_closure(selection_tail) == closure

    sealed_tail = controller.start_sealed(previous_tail_ref=selection_tail)
    with pytest.raises(StaleControllerTailError, match="stale"):
        controller.verify_experiment_selection_closure(selection_tail)
    sealed_event = ExperimentJournal(graph.store).replay(sealed_tail)[-1]
    authorization_ref = sealed_event.evidence_refs[0]
    assert authorization_ref.media_type == SEALED_RUN_AUTHORIZATION_MEDIA_TYPE
    authorization = graph.store.get_json(authorization_ref, SealedRunAuthorization)
    assert authorization.selection_closed_tail_ref == selection_tail
    assert authorization.sealed_split_ref == graph.sealed_split_ref
    assert authorization.usage_tail_ref == evidence.usage_tail_ref

    protocol = graph.store.get_json(graph.protocol_ref, ProtocolManifest)
    result_ref = put_json(graph.store, {"sealed_score": 0.8, "status": "complete"})
    sealed_evidence_ref = put_json(graph.store, {"sealed_trials": ["sealed-task"]})
    with pytest.raises(ExperimentControllerError, match="wrong media type"):
        controller.complete_experiment(
            previous_tail_ref=sealed_tail,
            sealed_evaluation_report_ref=result_ref,
        )
    sealed_report = SealedEvaluationReport(
        experiment_ref=graph.experiment_ref,
        protocol_ref=graph.protocol_ref,
        sealed_running_tail_ref=sealed_tail,
        sealed_authorization_ref=authorization_ref,
        selection_closure_ref=authorization.selection_closure_ref,
        champion_candidate_ref=authorization.champion_candidate_ref,
        champion_harness_ref=authorization.champion_harness_ref,
        analysis_plan_ref=authorization.analysis_plan_ref,
        sealed_split_ref=authorization.sealed_split_ref,
        usage_tail_ref=authorization.usage_tail_ref,
        model_fingerprint=protocol.model_fingerprint,
        inference_fingerprint=protocol.inference_fingerprint,
        runtime_fingerprint=protocol.runtime_fingerprint,
        sandbox_fingerprint=protocol.sandbox_fingerprint,
        grader_fingerprint=protocol.grader_fingerprint,
        capability_policy_ref=protocol.capability_policy_ref,
        result_ref=result_ref,
        evidence_refs=(sealed_evidence_ref,),
    )
    sealed_report_ref = put_json(
        graph.store,
        sealed_report,
        media_type=SEALED_EVALUATION_REPORT_MEDIA_TYPE,
    )
    complete_tail = controller.complete_experiment(
        previous_tail_ref=sealed_tail,
        sealed_evaluation_report_ref=sealed_report_ref,
    )
    events = ExperimentJournal(graph.store).replay(complete_tail)
    assert tuple(event.to_state for event in events) == (
        ExperimentState.FROZEN,
        ExperimentState.SEARCHING,
        ExperimentState.SELECTION_CLOSED,
        ExperimentState.SEALED_RUNNING,
        ExperimentState.COMPLETE,
    )
    completion_ref = next(
        ref
        for ref in events[-1].evidence_refs
        if ref.media_type == EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE
    )
    completion = graph.store.get_json(completion_ref, ExperimentCompletionReport)
    assert completion.sealed_running_tail_ref == sealed_tail
    assert completion.sealed_evaluation_report_ref == sealed_report_ref
    assert completion.usage_tail_ref == evidence.usage_tail_ref
    assert controller.experiment_state is ExperimentState.COMPLETE
    with pytest.raises(ExperimentControllerError, match="requires searching"):
        controller.register_candidate(candidate_ref=graph.candidate_ref)


def test_sealed_start_rejects_protocol_without_sealed_partition(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)
    _, _, _, gate_tail = advance_to_gate(graph, controller)
    evidence = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    _, report_ref = store_terminal_report(graph)
    candidate_terminal = controller.finalize_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=evidence.candidate_tail_ref,
        terminal_decision_report_ref=report_ref,
    )
    selection_tail = controller.close_selection(
        previous_tail_ref=search_tail,
        previous_usage_tail_ref=evidence.usage_tail_ref,
        champion_candidate_ref=graph.candidate_ref,
        champion_candidate_tail_ref=candidate_terminal.candidate_tail_ref,
        champion_harness_ref=graph.candidate.child_harness_ref,
        analysis_plan_ref=put_json(graph.store, {"metric": "sealed"}),
    )

    with pytest.raises(ExperimentControllerError, match="SEALED protocol split"):
        controller.start_sealed(previous_tail_ref=selection_tail)
    assert controller.experiment_state is ExperimentState.SELECTION_CLOSED


def test_selection_keeps_seed_when_no_candidate_was_promoted(tmp_path: Path) -> None:
    graph = build_graph(tmp_path, Decision.REJECT)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)
    _, _, _, gate_tail = advance_to_gate(graph, controller)
    evidence = controller.complete_evidence(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=gate_tail,
        evaluation_ref=graph.evaluation_ref,
        previous_usage_tail_ref=None,
    )
    report = graph.service.validate(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=graph.evaluation_ref,
        decision_ref=graph.decision_ref,
        terminal_state=CandidateState.REJECTED,
    )
    report_ref = put_json(
        graph.store,
        report,
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )
    rejected = controller.finalize_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=evidence.candidate_tail_ref,
        terminal_decision_report_ref=report_ref,
    )

    selection_tail = controller.close_selection(
        previous_tail_ref=search_tail,
        previous_usage_tail_ref=evidence.usage_tail_ref,
        champion_candidate_ref=None,
        champion_candidate_tail_ref=None,
        champion_harness_ref=graph.candidate.parent_harness_ref,
        analysis_plan_ref=put_json(graph.store, {"fallback": "seed"}),
    )
    event = ExperimentJournal(graph.store).replay(selection_tail)[-1]
    closure = graph.store.get_json(event.evidence_refs[0], SelectionClosure)
    assert closure.champion_candidate_ref is None
    assert closure.champion_candidate_tail_ref is None
    assert closure.champion_harness_ref == graph.candidate.parent_harness_ref
    assert closure.selection_reason is SelectionReason.NO_PROMOTABLE_CANDIDATES
    assert rejected.candidate_tail_ref is not None


def test_selection_can_keep_seed_without_issuing_any_gate_query(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)

    selection_tail = controller.close_selection(
        previous_tail_ref=search_tail,
        previous_usage_tail_ref=None,
        champion_candidate_ref=None,
        champion_candidate_tail_ref=None,
        champion_harness_ref=graph.candidate.parent_harness_ref,
        analysis_plan_ref=put_json(graph.store, {"fallback": "seed", "queries": 0}),
    )

    event = ExperimentJournal(graph.store).replay(selection_tail)[-1]
    closure = graph.store.get_json(event.evidence_refs[0], SelectionClosure)
    assert closure.selection_reason is SelectionReason.NO_PROMOTABLE_CANDIDATES
    assert closure.usage_tail_ref is None
    assert event.usage_tail_ref is None


def test_selection_cannot_close_while_any_registered_candidate_is_nonterminal(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    search_tail = controller.start_search(previous_tail_ref=frozen_tail)
    controller.register_candidate(candidate_ref=graph.candidate_ref)

    with pytest.raises(ExperimentControllerError, match="candidates remain nonterminal"):
        controller.close_selection(
            previous_tail_ref=search_tail,
            previous_usage_tail_ref=None,
            champion_candidate_ref=None,
            champion_candidate_tail_ref=None,
            champion_harness_ref=graph.candidate.parent_harness_ref,
            analysis_plan_ref=put_json(graph.store, {"fallback": "seed"}),
        )


@pytest.mark.parametrize("start_searching", [False, True])
def test_controller_authored_integrity_or_leakage_report_invalidates_nonterminal_lineage(
    tmp_path: Path,
    start_searching: bool,
) -> None:
    graph = build_graph(tmp_path)
    controller = controller_for(graph)
    frozen_tail = controller.freeze_experiment()
    source_tail = (
        controller.start_search(previous_tail_ref=frozen_tail) if start_searching else frozen_tail
    )
    expected_source = ExperimentState.SEARCHING if start_searching else ExperimentState.FROZEN
    evidence_ref = put_json(graph.store, {"violation": "private-task-read"})

    invalidated_tail = controller.invalidate_experiment(
        previous_tail_ref=source_tail,
        violation_code=ExperimentViolationCode.LEAKAGE,
        evidence_refs=(evidence_ref,),
        message="sealed task contents were observed before selection closed",
    )

    event = ExperimentJournal(graph.store).replay(invalidated_tail)[-1]
    assert event.to_state is ExperimentState.INVALIDATED
    report_ref = event.evidence_refs[0]
    assert report_ref.media_type == EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE
    report = graph.store.get_json(report_ref, ExperimentInvalidationReport)
    assert report.source_tail_ref == source_tail
    assert report.source_state is expected_source
    assert report.violation_code is ExperimentViolationCode.LEAKAGE
    assert report.evidence_refs == (evidence_ref,)
    with pytest.raises(StaleControllerTailError, match="experiment tail is stale"):
        controller.invalidate_experiment(
            previous_tail_ref=source_tail,
            violation_code=ExperimentViolationCode.INTEGRITY,
            evidence_refs=(evidence_ref,),
            message="attempted second terminal branch",
        )
