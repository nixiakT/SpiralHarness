from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT,
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
from spiral_harness.experiments.decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    GateEvaluationManifest,
    TerminalDecisionError,
    TerminalDecisionReport,
    TerminalDecisionService,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateBatchVerificationCapability,
    GateTrialArm,
    GateTrialBatch,
    GateTrialBatchContent,
    TrustedGateBatchService,
)
from spiral_harness.verification.gate import PromotionGate
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    AttestedMechanismEvidence,
    MechanismEvidenceVerificationCapability,
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

_STATE_BY_DECISION = {
    Decision.PROMOTE: CandidateState.PROMOTED,
    Decision.REJECT: CandidateState.REJECTED,
    Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
}


class ForgedGateVerificationCapability(GateBatchVerificationCapability):
    """A subclass can override the trust checks an ``isinstance`` guard assumes."""

    @property
    def attestor_id(self) -> str:
        return "0" * 64

    def verify(self, batch: object) -> object:
        return batch


class ForgedMechanismVerificationCapability(MechanismEvidenceVerificationCapability):
    """Adversarial verifier subclass rejected before any asserted ID is read."""

    @property
    def attestor_id(self) -> str:
        return "0" * 64

    def verify(self, value: object) -> object:
        return value


@dataclass(frozen=True)
class DecisionGraph:
    store: ArtifactStore
    gate_batch_service: TrustedGateBatchService
    mechanism_evidence_service: TrustedMechanismEvidenceService
    service: TerminalDecisionService
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    candidate: CandidateManifest
    candidate_ref: ArtifactRef
    experiment_ref: ArtifactRef
    gate_config: GateConfig
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    parent_trials: tuple[TrialObservation, ...]
    parent_batch: GateTrialBatch
    parent_batch_ref: ArtifactRef
    candidate_trials: tuple[TrialObservation, ...]
    candidate_batch: GateTrialBatch
    candidate_batch_ref: ArtifactRef
    mechanism_evidence: MechanismEvidence
    mechanism_evidence_ref: ArtifactRef
    evaluation: GateEvaluationManifest
    evaluation_ref: ArtifactRef
    decision: GateDecision
    decision_ref: ArtifactRef


class SubstitutingRepository:
    """Repository double whose typed decoder returns bytes it did not decode."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        target_ref: ArtifactRef,
        replacement: object,
    ) -> None:
        self.store = store
        self.target_ref = target_ref
        self.replacement = replacement

    def get_bytes(self, ref_or_digest: ArtifactRef | str) -> bytes:
        return self.store.get_bytes(ref_or_digest)

    def get_json(
        self,
        ref_or_digest: ArtifactRef | str,
        model_type: type[Any] | None = None,
    ) -> Any:
        if ref_or_digest == self.target_ref:
            return self.replacement
        return self.store.get_json(ref_or_digest, model_type)


def put_json(
    store: ArtifactStore,
    value: object,
    *,
    media_type: str = "application/json",
) -> ArtifactRef:
    return store.put_json(value, media_type=media_type)


def _deltas_for(decision: Decision) -> tuple[float, ...]:
    if decision is Decision.PROMOTE:
        return (0.1,) * 6
    if decision is Decision.REJECT:
        return (-0.1,) * 6
    return (0.2, -0.1, 0.2, -0.1, 0.2, -0.1)


def build_graph(
    tmp_path: Path,
    expected_decision: Decision = Decision.PROMOTE,
    *,
    gate_implementation_fingerprint: str = PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT,
) -> DecisionGraph:
    store = ArtifactStore(tmp_path / "artifacts")
    gate_batch_service = TrustedGateBatchService()
    mechanism_evidence_service = TrustedMechanismEvidenceService()
    exploration_ref = put_json(store, {"tasks": ["exploration"]})
    gate_split_ref = put_json(store, {"tasks": ["gate"]})
    gate_config = GateConfig(
        version="gate-v1",
        min_tasks=5,
        min_effect=0.04,
        bootstrap_samples=1_000,
        bootstrap_seed=123,
        required_mechanism_checks=("activation",),
    )
    gate_config_ref = put_json(store, gate_config)
    capability_policy_ref = put_json(
        store,
        CapabilityPolicy(),
        media_type=CAPABILITY_POLICY_MEDIA_TYPE,
    )

    protocol = ProtocolManifest(
        benchmark_fingerprint="benchmark-v1",
        splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=exploration_ref,
            ),
            ProtocolSplit(partition=ProtocolPartition.GATE, manifest_ref=gate_split_ref),
        ),
        model_fingerprint="model-v1",
        inference_fingerprint="inference-v1",
        runtime_fingerprint="runtime-v1",
        model_spec_fingerprint="9" * 64,
        sandbox_fingerprint="sandbox-v1",
        capability_policy_ref=capability_policy_ref,
        grader_fingerprint="grader-v1",
        gate_batch_attestor_id=gate_batch_service.attestor_id,
        mechanism_evidence_attestor_id=mechanism_evidence_service.attestor_id,
        gate_config_ref=gate_config_ref,
        gate_implementation_fingerprint=gate_implementation_fingerprint,
        trusted_plane_version="trusted-plane-v1",
        budget=BudgetPolicy(max_evaluations=20),
    )
    protocol_ref = put_json(store, protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)
    before_artifact_ref = store.put_bytes(b"old prompt", media_type="text/plain")
    after_artifact_ref = store.put_bytes(b"new prompt", media_type="text/plain")
    before = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=before_artifact_ref,
    )
    after = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=after_artifact_ref,
    )
    parent_harness = HarnessManifest(
        model_fingerprint=protocol.model_fingerprint,
        runtime_fingerprint=protocol.runtime_fingerprint,
        trusted_plane_version=protocol.trusted_plane_version,
        components=(before,),
        budget=BudgetPolicy(max_evaluations=10),
    )
    parent_harness_ref = put_json(
        store,
        parent_harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    diagnostic_evidence_ref = put_json(store, {"evidence": "diagnostic"})
    mutation = CandidateMutation(
        target_component="system",
        before=before,
        after=after,
        hypothesis=MutationHypothesis(
            evidence_refs=(diagnostic_evidence_ref,),
            where="system prompt",
            why="the old prompt omits the required check",
            expected_activation="the new prompt is loaded",
            expected_adherence="the new instruction is followed",
            expected_behavior="the verification probe passes",
            expected_benefit="paired gate score improves",
            protected_slices=("protected",),
            falsifier="the verification probe still fails",
            negative_control="unaffected tasks remain unchanged",
            risks=("over-specific instruction",),
        ),
    )
    mutation_ref = put_json(store, mutation, media_type=CANDIDATE_MUTATION_MEDIA_TYPE)
    mutation_policy = MutationPolicy(
        allowed_component_names=("system",),
        allowed_media_types=("text/plain",),
        max_artifact_size_bytes=1_024,
    )
    experiment = ExperimentManifest(
        protocol_ref=protocol_ref,
        seed_harness_ref=parent_harness_ref,
        mutation_policy=mutation_policy,
        objective="maximize gate score",
        baselines=("static",),
        stopping=("budget exhausted",),
        search_budget=BudgetPolicy(max_evaluations=20),
    )
    experiment_ref = put_json(
        store,
        experiment,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    child_harness = HarnessRegistry(mutation_policy).apply_mutation(
        parent=parent_harness,
        parent_ref=parent_harness_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(after_artifact_ref),
        artifact_media_type=after_artifact_ref.media_type,
    )
    child_harness_ref = put_json(
        store,
        child_harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate = CandidateManifest(
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_harness_ref,
        child_harness_ref=child_harness_ref,
        mutation_ref=mutation_ref,
        evidence_refs=(diagnostic_evidence_ref,),
        evaluation_plan_ref=gate_config_ref,
    )
    candidate_ref = put_json(store, candidate, media_type=CANDIDATE_MANIFEST_MEDIA_TYPE)
    admission_report = CandidateAdmissionService(store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
    )
    admission_report_ref = put_json(
        store,
        admission_report,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )

    parent_trials: list[TrialObservation] = []
    candidate_trials: list[TrialObservation] = []
    for index, delta in enumerate(_deltas_for(expected_decision)):
        task_id = f"task-{index:02d}"
        common = {
            "task_id": task_id,
            "seed": 11,
            "execution_fingerprint": f"runtime-v1:{task_id}:seed=11",
        }
        parent_trials.append(
            TrialObservation(
                harness_id=parent_harness_ref.sha256,
                score=0.5,
                **common,
            )
        )
        candidate_trials.append(
            TrialObservation(
                harness_id=child_harness_ref.sha256,
                score=0.5 + delta,
                **common,
            )
        )
    frozen_parent_trials = tuple(parent_trials)
    frozen_candidate_trials = tuple(candidate_trials)
    mechanism_source_ref = put_json(store, {"probe": "trusted activation trace"})
    mechanism_evidence = MechanismEvidence(
        candidate_harness_id=child_harness_ref.sha256,
        checks=(
            MechanismCheck(
                name="activation",
                passed=True,
                evidence_refs=(mechanism_source_ref.sha256,),
            ),
        ),
    )
    attested_mechanism_evidence = mechanism_evidence_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        candidate_harness_ref=child_harness_ref,
        source_refs=(mechanism_source_ref,),
        evidence=mechanism_evidence,
    )
    mechanism_evidence_ref = put_json(
        store,
        attested_mechanism_evidence,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    parent_source_ref = put_json(
        store,
        frozen_parent_trials,
        media_type="application/vnd.spiral-harness.test-parent-trials+json",
    )
    candidate_source_ref = put_json(
        store,
        frozen_candidate_trials,
        media_type="application/vnd.spiral-harness.test-candidate-trials+json",
    )
    parent_batch = gate_batch_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.PARENT,
        harness_ref=parent_harness_ref,
        gate_split_ref=gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(parent_source_ref,),
        observations=frozen_parent_trials,
    )
    parent_batch_ref = put_json(
        store,
        parent_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    candidate_batch = gate_batch_service.create(
        protocol_ref=protocol_ref,
        protocol=protocol,
        candidate_ref=candidate_ref,
        arm=GateTrialArm.CANDIDATE,
        harness_ref=child_harness_ref,
        gate_split_ref=gate_split_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
        source_refs=(candidate_source_ref,),
        observations=frozen_candidate_trials,
    )
    candidate_batch_ref = put_json(
        store,
        candidate_batch,
        media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
    )
    evaluation = GateEvaluationManifest(
        candidate_ref=candidate_ref,
        admission_report_ref=admission_report_ref,
        gate_config_ref=gate_config_ref,
        gate_split_ref=gate_split_ref,
        gate_implementation_fingerprint=gate_implementation_fingerprint,
        parent_batch_ref=parent_batch_ref,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence_ref=mechanism_evidence_ref,
    )
    evaluation_ref = put_json(
        store,
        evaluation,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    decision = PromotionGate(gate_config).evaluate(
        frozen_parent_trials,
        frozen_candidate_trials,
        mechanism_evidence,
        parent_harness_id=parent_harness_ref.sha256,
        candidate_harness_id=child_harness_ref.sha256,
    )
    assert decision.decision is expected_decision
    decision_ref = put_json(store, decision)
    return DecisionGraph(
        store=store,
        gate_batch_service=gate_batch_service,
        service=TerminalDecisionService(
            store,
            gate_batch_verifier=gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(mechanism_evidence_service.verification_capability),
        ),
        mechanism_evidence_service=mechanism_evidence_service,
        protocol=protocol,
        protocol_ref=protocol_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
        gate_config=gate_config,
        gate_config_ref=gate_config_ref,
        gate_split_ref=gate_split_ref,
        admission_report_ref=admission_report_ref,
        parent_trials=frozen_parent_trials,
        parent_batch=parent_batch,
        parent_batch_ref=parent_batch_ref,
        candidate_trials=frozen_candidate_trials,
        candidate_batch=candidate_batch,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence=mechanism_evidence,
        mechanism_evidence_ref=mechanism_evidence_ref,
        evaluation=evaluation,
        evaluation_ref=evaluation_ref,
        decision=decision,
        decision_ref=decision_ref,
    )


def store_evaluation(graph: DecisionGraph, **updates: object) -> ArtifactRef:
    return put_json(
        graph.store,
        graph.evaluation.model_copy(update=updates),
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )


def store_batch(
    graph: DecisionGraph,
    batch: object,
    *,
    media_type: str = GATE_TRIAL_BATCH_MEDIA_TYPE,
) -> ArtifactRef:
    return put_json(graph.store, batch, media_type=media_type)


def reissue_batch(
    graph: DecisionGraph,
    batch: GateTrialBatch,
    **updates: object,
) -> GateTrialBatch:
    """Apply trusted-fixture updates and issue a fresh valid attestation."""

    return graph.gate_batch_service.issue(batch.content.model_copy(update=updates))


def store_forged_score_attack(
    graph: DecisionGraph,
    *,
    signer: TrustedGateBatchService | None,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Persist a matching forged PROMOTE from observations that originally reject."""

    parent_trials = tuple(
        observation.model_copy(update={"score": 0.0}) for observation in graph.parent_trials
    )
    candidate_trials = tuple(
        observation.model_copy(update={"score": 1.0}) for observation in graph.candidate_trials
    )
    if signer is None:
        parent_batch = graph.parent_batch.model_copy(update={"observations": parent_trials})
        candidate_batch = graph.candidate_batch.model_copy(
            update={"observations": candidate_trials}
        )
    else:
        parent_batch = signer.issue(
            graph.parent_batch.content.model_copy(update={"observations": parent_trials})
        )
        candidate_batch = signer.issue(
            graph.candidate_batch.content.model_copy(update={"observations": candidate_trials})
        )
    parent_batch_ref = store_batch(graph, parent_batch)
    candidate_batch_ref = store_batch(graph, candidate_batch)
    evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=parent_batch_ref,
        candidate_batch_ref=candidate_batch_ref,
    )
    forged_decision = PromotionGate(graph.gate_config).evaluate(
        parent_trials,
        candidate_trials,
        graph.mechanism_evidence,
        parent_harness_id=graph.candidate.parent_harness_ref.sha256,
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
    )
    assert graph.decision.decision is Decision.REJECT
    assert forged_decision.decision is Decision.PROMOTE
    return evaluation_ref, put_json(graph.store, forged_decision)


@pytest.mark.parametrize(
    ("decision", "terminal_state"),
    [
        (Decision.PROMOTE, CandidateState.PROMOTED),
        (Decision.REJECT, CandidateState.REJECTED),
        (Decision.INCONCLUSIVE, CandidateState.INCONCLUSIVE),
    ],
)
def test_validate_recomputes_real_gate_outcomes_and_verify_replays_report(
    tmp_path: Path,
    decision: Decision,
    terminal_state: CandidateState,
) -> None:
    graph = build_graph(tmp_path, decision)

    report = graph.service.validate(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=graph.evaluation_ref,
        decision_ref=graph.decision_ref,
        terminal_state=terminal_state,
    )
    report_ref = put_json(
        graph.store,
        report,
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )

    assert report.evaluation_ref == graph.evaluation_ref
    assert report.admission_report_ref == graph.admission_report_ref
    assert report.gate_config_ref == graph.gate_config_ref
    assert report.gate_split_ref == graph.gate_split_ref
    assert report.gate_implementation_fingerprint == PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT
    assert report.gate_batch_attestor_id == graph.gate_batch_service.attestor_id
    assert report.decision is decision
    assert report.terminal_state is terminal_state
    assert (
        graph.service.verify_report(
            report_ref,
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )
        == report
    )


def test_gate_trial_batch_binds_attestor_protocol_context_sources_and_observations(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)

    assert graph.parent_batch.schema_version == "2"
    assert graph.parent_batch.protocol_ref == graph.protocol_ref
    assert graph.parent_batch.candidate_ref == graph.candidate_ref
    assert graph.parent_batch.arm is GateTrialArm.PARENT
    assert graph.parent_batch.harness_ref == graph.candidate.parent_harness_ref
    assert graph.parent_batch.gate_split_ref == graph.gate_split_ref
    assert graph.parent_batch.task_set_fingerprint == graph.gate_split_ref.sha256
    assert graph.parent_batch.mechanism_evidence_ref == graph.mechanism_evidence_ref
    assert graph.parent_batch.execution_context.grader_fingerprint == "grader-v1"
    assert graph.parent_batch.source_refs
    assert graph.parent_batch.observations == graph.parent_trials
    assert graph.parent_batch.attestor_id == graph.gate_batch_service.attestor_id
    assert (
        graph.gate_batch_service.verification_capability.verify(graph.parent_batch)
        == graph.parent_batch
    )
    assert graph.parent_batch_ref.media_type == GATE_TRIAL_BATCH_MEDIA_TYPE


def test_gate_trial_batch_rejects_empty_duplicate_or_wrong_harness_observations(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    values = graph.parent_batch.content.model_dump(mode="python", exclude={"observations"})

    with pytest.raises(ValidationError):
        GateTrialBatchContent(**values, observations=())
    with pytest.raises(ValidationError, match="unique task_id/seed pairs"):
        GateTrialBatchContent(
            **values,
            observations=(graph.parent_trials[0], graph.parent_trials[0]),
        )
    with pytest.raises(ValidationError, match="must belong to harness_ref"):
        GateTrialBatchContent(
            **values,
            observations=(
                graph.parent_trials[0].model_copy(update={"harness_id": "forged-harness"}),
            ),
        )


def test_raw_trial_tuple_cannot_masquerade_as_a_gate_batch(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    raw_tuple_ref = store_batch(graph, graph.parent_trials)
    evaluation_ref = store_evaluation(graph, parent_batch_ref=raw_tuple_ref)

    with pytest.raises(TerminalDecisionError, match="parent gate trial batch"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_gate_batch_and_terminal_services_require_explicit_capabilities(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    unsigned = graph.parent_batch.content.model_dump(mode="python")

    with pytest.raises(ValidationError, match=r"attestor_id|attestation_sha256"):
        GateTrialBatch.model_validate(unsigned)
    with pytest.raises(TypeError, match="gate_batch_verifier"):
        TerminalDecisionService(graph.store)
    with pytest.raises(TypeError, match="mechanism_evidence_verifier"):
        TerminalDecisionService(
            graph.store,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
        )


def test_terminal_service_rejects_verifier_subclasses_that_override_trust_checks(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    forged_gate = ForgedGateVerificationCapability(b"g" * 32)
    forged_mechanism = ForgedMechanismVerificationCapability(b"m" * 32)

    with pytest.raises(TypeError, match="gate_batch_verifier"):
        TerminalDecisionService(
            graph.store,
            gate_batch_verifier=forged_gate,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        )
    with pytest.raises(TypeError, match="mechanism_evidence_verifier"):
        TerminalDecisionService(
            graph.store,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=forged_mechanism,
        )


@pytest.mark.parametrize("attack", ["old-signature", "rogue-signer"])
def test_forged_scores_and_matching_promote_decision_fail_attestation(
    tmp_path: Path,
    attack: str,
) -> None:
    graph = build_graph(tmp_path, Decision.REJECT)
    rogue = TrustedGateBatchService() if attack == "rogue-signer" else None
    evaluation_ref, decision_ref = store_forged_score_attack(graph, signer=rogue)

    with pytest.raises(TerminalDecisionError, match=r"attestation|attestor"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

    if rogue is not None:
        rogue_verifier = TerminalDecisionService(
            graph.store,
            gate_batch_verifier=rogue.verification_capability,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        )
        with pytest.raises(TerminalDecisionError, match="protocol-frozen attestor"):
            rogue_verifier.validate(
                candidate_ref=graph.candidate_ref,
                experiment_ref=graph.experiment_ref,
                evaluation_ref=evaluation_ref,
                decision_ref=decision_ref,
                terminal_state=CandidateState.PROMOTED,
            )


def test_signed_execution_context_cannot_override_the_frozen_protocol(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    attacker_context = graph.parent_batch.execution_context.model_copy(
        update={"grader_fingerprint": "attacker-grader"}
    )
    batch_ref = store_batch(
        graph,
        reissue_batch(graph, graph.parent_batch, execution_context=attacker_context),
    )
    evaluation_ref = store_evaluation(graph, parent_batch_ref=batch_ref)

    with pytest.raises(TerminalDecisionError, match="execution context"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_parent_and_candidate_gate_batch_refs_cannot_be_swapped(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=graph.candidate_batch_ref,
        candidate_batch_ref=graph.parent_batch_ref,
    )

    with pytest.raises(TerminalDecisionError, match=r"arm 'candidate'.*expected 'parent'"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_gate_batches_cannot_cross_candidate_or_gate_split_boundaries(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    another_candidate_ref = put_json(
        graph.store,
        {"candidate": "other"},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    cross_candidate_ref = store_batch(
        graph,
        reissue_batch(graph, graph.parent_batch, candidate_ref=another_candidate_ref),
    )
    cross_candidate_evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=cross_candidate_ref,
    )
    with pytest.raises(TerminalDecisionError, match="batch belongs to another candidate"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=cross_candidate_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

    another_split_ref = put_json(graph.store, {"tasks": ["another-gate"]})
    wrong_split_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.parent_batch,
            gate_split_ref=another_split_ref,
            task_set_fingerprint=another_split_ref.sha256,
        ),
    )
    wrong_split_evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=wrong_split_batch_ref,
    )
    with pytest.raises(TerminalDecisionError, match="gate split"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=wrong_split_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_gate_batch_harness_ref_must_match_the_frozen_candidate_arm(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    another_harness_ref = put_json(
        graph.store,
        {"harness": "other"},
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    observations = tuple(
        observation.model_copy(update={"harness_id": another_harness_ref.sha256})
        for observation in graph.parent_trials
    )
    wrong_harness_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.parent_batch,
            harness_ref=another_harness_ref,
            observations=observations,
        ),
    )
    evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=wrong_harness_batch_ref,
    )

    with pytest.raises(TerminalDecisionError, match=r"parent.*wrong harness"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_gate_batch_refs_require_the_exact_media_type(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    payload = graph.evaluation.model_dump(mode="json")
    payload["parent_batch_ref"]["media_type"] = "application/json"
    evaluation_ref = put_json(
        graph.store,
        payload,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )

    with pytest.raises(TerminalDecisionError, match="exact gate trial batch media type"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_gate_implementation_is_bound_to_protocol_and_running_verifier(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    mismatched_evaluation_ref = store_evaluation(
        graph,
        gate_implementation_fingerprint="attacker-selected-gate",
    )
    with pytest.raises(TerminalDecisionError, match="does not match the protocol implementation"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=mismatched_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

    unsupported = build_graph(
        tmp_path / "unsupported",
        gate_implementation_fingerprint="untrusted-gate-implementation",
    )
    with pytest.raises(TerminalDecisionError, match="not supported by this verifier"):
        unsupported.service.validate(
            candidate_ref=unsupported.candidate_ref,
            experiment_ref=unsupported.experiment_ref,
            evaluation_ref=unsupported.evaluation_ref,
            decision_ref=unsupported.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_hand_forged_promote_cannot_override_recomputed_reject(tmp_path: Path) -> None:
    graph = build_graph(tmp_path, Decision.REJECT)
    forged_ref = put_json(
        graph.store,
        graph.decision.model_copy(update={"decision": Decision.PROMOTE}),
    )

    with pytest.raises(TerminalDecisionError, match="trusted recomputation"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=forged_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_persisted_decision_must_match_every_recomputed_field(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    tampered_ref = put_json(
        graph.store,
        graph.decision.model_copy(update={"reasons": ("rewritten after evaluation",)}),
    )

    with pytest.raises(TerminalDecisionError, match="exactly match trusted recomputation"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=tampered_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_evaluation_candidate_and_config_are_joined_to_frozen_lineage(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    another_candidate_ref = put_json(
        graph.store,
        {"candidate": "other"},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    wrong_candidate_evaluation_ref = store_evaluation(
        graph,
        candidate_ref=another_candidate_ref,
    )

    with pytest.raises(TerminalDecisionError, match="another candidate"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=wrong_candidate_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

    another_config_ref = put_json(graph.store, GateConfig(version="other-gate"))
    wrong_config_evaluation_ref = store_evaluation(
        graph,
        gate_config_ref=another_config_ref,
    )
    with pytest.raises(TerminalDecisionError, match="config does not match"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=wrong_config_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

    another_gate_split_ref = put_json(graph.store, {"tasks": ["another-gate"]})
    wrong_split_evaluation_ref = store_evaluation(
        graph,
        gate_split_ref=another_gate_split_ref,
    )
    with pytest.raises(TerminalDecisionError, match="protocol gate split"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=wrong_split_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_terminal_validation_requires_the_caller_frozen_experiment(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    experiment = graph.store.get_json(graph.experiment_ref, ExperimentManifest)
    another_experiment_ref = put_json(
        graph.store,
        experiment.model_copy(update={"objective": "attacker-selected objective"}),
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )

    with pytest.raises(TerminalDecisionError, match="caller-frozen experiment"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=another_experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_evaluation_requires_a_verified_admission_report(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    wrong_media_admission_ref = graph.admission_report_ref.model_copy(
        update={"media_type": "application/json"}
    )
    evaluation_ref = store_evaluation(
        graph,
        admission_report_ref=wrong_media_admission_ref,
    )

    with pytest.raises(TerminalDecisionError, match=r"admission report.*wrong media type"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_candidate_with_an_unadmitted_evaluation_plan_is_rejected(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    other_plan_ref = put_json(graph.store, GateConfig(version="other-gate"))
    candidate_ref = put_json(
        graph.store,
        graph.candidate.model_copy(update={"evaluation_plan_ref": other_plan_ref}),
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    evaluation_ref = store_evaluation(graph, candidate_ref=candidate_ref)

    with pytest.raises(TerminalDecisionError, match="admission report"):
        graph.service.validate(
            candidate_ref=candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_mechanism_evidence_from_another_candidate_changes_recomputed_result(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    wrong_evidence_ref = put_json(
        graph.store,
        graph.mechanism_evidence.model_copy(update={"candidate_harness_id": "another-candidate"}),
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    evaluation_ref = store_evaluation(
        graph,
        mechanism_evidence_ref=wrong_evidence_ref,
    )

    with pytest.raises(TerminalDecisionError, match="mechanism evidence"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_required_mechanism_check_cannot_be_swapped_after_batch_attestation(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    failed_evidence = MechanismEvidence(
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
        checks=(MechanismCheck(name="activation", passed=False),),
    )
    original_envelope = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    failed_envelope = graph.mechanism_evidence_service.create(
        protocol_ref=graph.protocol_ref,
        protocol=graph.protocol,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.candidate.child_harness_ref,
        source_refs=original_envelope.source_refs,
        evidence=failed_evidence,
    )
    failed_evidence_ref = put_json(
        graph.store,
        failed_envelope,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    parent_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.parent_batch,
            mechanism_evidence_ref=failed_evidence_ref,
        ),
    )
    candidate_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.candidate_batch,
            mechanism_evidence_ref=failed_evidence_ref,
        ),
    )
    rejected = PromotionGate(graph.gate_config).evaluate(
        graph.parent_trials,
        graph.candidate_trials,
        failed_evidence,
        parent_harness_id=graph.candidate.parent_harness_ref.sha256,
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
    )
    assert rejected.decision is Decision.REJECT

    # The attacker swaps in the original passing evidence and a matching
    # PROMOTE decision, but both signed batches still bind the failed evidence.
    evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=parent_batch_ref,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence_ref=graph.mechanism_evidence_ref,
    )
    with pytest.raises(TerminalDecisionError, match="mechanism evidence"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


@pytest.mark.parametrize("attack", ["old-signature", "rogue-signer"])
def test_terminal_replay_rejects_forged_mechanism_evidence_even_when_gate_batches_bind_it(
    tmp_path: Path,
    attack: str,
) -> None:
    graph = build_graph(tmp_path)
    original = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    if attack == "old-signature":
        changed = original.evidence.checks[0].model_copy(
            update={"details": "optimizer-authored passed result"}
        )
        attacked = original.model_copy(
            update={"evidence": original.evidence.model_copy(update={"checks": (changed,)})}
        )
    else:
        attacked = TrustedMechanismEvidenceService().issue(original.content)
    attacked_ref = put_json(
        graph.store,
        attacked,
        media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    )
    parent_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.parent_batch,
            mechanism_evidence_ref=attacked_ref,
        ),
    )
    candidate_batch_ref = store_batch(
        graph,
        reissue_batch(
            graph,
            graph.candidate_batch,
            mechanism_evidence_ref=attacked_ref,
        ),
    )
    evaluation_ref = store_evaluation(
        graph,
        parent_batch_ref=parent_batch_ref,
        candidate_batch_ref=candidate_batch_ref,
        mechanism_evidence_ref=attacked_ref,
    )

    with pytest.raises(TerminalDecisionError, match=r"attestation|attestor"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


@pytest.mark.parametrize("state", [CandidateState.EVIDENCE_COMPLETE, CandidateState.INVALID])
def test_validate_rejects_non_gate_terminal_states(tmp_path: Path, state: CandidateState) -> None:
    graph = build_graph(tmp_path)

    with pytest.raises(TerminalDecisionError, match="requires promoted, rejected, or inconclusive"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=state,
        )


def test_recomputed_decision_authorizes_only_its_matching_terminal_state(tmp_path: Path) -> None:
    graph = build_graph(tmp_path, Decision.INCONCLUSIVE)

    with pytest.raises(TerminalDecisionError, match="requires terminal state"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.REJECTED,
        )


def test_evaluation_and_report_require_exact_media_types(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    wrong_evaluation_ref = graph.evaluation_ref.model_copy(
        update={"media_type": "application/json"}
    )
    with pytest.raises(TerminalDecisionError, match=r"evaluation manifest.*wrong media type"):
        graph.service.validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=wrong_evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )

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
    wrong_report_ref = report_ref.model_copy(update={"media_type": "application/json"})
    with pytest.raises(TerminalDecisionError, match=r"report.*wrong media type"):
        graph.service.verify_report(
            wrong_report_ref,
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "candidate_ref",
        "admission_report_ref",
        "gate_config_ref",
        "gate_split_ref",
        "mechanism_evidence_ref",
    ],
)
def test_evaluation_manifest_requires_json_input_refs(tmp_path: Path, field_name: str) -> None:
    graph = build_graph(tmp_path)
    bad_ref = getattr(graph.evaluation, field_name).model_copy(update={"media_type": "text/plain"})
    unchecked = graph.evaluation.model_copy(update={field_name: bad_ref})

    with pytest.raises(ValidationError, match=field_name):
        GateEvaluationManifest.model_validate(unchecked)


def test_verify_report_binds_caller_supplied_candidate_experiment_and_evaluation(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
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
    another_candidate_ref = put_json(
        graph.store,
        {"candidate": "other"},
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(TerminalDecisionError, match="another candidate"):
        graph.service.verify_report(
            report_ref,
            candidate_ref=another_candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )

    another_experiment_ref = put_json(
        graph.store,
        {"experiment": "other"},
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(TerminalDecisionError, match="another experiment"):
        graph.service.verify_report(
            report_ref,
            candidate_ref=graph.candidate_ref,
            experiment_ref=another_experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )

    another_evaluation_ref = ArtifactRef(
        sha256="0" * 64,
        size=0,
        media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(TerminalDecisionError, match="another evaluation"):
        graph.service.verify_report(
            report_ref,
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=another_evaluation_ref,
        )


def test_verify_report_rejects_forged_report_fields(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    report = graph.service.validate(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=graph.evaluation_ref,
        decision_ref=graph.decision_ref,
        terminal_state=CandidateState.PROMOTED,
    )
    forged_report_ref = put_json(
        graph.store,
        report.model_copy(update={"decision": Decision.REJECT}),
        media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    )

    with pytest.raises(TerminalDecisionError, match="does not match replayed checks"):
        graph.service.verify_report(
            forged_report_ref,
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )


def test_service_does_not_trust_repository_typed_decode_without_bytes_match(
    tmp_path: Path,
) -> None:
    graph = build_graph(tmp_path)
    repository = SubstitutingRepository(
        graph.store,
        target_ref=graph.parent_batch_ref,
        replacement=graph.parent_batch.model_copy(
            update={"observations": graph.parent_trials[:-1]}
        ),
    )

    with pytest.raises(TerminalDecisionError, match="typed representation is not canonical"):
        TerminalDecisionService(  # type: ignore[arg-type]
            repository,
            gate_batch_verifier=graph.gate_batch_service.verification_capability,
            mechanism_evidence_verifier=(graph.mechanism_evidence_service.verification_capability),
        ).validate(
            candidate_ref=graph.candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
            decision_ref=graph.decision_ref,
            terminal_state=CandidateState.PROMOTED,
        )


def test_terminal_report_model_records_the_evaluation_reference(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    report = graph.service.validate(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
        evaluation_ref=graph.evaluation_ref,
        decision_ref=graph.decision_ref,
        terminal_state=_STATE_BY_DECISION[graph.decision.decision],
    )

    assert isinstance(report, TerminalDecisionReport)
    assert report.evaluation_ref == graph.evaluation_ref
    assert report.mechanism_evidence_ref == graph.mechanism_evidence_ref
    assert report.mechanism_evidence_attestor_id == graph.mechanism_evidence_service.attestor_id
    assert "mechanism_evidence_attestation_verified" in report.checks
