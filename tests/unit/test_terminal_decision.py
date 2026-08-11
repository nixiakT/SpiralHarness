from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spiral_harness.core import (
    ArtifactRef,
    BudgetPolicy,
    CandidateManifest,
    CandidateMutation,
    CandidateState,
    ComponentKind,
    ExperimentManifest,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
    MutationPolicy,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.experiments import (
    ADMISSION_REPORT_MEDIA_TYPE,
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    CandidateAdmissionService,
    GateEvaluationManifest,
    TerminalDecisionError,
    TerminalDecisionReport,
    TerminalDecisionService,
)
from spiral_harness.harness import HarnessRegistry
from spiral_harness.storage import ArtifactStore
from spiral_harness.verification import (
    Decision,
    GateConfig,
    GateDecision,
    MechanismCheck,
    MechanismEvidence,
    PromotionGate,
    TrialObservation,
)

_STATE_BY_DECISION = {
    Decision.PROMOTE: CandidateState.PROMOTED,
    Decision.REJECT: CandidateState.REJECTED,
    Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
}


@dataclass(frozen=True)
class DecisionGraph:
    store: ArtifactStore
    service: TerminalDecisionService
    candidate: CandidateManifest
    candidate_ref: ArtifactRef
    experiment_ref: ArtifactRef
    gate_config: GateConfig
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    parent_trials: tuple[TrialObservation, ...]
    parent_trials_ref: ArtifactRef
    candidate_trials: tuple[TrialObservation, ...]
    candidate_trials_ref: ArtifactRef
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


def build_graph(tmp_path: Path, expected_decision: Decision = Decision.PROMOTE) -> DecisionGraph:
    store = ArtifactStore(tmp_path / "artifacts")
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
        sandbox_fingerprint="sandbox-v1",
        grader_fingerprint="grader-v1",
        gate_config_ref=gate_config_ref,
        trusted_plane_version="trusted-plane-v1",
        budget=BudgetPolicy(max_evaluations=20),
    )
    protocol_ref = put_json(store, protocol)
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
    parent_harness_ref = put_json(store, parent_harness)
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
    mutation_ref = put_json(store, mutation)
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
        search_budget=BudgetPolicy(max_evaluations=10),
    )
    experiment_ref = put_json(store, experiment)
    child_harness = HarnessRegistry(mutation_policy).apply_mutation(
        parent=parent_harness,
        parent_ref=parent_harness_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(after_artifact_ref),
        artifact_media_type=after_artifact_ref.media_type,
    )
    child_harness_ref = put_json(store, child_harness)
    candidate = CandidateManifest(
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_harness_ref,
        child_harness_ref=child_harness_ref,
        mutation_ref=mutation_ref,
        evidence_refs=(diagnostic_evidence_ref,),
        evaluation_plan_ref=gate_config_ref,
    )
    candidate_ref = put_json(store, candidate)
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
    parent_trials_ref = put_json(store, frozen_parent_trials)
    candidate_trials_ref = put_json(store, frozen_candidate_trials)
    mechanism_evidence = MechanismEvidence(
        candidate_harness_id=child_harness_ref.sha256,
        checks=(
            MechanismCheck(
                name="activation",
                passed=True,
                evidence_refs=("trajectory-span:activation",),
            ),
        ),
    )
    mechanism_evidence_ref = put_json(store, mechanism_evidence)
    evaluation = GateEvaluationManifest(
        candidate_ref=candidate_ref,
        admission_report_ref=admission_report_ref,
        gate_config_ref=gate_config_ref,
        gate_split_ref=gate_split_ref,
        parent_trials_ref=parent_trials_ref,
        candidate_trials_ref=candidate_trials_ref,
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
        service=TerminalDecisionService(store),
        candidate=candidate,
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
        gate_config=gate_config,
        gate_config_ref=gate_config_ref,
        gate_split_ref=gate_split_ref,
        admission_report_ref=admission_report_ref,
        parent_trials=frozen_parent_trials,
        parent_trials_ref=parent_trials_ref,
        candidate_trials=frozen_candidate_trials,
        candidate_trials_ref=candidate_trials_ref,
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
    another_candidate_ref = put_json(graph.store, {"candidate": "other"})
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
    )
    evaluation_ref = store_evaluation(
        graph,
        mechanism_evidence_ref=wrong_evidence_ref,
    )

    with pytest.raises(TerminalDecisionError, match="trusted recomputation"):
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
        "parent_trials_ref",
        "candidate_trials_ref",
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
    another_candidate_ref = put_json(graph.store, {"candidate": "other"})
    with pytest.raises(TerminalDecisionError, match="another candidate"):
        graph.service.verify_report(
            report_ref,
            candidate_ref=another_candidate_ref,
            experiment_ref=graph.experiment_ref,
            evaluation_ref=graph.evaluation_ref,
        )

    another_experiment_ref = put_json(graph.store, {"experiment": "other"})
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
        target_ref=graph.parent_trials_ref,
        replacement=graph.parent_trials[:-1],
    )

    with pytest.raises(TerminalDecisionError, match="typed representation is not canonical"):
        TerminalDecisionService(repository).validate(  # type: ignore[arg-type]
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
