from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_sha256
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
    AdmissionReport,
    CandidateAdmissionError,
    CandidateAdmissionService,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.models import GateConfig

_PROTOCOL_MEDIA_TYPE = PROTOCOL_MANIFEST_MEDIA_TYPE
_EXPERIMENT_MEDIA_TYPE = EXPERIMENT_MANIFEST_MEDIA_TYPE
_MUTATION_MEDIA_TYPE = CANDIDATE_MUTATION_MEDIA_TYPE
_CANDIDATE_MEDIA_TYPE = CANDIDATE_MANIFEST_MEDIA_TYPE
_GATE_MEDIA_TYPE = "application/vnd.spiral-harness.gate-config+json"


@dataclass(frozen=True)
class AdmissionFixture:
    store: ArtifactStore
    policy: MutationPolicy
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    experiment: ExperimentManifest
    experiment_ref: ArtifactRef
    parent: HarnessManifest
    parent_ref: ArtifactRef
    child: HarnessManifest
    child_ref: ArtifactRef
    before: HarnessComponentRef
    after: HarnessComponentRef
    mutation: CandidateMutation
    mutation_ref: ArtifactRef
    evidence_ref: ArtifactRef
    gate_ref: ArtifactRef
    capability_policy_ref: ArtifactRef
    candidate: CandidateManifest
    candidate_ref: ArtifactRef


def _replace[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    content = model.model_dump(
        mode="python",
        by_alias=False,
        exclude_none=False,
        round_trip=True,
        warnings="none",
    )
    content.update(updates)
    return type(model).model_validate(content, strict=True, by_name=True)


def _candidate_ref(
    fixture: AdmissionFixture,
    *,
    candidate: CandidateManifest | None = None,
    **updates: object,
) -> ArtifactRef:
    value = candidate or _replace(fixture.candidate, **updates)
    return fixture.store.put_json(value, media_type=_CANDIDATE_MEDIA_TYPE)


def _experiment_candidate_refs(
    fixture: AdmissionFixture,
    *,
    experiment: ExperimentManifest,
) -> tuple[ArtifactRef, ArtifactRef]:
    experiment_ref = fixture.store.put_json(experiment, media_type=_EXPERIMENT_MEDIA_TYPE)
    candidate_ref = _candidate_ref(fixture, experiment_ref=experiment_ref)
    return experiment_ref, candidate_ref


def _fixture(root: Path) -> AdmissionFixture:
    store = ArtifactStore(root)
    gate_ref = store.put_json(
        GateConfig(min_tasks=1, bootstrap_samples=1_000),
        media_type=_GATE_MEDIA_TYPE,
    )
    capability_policy_ref = store.put_json(
        CapabilityPolicy(),
        media_type=CAPABILITY_POLICY_MEDIA_TYPE,
    )
    exploration_ref = store.put_json(
        {"task_ids": ["explore-1"]},
        media_type="application/vnd.spiral-harness.split+json",
    )
    gate_split_ref = store.put_json(
        {"task_ids": ["gate-1"]},
        media_type="application/vnd.spiral-harness.split+json",
    )
    protocol = ProtocolManifest(
        benchmark_fingerprint="benchmark-v1",
        splits=(
            ProtocolSplit(
                partition=ProtocolPartition.EXPLORATION,
                manifest_ref=exploration_ref,
            ),
            ProtocolSplit(
                partition=ProtocolPartition.GATE,
                manifest_ref=gate_split_ref,
            ),
        ),
        model_fingerprint="model-v1",
        inference_fingerprint="inference-v1",
        runtime_fingerprint="runtime-v1",
        model_spec_fingerprint="9" * 64,
        sandbox_fingerprint="sandbox-v1",
        capability_policy_ref=capability_policy_ref,
        grader_fingerprint="grader-v1",
        gate_batch_attestor_id="f" * 64,
        mechanism_evidence_attestor_id="e" * 64,
        gate_config_ref=gate_ref,
        trusted_plane_version="trusted-v1",
        budget=BudgetPolicy(max_evaluations=8),
    )
    protocol_ref = store.put_json(protocol, media_type=_PROTOCOL_MEDIA_TYPE)

    before_artifact = store.put_bytes(b"old prompt", media_type="text/plain")
    after_artifact = store.put_bytes(b"new prompt", media_type="text/plain")
    before = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=before_artifact,
    )
    after = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=after_artifact,
    )
    parent = HarnessManifest(
        model_fingerprint=protocol.model_fingerprint,
        runtime_fingerprint=protocol.runtime_fingerprint,
        trusted_plane_version=protocol.trusted_plane_version,
        components=(before,),
        budget=BudgetPolicy(max_evaluations=4),
    )
    parent_ref = store.put_json(parent, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    evidence_ref = store.put_json(
        {"failure": "old prompt fails the probe"},
        media_type="application/vnd.spiral-harness.evidence+json",
    )
    mutation = CandidateMutation(
        target_component="system",
        before=before,
        after=after,
        hypothesis=MutationHypothesis(
            evidence_refs=(evidence_ref,),
            where="system prompt",
            why="the old instruction is incomplete",
            expected_activation="the new prompt is loaded",
            expected_adherence="the new rule is followed",
            expected_behavior="the probe passes",
            expected_benefit="paired score improves",
            protected_slices=("protected",),
            falsifier="the probe still fails",
            negative_control="unaffected tasks remain unchanged",
            risks=("over-specific instruction",),
        ),
    )
    mutation_ref = store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    policy = MutationPolicy(
        allowed_component_names=("system",),
        allowed_media_types=("text/plain",),
        max_artifact_size_bytes=1_024,
    )
    experiment = ExperimentManifest(
        protocol_ref=protocol_ref,
        seed_harness_ref=parent_ref,
        mutation_policy=policy,
        objective="improve paired score",
        baselines=("static",),
        stopping=("one-candidate",),
        search_budget=BudgetPolicy(max_evaluations=8),
    )
    experiment_ref = store.put_json(experiment, media_type=_EXPERIMENT_MEDIA_TYPE)
    child = HarnessRegistry(policy).apply_mutation(
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(after_artifact),
        artifact_media_type=after_artifact.media_type,
    )
    child_ref = store.put_json(child, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = CandidateManifest(
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
        evidence_refs=(evidence_ref,),
        evaluation_plan_ref=gate_ref,
    )
    candidate_ref = store.put_json(candidate, media_type=_CANDIDATE_MEDIA_TYPE)
    return AdmissionFixture(
        store=store,
        policy=policy,
        protocol=protocol,
        protocol_ref=protocol_ref,
        experiment=experiment,
        experiment_ref=experiment_ref,
        parent=parent,
        parent_ref=parent_ref,
        child=child,
        child_ref=child_ref,
        before=before,
        after=after,
        mutation=mutation,
        mutation_ref=mutation_ref,
        evidence_ref=evidence_ref,
        gate_ref=gate_ref,
        capability_policy_ref=capability_policy_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
    )


def test_admission_reaches_seed_and_joins_current_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = CandidateAdmissionService(fixture.store).admit(
        candidate_ref=fixture.candidate_ref,
        experiment_ref=fixture.experiment_ref,
    )

    assert report.admitted is True
    assert report.candidate_ref == fixture.candidate_ref
    assert report.protocol_ref == fixture.protocol_ref
    assert report.parent_harness_ref == fixture.parent_ref
    assert report.child_harness_ref == fixture.child_ref
    assert report.mutation_ref == fixture.mutation_ref
    assert report.evaluation_plan_ref == fixture.gate_ref
    assert report.gate_config_ref == fixture.gate_ref
    assert report.capability_policy_ref == fixture.capability_policy_ref
    assert report.evidence_refs == (fixture.evidence_ref,)
    assert report.mutation_policy_sha256 == canonical_sha256(fixture.policy)
    assert report.component_contract == "prompt-atomic-replacement-v1"
    assert report.checks == (
        "canonical_artifacts_verified",
        "candidate_experiment_joined",
        "protocol_seed_planes_matched",
        "frozen_policy_applied",
        "ancestry_reaches_seed",
        "current_mutation_recomputed",
        "component_contract_verified",
        "evidence_joined",
        "evaluation_plan_joined",
        "capability_policy_joined",
    )


def test_admission_rejects_v3_protocol_payload_labeled_as_v1_media(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mislabeled_ref = fixture.store.put_json(
        fixture.protocol,
        media_type="application/vnd.spiral-harness.protocol-manifest.v1+json",
    )
    experiment_payload = fixture.experiment.model_dump(mode="python")
    experiment_payload["protocol_ref"] = mislabeled_ref.model_dump(mode="python")
    experiment_ref = fixture.store.put_json(
        experiment_payload,
        media_type=_EXPERIMENT_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(fixture, experiment_ref=experiment_ref)

    with pytest.raises(CandidateAdmissionError, match="experiment artifact could not be verified"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


def test_admission_rejects_candidate_payload_under_generic_json_media(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mislabeled_ref = fixture.store.put_json(
        fixture.candidate,
        media_type="application/json",
    )

    with pytest.raises(CandidateAdmissionError, match=r"candidate artifact.*wrong media type"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=mislabeled_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_candidate_from_another_experiment(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other_experiment = _replace(fixture.experiment, objective="a different experiment")
    other_experiment_ref = fixture.store.put_json(
        other_experiment,
        media_type=_EXPERIMENT_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(fixture, experiment_ref=other_experiment_ref)

    with pytest.raises(CandidateAdmissionError, match="does not match the frozen experiment"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_parent_that_is_not_the_experiment_seed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other_parent = _replace(fixture.parent, budget=BudgetPolicy(max_evaluations=3))
    other_parent_ref = fixture.store.put_json(
        other_parent,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(fixture, parent_harness_ref=other_parent_ref)

    with pytest.raises(CandidateAdmissionError, match="experiment seed harness"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_accepts_descendant_parent_with_intact_lineage_to_seed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    third_artifact = fixture.store.put_bytes(b"third prompt", media_type="text/plain")
    third_component = _replace(fixture.after, artifact=third_artifact)
    mutation = _replace(
        fixture.mutation,
        before=fixture.after,
        after=third_component,
    )
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    grandchild = HarnessRegistry(fixture.policy).apply_mutation(
        parent=fixture.child,
        parent_ref=fixture.child_ref,
        mutation=mutation,
        artifact_bytes=fixture.store.get_bytes(third_artifact),
        artifact_media_type=third_artifact.media_type,
    )
    grandchild_ref = fixture.store.put_json(
        grandchild,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate = _replace(
        fixture.candidate,
        parent_harness_ref=fixture.child_ref,
        child_harness_ref=grandchild_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = _candidate_ref(fixture, candidate=candidate)

    report = CandidateAdmissionService(fixture.store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=fixture.experiment_ref,
    )

    assert report.parent_harness_ref == fixture.child_ref
    assert report.child_harness_ref == grandchild_ref


def test_admission_rejects_descendant_with_missing_seed_lineage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_ancestor = ArtifactRef(
        sha256="e" * 64,
        size=10,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    disconnected_parent = _replace(fixture.parent, parent=missing_ancestor)
    disconnected_parent_ref = fixture.store.put_json(
        disconnected_parent,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(
        fixture,
        parent_harness_ref=disconnected_parent_ref,
    )

    with pytest.raises(CandidateAdmissionError, match="parent lineage harness"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


@pytest.mark.parametrize(
    "field_name",
    ["model_fingerprint", "runtime_fingerprint", "trusted_plane_version"],
)
def test_admission_rejects_seed_protocol_plane_mismatch(
    tmp_path: Path,
    field_name: str,
) -> None:
    fixture = _fixture(tmp_path)
    protocol = _replace(fixture.protocol, **{field_name: f"different-{field_name}"})
    protocol_ref = fixture.store.put_json(protocol, media_type=_PROTOCOL_MEDIA_TYPE)
    experiment = _replace(fixture.experiment, protocol_ref=protocol_ref)
    experiment_ref, candidate_ref = _experiment_candidate_refs(
        fixture,
        experiment=experiment,
    )

    with pytest.raises(CandidateAdmissionError, match=field_name):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


def test_admission_applies_the_policy_frozen_in_the_experiment(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    forbidden_policy = _replace(
        fixture.policy,
        allowed_component_names=("different-component",),
    )
    experiment = _replace(fixture.experiment, mutation_policy=forbidden_policy)
    experiment_ref, candidate_ref = _experiment_candidate_refs(
        fixture,
        experiment=experiment,
    )

    with pytest.raises(CandidateAdmissionError, match="frozen experiment policy"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


@pytest.mark.parametrize(
    "kind",
    [
        ComponentKind.MEMORY,
        ComponentKind.TOOL,
        ComponentKind.MIDDLEWARE,
        ComponentKind.CONTROL_FLOW,
    ],
)
def test_admission_rejects_component_without_a_semantic_contract(
    tmp_path: Path,
    kind: ComponentKind,
) -> None:
    fixture = _fixture(tmp_path)
    media_type = "application/octet-stream"
    before = _replace(
        fixture.before,
        kind=kind,
        artifact=fixture.store.put_bytes(b"old state", media_type=media_type),
    )
    after = _replace(
        fixture.after,
        kind=kind,
        artifact=fixture.store.put_bytes(b"new state", media_type=media_type),
    )
    parent = _replace(fixture.parent, components=(before,))
    parent_ref = fixture.store.put_json(parent, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    policy = MutationPolicy(
        allowed_kinds=(kind,),
        allowed_component_names=(before.name,),
        allowed_media_types=(media_type,),
        max_artifact_size_bytes=1_024,
    )
    experiment = _replace(
        fixture.experiment,
        seed_harness_ref=parent_ref,
        mutation_policy=policy,
    )
    experiment_ref = fixture.store.put_json(experiment, media_type=_EXPERIMENT_MEDIA_TYPE)
    mutation = _replace(fixture.mutation, before=before, after=after)
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    child = HarnessRegistry(policy).apply_mutation(
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        artifact_bytes=fixture.store.get_bytes(after.artifact),
        artifact_media_type=media_type,
    )
    child_ref = fixture.store.put_json(child, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        fixture.candidate,
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = fixture.store.put_json(candidate, media_type=_CANDIDATE_MEDIA_TYPE)

    with pytest.raises(CandidateAdmissionError, match=f"no semantic.*{kind.value}"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


def test_admission_rejects_search_budget_above_protocol_ceiling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    experiment = _replace(
        fixture.experiment,
        search_budget=BudgetPolicy(max_evaluations=9),
    )
    experiment_ref, candidate_ref = _experiment_candidate_refs(
        fixture,
        experiment=experiment,
    )

    with pytest.raises(CandidateAdmissionError, match="max_evaluations"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


def test_admission_rejects_mutation_before_that_is_not_the_parent_component(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    wrong_before = _replace(
        fixture.before,
        artifact=fixture.store.put_bytes(b"unrelated prompt", media_type="text/plain"),
    )
    mutation = _replace(fixture.mutation, before=wrong_before)
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    candidate_ref = _candidate_ref(fixture, mutation_ref=mutation_ref)

    with pytest.raises(CandidateAdmissionError, match="before component"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_missing_before_artifact_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_before_ref = ArtifactRef(
        sha256="d" * 64,
        size=10,
        media_type="text/plain",
    )
    missing_before = _replace(fixture.before, artifact=missing_before_ref)
    parent = _replace(fixture.parent, components=(missing_before,))
    parent_ref = fixture.store.put_json(parent, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    experiment = _replace(fixture.experiment, seed_harness_ref=parent_ref)
    experiment_ref = fixture.store.put_json(experiment, media_type=_EXPERIMENT_MEDIA_TYPE)
    mutation = _replace(fixture.mutation, before=missing_before)
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    candidate = _replace(
        fixture.candidate,
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = _candidate_ref(fixture, candidate=candidate)

    with pytest.raises(CandidateAdmissionError, match="before artifact"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )


def test_admission_rejects_child_not_recomputed_from_the_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_child = _replace(
        fixture.parent,
        parent=fixture.parent_ref,
    )
    wrong_child_ref = fixture.store.put_json(
        wrong_child,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(fixture, child_harness_ref=wrong_child_ref)

    with pytest.raises(CandidateAdmissionError, match="recomputed from parent and mutation"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_evidence_not_bound_to_the_hypothesis(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other_evidence_ref = fixture.store.put_json(
        {"failure": "different evidence"},
        media_type="application/vnd.spiral-harness.evidence+json",
    )
    candidate_ref = _candidate_ref(fixture, evidence_refs=(other_evidence_ref,))

    with pytest.raises(CandidateAdmissionError, match="hypothesis evidence"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_missing_evidence_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_ref = ArtifactRef(
        sha256="f" * 64,
        size=10,
        media_type="application/vnd.spiral-harness.evidence+json",
    )
    hypothesis = _replace(fixture.mutation.hypothesis, evidence_refs=(missing_ref,))
    mutation = _replace(fixture.mutation, hypothesis=hypothesis)
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    candidate_ref = _candidate_ref(
        fixture,
        mutation_ref=mutation_ref,
        evidence_refs=(missing_ref,),
    )

    with pytest.raises(CandidateAdmissionError, match="could not be verified"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_evaluation_plan_not_frozen_by_protocol(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other_gate_ref = fixture.store.put_json(
        GateConfig(min_tasks=2, bootstrap_samples=1_000),
        media_type=_GATE_MEDIA_TYPE,
    )
    candidate_ref = _candidate_ref(fixture, evaluation_plan_ref=other_gate_ref)

    with pytest.raises(CandidateAdmissionError, match="protocol gate_config_ref"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_admission_rejects_noncanonical_typed_artifact_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_payload = fixture.candidate.model_dump(
        mode="json",
        by_alias=False,
        exclude_none=False,
    )
    candidate_payload.pop("schema_version")
    candidate_ref = fixture.store.put_json(candidate_payload, media_type=_CANDIDATE_MEDIA_TYPE)

    with pytest.raises(CandidateAdmissionError, match="not canonical"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_persisted_admission_report_is_reverified_for_the_exact_candidate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    service = CandidateAdmissionService(fixture.store)
    report = service.admit(
        candidate_ref=fixture.candidate_ref,
        experiment_ref=fixture.experiment_ref,
    )
    report_ref = fixture.store.put_json(report, media_type=ADMISSION_REPORT_MEDIA_TYPE)

    assert (
        service.verify_report(
            candidate_ref=fixture.candidate_ref,
            experiment_ref=fixture.experiment_ref,
            report_ref=report_ref,
        )
        == report
    )

    other_candidate_ref = fixture.store.put_json(
        {"candidate": "other"},
        media_type=_CANDIDATE_MEDIA_TYPE,
    )
    with pytest.raises(CandidateAdmissionError, match="different candidate"):
        service.verify_report(
            candidate_ref=other_candidate_ref,
            experiment_ref=fixture.experiment_ref,
            report_ref=report_ref,
        )


def test_persisted_admission_report_must_equal_the_recomputed_proof(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    service = CandidateAdmissionService(fixture.store)
    report = service.admit(
        candidate_ref=fixture.candidate_ref,
        experiment_ref=fixture.experiment_ref,
    )
    forged_report: AdmissionReport = _replace(
        report,
        mutation_policy_sha256="0" * 64,
    )
    forged_report_ref = fixture.store.put_json(
        forged_report,
        media_type=ADMISSION_REPORT_MEDIA_TYPE,
    )

    with pytest.raises(CandidateAdmissionError, match="recomputed proof"):
        service.verify_report(
            candidate_ref=fixture.candidate_ref,
            experiment_ref=fixture.experiment_ref,
            report_ref=forged_report_ref,
        )
