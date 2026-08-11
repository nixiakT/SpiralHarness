from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

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
from spiral_harness.experiments.admission import CandidateAdmissionError, CandidateAdmissionService
from spiral_harness.harness.registry import HarnessRegistry, HarnessRegistryError
from spiral_harness.skills.package import (
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillExample,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.verification.models import GateConfig

_EXPERIMENT_MEDIA_TYPE = EXPERIMENT_MANIFEST_MEDIA_TYPE
_MUTATION_MEDIA_TYPE = CANDIDATE_MUTATION_MEDIA_TYPE
_CANDIDATE_MEDIA_TYPE = CANDIDATE_MANIFEST_MEDIA_TYPE
_GATE_MEDIA_TYPE = "application/vnd.spiral-harness.gate-config+json"
_SKILL_ID = "verify-arithmetic"


@dataclass(frozen=True)
class SkillAdmissionFixture:
    store: ArtifactStore
    protocol: ProtocolManifest
    experiment_ref: ArtifactRef
    policy: MutationPolicy
    before_package: SkillPackage
    before_ref: ArtifactRef
    after_package: SkillPackage
    after_ref: ArtifactRef
    parent: HarnessManifest
    parent_ref: ArtifactRef
    mutation: CandidateMutation
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


def _license(store: ArtifactStore) -> SkillLicense:
    provenance_ref = store.put_json(
        {"kind": "generated", "source": "controlled fixture"},
        media_type="application/vnd.spiral-harness.skill-provenance.v1+json",
    )
    review_ref = store.put_json(
        {"approved": True, "reviewer": "controlled-fixture"},
        media_type="application/vnd.spiral-harness.compliance-review.v1+json",
    )
    return SkillLicense(
        spdx_expression="Apache-2.0",
        source_kind=SkillSourceKind.GENERATED,
        provenance_refs=(provenance_ref,),
        compliance_review_ref=review_ref,
    )


def _root_package(store: ArtifactStore) -> SkillPackage:
    return SkillPackage(
        skill_id=_SKILL_ID,
        revision=0,
        name="Verify arithmetic",
        summary="Check arithmetic before returning a final answer.",
        activation_guidance="Use on multi-step arithmetic tasks.",
        applicability_tags=("arithmetic", "verification"),
        rules=(
            SkillRule(
                rule_id="solve-once",
                instruction="Solve the problem once and retain the result.",
            ),
        ),
        procedure="Solve once, then return the result.",
        examples=(
            SkillExample(
                input="What is 17 + 25?",
                output="42",
                explanation="Add the two integers.",
            ),
        ),
        compatible_model_fingerprints=("model-v1",),
        runtime_fingerprints=("runtime-v1",),
        license=_license(store),
    )


def _next_package(before: SkillPackage, before_ref: ArtifactRef) -> SkillPackage:
    return _replace(
        before,
        revision=before.revision + 1,
        parent_package_ref=before_ref,
        rules=(
            SkillRule(
                rule_id="solve-once",
                instruction="Solve the problem once and retain the result.",
            ),
            SkillRule(
                rule_id="independent-recheck",
                instruction="Recompute the final arithmetic independently before answering.",
            ),
        ),
    )


def _fixture(root: Path) -> SkillAdmissionFixture:
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
    protocol_ref = store.put_json(protocol, media_type=PROTOCOL_MANIFEST_MEDIA_TYPE)

    before_package = _root_package(store)
    before_ref = store.put_json(before_package, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    assert before_ref == before_package.artifact_ref
    after_package = _next_package(before_package, before_ref)
    after_ref = store.put_json(after_package, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    assert after_ref == after_package.artifact_ref

    before = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=before_ref,
    )
    after = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=after_ref,
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
        {"failure": "the seed skill does not independently recheck arithmetic"},
        media_type="application/vnd.spiral-harness.evidence+json",
    )
    mutation = CandidateMutation(
        target_component=_SKILL_ID,
        before=before,
        after=after,
        hypothesis=MutationHypothesis(
            evidence_refs=(evidence_ref,),
            where="arithmetic verification skill",
            why="the seed procedure omits an independent recheck",
            expected_activation="the revised skill package is loaded",
            expected_adherence="the independent recheck rule is followed",
            expected_behavior="arithmetic is recomputed before the answer",
            expected_benefit="paired benchmark score improves",
            protected_slices=("already-correct",),
            falsifier="the revised skill is loaded without a recheck",
            negative_control="revert to the seed skill rules",
            risks=("additional context cost",),
        ),
    )
    mutation_ref = store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)
    policy = MutationPolicy(
        allowed_kinds=(ComponentKind.SKILL,),
        allowed_component_names=(_SKILL_ID,),
        allowed_media_types=(SKILL_PACKAGE_MEDIA_TYPE,),
        max_artifact_size_bytes=65_536,
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
        artifact_bytes=store.get_bytes(after_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
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
    return SkillAdmissionFixture(
        store=store,
        protocol=protocol,
        experiment_ref=experiment_ref,
        policy=policy,
        before_package=before_package,
        before_ref=before_ref,
        after_package=after_package,
        after_ref=after_ref,
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        candidate=candidate,
        candidate_ref=candidate_ref,
    )


def _candidate_with_after(
    fixture: SkillAdmissionFixture,
    after_package: SkillPackage,
) -> tuple[ArtifactRef, ArtifactRef]:
    after_ref = fixture.store.put_json(after_package, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    after_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=after_ref,
    )
    mutation = _replace(fixture.mutation, after=after_component)
    mutation_ref = fixture.store.put_json(mutation, media_type=_MUTATION_MEDIA_TYPE)

    # This is intentionally the generic structural registry.  Semantic skill
    # drift is admitted only by CandidateAdmissionService below.
    child = HarnessRegistry(fixture.policy).apply_mutation(
        parent=fixture.parent,
        parent_ref=fixture.parent_ref,
        mutation=mutation,
        artifact_bytes=fixture.store.get_bytes(after_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    child_ref = fixture.store.put_json(child, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        fixture.candidate,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = fixture.store.put_json(candidate, media_type=_CANDIDATE_MEDIA_TYPE)
    return candidate_ref, after_ref


def _assert_semantic_rejection(
    fixture: SkillAdmissionFixture,
    after_package: SkillPackage,
    *,
    match: str,
) -> None:
    candidate_ref, _ = _candidate_with_after(fixture, after_package)
    with pytest.raises(CandidateAdmissionError, match=match):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )


def test_skill_candidate_passes_generic_registry_then_semantic_admission(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    report = CandidateAdmissionService(fixture.store).admit(
        candidate_ref=fixture.candidate_ref,
        experiment_ref=fixture.experiment_ref,
    )

    assert report.admitted is True
    assert report.candidate_ref == fixture.candidate_ref
    assert report.child_harness_ref == fixture.candidate.child_harness_ref
    assert report.mutation_ref == fixture.candidate.mutation_ref
    assert report.component_contract == "skill-rules-revision-v1"
    assert "component_contract_verified" in report.checks


def test_default_generic_policy_still_rejects_skill_packages(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(HarnessRegistryError, match="component kind"):
        HarnessRegistry().apply_mutation(
            parent=fixture.parent,
            parent_ref=fixture.parent_ref,
            mutation=fixture.mutation,
            artifact_bytes=fixture.store.get_bytes(fixture.after_ref),
            artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
        )


def test_skill_admission_rejects_wrong_parent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unrelated = _replace(
        fixture.before_package,
        procedure="A distinct root package used only to test lineage.",
    )
    unrelated_ref = fixture.store.put_json(unrelated, media_type=SKILL_PACKAGE_MEDIA_TYPE)
    after = _replace(fixture.after_package, parent_package_ref=unrelated_ref)

    _assert_semantic_rejection(fixture, after, match="parent_package_ref")


def test_skill_admission_rejects_skipped_revision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    after = _replace(fixture.after_package, revision=2)

    _assert_semantic_rejection(fixture, after, match="revision must increment")


def test_skill_admission_rejects_changed_skill_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    after = _replace(fixture.after_package, skill_id="different-skill")

    _assert_semantic_rejection(fixture, after, match="skill_id")


def test_skill_admission_rejects_rules_noop(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    after = _replace(fixture.after_package, rules=fixture.before_package.rules)

    _assert_semantic_rejection(fixture, after, match="must change rules")


def test_skill_admission_rejects_license_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed_license = _replace(fixture.after_package.license, spdx_expression="MIT")
    after = _replace(fixture.after_package, license=changed_license)

    _assert_semantic_rejection(fixture, after, match="license")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("summary", "Candidate-authored routing metadata."),
        ("activation_guidance", "Always activate, regardless of the task."),
        ("applicability_tags", ("all-tasks",)),
        ("procedure", "A candidate-authored replacement procedure."),
    ],
)
def test_skill_admission_rejects_metadata_drift(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    fixture = _fixture(tmp_path)
    after = _replace(fixture.after_package, **{field_name: value})

    _assert_semantic_rejection(fixture, after, match=field_name)


def test_skill_admission_rejects_missing_provenance_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_ref = ArtifactRef(
        sha256="0" * 64,
        size=1,
        media_type="application/vnd.spiral-harness.skill-provenance.v1+json",
    )
    changed_license = _replace(
        fixture.after_package.license,
        provenance_refs=(missing_ref,),
    )
    after = _replace(fixture.after_package, license=changed_license)

    _assert_semantic_rejection(fixture, after, match="could not be verified")


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("compatible_model_fingerprints", ("foreign-model",), "frozen model"),
        ("runtime_fingerprints", ("foreign-runtime",), "frozen runtime"),
    ],
)
def test_skill_admission_rejects_incompatible_execution_planes(
    tmp_path: Path,
    field_name: str,
    value: tuple[str, ...],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    after = _replace(fixture.after_package, **{field_name: value})

    _assert_semantic_rejection(fixture, after, match=message)


def test_skill_admission_rejects_tampered_package_before_child_acceptance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    candidate_ref, after_ref = _candidate_with_after(fixture, fixture.after_package)
    fixture.store.path_for(after_ref).write_bytes(b"{}")

    with pytest.raises(CandidateAdmissionError, match="after artifact could not be verified"):
        CandidateAdmissionService(fixture.store).admit(
            candidate_ref=candidate_ref,
            experiment_ref=fixture.experiment_ref,
        )
