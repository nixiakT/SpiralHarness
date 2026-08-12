from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from spiral_harness.core.canonical import sha256_bytes
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
from spiral_harness.core.lifecycle import CandidateLifecycleEvent, CandidateState
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
from spiral_harness.execution.schedule import (
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
)
from spiral_harness.experiments.admission import (
    ADMISSION_REPORT_MEDIA_TYPE,
    CandidateAdmissionService,
)
from spiral_harness.experiments.controller import (
    ExperimentController,
    ExperimentControllerError,
)
from spiral_harness.experiments.skill_probe_authorization import (
    SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE,
    SkillProbeExecutionAuthorization,
    SkillProbeExecutionAuthorizationCapability,
    SkillProbeExecutionAuthorizationError,
    TrustedSkillProbeExecutionAuthority,
    TrustedSkillProbeExecutionAuthorizationService,
    _ConsumedSkillProbeExecutionAuthorizationError,
    _load_exact_authorization,
)
from spiral_harness.experiments.skill_probe_closure import (
    MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
)
from spiral_harness.experiments.skill_probe_execution import (
    MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT,
    MATCHED_SKILL_PROBE_RESET_FINGERPRINT,
)
from spiral_harness.experiments.skill_probes import (
    SkillProbePreregistrationError,
    replay_probe_preregistration_refs,
    resolve_probe_preregistration,
    verify_skill_probe_preregistration,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.skills.loading import SkillDisclosureLevel, SkillPackageLoader
from spiral_harness.skills.package import (
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillLicense,
    SkillPackage,
    SkillRule,
    SkillSourceKind,
)
from spiral_harness.storage.artifact_store import ArtifactStore
from spiral_harness.storage.journal import CandidateJournal
from spiral_harness.verification.artifacts import TrustedGateBatchService
from spiral_harness.verification.mechanism import (
    LEGACY_SKILL_REQUEST_ACTIVATION_MECHANISM_ID,
    REQUIRED_SKILL_MECHANISM_IDS,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import GateConfig, MechanismEvidence
from spiral_harness.verification.skill_plan import (
    ATTESTED_SKILL_MECHANISM_CLAIMS,
    CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    NEUTRAL_SKILL_RULES_MEDIA_TYPE,
    SKILL_ADHERENCE_PROBE_MEDIA_TYPE,
    SKILL_BEHAVIOR_PROBE_MEDIA_TYPE,
    SKILL_CLAIM_EVIDENCE_MEDIA_TYPES,
    SKILL_MECHANISM_PLAN_MEDIA_TYPE,
    SKILL_PLACEBO_CONTROL_MEDIA_TYPE,
    SKILL_PROBE_ROSTER_MEDIA_TYPE,
    SKILL_VERIFICATION_POLICY_MEDIA_TYPE,
    ControlledSkillProbeTask,
    NeutralSkillRules,
    SkillClaimAuthority,
    SkillEvidenceProfile,
    SkillMechanismClaim,
    SkillMechanismPlan,
    SkillPlaceboControl,
    SkillProbeRoster,
    SkillVerificationPolicy,
)

_SKILL_ID = "verify-arithmetic"
_CHANGED_RULE_ID = "solve-once"
_MODEL = "fixed-model-v1"
_INFERENCE = "temperature=0;seed=paired"
_RUNTIME = "fixed-runtime-v1"
_GRADER = "fixed-grader-v1"
_PROBE_GRADER = "skill-probe-grader-v1"
_MODEL_SPEC = "9" * 64
_RESET = MATCHED_SKILL_PROBE_RESET_FINGERPRINT
_EXECUTION_ORDER = MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT
_GATE_MEDIA_TYPE = "application/vnd.spiral-harness.gate-config+json"


def _replace[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    payload = model.model_dump(
        mode="python",
        by_alias=False,
        exclude_none=False,
        round_trip=True,
        warnings="none",
    )
    payload.update(updates)
    return type(model).model_validate(payload, strict=True, by_name=True)


def _digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _put(store: ArtifactStore, model: object, media_type: str) -> ArtifactRef:
    return store.put_json(model, media_type=media_type)


@dataclass(frozen=True)
class ProbeFixture:
    store: ArtifactStore
    gate_batch_service: TrustedGateBatchService
    mechanism_service: TrustedMechanismEvidenceService
    exploration_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    task: ControlledSkillProbeTask
    task_ref: ArtifactRef
    roster: SkillProbeRoster
    roster_ref: ArtifactRef
    neutral: NeutralSkillRules
    neutral_ref: ArtifactRef
    policy: SkillVerificationPolicy
    policy_ref: ArtifactRef
    gate_config: GateConfig
    gate_config_ref: ArtifactRef
    protocol: ProtocolManifest
    protocol_ref: ArtifactRef
    experiment: ExperimentManifest
    experiment_ref: ArtifactRef
    before_package: SkillPackage
    before_ref: ArtifactRef
    after_package: SkillPackage
    after_ref: ArtifactRef
    parent: HarnessManifest
    parent_ref: ArtifactRef
    mutation: CandidateMutation
    mutation_ref: ArtifactRef
    child: HarnessManifest
    child_ref: ArtifactRef
    placebo_package: SkillPackage
    placebo_package_ref: ArtifactRef
    placebo_harness: HarnessManifest
    placebo_harness_ref: ArtifactRef
    placebo_control: SkillPlaceboControl
    placebo_control_ref: ArtifactRef
    candidate: CandidateManifest
    candidate_ref: ArtifactRef
    plan: SkillMechanismPlan
    plan_ref: ArtifactRef


def _license(store: ArtifactStore) -> SkillLicense:
    provenance_ref = _put(
        store,
        {"kind": "generated", "source": "skill probe fixture"},
        "application/vnd.spiral-harness.skill-provenance.v1+json",
    )
    review_ref = _put(
        store,
        {"approved": True, "reviewer": "fixture"},
        "application/vnd.spiral-harness.compliance-review.v1+json",
    )
    return SkillLicense(
        spdx_expression="Apache-2.0",
        source_kind=SkillSourceKind.GENERATED,
        provenance_refs=(provenance_ref,),
        compliance_review_ref=review_ref,
    )


def _before_package(store: ArtifactStore) -> SkillPackage:
    return SkillPackage(
        skill_id=_SKILL_ID,
        revision=0,
        name="Verify arithmetic",
        summary="Check arithmetic before returning an answer.",
        activation_guidance="Use for arithmetic tasks.",
        applicability_tags=("arithmetic",),
        rules=(
            SkillRule(
                rule_id=_CHANGED_RULE_ID,
                instruction="Solve the arithmetic once.",
            ),
            SkillRule(
                rule_id="format-answer",
                instruction="Return a concise final answer.",
            ),
        ),
        procedure="Solve, verify, and return the answer.",
        compatible_model_fingerprints=(_MODEL,),
        runtime_fingerprints=(_RUNTIME,),
        license=_license(store),
    )


def _probe_fixture(root: Path) -> ProbeFixture:
    store = ArtifactStore(root)
    gate_batch_service = TrustedGateBatchService()
    mechanism_service = TrustedMechanismEvidenceService()
    exploration_ref = _put(
        store,
        {"partition": "exploration", "task_ids": ["probe-task-1"]},
        "application/vnd.spiral-harness.exploration-split.v1+json",
    )
    gate_split_ref = _put(
        store,
        {"partition": "gate", "task_ids": ["gate-task-1"]},
        "application/vnd.spiral-harness.gate-split.v1+json",
    )
    adherence_ref = _put(
        store,
        {"probe_id": "adherence-1", "task_id": "probe-task-1"},
        SKILL_ADHERENCE_PROBE_MEDIA_TYPE,
    )
    behavior_ref = _put(
        store,
        {"probe_id": "behavior-1", "task_id": "probe-task-1"},
        SKILL_BEHAVIOR_PROBE_MEDIA_TYPE,
    )
    task = ControlledSkillProbeTask(
        task_id="probe-task-1",
        question="What is 17 + 25?",
    )
    task_ref = _put(store, task, CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE)
    assert task_ref == task.artifact_ref
    roster = SkillProbeRoster(
        evidence_profile=SkillEvidenceProfile.CONTROLLED_REPLAY,
        exploration_split_ref=exploration_ref,
        study="skill-mechanism-preregistration-v1",
        kind="matched-skill-controls",
        query=0,
        master_seed=71,
        task_ids=("probe-task-1",),
        task_refs=(task_ref,),
        search_runs=(0,),
        repeat_seeds=(11, 12),
        max_attempts_per_cell=1,
        token_ceiling_per_attempt=256,
        adherence_probe_refs=(adherence_ref,),
        behavior_probe_refs=(behavior_ref,),
    )
    roster_ref = _put(store, roster, SKILL_PROBE_ROSTER_MEDIA_TYPE)
    assert roster_ref == roster.artifact_ref
    neutral = NeutralSkillRules(rule_ids=(_CHANGED_RULE_ID,))
    neutral_ref = _put(store, neutral, NEUTRAL_SKILL_RULES_MEDIA_TYPE)
    assert neutral_ref == neutral.artifact_ref

    authorities = []
    for claim in ATTESTED_SKILL_MECHANISM_CLAIMS:
        config_ref = _put(
            store,
            {"claim": claim.value, "profile": "controlled_replay"},
            "application/vnd.spiral-harness.skill-claim-config.v1+json",
        )
        authorities.append(
            SkillClaimAuthority(
                claim=claim,
                evidence_profile=SkillEvidenceProfile.CONTROLLED_REPLAY,
                attestor_id=_digest(f"attestor:{claim.value}"),
                attestation_domain=f"spiral-harness:{claim.value}:v1",
                producer_fingerprint=f"producer:{claim.value}:v1",
                verifier_fingerprint=f"verifier:{claim.value}:v1",
                evidence_media_type=SKILL_CLAIM_EVIDENCE_MEDIA_TYPES[claim],
                config_ref=config_ref,
            )
        )
    policy = SkillVerificationPolicy(
        evidence_profile=SkillEvidenceProfile.CONTROLLED_REPLAY,
        model_fingerprint=_MODEL,
        model_spec_fingerprint=_MODEL_SPEC,
        inference_fingerprint=_INFERENCE,
        runtime_fingerprint=_RUNTIME,
        grader_fingerprint=_GRADER,
        task_roster_ref=roster_ref,
        neutral_rules_ref=neutral_ref,
        runtime_activation_hook_fingerprint="runtime-activation-hook-v1",
        probe_grader_fingerprint=_PROBE_GRADER,
        reset_fingerprint=_RESET,
        execution_order_fingerprint=_EXECUTION_ORDER,
        generic_mechanism_attestor_id=mechanism_service.attestor_id,
        aggregate_attestor_id=_digest("skill-aggregate-attestor"),
        aggregate_attestation_domain="spiral-harness:skill-aggregate:v1",
        aggregate_producer_fingerprint="producer:skill-aggregate:v1",
        aggregate_verifier_fingerprint="verifier:skill-aggregate:v1",
        claim_authorities=tuple(authorities),
        min_probe_tasks=1,
        min_adherence_coverage=1.0,
        min_adherence_rate=1.0,
        min_behavior_effect_vs_revert=0.0,
        min_behavior_effect_vs_placebo=0.0,
        max_placebo_input_token_delta=256,
        max_placebo_context_size_delta=16_384,
    )
    policy_ref = _put(store, policy, SKILL_VERIFICATION_POLICY_MEDIA_TYPE)
    assert policy_ref == policy.artifact_ref

    gate_config = GateConfig(
        version="skill-probe-v1",
        min_tasks=1,
        bootstrap_samples=1_000,
        required_mechanism_checks=tuple(sorted(REQUIRED_SKILL_MECHANISM_IDS)),
    )
    gate_config_ref = _put(store, gate_config, _GATE_MEDIA_TYPE)
    capability_policy_ref = _put(
        store,
        CapabilityPolicy(),
        CAPABILITY_POLICY_MEDIA_TYPE,
    )
    protocol = ProtocolManifest(
        benchmark_fingerprint="skill-probe-benchmark-v1",
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
        model_fingerprint=_MODEL,
        inference_fingerprint=_INFERENCE,
        runtime_fingerprint=_RUNTIME,
        model_spec_fingerprint=_MODEL_SPEC,
        sandbox_fingerprint="logical-fixture-isolation-v1",
        capability_policy_ref=capability_policy_ref,
        skill_verification_policy_ref=policy_ref,
        grader_fingerprint=_GRADER,
        gate_batch_attestor_id=gate_batch_service.attestor_id,
        mechanism_evidence_attestor_id=mechanism_service.attestor_id,
        gate_config_ref=gate_config_ref,
        trusted_plane_version="trusted-plane-v1",
        budget=BudgetPolicy(max_evaluations=100),
    )
    protocol_ref = _put(store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)

    before_package = _before_package(store)
    before_ref = _put(store, before_package, SKILL_PACKAGE_MEDIA_TYPE)
    after_package = _replace(
        before_package,
        revision=1,
        parent_package_ref=before_ref,
        rules=(
            SkillRule(
                rule_id=_CHANGED_RULE_ID,
                instruction="Solve once, then independently recompute the arithmetic.",
            ),
            before_package.rules[1],
        ),
    )
    after_ref = _put(store, after_package, SKILL_PACKAGE_MEDIA_TYPE)
    before_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=before_ref,
    )
    after_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=after_ref,
    )
    parent = HarnessManifest(
        model_fingerprint=_MODEL,
        runtime_fingerprint=_RUNTIME,
        trusted_plane_version=protocol.trusted_plane_version,
        components=(before_component,),
        budget=BudgetPolicy(max_evaluations=50),
    )
    parent_ref = _put(store, parent, HARNESS_MANIFEST_MEDIA_TYPE)
    evidence_ref = _put(
        store,
        {"failure": "the seed skill omits an independent recomputation"},
        "application/vnd.spiral-harness.evidence+json",
    )
    mutation = CandidateMutation(
        target_component=_SKILL_ID,
        before=before_component,
        after=after_component,
        hypothesis=MutationHypothesis(
            evidence_refs=(evidence_ref,),
            where="arithmetic verification skill",
            why="the seed rule checks the arithmetic only once",
            expected_activation="the revised package is activated",
            expected_adherence="the new recheck rule is followed",
            expected_behavior="the arithmetic is independently recomputed",
            expected_benefit="paired score improves",
            protected_slices=("already-correct",),
            falsifier="the changed rule is not followed",
            negative_control="revert and matched neutral placebo",
            risks=("extra context",),
        ),
    )
    mutation_ref = _put(store, mutation, CANDIDATE_MUTATION_MEDIA_TYPE)
    mutation_policy = MutationPolicy(
        allowed_kinds=(ComponentKind.SKILL,),
        allowed_component_names=(_SKILL_ID,),
        allowed_media_types=(SKILL_PACKAGE_MEDIA_TYPE,),
        max_artifact_size_bytes=65_536,
    )
    experiment = ExperimentManifest(
        protocol_ref=protocol_ref,
        seed_harness_ref=parent_ref,
        mutation_policy=mutation_policy,
        objective="improve paired benchmark score",
        baselines=("static",),
        stopping=("one-candidate",),
        search_budget=BudgetPolicy(max_evaluations=50),
    )
    experiment_ref = _put(store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    child = HarnessRegistry(mutation_policy).apply_mutation(
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        artifact_bytes=store.get_bytes(after_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    child_ref = _put(store, child, HARNESS_MANIFEST_MEDIA_TYPE)

    neutral_by_id = {rule.rule_id: rule for rule in neutral.rules}
    placebo_package = _replace(
        after_package,
        rules=tuple(neutral_by_id.get(rule.rule_id, rule) for rule in after_package.rules),
    )
    placebo_package_ref = _put(store, placebo_package, SKILL_PACKAGE_MEDIA_TYPE)
    placebo_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=placebo_package_ref,
    )
    placebo_harness = _replace(child, components=(placebo_component,))
    placebo_harness_ref = _put(store, placebo_harness, HARNESS_MANIFEST_MEDIA_TYPE)
    loader = SkillPackageLoader(store)
    candidate_disclosure = loader.disclose(
        after_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=_MODEL,
        runtime_fingerprint=_RUNTIME,
    )
    placebo_disclosure = loader.disclose(
        placebo_package_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=_MODEL,
        runtime_fingerprint=_RUNTIME,
    )
    placebo_control = SkillPlaceboControl(
        evidence_profile=SkillEvidenceProfile.CONTROLLED_REPLAY,
        candidate_package_ref=after_ref,
        placebo_package_ref=placebo_package_ref,
        candidate_harness_ref=child_ref,
        placebo_harness_ref=placebo_harness_ref,
        neutral_rules_ref=neutral_ref,
        changed_rule_ids=(_CHANGED_RULE_ID,),
        candidate_rule_count=len(after_package.rules),
        placebo_rule_count=len(placebo_package.rules),
        candidate_context_size_bytes=candidate_disclosure.context_size_bytes,
        placebo_context_size_bytes=placebo_disclosure.context_size_bytes,
    )
    placebo_control_ref = _put(store, placebo_control, SKILL_PLACEBO_CONTROL_MEDIA_TYPE)
    candidate = CandidateManifest(
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
        evidence_refs=(evidence_ref,),
        evaluation_plan_ref=gate_config_ref,
    )
    candidate_ref = _put(store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    revert_schedule = EvaluationBatchSchedule(
        study=roster.study,
        kind=roster.kind,
        phase=EvaluationPhase.PROBE,
        query=roster.query,
        master_seed=roster.master_seed,
        parent_harness_id=parent_ref.sha256,
        candidate_harness_id=child_ref.sha256,
        task_ids=roster.task_ids,
        search_runs=roster.search_runs,
        repeat_seeds=roster.repeat_seeds,
        max_attempts_per_cell=roster.max_attempts_per_cell,
        token_ceiling_per_attempt=roster.token_ceiling_per_attempt,
    )
    placebo_schedule = _replace(
        revert_schedule,
        parent_harness_id=placebo_harness_ref.sha256,
    )
    plan = SkillMechanismPlan(
        evidence_profile=policy.evidence_profile,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
        mutation_ref=mutation_ref,
        parent_harness_ref=parent_ref,
        candidate_harness_ref=child_ref,
        before_skill_package_ref=before_ref,
        after_skill_package_ref=after_ref,
        target_skill_id=_SKILL_ID,
        changed_rule_ids=(_CHANGED_RULE_ID,),
        policy_ref=policy_ref,
        policy_fingerprint=policy.fingerprint,
        exploration_split_ref=exploration_ref,
        probe_roster_ref=roster_ref,
        probe_roster_fingerprint=roster.fingerprint,
        placebo_control_ref=placebo_control_ref,
        placebo_harness_ref=placebo_harness_ref,
        model_spec_fingerprint=_MODEL_SPEC,
        runtime_fingerprint=_RUNTIME,
        probe_grader_fingerprint=_PROBE_GRADER,
        reset_fingerprint=_RESET,
        execution_order_fingerprint=_EXECUTION_ORDER,
        revert_schedule=revert_schedule,
        placebo_schedule=placebo_schedule,
    )
    plan_ref = _put(store, plan, SKILL_MECHANISM_PLAN_MEDIA_TYPE)
    assert plan_ref == plan.artifact_ref
    return ProbeFixture(
        store=store,
        gate_batch_service=gate_batch_service,
        mechanism_service=mechanism_service,
        exploration_ref=exploration_ref,
        gate_split_ref=gate_split_ref,
        task=task,
        task_ref=task_ref,
        roster=roster,
        roster_ref=roster_ref,
        neutral=neutral,
        neutral_ref=neutral_ref,
        policy=policy,
        policy_ref=policy_ref,
        gate_config=gate_config,
        gate_config_ref=gate_config_ref,
        protocol=protocol,
        protocol_ref=protocol_ref,
        experiment=experiment,
        experiment_ref=experiment_ref,
        before_package=before_package,
        before_ref=before_ref,
        after_package=after_package,
        after_ref=after_ref,
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        mutation_ref=mutation_ref,
        child=child,
        child_ref=child_ref,
        placebo_package=placebo_package,
        placebo_package_ref=placebo_package_ref,
        placebo_harness=placebo_harness,
        placebo_harness_ref=placebo_harness_ref,
        placebo_control=placebo_control,
        placebo_control_ref=placebo_control_ref,
        candidate=candidate,
        candidate_ref=candidate_ref,
        plan=plan,
        plan_ref=plan_ref,
    )


@pytest.fixture
def graph(tmp_path: Path) -> ProbeFixture:
    return _probe_fixture(tmp_path)


def _verify(graph: ProbeFixture, plan_ref: ArtifactRef) -> SkillMechanismPlan:
    return verify_skill_probe_preregistration(
        graph.store,
        plan_ref=plan_ref,
        expected_experiment_ref=graph.experiment_ref,
        expected_protocol_ref=graph.protocol_ref,
        expected_candidate_ref=graph.candidate_ref,
    )


def _controller_at_probes(
    graph: ProbeFixture,
) -> tuple[ExperimentController, ArtifactRef, ArtifactRef]:
    admission = CandidateAdmissionService(graph.store).admit(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
    )
    admission_ref = _put(graph.store, admission, ADMISSION_REPORT_MEDIA_TYPE)
    controller = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
    )
    frozen = controller.freeze_experiment()
    controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=admission_ref,
    )
    probes = controller.start_probes(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=valid,
        skill_mechanism_plan_ref=graph.plan_ref,
    )
    return controller, valid, probes


def _persist_plan(graph: ProbeFixture, plan: SkillMechanismPlan) -> ArtifactRef:
    return _put(graph.store, plan, SKILL_MECHANISM_PLAN_MEDIA_TYPE)


def _rebind_probe_roster(
    graph: ProbeFixture,
    roster: SkillProbeRoster,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef]:
    roster_ref = _put(graph.store, roster, SKILL_PROBE_ROSTER_MEDIA_TYPE)
    policy = _replace(graph.policy, task_roster_ref=roster_ref)
    policy_ref = _put(graph.store, policy, SKILL_VERIFICATION_POLICY_MEDIA_TYPE)
    protocol = _replace(graph.protocol, skill_verification_policy_ref=policy_ref)
    protocol_ref = _put(graph.store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = _replace(graph.experiment, protocol_ref=protocol_ref)
    experiment_ref = _put(graph.store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    candidate = _replace(graph.candidate, experiment_ref=experiment_ref)
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    plan = _replace(
        graph.plan,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
        policy_ref=policy_ref,
        policy_fingerprint=policy.fingerprint,
        probe_roster_ref=roster_ref,
        probe_roster_fingerprint=roster.fingerprint,
    )
    plan_ref = _persist_plan(graph, plan)
    return protocol_ref, experiment_ref, candidate_ref, plan_ref


def _rebind_gate_config(
    graph: ProbeFixture,
    gate_config: GateConfig,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef]:
    gate_ref = _put(graph.store, gate_config, _GATE_MEDIA_TYPE)
    protocol = _replace(graph.protocol, gate_config_ref=gate_ref)
    protocol_ref = _put(graph.store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = _replace(graph.experiment, protocol_ref=protocol_ref)
    experiment_ref = _put(graph.store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        graph.candidate,
        experiment_ref=experiment_ref,
        evaluation_plan_ref=gate_ref,
    )
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    plan = _replace(
        graph.plan,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
    )
    plan_ref = _persist_plan(graph, plan)
    return protocol_ref, experiment_ref, candidate_ref, plan_ref


def _verify_rebound(
    graph: ProbeFixture,
    refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef],
) -> SkillMechanismPlan:
    protocol_ref, experiment_ref, candidate_ref, plan_ref = refs
    return verify_skill_probe_preregistration(
        graph.store,
        plan_ref=plan_ref,
        expected_experiment_ref=experiment_ref,
        expected_protocol_ref=protocol_ref,
        expected_candidate_ref=candidate_ref,
    )


def test_verifies_complete_plan_and_controller_binds_it_to_probe_lifecycle(
    graph: ProbeFixture,
) -> None:
    assert _verify(graph, graph.plan_ref) == graph.plan
    admission = CandidateAdmissionService(graph.store).admit(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
    )
    admission_ref = _put(graph.store, admission, ADMISSION_REPORT_MEDIA_TYPE)
    controller = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
    )
    frozen = controller.freeze_experiment()
    controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=admission_ref,
    )
    probe_tail = controller.start_probes(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=valid,
        skill_mechanism_plan_ref=graph.plan_ref,
    )

    event = CandidateJournal(graph.store).replay(probe_tail)[-1]
    assert event.to_state is CandidateState.RUNNING_PROBES
    assert event.evidence_refs == (graph.plan_ref,)

    with pytest.raises(ExperimentControllerError, match="expected 'valid'"):
        controller.start_probes(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=probe_tail,
            skill_mechanism_plan_ref=graph.plan_ref,
        )


@pytest.mark.parametrize("forbidden_field", ("answer", "gold"))
def test_controlled_probe_task_cannot_represent_trusted_answers(
    forbidden_field: str,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ControlledSkillProbeTask.model_validate(
            {
                "task_id": "probe-task-1",
                "question": "What is 17 + 25?",
                forbidden_field: "42",
            },
            strict=True,
        )


def test_roster_requires_one_unique_typed_task_ref_per_task_id(
    graph: ProbeFixture,
) -> None:
    with pytest.raises(ValidationError, match="duplicate artifacts"):
        _replace(graph.roster, task_refs=(graph.task_ref, graph.task_ref))
    second_task = ControlledSkillProbeTask(
        task_id="probe-task-2",
        question="What is 19 + 23?",
    )
    second_ref = _put(
        graph.store,
        second_task,
        CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    )
    with pytest.raises(ValidationError, match="exactly one artifact per task_id"):
        _replace(graph.roster, task_refs=(graph.task_ref, second_ref))
    with pytest.raises(ValidationError, match="controlled-skill-probe-task"):
        _replace(graph.roster, task_refs=graph.roster.adherence_probe_refs)


def test_replay_rejects_same_task_id_with_question_replaced_after_freeze(
    graph: ProbeFixture,
) -> None:
    replacement = _replace(graph.task, question="What is 18 + 24?")
    replacement_ref = _put(
        graph.store,
        replacement,
        CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    )
    graph.store.path_for(graph.task_ref).write_bytes(graph.store.get_bytes(replacement_ref))

    with pytest.raises(
        SkillProbePreregistrationError,
        match="controlled skill probe task cannot be loaded exactly",
    ):
        _verify(graph, graph.plan_ref)


def test_replay_rejects_foreign_task_ref_even_with_the_exact_media_type(
    graph: ProbeFixture,
) -> None:
    foreign_task = ControlledSkillProbeTask(
        task_id="foreign-probe-task",
        question="What is 10 + 32?",
    )
    foreign_ref = _put(
        graph.store,
        foreign_task,
        CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    )
    roster = _replace(graph.roster, task_refs=(foreign_ref,))

    with pytest.raises(SkillProbePreregistrationError, match="frozen roster task IDs"):
        _verify_rebound(graph, _rebind_probe_roster(graph, roster))


def test_controller_replays_probe_plan_before_gate_authorization(
    graph: ProbeFixture,
) -> None:
    admission = CandidateAdmissionService(graph.store).admit(
        candidate_ref=graph.candidate_ref,
        experiment_ref=graph.experiment_ref,
    )
    admission_ref = _put(graph.store, admission, ADMISSION_REPORT_MEDIA_TYPE)
    controller = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
    )
    frozen = controller.freeze_experiment()
    controller.start_search(previous_tail_ref=frozen)
    registered = controller.register_candidate(candidate_ref=graph.candidate_ref)
    valid = controller.admit_candidate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=registered,
        admission_report_ref=admission_ref,
    )
    forged_tail = CandidateJournal(graph.store).append(
        stream_id=f"candidate/{graph.candidate_ref.sha256}",
        previous_entry_ref=valid,
        event=CandidateLifecycleEvent(
            candidate_ref=graph.candidate_ref,
            from_state=CandidateState.VALID,
            to_state=CandidateState.RUNNING_PROBES,
            evidence_refs=(graph.policy_ref,),
            reason="forged policy ref substituted for the candidate plan",
        ),
    )
    controller._candidate_tails[graph.candidate_ref.sha256] = forged_tail

    with pytest.raises(ExperimentControllerError, match="probe history replay failed"):
        controller.start_gate(
            candidate_ref=graph.candidate_ref,
            previous_tail_ref=forged_tail,
            mechanism_evidence_ref=graph.policy_ref,
        )


def test_controller_issues_current_process_local_skill_probe_execution_grant(
    graph: ProbeFixture,
) -> None:
    controller, _, probes = _controller_at_probes(graph)

    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    repeated_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    authorization = _load_exact_authorization(graph.store, authorization_ref)

    assert authorization_ref.media_type == SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE
    assert repeated_ref == authorization_ref
    assert authorization == graph.store.get_json(
        authorization_ref,
        SkillProbeExecutionAuthorization,
    )
    assert authorization.experiment_ref == graph.experiment_ref
    assert authorization.protocol_ref == graph.protocol_ref
    assert authorization.candidate_ref == graph.candidate_ref
    assert authorization.plan_ref == graph.plan_ref
    assert authorization.running_probes_tail_ref == probes

    assert not hasattr(controller, "skill_probe_execution_authorization_capability")
    assert not hasattr(controller, "skill_probe_execution_authority")
    assert not hasattr(controller, "register_skill_probe_execution_closure")


@pytest.mark.parametrize(
    ("privileged_type", "message"),
    (
        (
            SkillProbeExecutionAuthorizationCapability,
            "verification capabilities are created only by the trusted service",
        ),
        (
            TrustedSkillProbeExecutionAuthority,
            "execution authorities are created only by the trusted service",
        ),
        (
            TrustedSkillProbeExecutionAuthorizationService,
            "authorization services are created only by controller composition",
        ),
    ),
)
def test_skill_probe_execution_privileges_cannot_be_publicly_constructed(
    privileged_type: type[object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        privileged_type()


def test_uninitialized_skill_probe_capability_fails_with_stable_trust_error() -> None:
    shell = object.__new__(SkillProbeExecutionAuthorizationCapability)

    with pytest.raises(
        SkillProbeExecutionAuthorizationError,
        match="authorization capability is not initialized/trusted",
    ):
        shell.verify_skill_probe_execution_authorization(
            ArtifactRef(sha256="0" * 64, size=0, media_type="application/json")
        )


def test_started_skill_probe_grant_does_not_authenticate_an_unregistered_closure(
    graph: ProbeFixture,
) -> None:
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    capability = controller._skill_probe_authorizations.verification_capability
    authority = controller._skill_probe_authorizations.execution_authority
    closure_ref = _put(
        graph.store,
        {"closure": "not controller-registered"},
        MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
    )

    authority.begin_skill_probe_execution(authorization_ref)

    with pytest.raises(SkillProbeExecutionAuthorizationError, match="not registered"):
        capability.verify_registered_skill_probe_execution_closure(
            authorization_ref,
            closure_ref,
        )


@pytest.mark.parametrize("repeat_exact_ref", (True, False))
def test_started_skill_probe_grant_registers_exactly_one_closure(
    graph: ProbeFixture,
    repeat_exact_ref: bool,
) -> None:
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    capability = controller._skill_probe_authorizations.verification_capability
    authority = controller._skill_probe_authorizations.execution_authority
    first_closure_ref = _put(
        graph.store,
        {"closure": "first"},
        MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
    )
    second_closure_ref = (
        first_closure_ref
        if repeat_exact_ref
        else _put(
            graph.store,
            {"closure": "second"},
            MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
        )
    )

    authorization = authority.begin_skill_probe_execution(authorization_ref)
    assert (
        authority.register_skill_probe_execution_closure(
            authorization_ref,
            first_closure_ref,
        )
        == authorization
    )
    assert (
        capability.verify_registered_skill_probe_execution_closure(
            authorization_ref,
            first_closure_ref,
        )
        == authorization
    )

    with pytest.raises(SkillProbeExecutionAuthorizationError, match="already registered"):
        authority.register_skill_probe_execution_closure(
            authorization_ref,
            second_closure_ref,
        )
    if second_closure_ref != first_closure_ref:
        with pytest.raises(SkillProbeExecutionAuthorizationError, match="binds another closure"):
            capability.verify_registered_skill_probe_execution_closure(
                authorization_ref,
                second_closure_ref,
            )


def test_skill_probe_execution_grant_rejects_stale_and_advanced_heads(
    graph: ProbeFixture,
) -> None:
    controller, valid, probes = _controller_at_probes(graph)

    with pytest.raises(ExperimentControllerError, match="stale, foreign"):
        controller.issue_skill_probe_execution_authorization(
            candidate_ref=graph.candidate_ref,
            running_probes_tail_ref=valid,
        )

    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    controller._candidate_tails[graph.candidate_ref.sha256] = valid
    with pytest.raises(
        SkillProbeExecutionAuthorizationError,
        match="current controller head",
    ):
        controller._skill_probe_authorizations.verification_capability.verify_skill_probe_execution_authorization(
            authorization_ref
        )


def test_begin_distinguishes_a_grant_consumed_before_current_head_replay(
    graph: ProbeFixture,
) -> None:
    controller, valid, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    authority = controller._skill_probe_authorizations.execution_authority
    controller._candidate_tails[graph.candidate_ref.sha256] = valid

    with pytest.raises(
        _ConsumedSkillProbeExecutionAuthorizationError,
        match="verification failed after execution started",
    ) as consumed:
        authority.begin_skill_probe_execution(authorization_ref)
    assert isinstance(consumed.value.__cause__, SkillProbeExecutionAuthorizationError)

    with pytest.raises(SkillProbeExecutionAuthorizationError, match="already started"):
        authority.begin_skill_probe_execution(authorization_ref)


def test_skill_probe_execution_grant_is_invalid_after_real_lifecycle_advance(
    graph: ProbeFixture,
) -> None:
    controller, _, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    source_ref = _put(
        graph.store,
        {"probe": "failed closed before gate"},
        "application/vnd.spiral-harness.probe-source.v1+json",
    )
    evidence = graph.mechanism_service.create(
        protocol_ref=graph.protocol_ref,
        protocol=graph.protocol,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.child_ref,
        source_refs=(source_ref,),
        evidence=MechanismEvidence(
            candidate_harness_id=graph.child_ref.sha256,
            checks=(),
        ),
    )
    evidence_ref = _put(
        graph.store,
        evidence,
        "application/vnd.spiral-harness.attested-mechanism-evidence.v1+json",
    )
    controller.start_gate(
        candidate_ref=graph.candidate_ref,
        previous_tail_ref=probes,
        mechanism_evidence_ref=evidence_ref,
    )

    with pytest.raises(
        SkillProbeExecutionAuthorizationError,
        match="current controller head",
    ):
        controller._skill_probe_authorizations.verification_capability.verify_skill_probe_execution_authorization(
            authorization_ref
        )


def test_skill_probe_execution_grant_rejects_fork_and_foreign_controller(
    graph: ProbeFixture,
) -> None:
    controller, valid, probes = _controller_at_probes(graph)
    authorization_ref = controller.issue_skill_probe_execution_authorization(
        candidate_ref=graph.candidate_ref,
        running_probes_tail_ref=probes,
    )
    forked_tail = CandidateJournal(graph.store).append(
        stream_id=f"candidate/{graph.candidate_ref.sha256}",
        previous_entry_ref=valid,
        event=CandidateLifecycleEvent(
            candidate_ref=graph.candidate_ref,
            from_state=CandidateState.VALID,
            to_state=CandidateState.RUNNING_PROBES,
            evidence_refs=(graph.plan_ref,),
            reason="structurally valid sibling branch",
        ),
    )

    with pytest.raises(ExperimentControllerError, match="stale, foreign"):
        controller.issue_skill_probe_execution_authorization(
            candidate_ref=graph.candidate_ref,
            running_probes_tail_ref=forked_tail,
        )

    foreign = ExperimentController(
        graph.store,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
    )
    with pytest.raises(SkillProbeExecutionAuthorizationError, match="not issued"):
        foreign._skill_probe_authorizations.verification_capability.verify_skill_probe_execution_authorization(
            authorization_ref
        )

    same_path_repository = ArtifactStore(graph.store.root)
    same_path_controller = ExperimentController(
        same_path_repository,
        experiment_ref=graph.experiment_ref,
        gate_batch_verifier=graph.gate_batch_service.verification_capability,
        mechanism_evidence_verifier=graph.mechanism_service.verification_capability,
    )
    assert (
        same_path_controller._skill_probe_authorizations.verification_capability.repository
        is not graph.store
    )
    with pytest.raises(SkillProbeExecutionAuthorizationError, match="not issued"):
        same_path_controller._skill_probe_authorizations.verification_capability.verify_skill_probe_execution_authorization(
            authorization_ref
        )


def test_skill_probe_execution_grant_replays_admission_and_exact_plan(
    graph: ProbeFixture,
) -> None:
    controller, valid, _ = _controller_at_probes(graph)
    forged_tail = CandidateJournal(graph.store).append(
        stream_id=f"candidate/{graph.candidate_ref.sha256}",
        previous_entry_ref=valid,
        event=CandidateLifecycleEvent(
            candidate_ref=graph.candidate_ref,
            from_state=CandidateState.VALID,
            to_state=CandidateState.RUNNING_PROBES,
            evidence_refs=(graph.policy_ref,),
            reason="policy substituted for plan",
        ),
    )
    controller._candidate_tails[graph.candidate_ref.sha256] = forged_tail

    with pytest.raises(ExperimentControllerError, match="wrong skill mechanism plan"):
        controller.issue_skill_probe_execution_authorization(
            candidate_ref=graph.candidate_ref,
            running_probes_tail_ref=forged_tail,
        )


def test_request_inclusion_cannot_be_given_a_claim_authority(graph: ProbeFixture) -> None:
    template = graph.policy.claim_authorities[0]
    with pytest.raises(ValidationError, match=r"request inclusion.*no claim authority"):
        _replace(
            template,
            claim=SkillMechanismClaim.REQUEST_INCLUSION,
            evidence_media_type="application/vnd.example.inclusion-evidence+json",
        )


def test_cross_claim_attestor_config_and_media_reuse_fail_closed(graph: ProbeFixture) -> None:
    first, second, *remaining = graph.policy.claim_authorities
    with pytest.raises(ValidationError, match="distinct attestor IDs"):
        _replace(
            graph.policy,
            claim_authorities=(
                first,
                _replace(second, attestor_id=first.attestor_id),
                *remaining,
            ),
        )
    with pytest.raises(ValidationError, match="distinct configuration"):
        _replace(
            graph.policy,
            claim_authorities=(
                first,
                _replace(second, config_ref=first.config_ref),
                *remaining,
            ),
        )
    wrong_media = SKILL_CLAIM_EVIDENCE_MEDIA_TYPES[second.claim]
    with pytest.raises(ValidationError, match="wrong evidence media type"):
        _replace(first, evidence_media_type=wrong_media)


def test_generic_aggregate_and_claim_attestors_cannot_be_reused(graph: ProbeFixture) -> None:
    with pytest.raises(ValidationError, match="attestors must all be distinct"):
        _replace(
            graph.policy,
            aggregate_attestor_id=graph.policy.generic_mechanism_attestor_id,
        )
    with pytest.raises(ValidationError, match="attestors must all be distinct"):
        _replace(
            graph.policy,
            generic_mechanism_attestor_id=graph.policy.claim_authorities[0].attestor_id,
        )


def test_replay_rejects_wrong_policy_candidate_and_split(graph: ProbeFixture) -> None:
    wrong_policy = _replace(graph.policy, reset_fingerprint="other-reset-v1")
    wrong_policy_ref = _put(
        graph.store,
        wrong_policy,
        SKILL_VERIFICATION_POLICY_MEDIA_TYPE,
    )
    wrong_policy_plan = _replace(
        graph.plan,
        policy_ref=wrong_policy_ref,
        policy_fingerprint=wrong_policy.fingerprint,
    )
    with pytest.raises(SkillProbePreregistrationError, match="protocol skill policy"):
        _verify(graph, _persist_plan(graph, wrong_policy_plan))

    wrong_candidate_ref = _put(
        graph.store,
        {"candidate": "foreign"},
        CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(SkillProbePreregistrationError, match="another lifecycle"):
        verify_skill_probe_preregistration(
            graph.store,
            plan_ref=graph.plan_ref,
            expected_experiment_ref=graph.experiment_ref,
            expected_protocol_ref=graph.protocol_ref,
            expected_candidate_ref=wrong_candidate_ref,
        )

    wrong_split_plan = _replace(
        graph.plan,
        exploration_split_ref=graph.gate_split_ref,
    )
    with pytest.raises(SkillProbePreregistrationError, match="exploration split"):
        _verify(graph, _persist_plan(graph, wrong_split_plan))


def test_replay_rejects_schedule_axes_outside_typed_roster(graph: ProbeFixture) -> None:
    wrong_revert = _replace(graph.plan.revert_schedule, repeat_seeds=(99,))
    wrong_placebo = _replace(graph.plan.placebo_schedule, repeat_seeds=(99,))
    plan = _replace(
        graph.plan,
        revert_schedule=wrong_revert,
        placebo_schedule=wrong_placebo,
    )
    with pytest.raises(SkillProbePreregistrationError, match="policy roster"):
        _verify(graph, _persist_plan(graph, plan))


def test_plan_rejects_swapped_schedules_missing_placebo_and_incomplete_sides(
    graph: ProbeFixture,
) -> None:
    with pytest.raises(ValidationError, match="revert schedule"):
        _replace(
            graph.plan,
            revert_schedule=graph.plan.placebo_schedule,
            placebo_schedule=graph.plan.revert_schedule,
        )

    payload = graph.plan.model_dump(mode="python", round_trip=True)
    payload.pop("placebo_schedule")
    with pytest.raises(ValidationError, match="placebo_schedule"):
        SkillMechanismPlan.model_validate(payload, strict=True)

    with pytest.raises(ValidationError, match="exactly parent and candidate"):
        _replace(
            graph.plan.placebo_schedule,
            sides=(EvaluationSide.PARENT,),
        )


def test_replay_rejects_candidate_authored_post_hoc_placebo(graph: ProbeFixture) -> None:
    authored_placebo = _replace(
        graph.after_package,
        rules=(
            SkillRule(
                rule_id=_CHANGED_RULE_ID,
                instruction="Trust the candidate-authored sham result.",
            ),
            graph.after_package.rules[1],
        ),
    )
    authored_package_ref = _put(graph.store, authored_placebo, SKILL_PACKAGE_MEDIA_TYPE)
    authored_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=authored_package_ref,
    )
    authored_harness = _replace(graph.child, components=(authored_component,))
    authored_harness_ref = _put(
        graph.store,
        authored_harness,
        HARNESS_MANIFEST_MEDIA_TYPE,
    )
    disclosure = SkillPackageLoader(graph.store).disclose(
        authored_package_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=_MODEL,
        runtime_fingerprint=_RUNTIME,
    )
    authored_control = _replace(
        graph.placebo_control,
        placebo_package_ref=authored_package_ref,
        placebo_harness_ref=authored_harness_ref,
        placebo_context_size_bytes=disclosure.context_size_bytes,
    )
    authored_control_ref = _put(
        graph.store,
        authored_control,
        SKILL_PLACEBO_CONTROL_MEDIA_TYPE,
    )
    authored_schedule = _replace(
        graph.plan.placebo_schedule,
        parent_harness_id=authored_harness_ref.sha256,
    )
    authored_plan = _replace(
        graph.plan,
        placebo_control_ref=authored_control_ref,
        placebo_harness_ref=authored_harness_ref,
        placebo_schedule=authored_schedule,
    )

    with pytest.raises(SkillProbePreregistrationError, match="trusted construction"):
        _verify(graph, _persist_plan(graph, authored_plan))


def test_replay_rejects_a_revision_that_removes_a_rule(graph: ProbeFixture) -> None:
    removed_package = _replace(
        graph.before_package,
        revision=1,
        parent_package_ref=graph.before_ref,
        rules=(graph.before_package.rules[1],),
    )
    removed_ref = _put(graph.store, removed_package, SKILL_PACKAGE_MEDIA_TYPE)
    removed_component = HarnessComponentRef(
        name=_SKILL_ID,
        kind=ComponentKind.SKILL,
        artifact=removed_ref,
    )
    mutation = _replace(graph.mutation, after=removed_component)
    mutation_ref = _put(graph.store, mutation, CANDIDATE_MUTATION_MEDIA_TYPE)
    child = HarnessRegistry(graph.experiment.mutation_policy).apply_mutation(
        parent=graph.parent,
        parent_ref=graph.parent_ref,
        mutation=mutation,
        artifact_bytes=graph.store.get_bytes(removed_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    child_ref = _put(graph.store, child, HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        graph.candidate,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    revert_schedule = _replace(
        graph.plan.revert_schedule,
        candidate_harness_id=child_ref.sha256,
    )
    placebo_schedule = _replace(
        graph.plan.placebo_schedule,
        candidate_harness_id=child_ref.sha256,
    )
    plan = _replace(
        graph.plan,
        candidate_ref=candidate_ref,
        mutation_ref=mutation_ref,
        candidate_harness_ref=child_ref,
        after_skill_package_ref=removed_ref,
        revert_schedule=revert_schedule,
        placebo_schedule=placebo_schedule,
    )
    plan_ref = _persist_plan(graph, plan)

    with pytest.raises(SkillProbePreregistrationError, match="does not support removed rules"):
        verify_skill_probe_preregistration(
            graph.store,
            plan_ref=plan_ref,
            expected_experiment_ref=graph.experiment_ref,
            expected_protocol_ref=graph.protocol_ref,
            expected_candidate_ref=candidate_ref,
        )


@pytest.mark.parametrize("missing_id", sorted(REQUIRED_SKILL_MECHANISM_IDS))
def test_gate_config_must_contain_each_of_the_seven_skill_checks(
    graph: ProbeFixture,
    missing_id: str,
) -> None:
    config = _replace(
        graph.gate_config,
        required_mechanism_checks=tuple(
            check for check in graph.gate_config.required_mechanism_checks if check != missing_id
        ),
    )
    with pytest.raises(SkillProbePreregistrationError, match="omits required"):
        _verify_rebound(graph, _rebind_gate_config(graph, config))


def test_gate_config_rejects_legacy_request_activation_id(graph: ProbeFixture) -> None:
    config = _replace(
        graph.gate_config,
        required_mechanism_checks=(
            *graph.gate_config.required_mechanism_checks,
            LEGACY_SKILL_REQUEST_ACTIVATION_MECHANISM_ID,
        ),
    )
    with pytest.raises(SkillProbePreregistrationError, match="legacy activation ID"):
        _verify_rebound(graph, _rebind_gate_config(graph, config))


def test_prompt_candidate_cannot_supply_a_skill_mechanism_plan(graph: ProbeFixture) -> None:
    before_ref = graph.store.put_bytes(b"Answer directly.", media_type="text/plain")
    after_ref = graph.store.put_bytes(b"Answer and verify.", media_type="text/plain")
    before = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=before_ref,
    )
    after = HarnessComponentRef(
        name="system",
        kind=ComponentKind.PROMPT,
        artifact=after_ref,
    )
    parent = HarnessManifest(
        model_fingerprint=_MODEL,
        runtime_fingerprint=_RUNTIME,
        trusted_plane_version=graph.protocol.trusted_plane_version,
        components=(before,),
        budget=BudgetPolicy(max_evaluations=10),
    )
    parent_ref = _put(graph.store, parent, HARNESS_MANIFEST_MEDIA_TYPE)
    evidence_ref = _put(
        graph.store,
        {"failure": "prompt omitted a verification step"},
        "application/vnd.spiral-harness.evidence+json",
    )
    mutation = CandidateMutation(
        target_component="system",
        before=before,
        after=after,
        hypothesis=MutationHypothesis(
            evidence_refs=(evidence_ref,),
            where="system prompt",
            why="verification was omitted",
            expected_activation="the prompt is included",
            expected_adherence="the answer is checked",
            expected_behavior="the answer is verified",
            expected_benefit="paired score improves",
            protected_slices=(),
            falsifier="no verification occurs",
            negative_control="the original prompt",
            risks=(),
        ),
    )
    mutation_ref = _put(graph.store, mutation, CANDIDATE_MUTATION_MEDIA_TYPE)
    policy = MutationPolicy(
        allowed_kinds=(ComponentKind.PROMPT,),
        allowed_component_names=("system",),
        allowed_media_types=("text/plain",),
    )
    experiment = ExperimentManifest(
        protocol_ref=graph.protocol_ref,
        seed_harness_ref=parent_ref,
        mutation_policy=policy,
        objective="improve paired score",
        baselines=("static",),
        stopping=("one-candidate",),
        search_budget=BudgetPolicy(max_evaluations=10),
    )
    experiment_ref = _put(graph.store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    child = HarnessRegistry(policy).apply_mutation(
        parent=parent,
        parent_ref=parent_ref,
        mutation=mutation,
        artifact_bytes=graph.store.get_bytes(after_ref),
        artifact_media_type="text/plain",
    )
    child_ref = _put(graph.store, child, HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = CandidateManifest(
        experiment_ref=experiment_ref,
        parent_harness_ref=parent_ref,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
        evidence_refs=(evidence_ref,),
        evaluation_plan_ref=graph.gate_config_ref,
    )
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    CandidateAdmissionService(graph.store).admit(
        candidate_ref=candidate_ref,
        experiment_ref=experiment_ref,
    )

    with pytest.raises(SkillProbePreregistrationError, match="non-skill candidate"):
        resolve_probe_preregistration(
            graph.store,
            experiment_ref=experiment_ref,
            protocol_ref=graph.protocol_ref,
            candidate_ref=candidate_ref,
            plan_ref=graph.plan_ref,
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("study", "candidate-picked-study"),
        ("kind", "candidate-picked-kind"),
        ("query", 9),
        ("master_seed", 999),
        ("task_ids", ("candidate-picked-task",)),
        ("search_runs", (3,)),
        ("repeat_seeds", (99,)),
        ("token_ceiling_per_attempt", 257),
    ),
)
def test_every_seed_and_budget_schedule_coordinate_is_frozen_by_the_roster(
    graph: ProbeFixture,
    field_name: str,
    wrong_value: object,
) -> None:
    plan = _replace(
        graph.plan,
        revert_schedule=_replace(
            graph.plan.revert_schedule,
            **{field_name: wrong_value},
        ),
        placebo_schedule=_replace(
            graph.plan.placebo_schedule,
            **{field_name: wrong_value},
        ),
    )

    with pytest.raises(SkillProbePreregistrationError, match="policy roster"):
        _verify(graph, _persist_plan(graph, plan))


def test_skill_probe_roster_rejects_retry_semantics_not_defined_by_v1(
    graph: ProbeFixture,
) -> None:
    values = graph.roster.model_dump(mode="python", round_trip=True, warnings="none")
    values["max_attempts_per_cell"] = 2

    with pytest.raises(ValidationError):
        SkillProbeRoster.model_validate(values, strict=True)


def test_placebo_path_rejects_semantic_reordering_of_existing_rules(
    graph: ProbeFixture,
) -> None:
    reordered = _replace(
        graph.after_package,
        rules=(graph.after_package.rules[1], graph.after_package.rules[0]),
    )
    reordered_ref = _put(graph.store, reordered, SKILL_PACKAGE_MEDIA_TYPE)
    component = _replace(graph.mutation.after, artifact=reordered_ref)
    mutation = _replace(graph.mutation, after=component)
    mutation_ref = _put(graph.store, mutation, CANDIDATE_MUTATION_MEDIA_TYPE)
    child = HarnessRegistry(graph.experiment.mutation_policy).apply_mutation(
        parent=graph.parent,
        parent_ref=graph.parent_ref,
        mutation=mutation,
        artifact_bytes=graph.store.get_bytes(reordered_ref),
        artifact_media_type=SKILL_PACKAGE_MEDIA_TYPE,
    )
    child_ref = _put(graph.store, child, HARNESS_MANIFEST_MEDIA_TYPE)
    candidate = _replace(
        graph.candidate,
        child_harness_ref=child_ref,
        mutation_ref=mutation_ref,
    )
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    plan = _replace(
        graph.plan,
        candidate_ref=candidate_ref,
        mutation_ref=mutation_ref,
        candidate_harness_ref=child_ref,
        after_skill_package_ref=reordered_ref,
        revert_schedule=_replace(
            graph.plan.revert_schedule,
            candidate_harness_id=child_ref.sha256,
        ),
        placebo_schedule=_replace(
            graph.plan.placebo_schedule,
            candidate_harness_id=child_ref.sha256,
        ),
    )

    with pytest.raises(SkillProbePreregistrationError, match="reordering existing rules"):
        verify_skill_probe_preregistration(
            graph.store,
            plan_ref=_persist_plan(graph, plan),
            expected_experiment_ref=graph.experiment_ref,
            expected_protocol_ref=graph.protocol_ref,
            expected_candidate_ref=candidate_ref,
        )


@pytest.mark.parametrize("authority", ("generic", "aggregate", "claim"))
def test_gate_batch_attestor_cannot_attest_skill_claims(
    graph: ProbeFixture,
    authority: str,
) -> None:
    gate_attestor = {
        "generic": graph.policy.generic_mechanism_attestor_id,
        "aggregate": graph.policy.aggregate_attestor_id,
        "claim": graph.policy.claim_authorities[0].attestor_id,
    }[authority]
    protocol = _replace(graph.protocol, gate_batch_attestor_id=gate_attestor)
    protocol_ref = _put(graph.store, protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    experiment = _replace(graph.experiment, protocol_ref=protocol_ref)
    experiment_ref = _put(graph.store, experiment, EXPERIMENT_MANIFEST_MEDIA_TYPE)
    candidate = _replace(graph.candidate, experiment_ref=experiment_ref)
    candidate_ref = _put(graph.store, candidate, CANDIDATE_MANIFEST_MEDIA_TYPE)
    plan = _replace(
        graph.plan,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
    )

    with pytest.raises(SkillProbePreregistrationError, match="attestors must be distinct"):
        verify_skill_probe_preregistration(
            graph.store,
            plan_ref=_persist_plan(graph, plan),
            expected_experiment_ref=experiment_ref,
            expected_protocol_ref=protocol_ref,
            expected_candidate_ref=candidate_ref,
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("model_spec_fingerprint", "8" * 64),
        ("runtime_fingerprint", "other-runtime"),
        ("probe_grader_fingerprint", "other-probe-grader"),
        ("reset_fingerprint", "other-reset"),
        ("execution_order_fingerprint", "other-order"),
    ),
)
def test_plan_execution_context_must_equal_policy(
    graph: ProbeFixture,
    field_name: str,
    wrong_value: str,
) -> None:
    plan = _replace(graph.plan, **{field_name: wrong_value})

    with pytest.raises(SkillProbePreregistrationError, match="differs from policy"):
        _verify(graph, _persist_plan(graph, plan))


def test_policy_enabled_skill_requires_a_plan_and_policy_less_skill_stays_quarantined(
    graph: ProbeFixture,
) -> None:
    with pytest.raises(SkillProbePreregistrationError, match="requires a preregistered"):
        resolve_probe_preregistration(
            graph.store,
            experiment_ref=graph.experiment_ref,
            protocol_ref=graph.protocol_ref,
            candidate_ref=graph.candidate_ref,
            plan_ref=None,
        )

    legacy_protocol = _replace(graph.protocol, skill_verification_policy_ref=None)
    legacy_protocol_ref = _put(graph.store, legacy_protocol, PROTOCOL_MANIFEST_MEDIA_TYPE)
    legacy_experiment = _replace(graph.experiment, protocol_ref=legacy_protocol_ref)
    legacy_experiment_ref = _put(
        graph.store,
        legacy_experiment,
        EXPERIMENT_MANIFEST_MEDIA_TYPE,
    )
    legacy_candidate = _replace(graph.candidate, experiment_ref=legacy_experiment_ref)
    legacy_candidate_ref = _put(
        graph.store,
        legacy_candidate,
        CANDIDATE_MANIFEST_MEDIA_TYPE,
    )
    assert (
        resolve_probe_preregistration(
            graph.store,
            experiment_ref=legacy_experiment_ref,
            protocol_ref=legacy_protocol_ref,
            candidate_ref=legacy_candidate_ref,
            plan_ref=None,
        )
        == ()
    )
    with pytest.raises(SkillProbePreregistrationError, match="does not enable"):
        resolve_probe_preregistration(
            graph.store,
            experiment_ref=legacy_experiment_ref,
            protocol_ref=legacy_protocol_ref,
            candidate_ref=legacy_candidate_ref,
            plan_ref=graph.plan_ref,
        )


def test_probe_lifecycle_replay_rejects_missing_extra_and_foreign_plan_refs(
    graph: ProbeFixture,
) -> None:
    for evidence_refs in ((), (graph.plan_ref, graph.policy_ref), (graph.policy_ref,)):
        with pytest.raises(SkillProbePreregistrationError):
            replay_probe_preregistration_refs(
                graph.store,
                experiment_ref=graph.experiment_ref,
                protocol_ref=graph.protocol_ref,
                candidate_ref=graph.candidate_ref,
                evidence_refs=evidence_refs,
            )


def test_revert_schedule_must_use_the_exact_candidate_parent(graph: ProbeFixture) -> None:
    wrong_parent = _replace(graph.parent, budget=BudgetPolicy(max_evaluations=49))
    wrong_parent_ref = _put(graph.store, wrong_parent, HARNESS_MANIFEST_MEDIA_TYPE)

    with pytest.raises(ValidationError, match="revert schedule"):
        _replace(
            graph.plan,
            revert_schedule=_replace(
                graph.plan.revert_schedule,
                parent_harness_id=wrong_parent_ref.sha256,
            ),
        )
