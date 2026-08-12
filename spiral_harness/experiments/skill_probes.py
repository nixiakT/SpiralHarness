"""Trusted preregistration replay for matched skill mechanism probes."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.harness.registry import HarnessRegistry
from spiral_harness.skills.loading import SkillDisclosureLevel, SkillPackageLoader
from spiral_harness.skills.package import SKILL_PACKAGE_MEDIA_TYPE, SkillPackage
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.mechanism import (
    LEGACY_SKILL_REQUEST_ACTIVATION_MECHANISM_ID,
    REQUIRED_SKILL_MECHANISM_IDS,
)
from spiral_harness.verification.models import GateConfig
from spiral_harness.verification.skill_plan import (
    NEUTRAL_SKILL_RULES_MEDIA_TYPE,
    SKILL_MECHANISM_PLAN_MEDIA_TYPE,
    SKILL_PLACEBO_CONTROL_MEDIA_TYPE,
    SKILL_PROBE_ROSTER_MEDIA_TYPE,
    SKILL_VERIFICATION_POLICY_MEDIA_TYPE,
    NeutralSkillRules,
    SkillMechanismPlan,
    SkillPlaceboControl,
    SkillProbeRoster,
    SkillVerificationPolicy,
)


class SkillProbePreregistrationError(ValueError):
    """Raised when a skill probe plan is not an exact, pre-execution closure."""


def _load_exact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    media_type: str,
    label: str,
) -> ModelT:
    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
    except Exception as exc:
        raise SkillProbePreregistrationError(f"{label} reference is malformed") from exc
    if checked_ref.media_type != media_type:
        raise SkillProbePreregistrationError(f"{label} declares the wrong media type")
    try:
        payload = repository.get_bytes(checked_ref)
        value = repository.get_json(checked_ref, model_type)
        checked = model_type.model_validate(value, strict=True)
        canonical = canonical_json_bytes(checked)
    except Exception as exc:
        raise SkillProbePreregistrationError(f"{label} cannot be loaded exactly") from exc
    if (
        payload != canonical
        or len(payload) != checked_ref.size
        or sha256_bytes(payload) != checked_ref.sha256
    ):
        raise SkillProbePreregistrationError(f"{label} is not canonical under its reference")
    return checked


def _require_sources(repository: ArtifactRepository, refs: Iterable[ArtifactRef]) -> None:
    for ref in refs:
        try:
            repository.get_bytes(ref)
        except Exception as exc:
            raise SkillProbePreregistrationError(
                "a preregistered skill probe configuration is unavailable"
            ) from exc


def _artifact_ref(value: BaseModel, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(sha256=sha256_bytes(payload), size=len(payload), media_type=media_type)


def _changed_rule_ids(
    before: SkillPackage,
    after: SkillPackage,
) -> tuple[str, ...]:
    before_by_id = {rule.rule_id: rule for rule in before.rules}
    after_by_id = {rule.rule_id: rule for rule in after.rules}
    return tuple(
        sorted(
            rule_id
            for rule_id in before_by_id.keys() | after_by_id.keys()
            if before_by_id.get(rule_id) != after_by_id.get(rule_id)
        )
    )


def _placebo_package(
    after: SkillPackage,
    neutral: NeutralSkillRules,
) -> SkillPackage:
    replacements = {rule.rule_id: rule for rule in neutral.rules}
    missing = tuple(sorted(replacements.keys() - {rule.rule_id for rule in after.rules}))
    if missing:
        raise SkillProbePreregistrationError(
            "neutral rules refer to rule IDs absent from the candidate package"
        )
    rules = tuple(replacements.get(rule.rule_id, rule) for rule in after.rules)
    try:
        return SkillPackage.model_validate(
            after.model_copy(update={"rules": rules}).model_dump(
                mode="python",
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )
    except Exception as exc:
        raise SkillProbePreregistrationError("placebo package cannot be reproduced") from exc


def _placebo_harness(
    child: HarnessManifest,
    *,
    target_component: str,
    placebo_package_ref: ArtifactRef,
) -> HarnessManifest:
    components = tuple(
        HarnessComponentRef(
            name=component.name,
            kind=component.kind,
            artifact=placebo_package_ref,
        )
        if component.name == target_component
        else component
        for component in child.components
    )
    try:
        return HarnessManifest.model_validate(
            child.model_copy(update={"components": components}).model_dump(
                mode="python",
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )
    except Exception as exc:
        raise SkillProbePreregistrationError("placebo harness cannot be reproduced") from exc


def _verify_policy(
    repository: ArtifactRepository,
    *,
    protocol: ProtocolManifest,
    plan: SkillMechanismPlan,
) -> tuple[SkillVerificationPolicy, SkillProbeRoster, NeutralSkillRules]:
    if protocol.skill_verification_policy_ref != plan.policy_ref:
        raise SkillProbePreregistrationError("plan does not use the protocol skill policy")
    policy = _load_exact(
        repository,
        plan.policy_ref,
        SkillVerificationPolicy,
        media_type=SKILL_VERIFICATION_POLICY_MEDIA_TYPE,
        label="skill verification policy",
    )
    if policy.fingerprint != plan.policy_fingerprint:
        raise SkillProbePreregistrationError("plan skill policy fingerprint changed")
    protocol_coordinates = (
        protocol.model_fingerprint,
        protocol.model_spec_fingerprint,
        protocol.inference_fingerprint,
        protocol.runtime_fingerprint,
        protocol.grader_fingerprint,
        protocol.mechanism_evidence_attestor_id,
    )
    policy_coordinates = (
        policy.model_fingerprint,
        policy.model_spec_fingerprint,
        policy.inference_fingerprint,
        policy.runtime_fingerprint,
        policy.grader_fingerprint,
        policy.generic_mechanism_attestor_id,
    )
    if policy_coordinates != protocol_coordinates:
        raise SkillProbePreregistrationError("skill policy differs from the frozen protocol")
    skill_attestors = {
        policy.generic_mechanism_attestor_id,
        policy.aggregate_attestor_id,
        *(authority.attestor_id for authority in policy.claim_authorities),
    }
    if protocol.gate_batch_attestor_id in skill_attestors:
        raise SkillProbePreregistrationError(
            "gate-batch and skill-mechanism attestors must be distinct"
        )
    if (
        plan.evidence_profile is not policy.evidence_profile
        or plan.model_spec_fingerprint != policy.model_spec_fingerprint
        or plan.runtime_fingerprint != policy.runtime_fingerprint
        or plan.probe_grader_fingerprint != policy.probe_grader_fingerprint
        or plan.reset_fingerprint != policy.reset_fingerprint
        or plan.execution_order_fingerprint != policy.execution_order_fingerprint
    ):
        raise SkillProbePreregistrationError("skill plan execution context differs from policy")

    roster = _load_exact(
        repository,
        policy.task_roster_ref,
        SkillProbeRoster,
        media_type=SKILL_PROBE_ROSTER_MEDIA_TYPE,
        label="skill probe roster",
    )
    neutral = _load_exact(
        repository,
        policy.neutral_rules_ref,
        NeutralSkillRules,
        media_type=NEUTRAL_SKILL_RULES_MEDIA_TYPE,
        label="neutral skill rules",
    )
    if plan.probe_roster_ref != policy.task_roster_ref:
        raise SkillProbePreregistrationError("plan does not use the policy probe roster")
    if plan.probe_roster_fingerprint != roster.fingerprint:
        raise SkillProbePreregistrationError("plan probe roster fingerprint changed")
    if roster.evidence_profile is not policy.evidence_profile:
        raise SkillProbePreregistrationError("probe roster uses another evidence profile")
    if len(roster.task_ids) < policy.min_probe_tasks:
        raise SkillProbePreregistrationError("probe roster is below the policy task minimum")
    _require_sources(
        repository,
        (
            *(authority.config_ref for authority in policy.claim_authorities),
            *roster.adherence_probe_refs,
            *roster.behavior_probe_refs,
        ),
    )
    return policy, roster, neutral


def _verify_candidate(
    repository: ArtifactRepository,
    *,
    experiment: ExperimentManifest,
    protocol: ProtocolManifest,
    candidate: CandidateManifest,
    plan: SkillMechanismPlan,
) -> tuple[CandidateMutation, HarnessManifest, HarnessManifest, SkillPackage, SkillPackage]:
    if candidate.mutation_ref != plan.mutation_ref:
        raise SkillProbePreregistrationError("plan belongs to another candidate mutation")
    if candidate.evaluation_plan_ref != protocol.gate_config_ref:
        raise SkillProbePreregistrationError("candidate does not use the protocol gate config")
    if (
        candidate.parent_harness_ref != plan.parent_harness_ref
        or candidate.child_harness_ref != plan.candidate_harness_ref
    ):
        raise SkillProbePreregistrationError("plan harness lineage differs from the candidate")
    mutation = _load_exact(
        repository,
        candidate.mutation_ref,
        CandidateMutation,
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
        label="candidate mutation",
    )
    if mutation.after.kind is not ComponentKind.SKILL:
        raise SkillProbePreregistrationError("skill probe plan requires a skill mutation")
    if experiment.mutation_policy.allowed_kinds != (ComponentKind.SKILL,):
        raise SkillProbePreregistrationError("first skill probe policy must be skill-only")
    if (
        mutation.target_component != plan.target_skill_id
        or mutation.before.artifact != plan.before_skill_package_ref
        or mutation.after.artifact != plan.after_skill_package_ref
    ):
        raise SkillProbePreregistrationError("plan does not bind the exact skill mutation")
    parent = _load_exact(
        repository,
        candidate.parent_harness_ref,
        HarnessManifest,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        label="parent harness",
    )
    child = _load_exact(
        repository,
        candidate.child_harness_ref,
        HarnessManifest,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        label="candidate harness",
    )
    expected_harness_context = (
        protocol.model_fingerprint,
        protocol.runtime_fingerprint,
        protocol.trusted_plane_version,
    )
    for harness in (parent, child):
        if (
            harness.model_fingerprint,
            harness.runtime_fingerprint,
            harness.trusted_plane_version,
        ) != expected_harness_context:
            raise SkillProbePreregistrationError("skill harness differs from protocol context")
    expected_child = HarnessRegistry(experiment.mutation_policy).apply_mutation(
        parent=parent,
        parent_ref=candidate.parent_harness_ref,
        mutation=mutation,
        artifact_bytes=repository.get_bytes(mutation.after.artifact),
        artifact_media_type=mutation.after.artifact.media_type,
    )
    if child != expected_child:
        raise SkillProbePreregistrationError("candidate harness is not the atomic skill revision")
    loader = SkillPackageLoader(repository)
    try:
        before, after = loader.verify_revision(
            before_ref=plan.before_skill_package_ref,
            after_ref=plan.after_skill_package_ref,
            expected_component_name=plan.target_skill_id,
            model_fingerprint=protocol.model_fingerprint,
            runtime_fingerprint=protocol.runtime_fingerprint,
        )
    except Exception as exc:
        raise SkillProbePreregistrationError("skill revision cannot be replayed") from exc
    changed = _changed_rule_ids(before, after)
    if changed != plan.changed_rule_ids:
        raise SkillProbePreregistrationError("plan changed_rule_ids differ from the skill revision")
    before_rule_ids = tuple(rule.rule_id for rule in before.rules)
    after_rule_ids = tuple(rule.rule_id for rule in after.rules)
    if not set(before_rule_ids).issubset(after_rule_ids):
        raise SkillProbePreregistrationError("first placebo path does not support removed rules")
    retained_order = tuple(rule_id for rule_id in after_rule_ids if rule_id in before_rule_ids)
    if retained_order != before_rule_ids:
        raise SkillProbePreregistrationError(
            "first placebo path does not support reordering existing rules"
        )
    return mutation, parent, child, before, after


def _verify_placebo(
    repository: ArtifactRepository,
    *,
    protocol: ProtocolManifest,
    policy: SkillVerificationPolicy,
    neutral: NeutralSkillRules,
    mutation: CandidateMutation,
    child: HarnessManifest,
    after: SkillPackage,
    plan: SkillMechanismPlan,
) -> None:
    control = _load_exact(
        repository,
        plan.placebo_control_ref,
        SkillPlaceboControl,
        media_type=SKILL_PLACEBO_CONTROL_MEDIA_TYPE,
        label="skill placebo control",
    )
    expected_fields = (
        (control.evidence_profile, policy.evidence_profile),
        (control.candidate_package_ref, plan.after_skill_package_ref),
        (control.candidate_harness_ref, plan.candidate_harness_ref),
        (control.placebo_harness_ref, plan.placebo_harness_ref),
        (control.neutral_rules_ref, policy.neutral_rules_ref),
        (control.changed_rule_ids, plan.changed_rule_ids),
        (control.placebo_builder_fingerprint, policy.placebo_builder_fingerprint),
        (control.candidate_rule_count, len(after.rules)),
        (control.placebo_rule_count, len(after.rules)),
    )
    if any(actual != expected for actual, expected in expected_fields):
        raise SkillProbePreregistrationError("placebo control differs from its frozen plan")
    if neutral.rule_ids != plan.changed_rule_ids:
        raise SkillProbePreregistrationError("neutral rules do not cover the changed rules")
    placebo = _placebo_package(after, neutral)
    if placebo.artifact_ref != control.placebo_package_ref:
        raise SkillProbePreregistrationError("placebo package differs from trusted construction")
    persisted_placebo = _load_exact(
        repository,
        control.placebo_package_ref,
        SkillPackage,
        media_type=SKILL_PACKAGE_MEDIA_TYPE,
        label="placebo skill package",
    )
    if persisted_placebo != placebo or control.placebo_package_ref in {
        plan.before_skill_package_ref,
        plan.after_skill_package_ref,
    }:
        raise SkillProbePreregistrationError("placebo package is not a distinct matched sibling")

    loader = SkillPackageLoader(repository)
    candidate_disclosure = loader.disclose(
        plan.after_skill_package_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=protocol.model_fingerprint,
        runtime_fingerprint=protocol.runtime_fingerprint,
    )
    placebo_disclosure = loader.disclose(
        control.placebo_package_ref,
        level=SkillDisclosureLevel.RULES,
        model_fingerprint=protocol.model_fingerprint,
        runtime_fingerprint=protocol.runtime_fingerprint,
    )
    if (
        control.candidate_context_size_bytes != candidate_disclosure.context_size_bytes
        or control.placebo_context_size_bytes != placebo_disclosure.context_size_bytes
    ):
        raise SkillProbePreregistrationError("placebo disclosure sizes were not replayed")
    size_delta = abs(control.candidate_context_size_bytes - control.placebo_context_size_bytes)
    if size_delta > policy.max_placebo_context_size_delta:
        raise SkillProbePreregistrationError("placebo disclosure exceeds the size-match policy")

    expected_harness = _placebo_harness(
        child,
        target_component=mutation.target_component,
        placebo_package_ref=control.placebo_package_ref,
    )
    expected_ref = _artifact_ref(expected_harness, HARNESS_MANIFEST_MEDIA_TYPE)
    if expected_ref != plan.placebo_harness_ref:
        raise SkillProbePreregistrationError("placebo harness differs from trusted construction")
    persisted_harness = _load_exact(
        repository,
        plan.placebo_harness_ref,
        HarnessManifest,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        label="placebo harness",
    )
    if persisted_harness != expected_harness or plan.placebo_harness_ref in {
        plan.parent_harness_ref,
        plan.candidate_harness_ref,
    }:
        raise SkillProbePreregistrationError("placebo harness is not a distinct matched sibling")


def _verify_schedules(
    repository: ArtifactRepository,
    *,
    protocol: ProtocolManifest,
    roster: SkillProbeRoster,
    plan: SkillMechanismPlan,
) -> None:
    exploration_ref = next(
        split.manifest_ref
        for split in protocol.splits
        if split.partition is ProtocolPartition.EXPLORATION
    )
    if (
        plan.exploration_split_ref != exploration_ref
        or roster.exploration_split_ref != exploration_ref
    ):
        raise SkillProbePreregistrationError("skill probes do not use the exploration split")
    for schedule in (plan.revert_schedule, plan.placebo_schedule):
        coordinates = (
            schedule.study,
            schedule.kind,
            schedule.query,
            schedule.master_seed,
            schedule.task_ids,
            schedule.search_runs,
            schedule.repeat_seeds,
            schedule.max_attempts_per_cell,
            schedule.token_ceiling_per_attempt,
        )
        expected = (
            roster.study,
            roster.kind,
            roster.query,
            roster.master_seed,
            roster.task_ids,
            roster.search_runs,
            roster.repeat_seeds,
            roster.max_attempts_per_cell,
            roster.token_ceiling_per_attempt,
        )
        if coordinates != expected:
            raise SkillProbePreregistrationError("probe schedule differs from the policy roster")
    gate_config = _load_exact(
        repository,
        protocol.gate_config_ref,
        GateConfig,
        media_type=protocol.gate_config_ref.media_type,
        label="skill gate config",
    )
    required = set(gate_config.required_mechanism_checks)
    if not REQUIRED_SKILL_MECHANISM_IDS.issubset(required):
        raise SkillProbePreregistrationError("gate config omits required skill mechanism checks")
    if LEGACY_SKILL_REQUEST_ACTIVATION_MECHANISM_ID in required:
        raise SkillProbePreregistrationError("gate config still requires the legacy activation ID")


def verify_skill_probe_preregistration(
    repository: ArtifactRepository,
    *,
    plan_ref: ArtifactRef,
    expected_experiment_ref: ArtifactRef,
    expected_protocol_ref: ArtifactRef,
    expected_candidate_ref: ArtifactRef,
) -> SkillMechanismPlan:
    """Replay every frozen plan join without accepting a caller-authored verdict."""

    if not isinstance(repository, ArtifactRepository):
        raise SkillProbePreregistrationError("repository must implement ArtifactRepository")
    plan = _load_exact(
        repository,
        plan_ref,
        SkillMechanismPlan,
        media_type=SKILL_MECHANISM_PLAN_MEDIA_TYPE,
        label="skill mechanism plan",
    )
    expected_refs = (
        (plan.experiment_ref, expected_experiment_ref),
        (plan.protocol_ref, expected_protocol_ref),
        (plan.candidate_ref, expected_candidate_ref),
    )
    if any(actual != expected for actual, expected in expected_refs):
        raise SkillProbePreregistrationError("skill plan belongs to another lifecycle")
    experiment = _load_exact(
        repository,
        expected_experiment_ref,
        ExperimentManifest,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
        label="experiment",
    )
    protocol = _load_exact(
        repository,
        expected_protocol_ref,
        ProtocolManifest,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
        label="protocol",
    )
    candidate = _load_exact(
        repository,
        expected_candidate_ref,
        CandidateManifest,
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
        label="candidate",
    )
    if experiment.protocol_ref != expected_protocol_ref:
        raise SkillProbePreregistrationError("experiment uses another protocol")
    if candidate.experiment_ref != expected_experiment_ref:
        raise SkillProbePreregistrationError("candidate uses another experiment")
    policy, roster, neutral = _verify_policy(repository, protocol=protocol, plan=plan)
    mutation, _, child, _, after = _verify_candidate(
        repository,
        experiment=experiment,
        protocol=protocol,
        candidate=candidate,
        plan=plan,
    )
    _verify_placebo(
        repository,
        protocol=protocol,
        policy=policy,
        neutral=neutral,
        mutation=mutation,
        child=child,
        after=after,
        plan=plan,
    )
    _verify_schedules(repository, protocol=protocol, roster=roster, plan=plan)
    return plan


def resolve_probe_preregistration(
    repository: ArtifactRepository,
    *,
    experiment_ref: ArtifactRef,
    protocol_ref: ArtifactRef,
    candidate_ref: ArtifactRef,
    plan_ref: ArtifactRef | None,
) -> tuple[ArtifactRef, ...]:
    """Return lifecycle evidence refs; policy-less skills stay quarantined."""

    experiment = _load_exact(
        repository,
        experiment_ref,
        ExperimentManifest,
        media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
        label="experiment",
    )
    if experiment.protocol_ref != protocol_ref:
        raise SkillProbePreregistrationError("experiment uses another protocol")
    candidate = _load_exact(
        repository,
        candidate_ref,
        CandidateManifest,
        media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
        label="candidate",
    )
    if candidate.experiment_ref != experiment_ref:
        raise SkillProbePreregistrationError("candidate belongs to another experiment")
    mutation = _load_exact(
        repository,
        candidate.mutation_ref,
        CandidateMutation,
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
        label="candidate mutation",
    )
    protocol = _load_exact(
        repository,
        protocol_ref,
        ProtocolManifest,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
        label="protocol",
    )
    if mutation.after.kind is not ComponentKind.SKILL:
        if plan_ref is not None:
            raise SkillProbePreregistrationError("non-skill candidate supplied a skill probe plan")
        return ()
    if protocol.skill_verification_policy_ref is None:
        if plan_ref is not None:
            raise SkillProbePreregistrationError("protocol does not enable skill verification")
        return ()
    if plan_ref is None:
        raise SkillProbePreregistrationError(
            "skill-verification protocol requires a preregistered probe plan"
        )
    verified = verify_skill_probe_preregistration(
        repository,
        plan_ref=plan_ref,
        expected_experiment_ref=experiment_ref,
        expected_protocol_ref=protocol_ref,
        expected_candidate_ref=candidate_ref,
    )
    del verified
    return (ArtifactRef.model_validate(plan_ref, strict=True),)


def replay_probe_preregistration_refs(
    repository: ArtifactRepository,
    *,
    experiment_ref: ArtifactRef,
    protocol_ref: ArtifactRef,
    candidate_ref: ArtifactRef,
    evidence_refs: tuple[ArtifactRef, ...],
) -> None:
    """Re-derive the exact lifecycle refs recorded before probe execution."""

    if len(evidence_refs) > 1:
        raise SkillProbePreregistrationError("probe lifecycle contains extra evidence refs")
    plan_ref = evidence_refs[0] if evidence_refs else None
    expected = resolve_probe_preregistration(
        repository,
        experiment_ref=experiment_ref,
        protocol_ref=protocol_ref,
        candidate_ref=candidate_ref,
        plan_ref=plan_ref,
    )
    if evidence_refs != expected:
        raise SkillProbePreregistrationError("probe lifecycle preregistration changed")


__all__ = [
    "SkillProbePreregistrationError",
    "replay_probe_preregistration_refs",
    "resolve_probe_preregistration",
    "verify_skill_probe_preregistration",
]
