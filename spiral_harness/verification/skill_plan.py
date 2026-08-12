"""Candidate-bound preregistration contracts for skill mechanism probes.

Request inclusion is intentionally a separate claim: it is re-derived from
settled execution receipts and never receives a claim attestor.  The remaining
six mechanism claims each require a distinct authority, and their aggregate
attestor must also be distinct so a signature cannot be replayed across claims.

The placebo types below freeze a deterministic control construction.  They do
not assert that the resulting text is semantically neutral, nor do they turn a
placebo sibling into a revision eligible for promotion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.schedule import EvaluationBatchSchedule, EvaluationPhase
from spiral_harness.skills.package import (
    SKILL_PACKAGE_MEDIA_TYPE,
    SkillRule,
    Slug,
)
from spiral_harness.verification.mechanism import REQUIRED_SKILL_MECHANISM_IDS
from spiral_harness.verification.skill_inclusion import (
    SKILL_REQUEST_INCLUSION_VERIFIER_FINGERPRINT,
)

SKILL_VERIFICATION_POLICY_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-verification-policy.v1+json"
)
SKILL_PROBE_ROSTER_MEDIA_TYPE = "application/vnd.spiral-harness.skill-probe-roster.v1+json"
SKILL_ADHERENCE_PROBE_MEDIA_TYPE = "application/vnd.spiral-harness.skill-adherence-probe.v1+json"
SKILL_BEHAVIOR_PROBE_MEDIA_TYPE = "application/vnd.spiral-harness.skill-behavior-probe.v1+json"
NEUTRAL_SKILL_RULES_MEDIA_TYPE = "application/vnd.spiral-harness.neutral-skill-rules.v1+json"
SKILL_PLACEBO_CONTROL_MEDIA_TYPE = "application/vnd.spiral-harness.skill-placebo-control.v1+json"
SKILL_MECHANISM_PLAN_MEDIA_TYPE = "application/vnd.spiral-harness.skill-mechanism-plan.v1+json"
SKILL_PROVIDER_DELIVERY_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-provider-delivery-evidence.v1+json"
)
SKILL_RUNTIME_ACTIVATION_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-runtime-activation-evidence.v1+json"
)
SKILL_ADHERENCE_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-adherence-evidence.v1+json"
)
SKILL_BEHAVIOR_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-behavior-evidence.v1+json"
)
SKILL_REVERT_CONTROL_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-revert-control-evidence.v1+json"
)
SKILL_PLACEBO_CONTROL_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-placebo-control-evidence.v1+json"
)

NEUTRAL_SKILL_RULE_INSTRUCTION = "This preregistered placebo marker specifies no task behavior."
_NEUTRAL_RULE_RENDERER_ID = "spiral-harness/neutral-skill-rule/v1"
NEUTRAL_SKILL_RULE_RENDERER_FINGERPRINT = sha256_bytes(_NEUTRAL_RULE_RENDERER_ID.encode("utf-8"))
SKILL_PLACEBO_BUILDER_FINGERPRINT = sha256_bytes(b"spiral-harness/matched-skill-placebo-builder/v1")


class SkillEvidenceProfile(StrEnum):
    """Execution environment in which one claim authority may produce evidence."""

    CONTROLLED_REPLAY = "controlled_replay"
    LIVE_PROVIDER = "live_provider"


class SkillMechanismClaim(StrEnum):
    """Closed taxonomy; request inclusion has no producer attestor."""

    REQUEST_INCLUSION = "skill_request_inclusion"
    PROVIDER_DELIVERY = "skill_provider_delivery"
    RUNTIME_ACTIVATION = "skill_runtime_activation"
    ADHERENCE = "skill_adherence"
    BEHAVIOR = "skill_behavior"
    REVERT_CONTROL = "skill_revert_control"
    PLACEBO_CONTROL = "skill_placebo_control"


ATTESTED_SKILL_MECHANISM_CLAIMS = (
    SkillMechanismClaim.PROVIDER_DELIVERY,
    SkillMechanismClaim.RUNTIME_ACTIVATION,
    SkillMechanismClaim.ADHERENCE,
    SkillMechanismClaim.BEHAVIOR,
    SkillMechanismClaim.REVERT_CONTROL,
    SkillMechanismClaim.PLACEBO_CONTROL,
)
if frozenset(claim.value for claim in SkillMechanismClaim) != REQUIRED_SKILL_MECHANISM_IDS:
    raise RuntimeError("skill plan claims drifted from the reserved gate taxonomy")
SKILL_CLAIM_EVIDENCE_MEDIA_TYPES = {
    SkillMechanismClaim.PROVIDER_DELIVERY: SKILL_PROVIDER_DELIVERY_EVIDENCE_MEDIA_TYPE,
    SkillMechanismClaim.RUNTIME_ACTIVATION: SKILL_RUNTIME_ACTIVATION_EVIDENCE_MEDIA_TYPE,
    SkillMechanismClaim.ADHERENCE: SKILL_ADHERENCE_EVIDENCE_MEDIA_TYPE,
    SkillMechanismClaim.BEHAVIOR: SKILL_BEHAVIOR_EVIDENCE_MEDIA_TYPE,
    SkillMechanismClaim.REVERT_CONTROL: SKILL_REVERT_CONTROL_EVIDENCE_MEDIA_TYPE,
    SkillMechanismClaim.PLACEBO_CONTROL: SKILL_PLACEBO_CONTROL_EVIDENCE_MEDIA_TYPE,
}


def _require_media_type(ref: ArtifactRef, expected: str, field_name: str) -> None:
    if ref.media_type != expected:
        raise ValueError(f"{field_name} must declare the exact {expected} media type")


def _require_json_ref(ref: ArtifactRef, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _artifact_ref(value: ImmutableModel, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=sha256_bytes(payload),
        size=len(payload),
        media_type=media_type,
    )


class SkillClaimAuthority(ImmutableModel):
    """One claim-specific producer and verifier boundary."""

    schema_version: Literal["1"] = "1"
    claim: SkillMechanismClaim
    evidence_profile: SkillEvidenceProfile
    attestor_id: Sha256
    attestation_domain: NonEmptyStr
    producer_fingerprint: NonEmptyStr
    verifier_fingerprint: NonEmptyStr
    evidence_media_type: NonEmptyStr
    config_ref: ArtifactRef

    @model_validator(mode="after")
    def request_inclusion_has_no_attestor(self) -> Self:
        if self.claim is SkillMechanismClaim.REQUEST_INCLUSION:
            raise ValueError(
                "request inclusion is live-ledger re-derived and has no claim authority"
            )
        expected_media_type = SKILL_CLAIM_EVIDENCE_MEDIA_TYPES[self.claim]
        if self.evidence_media_type != expected_media_type:
            raise ValueError("claim authority declares the wrong evidence media type")
        _require_json_ref(self.config_ref, "config_ref")
        return self


class SkillProbeRoster(ImmutableModel):
    """Typed exploration split used by both matched probe schedules."""

    schema_version: Literal["1"] = "1"
    partition: Literal["exploration"] = "exploration"
    evidence_profile: SkillEvidenceProfile
    exploration_split_ref: ArtifactRef
    study: NonEmptyStr
    kind: NonEmptyStr
    query: Annotated[int, Field(ge=0, strict=True)]
    master_seed: Annotated[int, Field(ge=0, strict=True)]
    task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    search_runs: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=1),
    ]
    repeat_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=1),
    ]
    max_attempts_per_cell: Annotated[int, Field(ge=1, strict=True)]
    token_ceiling_per_attempt: Annotated[int, Field(ge=1, strict=True)]
    adherence_probe_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    behavior_probe_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]

    @field_validator("task_ids")
    @classmethod
    def canonicalize_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or value != value.strip():
                raise ValueError("task_ids must contain exact non-empty identifiers")
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("task_ids must not contain duplicates")
        return ordered

    @field_validator("search_runs", "repeat_seeds")
    @classmethod
    def canonicalize_integer_axes(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("probe roster integer axes must not contain duplicates")
        return ordered

    @field_validator("adherence_probe_refs", "behavior_probe_refs")
    @classmethod
    def canonicalize_probe_refs(
        cls,
        values: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(values, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("probe refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def refs_are_typed_and_disjoint(self) -> Self:
        _require_json_ref(self.exploration_split_ref, "exploration_split_ref")
        for ref in self.adherence_probe_refs:
            _require_media_type(ref, SKILL_ADHERENCE_PROBE_MEDIA_TYPE, "adherence_probe_ref")
        for ref in self.behavior_probe_refs:
            _require_media_type(ref, SKILL_BEHAVIOR_PROBE_MEDIA_TYPE, "behavior_probe_ref")
        adherence = {ref.sha256 for ref in self.adherence_probe_refs}
        behavior = {ref.sha256 for ref in self.behavior_probe_refs}
        if adherence.intersection(behavior):
            raise ValueError("adherence and behavior probes must be distinct artifacts")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        return _artifact_ref(self, SKILL_PROBE_ROSTER_MEDIA_TYPE)


class NeutralSkillRules(ImmutableModel):
    """Deterministic text replacements for changed child rule IDs.

    The fixed marker removes candidate authorship from the placebo text.  This
    is a structural negative control only; the schema does not claim or prove
    semantic neutrality to a model.
    """

    schema_version: Literal["1"] = "1"
    renderer_fingerprint: Sha256 = NEUTRAL_SKILL_RULE_RENDERER_FINGERPRINT
    rule_ids: Annotated[tuple[Slug, ...], Field(min_length=1)]

    @field_validator("rule_ids")
    @classmethod
    def canonicalize_rule_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("neutral rule_ids must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def renderer_is_fixed(self) -> Self:
        if self.renderer_fingerprint != NEUTRAL_SKILL_RULE_RENDERER_FINGERPRINT:
            raise ValueError("neutral rules require the fixed trusted renderer")
        return self

    @property
    def rules(self) -> tuple[SkillRule, ...]:
        """Materialize the only permitted neutral instruction deterministically."""

        return tuple(
            SkillRule(rule_id=rule_id, instruction=NEUTRAL_SKILL_RULE_INSTRUCTION)
            for rule_id in self.rule_ids
        )

    @property
    def artifact_ref(self) -> ArtifactRef:
        return _artifact_ref(self, NEUTRAL_SKILL_RULES_MEDIA_TYPE)


class SkillVerificationPolicy(ImmutableModel):
    """Protocol-frozen authority and implementation policy for skill probes."""

    schema_version: Literal["1"] = "1"
    evidence_profile: SkillEvidenceProfile
    model_fingerprint: NonEmptyStr
    model_spec_fingerprint: Sha256
    inference_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    grader_fingerprint: NonEmptyStr
    request_inclusion_verifier_fingerprint: Sha256 = SKILL_REQUEST_INCLUSION_VERIFIER_FINGERPRINT
    task_roster_ref: ArtifactRef
    neutral_rules_ref: ArtifactRef
    placebo_builder_fingerprint: Sha256 = SKILL_PLACEBO_BUILDER_FINGERPRINT
    runtime_activation_hook_fingerprint: NonEmptyStr
    probe_grader_fingerprint: NonEmptyStr
    reset_fingerprint: NonEmptyStr
    execution_order_fingerprint: NonEmptyStr
    generic_mechanism_attestor_id: Sha256
    aggregate_attestor_id: Sha256
    aggregate_attestation_domain: NonEmptyStr
    aggregate_producer_fingerprint: NonEmptyStr
    aggregate_verifier_fingerprint: NonEmptyStr
    claim_authorities: Annotated[
        tuple[SkillClaimAuthority, ...],
        Field(min_length=6, max_length=6),
    ]
    min_probe_tasks: Annotated[int, Field(ge=1, strict=True)]
    min_adherence_coverage: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    min_adherence_rate: Annotated[float, Field(gt=0.0, le=1.0, strict=True)]
    min_behavior_effect_vs_revert: Annotated[float, Field(ge=0.0, strict=True)]
    min_behavior_effect_vs_placebo: Annotated[float, Field(ge=0.0, strict=True)]
    max_placebo_input_token_delta: Annotated[int, Field(ge=0, strict=True)]
    max_placebo_context_size_delta: Annotated[int, Field(ge=0, strict=True)]

    @field_validator("claim_authorities")
    @classmethod
    def canonicalize_authorities(
        cls,
        values: tuple[SkillClaimAuthority, ...],
    ) -> tuple[SkillClaimAuthority, ...]:
        return tuple(sorted(values, key=lambda authority: authority.claim.value))

    @model_validator(mode="after")
    def authorities_are_exact_and_separated(self) -> Self:
        claims = tuple(authority.claim for authority in self.claim_authorities)
        required = set(ATTESTED_SKILL_MECHANISM_CLAIMS)
        if set(claims) != required or len(claims) != len(required):
            raise ValueError("claim_authorities must cover each attested skill claim exactly once")
        mismatched_profiles = tuple(
            authority.claim.value
            for authority in self.claim_authorities
            if authority.evidence_profile is not self.evidence_profile
        )
        if mismatched_profiles:
            raise ValueError("claim authorities must use the policy evidence profile")
        attestor_ids = tuple(authority.attestor_id for authority in self.claim_authorities)
        if len(attestor_ids) != len(set(attestor_ids)):
            raise ValueError("skill claim authorities must use distinct attestor IDs")
        domains = tuple(authority.attestation_domain for authority in self.claim_authorities)
        if len(domains) != len(set(domains)) or self.aggregate_attestation_domain in domains:
            raise ValueError("claim and aggregate attestation domains must all be distinct")
        producers = tuple(authority.producer_fingerprint for authority in self.claim_authorities)
        verifiers = tuple(authority.verifier_fingerprint for authority in self.claim_authorities)
        if (
            len(producers) != len(set(producers))
            or self.aggregate_producer_fingerprint in producers
        ):
            raise ValueError("claim and aggregate producers must have distinct fingerprints")
        if (
            len(verifiers) != len(set(verifiers))
            or self.aggregate_verifier_fingerprint in verifiers
        ):
            raise ValueError("claim and aggregate verifiers must have distinct fingerprints")
        all_attestors = (
            *attestor_ids,
            self.aggregate_attestor_id,
            self.generic_mechanism_attestor_id,
        )
        if len(all_attestors) != len(set(all_attestors)):
            raise ValueError(
                "claim, aggregate, and generic mechanism attestors must all be distinct"
            )
        if (
            self.request_inclusion_verifier_fingerprint
            != SKILL_REQUEST_INCLUSION_VERIFIER_FINGERPRINT
        ):
            raise ValueError("request inclusion verifier is not the trusted implementation")
        if self.placebo_builder_fingerprint != SKILL_PLACEBO_BUILDER_FINGERPRINT:
            raise ValueError("placebo builder is not the trusted implementation")
        config_digests = tuple(authority.config_ref.sha256 for authority in self.claim_authorities)
        if len(config_digests) != len(set(config_digests)):
            raise ValueError("claim authorities must use distinct configuration artifacts")
        _require_media_type(
            self.task_roster_ref,
            SKILL_PROBE_ROSTER_MEDIA_TYPE,
            "task_roster_ref",
        )
        _require_media_type(
            self.neutral_rules_ref,
            NEUTRAL_SKILL_RULES_MEDIA_TYPE,
            "neutral_rules_ref",
        )
        if self.task_roster_ref.sha256 == self.neutral_rules_ref.sha256:
            raise ValueError("task roster and neutral rules must be distinct artifacts")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        return _artifact_ref(self, SKILL_VERIFICATION_POLICY_MEDIA_TYPE)


class SkillPlaceboControl(ImmutableModel):
    """A matched sham sibling, never proof that its text is semantically neutral."""

    schema_version: Literal["1"] = "1"
    claim_scope: Literal["matched-sham-not-semantic-neutrality-proof"] = (
        "matched-sham-not-semantic-neutrality-proof"
    )
    evidence_profile: SkillEvidenceProfile
    candidate_package_ref: ArtifactRef
    placebo_package_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    placebo_harness_ref: ArtifactRef
    neutral_rules_ref: ArtifactRef
    changed_rule_ids: Annotated[tuple[Slug, ...], Field(min_length=1)]
    placebo_builder_fingerprint: Sha256 = SKILL_PLACEBO_BUILDER_FINGERPRINT
    candidate_rule_count: Annotated[int, Field(ge=1, strict=True)]
    placebo_rule_count: Annotated[int, Field(ge=1, strict=True)]
    candidate_context_size_bytes: Annotated[int, Field(ge=1, strict=True)]
    placebo_context_size_bytes: Annotated[int, Field(ge=1, strict=True)]

    @field_validator("changed_rule_ids")
    @classmethod
    def canonicalize_changed_rule_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("changed_rule_ids must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def matched_sham_shape_is_exact(self) -> Self:
        for field_name in ("candidate_package_ref", "placebo_package_ref"):
            _require_media_type(getattr(self, field_name), SKILL_PACKAGE_MEDIA_TYPE, field_name)
        for field_name in ("candidate_harness_ref", "placebo_harness_ref"):
            _require_media_type(getattr(self, field_name), HARNESS_MANIFEST_MEDIA_TYPE, field_name)
        _require_media_type(
            self.neutral_rules_ref,
            NEUTRAL_SKILL_RULES_MEDIA_TYPE,
            "neutral_rules_ref",
        )
        if self.candidate_package_ref.sha256 == self.placebo_package_ref.sha256:
            raise ValueError("placebo package must differ from the candidate package")
        if self.candidate_harness_ref.sha256 == self.placebo_harness_ref.sha256:
            raise ValueError("placebo harness must differ from the candidate harness")
        if self.candidate_rule_count != self.placebo_rule_count:
            raise ValueError("matched sham must preserve the candidate rule count")
        if self.placebo_builder_fingerprint != SKILL_PLACEBO_BUILDER_FINGERPRINT:
            raise ValueError("placebo control was not made by the trusted builder")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        return _artifact_ref(self, SKILL_PLACEBO_CONTROL_MEDIA_TYPE)


def _schedule_without_parent(schedule: EvaluationBatchSchedule) -> dict[str, object]:
    return schedule.model_dump(
        mode="python",
        exclude={"parent_harness_id"},
        round_trip=True,
        warnings="none",
    )


class SkillMechanismPlan(ImmutableModel):
    """One candidate-bound plan for matched revert and placebo probes.

    Candidate treatment executions must be performed and attested separately
    under both schedule fingerprints; matching coordinates do not authorize
    reusing one schedule's execution evidence in the other.
    """

    schema_version: Literal["1"] = "1"
    evidence_profile: SkillEvidenceProfile
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    mutation_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    before_skill_package_ref: ArtifactRef
    after_skill_package_ref: ArtifactRef
    target_skill_id: Slug
    changed_rule_ids: Annotated[tuple[Slug, ...], Field(min_length=1)]
    policy_ref: ArtifactRef
    policy_fingerprint: Sha256
    exploration_split_ref: ArtifactRef
    probe_roster_ref: ArtifactRef
    probe_roster_fingerprint: Sha256
    placebo_control_ref: ArtifactRef
    placebo_harness_ref: ArtifactRef
    model_spec_fingerprint: Sha256
    runtime_fingerprint: NonEmptyStr
    probe_grader_fingerprint: NonEmptyStr
    reset_fingerprint: NonEmptyStr
    execution_order_fingerprint: NonEmptyStr
    revert_schedule: EvaluationBatchSchedule
    placebo_schedule: EvaluationBatchSchedule

    @field_validator("changed_rule_ids")
    @classmethod
    def canonicalize_changed_rule_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("changed_rule_ids must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def bind_local_plan_shape(self) -> Self:
        expected_media_types = (
            ("experiment_ref", EXPERIMENT_MANIFEST_MEDIA_TYPE),
            ("protocol_ref", PROTOCOL_MANIFEST_MEDIA_TYPE),
            ("candidate_ref", CANDIDATE_MANIFEST_MEDIA_TYPE),
            ("mutation_ref", CANDIDATE_MUTATION_MEDIA_TYPE),
            ("parent_harness_ref", HARNESS_MANIFEST_MEDIA_TYPE),
            ("candidate_harness_ref", HARNESS_MANIFEST_MEDIA_TYPE),
            ("before_skill_package_ref", SKILL_PACKAGE_MEDIA_TYPE),
            ("after_skill_package_ref", SKILL_PACKAGE_MEDIA_TYPE),
            ("policy_ref", SKILL_VERIFICATION_POLICY_MEDIA_TYPE),
            ("probe_roster_ref", SKILL_PROBE_ROSTER_MEDIA_TYPE),
            ("placebo_control_ref", SKILL_PLACEBO_CONTROL_MEDIA_TYPE),
            ("placebo_harness_ref", HARNESS_MANIFEST_MEDIA_TYPE),
        )
        for field_name, media_type in expected_media_types:
            _require_media_type(getattr(self, field_name), media_type, field_name)

        _require_json_ref(self.exploration_split_ref, "exploration_split_ref")
        if self.exploration_split_ref.sha256 == self.probe_roster_ref.sha256:
            raise ValueError("probe roster must not be a self-referential exploration split")
        if self.policy_fingerprint != self.policy_ref.sha256:
            raise ValueError("policy_fingerprint must identify policy_ref")
        if self.probe_roster_fingerprint != self.probe_roster_ref.sha256:
            raise ValueError("probe_roster_fingerprint must identify probe_roster_ref")
        if self.parent_harness_ref == self.candidate_harness_ref:
            raise ValueError("candidate harness must differ from its parent")
        if self.before_skill_package_ref == self.after_skill_package_ref:
            raise ValueError("before and after skill packages must differ")
        if self.placebo_harness_ref in {
            self.parent_harness_ref,
            self.candidate_harness_ref,
        }:
            raise ValueError("placebo harness must differ from parent and candidate harnesses")

        revert = self.revert_schedule
        placebo = self.placebo_schedule
        if revert.phase is not EvaluationPhase.PROBE or placebo.phase is not EvaluationPhase.PROBE:
            raise ValueError("skill mechanism schedules must use the PROBE phase")
        if revert.parent_harness_id != self.parent_harness_ref.sha256:
            raise ValueError("revert schedule must pair the exact parent harness")
        if placebo.parent_harness_id != self.placebo_harness_ref.sha256:
            raise ValueError("placebo schedule must pair the exact placebo harness")
        for schedule in (revert, placebo):
            if schedule.candidate_harness_id != self.candidate_harness_ref.sha256:
                raise ValueError("both schedules must pair the exact candidate harness")
        if _schedule_without_parent(revert) != _schedule_without_parent(placebo):
            raise ValueError("revert and placebo schedules may differ only by parent_harness_id")
        if revert.fingerprint == placebo.fingerprint:
            raise ValueError("revert and placebo schedules require distinct fingerprints")
        return self

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=sha256_bytes(payload),
            size=len(payload),
            media_type=SKILL_MECHANISM_PLAN_MEDIA_TYPE,
        )


__all__ = [
    "ATTESTED_SKILL_MECHANISM_CLAIMS",
    "NEUTRAL_SKILL_RULES_MEDIA_TYPE",
    "NEUTRAL_SKILL_RULE_INSTRUCTION",
    "NEUTRAL_SKILL_RULE_RENDERER_FINGERPRINT",
    "SKILL_ADHERENCE_EVIDENCE_MEDIA_TYPE",
    "SKILL_ADHERENCE_PROBE_MEDIA_TYPE",
    "SKILL_BEHAVIOR_EVIDENCE_MEDIA_TYPE",
    "SKILL_BEHAVIOR_PROBE_MEDIA_TYPE",
    "SKILL_CLAIM_EVIDENCE_MEDIA_TYPES",
    "SKILL_MECHANISM_PLAN_MEDIA_TYPE",
    "SKILL_PLACEBO_BUILDER_FINGERPRINT",
    "SKILL_PLACEBO_CONTROL_EVIDENCE_MEDIA_TYPE",
    "SKILL_PLACEBO_CONTROL_MEDIA_TYPE",
    "SKILL_PROBE_ROSTER_MEDIA_TYPE",
    "SKILL_PROVIDER_DELIVERY_EVIDENCE_MEDIA_TYPE",
    "SKILL_REVERT_CONTROL_EVIDENCE_MEDIA_TYPE",
    "SKILL_RUNTIME_ACTIVATION_EVIDENCE_MEDIA_TYPE",
    "SKILL_VERIFICATION_POLICY_MEDIA_TYPE",
    "NeutralSkillRules",
    "SkillClaimAuthority",
    "SkillEvidenceProfile",
    "SkillMechanismClaim",
    "SkillMechanismPlan",
    "SkillPlaceboControl",
    "SkillProbeRoster",
    "SkillVerificationPolicy",
]
