"""Typed, non-promoting closure for matched skill-probe execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.contracts import ATTEMPT_OUTCOME_MEDIA_TYPE
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    TrustedExecutionUsage,
)
from spiral_harness.execution.schedule import SCHEDULE_PREFLIGHT_MEDIA_TYPE
from spiral_harness.storage.journal import JOURNAL_ENTRY_MEDIA_TYPE
from spiral_harness.verification.skill_inclusion import (
    SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
)
from spiral_harness.verification.skill_plan import (
    CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    SKILL_MECHANISM_PLAN_MEDIA_TYPE,
    SKILL_PROBE_ROSTER_MEDIA_TYPE,
    SkillEvidenceProfile,
    SkillMechanismClaim,
)

from .skill_probe_authorization import SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE

MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-skill-probe-closure.v1+json"
)
SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-probe-shadow-report.v1+json"
)


class SkillProbeArmClosure(ImmutableModel):
    """Exact accounting and inclusion closure for one control schedule."""

    schema_version: Literal["1"] = "1"
    control: Literal["revert", "placebo"]
    schedule_fingerprint: Sha256
    preflight_ref: ArtifactRef
    opening_ledger_tail_ref: ArtifactRef | None
    closing_ledger_tail_ref: ArtifactRef
    ledger_id: NonEmptyStr
    writer_epoch_id: Sha256
    budget_fingerprint: Sha256
    receipt_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    usage: TrustedExecutionUsage
    request_inclusion_ref: ArtifactRef

    @model_validator(mode="after")
    def references_and_usage_are_exact(self) -> Self:
        expected = (
            ("preflight_ref", self.preflight_ref, SCHEDULE_PREFLIGHT_MEDIA_TYPE),
            ("closing_ledger_tail_ref", self.closing_ledger_tail_ref, ATTEMPT_OUTCOME_MEDIA_TYPE),
            (
                "request_inclusion_ref",
                self.request_inclusion_ref,
                SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE,
            ),
        )
        for field_name, ref, media_type in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        if (
            self.opening_ledger_tail_ref is not None
            and self.opening_ledger_tail_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE
        ):
            raise ValueError("opening_ledger_tail_ref declares the wrong media type")
        if any(ref.media_type != EXECUTION_RECEIPT_MEDIA_TYPE for ref in self.receipt_refs):
            raise ValueError("receipt_refs contain the wrong media type")
        if len({ref.sha256 for ref in self.receipt_refs}) != len(self.receipt_refs):
            raise ValueError("receipt_refs must not contain duplicates")
        if self.usage.schedule_fingerprint != self.schedule_fingerprint:
            raise ValueError("usage belongs to another schedule")
        if self.usage.receipt_refs != self.receipt_refs:
            raise ValueError("receipt_refs differ from canonical trusted usage")
        if self.usage.ledger_tail_refs != (self.closing_ledger_tail_ref,):
            raise ValueError("closing ledger tail differs from trusted usage")
        if self.usage.burned_attempts or self.usage.poisoned_attempts:
            raise ValueError("a closed skill probe must settle every first attempt")
        if self.usage.settled_attempts != self.usage.attempt_count:
            raise ValueError("a closed skill probe contains a non-settled attempt")
        return self


class MatchedSkillProbeClosure(ImmutableModel):
    """Non-promoting proof of two independent, preregistered local executions."""

    schema_version: Literal["1"] = "1"
    claim_scope: Literal["controlled-local-execution-and-request-inclusion-only"] = (
        "controlled-local-execution-and-request-inclusion-only"
    )
    promotion_authority: Literal[False] = False
    evidence_profile: Literal[SkillEvidenceProfile.CONTROLLED_REPLAY] = (
        SkillEvidenceProfile.CONTROLLED_REPLAY
    )
    authorization_ref: ArtifactRef
    execution_nonce: Sha256
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    plan_ref: ArtifactRef
    running_probes_tail_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    probe_roster_ref: ArtifactRef
    task_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    model_spec_fingerprint: Sha256
    runtime_fingerprint: NonEmptyStr
    reset_fingerprint: Sha256
    execution_order_fingerprint: Sha256
    revert: SkillProbeArmClosure
    placebo: SkillProbeArmClosure

    @model_validator(mode="after")
    def context_and_arms_are_separate(self) -> Self:
        expected = (
            (
                "authorization_ref",
                self.authorization_ref,
                SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE,
            ),
            ("experiment_ref", self.experiment_ref, EXPERIMENT_MANIFEST_MEDIA_TYPE),
            ("protocol_ref", self.protocol_ref, PROTOCOL_MANIFEST_MEDIA_TYPE),
            ("candidate_ref", self.candidate_ref, CANDIDATE_MANIFEST_MEDIA_TYPE),
            ("plan_ref", self.plan_ref, SKILL_MECHANISM_PLAN_MEDIA_TYPE),
            ("running_probes_tail_ref", self.running_probes_tail_ref, JOURNAL_ENTRY_MEDIA_TYPE),
            ("candidate_harness_ref", self.candidate_harness_ref, HARNESS_MANIFEST_MEDIA_TYPE),
            ("probe_roster_ref", self.probe_roster_ref, SKILL_PROBE_ROSTER_MEDIA_TYPE),
        )
        for field_name, ref, media_type in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        if any(ref.media_type != CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE for ref in self.task_refs):
            raise ValueError("task_refs contain the wrong media type")
        if len({ref.sha256 for ref in self.task_refs}) != len(self.task_refs):
            raise ValueError("task_refs must not contain duplicates")
        if (self.revert.control, self.placebo.control) != ("revert", "placebo"):
            raise ValueError("closure arms are mislabeled")
        if self.revert.schedule_fingerprint == self.placebo.schedule_fingerprint:
            raise ValueError("matched arms must use distinct schedules")
        if self.revert.preflight_ref == self.placebo.preflight_ref:
            raise ValueError("matched arms must use distinct preflights")
        if self.revert.ledger_id == self.placebo.ledger_id:
            raise ValueError("matched arms must use distinct ledgers")
        if self.revert.writer_epoch_id == self.placebo.writer_epoch_id:
            raise ValueError("matched arms must use distinct writer epochs")
        if self.revert.request_inclusion_ref == self.placebo.request_inclusion_ref:
            raise ValueError("matched arms must use distinct inclusion evidence")
        return self


class SkillProbeShadowReport(ImmutableModel):
    """Human/audit summary with no mechanism-check or promotion authority."""

    schema_version: Literal["1"] = "1"
    claim_scope: Literal["request-inclusion-shadow-only"] = "request-inclusion-shadow-only"
    promotion_authority: Literal[False] = False
    authorization_ref: ArtifactRef
    plan_ref: ArtifactRef
    running_probes_tail_ref: ArtifactRef
    execution_closure_ref: ArtifactRef
    request_inclusion_refs: Annotated[
        tuple[ArtifactRef, ArtifactRef],
        Field(min_length=2, max_length=2),
    ]
    reported_claims: tuple[Literal[SkillMechanismClaim.REQUEST_INCLUSION]] = (
        SkillMechanismClaim.REQUEST_INCLUSION,
    )
    unavailable_claims: tuple[
        Literal[SkillMechanismClaim.PROVIDER_DELIVERY],
        Literal[SkillMechanismClaim.RUNTIME_ACTIVATION],
        Literal[SkillMechanismClaim.ADHERENCE],
        Literal[SkillMechanismClaim.BEHAVIOR],
        Literal[SkillMechanismClaim.REVERT_CONTROL],
        Literal[SkillMechanismClaim.PLACEBO_CONTROL],
    ] = (
        SkillMechanismClaim.PROVIDER_DELIVERY,
        SkillMechanismClaim.RUNTIME_ACTIVATION,
        SkillMechanismClaim.ADHERENCE,
        SkillMechanismClaim.BEHAVIOR,
        SkillMechanismClaim.REVERT_CONTROL,
        SkillMechanismClaim.PLACEBO_CONTROL,
    )

    @model_validator(mode="after")
    def refs_are_typed_and_distinct(self) -> Self:
        expected = (
            (self.authorization_ref, SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE),
            (self.plan_ref, SKILL_MECHANISM_PLAN_MEDIA_TYPE),
            (self.running_probes_tail_ref, JOURNAL_ENTRY_MEDIA_TYPE),
            (self.execution_closure_ref, MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE),
        )
        if any(ref.media_type != media_type for ref, media_type in expected):
            raise ValueError("shadow report contains a mistyped context reference")
        if any(
            ref.media_type != SETTLED_SKILL_REQUEST_INCLUSION_MEDIA_TYPE
            for ref in self.request_inclusion_refs
        ):
            raise ValueError("shadow report contains a mistyped inclusion reference")
        if self.request_inclusion_refs[0] == self.request_inclusion_refs[1]:
            raise ValueError("shadow report requires two independent inclusion artifacts")
        return self


class MatchedSkillProbeExecutionResult(ImmutableModel):
    """Published audit refs returned after one irrevocable execution."""

    closure_ref: ArtifactRef
    shadow_report_ref: ArtifactRef

    @model_validator(mode="after")
    def references_are_typed(self) -> Self:
        if self.closure_ref.media_type != MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE:
            raise ValueError("closure_ref declares the wrong media type")
        if self.shadow_report_ref.media_type != SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE:
            raise ValueError("shadow_report_ref declares the wrong media type")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedMatchedSkillProbeResult:
    """A live-ledger-verified closure and its non-promoting audit summary."""

    closure: MatchedSkillProbeClosure
    shadow_report: SkillProbeShadowReport


__all__ = [
    "MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE",
    "SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE",
    "MatchedSkillProbeClosure",
    "MatchedSkillProbeExecutionResult",
    "SkillProbeArmClosure",
    "SkillProbeShadowReport",
    "VerifiedMatchedSkillProbeResult",
]
