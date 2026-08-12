"""Typed artifacts authored or consumed by the trusted experiment controller."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
)
from spiral_harness.execution.schedule import SCHEDULE_PREFLIGHT_MEDIA_TYPE
from spiral_harness.storage.journal import JOURNAL_ENTRY_MEDIA_TYPE
from spiral_harness.verification.artifacts import GATE_TRIAL_BATCH_MEDIA_TYPE
from spiral_harness.verification.skill_plan import SKILL_MECHANISM_PLAN_MEDIA_TYPE

from .decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
)
from .skill_probe_authorization import SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE

_MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-skill-probe-closure.v1+json"
)

ADMISSION_FAILURE_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.admission-failure-report.v1+json"
)
EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE = "application/vnd.spiral-harness.experiment-usage-claim.v1+json"
EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-usage-entry.v1+json"
)
EXPERIMENT_USAGE_ENTRY_V2_MEDIA_TYPE = (
    "application/vnd.spiral-harness.experiment-usage-entry.v2+json"
)
# New controller writes always use v2.  The explicit V1 constant exists only
# for strict replay of historical gate-only ledgers.
EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE = EXPERIMENT_USAGE_ENTRY_V2_MEDIA_TYPE
EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES = frozenset(
    {EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE, EXPERIMENT_USAGE_ENTRY_V2_MEDIA_TYPE}
)
PROBE_REJECTION_REPORT_MEDIA_TYPE = "application/vnd.spiral-harness.probe-rejection-report.v1+json"
SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-probe-usage-claim.v1+json"
)
SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE = (
    "application/vnd.spiral-harness.skill-probe-usage-settlement-claim.v1+json"
)
SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.superseded-candidate-report.v1+json"
)
TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.terminal-transition-authorization.v1+json"
)


class AdmissionFailureCode(StrEnum):
    """Stable admission failure categories emitted by the trusted controller."""

    REPORT_REPLAY_FAILED = "report_replay_failed"
    VERIFIER_FAILURE = "verifier_failure"


class ProbeRejectionCode(StrEnum):
    """Stable outcomes that prevent a candidate from entering the gate."""

    REQUIRED_CHECK_FAILED = "required_check_failed"
    REQUIRED_CHECK_MISSING = "required_check_missing"
    REQUIRED_CHECKS_FAILED_AND_MISSING = "required_checks_failed_and_missing"


class CandidateSupersessionCode(StrEnum):
    """Stable controller-derived reasons that a local promotion cannot advance champion."""

    PARENT_CHAMPION_ADVANCED = "parent_champion_advanced"


class SkillProbeSettlementKind(StrEnum):
    """Irrevocable terminal outcome of one reserved matched probe."""

    COMPLETED = "completed"
    FAILED = "failed"


class AdmissionFailureReport(ImmutableModel):
    """Controller-authored evidence for a fail-closed ``INVALID`` transition."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    candidate_ref: ArtifactRef
    attempted_admission_report_ref: ArtifactRef
    error_code: AdmissionFailureCode
    message: NonEmptyStr


class ProbeRejectionReport(ImmutableModel):
    """Controller-authored evidence that required mechanism checks did not pass."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef
    error_code: ProbeRejectionCode
    required_checks: tuple[NonEmptyStr, ...]
    failed_checks: tuple[NonEmptyStr, ...]
    missing_checks: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def rejection_has_a_failed_or_missing_check(self) -> ProbeRejectionReport:
        if not self.failed_checks and not self.missing_checks:
            raise ValueError("probe rejection requires a failed or missing check")
        if set(self.failed_checks).intersection(self.missing_checks):
            raise ValueError("a required check cannot be both failed and missing")
        return self


class ExperimentUsageClaim(ImmutableModel):
    """One gate query whose cost was recomputed from immutable trial batches."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    running_gate_tail_ref: ArtifactRef
    evaluation_ref: ArtifactRef
    parent_batch_ref: ArtifactRef
    candidate_batch_ref: ArtifactRef
    evaluation_units: Annotated[int, Field(ge=1, strict=True)]
    tokens: Annotated[int, Field(ge=0, strict=True)]
    tool_calls: Annotated[int, Field(ge=0, strict=True)]
    wall_time_seconds: Annotated[float, Field(ge=0, strict=True)]
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None

    @model_validator(mode="after")
    def references_have_exact_media_types(self) -> ExperimentUsageClaim:
        expected = (
            ("running_gate_tail_ref", JOURNAL_ENTRY_MEDIA_TYPE),
            ("evaluation_ref", GATE_EVALUATION_MANIFEST_MEDIA_TYPE),
            ("parent_batch_ref", GATE_TRIAL_BATCH_MEDIA_TYPE),
            ("candidate_batch_ref", GATE_TRIAL_BATCH_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        return self


class SkillProbeUsageClaim(ImmutableModel):
    """Worst-case reservation for one controller-authorized matched probe."""

    schema_version: Literal["1"] = "1"
    reservation_kind: Literal["worst_case"] = "worst_case"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    authorization_ref: ArtifactRef
    execution_nonce: Sha256
    plan_ref: ArtifactRef
    running_probes_tail_ref: ArtifactRef
    revert_schedule_fingerprint: Sha256
    placebo_schedule_fingerprint: Sha256
    evaluation_units: Annotated[int, Field(ge=1, strict=True)]
    tokens: Annotated[int, Field(ge=1, strict=True)]
    tool_calls: Literal[0] = 0
    wall_time_seconds: None = None
    cost_usd: None = None

    @model_validator(mode="after")
    def references_have_exact_media_types(self) -> SkillProbeUsageClaim:
        expected = (
            ("experiment_ref", EXPERIMENT_MANIFEST_MEDIA_TYPE),
            ("protocol_ref", PROTOCOL_MANIFEST_MEDIA_TYPE),
            ("candidate_ref", CANDIDATE_MANIFEST_MEDIA_TYPE),
            ("authorization_ref", SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE),
            ("plan_ref", SKILL_MECHANISM_PLAN_MEDIA_TYPE),
            ("running_probes_tail_ref", JOURNAL_ENTRY_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        return self


class SkillProbeUsageArmSettlement(ImmutableModel):
    """Replay boundary for one prepared matched-probe arm."""

    control: Literal["revert", "placebo"]
    preflight_ref: ArtifactRef
    terminal_tail_ref: ArtifactRef | None
    encumbered_tokens: Annotated[int, Field(ge=0, strict=True)]
    poisoned: bool

    @model_validator(mode="after")
    def exact_boundary_media_types(self) -> SkillProbeUsageArmSettlement:
        if self.preflight_ref.media_type != SCHEDULE_PREFLIGHT_MEDIA_TYPE:
            raise ValueError("preflight_ref declares the wrong media type")
        if self.terminal_tail_ref is not None and self.terminal_tail_ref.media_type not in {
            ATTEMPT_RESERVATION_MEDIA_TYPE,
            ATTEMPT_OUTCOME_MEDIA_TYPE,
        }:
            raise ValueError("terminal_tail_ref declares the wrong media type")
        return self


class SkillProbeUsageSettlementClaim(ImmutableModel):
    """Replay-derived terminal adjustment for one worst-case probe reservation."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    authorization_ref: ArtifactRef
    execution_nonce: Sha256
    reservation_claim_ref: ArtifactRef
    terminal_kind: SkillProbeSettlementKind
    accounting_complete: Literal[True] = True
    revert: SkillProbeUsageArmSettlement
    placebo: SkillProbeUsageArmSettlement
    reserved_tokens: Annotated[int, Field(ge=1, strict=True)]
    encumbered_tokens: Annotated[int, Field(ge=0, strict=True)]
    token_adjustment: Annotated[int, Field(ge=0, strict=True)]
    poisoned: bool
    closure_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def context_and_arithmetic_are_exact(self) -> SkillProbeUsageSettlementClaim:
        expected = (
            ("experiment_ref", self.experiment_ref, EXPERIMENT_MANIFEST_MEDIA_TYPE),
            ("protocol_ref", self.protocol_ref, PROTOCOL_MANIFEST_MEDIA_TYPE),
            ("candidate_ref", self.candidate_ref, CANDIDATE_MANIFEST_MEDIA_TYPE),
            (
                "authorization_ref",
                self.authorization_ref,
                SKILL_PROBE_EXECUTION_AUTHORIZATION_MEDIA_TYPE,
            ),
            (
                "reservation_claim_ref",
                self.reservation_claim_ref,
                SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
            ),
        )
        for field_name, ref, media_type in expected:
            if ref.media_type != media_type:
                raise ValueError(f"{field_name} declares the wrong media type")
        if (self.revert.control, self.placebo.control) != ("revert", "placebo"):
            raise ValueError("skill-probe settlement arms are mislabeled")
        if self.encumbered_tokens != self.revert.encumbered_tokens + self.placebo.encumbered_tokens:
            raise ValueError("encumbered_tokens differs from the two arm settlements")
        if self.token_adjustment != max(0, self.encumbered_tokens - self.reserved_tokens):
            raise ValueError("token_adjustment is not the conservative overrun delta")
        if self.poisoned != (self.revert.poisoned or self.placebo.poisoned):
            raise ValueError("settlement poison state differs from its arm ledgers")
        if self.terminal_kind is SkillProbeSettlementKind.COMPLETED and self.closure_ref is None:
            raise ValueError("a completed settlement requires closure_ref")
        if (
            self.closure_ref is not None
            and self.closure_ref.media_type != _MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE
        ):
            raise ValueError("closure_ref declares the wrong media type")
        return self


class ExperimentUsageEntryV1(ImmutableModel):
    """Historical gate-only usage link; its v1 contract is frozen exactly."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    sequence: Annotated[int, Field(ge=0, strict=True)]
    claim_ref: ArtifactRef
    cumulative_evaluations: Annotated[int, Field(ge=1, strict=True)]
    cumulative_tokens: Annotated[int, Field(ge=0, strict=True)]
    cumulative_tool_calls: Annotated[int, Field(ge=0, strict=True)]
    cumulative_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)]
    cumulative_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None
    previous_entry_ref: ArtifactRef | None

    @model_validator(mode="after")
    def linked_entry_shape_is_valid(self) -> ExperimentUsageEntryV1:
        if self.claim_ref.media_type != EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE:
            raise ValueError("v1 claim_ref must declare the gate usage claim media type")
        if (
            self.previous_entry_ref is not None
            and self.previous_entry_ref.media_type != EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE
        ):
            raise ValueError("v1 previous_entry_ref must declare the v1 usage entry media type")
        if self.sequence == 0 and self.previous_entry_ref is not None:
            raise ValueError("usage sequence 0 must not have a previous entry")
        if self.sequence > 0 and self.previous_entry_ref is None:
            raise ValueError("usage sequence greater than 0 requires a previous entry")
        return self


class ExperimentUsageEntry(ImmutableModel):
    """Current usage link supporting gates, probe reservations, and settlements."""

    schema_version: Literal["2"] = "2"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    sequence: Annotated[int, Field(ge=0, strict=True)]
    claim_ref: ArtifactRef
    cumulative_evaluations: Annotated[int, Field(ge=1, strict=True)]
    cumulative_tokens: Annotated[int, Field(ge=0, strict=True)]
    cumulative_tool_calls: Annotated[int, Field(ge=0, strict=True)]
    cumulative_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)] | None
    cumulative_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None
    poisoned: bool = False
    accounting_complete: Literal[True] = True
    previous_entry_ref: ArtifactRef | None

    @model_validator(mode="after")
    def linked_entry_shape_is_valid(self) -> ExperimentUsageEntry:
        if self.claim_ref.media_type not in {
            EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
            SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
            SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
        }:
            raise ValueError("claim_ref declares an unsupported experiment usage media type")
        if (
            self.previous_entry_ref is not None
            and self.previous_entry_ref.media_type not in EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES
        ):
            raise ValueError("previous_entry_ref declares the wrong usage entry media type")
        if self.sequence == 0 and self.previous_entry_ref is not None:
            raise ValueError("usage sequence 0 must not have a previous entry")
        if self.sequence > 0 and self.previous_entry_ref is None:
            raise ValueError("usage sequence greater than 0 requires a previous entry")
        return self


class ExperimentUsage(ImmutableModel):
    """Replay-derived query result for one exact experiment usage tail."""

    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    tail_ref: ArtifactRef | None
    entry_refs: tuple[ArtifactRef, ...]
    entry_count: Annotated[int, Field(ge=0, strict=True)]
    claim_refs: tuple[ArtifactRef, ...]
    candidate_refs: tuple[ArtifactRef, ...]
    evaluation_refs: tuple[ArtifactRef, ...]
    skill_probe_claim_refs: tuple[ArtifactRef, ...]
    skill_probe_authorization_refs: tuple[ArtifactRef, ...]
    skill_probe_settlement_refs: tuple[ArtifactRef, ...] = ()
    poisoned: bool = False
    accounting_complete: bool = True
    query_count: Annotated[int, Field(ge=0, strict=True)]
    total_evaluations: Annotated[int, Field(ge=0, strict=True)]
    total_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_tool_calls: Annotated[int, Field(ge=0, strict=True)]
    total_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)] | None
    total_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None
    max_evaluations: Annotated[int, Field(ge=0, strict=True)]
    max_tokens: Annotated[int, Field(ge=0, strict=True)] | None
    max_tool_calls: Annotated[int, Field(ge=0, strict=True)] | None
    max_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)] | None
    max_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None
    remaining_evaluations: Annotated[int, Field(ge=0, strict=True)]
    remaining_tokens: Annotated[int, Field(ge=0, strict=True)] | None
    remaining_tool_calls: Annotated[int, Field(ge=0, strict=True)] | None
    remaining_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)] | None
    remaining_cost_usd: Annotated[float, Field(ge=0, strict=True)] | None


class EvidenceCompletion(ImmutableModel):
    """The two published heads resulting from a budgeted gate completion."""

    candidate_tail_ref: ArtifactRef
    usage_tail_ref: ArtifactRef
    usage_claim_ref: ArtifactRef
    total_evaluations: Annotated[int, Field(ge=1, strict=True)]
    remaining_evaluations: Annotated[int, Field(ge=0, strict=True)]


class SupersededCandidateReport(ImmutableModel):
    """Resolve a valid local PROMOTE after another sibling advanced champion."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    evidence_complete_tail_ref: ArtifactRef
    terminal_decision_report_ref: ArtifactRef
    decision_ref: ArtifactRef
    stale_parent_harness_ref: ArtifactRef
    current_champion_harness_ref: ArtifactRef
    superseding_candidate_ref: ArtifactRef
    error_code: CandidateSupersessionCode = CandidateSupersessionCode.PARENT_CHAMPION_ADVANCED
    gate_terminal_state: Literal[CandidateState.PROMOTED] = CandidateState.PROMOTED
    resolved_terminal_state: Literal[CandidateState.INCONCLUSIVE] = CandidateState.INCONCLUSIVE

    @model_validator(mode="after")
    def exact_proof_media_types(self) -> SupersededCandidateReport:
        expected = (
            ("evidence_complete_tail_ref", JOURNAL_ENTRY_MEDIA_TYPE),
            ("terminal_decision_report_ref", TERMINAL_DECISION_REPORT_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        return self


class TerminalTransitionAuthorization(ImmutableModel):
    """Bind a replayed decision to one exact evidence-complete journal branch."""

    schema_version: Literal["1"] = "1"
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    prior_champion_harness_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    child_harness_ref: ArtifactRef
    evidence_complete_tail_ref: ArtifactRef
    usage_entry_ref: ArtifactRef
    evaluation_ref: ArtifactRef
    terminal_decision_report_ref: ArtifactRef
    decision_ref: ArtifactRef
    gate_terminal_state: CandidateState
    terminal_state: CandidateState
    superseded_report_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def references_and_state_are_terminal(self) -> TerminalTransitionAuthorization:
        expected = (
            ("evidence_complete_tail_ref", JOURNAL_ENTRY_MEDIA_TYPE),
            ("evaluation_ref", GATE_EVALUATION_MANIFEST_MEDIA_TYPE),
            ("terminal_decision_report_ref", TERMINAL_DECISION_REPORT_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        if self.usage_entry_ref.media_type not in EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES:
            raise ValueError("usage_entry_ref declares an unsupported usage entry version")
        terminal_states = {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        }
        if self.terminal_state not in terminal_states:
            raise ValueError("terminal_state is not a gate terminal outcome")
        if self.gate_terminal_state not in terminal_states:
            raise ValueError("gate_terminal_state is not a gate terminal outcome")
        is_superseded = self.superseded_report_ref is not None
        if is_superseded:
            if self.superseded_report_ref.media_type != SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE:
                raise ValueError("superseded_report_ref declares the wrong media type")
            if self.gate_terminal_state is not CandidateState.PROMOTED:
                raise ValueError("only a local PROMOTE may be superseded")
            if self.terminal_state is not CandidateState.INCONCLUSIVE:
                raise ValueError("a superseded local promotion resolves to INCONCLUSIVE")
        elif self.gate_terminal_state is not self.terminal_state:
            raise ValueError("terminal state may differ from gate state only when superseded")
        return self


class TerminalCompletion(ImmutableModel):
    """Published terminal tail and its exact-branch authorization artifact."""

    candidate_tail_ref: ArtifactRef
    authorization_ref: ArtifactRef
    superseded_report_ref: ArtifactRef | None = None


__all__ = [
    "ADMISSION_FAILURE_REPORT_MEDIA_TYPE",
    "EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE",
    "EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE",
    "EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES",
    "EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE",
    "EXPERIMENT_USAGE_ENTRY_V2_MEDIA_TYPE",
    "PROBE_REJECTION_REPORT_MEDIA_TYPE",
    "SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE",
    "SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE",
    "SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE",
    "TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE",
    "AdmissionFailureCode",
    "AdmissionFailureReport",
    "CandidateSupersessionCode",
    "EvidenceCompletion",
    "ExperimentUsage",
    "ExperimentUsageClaim",
    "ExperimentUsageEntry",
    "ExperimentUsageEntryV1",
    "ProbeRejectionCode",
    "ProbeRejectionReport",
    "SkillProbeSettlementKind",
    "SkillProbeUsageArmSettlement",
    "SkillProbeUsageClaim",
    "SkillProbeUsageSettlementClaim",
    "SupersededCandidateReport",
    "TerminalCompletion",
    "TerminalTransitionAuthorization",
]
