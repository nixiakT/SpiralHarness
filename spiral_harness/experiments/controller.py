"""Trusted, single-writer control of one frozen experiment.

The structural :class:`~spiral_harness.storage.CandidateJournal` deliberately
accepts any locally legal lifecycle event.  This module is the semantic owner
of those events: it rejoins admission, mechanism, gate, usage, and terminal
artifacts before publishing a transition. Mechanism checks must arrive in a
protocol-pinned trusted-producer envelope; a caller-authored ``passed`` value
cannot authorize entry into the gate.

M0.2 uses caller-held content-addressed tails plus one in-process controller as
the single writer.  Stale tails and branches are rejected inside that process.
This is not a cross-process compare-and-swap protocol: a distributed service
must add a durable lease/transaction around the experiment usage head and each
candidate head before multiple workers may write concurrently.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.lifecycle import (
    TERMINAL_CANDIDATE_STATES,
    CandidateLifecycleEvent,
    CandidateState,
)
from spiral_harness.core.models import (
    ArtifactRef,
    BudgetPolicy,
    ImmutableModel,
    NonEmptyStr,
)
from spiral_harness.storage.journal import (
    JOURNAL_ENTRY_MEDIA_TYPE,
    CandidateJournal,
    JournalEntry,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateBatchExecutionContext,
    GateBatchVerificationCapability,
    GateTrialArm,
    GateTrialBatch,
)
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    AttestedMechanismEvidence,
    MechanismEvidenceVerificationCapability,
)
from spiral_harness.verification.models import GateConfig, MechanismEvidence

from .admission import CandidateAdmissionError, CandidateAdmissionService
from .decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    GateEvaluationManifest,
    TerminalDecisionError,
    TerminalDecisionReport,
    TerminalDecisionService,
)
from .lifecycle import (
    EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE,
    EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE,
    SEALED_EVALUATION_REPORT_MEDIA_TYPE,
    SEALED_RUN_AUTHORIZATION_MEDIA_TYPE,
    SELECTION_CLOSURE_MEDIA_TYPE,
    TERMINAL_EXPERIMENT_STATES,
    ExperimentCompletionReport,
    ExperimentInvalidationReport,
    ExperimentJournal,
    ExperimentLifecycleEvent,
    ExperimentState,
    ExperimentViolationCode,
    SealedEvaluationReport,
    SealedRunAuthorization,
    SelectionClosure,
    SelectionReason,
)

EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE = "application/vnd.spiral-harness.experiment-usage-claim.v1+json"
EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE = "application/vnd.spiral-harness.experiment-usage-entry.v1+json"
ADMISSION_FAILURE_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.admission-failure-report.v1+json"
)
PROBE_REJECTION_REPORT_MEDIA_TYPE = "application/vnd.spiral-harness.probe-rejection-report.v1+json"
SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.superseded-candidate-report.v1+json"
)
TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.terminal-transition-authorization.v1+json"
)


class ExperimentControllerError(RuntimeError):
    """Raised when a requested semantic transition cannot be authorized."""


class ExperimentBudgetError(ExperimentControllerError):
    """Raised when persisted evaluation use would exceed a frozen ceiling."""


class StaleControllerTailError(ExperimentControllerError):
    """Raised when a caller tries to append from an old or foreign branch."""


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


class ExperimentUsageEntry(ImmutableModel):
    """One immutable link in the experiment-wide evaluation/query ledger."""

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
    def linked_entry_shape_is_valid(self) -> ExperimentUsageEntry:
        if self.claim_ref.media_type != EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE:
            raise ValueError("claim_ref declares the wrong experiment usage media type")
        if (
            self.previous_entry_ref is not None
            and self.previous_entry_ref.media_type != EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE
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
    claim_refs: tuple[ArtifactRef, ...]
    candidate_refs: tuple[ArtifactRef, ...]
    evaluation_refs: tuple[ArtifactRef, ...]
    query_count: Annotated[int, Field(ge=0, strict=True)]
    total_evaluations: Annotated[int, Field(ge=0, strict=True)]
    total_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_tool_calls: Annotated[int, Field(ge=0, strict=True)]
    total_wall_time_seconds: Annotated[float, Field(ge=0, strict=True)]
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
            ("usage_entry_ref", EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE),
            ("evaluation_ref", GATE_EVALUATION_MANIFEST_MEDIA_TYPE),
            ("terminal_decision_report_ref", TERMINAL_DECISION_REPORT_MEDIA_TYPE),
        )
        for field_name, media_type in expected:
            if getattr(self, field_name).media_type != media_type:
                raise ValueError(f"{field_name} must declare {media_type!r}")
        if self.terminal_state not in {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        }:
            raise ValueError("terminal_state is not a gate terminal outcome")
        if self.gate_terminal_state not in {
            CandidateState.PROMOTED,
            CandidateState.REJECTED,
            CandidateState.INCONCLUSIVE,
        }:
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


class ExperimentController:
    """Sole semantic transition writer for one explicitly frozen experiment.

    The controller never accepts a caller-reported evaluation count.  It loads
    both exact-media gate trial batches and charges every observation in both
    arms against both the experiment and protocol ``max_evaluations`` limits.

    The optional usage tail supports controller re-instantiation and read-only
    replay only while the original in-memory gate-batch and mechanism-evidence
    verification capabilities remain available.  This is not process-restart
    recovery.  Candidate lifecycle heads intentionally remain process-local in
    M0.2; callers must register candidates through this instance.  Durable
    multi-process head discovery and locking are future distributed-controller
    work, not a property of content addressing alone.
    """

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        experiment_ref: ArtifactRef,
        gate_batch_verifier: GateBatchVerificationCapability,
        mechanism_evidence_verifier: MechanismEvidenceVerificationCapability,
        usage_tail_ref: ArtifactRef | None = None,
        experiment_tail_ref: ArtifactRef | None = None,
    ) -> None:
        # These process-local objects are verification capabilities, not
        # extension points.  Accepting a subclass would let caller code
        # override ``attestor_id`` and ``verify`` while still satisfying an
        # ``isinstance`` check.
        if type(gate_batch_verifier) is not GateBatchVerificationCapability:
            raise TypeError("gate_batch_verifier must be a GateBatchVerificationCapability")
        if type(mechanism_evidence_verifier) is not MechanismEvidenceVerificationCapability:
            raise TypeError(
                "mechanism_evidence_verifier must be a MechanismEvidenceVerificationCapability"
            )
        if experiment_tail_ref is not None:
            raise ExperimentControllerError(
                "M0.2 does not support trusted experiment lifecycle resume; "
                "experiment_tail_ref must be omitted"
            )
        self._repository = repository
        self._gate_batch_verifier = gate_batch_verifier
        self._mechanism_evidence_verifier = mechanism_evidence_verifier
        self.experiment_ref = ArtifactRef.model_validate(experiment_ref)
        self._experiment = self._load(
            self.experiment_ref,
            ExperimentManifest,
            "frozen experiment",
            expected_media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
        )
        self._protocol = self._load(
            self._experiment.protocol_ref,
            ProtocolManifest,
            "frozen protocol",
            expected_media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
        )
        if self._gate_batch_verifier.attestor_id != self._protocol.gate_batch_attestor_id:
            raise ExperimentControllerError(
                "gate batch verifier does not match the protocol-frozen attestor"
            )
        if (
            self._mechanism_evidence_verifier.attestor_id
            != self._protocol.mechanism_evidence_attestor_id
        ):
            raise ExperimentControllerError(
                "mechanism evidence verifier does not match the protocol-frozen attestor"
            )
        self.protocol_ref = self._experiment.protocol_ref
        self._budget_limits = self._validated_budget_limits()
        if self._budget_limits.max_evaluations is None:  # pragma: no cover - schema invariant
            raise ExperimentControllerError("frozen budgets must cap max_evaluations")
        self._max_evaluations = self._budget_limits.max_evaluations
        self._journal = CandidateJournal(repository)
        self._experiment_journal = ExperimentJournal(repository)
        self._admission = CandidateAdmissionService(repository)
        self._terminal = TerminalDecisionService(
            repository,
            gate_batch_verifier=self._gate_batch_verifier,
            mechanism_evidence_verifier=self._mechanism_evidence_verifier,
        )
        self._candidate_tails: dict[str, ArtifactRef] = {}
        self._candidate_refs: dict[str, ArtifactRef] = {}
        self._champion_harness_ref = self._experiment.seed_harness_ref
        self._champion_candidate_ref: ArtifactRef | None = None
        self._usage_tail_ref = (
            None if usage_tail_ref is None else ArtifactRef.model_validate(usage_tail_ref)
        )
        # A supplied restart head is not trusted merely because it exists.
        self._replay_usage(self._usage_tail_ref)
        self._experiment_tail_ref: ArtifactRef | None = None
        self._candidate_resume_blocked = usage_tail_ref is not None

    @property
    def usage_tail_ref(self) -> ArtifactRef | None:
        """Return the current caller-held usage head for this writer."""

        return self._usage_tail_ref

    @property
    def experiment_tail_ref(self) -> ArtifactRef | None:
        """Return the current caller-held experiment lifecycle head."""

        return self._experiment_tail_ref

    @property
    def experiment_state(self) -> ExperimentState | None:
        """Return the replayed current experiment state, if frozen."""

        if self._experiment_tail_ref is None:
            return None
        return self._experiment_journal.replay(self._experiment_tail_ref)[-1].to_state

    def freeze_experiment(self) -> ArtifactRef:
        """Publish the unique ``FROZEN`` root for this explicit manifest."""

        if self._experiment_tail_ref is not None:
            raise ExperimentControllerError("experiment lifecycle is already frozen")
        event = ExperimentLifecycleEvent(
            experiment_ref=self.experiment_ref,
            from_state=None,
            to_state=ExperimentState.FROZEN,
            reason="protocol, experiment, seed harness, and budgets were frozen",
        )
        tail_ref = self._experiment_journal.append(
            experiment_ref=self.experiment_ref,
            event=event,
        )
        self._experiment_tail_ref = tail_ref
        return tail_ref

    def start_search(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        """Open bounded candidate search from the exact frozen head."""

        self._require_experiment_tail(
            previous_tail_ref,
            expected_state=ExperimentState.FROZEN,
        )
        if self._usage_tail_ref is not None:
            raise ExperimentControllerError("experiment has usage before search was opened")
        return self._append_experiment_event(
            previous_tail_ref=previous_tail_ref,
            from_state=ExperimentState.FROZEN,
            to_state=ExperimentState.SEARCHING,
            evidence_refs=(),
            usage_tail_ref=None,
            reason="bounded candidate search opened",
        )

    def close_selection(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        previous_usage_tail_ref: ArtifactRef | None,
        champion_candidate_ref: ArtifactRef | None,
        champion_candidate_tail_ref: ArtifactRef | None,
        champion_harness_ref: ArtifactRef,
        analysis_plan_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Freeze the promoted champion, query head, and sealed analysis plan."""

        self._require_experiment_tail(
            previous_tail_ref,
            expected_state=ExperimentState.SEARCHING,
        )
        if self._candidate_resume_blocked:
            raise ExperimentControllerError(
                "M0.2 cannot close a resumed search without durable candidate-head recovery"
            )
        self._require_current_usage_tail(previous_usage_tail_ref)
        active = tuple(
            candidate_ref
            for digest, candidate_ref in self._candidate_refs.items()
            if self._journal.replay(self._candidate_tails[digest])[-1].to_state
            not in TERMINAL_CANDIDATE_STATES
        )
        if active:
            raise ExperimentControllerError(
                "selection cannot close while registered candidates remain nonterminal"
            )
        champion_harness_ref = ArtifactRef.model_validate(champion_harness_ref)
        if champion_harness_ref != self._champion_harness_ref:
            raise ExperimentControllerError(
                "selection harness is not the controller's current champion"
            )
        usage = self._replay_usage(self._usage_tail_ref)
        if self._champion_candidate_ref is None:
            if champion_candidate_ref is not None or champion_candidate_tail_ref is not None:
                raise ExperimentControllerError(
                    "seed fallback must not claim a promoted champion candidate"
                )
            selection_reason = SelectionReason.NO_PROMOTABLE_CANDIDATES
        else:
            if champion_candidate_ref != self._champion_candidate_ref:
                raise ExperimentControllerError(
                    "selection candidate is not the controller's current promoted champion"
                )
            if champion_candidate_tail_ref is None:
                raise ExperimentControllerError("promoted champion requires its terminal tail")
            candidate_events = self._require_current_tail(
                champion_candidate_ref,
                champion_candidate_tail_ref,
                expected_state=CandidateState.PROMOTED,
            )
            if candidate_events[-1].to_state is not CandidateState.PROMOTED:
                raise ExperimentControllerError("selection champion was not promoted")
            champion = self._load(
                champion_candidate_ref,
                CandidateManifest,
                "champion candidate",
            )
            if champion.child_harness_ref != champion_harness_ref:
                raise ExperimentControllerError(
                    "selected champion harness does not match the candidate child"
                )
            if champion_candidate_ref not in usage.candidate_refs:
                raise ExperimentControllerError("selected champion has no charged gate evaluation")
            selection_reason = SelectionReason.PROMOTED_CHAMPION
        self._load_json(analysis_plan_ref, "sealed analysis plan")

        closure = SelectionClosure(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            champion_candidate_ref=champion_candidate_ref,
            champion_candidate_tail_ref=champion_candidate_tail_ref,
            champion_harness_ref=champion_harness_ref,
            analysis_plan_ref=analysis_plan_ref,
            usage_tail_ref=self._usage_tail_ref,
            selection_reason=selection_reason,
            stopping_criteria=self._experiment.stopping,
        )
        closure_ref = self._repository.put_json(
            closure,
            media_type=SELECTION_CLOSURE_MEDIA_TYPE,
        )
        return self._append_experiment_event(
            previous_tail_ref=previous_tail_ref,
            from_state=ExperimentState.SEARCHING,
            to_state=ExperimentState.SELECTION_CLOSED,
            evidence_refs=(closure_ref,),
            usage_tail_ref=self._usage_tail_ref,
            reason="champion, usage head, and sealed analysis plan were frozen",
        )

    def close_current_selection(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        analysis_plan_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Close selection from controller-owned heads for search orchestration.

        The lower-level compatibility API accepts explicit champion and usage
        references so old callers can prove their expected view.  An automatic
        search loop must not reconstruct those values from plugin output.  This
        method projects the only current candidate/usage heads owned by this
        controller and leaves the caller responsible solely for supplying the
        analysis plan already frozen in its typed search-run manifest.
        """

        champion_candidate_ref = self._champion_candidate_ref
        champion_candidate_tail_ref = (
            None
            if champion_candidate_ref is None
            else self._candidate_tails[champion_candidate_ref.sha256]
        )
        return self.close_selection(
            previous_tail_ref=previous_tail_ref,
            previous_usage_tail_ref=self._usage_tail_ref,
            champion_candidate_ref=champion_candidate_ref,
            champion_candidate_tail_ref=champion_candidate_tail_ref,
            champion_harness_ref=self._champion_harness_ref,
            analysis_plan_ref=analysis_plan_ref,
        )

    def verify_experiment_selection_closure(
        self,
        tail_ref: ArtifactRef,
    ) -> SelectionClosure:
        """Re-authenticate this controller's exact current selection closure.

        Study-level orchestration must not treat a caller-authored
        :class:`SelectionClosure` as evidence.  This read-only capability
        replays the current controller-owned lifecycle branch and rejoins the
        closure with champion, candidate, usage, and analysis heads without
        starting sealed evaluation.
        """

        checked_ref = ArtifactRef.model_validate(tail_ref)
        events = self._require_experiment_tail(
            checked_ref,
            expected_state=ExperimentState.SELECTION_CLOSED,
        )
        terminal_event = events[-1]
        closure_ref = self._only_evidence_ref(
            terminal_event,
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
            label="selection closure",
        )
        closure = self._load(
            closure_ref,
            SelectionClosure,
            "selection closure",
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
        )
        if closure.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("selection closure belongs to another experiment")
        if closure.protocol_ref != self.protocol_ref:
            raise ExperimentControllerError("selection closure belongs to another protocol")
        if closure.usage_tail_ref != self._usage_tail_ref:
            raise ExperimentControllerError("selection closure does not bind current usage")
        if terminal_event.usage_tail_ref != self._usage_tail_ref:
            raise ExperimentControllerError("selection event does not bind current usage")
        if closure.champion_harness_ref != self._champion_harness_ref:
            raise ExperimentControllerError("selection closure does not bind current champion")
        if closure.stopping_criteria != self._experiment.stopping:
            raise ExperimentControllerError("selection closure stopping criteria changed")
        self._load_json(closure.analysis_plan_ref, "sealed analysis plan")

        champion_candidate_ref = self._champion_candidate_ref
        if champion_candidate_ref is None:
            if (
                closure.champion_candidate_ref is not None
                or closure.champion_candidate_tail_ref is not None
            ):
                raise ExperimentControllerError(
                    "seed selection closure must not claim a champion candidate"
                )
            if closure.selection_reason is not SelectionReason.NO_PROMOTABLE_CANDIDATES:
                raise ExperimentControllerError("seed selection closure has the wrong reason")
        else:
            if closure.champion_candidate_ref != champion_candidate_ref:
                raise ExperimentControllerError(
                    "selection closure does not bind current champion candidate"
                )
            expected_candidate_tail = self._candidate_tails[champion_candidate_ref.sha256]
            if closure.champion_candidate_tail_ref != expected_candidate_tail:
                raise ExperimentControllerError(
                    "selection closure does not bind the champion candidate tail"
                )
            candidate_events = self._require_current_tail(
                champion_candidate_ref,
                expected_candidate_tail,
                expected_state=CandidateState.PROMOTED,
            )
            if candidate_events[-1].to_state is not CandidateState.PROMOTED:
                raise ExperimentControllerError("selection champion is not promoted")
            candidate = self._load(
                champion_candidate_ref,
                CandidateManifest,
                "champion candidate",
            )
            if candidate.child_harness_ref != closure.champion_harness_ref:
                raise ExperimentControllerError(
                    "selection candidate child does not match champion harness"
                )
            usage = self._replay_usage(self._usage_tail_ref)
            if champion_candidate_ref not in usage.candidate_refs:
                raise ExperimentControllerError("selection champion has no charged gate evaluation")
            if closure.selection_reason is not SelectionReason.PROMOTED_CHAMPION:
                raise ExperimentControllerError("promoted selection closure has the wrong reason")
        return closure

    def start_sealed(self, *, previous_tail_ref: ArtifactRef) -> ArtifactRef:
        """Authorize sealed access only along the exact selection branch."""

        events = self._require_experiment_tail(
            previous_tail_ref,
            expected_state=ExperimentState.SELECTION_CLOSED,
        )
        closure_ref = self._only_evidence_ref(
            events[-1],
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
            label="selection closure",
        )
        closure = self._load(
            closure_ref,
            SelectionClosure,
            "selection closure",
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
        )
        if closure.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("selection closure belongs to another experiment")
        if closure.protocol_ref != self.protocol_ref:
            raise ExperimentControllerError("selection closure belongs to another protocol")
        if closure.usage_tail_ref != self._usage_tail_ref:
            raise ExperimentControllerError("selection closure does not bind current usage")
        sealed_splits = tuple(
            split.manifest_ref
            for split in self._protocol.splits
            if split.partition is ProtocolPartition.SEALED
        )
        if len(sealed_splits) != 1:
            raise ExperimentControllerError(
                "sealed evaluation requires exactly one frozen SEALED protocol split"
            )
        sealed_split_ref = sealed_splits[0]
        self._load_json(sealed_split_ref, "sealed split manifest")
        authorization = SealedRunAuthorization(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            selection_closed_tail_ref=previous_tail_ref,
            selection_closure_ref=closure_ref,
            champion_candidate_ref=closure.champion_candidate_ref,
            champion_harness_ref=closure.champion_harness_ref,
            analysis_plan_ref=closure.analysis_plan_ref,
            sealed_split_ref=sealed_split_ref,
            usage_tail_ref=closure.usage_tail_ref,
        )
        authorization_ref = self._repository.put_json(
            authorization,
            media_type=SEALED_RUN_AUTHORIZATION_MEDIA_TYPE,
        )
        return self._append_experiment_event(
            previous_tail_ref=previous_tail_ref,
            from_state=ExperimentState.SELECTION_CLOSED,
            to_state=ExperimentState.SEALED_RUNNING,
            evidence_refs=(authorization_ref,),
            usage_tail_ref=closure.usage_tail_ref,
            reason="sealed evaluation authorized from the exact closed selection branch",
        )

    def complete_experiment(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        sealed_evaluation_report_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Replay a typed sealed-evaluator report and finish the experiment."""

        events = self._require_experiment_tail(
            previous_tail_ref,
            expected_state=ExperimentState.SEALED_RUNNING,
        )
        authorization_ref = self._only_evidence_ref(
            events[-1],
            expected_media_type=SEALED_RUN_AUTHORIZATION_MEDIA_TYPE,
            label="sealed run authorization",
        )
        authorization = self._load(
            authorization_ref,
            SealedRunAuthorization,
            "sealed run authorization",
            expected_media_type=SEALED_RUN_AUTHORIZATION_MEDIA_TYPE,
        )
        if authorization.usage_tail_ref != self._usage_tail_ref:
            raise ExperimentControllerError("sealed authorization usage head changed")
        selection_events = self._experiment_journal.replay(authorization.selection_closed_tail_ref)
        closure_ref = self._only_evidence_ref(
            selection_events[-1],
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
            label="selection closure",
        )
        if closure_ref != authorization.selection_closure_ref:
            raise ExperimentControllerError(
                "sealed authorization does not bind its selection closure"
            )
        closure = self._load(
            closure_ref,
            SelectionClosure,
            "selection closure",
            expected_media_type=SELECTION_CLOSURE_MEDIA_TYPE,
        )
        sealed_report = self._load(
            sealed_evaluation_report_ref,
            SealedEvaluationReport,
            "sealed evaluation report",
            expected_media_type=SEALED_EVALUATION_REPORT_MEDIA_TYPE,
        )
        expected_report_fields = {
            "experiment_ref": self.experiment_ref,
            "protocol_ref": self.protocol_ref,
            "sealed_running_tail_ref": previous_tail_ref,
            "sealed_authorization_ref": authorization_ref,
            "selection_closure_ref": closure_ref,
            "champion_candidate_ref": closure.champion_candidate_ref,
            "champion_harness_ref": closure.champion_harness_ref,
            "analysis_plan_ref": closure.analysis_plan_ref,
            "sealed_split_ref": authorization.sealed_split_ref,
            "usage_tail_ref": closure.usage_tail_ref,
            "model_fingerprint": self._protocol.model_fingerprint,
            "inference_fingerprint": self._protocol.inference_fingerprint,
            "runtime_fingerprint": self._protocol.runtime_fingerprint,
            "sandbox_fingerprint": self._protocol.sandbox_fingerprint,
            "grader_fingerprint": self._protocol.grader_fingerprint,
            "capability_policy_ref": self._protocol.capability_policy_ref,
        }
        mismatched = tuple(
            field_name
            for field_name, expected in expected_report_fields.items()
            if getattr(sealed_report, field_name) != expected
        )
        if mismatched:
            raise ExperimentControllerError(
                "sealed evaluation report does not match frozen authorization: "
                + ", ".join(mismatched)
            )
        self._load_json(sealed_report.result_ref, "sealed evaluation result")
        for evidence_ref in sealed_report.evidence_refs:
            try:
                self._repository.get_bytes(evidence_ref)
            except Exception as exc:
                raise ExperimentControllerError(
                    f"could not verify sealed evaluation evidence: {exc}"
                ) from exc
        completion = ExperimentCompletionReport(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            sealed_running_tail_ref=previous_tail_ref,
            sealed_authorization_ref=authorization_ref,
            sealed_evaluation_report_ref=sealed_evaluation_report_ref,
            usage_tail_ref=authorization.usage_tail_ref,
        )
        completion_ref = self._repository.put_json(
            completion,
            media_type=EXPERIMENT_COMPLETION_REPORT_MEDIA_TYPE,
        )
        return self._append_experiment_event(
            previous_tail_ref=previous_tail_ref,
            from_state=ExperimentState.SEALED_RUNNING,
            to_state=ExperimentState.COMPLETE,
            evidence_refs=(sealed_evaluation_report_ref, completion_ref),
            usage_tail_ref=authorization.usage_tail_ref,
            reason="sealed final report was bound to the frozen selection lineage",
        )

    def invalidate_experiment(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        violation_code: ExperimentViolationCode,
        evidence_refs: tuple[ArtifactRef, ...],
        message: str,
    ) -> ArtifactRef:
        """Create typed integrity/leakage evidence and invalidate a nonterminal lineage."""

        events = self._require_experiment_tail(previous_tail_ref)
        source_state = events[-1].to_state
        if source_state in TERMINAL_EXPERIMENT_STATES:
            raise ExperimentControllerError("terminal experiment cannot be invalidated again")
        if not evidence_refs:
            raise ExperimentControllerError("experiment invalidation requires evidence")
        validated_evidence = tuple(ArtifactRef.model_validate(ref) for ref in evidence_refs)
        for evidence_ref in validated_evidence:
            try:
                self._repository.get_bytes(evidence_ref)
            except Exception as exc:
                raise ExperimentControllerError(
                    f"could not verify invalidation evidence: {exc}"
                ) from exc
        report = ExperimentInvalidationReport(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            source_tail_ref=previous_tail_ref,
            source_state=source_state,
            violation_code=violation_code,
            evidence_refs=validated_evidence,
            usage_tail_ref=self._usage_tail_ref,
            message=message,
        )
        report_ref = self._repository.put_json(
            report,
            media_type=EXPERIMENT_INVALIDATION_REPORT_MEDIA_TYPE,
        )
        return self._append_experiment_event(
            previous_tail_ref=previous_tail_ref,
            from_state=source_state,
            to_state=ExperimentState.INVALIDATED,
            evidence_refs=(report_ref,),
            usage_tail_ref=self._usage_tail_ref,
            reason="trusted controller invalidated the experiment after a proved violation",
        )

    def register_candidate(self, *, candidate_ref: ArtifactRef) -> ArtifactRef:
        """Register one canonical candidate from the frozen experiment."""

        self._require_searching()
        candidate_ref = ArtifactRef.model_validate(candidate_ref)
        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")
        if candidate.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError(
                "candidate does not belong to the controller's frozen experiment"
            )
        if candidate.parent_harness_ref != self._champion_harness_ref:
            raise ExperimentControllerError(
                "candidate parent is not the controller's current champion harness"
            )
        if candidate_ref.sha256 in self._candidate_tails:
            raise ExperimentControllerError("candidate is already registered in this controller")

        event = CandidateLifecycleEvent(
            candidate_ref=candidate_ref,
            from_state=None,
            to_state=CandidateState.REGISTERED,
            reason="candidate manifest was frozen under the controller experiment",
        )
        tail_ref = self._journal.append(
            stream_id=self._stream_id(candidate_ref),
            event=event,
        )
        self._candidate_refs[candidate_ref.sha256] = candidate_ref
        self._candidate_tails[candidate_ref.sha256] = tail_ref
        return tail_ref

    def admit_candidate(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        admission_report_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Replay admission, publishing ``VALID`` or a proved ``INVALID``.

        The caller cannot choose the failure state or supply a success flag.
        Any failed replay is converted into a typed controller-authored report
        before the candidate is irreversibly invalidated.
        """

        self._require_searching()
        events = self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.REGISTERED,
        )
        if len(events) != 1 or events[0].evidence_refs:
            raise ExperimentControllerError("registered tail is not controller-canonical")
        try:
            self._admission.verify_report(
                candidate_ref=candidate_ref,
                experiment_ref=self.experiment_ref,
                report_ref=admission_report_ref,
            )
        except Exception as exc:
            error_code = (
                AdmissionFailureCode.REPORT_REPLAY_FAILED
                if isinstance(exc, CandidateAdmissionError)
                else AdmissionFailureCode.VERIFIER_FAILURE
            )
            message = str(exc).strip() or type(exc).__name__
            failure = AdmissionFailureReport(
                experiment_ref=self.experiment_ref,
                candidate_ref=candidate_ref,
                attempted_admission_report_ref=admission_report_ref,
                error_code=error_code,
                message=message,
            )
            failure_ref = self._repository.put_json(
                failure,
                media_type=ADMISSION_FAILURE_REPORT_MEDIA_TYPE,
            )
            return self._append_candidate_event(
                candidate_ref=candidate_ref,
                previous_tail_ref=previous_tail_ref,
                from_state=CandidateState.REGISTERED,
                to_state=CandidateState.INVALID,
                evidence_refs=(failure_ref,),
                reason="trusted admission replay failed closed",
            )

        return self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.REGISTERED,
            to_state=CandidateState.VALID,
            evidence_refs=(admission_report_ref,),
            reason="trusted admission report replayed against the frozen experiment",
        )

    def start_probes(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Enter the preregistered mechanism-probe stage."""

        self._require_searching()
        self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.VALID,
        )
        return self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.VALID,
            to_state=CandidateState.RUNNING_PROBES,
            evidence_refs=(),
            reason="candidate entered preregistered mechanism probes",
        )

    def start_gate(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Resolve probe evidence into ``RUNNING_GATE`` or ``REJECTED``.

        The transition is derived exclusively from the loaded mechanism
        artifact and frozen gate config.  Callers cannot report their own
        passed/failed boolean.
        """

        self._require_searching()
        self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.RUNNING_PROBES,
        )
        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")
        evidence, required, failed, missing = self._mechanism_status(
            candidate_ref,
            candidate,
            mechanism_evidence_ref,
        )
        del evidence
        if failed or missing:
            if failed and missing:
                error_code = ProbeRejectionCode.REQUIRED_CHECKS_FAILED_AND_MISSING
            elif failed:
                error_code = ProbeRejectionCode.REQUIRED_CHECK_FAILED
            else:
                error_code = ProbeRejectionCode.REQUIRED_CHECK_MISSING
            rejection = ProbeRejectionReport(
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=candidate_ref,
                mechanism_evidence_ref=mechanism_evidence_ref,
                error_code=error_code,
                required_checks=required,
                failed_checks=failed,
                missing_checks=missing,
            )
            rejection_ref = self._repository.put_json(
                rejection,
                media_type=PROBE_REJECTION_REPORT_MEDIA_TYPE,
            )
            return self._append_candidate_event(
                candidate_ref=candidate_ref,
                previous_tail_ref=previous_tail_ref,
                from_state=CandidateState.RUNNING_PROBES,
                to_state=CandidateState.REJECTED,
                evidence_refs=(mechanism_evidence_ref, rejection_ref),
                reason="required mechanism probes failed or were missing",
            )
        return self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.RUNNING_PROBES,
            to_state=CandidateState.RUNNING_GATE,
            evidence_refs=(mechanism_evidence_ref,),
            reason="candidate-bound required mechanism checks passed",
        )

    def complete_evidence(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
        previous_usage_tail_ref: ArtifactRef | None,
    ) -> EvidenceCompletion:
        """Account a gate query and publish its evidence-complete lifecycle head."""

        self._require_searching()
        events = self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.RUNNING_GATE,
        )
        candidate, admission_report_ref, mechanism_evidence_ref = self._verify_running_gate_history(
            previous_tail_ref, expected_events=events
        )
        evaluation, parent_batch, candidate_batch, evaluation_units = self._verify_gate_evaluation(
            candidate_ref=candidate_ref,
            candidate=candidate,
            admission_report_ref=admission_report_ref,
            mechanism_evidence_ref=mechanism_evidence_ref,
            evaluation_ref=evaluation_ref,
        )
        self._require_current_usage_tail(previous_usage_tail_ref)
        usage = self._replay_usage(self._usage_tail_ref)
        self._reject_duplicate_usage(
            usage,
            candidate_ref=candidate_ref,
            evaluation_ref=evaluation_ref,
            parent_batch_ref=evaluation.parent_batch_ref,
            candidate_batch_ref=evaluation.candidate_batch_ref,
        )
        tokens, tool_calls, wall_time_seconds, cost_usd = self._resource_charge(
            parent_batch,
            candidate_batch,
        )
        cumulative = usage.total_evaluations + evaluation_units
        cumulative_tokens = usage.total_tokens + tokens
        cumulative_tool_calls = usage.total_tool_calls + tool_calls
        cumulative_wall_time = usage.total_wall_time_seconds + wall_time_seconds
        cumulative_cost = (
            None
            if usage.total_cost_usd is None or cost_usd is None
            else usage.total_cost_usd + cost_usd
        )
        self._enforce_budget(
            evaluations=cumulative,
            tokens=cumulative_tokens,
            tool_calls=cumulative_tool_calls,
            wall_time_seconds=cumulative_wall_time,
            cost_usd=cumulative_cost,
            requested_evaluations=evaluation_units,
            prior_evaluations=usage.total_evaluations,
        )

        # These objects are written before the lifecycle event, but no mutable
        # head is published until every append succeeds.  A crash can therefore
        # leave unreachable CAS objects, never an authorized partial state.
        claim = ExperimentUsageClaim(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            candidate_ref=candidate_ref,
            running_gate_tail_ref=previous_tail_ref,
            evaluation_ref=evaluation_ref,
            parent_batch_ref=evaluation.parent_batch_ref,
            candidate_batch_ref=evaluation.candidate_batch_ref,
            evaluation_units=evaluation_units,
            tokens=tokens,
            tool_calls=tool_calls,
            wall_time_seconds=wall_time_seconds,
            cost_usd=cost_usd,
        )
        claim_ref = self._repository.put_json(
            claim,
            media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
        )
        entry = ExperimentUsageEntry(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            sequence=usage.query_count,
            claim_ref=claim_ref,
            cumulative_evaluations=cumulative,
            cumulative_tokens=cumulative_tokens,
            cumulative_tool_calls=cumulative_tool_calls,
            cumulative_wall_time_seconds=cumulative_wall_time,
            cumulative_cost_usd=cumulative_cost,
            previous_entry_ref=self._usage_tail_ref,
        )
        usage_tail_ref = self._repository.put_json(
            entry,
            media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
        )
        candidate_tail_ref = self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.RUNNING_GATE,
            to_state=CandidateState.EVIDENCE_COMPLETE,
            evidence_refs=(evaluation_ref, usage_tail_ref),
            reason="gate evidence closed and both trial arms charged to the frozen budget",
        )
        self._usage_tail_ref = usage_tail_ref
        return EvidenceCompletion(
            candidate_tail_ref=candidate_tail_ref,
            usage_tail_ref=usage_tail_ref,
            usage_claim_ref=claim_ref,
            total_evaluations=cumulative,
            remaining_evaluations=self._max_evaluations - cumulative,
        )

    def finalize_candidate(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        terminal_decision_report_ref: ArtifactRef,
    ) -> TerminalCompletion:
        """Replay a terminal proof and bind it to the exact current branch."""

        self._require_searching()
        events = self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.EVIDENCE_COMPLETE,
        )
        evidence_complete_event = events[-1]
        evaluation_ref, usage_entry_ref = self._evidence_complete_refs(
            evidence_complete_event.evidence_refs
        )
        report = self._load(
            terminal_decision_report_ref,
            TerminalDecisionReport,
            "terminal decision report",
            expected_media_type=TERMINAL_DECISION_REPORT_MEDIA_TYPE,
        )
        try:
            verified = self._terminal.verify_report(
                terminal_decision_report_ref,
                candidate_ref=candidate_ref,
                experiment_ref=self.experiment_ref,
                evaluation_ref=evaluation_ref,
            )
        except TerminalDecisionError as exc:
            raise ExperimentControllerError(f"terminal decision replay failed: {exc}") from exc
        if verified != report:
            raise ExperimentControllerError("terminal report changed during trusted replay")
        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")

        usage = self._replay_usage(self._usage_tail_ref)
        if usage_entry_ref not in usage.entry_refs:
            raise ExperimentControllerError(
                "evidence-complete usage entry is not in the current experiment ledger"
            )
        usage_entry = self._load(
            usage_entry_ref,
            ExperimentUsageEntry,
            "experiment usage entry",
            expected_media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
        )
        claim = self._load(
            usage_entry.claim_ref,
            ExperimentUsageClaim,
            "experiment usage claim",
            expected_media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
        )
        evidence_complete_entry = self._load(
            previous_tail_ref,
            JournalEntry,
            "evidence-complete journal entry",
            expected_media_type=JOURNAL_ENTRY_MEDIA_TYPE,
        )
        if claim.candidate_ref != candidate_ref or claim.evaluation_ref != evaluation_ref:
            raise ExperimentControllerError(
                "evidence-complete usage claim belongs to another candidate or evaluation"
            )
        if claim.running_gate_tail_ref != evidence_complete_entry.previous_entry_ref:
            raise ExperimentControllerError(
                "usage claim is not bound to this evidence-complete journal branch"
            )

        valid_event = next(event for event in events if event.to_state is CandidateState.VALID)
        if verified.admission_report_ref not in valid_event.evidence_refs:
            raise ExperimentControllerError(
                "terminal report does not reuse this branch's admission report"
            )

        resolved_terminal_state = verified.terminal_state
        superseded_report_ref: ArtifactRef | None = None
        if (
            verified.terminal_state is CandidateState.PROMOTED
            and candidate.parent_harness_ref != self._champion_harness_ref
        ):
            if self._champion_candidate_ref is None:
                raise ExperimentControllerError(
                    "champion changed without a replayable superseding candidate"
                )
            superseded = SupersededCandidateReport(
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=candidate_ref,
                evidence_complete_tail_ref=previous_tail_ref,
                terminal_decision_report_ref=terminal_decision_report_ref,
                decision_ref=verified.decision_ref,
                stale_parent_harness_ref=candidate.parent_harness_ref,
                current_champion_harness_ref=self._champion_harness_ref,
                superseding_candidate_ref=self._champion_candidate_ref,
            )
            superseded_report_ref = self._repository.put_json(
                superseded,
                media_type=SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE,
            )
            resolved_terminal_state = CandidateState.INCONCLUSIVE

        authorization = TerminalTransitionAuthorization(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            candidate_ref=candidate_ref,
            prior_champion_harness_ref=self._champion_harness_ref,
            parent_harness_ref=candidate.parent_harness_ref,
            child_harness_ref=candidate.child_harness_ref,
            evidence_complete_tail_ref=previous_tail_ref,
            usage_entry_ref=usage_entry_ref,
            evaluation_ref=evaluation_ref,
            terminal_decision_report_ref=terminal_decision_report_ref,
            decision_ref=verified.decision_ref,
            gate_terminal_state=verified.terminal_state,
            terminal_state=resolved_terminal_state,
            superseded_report_ref=superseded_report_ref,
        )
        authorization_ref = self._repository.put_json(
            authorization,
            media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
        )
        terminal_evidence = [
            verified.decision_ref,
            terminal_decision_report_ref,
            authorization_ref,
        ]
        if superseded_report_ref is not None:
            terminal_evidence.append(superseded_report_ref)
        tail_ref = self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.EVIDENCE_COMPLETE,
            to_state=resolved_terminal_state,
            evidence_refs=tuple(terminal_evidence),
            reason=(
                "local promotion was superseded by a sibling champion advance"
                if superseded_report_ref is not None
                else "terminal gate decision replayed and bound to this exact evidence branch"
            ),
        )
        if resolved_terminal_state is CandidateState.PROMOTED:
            self._champion_harness_ref = candidate.child_harness_ref
            self._champion_candidate_ref = candidate_ref
        return TerminalCompletion(
            candidate_tail_ref=tail_ref,
            authorization_ref=authorization_ref,
            superseded_report_ref=superseded_report_ref,
        )

    def verify_terminal_authorization(
        self,
        authorization_ref: ArtifactRef,
    ) -> TerminalTransitionAuthorization:
        """Re-authenticate one terminal result produced by this controller.

        A content-addressed ``TerminalTransitionAuthorization`` is only a typed
        assertion: any repository writer can persist an object with that
        schema.  Adaptive-search code must therefore consume this method as a
        process-local verification capability instead of trusting a bare CAS
        read.  The authorization is accepted only when it is embedded in the
        current controller-owned terminal candidate branch and its complete
        decision proof still replays under the protocol-pinned verifiers.
        """

        checked_ref = ArtifactRef.model_validate(authorization_ref)
        authorization = self._load(
            checked_ref,
            TerminalTransitionAuthorization,
            "terminal transition authorization",
            expected_media_type=TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
        )
        if authorization.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("terminal authorization belongs to another experiment")
        if authorization.protocol_ref != self.protocol_ref:
            raise ExperimentControllerError("terminal authorization belongs to another protocol")

        candidate_ref = authorization.candidate_ref
        known_ref = self._candidate_refs.get(candidate_ref.sha256)
        if known_ref is None:
            raise ExperimentControllerError(
                "terminal authorization candidate is not registered with this controller"
            )
        if known_ref != candidate_ref:
            raise ExperimentControllerError(
                "terminal authorization candidate reference metadata changed"
            )
        candidate_tail_ref = self._candidate_tails[candidate_ref.sha256]
        try:
            events = self._journal.replay(candidate_tail_ref)
        except Exception as exc:
            raise ExperimentControllerError(
                f"terminal authorization candidate replay failed: {exc}"
            ) from exc
        terminal_event = events[-1]
        if terminal_event.to_state is not authorization.terminal_state:
            raise ExperimentControllerError(
                "terminal authorization does not match the candidate terminal state"
            )
        if checked_ref not in terminal_event.evidence_refs:
            raise ExperimentControllerError(
                "terminal authorization was not published by this controller branch"
            )

        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")
        if authorization.parent_harness_ref != candidate.parent_harness_ref:
            raise ExperimentControllerError("terminal authorization parent harness changed")
        if authorization.child_harness_ref != candidate.child_harness_ref:
            raise ExperimentControllerError("terminal authorization child harness changed")

        try:
            report = self._terminal.verify_report(
                authorization.terminal_decision_report_ref,
                candidate_ref=candidate_ref,
                experiment_ref=self.experiment_ref,
                evaluation_ref=authorization.evaluation_ref,
            )
        except TerminalDecisionError as exc:
            raise ExperimentControllerError(
                f"terminal authorization decision replay failed: {exc}"
            ) from exc
        if report.decision_ref != authorization.decision_ref:
            raise ExperimentControllerError("terminal authorization decision reference changed")
        if report.terminal_state is not authorization.gate_terminal_state:
            raise ExperimentControllerError("terminal authorization gate outcome changed")

        usage = self._replay_usage(self._usage_tail_ref)
        if authorization.usage_entry_ref not in usage.entry_refs:
            raise ExperimentControllerError(
                "terminal authorization usage entry is not in the controller ledger"
            )
        usage_entry = self._load(
            authorization.usage_entry_ref,
            ExperimentUsageEntry,
            "terminal authorization usage entry",
            expected_media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
        )
        claim = self._load(
            usage_entry.claim_ref,
            ExperimentUsageClaim,
            "terminal authorization usage claim",
            expected_media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
        )
        if (
            claim.candidate_ref != candidate_ref
            or claim.evaluation_ref != authorization.evaluation_ref
        ):
            raise ExperimentControllerError(
                "terminal authorization usage belongs to another candidate or evaluation"
            )
        return authorization

    def current_usage(self) -> ExperimentUsage:
        """Replay and query the controller's current experiment usage head."""

        return self._replay_usage(self._usage_tail_ref)

    def query_usage(self, tail_ref: ArtifactRef | None) -> ExperimentUsage:
        """Replay any explicit historical head from this frozen experiment."""

        return self._replay_usage(tail_ref)

    def _require_searching(self) -> None:
        if self._experiment_tail_ref is None:
            raise ExperimentControllerError(
                "experiment must be frozen and opened for search before candidate work"
            )
        events = self._experiment_journal.replay(self._experiment_tail_ref)
        if events[-1].to_state is not ExperimentState.SEARCHING:
            raise ExperimentControllerError(
                f"candidate work requires searching experiment; got {events[-1].to_state.value!r}"
            )
        if self._candidate_resume_blocked:
            raise ExperimentControllerError(
                "M0.2 resumed usage/search state is read-only without durable candidate heads"
            )

    def _require_experiment_tail(
        self,
        previous_tail_ref: ArtifactRef,
        *,
        expected_state: ExperimentState | None = None,
    ) -> tuple[ExperimentLifecycleEvent, ...]:
        if self._experiment_tail_ref is None:
            raise ExperimentControllerError("experiment lifecycle has not been frozen")
        previous_tail_ref = ArtifactRef.model_validate(previous_tail_ref)
        if previous_tail_ref != self._experiment_tail_ref:
            raise StaleControllerTailError(
                "experiment tail is stale, foreign, or belongs to another branch"
            )
        try:
            events = self._experiment_journal.replay(previous_tail_ref)
        except Exception as exc:
            raise ExperimentControllerError(f"experiment lifecycle replay failed: {exc}") from exc
        if events[0].experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("experiment lifecycle belongs to another experiment")
        if expected_state is not None and events[-1].to_state is not expected_state:
            raise ExperimentControllerError(
                f"experiment is {events[-1].to_state.value!r}; expected {expected_state.value!r}"
            )
        return events

    def _append_experiment_event(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        from_state: ExperimentState,
        to_state: ExperimentState,
        evidence_refs: tuple[ArtifactRef, ...],
        usage_tail_ref: ArtifactRef | None,
        reason: str,
    ) -> ArtifactRef:
        event = ExperimentLifecycleEvent(
            experiment_ref=self.experiment_ref,
            from_state=from_state,
            to_state=to_state,
            evidence_refs=evidence_refs,
            usage_tail_ref=usage_tail_ref,
            reason=reason,
        )
        try:
            tail_ref = self._experiment_journal.append(
                experiment_ref=self.experiment_ref,
                event=event,
                previous_entry_ref=previous_tail_ref,
            )
        except Exception as exc:
            raise ExperimentControllerError(f"experiment journal append failed: {exc}") from exc
        self._experiment_tail_ref = tail_ref
        return tail_ref

    @staticmethod
    def _only_evidence_ref(
        event: ExperimentLifecycleEvent,
        *,
        expected_media_type: str,
        label: str,
    ) -> ArtifactRef:
        if len(event.evidence_refs) != 1:
            raise ExperimentControllerError(f"experiment event requires exactly one {label}")
        evidence_ref = event.evidence_refs[0]
        if evidence_ref.media_type != expected_media_type:
            raise ExperimentControllerError(f"experiment event binds the wrong {label} type")
        return evidence_ref

    def _validated_budget_limits(self) -> BudgetPolicy:
        """Join experiment/protocol ceilings, rejecting every looser search limit."""

        fields = (
            "max_tokens",
            "max_tool_calls",
            "max_wall_time_seconds",
            "max_cost_usd",
            "max_evaluations",
        )
        effective: dict[str, int | float | None] = {}
        for field_name in fields:
            experiment_limit = getattr(self._experiment.search_budget, field_name)
            protocol_limit = getattr(self._protocol.budget, field_name)
            if protocol_limit is not None and (
                experiment_limit is None or experiment_limit > protocol_limit
            ):
                raise ExperimentControllerError(
                    f"experiment {field_name} exceeds or omits its protocol ceiling"
                )
            effective[field_name] = (
                experiment_limit if experiment_limit is not None else protocol_limit
            )
        if effective["max_evaluations"] is None:
            raise ExperimentControllerError(
                "experiment and protocol must both freeze max_evaluations"
            )
        return BudgetPolicy.model_validate(effective)

    @staticmethod
    def _resource_charge(
        parent_batch: GateTrialBatch,
        candidate_batch: GateTrialBatch,
    ) -> tuple[int, int, float, float | None]:
        observations = (*parent_batch.observations, *candidate_batch.observations)
        tokens = sum(observation.tokens for observation in observations)
        tool_calls = sum(observation.tool_calls for observation in observations)
        wall_time_seconds = sum(observation.latency_ms for observation in observations) / 1_000
        costs = tuple(observation.cost_usd for observation in observations)
        cost_usd = (
            None
            if any(cost is None for cost in costs)
            else sum(cost for cost in costs if cost is not None)
        )
        return tokens, tool_calls, wall_time_seconds, cost_usd

    def _enforce_budget(
        self,
        *,
        evaluations: int,
        tokens: int,
        tool_calls: int,
        wall_time_seconds: float,
        cost_usd: float | None,
        requested_evaluations: int,
        prior_evaluations: int,
    ) -> None:
        limits = self._budget_limits
        if evaluations > self._max_evaluations:
            raise ExperimentBudgetError(
                "gate evaluation would exceed max_evaluations: "
                f"used={prior_evaluations}, requested={requested_evaluations}, "
                f"limit={self._max_evaluations}"
            )
        checks = (
            ("max_tokens", tokens, limits.max_tokens),
            ("max_tool_calls", tool_calls, limits.max_tool_calls),
            ("max_wall_time_seconds", wall_time_seconds, limits.max_wall_time_seconds),
        )
        for field_name, used, limit in checks:
            if limit is not None and used > limit:
                raise ExperimentBudgetError(
                    f"persisted gate usage exceeds {field_name}: used={used}, limit={limit}"
                )
        if limits.max_cost_usd is not None:
            if cost_usd is None:
                raise ExperimentBudgetError(
                    "cost ceiling is active but one or more observations omit cost_usd"
                )
            if cost_usd > limits.max_cost_usd:
                raise ExperimentBudgetError(
                    "persisted gate usage exceeds max_cost_usd: "
                    f"used={cost_usd}, limit={limits.max_cost_usd}"
                )

    def _require_current_tail(
        self,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        *,
        expected_state: CandidateState,
    ) -> tuple[CandidateLifecycleEvent, ...]:
        candidate_ref = ArtifactRef.model_validate(candidate_ref)
        previous_tail_ref = ArtifactRef.model_validate(previous_tail_ref)
        known_candidate_ref = self._candidate_refs.get(candidate_ref.sha256)
        if known_candidate_ref is None:
            raise ExperimentControllerError("candidate is not registered with this controller")
        if known_candidate_ref != candidate_ref:
            raise ExperimentControllerError(
                "candidate reference metadata changed after registration"
            )
        current_tail = self._candidate_tails[candidate_ref.sha256]
        if previous_tail_ref != current_tail:
            raise StaleControllerTailError(
                "candidate tail is stale, foreign, or belongs to another branch"
            )
        try:
            events = self._journal.replay(previous_tail_ref)
        except Exception as exc:
            raise ExperimentControllerError(f"candidate journal replay failed: {exc}") from exc
        if events[-1].candidate_ref != candidate_ref:
            raise ExperimentControllerError("candidate tail belongs to another candidate")
        if events[-1].to_state is not expected_state:
            raise ExperimentControllerError(
                f"candidate is {events[-1].to_state.value!r}; expected {expected_state.value!r}"
            )
        return events

    def _append_candidate_event(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        from_state: CandidateState,
        to_state: CandidateState,
        evidence_refs: Sequence[ArtifactRef],
        reason: str,
    ) -> ArtifactRef:
        event = CandidateLifecycleEvent(
            candidate_ref=candidate_ref,
            from_state=from_state,
            to_state=to_state,
            evidence_refs=tuple(evidence_refs),
            reason=reason,
        )
        try:
            tail_ref = self._journal.append(
                stream_id=self._stream_id(candidate_ref),
                event=event,
                previous_entry_ref=previous_tail_ref,
            )
        except Exception as exc:
            raise ExperimentControllerError(f"candidate journal append failed: {exc}") from exc
        self._candidate_tails[candidate_ref.sha256] = tail_ref
        return tail_ref

    def _verify_running_gate_history(
        self,
        tail_ref: ArtifactRef,
        *,
        expected_events: tuple[CandidateLifecycleEvent, ...] | None = None,
    ) -> tuple[CandidateManifest, ArtifactRef, ArtifactRef]:
        try:
            events = self._journal.replay(tail_ref)
        except Exception as exc:
            raise ExperimentControllerError(f"running-gate journal replay failed: {exc}") from exc
        if expected_events is not None and events != expected_events:
            raise ExperimentControllerError("running-gate history changed during replay")
        states = tuple(event.to_state for event in events)
        expected_states = (
            CandidateState.REGISTERED,
            CandidateState.VALID,
            CandidateState.RUNNING_PROBES,
            CandidateState.RUNNING_GATE,
        )
        if states != expected_states:
            raise ExperimentControllerError("usage claim source is not a canonical gate history")
        if events[0].evidence_refs or events[2].evidence_refs:
            raise ExperimentControllerError("controller-empty lifecycle stages contain evidence")
        if len(events[1].evidence_refs) != 1 or len(events[3].evidence_refs) != 1:
            raise ExperimentControllerError("admission and mechanism stages require exact evidence")

        candidate_ref = events[0].candidate_ref
        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")
        if candidate.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("candidate history belongs to another experiment")
        admission_report_ref = events[1].evidence_refs[0]
        try:
            self._admission.verify_report(
                candidate_ref=candidate_ref,
                experiment_ref=self.experiment_ref,
                report_ref=admission_report_ref,
            )
        except CandidateAdmissionError as exc:
            raise ExperimentControllerError(f"admission history replay failed: {exc}") from exc
        mechanism_evidence_ref = events[3].evidence_refs[0]
        self._verify_mechanism_evidence(
            candidate_ref,
            candidate,
            mechanism_evidence_ref,
            require_all_checks=True,
        )
        return candidate, admission_report_ref, mechanism_evidence_ref

    def _verify_mechanism_evidence(
        self,
        candidate_ref: ArtifactRef,
        candidate: CandidateManifest,
        mechanism_evidence_ref: ArtifactRef,
        *,
        require_all_checks: bool,
    ) -> MechanismEvidence:
        evidence, _, failed, missing = self._mechanism_status(
            candidate_ref,
            candidate,
            mechanism_evidence_ref,
        )
        if require_all_checks and (failed or missing):
            not_passed = (*failed, *missing)
            raise ExperimentControllerError(
                "required mechanism checks did not pass: " + ", ".join(not_passed)
            )
        return evidence

    def _mechanism_status(
        self,
        candidate_ref: ArtifactRef,
        candidate: CandidateManifest,
        mechanism_evidence_ref: ArtifactRef,
    ) -> tuple[MechanismEvidence, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Return trusted required/failed/missing probe status from persisted evidence."""

        envelope = self._load(
            mechanism_evidence_ref,
            AttestedMechanismEvidence,
            "attested mechanism evidence",
            expected_media_type=ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
        )
        try:
            verified = self._mechanism_evidence_verifier.verify(envelope)
        except (TypeError, ValueError) as exc:
            raise ExperimentControllerError(
                f"mechanism evidence attestation failed: {exc}"
            ) from exc
        if verified.attestor_id != self._protocol.mechanism_evidence_attestor_id:
            raise ExperimentControllerError(
                "mechanism evidence attestor does not match the frozen protocol"
            )
        if verified.protocol_ref != self.protocol_ref:
            raise ExperimentControllerError("mechanism evidence belongs to another protocol")
        if verified.candidate_ref != candidate_ref:
            raise ExperimentControllerError("mechanism evidence belongs to another candidate")
        if verified.candidate_harness_ref != candidate.child_harness_ref:
            raise ExperimentControllerError(
                "mechanism evidence is not bound to the candidate child harness"
            )
        exploration_split_ref = next(
            split.manifest_ref
            for split in self._protocol.splits
            if split.partition is ProtocolPartition.EXPLORATION
        )
        if verified.exploration_split_ref != exploration_split_ref:
            raise ExperimentControllerError(
                "mechanism evidence does not use the frozen exploration split"
            )
        if verified.task_set_fingerprint != exploration_split_ref.sha256:
            raise ExperimentControllerError(
                "mechanism evidence task set does not match the exploration split"
            )
        if verified.execution_context != GateBatchExecutionContext.from_protocol(self._protocol):
            raise ExperimentControllerError(
                "mechanism evidence execution context does not match the frozen protocol"
            )
        for source_ref in verified.source_refs:
            try:
                self._repository.get_bytes(source_ref)
            except Exception as exc:
                raise ExperimentControllerError(
                    f"mechanism evidence source artifact could not be verified: {exc}"
                ) from exc
        evidence = verified.evidence
        if evidence.candidate_harness_id != candidate.child_harness_ref.sha256:
            raise ExperimentControllerError(
                "mechanism evidence is not bound to the candidate child harness"
            )
        names = [check.name for check in evidence.checks]
        if len(names) != len(set(names)):
            raise ExperimentControllerError("mechanism evidence contains duplicate checks")
        gate_config = self._load(
            self._protocol.gate_config_ref,
            GateConfig,
            "frozen gate config",
        )
        required = gate_config.required_mechanism_checks
        by_name = {check.name: check for check in evidence.checks}
        failed = tuple(
            name for name in required if name in by_name and by_name[name].passed is False
        )
        missing = tuple(
            name for name in required if name not in by_name or by_name[name].passed is None
        )
        return evidence, required, failed, missing

    def _verify_gate_evaluation(
        self,
        *,
        candidate_ref: ArtifactRef,
        candidate: CandidateManifest,
        admission_report_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
    ) -> tuple[GateEvaluationManifest, GateTrialBatch, GateTrialBatch, int]:
        evaluation = self._load(
            evaluation_ref,
            GateEvaluationManifest,
            "gate evaluation manifest",
            expected_media_type=GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
        )
        if evaluation.candidate_ref != candidate_ref:
            raise ExperimentControllerError("gate evaluation belongs to another candidate")
        if evaluation.admission_report_ref != admission_report_ref:
            raise ExperimentControllerError(
                "gate evaluation does not bind this lifecycle's admission report"
            )
        if evaluation.mechanism_evidence_ref != mechanism_evidence_ref:
            raise ExperimentControllerError(
                "gate evaluation does not bind this lifecycle's mechanism evidence"
            )
        if evaluation.gate_config_ref != self._protocol.gate_config_ref:
            raise ExperimentControllerError("gate evaluation uses a non-frozen gate config")
        gate_split_ref = next(
            split.manifest_ref
            for split in self._protocol.splits
            if split.partition is ProtocolPartition.GATE
        )
        if evaluation.gate_split_ref != gate_split_ref:
            raise ExperimentControllerError("gate evaluation uses a non-frozen gate split")
        if (
            evaluation.gate_implementation_fingerprint
            != self._protocol.gate_implementation_fingerprint
        ):
            raise ExperimentControllerError("gate evaluation uses a non-frozen gate implementation")

        parent_batch = self._load(
            evaluation.parent_batch_ref,
            GateTrialBatch,
            "parent gate trial batch",
            expected_media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
        )
        candidate_batch = self._load(
            evaluation.candidate_batch_ref,
            GateTrialBatch,
            "candidate gate trial batch",
            expected_media_type=GATE_TRIAL_BATCH_MEDIA_TYPE,
        )
        parent_batch = self._verify_attested_gate_batch(
            parent_batch,
            expected_mechanism_evidence_ref=evaluation.mechanism_evidence_ref,
        )
        candidate_batch = self._verify_attested_gate_batch(
            candidate_batch,
            expected_mechanism_evidence_ref=evaluation.mechanism_evidence_ref,
        )
        expected_batches = (
            (parent_batch, GateTrialArm.PARENT, candidate.parent_harness_ref),
            (candidate_batch, GateTrialArm.CANDIDATE, candidate.child_harness_ref),
        )
        for batch, arm, harness_ref in expected_batches:
            if batch.candidate_ref != candidate_ref:
                raise ExperimentControllerError("gate trial batch belongs to another candidate")
            if batch.arm is not arm:
                raise ExperimentControllerError("gate trial batch is assigned to the wrong arm")
            if batch.harness_ref != harness_ref:
                raise ExperimentControllerError("gate trial batch uses the wrong harness")
            if batch.gate_split_ref != gate_split_ref:
                raise ExperimentControllerError("gate trial batch uses the wrong split")
        evaluation_units = len(parent_batch.observations) + len(candidate_batch.observations)
        if evaluation_units <= 0:  # pragma: no cover - batches enforce non-empty observations
            raise ExperimentControllerError("gate evaluation did not contain any observations")
        return evaluation, parent_batch, candidate_batch, evaluation_units

    def _verify_attested_gate_batch(
        self,
        batch: GateTrialBatch,
        *,
        expected_mechanism_evidence_ref: ArtifactRef,
    ) -> GateTrialBatch:
        """Authenticate one batch and rejoin its signed protocol context."""

        try:
            verified = self._gate_batch_verifier.verify(batch)
        except (TypeError, ValueError) as exc:
            raise ExperimentControllerError(f"gate trial batch attestation failed: {exc}") from exc
        if verified.attestor_id != self._protocol.gate_batch_attestor_id:
            raise ExperimentControllerError(
                "gate trial batch attestor does not match the frozen protocol"
            )
        if verified.protocol_ref != self.protocol_ref:
            raise ExperimentControllerError("gate trial batch belongs to another protocol")
        if verified.execution_context != GateBatchExecutionContext.from_protocol(self._protocol):
            raise ExperimentControllerError(
                "gate trial batch execution context does not match the frozen protocol"
            )
        gate_split_ref = next(
            split.manifest_ref
            for split in self._protocol.splits
            if split.partition is ProtocolPartition.GATE
        )
        if verified.task_set_fingerprint != gate_split_ref.sha256:
            raise ExperimentControllerError(
                "gate trial batch task set does not match the frozen gate split"
            )
        if verified.mechanism_evidence_ref != expected_mechanism_evidence_ref:
            raise ExperimentControllerError(
                "gate trial batch does not bind the evaluation mechanism evidence"
            )
        for source_ref in verified.source_refs:
            try:
                self._repository.get_bytes(source_ref)
            except Exception as exc:
                raise ExperimentControllerError(
                    f"gate trial batch source artifact could not be verified: {exc}"
                ) from exc
        return verified

    def _require_current_usage_tail(
        self,
        previous_usage_tail_ref: ArtifactRef | None,
    ) -> None:
        supplied = (
            None
            if previous_usage_tail_ref is None
            else ArtifactRef.model_validate(previous_usage_tail_ref)
        )
        if supplied != self._usage_tail_ref:
            raise StaleControllerTailError(
                "experiment usage tail is stale, foreign, or belongs to another branch"
            )

    def _replay_usage(self, tail_ref: ArtifactRef | None) -> ExperimentUsage:
        if tail_ref is None:
            return ExperimentUsage(
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                tail_ref=None,
                entry_refs=(),
                claim_refs=(),
                candidate_refs=(),
                evaluation_refs=(),
                query_count=0,
                total_evaluations=0,
                total_tokens=0,
                total_tool_calls=0,
                total_wall_time_seconds=0.0,
                total_cost_usd=0.0,
                max_evaluations=self._max_evaluations,
                max_tokens=self._budget_limits.max_tokens,
                max_tool_calls=self._budget_limits.max_tool_calls,
                max_wall_time_seconds=self._budget_limits.max_wall_time_seconds,
                max_cost_usd=self._budget_limits.max_cost_usd,
                remaining_evaluations=self._max_evaluations,
                remaining_tokens=self._budget_limits.max_tokens,
                remaining_tool_calls=self._budget_limits.max_tool_calls,
                remaining_wall_time_seconds=self._budget_limits.max_wall_time_seconds,
                remaining_cost_usd=self._budget_limits.max_cost_usd,
            )

        cursor: ArtifactRef | None = ArtifactRef.model_validate(tail_ref)
        backwards: list[tuple[ArtifactRef, ExperimentUsageEntry]] = []
        seen_entries: set[str] = set()
        while cursor is not None:
            if cursor.sha256 in seen_entries:
                raise ExperimentControllerError("experiment usage ledger contains a cycle")
            seen_entries.add(cursor.sha256)
            entry = self._load(
                cursor,
                ExperimentUsageEntry,
                "experiment usage entry",
                expected_media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
            )
            backwards.append((cursor, entry))
            cursor = entry.previous_entry_ref
        entries = tuple(reversed(backwards))

        claim_refs: list[ArtifactRef] = []
        candidate_refs: list[ArtifactRef] = []
        evaluation_refs: list[ArtifactRef] = []
        seen_candidates: set[str] = set()
        seen_evaluations: set[str] = set()
        seen_batches: set[str] = set()
        cumulative = 0
        cumulative_tokens = 0
        cumulative_tool_calls = 0
        cumulative_wall_time = 0.0
        cumulative_cost: float | None = 0.0
        previous_entry_ref: ArtifactRef | None = None
        for sequence, (entry_ref, entry) in enumerate(entries):
            if entry.sequence != sequence:
                raise ExperimentControllerError("experiment usage sequence is not contiguous")
            if entry.previous_entry_ref != previous_entry_ref:
                raise ExperimentControllerError("experiment usage link does not match prior entry")
            if entry.experiment_ref != self.experiment_ref:
                raise ExperimentControllerError("usage entry belongs to another experiment")
            if entry.protocol_ref != self.protocol_ref:
                raise ExperimentControllerError("usage entry belongs to another protocol")
            claim = self._load(
                entry.claim_ref,
                ExperimentUsageClaim,
                "experiment usage claim",
                expected_media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
            )
            if claim.experiment_ref != self.experiment_ref:
                raise ExperimentControllerError("usage claim belongs to another experiment")
            if claim.protocol_ref != self.protocol_ref:
                raise ExperimentControllerError("usage claim belongs to another protocol")
            if claim.candidate_ref.sha256 in seen_candidates:
                raise ExperimentControllerError("candidate was charged more than once")
            if claim.evaluation_ref.sha256 in seen_evaluations:
                raise ExperimentControllerError("gate evaluation was charged more than once")
            batch_hashes = {claim.parent_batch_ref.sha256, claim.candidate_batch_ref.sha256}
            if seen_batches.intersection(batch_hashes):
                raise ExperimentControllerError("gate trial batch was charged more than once")
            if len(batch_hashes) != 2:
                raise ExperimentControllerError("parent and candidate batch refs must differ")

            source_events = self._journal.replay(claim.running_gate_tail_ref)
            candidate, admission_ref, mechanism_ref = self._verify_running_gate_history(
                claim.running_gate_tail_ref
            )
            if claim.candidate_ref != source_events[0].candidate_ref:
                raise ExperimentControllerError(
                    "usage claim source tail belongs to another candidate"
                )
            evaluation, parent_batch, candidate_batch, recomputed_units = (
                self._verify_gate_evaluation(
                    candidate_ref=claim.candidate_ref,
                    candidate=candidate,
                    admission_report_ref=admission_ref,
                    mechanism_evidence_ref=mechanism_ref,
                    evaluation_ref=claim.evaluation_ref,
                )
            )
            if evaluation.parent_batch_ref != claim.parent_batch_ref:
                raise ExperimentControllerError(
                    "usage claim parent batch does not match evaluation"
                )
            if evaluation.candidate_batch_ref != claim.candidate_batch_ref:
                raise ExperimentControllerError(
                    "usage claim candidate batch does not match evaluation"
                )
            if claim.evaluation_units != recomputed_units:
                raise ExperimentControllerError(
                    "usage claim evaluation count does not match persisted batches"
                )
            tokens, tool_calls, wall_time_seconds, cost_usd = self._resource_charge(
                parent_batch,
                candidate_batch,
            )
            if (
                claim.tokens != tokens
                or claim.tool_calls != tool_calls
                or claim.wall_time_seconds != wall_time_seconds
                or claim.cost_usd != cost_usd
            ):
                raise ExperimentControllerError(
                    "usage claim resources do not match persisted trial batches"
                )
            cumulative += recomputed_units
            cumulative_tokens += tokens
            cumulative_tool_calls += tool_calls
            cumulative_wall_time += wall_time_seconds
            cumulative_cost = (
                None if cumulative_cost is None or cost_usd is None else cumulative_cost + cost_usd
            )
            self._enforce_budget(
                evaluations=cumulative,
                tokens=cumulative_tokens,
                tool_calls=cumulative_tool_calls,
                wall_time_seconds=cumulative_wall_time,
                cost_usd=cumulative_cost,
                requested_evaluations=recomputed_units,
                prior_evaluations=cumulative - recomputed_units,
            )
            cumulative_fields = (
                ("evaluations", entry.cumulative_evaluations, cumulative),
                ("tokens", entry.cumulative_tokens, cumulative_tokens),
                ("tool calls", entry.cumulative_tool_calls, cumulative_tool_calls),
                ("wall time", entry.cumulative_wall_time_seconds, cumulative_wall_time),
                ("cost", entry.cumulative_cost_usd, cumulative_cost),
            )
            for label, persisted, recomputed in cumulative_fields:
                if persisted != recomputed:
                    raise ExperimentControllerError(
                        f"usage entry cumulative {label} is inconsistent"
                    )

            claim_refs.append(entry.claim_ref)
            candidate_refs.append(claim.candidate_ref)
            evaluation_refs.append(claim.evaluation_ref)
            seen_candidates.add(claim.candidate_ref.sha256)
            seen_evaluations.add(claim.evaluation_ref.sha256)
            seen_batches.update(batch_hashes)
            previous_entry_ref = entry_ref

        return ExperimentUsage(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            tail_ref=ArtifactRef.model_validate(tail_ref),
            entry_refs=tuple(entry_ref for entry_ref, _ in entries),
            claim_refs=tuple(claim_refs),
            candidate_refs=tuple(candidate_refs),
            evaluation_refs=tuple(evaluation_refs),
            query_count=len(entries),
            total_evaluations=cumulative,
            total_tokens=cumulative_tokens,
            total_tool_calls=cumulative_tool_calls,
            total_wall_time_seconds=cumulative_wall_time,
            total_cost_usd=cumulative_cost,
            max_evaluations=self._max_evaluations,
            max_tokens=self._budget_limits.max_tokens,
            max_tool_calls=self._budget_limits.max_tool_calls,
            max_wall_time_seconds=self._budget_limits.max_wall_time_seconds,
            max_cost_usd=self._budget_limits.max_cost_usd,
            remaining_evaluations=self._max_evaluations - cumulative,
            remaining_tokens=(
                None
                if self._budget_limits.max_tokens is None
                else self._budget_limits.max_tokens - cumulative_tokens
            ),
            remaining_tool_calls=(
                None
                if self._budget_limits.max_tool_calls is None
                else self._budget_limits.max_tool_calls - cumulative_tool_calls
            ),
            remaining_wall_time_seconds=(
                None
                if self._budget_limits.max_wall_time_seconds is None
                else self._budget_limits.max_wall_time_seconds - cumulative_wall_time
            ),
            remaining_cost_usd=(
                None
                if self._budget_limits.max_cost_usd is None or cumulative_cost is None
                else self._budget_limits.max_cost_usd - cumulative_cost
            ),
        )

    def _reject_duplicate_usage(
        self,
        usage: ExperimentUsage,
        *,
        candidate_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
        parent_batch_ref: ArtifactRef,
        candidate_batch_ref: ArtifactRef,
    ) -> None:
        if any(ref.sha256 == candidate_ref.sha256 for ref in usage.candidate_refs):
            raise ExperimentControllerError("candidate already consumed a gate query")
        if any(ref.sha256 == evaluation_ref.sha256 for ref in usage.evaluation_refs):
            raise ExperimentControllerError("gate evaluation was already charged")
        requested_batches = {parent_batch_ref.sha256, candidate_batch_ref.sha256}
        for claim_ref in usage.claim_refs:
            claim = self._load(
                claim_ref,
                ExperimentUsageClaim,
                "experiment usage claim",
                expected_media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
            )
            existing_batches = {
                claim.parent_batch_ref.sha256,
                claim.candidate_batch_ref.sha256,
            }
            if requested_batches.intersection(existing_batches):
                raise ExperimentControllerError("gate trial batch was already charged")

    @staticmethod
    def _evidence_complete_refs(
        evidence_refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ArtifactRef]:
        evaluations = tuple(
            ref for ref in evidence_refs if ref.media_type == GATE_EVALUATION_MANIFEST_MEDIA_TYPE
        )
        usage_entries = tuple(
            ref for ref in evidence_refs if ref.media_type == EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE
        )
        if len(evidence_refs) != 2 or len(evaluations) != 1 or len(usage_entries) != 1:
            raise ExperimentControllerError(
                "evidence-complete event must bind exactly one evaluation and usage entry"
            )
        return evaluations[0], usage_entries[0]

    def _load[ModelT](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        label: str,
        *,
        expected_media_type: str | None = None,
    ) -> ModelT:
        ref = ArtifactRef.model_validate(ref)
        if expected_media_type is not None and ref.media_type != expected_media_type:
            raise ExperimentControllerError(f"{label} declares the wrong media type")
        try:
            payload = self._repository.get_bytes(ref)
            loaded = self._repository.get_json(ref, model_type)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise ExperimentControllerError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise ExperimentControllerError(
                f"could not load canonical {label}: typed representation is not canonical"
            )
        return loaded

    def _load_json(self, ref: ArtifactRef, label: str) -> object:
        ref = ArtifactRef.model_validate(ref)
        try:
            payload = self._repository.get_bytes(ref)
            loaded = self._repository.get_json(ref)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise ExperimentControllerError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise ExperimentControllerError(
                f"could not load canonical {label}: JSON representation is not canonical"
            )
        return loaded

    @staticmethod
    def _stream_id(candidate_ref: ArtifactRef) -> str:
        return f"candidate/{candidate_ref.sha256}"


__all__ = [
    "ADMISSION_FAILURE_REPORT_MEDIA_TYPE",
    "EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE",
    "EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE",
    "PROBE_REJECTION_REPORT_MEDIA_TYPE",
    "SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE",
    "TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE",
    "AdmissionFailureCode",
    "AdmissionFailureReport",
    "CandidateSupersessionCode",
    "EvidenceCompletion",
    "ExperimentBudgetError",
    "ExperimentController",
    "ExperimentControllerError",
    "ExperimentUsage",
    "ExperimentUsageClaim",
    "ExperimentUsageEntry",
    "ProbeRejectionCode",
    "ProbeRejectionReport",
    "StaleControllerTailError",
    "SupersededCandidateReport",
    "TerminalCompletion",
    "TerminalTransitionAuthorization",
]
