"""Trusted in-process single-writer controller for one frozen experiment."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

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
    CandidateMutation,
    ComponentKind,
)
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.execution.model import ModelBackend
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
    REQUIRED_SKILL_MECHANISM_IDS,
    RESERVED_SKILL_MECHANISM_IDS,
    AttestedMechanismEvidence,
    MechanismEvidenceVerificationCapability,
)
from spiral_harness.verification.models import GateConfig, MechanismEvidence
from spiral_harness.verification.skill_plan import SkillMechanismPlan

from .admission import CandidateAdmissionError, CandidateAdmissionService
from .controller_artifacts import (
    ADMISSION_FAILURE_REPORT_MEDIA_TYPE,
    EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES,
    EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE,
    PROBE_REJECTION_REPORT_MEDIA_TYPE,
    SUPERSEDED_CANDIDATE_REPORT_MEDIA_TYPE,
    TERMINAL_TRANSITION_AUTHORIZATION_MEDIA_TYPE,
    AdmissionFailureCode,
    AdmissionFailureReport,
    EvidenceCompletion,
    ExperimentUsage,
    ExperimentUsageClaim,
    ExperimentUsageEntry,
    ExperimentUsageEntryV1,
    ProbeRejectionCode,
    ProbeRejectionReport,
    SkillProbeSettlementKind,
    SupersededCandidateReport,
    TerminalCompletion,
    TerminalTransitionAuthorization,
)
from .decision import (
    GATE_EVALUATION_MANIFEST_MEDIA_TYPE,
    TERMINAL_DECISION_REPORT_MEDIA_TYPE,
    GateEvaluationManifest,
    TerminalDecisionError,
    TerminalDecisionReport,
    TerminalDecisionService,
)
from .experiment_usage import (
    ExperimentUsageBudgetError,
    ExperimentUsageLedger,
    ExperimentUsageLedgerError,
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
from .skill_probe_authorization import (
    SkillProbeExecutionAuthorization,
    SkillProbeExecutionAuthorizationError,
    _create_trusted_skill_probe_execution_authorization_service,
    probe_plan_ref_from_history,
    verify_skill_probe_execution_authorization_history,
)
from .skill_probe_closure import VerifiedMatchedSkillProbeResult
from .skill_probe_execution import (
    MatchedSkillProbeExecution,
    _execute_matched_skill_probes,
    _verify_matched_skill_probe_result,
)
from .skill_probes import (
    SkillProbePreregistrationError,
    replay_probe_preregistration_refs,
    resolve_probe_preregistration,
)


class ExperimentControllerError(RuntimeError):
    """Raised when a requested semantic transition cannot be authorized."""


class ExperimentBudgetError(ExperimentControllerError):
    """Raised when persisted evaluation use would exceed a frozen ceiling."""


class StaleControllerTailError(ExperimentControllerError):
    """Raised when a caller tries to append from an old or foreign branch."""


class ExperimentController:
    """Single process-local writer for a frozen experiment and its checked budgets."""

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
        # Exact types keep subclasses from overriding attestor identity or verify().
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
        self._state_lock = RLock()
        self._usage_accounting_blocked = False
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
        self._usage_ledger = ExperimentUsageLedger(
            repository,
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            budget_limits=self._budget_limits,
            verify_gate_history=self._verify_running_gate_history,
            verify_gate_evaluation=self._verify_gate_evaluation,
        )
        self._skill_probe_authorizations = (
            _create_trusted_skill_probe_execution_authorization_service(
                repository,
                verify_current=self._verify_current_skill_probe_execution_authorization,
            )
        )
        self._candidate_tails: dict[str, ArtifactRef] = {}
        self._candidate_refs: dict[str, ArtifactRef] = {}
        self._champion_harness_ref = self._experiment.seed_harness_ref
        self._champion_candidate_ref: ArtifactRef | None = None
        self._usage_tail_ref = (
            None if usage_tail_ref is None else ArtifactRef.model_validate(usage_tail_ref)
        )
        # A supplied restart head is not trusted merely because it exists.
        # A replayed poison is terminal even for a fresh controller object: a
        # new process must not turn an immutable poisoned usage head back into
        # a writable experiment merely because its local flag started false.
        replayed_usage = self._replay_usage(self._usage_tail_ref)
        self._usage_accounting_blocked = (
            replayed_usage.poisoned or not replayed_usage.accounting_complete
        )
        self._experiment_tail_ref: ArtifactRef | None = None
        self._candidate_resume_blocked = usage_tail_ref is not None

    @property
    def usage_tail_ref(self) -> ArtifactRef | None:
        """Return the current caller-held usage head for this writer."""

        with self._state_lock:
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

        with self._state_lock:
            self._require_accounting_open()
            return self._close_selection(
                previous_tail_ref=previous_tail_ref,
                previous_usage_tail_ref=previous_usage_tail_ref,
                champion_candidate_ref=champion_candidate_ref,
                champion_candidate_tail_ref=champion_candidate_tail_ref,
                champion_harness_ref=champion_harness_ref,
                analysis_plan_ref=analysis_plan_ref,
            )

    def _close_selection(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        previous_usage_tail_ref: ArtifactRef | None,
        champion_candidate_ref: ArtifactRef | None,
        champion_candidate_tail_ref: ArtifactRef | None,
        champion_harness_ref: ArtifactRef,
        analysis_plan_ref: ArtifactRef,
    ) -> ArtifactRef:
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
        """Close selection from controller-owned champion and usage heads."""

        with self._state_lock:
            self._require_accounting_open()
            champion_candidate_ref = self._champion_candidate_ref
            champion_candidate_tail_ref = (
                None
                if champion_candidate_ref is None
                else self._candidate_tails[champion_candidate_ref.sha256]
            )
            return self._close_selection(
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
        """Replay and rejoin the current controller-owned selection closure."""

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
        skill_mechanism_plan_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Enter the preregistered mechanism-probe stage."""

        self._require_searching()
        self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.VALID,
        )
        try:
            evidence_refs = resolve_probe_preregistration(
                self._repository,
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=candidate_ref,
                plan_ref=skill_mechanism_plan_ref,
            )
        except SkillProbePreregistrationError as exc:
            raise ExperimentControllerError(f"probe preregistration failed: {exc}") from exc
        return self._append_candidate_event(
            candidate_ref=candidate_ref,
            previous_tail_ref=previous_tail_ref,
            from_state=CandidateState.VALID,
            to_state=CandidateState.RUNNING_PROBES,
            evidence_refs=evidence_refs,
            reason="candidate entered preregistered mechanism probes",
        )

    def start_gate(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Replay preregistration and resolve probes into a gate or rejection."""

        self._require_searching()
        events = self._require_current_tail(
            candidate_ref,
            previous_tail_ref,
            expected_state=CandidateState.RUNNING_PROBES,
        )
        try:
            replay_probe_preregistration_refs(
                self._repository,
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=candidate_ref,
                evidence_refs=events[-1].evidence_refs,
            )
        except SkillProbePreregistrationError as exc:
            raise ExperimentControllerError(f"probe history replay failed: {exc}") from exc
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

    def issue_skill_probe_execution_authorization(
        self,
        *,
        candidate_ref: ArtifactRef,
        running_probes_tail_ref: ArtifactRef,
    ) -> ArtifactRef:
        """Issue one idempotent execution grant for the exact current probe head."""

        self._require_searching()
        events = self._require_current_tail(
            candidate_ref,
            running_probes_tail_ref,
            expected_state=CandidateState.RUNNING_PROBES,
        )
        try:
            plan_ref = probe_plan_ref_from_history(events)
            return self._skill_probe_authorizations.issue(
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=ArtifactRef.model_validate(candidate_ref, strict=True),
                plan_ref=plan_ref,
                running_probes_tail_ref=ArtifactRef.model_validate(
                    running_probes_tail_ref,
                    strict=True,
                ),
            )
        except SkillProbeExecutionAuthorizationError as exc:
            raise ExperimentControllerError(
                f"skill-probe execution authorization failed: {exc}"
            ) from exc

    def execute_matched_skill_probes(
        self,
        *,
        authorization_ref: ArtifactRef,
        model_spec: FrozenModelSpec,
        revert_backend: ModelBackend,
        placebo_backend: ModelBackend,
    ) -> MatchedSkillProbeExecution:
        """Reserve global budget, then execute one issued matched probe."""

        with self._state_lock:
            self._require_accounting_open()
            return _execute_matched_skill_probes(
                self._repository,
                authorization_ref=authorization_ref,
                execution_authority=self._skill_probe_authorizations.execution_authority,
                authorization_capability=self._skill_probe_authorizations.verification_capability,
                model_spec=model_spec,
                revert_backend=revert_backend,
                placebo_backend=placebo_backend,
                reserve_usage=self._reserve_skill_probe_usage,
                settle_usage=self._settle_skill_probe_usage,
            )

    def verify_matched_skill_probe_result(
        self,
        execution: MatchedSkillProbeExecution,
    ) -> VerifiedMatchedSkillProbeResult:
        """Replay one result with this controller's original live ledgers."""

        if type(execution) is not MatchedSkillProbeExecution:
            raise ExperimentControllerError(
                "skill-probe execution must be the controller result object"
            )
        try:
            return _verify_matched_skill_probe_result(
                self._repository,
                result=execution.result,
                authorization_capability=(self._skill_probe_authorizations.verification_capability),
                revert_attempt_ledger=execution.revert_attempt_ledger,
                placebo_attempt_ledger=execution.placebo_attempt_ledger,
            )
        except Exception as exc:
            raise ExperimentControllerError(
                f"matched skill-probe result verification failed: {exc}"
            ) from exc

    def _reserve_skill_probe_usage(
        self,
        authorization_ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
        plan: SkillMechanismPlan,
    ) -> None:
        """Charge the two schedules conservatively before consuming the grant."""
        with self._state_lock:
            self._require_accounting_open()
            try:
                self._usage_tail_ref = self._usage_ledger.reserve_skill_probe(
                    tail_ref=self._usage_tail_ref,
                    authorization_ref=authorization_ref,
                    authorization=authorization,
                    plan=plan,
                )
            except ExperimentUsageBudgetError as exc:
                raise ExperimentBudgetError(str(exc)) from exc
            except ExperimentUsageLedgerError as exc:
                raise ExperimentControllerError(str(exc)) from exc

    def _settle_skill_probe_usage(
        self,
        authorization_ref: ArtifactRef,
        revert_preflight_ref: ArtifactRef,
        revert_terminal_tail_ref: ArtifactRef | None,
        placebo_preflight_ref: ArtifactRef,
        placebo_terminal_tail_ref: ArtifactRef | None,
        terminal_kind: SkillProbeSettlementKind,
        closure_ref: ArtifactRef | None,
    ) -> None:
        """Publish the terminal probe charge or permanently close accounting."""

        with self._state_lock:
            self._require_accounting_open()
            try:
                usage_tail_ref = self._usage_ledger.settle_skill_probe(
                    tail_ref=self._usage_tail_ref,
                    authorization_ref=authorization_ref,
                    revert_preflight_ref=revert_preflight_ref,
                    revert_terminal_tail_ref=revert_terminal_tail_ref,
                    placebo_preflight_ref=placebo_preflight_ref,
                    placebo_terminal_tail_ref=placebo_terminal_tail_ref,
                    terminal_kind=terminal_kind,
                    closure_ref=closure_ref,
                )
            except ExperimentUsageBudgetError as exc:
                self._usage_accounting_blocked = True
                raise ExperimentBudgetError(str(exc)) from exc
            except ExperimentUsageLedgerError as exc:
                self._usage_accounting_blocked = True
                raise ExperimentControllerError(str(exc)) from exc
            except Exception as exc:
                self._usage_accounting_blocked = True
                raise ExperimentControllerError(
                    f"skill-probe usage settlement publication failed: {exc}"
                ) from exc
            self._usage_tail_ref = usage_tail_ref
            settled_usage = self._replay_usage(usage_tail_ref)
            if settled_usage.poisoned or not settled_usage.accounting_complete:
                self._usage_accounting_blocked = True

    def complete_evidence(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
        previous_usage_tail_ref: ArtifactRef | None,
    ) -> EvidenceCompletion:
        """Account a gate query and publish its evidence-complete lifecycle head."""

        with self._state_lock:
            return self._complete_evidence(
                candidate_ref=candidate_ref,
                previous_tail_ref=previous_tail_ref,
                evaluation_ref=evaluation_ref,
                previous_usage_tail_ref=previous_usage_tail_ref,
            )

    def _complete_evidence(
        self,
        *,
        candidate_ref: ArtifactRef,
        previous_tail_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
        previous_usage_tail_ref: ArtifactRef | None,
    ) -> EvidenceCompletion:
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
        tokens, tool_calls, wall_time_seconds, cost_usd = self._usage_ledger.resource_charge(
            parent_batch,
            candidate_batch,
        )
        cumulative = usage.total_evaluations + evaluation_units
        cumulative_tokens = usage.total_tokens + tokens
        cumulative_tool_calls = usage.total_tool_calls + tool_calls
        cumulative_wall_time = (
            None
            if usage.total_wall_time_seconds is None
            else usage.total_wall_time_seconds + wall_time_seconds
        )
        cumulative_cost = (
            None
            if usage.total_cost_usd is None or cost_usd is None
            else usage.total_cost_usd + cost_usd
        )
        try:
            self._usage_ledger.enforce_budget(
                evaluations=cumulative,
                tokens=cumulative_tokens,
                tool_calls=cumulative_tool_calls,
                wall_time_seconds=cumulative_wall_time or 0.0,
                cost_usd=cumulative_cost,
                requested_evaluations=evaluation_units,
                prior_evaluations=usage.total_evaluations,
            )
        except ExperimentUsageBudgetError as exc:
            raise ExperimentBudgetError(str(exc)) from exc

        # Publish the mutable head only after every immutable object exists.
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
            sequence=usage.entry_count,
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
        usage_entry = self._load_usage_entry(
            usage_entry_ref,
            "experiment usage entry",
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
        usage_entry = self._load_usage_entry(
            authorization.usage_entry_ref,
            "terminal authorization usage entry",
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

        with self._state_lock:
            return self._replay_usage(self._usage_tail_ref)

    def query_usage(self, tail_ref: ArtifactRef | None) -> ExperimentUsage:
        """Replay any explicit historical head from this frozen experiment."""

        with self._state_lock:
            return self._replay_usage(tail_ref)

    def _require_accounting_open(self) -> None:
        if self._usage_accounting_blocked:
            raise ExperimentControllerError(
                "experiment usage accounting is blocked after a settlement failure, "
                "incomplete accounting, or a poisoned usage entry"
            )
        usage = self._replay_usage(self._usage_tail_ref)
        if usage.poisoned or not usage.accounting_complete:
            self._usage_accounting_blocked = True
            raise ExperimentControllerError(
                "experiment usage accounting is blocked after a settlement failure, "
                "incomplete accounting, or a poisoned usage entry"
            )

    def _require_searching(self) -> None:
        self._require_accounting_open()
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

    def _verify_current_skill_probe_execution_authorization(
        self,
        authorization: SkillProbeExecutionAuthorization,
    ) -> None:
        if authorization.experiment_ref != self.experiment_ref:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe authorization belongs to another experiment"
            )
        if authorization.protocol_ref != self.protocol_ref:
            raise SkillProbeExecutionAuthorizationError(
                "skill-probe authorization belongs to another protocol"
            )
        try:
            self._require_searching()
            events = self._require_current_tail(
                authorization.candidate_ref,
                authorization.running_probes_tail_ref,
                expected_state=CandidateState.RUNNING_PROBES,
            )
        except ExperimentControllerError as exc:
            raise SkillProbeExecutionAuthorizationError(
                f"skill-probe authorization does not bind the current controller head: {exc}"
            ) from exc
        verify_skill_probe_execution_authorization_history(
            self._repository,
            authorization=authorization,
            events=events,
        )

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
        if events[0].evidence_refs:
            raise ExperimentControllerError("registered lifecycle stage contains evidence")
        if len(events[1].evidence_refs) != 1 or len(events[3].evidence_refs) != 1:
            raise ExperimentControllerError("admission and mechanism stages require exact evidence")

        candidate_ref = events[0].candidate_ref
        candidate = self._load(candidate_ref, CandidateManifest, "candidate manifest")
        if candidate.experiment_ref != self.experiment_ref:
            raise ExperimentControllerError("candidate history belongs to another experiment")
        try:
            replay_probe_preregistration_refs(
                self._repository,
                experiment_ref=self.experiment_ref,
                protocol_ref=self.protocol_ref,
                candidate_ref=candidate_ref,
                evidence_refs=events[2].evidence_refs,
            )
        except SkillProbePreregistrationError as exc:
            raise ExperimentControllerError(f"probe history replay failed: {exc}") from exc
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
        mutation = self._load(candidate.mutation_ref, CandidateMutation, "candidate mutation")
        is_skill_mutation = mutation.after.kind is ComponentKind.SKILL
        if is_skill_mutation:
            skill_required = sorted(REQUIRED_SKILL_MECHANISM_IDS)
            required = tuple(dict.fromkeys((*required, *skill_required)))
        by_name = {
            check.name: check
            for check in evidence.checks
            if not (
                is_skill_mutation and check.name.strip().casefold() in RESERVED_SKILL_MECHANISM_IDS
            )
        }
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
        try:
            return self._usage_ledger.replay(tail_ref)
        except ExperimentUsageBudgetError as exc:
            raise ExperimentBudgetError(str(exc)) from exc
        except ExperimentUsageLedgerError as exc:
            raise ExperimentControllerError(str(exc)) from exc

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
            ref for ref in evidence_refs if ref.media_type in EXPERIMENT_USAGE_ENTRY_MEDIA_TYPES
        )
        if len(evidence_refs) != 2 or len(evaluations) != 1 or len(usage_entries) != 1:
            raise ExperimentControllerError(
                "evidence-complete event must bind exactly one evaluation and usage entry"
            )
        return evaluations[0], usage_entries[0]

    def _load_usage_entry(
        self,
        ref: ArtifactRef,
        label: str,
    ) -> ExperimentUsageEntryV1 | ExperimentUsageEntry:
        """Load one frozen usage-entry version from its declared media type."""

        checked_ref = ArtifactRef.model_validate(ref)
        if checked_ref.media_type == EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE:
            return self._load(
                checked_ref,
                ExperimentUsageEntryV1,
                label,
                expected_media_type=EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE,
            )
        if checked_ref.media_type == EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE:
            return self._load(
                checked_ref,
                ExperimentUsageEntry,
                label,
                expected_media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
            )
        raise ExperimentControllerError(f"{label} declares an unsupported version")

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
    "ExperimentBudgetError",
    "ExperimentController",
    "ExperimentControllerError",
    "StaleControllerTailError",
]
