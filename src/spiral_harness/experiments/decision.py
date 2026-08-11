"""Trusted experiment validation for candidate terminal decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import (
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import ArtifactRef, ImmutableModel
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.gate import PromotionGate
from spiral_harness.verification.models import (
    Decision,
    GateConfig,
    GateDecision,
    MechanismEvidence,
    TrialObservation,
)

from .admission import CandidateAdmissionError, CandidateAdmissionService

GATE_EVALUATION_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.gate-evaluation-manifest.v1+json"
)
TERMINAL_DECISION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.terminal-decision-report.v1+json"
)

_DECISION_STATES = {
    Decision.PROMOTE: CandidateState.PROMOTED,
    Decision.REJECT: CandidateState.REJECTED,
    Decision.INCONCLUSIVE: CandidateState.INCONCLUSIVE,
}
_GATE_TERMINAL_STATES = frozenset(_DECISION_STATES.values())

_REPORT_CHECKS = (
    "evaluation_manifest_bound",
    "admission_report_verified",
    "candidate_plan_matches_protocol_gate",
    "gate_split_bound",
    "gate_inputs_canonical_verified",
    "gate_decision_recomputed",
    "terminal_state_verified",
)


class TerminalDecisionError(ValueError):
    """Raised when a gate decision cannot authorize a terminal transition."""


class GateEvaluationManifest(ImmutableModel):
    """Immutable closure of every input consumed by one gate evaluation."""

    schema_version: Literal["1"] = "1"
    candidate_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    parent_trials_ref: ArtifactRef
    candidate_trials_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef

    @model_validator(mode="after")
    def inputs_are_json_artifacts(self) -> GateEvaluationManifest:
        for field_name in (
            "candidate_ref",
            "admission_report_ref",
            "gate_config_ref",
            "gate_split_ref",
            "parent_trials_ref",
            "candidate_trials_ref",
            "mechanism_evidence_ref",
        ):
            ref = getattr(self, field_name)
            media_type = ref.media_type.partition(";")[0].strip().lower()
            if media_type != "application/json" and not media_type.endswith("+json"):
                raise ValueError(f"{field_name} must declare a JSON media type")
        return self


class TerminalDecisionReport(ImmutableModel):
    """Replayable proof of every semantic join used for a terminal transition."""

    schema_version: Literal["1"] = "1"
    candidate_ref: ArtifactRef
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    evaluation_ref: ArtifactRef
    decision_ref: ArtifactRef
    terminal_state: CandidateState
    decision: Decision
    checks: tuple[
        Literal[
            "evaluation_manifest_bound",
            "admission_report_verified",
            "candidate_plan_matches_protocol_gate",
            "gate_split_bound",
            "gate_inputs_canonical_verified",
            "gate_decision_recomputed",
            "terminal_state_verified",
        ],
        ...,
    ] = _REPORT_CHECKS


class TerminalDecisionService:
    """Rejoin frozen candidate lineage with a persisted gate decision.

    The linked journal intentionally validates only append structure.  This
    service is the trusted semantic boundary that must run before a terminal
    lifecycle event is appended.
    """

    def __init__(self, repository: ArtifactRepository) -> None:
        self._repository = repository

    def validate(
        self,
        *,
        candidate_ref: ArtifactRef,
        experiment_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
        decision_ref: ArtifactRef,
        terminal_state: CandidateState,
    ) -> TerminalDecisionReport:
        """Recompute a decision from referenced evidence and authorize its terminal state."""

        if (
            not isinstance(terminal_state, CandidateState)
            or terminal_state not in _GATE_TERMINAL_STATES
        ):
            raise TerminalDecisionError(
                "terminal decision requires promoted, rejected, or inconclusive state"
            )
        if evaluation_ref.media_type != GATE_EVALUATION_MANIFEST_MEDIA_TYPE:
            raise TerminalDecisionError("gate evaluation manifest declares the wrong media type")

        evaluation = self._load(
            evaluation_ref,
            GateEvaluationManifest,
            "gate evaluation manifest",
        )
        if evaluation.candidate_ref != candidate_ref:
            raise TerminalDecisionError("gate evaluation belongs to another candidate")

        candidate = self._load(candidate_ref, CandidateManifest, "candidate")
        if candidate.experiment_ref != experiment_ref:
            raise TerminalDecisionError(
                "candidate experiment does not match the caller-frozen experiment"
            )
        try:
            CandidateAdmissionService(self._repository).verify_report(
                candidate_ref=candidate_ref,
                experiment_ref=experiment_ref,
                report_ref=evaluation.admission_report_ref,
            )
        except CandidateAdmissionError as exc:
            raise TerminalDecisionError(
                f"candidate admission report could not be verified: {exc}"
            ) from exc

        experiment = self._load(experiment_ref, ExperimentManifest, "experiment")
        protocol = self._load(experiment.protocol_ref, ProtocolManifest, "protocol")
        gate_split_ref = next(
            split.manifest_ref
            for split in protocol.splits
            if split.partition is ProtocolPartition.GATE
        )
        if evaluation.gate_split_ref != gate_split_ref:
            raise TerminalDecisionError(
                "gate evaluation split does not match the protocol gate split"
            )
        self._load_json(gate_split_ref, "gate split")

        if candidate.evaluation_plan_ref != protocol.gate_config_ref:
            raise TerminalDecisionError(
                "candidate evaluation plan does not match the protocol gate config"
            )
        if evaluation.gate_config_ref != protocol.gate_config_ref:
            raise TerminalDecisionError(
                "gate evaluation config does not match the protocol gate config"
            )

        gate_config = self._load(evaluation.gate_config_ref, GateConfig, "gate config")
        parent_trials = self._load(
            evaluation.parent_trials_ref,
            tuple[TrialObservation, ...],
            "parent gate trials",
        )
        candidate_trials = self._load(
            evaluation.candidate_trials_ref,
            tuple[TrialObservation, ...],
            "candidate gate trials",
        )
        mechanism_evidence = self._load(
            evaluation.mechanism_evidence_ref,
            MechanismEvidence,
            "mechanism evidence",
        )
        recomputed = PromotionGate(gate_config).evaluate(
            parent_trials,
            candidate_trials,
            mechanism_evidence,
            parent_harness_id=candidate.parent_harness_ref.sha256,
            candidate_harness_id=candidate.child_harness_ref.sha256,
        )
        persisted = self._load(decision_ref, GateDecision, "gate decision")
        if persisted != recomputed:
            raise TerminalDecisionError(
                "persisted gate decision does not exactly match trusted recomputation"
            )

        expected_state = _DECISION_STATES[recomputed.decision]
        if terminal_state is not expected_state:
            raise TerminalDecisionError(
                f"decision {recomputed.decision.value!r} requires terminal state "
                f"{expected_state.value!r}, got {terminal_state.value!r}"
            )

        return TerminalDecisionReport(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
            protocol_ref=experiment.protocol_ref,
            admission_report_ref=evaluation.admission_report_ref,
            gate_config_ref=protocol.gate_config_ref,
            gate_split_ref=gate_split_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=decision_ref,
            terminal_state=terminal_state,
            decision=recomputed.decision,
        )

    def verify_report(
        self,
        report_ref: ArtifactRef,
        *,
        candidate_ref: ArtifactRef,
        experiment_ref: ArtifactRef,
        evaluation_ref: ArtifactRef,
    ) -> TerminalDecisionReport:
        """Reload a report and reproduce every claimed semantic check."""

        if report_ref.media_type != TERMINAL_DECISION_REPORT_MEDIA_TYPE:
            raise TerminalDecisionError("terminal decision report declares the wrong media type")
        report = self._load(report_ref, TerminalDecisionReport, "terminal decision report")
        if report.candidate_ref != candidate_ref:
            raise TerminalDecisionError("terminal decision report belongs to another candidate")
        if report.experiment_ref != experiment_ref:
            raise TerminalDecisionError("terminal decision report belongs to another experiment")
        if report.evaluation_ref != evaluation_ref:
            raise TerminalDecisionError("terminal decision report belongs to another evaluation")
        expected = self.validate(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
            evaluation_ref=evaluation_ref,
            decision_ref=report.decision_ref,
            terminal_state=report.terminal_state,
        )
        if report != expected:
            raise TerminalDecisionError("terminal decision report does not match replayed checks")
        return report

    def _load[ModelT](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        label: str,
    ) -> ModelT:
        try:
            payload = self._repository.get_bytes(ref)
            loaded = self._repository.get_json(ref, model_type)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise TerminalDecisionError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise TerminalDecisionError(
                f"could not load canonical {label}: typed representation is not canonical"
            )
        return loaded

    def _load_json(self, ref: ArtifactRef, label: str) -> object:
        """Load an untyped JSON manifest without trusting backend normalization."""

        try:
            payload = self._repository.get_bytes(ref)
            loaded = self._repository.get_json(ref)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise TerminalDecisionError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise TerminalDecisionError(
                f"could not load canonical {label}: JSON representation is not canonical"
            )
        return loaded


__all__ = [
    "GATE_EVALUATION_MANIFEST_MEDIA_TYPE",
    "TERMINAL_DECISION_REPORT_MEDIA_TYPE",
    "GateEvaluationManifest",
    "TerminalDecisionError",
    "TerminalDecisionReport",
    "TerminalDecisionService",
]
