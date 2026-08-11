"""Trusted experiment validation for candidate terminal decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import (
    GATE_TRIAL_BATCH_MEDIA_TYPE,
    GateBatchExecutionContext,
    GateBatchVerificationCapability,
    GateTrialArm,
    GateTrialBatch,
)
from spiral_harness.verification.gate import PromotionGate
from spiral_harness.verification.mechanism import (
    ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
    AttestedMechanismEvidence,
    MechanismEvidenceVerificationCapability,
)
from spiral_harness.verification.models import (
    Decision,
    GateConfig,
    GateDecision,
    MechanismEvidence,
)

from .admission import CandidateAdmissionError, CandidateAdmissionService

GATE_EVALUATION_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.gate-evaluation-manifest.v2+json"
)
TERMINAL_DECISION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.terminal-decision-report.v2+json"
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
    "gate_implementation_bound",
    "gate_trial_batches_bound",
    "gate_batch_attestations_verified",
    "mechanism_evidence_attestation_verified",
    "gate_inputs_canonical_verified",
    "gate_decision_recomputed",
    "terminal_state_verified",
)


class TerminalDecisionError(ValueError):
    """Raised when a gate decision cannot authorize a terminal transition."""


class GateEvaluationManifest(ImmutableModel):
    """Immutable closure of every input consumed by one gate evaluation."""

    schema_version: Literal["2"] = "2"
    candidate_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    gate_implementation_fingerprint: NonEmptyStr = PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT
    parent_batch_ref: ArtifactRef
    candidate_batch_ref: ArtifactRef
    mechanism_evidence_ref: ArtifactRef

    @model_validator(mode="after")
    def inputs_are_json_artifacts(self) -> GateEvaluationManifest:
        for field_name in (
            "candidate_ref",
            "admission_report_ref",
            "gate_config_ref",
            "gate_split_ref",
        ):
            ref = getattr(self, field_name)
            media_type = ref.media_type.partition(";")[0].strip().lower()
            if media_type != "application/json" and not media_type.endswith("+json"):
                raise ValueError(f"{field_name} must declare a JSON media type")
        if self.mechanism_evidence_ref.media_type != ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE:
            raise ValueError(
                "mechanism_evidence_ref must declare the exact attested evidence media type"
            )
        for field_name in ("parent_batch_ref", "candidate_batch_ref"):
            if getattr(self, field_name).media_type != GATE_TRIAL_BATCH_MEDIA_TYPE:
                raise ValueError(f"{field_name} must declare the exact gate trial batch media type")
        return self


class TerminalDecisionReport(ImmutableModel):
    """Replayable proof of every semantic join used for a terminal transition."""

    schema_version: Literal["2"] = "2"
    candidate_ref: ArtifactRef
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    gate_implementation_fingerprint: NonEmptyStr
    gate_batch_attestor_id: Sha256
    mechanism_evidence_attestor_id: Sha256
    mechanism_evidence_ref: ArtifactRef
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
            "gate_implementation_bound",
            "gate_trial_batches_bound",
            "gate_batch_attestations_verified",
            "mechanism_evidence_attestation_verified",
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

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        gate_batch_verifier: GateBatchVerificationCapability,
        mechanism_evidence_verifier: MechanismEvidenceVerificationCapability,
    ) -> None:
        # Verification capabilities are deliberately closed to subclass
        # overrides: a forged subclass could otherwise self-report the
        # protocol's attestor ID and bypass the real HMAC verifier.
        if type(gate_batch_verifier) is not GateBatchVerificationCapability:
            raise TypeError("gate_batch_verifier must be a GateBatchVerificationCapability")
        if type(mechanism_evidence_verifier) is not MechanismEvidenceVerificationCapability:
            raise TypeError(
                "mechanism_evidence_verifier must be a MechanismEvidenceVerificationCapability"
            )
        self._repository = repository
        self._gate_batch_verifier = gate_batch_verifier
        self._mechanism_evidence_verifier = mechanism_evidence_verifier

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

        experiment = self._load_exact_media(
            experiment_ref,
            ExperimentManifest,
            "experiment",
            EXPERIMENT_MANIFEST_MEDIA_TYPE,
        )
        protocol = self._load_exact_media(
            experiment.protocol_ref,
            ProtocolManifest,
            "protocol",
            PROTOCOL_MANIFEST_MEDIA_TYPE,
        )
        if self._gate_batch_verifier.attestor_id != protocol.gate_batch_attestor_id:
            raise TerminalDecisionError(
                "gate batch verifier does not match the protocol-frozen attestor"
            )
        if self._mechanism_evidence_verifier.attestor_id != protocol.mechanism_evidence_attestor_id:
            raise TerminalDecisionError(
                "mechanism evidence verifier does not match the protocol-frozen attestor"
            )
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
        if protocol.gate_implementation_fingerprint != PROMOTION_GATE_IMPLEMENTATION_FINGERPRINT:
            raise TerminalDecisionError(
                "protocol gate implementation fingerprint is not supported by this verifier"
            )
        if evaluation.gate_implementation_fingerprint != protocol.gate_implementation_fingerprint:
            raise TerminalDecisionError(
                "gate evaluation implementation does not match the protocol implementation"
            )

        gate_config = self._load(evaluation.gate_config_ref, GateConfig, "gate config")
        parent_batch = self._load_exact_media(
            evaluation.parent_batch_ref,
            GateTrialBatch,
            "parent gate trial batch",
            GATE_TRIAL_BATCH_MEDIA_TYPE,
        )
        candidate_batch = self._load_exact_media(
            evaluation.candidate_batch_ref,
            GateTrialBatch,
            "candidate gate trial batch",
            GATE_TRIAL_BATCH_MEDIA_TYPE,
        )
        parent_batch = self._verify_gate_batch(
            parent_batch,
            expected_protocol_ref=experiment.protocol_ref,
            expected_protocol=protocol,
            expected_candidate_ref=candidate_ref,
            expected_arm=GateTrialArm.PARENT,
            expected_harness_ref=candidate.parent_harness_ref,
            expected_gate_split_ref=gate_split_ref,
            expected_mechanism_evidence_ref=evaluation.mechanism_evidence_ref,
        )
        candidate_batch = self._verify_gate_batch(
            candidate_batch,
            expected_protocol_ref=experiment.protocol_ref,
            expected_protocol=protocol,
            expected_candidate_ref=candidate_ref,
            expected_arm=GateTrialArm.CANDIDATE,
            expected_harness_ref=candidate.child_harness_ref,
            expected_gate_split_ref=gate_split_ref,
            expected_mechanism_evidence_ref=evaluation.mechanism_evidence_ref,
        )
        mechanism_evidence = self._verify_mechanism_evidence(
            evaluation.mechanism_evidence_ref,
            expected_protocol_ref=experiment.protocol_ref,
            expected_protocol=protocol,
            expected_candidate_ref=candidate_ref,
            expected_candidate_harness_ref=candidate.child_harness_ref,
        )
        recomputed = PromotionGate(gate_config).evaluate(
            parent_batch.observations,
            candidate_batch.observations,
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
            gate_implementation_fingerprint=protocol.gate_implementation_fingerprint,
            gate_batch_attestor_id=protocol.gate_batch_attestor_id,
            mechanism_evidence_attestor_id=protocol.mechanism_evidence_attestor_id,
            mechanism_evidence_ref=evaluation.mechanism_evidence_ref,
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

    def _load_exact_media[ModelT](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        label: str,
        expected_media_type: str,
    ) -> ModelT:
        if ref.media_type != expected_media_type:
            raise TerminalDecisionError(
                f"{label} declares the wrong media type: expected {expected_media_type!r}"
            )
        return self._load(ref, model_type, label)

    def _verify_mechanism_evidence(
        self,
        ref: ArtifactRef,
        *,
        expected_protocol_ref: ArtifactRef,
        expected_protocol: ProtocolManifest,
        expected_candidate_ref: ArtifactRef,
        expected_candidate_harness_ref: ArtifactRef,
    ) -> MechanismEvidence:
        envelope = self._load_exact_media(
            ref,
            AttestedMechanismEvidence,
            "attested mechanism evidence",
            ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE,
        )
        try:
            verified = self._mechanism_evidence_verifier.verify(envelope)
        except (TypeError, ValueError) as exc:
            raise TerminalDecisionError(f"mechanism evidence attestation failed: {exc}") from exc
        if verified.attestor_id != expected_protocol.mechanism_evidence_attestor_id:
            raise TerminalDecisionError(
                "mechanism evidence attestor does not match the frozen protocol"
            )
        if verified.protocol_ref != expected_protocol_ref:
            raise TerminalDecisionError("mechanism evidence belongs to another protocol")
        if verified.candidate_ref != expected_candidate_ref:
            raise TerminalDecisionError("mechanism evidence belongs to another candidate")
        if verified.candidate_harness_ref != expected_candidate_harness_ref:
            raise TerminalDecisionError("mechanism evidence belongs to the wrong child harness")
        expected_split_ref = next(
            split.manifest_ref
            for split in expected_protocol.splits
            if split.partition is ProtocolPartition.EXPLORATION
        )
        if verified.exploration_split_ref != expected_split_ref:
            raise TerminalDecisionError(
                "mechanism evidence does not use the frozen exploration split"
            )
        if verified.task_set_fingerprint != expected_split_ref.sha256:
            raise TerminalDecisionError(
                "mechanism evidence task set does not match the exploration split"
            )
        if verified.execution_context != GateBatchExecutionContext.from_protocol(expected_protocol):
            raise TerminalDecisionError(
                "mechanism evidence execution context does not match the frozen protocol"
            )
        for source_ref in verified.source_refs:
            try:
                self._repository.get_bytes(source_ref)
            except Exception as exc:
                raise TerminalDecisionError(
                    f"mechanism evidence source artifact could not be verified: {exc}"
                ) from exc
        return verified.evidence

    def _verify_gate_batch(
        self,
        batch: GateTrialBatch,
        *,
        expected_protocol_ref: ArtifactRef,
        expected_protocol: ProtocolManifest,
        expected_candidate_ref: ArtifactRef,
        expected_arm: GateTrialArm,
        expected_harness_ref: ArtifactRef,
        expected_gate_split_ref: ArtifactRef,
        expected_mechanism_evidence_ref: ArtifactRef,
    ) -> GateTrialBatch:
        try:
            verified = self._gate_batch_verifier.verify(batch)
        except (TypeError, ValueError) as exc:
            raise TerminalDecisionError(f"gate trial batch attestation failed: {exc}") from exc
        if verified.attestor_id != expected_protocol.gate_batch_attestor_id:
            raise TerminalDecisionError(
                "gate trial batch attestor does not match the frozen protocol"
            )
        if verified.protocol_ref != expected_protocol_ref:
            raise TerminalDecisionError("gate trial batch belongs to another protocol")
        expected_context = GateBatchExecutionContext.from_protocol(expected_protocol)
        if verified.execution_context != expected_context:
            raise TerminalDecisionError(
                "gate trial batch execution context does not match the frozen protocol"
            )
        if verified.task_set_fingerprint != expected_gate_split_ref.sha256:
            raise TerminalDecisionError(
                "gate trial batch task set does not match the frozen gate split"
            )
        if verified.mechanism_evidence_ref != expected_mechanism_evidence_ref:
            raise TerminalDecisionError(
                "gate trial batch does not bind the evaluation mechanism evidence"
            )
        if verified.candidate_ref != expected_candidate_ref:
            raise TerminalDecisionError("gate trial batch belongs to another candidate")
        if verified.arm is not expected_arm:
            raise TerminalDecisionError(
                f"gate trial batch has arm {verified.arm.value!r}; expected {expected_arm.value!r}"
            )
        if verified.harness_ref != expected_harness_ref:
            raise TerminalDecisionError(
                f"{expected_arm.value} gate trial batch belongs to the wrong harness"
            )
        if verified.gate_split_ref != expected_gate_split_ref:
            raise TerminalDecisionError(
                f"{expected_arm.value} gate trial batch belongs to the wrong gate split"
            )
        for source_ref in verified.source_refs:
            try:
                self._repository.get_bytes(source_ref)
            except Exception as exc:
                raise TerminalDecisionError(
                    f"gate trial batch source artifact could not be verified: {exc}"
                ) from exc
        return verified

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
