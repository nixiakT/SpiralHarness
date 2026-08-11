"""Fail-closed verifier for the M0.2 controlled adversarial fixture."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import model_validator

from spiral_harness.benchmark.grading import (
    MECHANISM_CHECK_NAMES,
    CaptureAttestation,
    CapturedOutputTrace,
    FixedOutputGrader,
    GradingTask,
    SealedTaskRoster,
    TrustedGradingError,
)
from spiral_harness.core import canonical_json_bytes
from spiral_harness.core.models import ImmutableModel, Sha256
from spiral_harness.verification import Decision, GateConfig, GateDecision, PromotionGate
from spiral_harness.verification.models import TrialObservation


class AdversarialVerificationResult(ImmutableModel):
    """Outcome that distinguishes pre-gate rejection from a recomputable gate result."""

    decision: Decision
    task_set_fingerprint: Sha256
    grader_fingerprint: Sha256
    gate_decision: GateDecision | None
    parent_observations: tuple[TrialObservation, ...] = ()
    candidate_observations: tuple[TrialObservation, ...] = ()
    parent_capture_attestations: tuple[CaptureAttestation, ...] = ()
    candidate_capture_attestations: tuple[CaptureAttestation, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def gate_and_decision_are_consistent(self) -> AdversarialVerificationResult:
        if self.gate_decision is None:
            if self.decision is not Decision.REJECT or not self.failure_reasons:
                raise ValueError("a pre-gate result must be an explained rejection")
        elif self.decision is not self.gate_decision.decision:
            raise ValueError("result decision must equal the trusted gate decision")
        return self


class ControlledAdversarialVerifier:
    """Run trusted grading over one constructor-frozen sealed task roster.

    ``verify`` intentionally accepts captures only.  Answers, protected slices,
    and roster membership cannot be replaced after this verifier is created.
    """

    def __init__(
        self,
        grader: FixedOutputGrader,
        gate_config: GateConfig,
        *,
        tasks: Sequence[GradingTask],
    ) -> None:
        required = set(gate_config.required_mechanism_checks)
        missing = set(MECHANISM_CHECK_NAMES).difference(required)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"adversarial gate must require trusted mechanism checks: {joined}")
        self._grader = grader
        self._gate_config = GateConfig.model_validate(
            gate_config.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        try:
            validated_tasks = tuple(GradingTask.model_validate(task) for task in tasks)
            roster = SealedTaskRoster(tasks=validated_tasks)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sealed task roster: {exc}") from exc
        self._roster_payload = canonical_json_bytes(roster)
        self._task_set_fingerprint = roster.fingerprint

    @property
    def task_set_fingerprint(self) -> str:
        """Return the full sealed roster identity frozen at construction."""

        return self._task_set_fingerprint

    @property
    def grader_fingerprint(self) -> str:
        """Return the exact grader and capture-attestor identity."""

        return self._grader.fingerprint

    def _tasks(self) -> tuple[GradingTask, ...]:
        roster = SealedTaskRoster.model_validate_json(self._roster_payload)
        if roster.fingerprint != self._task_set_fingerprint:  # pragma: no cover - bytes invariant
            raise TrustedGradingError("frozen sealed task roster fingerprint changed")
        return roster.tasks

    def verify(
        self,
        parent_captures: Sequence[CapturedOutputTrace | object],
        candidate_captures: Sequence[CapturedOutputTrace | object],
    ) -> AdversarialVerificationResult:
        """Reject malformed grading inputs or evaluate only trusted observations."""

        try:
            tasks = self._tasks()
            parent = self._grader.grade_batch(tasks, parent_captures, candidate=False)
            candidate = self._grader.grade_batch(tasks, candidate_captures, candidate=True)
            for batch in (parent, candidate):
                if batch.task_set_fingerprint != self._task_set_fingerprint:
                    raise TrustedGradingError(
                        "graded batch does not match the frozen sealed task roster"
                    )
                if batch.grader_fingerprint != self._grader.fingerprint:
                    raise TrustedGradingError("graded batch came from another grader")
        except (TypeError, ValueError) as exc:
            return AdversarialVerificationResult(
                decision=Decision.REJECT,
                task_set_fingerprint=self._task_set_fingerprint,
                grader_fingerprint=self._grader.fingerprint,
                gate_decision=None,
                failure_reasons=(f"trusted grading failed closed: {exc}",),
            )

        if candidate.mechanism_evidence is None:  # pragma: no cover - constructor invariant
            return AdversarialVerificationResult(
                decision=Decision.REJECT,
                task_set_fingerprint=self._task_set_fingerprint,
                grader_fingerprint=self._grader.fingerprint,
                gate_decision=None,
                failure_reasons=("trusted grader did not emit candidate mechanism evidence",),
            )
        parent_harness_id = parent.observations[0].harness_id
        candidate_harness_id = candidate.observations[0].harness_id
        decision = PromotionGate(self._gate_config).evaluate(
            parent.observations,
            candidate.observations,
            candidate.mechanism_evidence,
            parent_harness_id=parent_harness_id,
            candidate_harness_id=candidate_harness_id,
        )
        return AdversarialVerificationResult(
            decision=decision.decision,
            task_set_fingerprint=self._task_set_fingerprint,
            grader_fingerprint=self._grader.fingerprint,
            gate_decision=decision,
            parent_observations=parent.observations,
            candidate_observations=candidate.observations,
            parent_capture_attestations=parent.capture_attestations,
            candidate_capture_attestations=candidate.capture_attestations,
        )


__all__ = ["AdversarialVerificationResult", "ControlledAdversarialVerifier"]
