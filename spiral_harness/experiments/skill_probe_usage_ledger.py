"""Global reservation and settlement accounting for matched skill probes."""

from __future__ import annotations

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.models import ArtifactRef, BudgetPolicy
from spiral_harness.storage.journal import CandidateJournal
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_plan import SkillMechanismPlan

from .controller_artifacts import (
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    ExperimentUsage,
    ExperimentUsageEntry,
    SkillProbeSettlementKind,
    SkillProbeUsageClaim,
    SkillProbeUsageSettlementClaim,
)
from .skill_probe_authorization import (
    SkillProbeExecutionAuthorization,
    _load_exact_authorization,
    verify_skill_probe_execution_authorization_history,
)
from .skill_probe_usage import SkillProbeUsageReplayError, replay_skill_probe_settlement
from .skill_probes import verify_skill_probe_preregistration


class SkillProbeUsageLedgerError(RuntimeError):
    """Raised when probe-wide accounting cannot be reconstructed exactly."""


class SkillProbeUsageBudgetError(SkillProbeUsageLedgerError):
    """Raised before execution when a complete probe reservation cannot fit."""


class SkillProbeUsageLedger:
    """Own probe-specific claims within one experiment-wide usage chain."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        experiment_ref: ArtifactRef,
        protocol_ref: ArtifactRef,
        budget_limits: BudgetPolicy,
    ) -> None:
        self._repository = repository
        self._experiment_ref = experiment_ref
        self._protocol_ref = protocol_ref
        self._limits = budget_limits
        self._candidate_journal = CandidateJournal(repository)

    def reserve(
        self,
        *,
        tail_ref: ArtifactRef | None,
        usage: ExperimentUsage,
        authorization_ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
        plan: SkillMechanismPlan,
    ) -> ArtifactRef:
        """Reserve both frozen schedules before either backend can be called."""

        checked_ref, checked_plan = self._verify_reservation_request(
            authorization_ref=authorization_ref,
            authorization=authorization,
            plan=plan,
        )
        if usage.poisoned or not usage.accounting_complete:
            raise SkillProbeUsageBudgetError(
                "experiment usage ledger is poisoned or has incomplete accounting"
            )
        matching = tuple(
            claim
            for ref in usage.skill_probe_claim_refs
            if (
                claim := self._load(
                    ref,
                    SkillProbeUsageClaim,
                    label="skill-probe usage claim",
                    media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
                )
            ).execution_nonce
            == authorization.execution_nonce
        )
        if matching:
            return self._existing_reservation_tail(
                usage=usage,
                matching=matching,
                authorization_ref=checked_ref,
                authorization=authorization,
                plan=checked_plan,
            )
        if self._limits.max_wall_time_seconds is not None:
            raise SkillProbeUsageBudgetError(
                "skill-probe execution has no preregistered wall-time ceiling"
            )
        if self._limits.max_cost_usd is not None:
            raise SkillProbeUsageBudgetError(
                "skill-probe execution has no preregistered cost ceiling"
            )
        evaluations = (
            checked_plan.revert_schedule.required_attempts
            + checked_plan.placebo_schedule.required_attempts
        )
        tokens = (
            checked_plan.revert_schedule.required_tokens
            + checked_plan.placebo_schedule.required_tokens
        )
        self._enforce_reservation_budget(
            usage=usage,
            requested_evaluations=evaluations,
            requested_tokens=tokens,
        )
        claim = SkillProbeUsageClaim(
            experiment_ref=self._experiment_ref,
            protocol_ref=self._protocol_ref,
            candidate_ref=authorization.candidate_ref,
            authorization_ref=checked_ref,
            execution_nonce=authorization.execution_nonce,
            plan_ref=authorization.plan_ref,
            running_probes_tail_ref=authorization.running_probes_tail_ref,
            revert_schedule_fingerprint=checked_plan.revert_schedule.fingerprint,
            placebo_schedule_fingerprint=checked_plan.placebo_schedule.fingerprint,
            evaluation_units=evaluations,
            tokens=tokens,
        )
        claim_ref = self._repository.put_json(
            claim,
            media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
        )
        entry = ExperimentUsageEntry(
            experiment_ref=self._experiment_ref,
            protocol_ref=self._protocol_ref,
            sequence=usage.entry_count,
            claim_ref=claim_ref,
            cumulative_evaluations=usage.total_evaluations + evaluations,
            cumulative_tokens=usage.total_tokens + tokens,
            cumulative_tool_calls=usage.total_tool_calls,
            cumulative_wall_time_seconds=None,
            cumulative_cost_usd=None,
            previous_entry_ref=tail_ref,
        )
        return self._repository.put_json(entry, media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE)

    def settle(
        self,
        *,
        tail_ref: ArtifactRef | None,
        usage: ExperimentUsage,
        authorization_ref: ArtifactRef,
        revert_preflight_ref: ArtifactRef,
        revert_terminal_tail_ref: ArtifactRef | None,
        placebo_preflight_ref: ArtifactRef,
        placebo_terminal_tail_ref: ArtifactRef | None,
        terminal_kind: SkillProbeSettlementKind,
        closure_ref: ArtifactRef | None,
    ) -> ArtifactRef:
        """Append one replay-derived terminal adjustment without refunding capacity."""

        try:
            checked_ref = ArtifactRef.model_validate(authorization_ref, strict=True)
            authorization = _load_exact_authorization(self._repository, checked_ref)
            checked_kind = SkillProbeSettlementKind(terminal_kind)
        except Exception as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe settlement authorization is invalid: {exc}"
            ) from exc
        reservation_ref, reservation = self._one_reservation(usage, checked_ref)
        existing = self._settlements(usage, checked_ref)
        if existing:
            return self._existing_settlement_tail(
                usage=usage,
                existing=existing,
                revert_preflight_ref=revert_preflight_ref,
                revert_terminal_tail_ref=revert_terminal_tail_ref,
                placebo_preflight_ref=placebo_preflight_ref,
                placebo_terminal_tail_ref=placebo_terminal_tail_ref,
                terminal_kind=checked_kind,
                closure_ref=closure_ref,
            )
        if (
            reservation.execution_nonce != authorization.execution_nonce
            or reservation.candidate_ref != authorization.candidate_ref
            or reservation.experiment_ref != self._experiment_ref
            or reservation.protocol_ref != self._protocol_ref
        ):
            raise SkillProbeUsageLedgerError(
                "skill-probe settlement differs from its reservation authorization"
            )
        plan = self._verify_plan(reservation)
        try:
            revert, placebo = replay_skill_probe_settlement(
                self._repository,
                authorization_ref=checked_ref,
                execution_nonce=reservation.execution_nonce,
                running_probes_tail_ref=reservation.running_probes_tail_ref,
                plan=plan,
                revert_preflight_ref=revert_preflight_ref,
                revert_terminal_tail_ref=revert_terminal_tail_ref,
                placebo_preflight_ref=placebo_preflight_ref,
                placebo_terminal_tail_ref=placebo_terminal_tail_ref,
                terminal_kind=checked_kind,
                closure_ref=closure_ref,
            )
        except SkillProbeUsageReplayError as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe settlement replay failed: {exc}"
            ) from exc
        encumbered = revert.encumbered_tokens + placebo.encumbered_tokens
        adjustment = max(0, encumbered - reservation.tokens)
        poisoned = revert.poisoned or placebo.poisoned
        claim = SkillProbeUsageSettlementClaim(
            experiment_ref=self._experiment_ref,
            protocol_ref=self._protocol_ref,
            candidate_ref=reservation.candidate_ref,
            authorization_ref=checked_ref,
            execution_nonce=reservation.execution_nonce,
            reservation_claim_ref=reservation_ref,
            terminal_kind=checked_kind,
            revert=revert,
            placebo=placebo,
            reserved_tokens=reservation.tokens,
            encumbered_tokens=encumbered,
            token_adjustment=adjustment,
            poisoned=poisoned,
            closure_ref=closure_ref,
        )
        claim_ref = self._repository.put_json(
            claim,
            media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
        )
        entry = ExperimentUsageEntry(
            experiment_ref=self._experiment_ref,
            protocol_ref=self._protocol_ref,
            sequence=usage.entry_count,
            claim_ref=claim_ref,
            cumulative_evaluations=usage.total_evaluations,
            cumulative_tokens=usage.total_tokens + adjustment,
            cumulative_tool_calls=usage.total_tool_calls,
            cumulative_wall_time_seconds=usage.total_wall_time_seconds,
            cumulative_cost_usd=usage.total_cost_usd,
            poisoned=usage.poisoned or poisoned,
            accounting_complete=usage.accounting_complete,
            previous_entry_ref=tail_ref,
        )
        return self._repository.put_json(entry, media_type=EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE)

    def replay_reservation(
        self,
        claim: SkillProbeUsageClaim,
        *,
        seen_authorizations: set[str],
        seen_nonces: set[str],
    ) -> tuple[int, int, int, None, None]:
        """Verify one reservation claim and recompute its schedule charge."""

        if claim.authorization_ref.sha256 in seen_authorizations:
            raise SkillProbeUsageLedgerError("skill-probe authorization was charged more than once")
        if claim.execution_nonce in seen_nonces:
            raise SkillProbeUsageLedgerError(
                "skill-probe execution nonce was charged more than once"
            )
        try:
            authorization = _load_exact_authorization(self._repository, claim.authorization_ref)
            events = self._candidate_journal.replay(claim.running_probes_tail_ref)
            verify_skill_probe_execution_authorization_history(
                self._repository,
                authorization=authorization,
                events=events,
            )
        except Exception as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe usage authorization replay failed: {exc}"
            ) from exc
        expected = (
            claim.experiment_ref,
            claim.protocol_ref,
            claim.candidate_ref,
            claim.plan_ref,
            claim.running_probes_tail_ref,
            claim.execution_nonce,
        )
        actual = (
            authorization.experiment_ref,
            authorization.protocol_ref,
            authorization.candidate_ref,
            authorization.plan_ref,
            authorization.running_probes_tail_ref,
            authorization.execution_nonce,
        )
        if expected != actual:
            raise SkillProbeUsageLedgerError(
                "skill-probe usage claim differs from its authorization"
            )
        plan = self._verify_plan(claim)
        schedules = (plan.revert_schedule.fingerprint, plan.placebo_schedule.fingerprint)
        if schedules != (
            claim.revert_schedule_fingerprint,
            claim.placebo_schedule_fingerprint,
        ):
            raise SkillProbeUsageLedgerError("skill-probe usage schedule reservation changed")
        evaluations = (
            plan.revert_schedule.required_attempts + plan.placebo_schedule.required_attempts
        )
        tokens = plan.revert_schedule.required_tokens + plan.placebo_schedule.required_tokens
        if (
            claim.evaluation_units,
            claim.tokens,
            claim.tool_calls,
            claim.wall_time_seconds,
            claim.cost_usd,
        ) != (evaluations, tokens, 0, None, None):
            raise SkillProbeUsageLedgerError(
                "skill-probe usage reservation differs from the frozen schedules"
            )
        return evaluations, tokens, 0, None, None

    def replay_settlement(
        self,
        claim: SkillProbeUsageSettlementClaim,
        *,
        reservation_refs: tuple[ArtifactRef, ...],
        seen_authorizations: set[str],
    ) -> tuple[tuple[int, int, int, float, float], bool]:
        """Verify a terminal settlement and recompute its overrun adjustment."""

        if claim.authorization_ref.sha256 in seen_authorizations:
            raise SkillProbeUsageLedgerError("skill-probe authorization was settled more than once")
        if claim.reservation_claim_ref not in reservation_refs:
            raise SkillProbeUsageLedgerError(
                "skill-probe settlement does not follow its reservation"
            )
        reservation = self._load(
            claim.reservation_claim_ref,
            SkillProbeUsageClaim,
            label="skill-probe usage reservation claim",
            media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
        )
        if (
            claim.experiment_ref,
            claim.protocol_ref,
            claim.candidate_ref,
            claim.authorization_ref,
            claim.execution_nonce,
            claim.reserved_tokens,
        ) != (
            reservation.experiment_ref,
            reservation.protocol_ref,
            reservation.candidate_ref,
            reservation.authorization_ref,
            reservation.execution_nonce,
            reservation.tokens,
        ):
            raise SkillProbeUsageLedgerError("skill-probe settlement differs from its reservation")
        plan = self._verify_plan(reservation)
        try:
            revert, placebo = replay_skill_probe_settlement(
                self._repository,
                authorization_ref=reservation.authorization_ref,
                execution_nonce=reservation.execution_nonce,
                running_probes_tail_ref=reservation.running_probes_tail_ref,
                plan=plan,
                revert_preflight_ref=claim.revert.preflight_ref,
                revert_terminal_tail_ref=claim.revert.terminal_tail_ref,
                placebo_preflight_ref=claim.placebo.preflight_ref,
                placebo_terminal_tail_ref=claim.placebo.terminal_tail_ref,
                terminal_kind=claim.terminal_kind,
                closure_ref=claim.closure_ref,
            )
        except SkillProbeUsageReplayError as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe settlement replay failed: {exc}"
            ) from exc
        encumbered = revert.encumbered_tokens + placebo.encumbered_tokens
        adjustment = max(0, encumbered - reservation.tokens)
        recomputed = (
            revert,
            placebo,
            encumbered,
            adjustment,
            revert.poisoned or placebo.poisoned,
        )
        persisted = (
            claim.revert,
            claim.placebo,
            claim.encumbered_tokens,
            claim.token_adjustment,
            claim.poisoned,
        )
        if persisted != recomputed:
            raise SkillProbeUsageLedgerError(
                "skill-probe settlement differs from replayed attempt accounting"
            )
        return (0, adjustment, 0, 0.0, 0.0), claim.poisoned

    def _verify_reservation_request(
        self,
        *,
        authorization_ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
        plan: SkillMechanismPlan,
    ) -> tuple[ArtifactRef, SkillMechanismPlan]:
        try:
            checked_ref = ArtifactRef.model_validate(authorization_ref, strict=True)
            checked_plan = SkillMechanismPlan.model_validate(plan, strict=True)
            stored = _load_exact_authorization(self._repository, checked_ref)
        except Exception as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe usage authorization replay failed: {exc}"
            ) from exc
        if stored != authorization:
            raise SkillProbeUsageLedgerError(
                "skill-probe usage reservation received a substituted authorization"
            )
        replayed = self._verify_plan(authorization)
        if replayed != checked_plan:
            raise SkillProbeUsageLedgerError(
                "skill-probe usage reservation received a substituted plan"
            )
        if (
            authorization.experiment_ref,
            authorization.protocol_ref,
            authorization.plan_ref,
            authorization.candidate_ref,
        ) != (
            self._experiment_ref,
            self._protocol_ref,
            checked_plan.artifact_ref,
            checked_plan.candidate_ref,
        ):
            raise SkillProbeUsageLedgerError(
                "skill-probe usage reservation differs from its authorization"
            )
        return checked_ref, checked_plan

    def _verify_plan(
        self,
        context: SkillProbeExecutionAuthorization | SkillProbeUsageClaim,
    ) -> SkillMechanismPlan:
        try:
            return verify_skill_probe_preregistration(
                self._repository,
                plan_ref=context.plan_ref,
                expected_experiment_ref=context.experiment_ref,
                expected_protocol_ref=context.protocol_ref,
                expected_candidate_ref=context.candidate_ref,
            )
        except Exception as exc:
            raise SkillProbeUsageLedgerError(
                f"skill-probe usage plan replay failed: {exc}"
            ) from exc

    def _one_reservation(
        self,
        usage: ExperimentUsage,
        authorization_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, SkillProbeUsageClaim]:
        matching = tuple(
            (ref, claim)
            for ref in usage.skill_probe_claim_refs
            if (
                claim := self._load(
                    ref,
                    SkillProbeUsageClaim,
                    label="skill-probe usage claim",
                    media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
                )
            ).authorization_ref
            == authorization_ref
        )
        if len(matching) != 1:
            raise SkillProbeUsageLedgerError(
                "skill-probe settlement requires exactly one prior reservation"
            )
        return matching[0]

    def _settlements(
        self,
        usage: ExperimentUsage,
        authorization_ref: ArtifactRef,
    ) -> tuple[SkillProbeUsageSettlementClaim, ...]:
        return tuple(
            claim
            for ref in usage.skill_probe_settlement_refs
            if (
                claim := self._load(
                    ref,
                    SkillProbeUsageSettlementClaim,
                    label="skill-probe usage settlement claim",
                    media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
                )
            ).authorization_ref
            == authorization_ref
        )

    def _existing_reservation_tail(
        self,
        *,
        usage: ExperimentUsage,
        matching: tuple[SkillProbeUsageClaim, ...],
        authorization_ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
        plan: SkillMechanismPlan,
    ) -> ArtifactRef:
        if len(matching) != 1:
            raise SkillProbeUsageLedgerError(
                "skill-probe authorization has multiple usage reservations"
            )
        claim = matching[0]
        if (
            claim.authorization_ref,
            claim.plan_ref,
            claim.candidate_ref,
            claim.running_probes_tail_ref,
            claim.revert_schedule_fingerprint,
            claim.placebo_schedule_fingerprint,
        ) != (
            authorization_ref,
            authorization.plan_ref,
            authorization.candidate_ref,
            authorization.running_probes_tail_ref,
            plan.revert_schedule.fingerprint,
            plan.placebo_schedule.fingerprint,
        ):
            raise SkillProbeUsageLedgerError(
                "skill-probe authorization reservation context changed"
            )
        if usage.tail_ref is None:  # pragma: no cover - matching implies an entry
            raise SkillProbeUsageLedgerError("skill-probe reservation has no usage tail")
        return usage.tail_ref

    @staticmethod
    def _existing_settlement_tail(
        *,
        usage: ExperimentUsage,
        existing: tuple[SkillProbeUsageSettlementClaim, ...],
        revert_preflight_ref: ArtifactRef,
        revert_terminal_tail_ref: ArtifactRef | None,
        placebo_preflight_ref: ArtifactRef,
        placebo_terminal_tail_ref: ArtifactRef | None,
        terminal_kind: SkillProbeSettlementKind,
        closure_ref: ArtifactRef | None,
    ) -> ArtifactRef:
        if len(existing) != 1:
            raise SkillProbeUsageLedgerError(
                "skill-probe authorization has multiple usage settlements"
            )
        claim = existing[0]
        if (
            claim.revert.preflight_ref,
            claim.revert.terminal_tail_ref,
            claim.placebo.preflight_ref,
            claim.placebo.terminal_tail_ref,
            claim.terminal_kind,
            claim.closure_ref,
        ) != (
            revert_preflight_ref,
            revert_terminal_tail_ref,
            placebo_preflight_ref,
            placebo_terminal_tail_ref,
            terminal_kind,
            closure_ref,
        ):
            raise SkillProbeUsageLedgerError("skill-probe authorization settlement context changed")
        if usage.tail_ref is None:  # pragma: no cover - settlement implies a tail
            raise SkillProbeUsageLedgerError("skill-probe settlement has no usage tail")
        return usage.tail_ref

    def _enforce_reservation_budget(
        self,
        *,
        usage: ExperimentUsage,
        requested_evaluations: int,
        requested_tokens: int,
    ) -> None:
        evaluations = usage.total_evaluations + requested_evaluations
        tokens = usage.total_tokens + requested_tokens
        if self._limits.max_evaluations is not None and evaluations > self._limits.max_evaluations:
            raise SkillProbeUsageBudgetError(
                "gate evaluation would exceed max_evaluations: "
                f"used={usage.total_evaluations}, requested={requested_evaluations}, "
                f"limit={self._limits.max_evaluations}"
            )
        if self._limits.max_tokens is not None and tokens > self._limits.max_tokens:
            raise SkillProbeUsageBudgetError(
                f"persisted gate usage exceeds max_tokens: used={tokens}, "
                f"limit={self._limits.max_tokens}"
            )
        if (
            self._limits.max_tool_calls is not None
            and usage.total_tool_calls > self._limits.max_tool_calls
        ):
            raise SkillProbeUsageBudgetError(
                "persisted gate usage exceeds max_tool_calls: "
                f"used={usage.total_tool_calls}, limit={self._limits.max_tool_calls}"
            )

    def _load[ModelT: BaseModel](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        *,
        label: str,
        media_type: str,
    ) -> ModelT:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
        if checked_ref.media_type != media_type:
            raise SkillProbeUsageLedgerError(f"{label} declares the wrong media type")
        try:
            payload = self._repository.get_bytes(checked_ref)
            loaded = self._repository.get_json(checked_ref, model_type)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise SkillProbeUsageLedgerError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise SkillProbeUsageLedgerError(
                f"could not load canonical {label}: typed representation is not canonical"
            )
        return loaded


__all__ = [
    "SkillProbeUsageBudgetError",
    "SkillProbeUsageLedger",
    "SkillProbeUsageLedgerError",
]
