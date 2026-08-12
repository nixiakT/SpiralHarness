"""Replay experiment usage and reserve controller-authorized skill probes."""

from __future__ import annotations

from collections.abc import Callable

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import CandidateManifest
from spiral_harness.core.models import ArtifactRef, BudgetPolicy
from spiral_harness.storage.journal import CandidateJournal
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.artifacts import GateTrialBatch
from spiral_harness.verification.skill_plan import SkillMechanismPlan

from .controller_artifacts import (
    EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE,
    EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE,
    SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
    SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
    ExperimentUsage,
    ExperimentUsageClaim,
    ExperimentUsageEntry,
    ExperimentUsageEntryV1,
    SkillProbeSettlementKind,
    SkillProbeUsageClaim,
    SkillProbeUsageSettlementClaim,
)
from .decision import GateEvaluationManifest
from .skill_probe_authorization import (
    SkillProbeExecutionAuthorization,
)
from .skill_probe_usage_ledger import (
    SkillProbeUsageBudgetError,
    SkillProbeUsageLedger,
    SkillProbeUsageLedgerError,
)

UsageEntry = ExperimentUsageEntryV1 | ExperimentUsageEntry


class ExperimentUsageLedgerError(RuntimeError):
    """Raised when a persisted usage chain cannot be replayed exactly."""


class ExperimentUsageBudgetError(ExperimentUsageLedgerError):
    """Raised when a usage charge exceeds a frozen experiment ceiling."""


GateHistoryVerifier = Callable[
    [ArtifactRef],
    tuple[CandidateManifest, ArtifactRef, ArtifactRef],
]
GateEvaluationVerifier = Callable[
    [ArtifactRef, CandidateManifest, ArtifactRef, ArtifactRef, ArtifactRef],
    tuple[GateEvaluationManifest, GateTrialBatch, GateTrialBatch, int],
]


class ExperimentUsageLedger:
    """Replay exact usage and reserve matched probes for one frozen experiment."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        experiment_ref: ArtifactRef,
        protocol_ref: ArtifactRef,
        budget_limits: BudgetPolicy,
        verify_gate_history: GateHistoryVerifier,
        verify_gate_evaluation: GateEvaluationVerifier,
    ) -> None:
        self._repository = repository
        self.experiment_ref = experiment_ref
        self.protocol_ref = protocol_ref
        self._limits = budget_limits
        if budget_limits.max_evaluations is None:
            raise ExperimentUsageLedgerError("usage ledger requires max_evaluations")
        self._max_evaluations = budget_limits.max_evaluations
        self._verify_gate_history = verify_gate_history
        self._verify_gate_evaluation = verify_gate_evaluation
        self._candidate_journal = CandidateJournal(repository)
        self._probe_usage = SkillProbeUsageLedger(
            repository,
            experiment_ref=experiment_ref,
            protocol_ref=protocol_ref,
            budget_limits=budget_limits,
        )

    def load_claim[ModelT](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        *,
        label: str,
        media_type: str,
    ) -> ModelT:
        checked_ref = ArtifactRef.model_validate(ref)
        if checked_ref.media_type != media_type:
            raise ExperimentUsageLedgerError(f"{label} declares the wrong media type")
        try:
            payload = self._repository.get_bytes(checked_ref)
            loaded = self._repository.get_json(checked_ref, model_type)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise ExperimentUsageLedgerError(f"could not load canonical {label}: {exc}") from exc
        if payload != canonical:
            raise ExperimentUsageLedgerError(
                f"could not load canonical {label}: typed representation is not canonical"
            )
        return loaded

    @staticmethod
    def resource_charge(
        parent_batch: GateTrialBatch,
        candidate_batch: GateTrialBatch,
    ) -> tuple[int, int, float, float | None]:
        observations = (*parent_batch.observations, *candidate_batch.observations)
        tokens = sum(observation.tokens for observation in observations)
        tool_calls = sum(observation.tool_calls for observation in observations)
        wall_time = sum(observation.latency_ms for observation in observations) / 1_000
        costs = tuple(observation.cost_usd for observation in observations)
        cost = (
            None
            if any(value is None for value in costs)
            else sum(value for value in costs if value is not None)
        )
        return tokens, tool_calls, wall_time, cost

    def enforce_budget(
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
        if evaluations > self._max_evaluations:
            raise ExperimentUsageBudgetError(
                "gate evaluation would exceed max_evaluations: "
                f"used={prior_evaluations}, requested={requested_evaluations}, "
                f"limit={self._max_evaluations}"
            )
        checks = (
            ("max_tokens", tokens, self._limits.max_tokens),
            ("max_tool_calls", tool_calls, self._limits.max_tool_calls),
            ("max_wall_time_seconds", wall_time_seconds, self._limits.max_wall_time_seconds),
        )
        for name, used, limit in checks:
            if limit is not None and used > limit:
                raise ExperimentUsageBudgetError(
                    f"persisted gate usage exceeds {name}: used={used}, limit={limit}"
                )
        if self._limits.max_cost_usd is not None:
            if cost_usd is None:
                raise ExperimentUsageBudgetError(
                    "cost ceiling is active but one or more observations omit cost_usd"
                )
            if cost_usd > self._limits.max_cost_usd:
                raise ExperimentUsageBudgetError(
                    "persisted gate usage exceeds max_cost_usd: "
                    f"used={cost_usd}, limit={self._limits.max_cost_usd}"
                )

    def reserve_skill_probe(
        self,
        *,
        tail_ref: ArtifactRef | None,
        authorization_ref: ArtifactRef,
        authorization: SkillProbeExecutionAuthorization,
        plan: SkillMechanismPlan,
    ) -> ArtifactRef:
        try:
            return self._probe_usage.reserve(
                tail_ref=tail_ref,
                usage=self.replay(tail_ref),
                authorization_ref=authorization_ref,
                authorization=authorization,
                plan=plan,
            )
        except SkillProbeUsageBudgetError as exc:
            raise ExperimentUsageBudgetError(str(exc)) from exc
        except SkillProbeUsageLedgerError as exc:
            raise ExperimentUsageLedgerError(str(exc)) from exc

    def settle_skill_probe(
        self,
        *,
        tail_ref: ArtifactRef | None,
        authorization_ref: ArtifactRef,
        revert_preflight_ref: ArtifactRef,
        revert_terminal_tail_ref: ArtifactRef | None,
        placebo_preflight_ref: ArtifactRef,
        placebo_terminal_tail_ref: ArtifactRef | None,
        terminal_kind: SkillProbeSettlementKind,
        closure_ref: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Append one replay-derived terminal adjustment for a reserved probe."""

        try:
            return self._probe_usage.settle(
                tail_ref=tail_ref,
                usage=self.replay(tail_ref),
                authorization_ref=authorization_ref,
                revert_preflight_ref=revert_preflight_ref,
                revert_terminal_tail_ref=revert_terminal_tail_ref,
                placebo_preflight_ref=placebo_preflight_ref,
                placebo_terminal_tail_ref=placebo_terminal_tail_ref,
                terminal_kind=terminal_kind,
                closure_ref=closure_ref,
            )
        except SkillProbeUsageBudgetError as exc:
            raise ExperimentUsageBudgetError(str(exc)) from exc
        except SkillProbeUsageLedgerError as exc:
            raise ExperimentUsageLedgerError(str(exc)) from exc

    def replay(self, tail_ref: ArtifactRef | None) -> ExperimentUsage:
        if tail_ref is None:
            return self._empty_usage()
        try:
            entries = self._entries(tail_ref)
        except ExperimentUsageLedgerError:
            raise
        except Exception as exc:
            raise ExperimentUsageLedgerError(
                f"could not load experiment usage ledger: {exc}"
            ) from exc
        gate_claim_refs: list[ArtifactRef] = []
        candidate_refs: list[ArtifactRef] = []
        evaluation_refs: list[ArtifactRef] = []
        probe_claim_refs: list[ArtifactRef] = []
        authorization_refs: list[ArtifactRef] = []
        settlement_refs: list[ArtifactRef] = []
        settled_authorizations: set[str] = set()
        probe_nonces: set[str] = set()
        seen_candidates: set[str] = set()
        seen_evaluations: set[str] = set()
        seen_batches: set[str] = set()
        totals: tuple[int, int, int, float | None, float | None] = (0, 0, 0, 0.0, 0.0)
        poisoned = False
        accounting_complete = True
        previous: ArtifactRef | None = None
        for sequence, (entry_ref, entry) in enumerate(entries):
            if poisoned:
                raise ExperimentUsageLedgerError(
                    "experiment usage ledger continues after a poisoned terminal entry"
                )
            self._verify_entry(entry, sequence=sequence, previous=previous)
            if entry.claim_ref.media_type == EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE:
                claim = self.load_claim(
                    entry.claim_ref,
                    ExperimentUsageClaim,
                    label="experiment usage claim",
                    media_type=EXPERIMENT_USAGE_CLAIM_MEDIA_TYPE,
                )
                charge = self._replay_gate_claim(
                    claim,
                    seen_candidates=seen_candidates,
                    seen_evaluations=seen_evaluations,
                    seen_batches=seen_batches,
                )
                gate_claim_refs.append(entry.claim_ref)
                candidate_refs.append(claim.candidate_ref)
                evaluation_refs.append(claim.evaluation_ref)
                seen_candidates.add(claim.candidate_ref.sha256)
                seen_evaluations.add(claim.evaluation_ref.sha256)
                seen_batches.update(
                    {claim.parent_batch_ref.sha256, claim.candidate_batch_ref.sha256}
                )
            elif entry.claim_ref.media_type == SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE:
                claim = self.load_claim(
                    entry.claim_ref,
                    SkillProbeUsageClaim,
                    label="skill-probe usage claim",
                    media_type=SKILL_PROBE_USAGE_CLAIM_MEDIA_TYPE,
                )
                charge = self._replay_probe_claim(
                    claim,
                    seen_authorizations={ref.sha256 for ref in authorization_refs},
                    seen_nonces=probe_nonces,
                )
                probe_claim_refs.append(entry.claim_ref)
                authorization_refs.append(claim.authorization_ref)
                probe_nonces.add(claim.execution_nonce)
            elif entry.claim_ref.media_type == SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE:
                claim = self.load_claim(
                    entry.claim_ref,
                    SkillProbeUsageSettlementClaim,
                    label="skill-probe usage settlement claim",
                    media_type=SKILL_PROBE_USAGE_SETTLEMENT_CLAIM_MEDIA_TYPE,
                )
                charge, settlement_poison = self._replay_probe_settlement_claim(
                    claim,
                    reservation_refs=tuple(probe_claim_refs),
                    seen_authorizations=settled_authorizations,
                )
                settlement_refs.append(entry.claim_ref)
                settled_authorizations.add(claim.authorization_ref.sha256)
                poisoned = poisoned or settlement_poison
            else:  # pragma: no cover - entry schema rejects this first
                raise ExperimentUsageLedgerError("usage claim type is unsupported")
            if claim.experiment_ref != self.experiment_ref:
                raise ExperimentUsageLedgerError("usage claim belongs to another experiment")
            if claim.protocol_ref != self.protocol_ref:
                raise ExperimentUsageLedgerError("usage claim belongs to another protocol")
            totals = self._add_charge(totals, charge)
            if not poisoned:
                self._enforce_replayed_budget(totals, requested_evaluations=charge[0])
            persisted = (
                entry.cumulative_evaluations,
                entry.cumulative_tokens,
                entry.cumulative_tool_calls,
                entry.cumulative_wall_time_seconds,
                entry.cumulative_cost_usd,
            )
            if persisted != totals:
                raise ExperimentUsageLedgerError(
                    "usage entry cumulative resources are inconsistent"
                )
            entry_poisoned = False if isinstance(entry, ExperimentUsageEntryV1) else entry.poisoned
            entry_complete = (
                True if isinstance(entry, ExperimentUsageEntryV1) else entry.accounting_complete
            )
            if entry_poisoned != poisoned:
                raise ExperimentUsageLedgerError("usage entry poison state is inconsistent")
            accounting_complete = accounting_complete and entry_complete
            previous = entry_ref
        return self._usage_result(
            tail_ref=entries[-1][0],
            entries=entries,
            gate_claim_refs=tuple(gate_claim_refs),
            candidate_refs=tuple(candidate_refs),
            evaluation_refs=tuple(evaluation_refs),
            probe_claim_refs=tuple(probe_claim_refs),
            authorization_refs=tuple(authorization_refs),
            settlement_refs=tuple(settlement_refs),
            totals=totals,
            poisoned=poisoned,
            accounting_complete=accounting_complete,
        )

    def _entries(
        self,
        tail_ref: ArtifactRef,
    ) -> tuple[tuple[ArtifactRef, UsageEntry], ...]:
        cursor: ArtifactRef | None = ArtifactRef.model_validate(tail_ref)
        backwards: list[tuple[ArtifactRef, UsageEntry]] = []
        seen: set[str] = set()
        while cursor is not None:
            if cursor.sha256 in seen:
                raise ExperimentUsageLedgerError("experiment usage ledger contains a cycle")
            seen.add(cursor.sha256)
            if cursor.media_type == EXPERIMENT_USAGE_ENTRY_V1_MEDIA_TYPE:
                model_type = ExperimentUsageEntryV1
            elif cursor.media_type == EXPERIMENT_USAGE_ENTRY_MEDIA_TYPE:
                model_type = ExperimentUsageEntry
            else:
                raise ExperimentUsageLedgerError(
                    "experiment usage entry declares an unsupported version"
                )
            entry = self.load_claim(
                cursor,
                model_type,
                label="experiment usage entry",
                media_type=cursor.media_type,
            )
            backwards.append((cursor, entry))
            cursor = entry.previous_entry_ref
        return tuple(reversed(backwards))

    def _verify_entry(
        self,
        entry: UsageEntry,
        *,
        sequence: int,
        previous: ArtifactRef | None,
    ) -> None:
        if entry.sequence != sequence:
            raise ExperimentUsageLedgerError("experiment usage sequence is not contiguous")
        if entry.previous_entry_ref != previous:
            raise ExperimentUsageLedgerError("experiment usage link does not match prior entry")
        if entry.experiment_ref != self.experiment_ref:
            raise ExperimentUsageLedgerError("usage entry belongs to another experiment")
        if entry.protocol_ref != self.protocol_ref:
            raise ExperimentUsageLedgerError("usage entry belongs to another protocol")

    def _replay_gate_claim(
        self,
        claim: ExperimentUsageClaim,
        *,
        seen_candidates: set[str],
        seen_evaluations: set[str],
        seen_batches: set[str],
    ) -> tuple[int, int, int, float, float | None]:
        if claim.candidate_ref.sha256 in seen_candidates:
            raise ExperimentUsageLedgerError("candidate was charged more than once")
        if claim.evaluation_ref.sha256 in seen_evaluations:
            raise ExperimentUsageLedgerError("gate evaluation was charged more than once")
        batches = {claim.parent_batch_ref.sha256, claim.candidate_batch_ref.sha256}
        if seen_batches.intersection(batches):
            raise ExperimentUsageLedgerError("gate trial batch was charged more than once")
        if len(batches) != 2:
            raise ExperimentUsageLedgerError("parent and candidate batch refs must differ")
        try:
            events = self._candidate_journal.replay(claim.running_gate_tail_ref)
            candidate, admission_ref, mechanism_ref = self._verify_gate_history(
                claim.running_gate_tail_ref
            )
            evaluation, parent, child, units = self._verify_gate_evaluation(
                candidate_ref=claim.candidate_ref,
                candidate=candidate,
                admission_report_ref=admission_ref,
                mechanism_evidence_ref=mechanism_ref,
                evaluation_ref=claim.evaluation_ref,
            )
        except Exception as exc:
            raise ExperimentUsageLedgerError(f"gate usage replay failed: {exc}") from exc
        if claim.candidate_ref != events[0].candidate_ref:
            raise ExperimentUsageLedgerError("usage claim source tail belongs to another candidate")
        if evaluation.parent_batch_ref != claim.parent_batch_ref:
            raise ExperimentUsageLedgerError("usage claim parent batch does not match evaluation")
        if evaluation.candidate_batch_ref != claim.candidate_batch_ref:
            raise ExperimentUsageLedgerError(
                "usage claim candidate batch does not match evaluation"
            )
        if claim.evaluation_units != units:
            raise ExperimentUsageLedgerError(
                "usage claim evaluation count does not match persisted batches"
            )
        tokens, tools, wall_time, cost = self.resource_charge(parent, child)
        if (claim.tokens, claim.tool_calls, claim.wall_time_seconds, claim.cost_usd) != (
            tokens,
            tools,
            wall_time,
            cost,
        ):
            raise ExperimentUsageLedgerError(
                "usage claim resources do not match persisted trial batches"
            )
        return units, tokens, tools, wall_time, cost

    def _replay_probe_claim(
        self,
        claim: SkillProbeUsageClaim,
        *,
        seen_authorizations: set[str],
        seen_nonces: set[str],
    ) -> tuple[int, int, int, None, None]:
        try:
            return self._probe_usage.replay_reservation(
                claim,
                seen_authorizations=seen_authorizations,
                seen_nonces=seen_nonces,
            )
        except SkillProbeUsageLedgerError as exc:
            raise ExperimentUsageLedgerError(str(exc)) from exc

    def _replay_probe_settlement_claim(
        self,
        claim: SkillProbeUsageSettlementClaim,
        *,
        reservation_refs: tuple[ArtifactRef, ...],
        seen_authorizations: set[str],
    ) -> tuple[tuple[int, int, int, float, float], bool]:
        try:
            return self._probe_usage.replay_settlement(
                claim,
                reservation_refs=reservation_refs,
                seen_authorizations=seen_authorizations,
            )
        except SkillProbeUsageLedgerError as exc:
            raise ExperimentUsageLedgerError(str(exc)) from exc

    @staticmethod
    def _add_charge(
        totals: tuple[int, int, int, float | None, float | None],
        charge: tuple[int, int, int, float | None, float | None],
    ) -> tuple[int, int, int, float | None, float | None]:
        evaluations, tokens, tools, wall_time, cost = totals
        add_evaluations, add_tokens, add_tools, add_wall_time, add_cost = charge
        return (
            evaluations + add_evaluations,
            tokens + add_tokens,
            tools + add_tools,
            None if wall_time is None or add_wall_time is None else wall_time + add_wall_time,
            None if cost is None or add_cost is None else cost + add_cost,
        )

    def _enforce_replayed_budget(
        self,
        totals: tuple[int, int, int, float | None, float | None],
        *,
        requested_evaluations: int,
    ) -> None:
        evaluations, tokens, tools, wall_time, cost = totals
        if self._limits.max_wall_time_seconds is not None and wall_time is None:
            raise ExperimentUsageBudgetError(
                "wall-time ceiling is active but usage has no trusted upper bound"
            )
        if self._limits.max_cost_usd is not None and cost is None:
            raise ExperimentUsageBudgetError(
                "cost ceiling is active but usage has no trusted upper bound"
            )
        self.enforce_budget(
            evaluations=evaluations,
            tokens=tokens,
            tool_calls=tools,
            wall_time_seconds=wall_time or 0.0,
            cost_usd=cost,
            requested_evaluations=requested_evaluations,
            prior_evaluations=evaluations - requested_evaluations,
        )

    def _empty_usage(self) -> ExperimentUsage:
        return self._usage_result(
            tail_ref=None,
            entries=(),
            gate_claim_refs=(),
            candidate_refs=(),
            evaluation_refs=(),
            probe_claim_refs=(),
            authorization_refs=(),
            settlement_refs=(),
            totals=(0, 0, 0, 0.0, 0.0),
            poisoned=False,
            accounting_complete=True,
        )

    def _usage_result(
        self,
        *,
        tail_ref: ArtifactRef | None,
        entries: tuple[tuple[ArtifactRef, UsageEntry], ...],
        gate_claim_refs: tuple[ArtifactRef, ...],
        candidate_refs: tuple[ArtifactRef, ...],
        evaluation_refs: tuple[ArtifactRef, ...],
        probe_claim_refs: tuple[ArtifactRef, ...],
        authorization_refs: tuple[ArtifactRef, ...],
        settlement_refs: tuple[ArtifactRef, ...],
        totals: tuple[int, int, int, float | None, float | None],
        poisoned: bool,
        accounting_complete: bool,
    ) -> ExperimentUsage:
        evaluations, tokens, tools, wall_time, cost = totals
        return ExperimentUsage(
            experiment_ref=self.experiment_ref,
            protocol_ref=self.protocol_ref,
            tail_ref=tail_ref,
            entry_refs=tuple(ref for ref, _ in entries),
            entry_count=len(entries),
            claim_refs=gate_claim_refs,
            candidate_refs=candidate_refs,
            evaluation_refs=evaluation_refs,
            skill_probe_claim_refs=probe_claim_refs,
            skill_probe_authorization_refs=authorization_refs,
            skill_probe_settlement_refs=settlement_refs,
            poisoned=poisoned,
            accounting_complete=accounting_complete,
            query_count=len(gate_claim_refs),
            total_evaluations=evaluations,
            total_tokens=tokens,
            total_tool_calls=tools,
            total_wall_time_seconds=wall_time,
            total_cost_usd=cost,
            max_evaluations=self._max_evaluations,
            max_tokens=self._limits.max_tokens,
            max_tool_calls=self._limits.max_tool_calls,
            max_wall_time_seconds=self._limits.max_wall_time_seconds,
            max_cost_usd=self._limits.max_cost_usd,
            remaining_evaluations=self._max_evaluations - evaluations,
            remaining_tokens=(
                None
                if self._limits.max_tokens is None
                # A poisoned stream cannot authorize any further work.  Its
                # numeric token head may still be below a loose global ceiling
                # when an early arm overruns while another arm's reservation
                # remains unused, so the remaining capacity must fail closed.
                else 0
                if poisoned
                else max(0, self._limits.max_tokens - tokens)
            ),
            remaining_tool_calls=(
                None if self._limits.max_tool_calls is None else self._limits.max_tool_calls - tools
            ),
            remaining_wall_time_seconds=(
                None
                if self._limits.max_wall_time_seconds is None or wall_time is None
                else self._limits.max_wall_time_seconds - wall_time
            ),
            remaining_cost_usd=(
                None
                if self._limits.max_cost_usd is None or cost is None
                else self._limits.max_cost_usd - cost
            ),
        )


__all__ = [
    "ExperimentUsageBudgetError",
    "ExperimentUsageLedger",
    "ExperimentUsageLedgerError",
]
