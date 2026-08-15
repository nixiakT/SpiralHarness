"""Injected stage-major executor and append-only BFCL public v2 journal."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts import (
    BfclV4PublicV2PureAtBBatchGradeRequest,
    build_bfcl_v4_public_v2_pure_at_b_batch_grade_request,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2CallReservation,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2ExecutionReceipt,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2JournalSnapshot,
    BfclV4PublicV2ProposalDisposition,
    BfclV4PublicV2ProviderAttempt,
    BfclV4PublicV2ProviderRequest,
    BfclV4PublicV2PureAtBBatchGradeProjection,
    BfclV4PublicV2TrustedGradeProjection,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_replay import (
    BfclV4PublicV2ExecutorError,
    BfclV4PublicV2ReplayError,
    _build_control_event,
    _checked_campaign,
    _expected_executed_variant,
    _request,
    _validate_event,
    replay_bfcl_v4_public_v2_journal,
)


class BfclV4PublicV2StaleTailError(BfclV4PublicV2ExecutorError):
    """An append attempted to extend a stale or foreign journal tail."""


class BfclV4PublicV2CheckpointError(BfclV4PublicV2ExecutorError):
    """A durable reservation or terminal checkpoint failed before acknowledgement."""


class BfclV4PublicV2ProviderTransport(Protocol):
    """One-shot provider boundary; the core never retries this method."""

    def execute(
        self,
        request: BfclV4PublicV2ProviderRequest,
        node: BfclV4PublicDevelopmentV2DagNode,
    ) -> BfclV4PublicV2ProviderAttempt: ...


class BfclV4PublicV2RequestBinder(Protocol):
    """Bind actual request bytes without exposing them to the journal core."""

    def request_payload_sha256(
        self,
        node: BfclV4PublicDevelopmentV2DagNode,
    ) -> str: ...


class BfclV4PublicV2CheckpointSink(Protocol):
    """Atomic external CAS sink for reservations and terminal snapshots."""

    def checkpoint(
        self,
        snapshot: BfclV4PublicV2JournalSnapshot,
        *,
        expected_previous_snapshot_fingerprint: str | None,
    ) -> None: ...


class BfclV4PublicV2PureAtBBatchGrader(Protocol):
    """Trusted adapter returning only the final 48-cell receipt projection."""

    def grade_pure_at_b_batch(
        self,
        request: BfclV4PublicV2PureAtBBatchGradeRequest,
    ) -> BfclV4PublicV2PureAtBBatchGradeProjection: ...


class BfclV4PublicV2TrustedBinaryGrader(Protocol):
    """Answer-side adapter exposing only opaque authority and grade projections."""

    def issue_evaluation_unlock(
        self,
        evidence: BfclV4PublicV2DecisionBarrierEvidence,
    ) -> BfclV4PublicV2EvaluationUnlock: ...

    def grade(
        self,
        node: BfclV4PublicDevelopmentV2DagNode,
        canonical_response: str,
        *,
        request_payload_sha256: str,
        provider_response_fingerprint: str,
        evaluation_unlock_fingerprint: str | None,
    ) -> BfclV4PublicV2TrustedGradeProjection: ...


_GRADABLE_KINDS = frozenset(
    {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        BfclV4PublicDevelopmentV2NodeKind.GATE,
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
    }
)

_EVALUATION_KINDS = frozenset(
    {
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
    }
)


class BfclV4PublicV2AppendOnlyJournal:
    """Process-local CAS writer whose portable snapshot is independently replayable."""

    def __init__(
        self,
        *,
        campaign: BfclV4PublicDevelopmentV2CampaignPlan | None = None,
        runtime_fingerprint: str,
        semantic_release_fingerprint: str,
        checkpoint_sink: BfclV4PublicV2CheckpointSink | None = None,
        resume_snapshot: BfclV4PublicV2JournalSnapshot | None = None,
    ) -> None:
        self._campaign = _checked_campaign(campaign)
        self._runtime_fingerprint = runtime_fingerprint
        self._semantic_release_fingerprint = semantic_release_fingerprint
        if resume_snapshot is None:
            events: tuple[BfclV4PublicV2JournalEvent, ...] = ()
            pending = None
            checkpoint_fingerprint = None
        else:
            try:
                checked = BfclV4PublicV2JournalSnapshot.model_validate(
                    resume_snapshot,
                    strict=True,
                )
                replay_bfcl_v4_public_v2_journal(
                    checked,
                    campaign=self._campaign,
                    require_complete=False,
                )
            except Exception as exc:
                raise BfclV4PublicV2ReplayError("resume snapshot failed verification") from exc
            if (
                checked.runtime_fingerprint != runtime_fingerprint
                or checked.semantic_release_fingerprint != semantic_release_fingerprint
            ):
                raise BfclV4PublicV2ReplayError("resume snapshot runtime or release changed")
            events = checked.events
            pending = checked.pending_call_reservation
            checkpoint_fingerprint = checked.fingerprint
        self._events = events
        self._pending_call_reservation = pending
        self._checkpoint_sink = checkpoint_sink
        self._checkpoint_fingerprint = checkpoint_fingerprint
        self._lock = RLock()

    @property
    def events(self) -> tuple[BfclV4PublicV2JournalEvent, ...]:
        with self._lock:
            return self._events

    @property
    def tail_event_sha256(self) -> str | None:
        with self._lock:
            return self._events[-1].fingerprint if self._events else None

    @property
    def pending_call_reservation(self) -> BfclV4PublicV2CallReservation | None:
        with self._lock:
            return self._pending_call_reservation

    def _snapshot(
        self,
        events: tuple[BfclV4PublicV2JournalEvent, ...],
        pending: BfclV4PublicV2CallReservation | None,
    ) -> BfclV4PublicV2JournalSnapshot:
        return BfclV4PublicV2JournalSnapshot(
            campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
            node_schedule_content_sha256=self._campaign.node_schedule_content_sha256,
            runtime_fingerprint=self._runtime_fingerprint,
            semantic_release_fingerprint=self._semantic_release_fingerprint,
            events=events,
            tail_event_sha256=events[-1].fingerprint if events else None,
            pending_call_reservation=pending,
        )

    def _checkpoint(self, snapshot: BfclV4PublicV2JournalSnapshot) -> None:
        if self._checkpoint_sink is None:
            return
        try:
            self._checkpoint_sink.checkpoint(
                snapshot,
                expected_previous_snapshot_fingerprint=self._checkpoint_fingerprint,
            )
        except Exception as exc:
            raise BfclV4PublicV2CheckpointError("durable journal checkpoint failed") from exc
        self._checkpoint_fingerprint = snapshot.fingerprint

    def reserve_call(
        self,
        *,
        node: BfclV4PublicDevelopmentV2DagNode,
        request: BfclV4PublicV2ProviderRequest,
    ) -> BfclV4PublicV2CallReservation:
        """Durably reserve the exact next call before crossing the provider boundary."""

        with self._lock:
            if self._pending_call_reservation is not None:
                raise BfclV4PublicV2CheckpointError("journal already has a pending call")
            sequence = len(self._events)
            if sequence >= len(self._campaign.nodes) or self._campaign.nodes[sequence] != node:
                raise BfclV4PublicV2ReplayError("call reservation is not the exact next DAG node")
            reservation = BfclV4PublicV2CallReservation(
                sequence=sequence,
                expected_tail_event_sha256=(self._events[-1].fingerprint if self._events else None),
                node=node,
                node_reference_sha256=canonical_sha256(node),
                request=request,
                request_fingerprint=request.fingerprint,
            )
            prospective = self._snapshot(self._events, reservation)
            self._checkpoint(prospective)
            self._pending_call_reservation = reservation
            return reservation

    def append(
        self,
        event: BfclV4PublicV2JournalEvent,
        *,
        expected_tail_event_sha256: str | None,
    ) -> str:
        with self._lock:
            current_tail = self._events[-1].fingerprint if self._events else None
            if expected_tail_event_sha256 != current_tail:
                raise BfclV4PublicV2StaleTailError("expected journal tail is stale")
            try:
                checked = BfclV4PublicV2JournalEvent.model_validate(event, strict=True)
            except Exception as exc:
                raise BfclV4PublicV2ReplayError("journal event failed strict validation") from exc
            _validate_event(
                self._campaign,
                self._events,
                checked,
                runtime_fingerprint=self._runtime_fingerprint,
                semantic_release_fingerprint=self._semantic_release_fingerprint,
            )
            pending = self._pending_call_reservation
            if pending is not None and (
                checked.event_kind is not BfclV4PublicV2EventKind.CALL
                or checked.node_id != pending.node.node_id
                or checked.request_fingerprint != pending.request_fingerprint
                or checked.request_payload_sha256 != pending.request.request_payload_sha256
            ):
                raise BfclV4PublicV2ReplayError("terminal event differs from pending reservation")
            if (
                pending is None
                and self._checkpoint_sink is not None
                and checked.event_kind is BfclV4PublicV2EventKind.CALL
            ):
                raise BfclV4PublicV2CheckpointError(
                    "durable call event lacks a pre-call reservation"
                )
            prospective_events = (*self._events, checked)
            prospective = self._snapshot(prospective_events, None)
            self._checkpoint(prospective)
            self._events = prospective_events
            self._pending_call_reservation = None
            return checked.fingerprint

    def snapshot(self) -> BfclV4PublicV2JournalSnapshot:
        with self._lock:
            return self._snapshot(self._events, self._pending_call_reservation)


def _provider_attempt(
    *,
    provider: BfclV4PublicV2ProviderTransport,
    request: BfclV4PublicV2ProviderRequest,
    node: BfclV4PublicDevelopmentV2DagNode,
) -> BfclV4PublicV2ProviderAttempt:
    try:
        raw_attempt = provider.execute(request, node)
        return BfclV4PublicV2ProviderAttempt.model_validate(raw_attempt, strict=True)
    except Exception:
        return BfclV4PublicV2ProviderAttempt(
            disposition=BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE,
            proposal_disposition=(
                BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
                else None
            ),
        )


def _decision_barrier_evidence(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    semantic_release_fingerprint: str,
) -> BfclV4PublicV2DecisionBarrierEvidence:
    decision_nodes = tuple(
        node
        for node in campaign.nodes
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
    )
    by_node = {event.node_id: event for event in events}
    try:
        decision_events = tuple(by_node[node.node_id] for node in decision_nodes)
    except KeyError as exc:
        raise BfclV4PublicV2ExecutorError(
            "evaluation authorization precedes the complete decision barrier"
        ) from exc
    if len(decision_nodes) != 6 or any(
        event.event_kind is not BfclV4PublicV2EventKind.DECISION for event in decision_events
    ):
        raise BfclV4PublicV2ExecutorError("evaluation authorization lacks six decisions")
    return BfclV4PublicV2DecisionBarrierEvidence(
        semantic_release_fingerprint=semantic_release_fingerprint,
        decision_node_references=tuple(canonical_sha256(node) for node in decision_nodes),
        decision_event_fingerprints=tuple(event.fingerprint for event in decision_events),
        final_decision_event_fingerprint=decision_events[-1].fingerprint,
    )


def _authorize_evaluation(
    *,
    grader: BfclV4PublicV2TrustedBinaryGrader,
    evidence: BfclV4PublicV2DecisionBarrierEvidence,
) -> BfclV4PublicV2EvaluationUnlock:
    try:
        unlock = BfclV4PublicV2EvaluationUnlock.model_validate(
            grader.issue_evaluation_unlock(evidence),
            strict=True,
        )
    except Exception as exc:
        raise BfclV4PublicV2ExecutorError("trusted evaluation authorization failed") from exc
    if unlock.barrier_evidence != evidence:
        raise BfclV4PublicV2ExecutorError("evaluation unlock binds another decision barrier")
    return unlock


def _trusted_grade(
    *,
    grader: BfclV4PublicV2TrustedBinaryGrader,
    node: BfclV4PublicDevelopmentV2DagNode,
    request: BfclV4PublicV2ProviderRequest,
    attempt: BfclV4PublicV2ProviderAttempt,
) -> BfclV4PublicV2TrustedGradeProjection:
    assert attempt.canonical_response is not None
    assert attempt.provider_response_fingerprint is not None
    try:
        projection = BfclV4PublicV2TrustedGradeProjection.model_validate(
            grader.grade(
                node,
                attempt.canonical_response,
                request_payload_sha256=request.request_payload_sha256,
                provider_response_fingerprint=attempt.provider_response_fingerprint,
                evaluation_unlock_fingerprint=request.evaluation_unlock_fingerprint,
            ),
            strict=True,
        )
    except Exception as exc:
        raise BfclV4PublicV2ExecutorError("trusted binary grading failed") from exc
    observed = (
        projection.request_payload_sha256,
        projection.provider_response_fingerprint,
        projection.evaluation_unlock_fingerprint,
    )
    expected = (
        request.request_payload_sha256,
        attempt.provider_response_fingerprint,
        request.evaluation_unlock_fingerprint,
    )
    if observed != expected:
        raise BfclV4PublicV2ExecutorError("trusted grade projection binds another call")
    return projection


def _trusted_pure_at_b_grade(
    *,
    grader: BfclV4PublicV2PureAtBBatchGrader,
    request: BfclV4PublicV2PureAtBBatchGradeRequest,
    aggregations: tuple,
) -> BfclV4PublicV2PureAtBBatchGradeProjection:
    try:
        projection = BfclV4PublicV2PureAtBBatchGradeProjection.model_validate(
            grader.grade_pure_at_b_batch(request),
            strict=True,
        )
    except Exception as exc:
        raise BfclV4PublicV2ExecutorError("trusted PURE@B batch grading failed") from exc
    if (
        projection.batch_grade_request_fingerprint != request.fingerprint
        or projection.decision_barrier_evidence_fingerprint
        != request.decision_barrier_evidence_fingerprint
        or projection.evaluation_unlock_fingerprint != request.evaluation_unlock_fingerprint
    ):
        raise BfclV4PublicV2ExecutorError("PURE@B batch projection binds another request")
    expected_cells = tuple(
        (
            request_cell.outer_seed_u64,
            request_cell.task_ref,
            aggregation.fingerprint,
            request_cell.fingerprint,
        )
        for request_cell, aggregation in zip(request.cells, aggregations, strict=True)
    )
    observed_cells = tuple(
        (
            cell.outer_seed_u64,
            cell.task_ref,
            cell.aggregation_record_fingerprint,
            cell.cell_grade_request_fingerprint,
        )
        for cell in projection.cells
    )
    if observed_cells != expected_cells:
        raise BfclV4PublicV2ExecutorError("PURE@B cell projections changed order or lineage")
    return projection


def _call_event(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    prefix: tuple[BfclV4PublicV2JournalEvent, ...],
    node: BfclV4PublicDevelopmentV2DagNode,
    request: BfclV4PublicV2ProviderRequest,
    attempt: BfclV4PublicV2ProviderAttempt,
    grade: BfclV4PublicV2TrustedGradeProjection | None,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> BfclV4PublicV2JournalEvent:
    proposal_disposition = attempt.proposal_disposition
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL:
        proposal_disposition = (
            BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
            if attempt.disposition is not BfclV4PublicV2AttemptDisposition.SUCCEEDED
            else proposal_disposition
        )
    return BfclV4PublicV2JournalEvent(
        sequence=node.node_slot,
        previous_event_sha256=prefix[-1].fingerprint if prefix else None,
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        node_id=node.node_id,
        node_slot=node.node_slot,
        node_reference_sha256=canonical_sha256(node),
        event_kind=BfclV4PublicV2EventKind.CALL,
        request_fingerprint=request.fingerprint,
        request_payload_sha256=request.request_payload_sha256,
        provider_attempt_disposition=attempt.disposition,
        provider_attempts_consumed=attempt.provider_attempts_consumed,
        executed_harness_variant=_expected_executed_variant(campaign, prefix, node),
        canonical_response=attempt.canonical_response,
        provider_response_fingerprint=attempt.provider_response_fingerprint,
        binary_grade=(
            grade.correct if grade is not None else False if node.kind in _GRADABLE_KINDS else None
        ),
        trusted_grade_request_fingerprint=(
            None if grade is None else grade.trusted_grade_request_fingerprint
        ),
        trusted_grader_receipt_fingerprint=(
            None if grade is None else grade.trusted_grader_receipt_fingerprint
        ),
        trusted_grade_attempts_consumed=0 if grade is None else 1,
        decision_barrier_evidence_fingerprint=(request.decision_barrier_evidence_fingerprint),
        evaluation_unlock_fingerprint=request.evaluation_unlock_fingerprint,
        proposal_disposition=proposal_disposition,
        candidate_artifact_sha256=attempt.candidate_artifact_sha256,
    )


def execute_bfcl_v4_public_v2_rehearsal(
    *,
    provider: BfclV4PublicV2ProviderTransport,
    request_binder: BfclV4PublicV2RequestBinder,
    grader: BfclV4PublicV2TrustedBinaryGrader,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan | None = None,
    pure_at_b_batch_grader: BfclV4PublicV2PureAtBBatchGrader | None = None,
    checkpoint_sink: BfclV4PublicV2CheckpointSink | None = None,
    resume_snapshot: BfclV4PublicV2JournalSnapshot | None = None,
) -> BfclV4PublicV2ExecutionReceipt:
    """Consume every frozen node once using injected provider and grader boundaries."""

    checked_campaign = _checked_campaign(campaign)
    journal = BfclV4PublicV2AppendOnlyJournal(
        campaign=checked_campaign,
        runtime_fingerprint=runtime_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        checkpoint_sink=checkpoint_sink,
        resume_snapshot=resume_snapshot,
    )
    evaluation_authority: BfclV4PublicV2EvaluationUnlock | None = None
    for node in checked_campaign.nodes[len(journal.events) :]:
        prefix = journal.events
        if node.consumes_model_call:
            if node.kind in _EVALUATION_KINDS and evaluation_authority is None:
                evidence = _decision_barrier_evidence(
                    campaign=checked_campaign,
                    events=prefix,
                    semantic_release_fingerprint=semantic_release_fingerprint,
                )
                evaluation_authority = _authorize_evaluation(
                    grader=grader,
                    evidence=evidence,
                )
            pending = journal.pending_call_reservation
            if pending is not None:
                if pending.node != node:
                    raise BfclV4PublicV2ReplayError("pending call is not the next DAG node")
                request = pending.request
                attempt = BfclV4PublicV2ProviderAttempt(
                    disposition=BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN,
                    proposal_disposition=(
                        BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE
                        if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL
                        else None
                    ),
                    provider_attempts_consumed=0,
                )
            else:
                payload_hash = request_binder.request_payload_sha256(node)
                request = _request(
                    checked_campaign,
                    node,
                    runtime_fingerprint=runtime_fingerprint,
                    semantic_release_fingerprint=semantic_release_fingerprint,
                    request_payload_sha256=payload_hash,
                    decision_barrier_evidence_fingerprint=(
                        None
                        if evaluation_authority is None
                        else evaluation_authority.barrier_evidence_fingerprint
                    ),
                    evaluation_unlock_fingerprint=(
                        None if evaluation_authority is None else evaluation_authority.fingerprint
                    ),
                )
                journal.reserve_call(node=node, request=request)
                attempt = _provider_attempt(provider=provider, request=request, node=node)
            if node.kind in _EVALUATION_KINDS and (
                evaluation_authority is None
                or request.decision_barrier_evidence_fingerprint
                != evaluation_authority.barrier_evidence_fingerprint
                or request.evaluation_unlock_fingerprint != evaluation_authority.fingerprint
            ):
                raise BfclV4PublicV2ReplayError("evaluation request differs from reissued unlock")
            gradable = node.kind in _GRADABLE_KINDS
            if gradable and (
                attempt.disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
                and attempt.canonical_response is not None
            ):
                grade = _trusted_grade(
                    grader=grader,
                    node=node,
                    request=request,
                    attempt=attempt,
                )
            else:
                grade = None
            event = _call_event(
                campaign=checked_campaign,
                prefix=prefix,
                node=node,
                request=request,
                attempt=attempt,
                grade=grade,
                runtime_fingerprint=runtime_fingerprint,
                semantic_release_fingerprint=semantic_release_fingerprint,
            )
        else:
            event = _build_control_event(
                checked_campaign,
                prefix,
                node,
                runtime_fingerprint=runtime_fingerprint,
                semantic_release_fingerprint=semantic_release_fingerprint,
            )
        journal.append(event, expected_tail_event_sha256=journal.tail_event_sha256)

    snapshot = journal.snapshot()
    state = replay_bfcl_v4_public_v2_journal(
        snapshot,
        campaign=checked_campaign,
        require_complete=True,
    )
    batch_grade = None
    if pure_at_b_batch_grader is not None:
        if evaluation_authority is None:
            evidence = _decision_barrier_evidence(
                campaign=checked_campaign,
                events=snapshot.events,
                semantic_release_fingerprint=semantic_release_fingerprint,
            )
            evaluation_authority = _authorize_evaluation(grader=grader, evidence=evidence)
        batch_request = build_bfcl_v4_public_v2_pure_at_b_batch_grade_request(
            campaign=checked_campaign,
            events=snapshot.events,
            aggregations=state.pure_at_b_aggregations,
            evaluation_unlock=evaluation_authority,
            semantic_release_fingerprint=semantic_release_fingerprint,
        )
        batch_grade = _trusted_pure_at_b_grade(
            grader=pure_at_b_batch_grader,
            request=batch_request,
            aggregations=state.pure_at_b_aggregations,
        )
    return BfclV4PublicV2ExecutionReceipt(
        journal=snapshot,
        replayed_state=state,
        pure_at_b_batch_grade=batch_grade,
    )


__all__ = [
    "BfclV4PublicV2AppendOnlyJournal",
    "BfclV4PublicV2CheckpointError",
    "BfclV4PublicV2CheckpointSink",
    "BfclV4PublicV2ExecutorError",
    "BfclV4PublicV2ProviderTransport",
    "BfclV4PublicV2PureAtBBatchGrader",
    "BfclV4PublicV2ReplayError",
    "BfclV4PublicV2RequestBinder",
    "BfclV4PublicV2StaleTailError",
    "BfclV4PublicV2TrustedBinaryGrader",
    "execute_bfcl_v4_public_v2_rehearsal",
    "replay_bfcl_v4_public_v2_journal",
]
