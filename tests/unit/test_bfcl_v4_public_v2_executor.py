from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2GateVariant,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2Stage,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2EvaluationUnlock,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_dispatch_contracts import (
    BfclV4PublicV2DispatchReceipt,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor import (
    BfclV4PublicV2AppendOnlyJournal,
    BfclV4PublicV2CheckpointError,
    BfclV4PublicV2ExecutorError,
    BfclV4PublicV2ReplayError,
    BfclV4PublicV2StaleTailError,
    execute_bfcl_v4_public_v2_rehearsal,
    replay_bfcl_v4_public_v2_journal,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2ControlValue,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2JournalSnapshot,
    BfclV4PublicV2ProposalDisposition,
    BfclV4PublicV2ProviderAttempt,
    BfclV4PublicV2PureAtBAggregationRecord,
    BfclV4PublicV2PureAtBBatchGradeProjection,
    BfclV4PublicV2PureAtBCellGradeProjection,
    BfclV4PublicV2TrustedGradeProjection,
)

RUNTIME = "a" * 64
SEMANTIC_RELEASE = "b" * 64


@dataclass
class _RequestBinder:
    calls: int = 0

    def bind(
        self,
        *,
        node: BfclV4PublicDevelopmentV2DagNode,
        context,
    ) -> BfclV4PublicV2DispatchReceipt:
        self.calls += 1
        request = canonical_sha256(
            {
                "domain": "bfcl-v2-test-request/v1",
                "node_id": node.node_id,
                "provider_seed_u63": node.provider_seed_u63,
            }
        )
        return BfclV4PublicV2DispatchReceipt(
            node=node,
            node_reference_sha256=canonical_sha256(node),
            journal_prefix_fingerprint=context.journal_prefix_fingerprint,
            journal_prefix_event_count=context.journal_prefix_event_count,
            journal_prefix_tail_event_sha256=context.journal_prefix_tail_event_sha256,
            campaign_plan_fingerprint=context.campaign_plan_fingerprint,
            node_schedule_content_sha256=context.node_schedule_content_sha256,
            runtime_fingerprint=context.runtime_fingerprint,
            semantic_release_fingerprint=context.semantic_release_fingerprint,
            request_materialization_fingerprint=canonical_sha256(
                {"domain": "bfcl-v2-test-materialization/v1", "request": request}
            ),
            native_request_fingerprint=request,
            request_payload_sha256=request,
            proposal_batch_set_fingerprint=context.proposal_batch_set_fingerprint,
        )


@dataclass
class _Provider:
    calls: int = 0

    def execute(self, request, node: BfclV4PublicDevelopmentV2DagNode, dispatch):
        self.calls += 1
        assert dispatch.node == node
        assert dispatch.request_payload_sha256 == request.request_payload_sha256
        if (
            node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
            and node.sample_index == 6
        ):
            return BfclV4PublicV2ProviderAttempt(
                dispatch_fingerprint=dispatch.fingerprint,
                disposition=BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE,
            )
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.PROPOSAL:
            response = f"typed-proposal-{node.candidate_index}"
            return BfclV4PublicV2ProviderAttempt(
                dispatch_fingerprint=dispatch.fingerprint,
                disposition=BfclV4PublicV2AttemptDisposition.SUCCEEDED,
                canonical_response=response,
                provider_response_fingerprint=canonical_sha256(
                    {"node_id": node.node_id, "response": response}
                ),
                proposal_disposition=BfclV4PublicV2ProposalDisposition.VALID,
                candidate_artifact_sha256=canonical_sha256(
                    {"candidate": node.candidate_index, "node": node.node_id}
                ),
            )
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE and node.gate_variant in {
            BfclV4PublicDevelopmentV2GateVariant.PARENT,
            BfclV4PublicDevelopmentV2GateVariant.REVERT,
            BfclV4PublicDevelopmentV2GateVariant.NEGATIVE_CONTROL,
        }:
            response = (
                f"shadow:{node.outer_seed_u64}:{node.arm}:{node.task_ref}:{node.rollout_index}"
            )
        elif node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE:
            response = (
                f"modal:{node.outer_seed_u64}:{node.task_ref}"
                if node.sample_index is not None and node.sample_index < 4
                else f"minority:{node.outer_seed_u64}:{node.task_ref}:{node.sample_index}"
            )
        else:
            response = f"response:{node.node_id}"
        return BfclV4PublicV2ProviderAttempt(
            dispatch_fingerprint=dispatch.fingerprint,
            disposition=BfclV4PublicV2AttemptDisposition.SUCCEEDED,
            canonical_response=response,
            provider_response_fingerprint=canonical_sha256(
                {"node_id": node.node_id, "response": response}
            ),
        )


@dataclass
class _Grader:
    calls: int = 0
    authorizations: int = 0

    def issue_evaluation_unlock(self, capability):
        self.authorizations += 1
        evidence = capability.receipt.evidence
        return BfclV4PublicV2EvaluationUnlock(
            barrier_evidence=evidence,
            barrier_evidence_fingerprint=evidence.fingerprint,
            verified_barrier_receipt_fingerprint=capability.receipt.fingerprint,
            authority_key_id="f" * 64,
            authentication_tag_hmac_sha256="e" * 64,
        )

    def grade(
        self,
        node: BfclV4PublicDevelopmentV2DagNode,
        canonical_response: str,
        *,
        request_payload_sha256: str,
        provider_response_fingerprint: str,
        evaluation_unlock_fingerprint: str | None,
    ):
        self.calls += 1
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT:
            correct = node.candidate_index == 0
        elif node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE:
            correct = node.gate_variant is BfclV4PublicDevelopmentV2GateVariant.NOMINATED_CANDIDATE
        elif node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE:
            correct = canonical_response.startswith("modal:")
        else:
            correct = False
        grade_request = canonical_sha256(
            {
                "node_id": node.node_id,
                "request": request_payload_sha256,
                "response": provider_response_fingerprint,
                "unlock": evaluation_unlock_fingerprint,
            }
        )
        return BfclV4PublicV2TrustedGradeProjection(
            correct=correct,
            request_payload_sha256=request_payload_sha256,
            provider_response_fingerprint=provider_response_fingerprint,
            trusted_grade_request_fingerprint=grade_request,
            trusted_grader_receipt_fingerprint=canonical_sha256(
                {"grade_request": grade_request, "correct": correct}
            ),
            evaluation_unlock_fingerprint=evaluation_unlock_fingerprint,
        )


@dataclass
class _BatchGrader:
    calls: int = 0

    def grade_pure_at_b_batch(self, request):
        self.calls += 1
        cells = tuple(
            BfclV4PublicV2PureAtBCellGradeProjection(
                outer_seed_u64=cell.outer_seed_u64,
                task_ref=cell.task_ref,
                aggregation_record_fingerprint=BfclV4PublicV2PureAtBAggregationRecord(
                    outer_seed_u64=cell.outer_seed_u64,
                    task_ref=cell.task_ref,
                    source_event_sha256=cell.source_event_sha256,
                    result=cell.aggregation_result,
                ).fingerprint,
                cell_grade_request_fingerprint=cell.fingerprint,
                cell_grade_receipt_fingerprint=canonical_sha256(
                    {"cell_request": cell.fingerprint, "correct": True}
                ),
                correct=True,
            )
            for cell in request.cells
        )
        return BfclV4PublicV2PureAtBBatchGradeProjection(
            batch_grade_request_fingerprint=request.fingerprint,
            batch_grade_receipt_fingerprint=canonical_sha256(
                {"batch_request": request.fingerprint, "cells": cells}
            ),
            decision_barrier_evidence_fingerprint=(request.decision_barrier_evidence_fingerprint),
            evaluation_unlock_fingerprint=request.evaluation_unlock_fingerprint,
            cells=cells,
            correct_count=48,
        )


@dataclass
class _CheckpointSink:
    fail_on_call: int | None = None
    calls: int = 0
    snapshots: list[BfclV4PublicV2JournalSnapshot] = field(default_factory=list)

    def checkpoint(
        self,
        snapshot: BfclV4PublicV2JournalSnapshot,
        *,
        expected_previous_snapshot_fingerprint: str | None,
    ) -> None:
        expected = self.snapshots[-1].fingerprint if self.snapshots else None
        assert expected_previous_snapshot_fingerprint == expected
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected checkpoint failure")
        self.snapshots.append(snapshot)


@dataclass
class _ExplodingBinder:
    calls: int = 0

    def bind(self, *, node, context):
        self.calls += 1
        raise RuntimeError(f"stop after recovered node: {node.node_id}")


@pytest.fixture(scope="module")
def execution():
    provider = _Provider()
    binder = _RequestBinder()
    grader = _Grader()
    batch_grader = _BatchGrader()
    receipt = execute_bfcl_v4_public_v2_rehearsal(
        provider=provider,
        request_binder=binder,
        grader=grader,
        runtime_fingerprint=RUNTIME,
        semantic_release_fingerprint=SEMANTIC_RELEASE,
        pure_at_b_batch_grader=batch_grader,
    )
    assert batch_grader.calls == 1
    return receipt, provider, binder, grader


def _rechain(
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> tuple[BfclV4PublicV2JournalEvent, ...]:
    result: list[BfclV4PublicV2JournalEvent] = []
    previous: str | None = None
    for sequence, event in enumerate(events):
        payload = event.model_dump(mode="python")
        payload["sequence"] = sequence
        payload["previous_event_sha256"] = previous
        rebuilt = BfclV4PublicV2JournalEvent.model_validate(payload, strict=True)
        result.append(rebuilt)
        previous = rebuilt.fingerprint
    return tuple(result)


def _snapshot(
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> BfclV4PublicV2JournalSnapshot:
    campaign = build_bfcl_v4_public_development_v2_campaign_plan()
    return BfclV4PublicV2JournalSnapshot(
        campaign_plan_fingerprint=BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        runtime_fingerprint=RUNTIME,
        semantic_release_fingerprint=SEMANTIC_RELEASE,
        events=events,
        tail_event_sha256=events[-1].fingerprint if events else None,
    )


def test_full_stage_major_rehearsal_consumes_exact_frozen_dag(execution) -> None:
    receipt, provider, binder, grader = execution
    state = receipt.replayed_state
    events = receipt.journal.events

    assert len(events) == 1_098
    assert state.completed_node_count == 1_098
    assert state.burned_call_slot_count == 1_086
    assert state.succeeded_call_count == 1_044
    assert state.failed_call_count == 42
    assert provider.calls == binder.calls == 1_086
    assert state.provider_attempt_count == 1_086
    assert state.crash_recovery_burn_count == 0
    assert grader.calls == 720
    assert grader.authorizations == 1
    assert len(state.nominations) == len(state.decisions) == 6
    assert {item.value for item in state.nominations} == {BfclV4PublicV2ControlValue.CANDIDATE_0}
    assert {item.value for item in state.decisions} == {BfclV4PublicV2ControlValue.PROMOTE}
    assert len(state.pure_at_b_aggregations) == 48
    assert state.retries_used == state.backfills_used == state.adaptive_stops_used == 0
    assert state.complete is True
    assert receipt.real_api_called_by_core is False
    assert receipt.score_bearing_execution is False

    batch = receipt.pure_at_b_batch_grade
    assert batch is not None
    assert len(batch.cells) == 48
    assert batch.source_event_count == 330
    assert batch.trusted_grade_attempt_count == 48
    assert batch.individual_sample_grade_count == 0


def test_event_order_has_global_stage_barriers_and_exact_control_counts(execution) -> None:
    receipt, *_ = execution
    campaign = build_bfcl_v4_public_development_v2_campaign_plan()
    assert tuple(event.node_id for event in receipt.journal.events) == tuple(
        node.node_id for node in campaign.nodes
    )
    assert tuple(event.sequence for event in receipt.journal.events) == tuple(range(1_098))
    stages = tuple(campaign.nodes[event.node_slot].stage for event in receipt.journal.events)
    assert stages == tuple(sorted(stages, key=lambda stage: stage.value))
    assert (
        sum(
            event.event_kind is BfclV4PublicV2EventKind.NOMINATION
            for event in receipt.journal.events
        )
        == 6
    )
    assert (
        sum(
            event.event_kind is BfclV4PublicV2EventKind.DECISION for event in receipt.journal.events
        )
        == 6
    )
    assert (
        sum(
            campaign.nodes[event.node_slot].stage is BfclV4PublicDevelopmentV2Stage.GATE
            for event in receipt.journal.events
        )
        == 288
    )


def test_every_event_binds_plan_node_catalog_runtime_request_and_release(execution) -> None:
    receipt, *_ = execution
    campaign = build_bfcl_v4_public_development_v2_campaign_plan()
    for event, node in zip(receipt.journal.events, campaign.nodes, strict=True):
        assert event.campaign_plan_fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        assert event.node_schedule_content_sha256 == campaign.node_schedule_content_sha256
        assert event.mutation_catalog_fingerprint == campaign.mutation_catalog_fingerprint
        assert event.runtime_fingerprint == RUNTIME
        assert event.semantic_release_fingerprint == SEMANTIC_RELEASE
        assert event.node_id == node.node_id
        assert event.node_reference_sha256 == canonical_sha256(node)
        assert event.request_fingerprint
        assert event.request_payload_sha256
        assert event.semantic_release_authenticity_attested is False
        assert event.task_payload_present is False
        assert event.possible_answer_present is False

        if node.kind in {
            BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
            BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
        }:
            assert event.decision_barrier_evidence_fingerprint
            assert event.evaluation_unlock_fingerprint
        else:
            assert event.decision_barrier_evidence_fingerprint is None
            assert event.evaluation_unlock_fingerprint is None

        gradable_success = (
            node.kind
            in {
                BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
                BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
                BfclV4PublicDevelopmentV2NodeKind.GATE,
                BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
            }
            and event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
        )
        assert bool(event.trusted_grader_receipt_fingerprint) is gradable_success


def test_pure_at_b_modal_aggregation_counts_failed_slots_as_no_response(execution) -> None:
    receipt, *_ = execution
    campaign = build_bfcl_v4_public_development_v2_campaign_plan()
    pure_events = tuple(
        event
        for event, node in zip(receipt.journal.events, campaign.nodes, strict=True)
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
    )
    assert len(pure_events) == 330
    assert all(event.binary_grade is None for event in pure_events)
    assert all(event.trusted_grade_attempts_consumed == 0 for event in pure_events)
    for record in receipt.replayed_state.pure_at_b_aggregations:
        assert record.result.selected_canonical_response == (
            f"modal:{record.outer_seed_u64}:{record.task_ref}"
        )
        assert record.result.modal_count == 4
        assert len(record.source_event_sha256) in {6, 7}


def test_independent_replay_exactly_reconstructs_terminal_state(execution) -> None:
    receipt, *_ = execution
    replayed = replay_bfcl_v4_public_v2_journal(receipt.journal)
    assert replayed == receipt.replayed_state


def test_replay_rejects_missing_duplicate_reordered_and_lineage_tampering(execution) -> None:
    receipt, *_ = execution
    events = receipt.journal.events

    with pytest.raises(BfclV4PublicV2ReplayError, match="missing"):
        replay_bfcl_v4_public_v2_journal(_snapshot(events[:-1]))

    duplicate = _rechain((*events[:10], events[9], *events[11:]))
    with pytest.raises(BfclV4PublicV2ReplayError):
        replay_bfcl_v4_public_v2_journal(_snapshot(duplicate))

    reordered = _rechain((*events[:20], events[21], events[20], *events[22:]))
    with pytest.raises(BfclV4PublicV2ReplayError):
        replay_bfcl_v4_public_v2_journal(_snapshot(reordered))

    payload = events[30].model_dump(mode="python")
    payload["runtime_fingerprint"] = "c" * 64
    tampered = BfclV4PublicV2JournalEvent.model_validate(payload, strict=True)
    lineage_tamper = _rechain((*events[:30], tampered, *events[31:]))
    with pytest.raises(BfclV4PublicV2ReplayError):
        replay_bfcl_v4_public_v2_journal(_snapshot(lineage_tamper))


def test_replay_rejects_illegal_promotion_even_after_attacker_rechains(execution) -> None:
    receipt, *_ = execution
    events = receipt.journal.events
    decision_index = next(
        index
        for index, event in enumerate(events)
        if event.event_kind is BfclV4PublicV2EventKind.DECISION
    )
    payload = events[decision_index].model_dump(mode="python")
    payload["control_value"] = BfclV4PublicV2ControlValue.PARENT_FALLBACK
    illegal = BfclV4PublicV2JournalEvent.model_validate(payload, strict=True)
    rechained = _rechain((*events[:decision_index], illegal, *events[decision_index + 1 :]))
    with pytest.raises(BfclV4PublicV2ReplayError, match="promotion"):
        replay_bfcl_v4_public_v2_journal(_snapshot(rechained))


def test_append_only_journal_rejects_stale_tail(execution) -> None:
    receipt, *_ = execution
    journal = BfclV4PublicV2AppendOnlyJournal(
        runtime_fingerprint=RUNTIME,
        semantic_release_fingerprint=SEMANTIC_RELEASE,
    )
    with pytest.raises(BfclV4PublicV2StaleTailError):
        journal.append(
            receipt.journal.events[0],
            expected_tail_event_sha256="d" * 64,
        )


def test_failed_reservation_checkpoint_never_crosses_provider_boundary() -> None:
    provider = _Provider()
    binder = _RequestBinder()
    sink = _CheckpointSink(fail_on_call=1)

    with pytest.raises(BfclV4PublicV2CheckpointError):
        execute_bfcl_v4_public_v2_rehearsal(
            provider=provider,
            request_binder=binder,
            grader=_Grader(),
            runtime_fingerprint=RUNTIME,
            semantic_release_fingerprint=SEMANTIC_RELEASE,
            checkpoint_sink=sink,
        )

    assert binder.calls == 1
    assert provider.calls == 0
    assert sink.calls == 1
    assert sink.snapshots == []


def test_resume_burns_pending_slot_without_repeating_provider_attempt() -> None:
    provider = _Provider()
    sink = _CheckpointSink(fail_on_call=2)

    with pytest.raises(BfclV4PublicV2CheckpointError):
        execute_bfcl_v4_public_v2_rehearsal(
            provider=provider,
            request_binder=_RequestBinder(),
            grader=_Grader(),
            runtime_fingerprint=RUNTIME,
            semantic_release_fingerprint=SEMANTIC_RELEASE,
            checkpoint_sink=sink,
        )

    assert provider.calls == 1
    assert len(sink.snapshots) == 1
    resume_snapshot = sink.snapshots[-1]
    assert resume_snapshot.events == ()
    assert resume_snapshot.pending_call_reservation is not None

    binder = _ExplodingBinder()
    with pytest.raises(
        BfclV4PublicV2ExecutorError,
        match="request binder failed exact materialization dispatch",
    ):
        execute_bfcl_v4_public_v2_rehearsal(
            provider=provider,
            request_binder=binder,
            grader=_Grader(),
            runtime_fingerprint=RUNTIME,
            semantic_release_fingerprint=SEMANTIC_RELEASE,
            checkpoint_sink=sink,
            resume_snapshot=resume_snapshot,
        )

    assert provider.calls == 1
    assert binder.calls == 1
    recovered = sink.snapshots[-1]
    assert len(recovered.events) == 1
    assert recovered.pending_call_reservation is None
    assert (
        recovered.events[0].provider_attempt_disposition
        is BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN
    )
    replayed = replay_bfcl_v4_public_v2_journal(recovered, require_complete=False)
    assert replayed.burned_call_slot_count == 1
    assert replayed.provider_attempt_count == 0
    assert replayed.crash_recovery_burn_count == 1
