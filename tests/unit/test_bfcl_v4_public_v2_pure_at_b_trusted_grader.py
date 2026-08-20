from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
    BfclV4PublicDevelopmentV2NodeKind,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_barrier_capability import (
    BfclV4PublicV2VerifiedDecisionBarrierReceipt,
    _mint_bfcl_v4_public_v2_verified_decision_barrier,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader import (
    grade_bfcl_v4_public_v2_pure_at_b_batch,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts import (
    BfclV4PublicV2PureAtBBatchGradeRequest,
    BfclV4PublicV2PureAtBCellGradeRequest,
    bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint,
    build_bfcl_v4_public_v2_pure_at_b_batch_grade_request,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader import (
    BfclV4PublicV2TrustedGrader,
    open_bfcl_v4_public_v2_trusted_grader,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_dispatch_contracts import (
    BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    BfclV4PublicV2DispatchReceipt,
    bfcl_v4_public_v2_journal_prefix_fingerprint,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2ProviderRequest,
    BfclV4PublicV2PureAtBAggregationRecord,
)

_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")
_SEMANTIC_RELEASE = "a" * 64
_RUNTIME_FINGERPRINT = "b" * 64
_AUTHORITY_SECRET = b"bfcl-v2-pure-at-b-test-authority-secret"
_KNOWN_HOLDOUT_00_RESPONSE = (
    '[{"arguments":{"patient_id":"546382","status":"concluded"},'
    '"function_name":"patient.get_mri_report"}]'
)


def _dispatch_receipt(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    node,
    *,
    request_payload_sha256: str,
    previous_event_sha256: str | None,
) -> BfclV4PublicV2DispatchReceipt:
    return BfclV4PublicV2DispatchReceipt(
        node=node,
        node_reference_sha256=canonical_sha256(node),
        journal_prefix_fingerprint=bfcl_v4_public_v2_journal_prefix_fingerprint(
            campaign_plan_fingerprint=campaign.fingerprint,
            node_schedule_content_sha256=campaign.node_schedule_content_sha256,
            runtime_fingerprint=_RUNTIME_FINGERPRINT,
            semantic_release_fingerprint=_SEMANTIC_RELEASE,
            event_count=node.node_slot,
            tail_event_sha256=previous_event_sha256,
        ),
        journal_prefix_event_count=node.node_slot,
        journal_prefix_tail_event_sha256=previous_event_sha256,
        campaign_plan_fingerprint=campaign.fingerprint,
        node_schedule_content_sha256=campaign.node_schedule_content_sha256,
        runtime_fingerprint=_RUNTIME_FINGERPRINT,
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
        request_materialization_fingerprint=canonical_sha256(
            {
                "domain": "bfcl-v2-test-materialization/v1",
                "node_id": node.node_id,
                "payload": request_payload_sha256,
            }
        ),
        native_request_fingerprint=request_payload_sha256,
        request_payload_sha256=request_payload_sha256,
        proposal_batch_set_fingerprint=BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT,
    )


@pytest.fixture(scope="module")
def campaign() -> BfclV4PublicDevelopmentV2CampaignPlan:
    return build_bfcl_v4_public_development_v2_campaign_plan()


@pytest.fixture(scope="module")
def grader(campaign: BfclV4PublicDevelopmentV2CampaignPlan) -> BfclV4PublicV2TrustedGrader:
    if not _PINNED_CHECKOUT.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for trusted grader integration")
    return open_bfcl_v4_public_v2_trusted_grader(
        _PINNED_CHECKOUT,
        campaign,
        evaluation_authority_secret=_AUTHORITY_SECRET,
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
    )


@pytest.fixture(scope="module")
def unlock(grader: BfclV4PublicV2TrustedGrader) -> BfclV4PublicV2EvaluationUnlock:
    decision_events = tuple(f"{index + 1:064x}" for index in range(6))
    evidence = BfclV4PublicV2DecisionBarrierEvidence(
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
        decision_node_references=grader.decision_node_references,
        decision_event_fingerprints=decision_events,
        final_decision_event_fingerprint=decision_events[-1],
    )
    # Test-only mint: this grader unit does not own the durable executor journal.
    verified_receipt = BfclV4PublicV2VerifiedDecisionBarrierReceipt(
        evidence=evidence,
        evidence_fingerprint=evidence.fingerprint,
        journal_snapshot_fingerprint=canonical_sha256("test-only-journal-snapshot"),
        journal_prefix_event_count=1_000,
        journal_tail_event_fingerprint=evidence.final_decision_event_fingerprint,
        runtime_fingerprint=_RUNTIME_FINGERPRINT,
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
        replay_state_fingerprint=canonical_sha256("test-only-independent-replay"),
    )
    capability = _mint_bfcl_v4_public_v2_verified_decision_barrier(verified_receipt)
    return grader.issue_evaluation_unlock(capability)


def _source_events(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    unlock: BfclV4PublicV2EvaluationUnlock,
    *,
    selected_response: str | None,
) -> tuple[BfclV4PublicV2JournalEvent, ...]:
    events: list[BfclV4PublicV2JournalEvent] = []
    campaign_fingerprint = campaign.fingerprint
    for node in campaign.nodes:
        if node.kind is not BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE:
            continue
        assert node.campaign_call_slot is not None
        assert node.provider_seed_u63 is not None
        response = (
            selected_response
            if node.outer_seed_u64 == BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64[0]
            and node.task_ref == "holdout-00"
            else None
        )
        payload_sha256 = canonical_sha256({"node_id": node.node_id})
        provider_request = BfclV4PublicV2ProviderRequest(
            campaign_plan_fingerprint=campaign_fingerprint,
            node_schedule_content_sha256=campaign.node_schedule_content_sha256,
            mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
            runtime_fingerprint=_RUNTIME_FINGERPRINT,
            semantic_release_fingerprint=_SEMANTIC_RELEASE,
            node_id=node.node_id,
            node_reference_sha256=canonical_sha256(node),
            campaign_call_slot=node.campaign_call_slot,
            provider_seed_u63=node.provider_seed_u63,
            request_payload_sha256=payload_sha256,
            decision_barrier_evidence_fingerprint=unlock.barrier_evidence_fingerprint,
            evaluation_unlock_fingerprint=unlock.fingerprint,
        )
        previous_event_sha256 = canonical_sha256(
            {"domain": "bfcl-v2-test-prefix-tail/v1", "node_slot": node.node_slot}
        )
        dispatch = _dispatch_receipt(
            campaign,
            node,
            request_payload_sha256=payload_sha256,
            previous_event_sha256=previous_event_sha256,
        )
        events.append(
            BfclV4PublicV2JournalEvent(
                sequence=node.node_slot,
                previous_event_sha256=previous_event_sha256,
                campaign_plan_fingerprint=campaign_fingerprint,
                node_schedule_content_sha256=campaign.node_schedule_content_sha256,
                mutation_catalog_fingerprint=campaign.mutation_catalog_fingerprint,
                runtime_fingerprint=_RUNTIME_FINGERPRINT,
                semantic_release_fingerprint=_SEMANTIC_RELEASE,
                node_id=node.node_id,
                node_slot=node.node_slot,
                node_reference_sha256=canonical_sha256(node),
                event_kind=BfclV4PublicV2EventKind.CALL,
                request_fingerprint=provider_request.fingerprint,
                request_payload_sha256=payload_sha256,
                dispatch_fingerprint=dispatch.fingerprint,
                journal_prefix_fingerprint=dispatch.journal_prefix_fingerprint,
                request_materialization_fingerprint=dispatch.request_materialization_fingerprint,
                native_request_fingerprint=dispatch.native_request_fingerprint,
                proposal_batch_set_fingerprint=dispatch.proposal_batch_set_fingerprint,
                provider_attempt_disposition=(
                    BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE
                    if response is None
                    else BfclV4PublicV2AttemptDisposition.SUCCEEDED
                ),
                provider_attempts_consumed=1,
                executed_harness_variant="bare",
                canonical_response=response,
                provider_response_fingerprint=(
                    None if response is None else canonical_sha256({"canonical_response": response})
                ),
                decision_barrier_evidence_fingerprint=(unlock.barrier_evidence_fingerprint),
                evaluation_unlock_fingerprint=unlock.fingerprint,
            )
        )
    assert len(events) == 330
    return tuple(events)


def _aggregations(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
) -> tuple[BfclV4PublicV2PureAtBAggregationRecord, ...]:
    by_node = {event.node_id: event for event in events}
    records: list[BfclV4PublicV2PureAtBAggregationRecord] = []
    for outer_seed in campaign.outer_seeds_u64:
        for allocation in campaign.pure_at_b_allocation:
            nodes = tuple(
                node
                for node in campaign.nodes
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
                and node.outer_seed_u64 == outer_seed
                and node.task_ref == allocation.task_ref
            )
            sources = tuple(by_node[node.node_id] for node in nodes)
            records.append(
                BfclV4PublicV2PureAtBAggregationRecord(
                    outer_seed_u64=outer_seed,
                    task_ref=allocation.task_ref,
                    source_event_sha256=tuple(event.fingerprint for event in sources),
                    result=aggregate_bfcl_v4_public_development_v2_pure_at_b(
                        tuple(event.canonical_response for event in sources)
                    ),
                )
            )
    assert len(records) == 48
    return tuple(records)


def _batch(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    unlock: BfclV4PublicV2EvaluationUnlock,
    *,
    selected_response: str | None,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    events = _source_events(campaign, unlock, selected_response=selected_response)
    return build_bfcl_v4_public_v2_pure_at_b_batch_grade_request(
        campaign=campaign,
        events=events,
        aggregations=_aggregations(campaign, events),
        evaluation_unlock=unlock,
        semantic_release_fingerprint=_SEMANTIC_RELEASE,
    )


@pytest.fixture(scope="module")
def no_response_batch(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    unlock: BfclV4PublicV2EvaluationUnlock,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    return _batch(campaign, unlock, selected_response=None)


@pytest.fixture(scope="module")
def selected_batch(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    unlock: BfclV4PublicV2EvaluationUnlock,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    return _batch(campaign, unlock, selected_response=_KNOWN_HOLDOUT_00_RESPONSE)


def _unchecked_replace(model: Any, **updates: Any) -> Any:
    content = {name: getattr(model, name) for name in type(model).model_fields}
    content.update(updates)
    return type(model).model_construct(**content)


def _replace_cell(
    batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    index: int,
    cell: BfclV4PublicV2PureAtBCellGradeRequest,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    cells = list(batch.cells)
    cells[index] = cell
    return _unchecked_replace(batch, cells=tuple(cells))


def _forbidden_worker(*args: Any, **kwargs: Any) -> bool:
    raise AssertionError("isolated answer worker must not run")


def test_incomplete_batch_is_rejected_before_any_worker(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    forged = _unchecked_replace(no_response_batch, cells=no_response_batch.cells[:-1])

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(grader, forged)


def test_wrong_source_reference_is_rejected_before_any_worker(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    cell = no_response_batch.cells[0]
    references = ("f" * 64, *cell.source_node_references[1:])
    forged = _replace_cell(
        no_response_batch,
        0,
        _unchecked_replace(cell, source_node_references=references),
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(grader, forged)


def test_wrong_six_or_seven_source_count_is_rejected(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> None:
    index = next(
        index
        for index, cell in enumerate(no_response_batch.cells)
        if cell.allocation.sample_count == 7
    )
    cell = no_response_batch.cells[index]
    forged_cell = _unchecked_replace(
        cell,
        source_nodes=cell.source_nodes[:-1],
        source_node_references=cell.source_node_references[:-1],
        source_events=cell.source_events[:-1],
        source_event_sha256=cell.source_event_sha256[:-1],
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(no_response_batch, index, forged_cell),
        )


def test_source_event_with_individual_grade_is_rejected(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> None:
    cell = no_response_batch.cells[0]
    event = _unchecked_replace(
        cell.source_events[0],
        binary_grade=False,
        trusted_grade_request_fingerprint="c" * 64,
        trusted_grader_receipt_fingerprint="d" * 64,
        trusted_grade_attempts_consumed=1,
    )
    events = (event, *cell.source_events[1:])
    forged_cell = _unchecked_replace(
        cell,
        source_events=events,
        source_event_sha256=tuple(item.fingerprint for item in events),
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(no_response_batch, 0, forged_cell),
        )


def test_source_event_with_wrong_harness_variant_is_rejected(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> None:
    cell = no_response_batch.cells[0]
    event = _unchecked_replace(
        cell.source_events[0],
        executed_harness_variant="selected-or-parent-fallback",
    )
    events = (event, *cell.source_events[1:])
    forged_cell = _unchecked_replace(
        cell,
        source_events=events,
        source_event_sha256=tuple(item.fingerprint for item in events),
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(no_response_batch, 0, forged_cell),
        )


def test_wrong_modal_result_is_rejected_before_any_worker(
    grader: BfclV4PublicV2TrustedGrader,
    selected_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    cell = selected_batch.cells[0]
    wrong = aggregate_bfcl_v4_public_development_v2_pure_at_b(
        (None,) * cell.allocation.sample_count
    )
    forged_cell = _unchecked_replace(
        cell,
        aggregation_result=wrong,
        aggregation_result_fingerprint=canonical_sha256(wrong),
        selected_canonical_response_fingerprint=(
            bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(None)
        ),
    )

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(selected_batch, 0, forged_cell),
        )


def test_tampered_unlock_is_rejected_before_any_worker(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    forged_unlock = _unchecked_replace(
        no_response_batch.evaluation_unlock,
        authentication_tag_hmac_sha256="0" * 64,
    )
    forged = _unchecked_replace(no_response_batch, evaluation_unlock=forged_unlock)

    with pytest.raises(ValueError, match="revalidation failed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(grader, forged)


def test_mixed_source_runtimes_are_rejected_before_any_worker(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    cell = no_response_batch.cells[0]
    event = _unchecked_replace(cell.source_events[0], runtime_fingerprint="e" * 64)
    events = (event, *cell.source_events[1:])
    forged_cell = _unchecked_replace(
        cell,
        source_events=events,
        source_event_sha256=tuple(item.fingerprint for item in events),
    )

    with pytest.raises(ValueError, match="one frozen runtime"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(no_response_batch, 0, forged_cell),
        )


def test_malformed_source_canonical_response_is_rejected_before_worker(
    grader: BfclV4PublicV2TrustedGrader,
    selected_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    cell = selected_batch.cells[0]
    event = _unchecked_replace(cell.source_events[0], canonical_response="not-json")
    events = (event, *cell.source_events[1:])
    result = aggregate_bfcl_v4_public_development_v2_pure_at_b(
        tuple(item.canonical_response for item in events)
    )
    forged_cell = _unchecked_replace(
        cell,
        source_events=events,
        source_event_sha256=tuple(item.fingerprint for item in events),
        aggregation_result=result,
        aggregation_result_fingerprint=canonical_sha256(result),
        selected_canonical_response_fingerprint=(
            bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
                result.selected_canonical_response
            )
        ),
    )

    with pytest.raises(ValueError, match="canonical BFCL v2 response is malformed"):
        grade_bfcl_v4_public_v2_pure_at_b_batch(
            grader,
            _replace_cell(selected_batch, 0, forged_cell),
        )


def test_all_no_response_cells_are_false_without_reading_answers(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)

    receipt = grade_bfcl_v4_public_v2_pure_at_b_batch(grader, no_response_batch)

    assert receipt.correct_count == 0
    assert receipt.isolated_worker_execution_count == 0
    assert len(receipt.cell_receipts) == 48
    assert all(not cell.correct for cell in receipt.cell_receipts)
    assert all(not cell.isolated_worker_executed for cell in receipt.cell_receipts)


def test_only_final_selected_response_is_sent_once_to_worker(
    grader: BfclV4PublicV2TrustedGrader,
    selected_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[dict[str, Any], ...]] = []

    def worker(
        self: BfclV4PublicV2TrustedGrader,
        *,
        task: Any,
        calls: tuple[dict[str, Any], ...],
    ) -> bool:
        observed.append(calls)
        return True

    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", worker)

    receipt = grade_bfcl_v4_public_v2_pure_at_b_batch(grader, selected_batch)

    assert observed == [
        (
            {
                "arguments": {"patient_id": "546382", "status": "concluded"},
                "function_name": "patient.get_mri_report",
            },
        )
    ]
    assert receipt.correct_count == 1
    assert receipt.isolated_worker_execution_count == 1
    assert receipt.cell_receipts[0].correct is True
    assert receipt.cell_receipts[0].selected_canonical_response == _KNOWN_HOLDOUT_00_RESPONSE
    assert all(not item.correct for item in receipt.cell_receipts[1:])


def test_known_public_holdout_mode_reaches_exact_upstream_checker(
    grader: BfclV4PublicV2TrustedGrader,
    selected_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> None:
    receipt = grade_bfcl_v4_public_v2_pure_at_b_batch(grader, selected_batch)

    assert receipt.correct_count == 1
    assert receipt.isolated_worker_execution_count == 1
    assert receipt.cell_receipts[0].correct is True
    assert receipt.cell_receipts[0].exact_upstream_ast_checker_executed is True


def test_full_receipt_binds_48_cells_and_330_ungraded_sources_without_leakage(
    grader: BfclV4PublicV2TrustedGrader,
    no_response_batch: BfclV4PublicV2PureAtBBatchGradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BfclV4PublicV2TrustedGrader, "_run_worker_calls", _forbidden_worker)
    receipt = grade_bfcl_v4_public_v2_pure_at_b_batch(grader, no_response_batch)
    payload = receipt.model_dump(mode="json")
    encoded = str(payload)

    assert receipt.cell_count == 48
    assert receipt.source_event_count == 330
    assert receipt.trusted_grade_attempt_count == 48
    assert receipt.individual_sample_grade_count == 0
    assert len(set(receipt.cell_receipt_fingerprints)) == 48
    assert sum(len(cell.source_event_sha256) for cell in receipt.cell_receipts) == 330
    assert receipt.source_sample_grades_present is False
    assert receipt.answers_present is False
    assert receipt.answer_derived_identities_present is False
    assert receipt.checker_diagnostics_present is False
    assert "simple_python_" not in encoded
    assert "ground_truth" not in encoded
    assert "checker_error" not in encoded
