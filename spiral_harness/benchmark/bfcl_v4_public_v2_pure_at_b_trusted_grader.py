"""Atomic trusted grading for all BFCL v2 PURE@B modal responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2NodeKind,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
    BfclV4PublicDevelopmentV2Task,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts import (
    BfclV4PublicV2PureAtBBatchGradeReceipt,
    BfclV4PublicV2PureAtBBatchGradeRequest,
    BfclV4PublicV2PureAtBCellGradeReceipt,
    BfclV4PublicV2PureAtBCellGradeRequest,
    bfcl_v4_public_v2_pure_at_b_source_set_fingerprint,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader import (
    BfclV4PublicV2TrustedGrader,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2AttemptDisposition,
    BfclV4PublicV2ProviderRequest,
)
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    parse_bfcl_v4_public_v2_canonical_response,
)


class _PreparedCell(NamedTuple):
    request: BfclV4PublicV2PureAtBCellGradeRequest
    task: BfclV4PublicDevelopmentV2Task
    calls: tuple[dict[str, Any], ...] | None


def _raw_model_content(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {name: _raw_model_content(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _raw_model_content(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_raw_model_content(item) for item in value)
    if isinstance(value, list):
        return [_raw_model_content(item) for item in value]
    return value


def _checked_batch(
    value: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    try:
        return BfclV4PublicV2PureAtBBatchGradeRequest.model_validate(
            _raw_model_content(value),
            strict=True,
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ValueError("PURE@B batch contract revalidation failed") from error


def _expected_source_nodes(
    grader: BfclV4PublicV2TrustedGrader,
    cell: BfclV4PublicV2PureAtBCellGradeRequest,
) -> tuple[Any, ...]:
    return tuple(
        node
        for node in grader._campaign.nodes
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
        and node.arm is BfclV4PublicDevelopmentV2Arm.PURE_AT_B
        and node.outer_seed_u64 == cell.outer_seed_u64
        and node.task_ref == cell.task_ref
    )


def _validate_source_request(
    grader: BfclV4PublicV2TrustedGrader,
    cell: BfclV4PublicV2PureAtBCellGradeRequest,
) -> None:
    campaign_fingerprint = cell.campaign_plan_fingerprint
    schedule_fingerprint = grader._campaign.node_schedule_content_sha256
    mutation_fingerprint = grader._campaign.mutation_catalog_fingerprint
    for node, event in zip(cell.source_nodes, cell.source_events, strict=True):
        if node.campaign_call_slot is None or node.provider_seed_u63 is None:
            raise ValueError("PURE@B source node is not a frozen model-call slot")
        expected = BfclV4PublicV2ProviderRequest(
            campaign_plan_fingerprint=campaign_fingerprint,
            node_schedule_content_sha256=schedule_fingerprint,
            mutation_catalog_fingerprint=mutation_fingerprint,
            runtime_fingerprint=event.runtime_fingerprint,
            semantic_release_fingerprint=cell.semantic_release_fingerprint,
            node_id=node.node_id,
            node_reference_sha256=canonical_sha256(node),
            campaign_call_slot=node.campaign_call_slot,
            provider_seed_u63=node.provider_seed_u63,
            request_payload_sha256=event.request_payload_sha256,
            decision_barrier_evidence_fingerprint=(cell.decision_barrier_evidence_fingerprint),
            evaluation_unlock_fingerprint=cell.evaluation_unlock_fingerprint,
        )
        if event.request_fingerprint != expected.fingerprint:
            raise ValueError("PURE@B source event request lineage changed")


def _canonical_calls(
    cell: BfclV4PublicV2PureAtBCellGradeRequest,
) -> tuple[dict[str, Any], ...] | None:
    for event in cell.source_events:
        if event.canonical_response is not None:
            parse_bfcl_v4_public_v2_canonical_response(event.canonical_response)
    recomputed = aggregate_bfcl_v4_public_development_v2_pure_at_b(
        tuple(event.canonical_response for event in cell.source_events)
    )
    if recomputed != cell.aggregation_result:
        raise ValueError("PURE@B modal response changed during trusted revalidation")
    selected = recomputed.selected_canonical_response
    if selected is None:
        return None
    parsed = parse_bfcl_v4_public_v2_canonical_response(selected)
    return tuple(
        {
            "arguments": json.loads(call.arguments_json),
            "function_name": call.function_name,
        }
        for call in parsed.calls
    )


def _prepare_cells(
    grader: BfclV4PublicV2TrustedGrader,
    batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> tuple[_PreparedCell, ...]:
    prepared: list[_PreparedCell] = []
    runtime_fingerprints = {
        event.runtime_fingerprint for cell in batch.cells for event in cell.source_events
    }
    if len(runtime_fingerprints) != 1:
        raise ValueError("PURE@B source events do not share one frozen runtime")
    for cell in batch.cells:
        expected_nodes = _expected_source_nodes(grader, cell)
        if expected_nodes != cell.source_nodes:
            raise ValueError("PURE@B source nodes differ from the frozen campaign")
        if tuple(node.sample_index for node in expected_nodes) != tuple(
            range(cell.allocation.sample_count)
        ):
            raise ValueError("PURE@B frozen campaign has an invalid sample allocation")
        _validate_source_request(grader, cell)
        for event in cell.source_events:
            if event.mutation_catalog_fingerprint != grader._campaign.mutation_catalog_fingerprint:
                raise ValueError("PURE@B source event mutation catalogue changed")
            succeeded = (
                event.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
            )
            if succeeded is (event.canonical_response is None) or succeeded is (
                event.provider_response_fingerprint is None
            ):
                raise ValueError("PURE@B source disposition and response lineage disagree")
        split, task = grader._resolve_task(cell.task_ref)
        if split is not BfclV4PublicDevelopmentV2Split.HOLDOUT:
            raise ValueError("PURE@B aggregate grading is restricted to HOLDOUT")
        prepared.append(_PreparedCell(cell, task, _canonical_calls(cell)))
    return tuple(prepared)


def _cell_receipt(
    grader: BfclV4PublicV2TrustedGrader,
    prepared: _PreparedCell,
    *,
    correct: bool,
    pure_at_b_source_sha256: str,
) -> BfclV4PublicV2PureAtBCellGradeReceipt:
    cell = prepared.request
    result = cell.aggregation_result
    worker_executed = prepared.calls is not None
    return BfclV4PublicV2PureAtBCellGradeReceipt(
        semantic_release_fingerprint=cell.semantic_release_fingerprint,
        outer_seed_u64=cell.outer_seed_u64,
        task_ref=cell.task_ref,
        cell_grade_request_fingerprint=cell.fingerprint,
        allocation=cell.allocation,
        allocation_fingerprint=canonical_sha256(cell.allocation),
        source_event_sha256=cell.source_event_sha256,
        source_set_fingerprint=bfcl_v4_public_v2_pure_at_b_source_set_fingerprint(
            cell.source_event_sha256
        ),
        aggregation_spec_fingerprint=(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
        ),
        aggregation_result=result,
        aggregation_result_fingerprint=canonical_sha256(result),
        selected_canonical_response=result.selected_canonical_response,
        selected_canonical_response_fingerprint=(cell.selected_canonical_response_fingerprint),
        decision_barrier_evidence_fingerprint=(cell.decision_barrier_evidence_fingerprint),
        evaluation_unlock_fingerprint=cell.evaluation_unlock_fingerprint,
        loaded_question_bundle_fingerprint=grader._loaded_bundle_fingerprint,
        grader_source_sha256=grader._source_sha256,
        pure_at_b_grader_source_sha256=pure_at_b_source_sha256,
        correct=correct,
        isolated_worker_executed=worker_executed,
        exact_upstream_ast_checker_executed=worker_executed,
        possible_answers_read_in_isolated_trusted_worker=worker_executed,
    )


def grade_bfcl_v4_public_v2_pure_at_b_batch(
    grader: BfclV4PublicV2TrustedGrader,
    batch: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> BfclV4PublicV2PureAtBBatchGradeReceipt:
    """Grade 48 final modes only after all 330 ungraded samples validate."""

    if not isinstance(grader, BfclV4PublicV2TrustedGrader):
        raise TypeError("grader must be a BFCL v2 trusted grader")
    checked = _checked_batch(batch)
    if (
        checked.campaign_plan_fingerprint != grader._campaign.fingerprint
        or checked.node_schedule_content_sha256 != grader._campaign.node_schedule_content_sha256
        or checked.semantic_release_fingerprint != grader._semantic_release_fingerprint
    ):
        raise ValueError("PURE@B batch differs from the trusted frozen release")
    authorized_unlock = grader._authorize_split(
        BfclV4PublicDevelopmentV2Split.HOLDOUT,
        checked.evaluation_unlock,
    )
    assert authorized_unlock is not None
    if authorized_unlock.fingerprint != checked.evaluation_unlock_fingerprint:
        raise ValueError("PURE@B batch evaluation unlock changed")

    # This pass must finish for all 48 cells before any isolated worker can run.
    prepared_cells = _prepare_cells(grader, checked)
    try:
        pure_at_b_source_sha256 = sha256_bytes(Path(__file__).resolve(strict=True).read_bytes())
    except OSError as error:
        raise ValueError("PURE@B aggregate grader source binding failed") from error

    cell_receipts: list[BfclV4PublicV2PureAtBCellGradeReceipt] = []
    for prepared in prepared_cells:
        correct = (
            False
            if prepared.calls is None
            else grader._run_worker_calls(task=prepared.task, calls=prepared.calls)
        )
        cell_receipts.append(
            _cell_receipt(
                grader,
                prepared,
                correct=correct,
                pure_at_b_source_sha256=pure_at_b_source_sha256,
            )
        )

    receipts = tuple(cell_receipts)
    return BfclV4PublicV2PureAtBBatchGradeReceipt(
        semantic_release_fingerprint=checked.semantic_release_fingerprint,
        batch_grade_request_fingerprint=checked.fingerprint,
        evaluation_unlock_fingerprint=checked.evaluation_unlock_fingerprint,
        decision_barrier_evidence_fingerprint=(checked.decision_barrier_evidence_fingerprint),
        cell_receipts=receipts,
        cell_receipt_fingerprints=tuple(receipt.fingerprint for receipt in receipts),
        correct_count=sum(receipt.correct for receipt in receipts),
        isolated_worker_execution_count=sum(
            receipt.isolated_worker_executed for receipt in receipts
        ),
        loaded_question_bundle_fingerprint=grader._loaded_bundle_fingerprint,
        grader_source_sha256=grader._source_sha256,
        pure_at_b_grader_source_sha256=pure_at_b_source_sha256,
    )


__all__ = ["grade_bfcl_v4_public_v2_pure_at_b_batch"]
