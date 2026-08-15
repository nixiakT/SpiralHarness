"""Composition-root adapters for BFCL v2 trusted grading receipts.

The executor deliberately carries only sanitized response text and opaque
fingerprints.  Ordinary trusted grading therefore uses a one-shot, trusted-plane
registry populated by the native provider composition after it has received the
full response.  This module never reconstructs a native response from sanitized
text and never exposes full grader receipts to the executor journal.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader import (
    grade_bfcl_v4_public_v2_pure_at_b_batch,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts import (
    BfclV4PublicV2PureAtBBatchGradeReceipt,
    BfclV4PublicV2PureAtBBatchGradeRequest,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
    BfclV4PublicV2TrustedGradeRequest,
    BfclV4PublicV2TrustedGraderReceipt,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2PureAtBAggregationRecord,
    BfclV4PublicV2PureAtBBatchGradeProjection,
    BfclV4PublicV2PureAtBCellGradeProjection,
    BfclV4PublicV2TrustedGradeProjection,
)
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2LiveExecutionConfig,
)
from spiral_harness.experiments.bfcl_v4_public_v2_trusted_adapter_contracts import (
    BfclV4PublicV2IssuedTrustedCall,
    BfclV4PublicV2TrustedCallProducer,
    BfclV4PublicV2TrustedCallProducerError,
    BfclV4PublicV2TrustedCallRecord,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GRADABLE_KINDS = frozenset(
    {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        BfclV4PublicDevelopmentV2NodeKind.GATE,
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
    }
)


class BfclV4PublicV2TrustedAdapterError(ValueError):
    """A full trusted artifact differs from the executor-visible lineage."""


class BfclV4PublicV2FullTrustedGrader(Protocol):
    """Full trusted grader surface consumed only inside the composition root."""

    def issue_evaluation_unlock(
        self,
        evidence: BfclV4PublicV2DecisionBarrierEvidence,
    ) -> BfclV4PublicV2EvaluationUnlock: ...

    def grade(
        self,
        request: BfclV4PublicV2TrustedGradeRequest,
        *,
        evaluation_unlock: BfclV4PublicV2EvaluationUnlock | None = None,
    ) -> BfclV4PublicV2TrustedGraderReceipt: ...


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


def _checked[ModelT: BaseModel](
    value: ModelT,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return model.model_validate(_raw_model_content(value), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicV2TrustedAdapterError(
            f"{label} failed strict trusted-adapter validation"
        ) from error


def _checked_sha256(value: str, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BfclV4PublicV2TrustedAdapterError(f"{label} is not a lowercase SHA-256")
    return value


def _checked_campaign(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> BfclV4PublicDevelopmentV2CampaignPlan:
    checked = _checked(campaign, BfclV4PublicDevelopmentV2CampaignPlan, "campaign")
    if (
        checked.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
        or checked.node_schedule_content_sha256
        != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    ):
        raise BfclV4PublicV2TrustedAdapterError("campaign differs from the frozen BFCL v2 plan")
    return checked


class BfclV4PublicV2TrustedCallRegistry:
    """Thread-safe, one-shot handoff from native provider to trusted grader."""

    def __init__(self, *, producer: BfclV4PublicV2TrustedCallProducer) -> None:
        if type(producer) is not BfclV4PublicV2TrustedCallProducer:
            raise BfclV4PublicV2TrustedAdapterError(
                "trusted call registry requires the exact verified producer"
            )
        self._producer = producer
        self._records: dict[str, BfclV4PublicV2TrustedCallRecord] = {}
        self._lock = RLock()

    def register(self, issued: BfclV4PublicV2IssuedTrustedCall) -> str:
        try:
            checked = self._producer.verify_issued(issued)
        except BfclV4PublicV2TrustedCallProducerError as error:
            raise BfclV4PublicV2TrustedAdapterError(
                "trusted call registry rejected an unverified producer result"
            ) from error
        node_id = checked.grade_request.node.node_id
        with self._lock:
            if node_id in self._records:
                raise BfclV4PublicV2TrustedAdapterError(
                    "trusted call registry already contains this DAG node"
                )
            self._records[node_id] = checked
        return checked.fingerprint

    def take(
        self,
        *,
        node: BfclV4PublicDevelopmentV2DagNode,
        canonical_response: str,
        request_payload_sha256: str,
        provider_response_fingerprint: str,
        semantic_release_fingerprint: str,
    ) -> BfclV4PublicV2TrustedCallRecord:
        checked_node = _checked(node, BfclV4PublicDevelopmentV2DagNode, "grade node")
        with self._lock:
            record = self._records.get(checked_node.node_id)
            if record is None:
                raise BfclV4PublicV2TrustedAdapterError(
                    "trusted call registry lacks the exact provider result"
                )
            observed = (
                record.grade_request.node,
                record.canonical_response,
                record.request_payload_sha256,
                record.provider_response_fingerprint,
                record.live_execution_config.semantic_release_fingerprint,
            )
            expected = (
                checked_node,
                canonical_response,
                request_payload_sha256,
                provider_response_fingerprint,
                semantic_release_fingerprint,
            )
            if observed != expected:
                raise BfclV4PublicV2TrustedAdapterError(
                    "trusted call registry lineage differs from the executor call"
                )
            del self._records[checked_node.node_id]
            return record

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def producer(self) -> BfclV4PublicV2TrustedCallProducer:
        return self._producer


def _split_for_node(node: BfclV4PublicDevelopmentV2DagNode) -> BfclV4PublicDevelopmentV2Split:
    if node.kind in {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
    }:
        return BfclV4PublicDevelopmentV2Split.FIT
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE:
        return BfclV4PublicDevelopmentV2Split.GATE
    if node.kind is BfclV4PublicDevelopmentV2NodeKind.HOLDOUT:
        return BfclV4PublicDevelopmentV2Split.HOLDOUT
    raise BfclV4PublicV2TrustedAdapterError("executor requested a non-gradable BFCL node")


def project_bfcl_v4_public_v2_trusted_grade(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    semantic_release_fingerprint: str,
    call: BfclV4PublicV2TrustedCallRecord,
    receipt: BfclV4PublicV2TrustedGraderReceipt,
    evaluation_unlock: BfclV4PublicV2EvaluationUnlock | None,
) -> BfclV4PublicV2TrustedGradeProjection:
    """Fail closed before reducing one full grader receipt to journal fields."""

    checked_campaign = _checked_campaign(campaign)
    release = _checked_sha256(semantic_release_fingerprint, "semantic release fingerprint")
    checked_call = _checked(call, BfclV4PublicV2TrustedCallRecord, "trusted call record")
    checked_receipt = _checked(receipt, BfclV4PublicV2TrustedGraderReceipt, "grader receipt")
    request = checked_call.grade_request
    node = request.node
    if (
        node.kind not in _GRADABLE_KINDS
        or checked_call.live_execution_config.semantic_release_fingerprint != release
    ):
        raise BfclV4PublicV2TrustedAdapterError("trusted call kind or semantic release changed")
    if (
        node.node_slot >= len(checked_campaign.nodes)
        or checked_campaign.nodes[node.node_slot] != node
    ):
        raise BfclV4PublicV2TrustedAdapterError(
            "trusted call node differs from the frozen campaign"
        )

    split = _split_for_node(node)
    if evaluation_unlock is None:
        unlock_fingerprint = None
        if split is BfclV4PublicDevelopmentV2Split.HOLDOUT:
            raise BfclV4PublicV2TrustedAdapterError("HOLDOUT grade lacks an evaluation unlock")
    else:
        checked_unlock = _checked(
            evaluation_unlock,
            BfclV4PublicV2EvaluationUnlock,
            "evaluation unlock",
        )
        barrier = checked_unlock.barrier_evidence
        decision_nodes = tuple(
            item
            for item in checked_campaign.nodes
            if item.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        )
        expected_decision_references = tuple(canonical_sha256(item) for item in decision_nodes)
        if (
            split is not BfclV4PublicDevelopmentV2Split.HOLDOUT
            or barrier.campaign_plan_fingerprint != checked_campaign.fingerprint
            or barrier.node_schedule_content_sha256 != checked_campaign.node_schedule_content_sha256
            or barrier.semantic_release_fingerprint != release
            or barrier.decision_node_references != expected_decision_references
        ):
            raise BfclV4PublicV2TrustedAdapterError(
                "evaluation unlock differs from the campaign or semantic release"
            )
        unlock_fingerprint = checked_unlock.fingerprint

    expected_receipt = (
        checked_campaign.fingerprint,
        checked_campaign.node_schedule_content_sha256,
        request.node_reference_sha256,
        node.campaign_call_slot,
        node.task_ref,
        canonical_sha256(
            {
                "domain": ("spiral-bfcl-v4-public-development-v2-structural-task-reference/v1"),
                "campaign_plan_fingerprint": checked_campaign.fingerprint,
                "task_ref": node.task_ref,
                "task_payload_sha256": request.task_payload_sha256,
            }
        ),
        split,
        request.request_fingerprint,
        request.response_fingerprint,
        unlock_fingerprint,
    )
    observed_receipt = (
        checked_receipt.campaign_plan_fingerprint,
        checked_receipt.node_schedule_content_sha256,
        checked_receipt.node_reference_sha256,
        checked_receipt.campaign_call_slot,
        checked_receipt.task_ref,
        checked_receipt.task_reference_sha256,
        checked_receipt.split_role,
        checked_receipt.request_fingerprint,
        checked_receipt.response_fingerprint,
        checked_receipt.evaluation_unlock_fingerprint,
    )
    if observed_receipt != expected_receipt:
        raise BfclV4PublicV2TrustedAdapterError(
            "full grader receipt differs from its exact request, response, or unlock"
        )
    return BfclV4PublicV2TrustedGradeProjection(
        correct=checked_receipt.correct,
        request_payload_sha256=checked_call.request_payload_sha256,
        provider_response_fingerprint=checked_call.provider_response_fingerprint,
        trusted_grade_request_fingerprint=request.fingerprint,
        trusted_grader_receipt_fingerprint=checked_receipt.fingerprint,
        evaluation_unlock_fingerprint=unlock_fingerprint,
    )


class BfclV4PublicV2TrustedGraderAdapter:
    """Concrete executor grader backed by full trusted requests and receipts."""

    def __init__(
        self,
        *,
        grader: BfclV4PublicV2FullTrustedGrader,
        campaign: BfclV4PublicDevelopmentV2CampaignPlan,
        live_config: BfclV4PublicV2LiveExecutionConfig,
        call_registry: BfclV4PublicV2TrustedCallRegistry,
    ) -> None:
        self._grader = grader
        self._campaign = _checked_campaign(campaign)
        self._live_config = _checked(
            live_config,
            BfclV4PublicV2LiveExecutionConfig,
            "live execution config",
        )
        self._semantic_release_fingerprint = self._live_config.semantic_release_fingerprint
        if type(call_registry) is not BfclV4PublicV2TrustedCallRegistry:
            raise BfclV4PublicV2TrustedAdapterError(
                "call registry must use the exact trusted registry implementation"
            )
        producer = call_registry.producer
        if producer.campaign != self._campaign or producer.live_config != self._live_config:
            raise BfclV4PublicV2TrustedAdapterError(
                "trusted registry producer differs from adapter campaign or live config"
            )
        self._call_registry = call_registry
        self._unlocks: dict[str, BfclV4PublicV2EvaluationUnlock] = {}
        self._lock = RLock()

    def issue_evaluation_unlock(
        self,
        evidence: BfclV4PublicV2DecisionBarrierEvidence,
    ) -> BfclV4PublicV2EvaluationUnlock:
        checked_evidence = _checked(
            evidence,
            BfclV4PublicV2DecisionBarrierEvidence,
            "decision barrier evidence",
        )
        decision_nodes = tuple(
            node
            for node in self._campaign.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        )
        expected_references = tuple(canonical_sha256(node) for node in decision_nodes)
        if (
            checked_evidence.campaign_plan_fingerprint != self._campaign.fingerprint
            or checked_evidence.node_schedule_content_sha256
            != self._campaign.node_schedule_content_sha256
            or checked_evidence.semantic_release_fingerprint != self._semantic_release_fingerprint
            or checked_evidence.decision_node_references != expected_references
        ):
            raise BfclV4PublicV2TrustedAdapterError(
                "decision barrier differs from the frozen campaign or release"
            )
        try:
            raw_unlock = self._grader.issue_evaluation_unlock(checked_evidence)
        except Exception as error:
            raise BfclV4PublicV2TrustedAdapterError(
                "full trusted grader rejected evaluation authorization"
            ) from error
        unlock = _checked(raw_unlock, BfclV4PublicV2EvaluationUnlock, "evaluation unlock")
        if unlock.barrier_evidence != checked_evidence:
            raise BfclV4PublicV2TrustedAdapterError(
                "full trusted grader authorized another decision barrier"
            )
        with self._lock:
            prior = self._unlocks.get(unlock.fingerprint)
            if prior is not None and prior != unlock:
                raise BfclV4PublicV2TrustedAdapterError("evaluation unlock identity collision")
            self._unlocks[unlock.fingerprint] = unlock
        return unlock

    def grade(
        self,
        node: BfclV4PublicDevelopmentV2DagNode,
        canonical_response: str,
        *,
        request_payload_sha256: str,
        provider_response_fingerprint: str,
        evaluation_unlock_fingerprint: str | None,
    ) -> BfclV4PublicV2TrustedGradeProjection:
        if evaluation_unlock_fingerprint is None:
            unlock = None
        else:
            with self._lock:
                unlock = self._unlocks.get(evaluation_unlock_fingerprint)
            if unlock is None:
                raise BfclV4PublicV2TrustedAdapterError(
                    "executor supplied an unknown evaluation unlock"
                )
        call = self._call_registry.take(
            node=node,
            canonical_response=canonical_response,
            request_payload_sha256=request_payload_sha256,
            provider_response_fingerprint=provider_response_fingerprint,
            semantic_release_fingerprint=self._semantic_release_fingerprint,
        )
        try:
            raw_receipt = self._grader.grade(
                call.grade_request,
                evaluation_unlock=unlock,
            )
        except Exception as error:
            raise BfclV4PublicV2TrustedAdapterError("full trusted grading failed") from error
        return project_bfcl_v4_public_v2_trusted_grade(
            campaign=self._campaign,
            semantic_release_fingerprint=self._semantic_release_fingerprint,
            call=call,
            receipt=raw_receipt,
            evaluation_unlock=unlock,
        )


def _checked_batch_context(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    semantic_release_fingerprint: str,
    request: BfclV4PublicV2PureAtBBatchGradeRequest,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    checked = _checked(request, BfclV4PublicV2PureAtBBatchGradeRequest, "PURE@B batch request")
    release = _checked_sha256(semantic_release_fingerprint, "semantic release fingerprint")
    barrier = checked.evaluation_unlock.barrier_evidence
    decision_nodes = tuple(
        node
        for node in campaign.nodes
        if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
    )
    expected_references = tuple(canonical_sha256(node) for node in decision_nodes)
    observed = (
        checked.campaign_plan_fingerprint,
        checked.node_schedule_content_sha256,
        checked.semantic_release_fingerprint,
        barrier.campaign_plan_fingerprint,
        barrier.node_schedule_content_sha256,
        barrier.semantic_release_fingerprint,
        barrier.decision_node_references,
    )
    expected = (
        campaign.fingerprint,
        campaign.node_schedule_content_sha256,
        release,
        campaign.fingerprint,
        campaign.node_schedule_content_sha256,
        release,
        expected_references,
    )
    if observed != expected:
        raise BfclV4PublicV2TrustedAdapterError(
            "PURE@B batch differs from the frozen campaign, release, or unlock"
        )
    return checked


def project_bfcl_v4_public_v2_pure_at_b_batch_grade(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    semantic_release_fingerprint: str,
    request: BfclV4PublicV2PureAtBBatchGradeRequest,
    receipt: BfclV4PublicV2PureAtBBatchGradeReceipt,
) -> BfclV4PublicV2PureAtBBatchGradeProjection:
    """Verify a full 48-cell receipt before retaining its minimal projection."""

    checked_campaign = _checked_campaign(campaign)
    checked_request = _checked_batch_context(
        campaign=checked_campaign,
        semantic_release_fingerprint=semantic_release_fingerprint,
        request=request,
    )
    checked_receipt = _checked(
        receipt,
        BfclV4PublicV2PureAtBBatchGradeReceipt,
        "PURE@B batch receipt",
    )
    expected_batch = (
        checked_request.campaign_plan_fingerprint,
        checked_request.node_schedule_content_sha256,
        checked_request.semantic_release_fingerprint,
        checked_request.fingerprint,
        checked_request.evaluation_unlock_fingerprint,
        checked_request.decision_barrier_evidence_fingerprint,
        checked_request.source_event_count,
        checked_request.individual_sample_grade_count,
    )
    observed_batch = (
        checked_receipt.campaign_plan_fingerprint,
        checked_receipt.node_schedule_content_sha256,
        checked_receipt.semantic_release_fingerprint,
        checked_receipt.batch_grade_request_fingerprint,
        checked_receipt.evaluation_unlock_fingerprint,
        checked_receipt.decision_barrier_evidence_fingerprint,
        checked_receipt.source_event_count,
        checked_receipt.individual_sample_grade_count,
    )
    if observed_batch != expected_batch:
        raise BfclV4PublicV2TrustedAdapterError(
            "PURE@B full receipt differs from its batch request or authority"
        )

    projections: list[BfclV4PublicV2PureAtBCellGradeProjection] = []
    for cell_request, cell_receipt in zip(
        checked_request.cells,
        checked_receipt.cell_receipts,
        strict=True,
    ):
        expected_cell = (
            cell_request.campaign_plan_fingerprint,
            cell_request.node_schedule_content_sha256,
            cell_request.semantic_release_fingerprint,
            cell_request.outer_seed_u64,
            cell_request.task_ref,
            cell_request.fingerprint,
            cell_request.allocation,
            cell_request.source_event_sha256,
            cell_request.aggregation_spec_fingerprint,
            cell_request.aggregation_result,
            cell_request.aggregation_result_fingerprint,
            cell_request.selected_canonical_response_fingerprint,
            cell_request.decision_barrier_evidence_fingerprint,
            cell_request.evaluation_unlock_fingerprint,
        )
        observed_cell = (
            cell_receipt.campaign_plan_fingerprint,
            cell_receipt.node_schedule_content_sha256,
            cell_receipt.semantic_release_fingerprint,
            cell_receipt.outer_seed_u64,
            cell_receipt.task_ref,
            cell_receipt.cell_grade_request_fingerprint,
            cell_receipt.allocation,
            cell_receipt.source_event_sha256,
            cell_receipt.aggregation_spec_fingerprint,
            cell_receipt.aggregation_result,
            cell_receipt.aggregation_result_fingerprint,
            cell_receipt.selected_canonical_response_fingerprint,
            cell_receipt.decision_barrier_evidence_fingerprint,
            cell_receipt.evaluation_unlock_fingerprint,
        )
        if observed_cell != expected_cell:
            raise BfclV4PublicV2TrustedAdapterError(
                "PURE@B cell receipt differs from its exact modal request"
            )
        aggregation = BfclV4PublicV2PureAtBAggregationRecord(
            outer_seed_u64=cell_request.outer_seed_u64,
            task_ref=cell_request.task_ref,
            source_event_sha256=cell_request.source_event_sha256,
            result=cell_request.aggregation_result,
        )
        projections.append(
            BfclV4PublicV2PureAtBCellGradeProjection(
                outer_seed_u64=cell_request.outer_seed_u64,
                task_ref=cell_request.task_ref,
                aggregation_record_fingerprint=aggregation.fingerprint,
                cell_grade_request_fingerprint=cell_request.fingerprint,
                cell_grade_receipt_fingerprint=cell_receipt.fingerprint,
                correct=cell_receipt.correct,
            )
        )
    return BfclV4PublicV2PureAtBBatchGradeProjection(
        batch_grade_request_fingerprint=checked_request.fingerprint,
        batch_grade_receipt_fingerprint=checked_receipt.fingerprint,
        decision_barrier_evidence_fingerprint=(
            checked_request.decision_barrier_evidence_fingerprint
        ),
        evaluation_unlock_fingerprint=checked_request.evaluation_unlock_fingerprint,
        cells=tuple(projections),
        correct_count=checked_receipt.correct_count,
    )


class BfclV4PublicV2PureAtBBatchGraderAdapter:
    """Concrete executor batch grader backed by the full trusted batch grader."""

    def __init__(
        self,
        *,
        grader: BfclV4PublicV2FullTrustedGrader,
        campaign: BfclV4PublicDevelopmentV2CampaignPlan,
        semantic_release_fingerprint: str,
        full_batch_grader: Callable[
            [BfclV4PublicV2FullTrustedGrader, BfclV4PublicV2PureAtBBatchGradeRequest],
            BfclV4PublicV2PureAtBBatchGradeReceipt,
        ] = grade_bfcl_v4_public_v2_pure_at_b_batch,
    ) -> None:
        self._grader = grader
        self._campaign = _checked_campaign(campaign)
        self._semantic_release_fingerprint = _checked_sha256(
            semantic_release_fingerprint,
            "semantic release fingerprint",
        )
        if not callable(full_batch_grader):
            raise BfclV4PublicV2TrustedAdapterError("full PURE@B batch grader is not callable")
        self._full_batch_grader = full_batch_grader

    def grade_pure_at_b_batch(
        self,
        request: BfclV4PublicV2PureAtBBatchGradeRequest,
    ) -> BfclV4PublicV2PureAtBBatchGradeProjection:
        checked_request = _checked_batch_context(
            campaign=self._campaign,
            semantic_release_fingerprint=self._semantic_release_fingerprint,
            request=request,
        )
        try:
            full_receipt = self._full_batch_grader(self._grader, checked_request)
        except Exception as error:
            raise BfclV4PublicV2TrustedAdapterError(
                "full trusted PURE@B batch grading failed"
            ) from error
        return project_bfcl_v4_public_v2_pure_at_b_batch_grade(
            campaign=self._campaign,
            semantic_release_fingerprint=self._semantic_release_fingerprint,
            request=checked_request,
            receipt=full_receipt,
        )


__all__ = [name for name in globals() if name.startswith("Bfcl") or name.startswith("project")]
