"""Fail-closed aggregate-grade contracts for BFCL v2 PURE@B."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2OuterSeed,
    BfclV4PublicDevelopmentV2PureAtBAggregationResult,
    BfclV4PublicDevelopmentV2PureAtBAggregationSpec,
    BfclV4PublicDevelopmentV2PureAtBAllocation,
    aggregate_bfcl_v4_public_development_v2_pure_at_b,
    expected_bfcl_v4_public_development_v2_pure_at_b_allocation,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256,
    BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256,
    BfclV4PublicV2EvaluationUnlock,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, Sha256
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2EventKind,
    BfclV4PublicV2JournalEvent,
    BfclV4PublicV2PureAtBAggregationRecord,
)

BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL = (
    "spiral-bfcl-v4-public-development-v2-pure-at-b-aggregate-grader/v1"
)
BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT = canonical_sha256(
    {
        "domain": BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL,
        "base_grader_implementation_fingerprint": (
            BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
        ),
        "cell_order": "frozen-outer-seed-then-frozen-holdout-allocation",
        "source_count": 330,
        "cell_count": 48,
        "grade_boundary": "one-final-modal-response-per-cell-after-full-batch-validation",
    }
)


def bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
    value: str | None,
) -> str:
    """Bind the selected vote, including the distinguished no-response value."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-pure-at-b-selected-response/v1",
            "selected_canonical_response": value,
        }
    )


def bfcl_v4_public_v2_pure_at_b_source_set_fingerprint(
    source_event_sha256: tuple[str, ...],
) -> str:
    """Bind the exact ordered sample-event set used by one modal decision."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-pure-at-b-source-set/v1",
            "source_event_sha256": source_event_sha256,
        }
    )


class BfclV4PublicV2PureAtBCellGradeRequest(ImmutableModel):
    """One label-free modal response and all of its ungraded source events."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_fingerprint: Sha256
    task_ref: Annotated[str, Field(pattern=r"^holdout-[0-9]{2}$")]
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed
    allocation: BfclV4PublicDevelopmentV2PureAtBAllocation
    source_nodes: Annotated[
        tuple[BfclV4PublicDevelopmentV2DagNode, ...],
        Field(min_length=6, max_length=7),
    ]
    source_node_references: Annotated[tuple[Sha256, ...], Field(min_length=6, max_length=7)]
    source_events: Annotated[
        tuple[BfclV4PublicV2JournalEvent, ...],
        Field(min_length=6, max_length=7),
    ]
    source_event_sha256: Annotated[tuple[Sha256, ...], Field(min_length=6, max_length=7)]
    aggregation_spec: BfclV4PublicDevelopmentV2PureAtBAggregationSpec
    aggregation_spec_fingerprint: Sha256
    aggregation_result: BfclV4PublicDevelopmentV2PureAtBAggregationResult
    aggregation_result_fingerprint: Sha256
    selected_canonical_response_fingerprint: Sha256
    decision_barrier_evidence_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256
    source_sample_grades_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_complete_cell(self) -> Self:
        allocation = self.allocation
        if allocation.task_ref != self.task_ref:
            raise ValueError("PURE@B allocation points to another HOLDOUT task")
        if len(self.source_nodes) != allocation.sample_count or len(self.source_events) != (
            allocation.sample_count
        ):
            raise ValueError("PURE@B source count differs from its frozen allocation")
        if self.source_node_references != tuple(
            canonical_sha256(node) for node in self.source_nodes
        ):
            raise ValueError("PURE@B source-node references changed")
        if self.source_event_sha256 != tuple(event.fingerprint for event in self.source_events):
            raise ValueError("PURE@B source-event references changed")
        if (
            len(set(self.source_node_references)) != allocation.sample_count
            or len(set(self.source_event_sha256)) != allocation.sample_count
        ):
            raise ValueError("PURE@B cell repeats a source node or event")

        for sample_index, (node, event) in enumerate(
            zip(self.source_nodes, self.source_events, strict=True)
        ):
            if (
                node.kind is not BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
                or node.arm is not BfclV4PublicDevelopmentV2Arm.PURE_AT_B
                or node.task_ref != self.task_ref
                or node.outer_seed_u64 != self.outer_seed_u64
                or node.sample_index != sample_index
                or not node.consumes_model_call
            ):
                raise ValueError("PURE@B source node differs from its cell coordinates")
            if (
                event.event_kind is not BfclV4PublicV2EventKind.CALL
                or event.campaign_plan_fingerprint != self.campaign_plan_fingerprint
                or event.node_schedule_content_sha256 != self.node_schedule_content_sha256
                or event.semantic_release_fingerprint != self.semantic_release_fingerprint
                or event.node_id != node.node_id
                or event.sequence != node.node_slot
                or event.node_slot != node.node_slot
                or event.node_reference_sha256 != self.source_node_references[sample_index]
                # The per-sample plan labels (``bare-sample-{i}``) identify
                # frozen stochastic slots; they are not distinct harnesses.
                # Every PURE@B source must execute the same bare harness.
                or event.executed_harness_variant != "bare"
                or event.binary_grade is not None
                or event.trusted_grade_request_fingerprint is not None
                or event.trusted_grader_receipt_fingerprint is not None
                or event.trusted_grade_attempts_consumed != 0
                or event.decision_barrier_evidence_fingerprint
                != self.decision_barrier_evidence_fingerprint
                or event.evaluation_unlock_fingerprint != self.evaluation_unlock_fingerprint
                or event.proposal_disposition is not None
                or event.candidate_artifact_sha256 is not None
            ):
                raise ValueError("PURE@B source event is not an ungraded sample call")

        if (
            self.aggregation_spec != BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION
            or self.aggregation_spec_fingerprint != self.aggregation_spec.fingerprint
        ):
            raise ValueError("PURE@B aggregation rule changed")
        recomputed = aggregate_bfcl_v4_public_development_v2_pure_at_b(
            tuple(event.canonical_response for event in self.source_events)
        )
        if self.aggregation_result != recomputed:
            raise ValueError("PURE@B modal result differs from its source events")
        if self.aggregation_result_fingerprint != canonical_sha256(self.aggregation_result):
            raise ValueError("PURE@B aggregation-result fingerprint changed")
        if self.selected_canonical_response_fingerprint != (
            bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
                self.aggregation_result.selected_canonical_response
            )
        ):
            raise ValueError("PURE@B selected-response fingerprint changed")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2PureAtBBatchGradeRequest(ImmutableModel):
    """The sole public grade request: all 48 frozen seed/task modal cells."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_fingerprint: Sha256
    cells: Annotated[
        tuple[BfclV4PublicV2PureAtBCellGradeRequest, ...],
        Field(min_length=48, max_length=48),
    ]
    evaluation_unlock: BfclV4PublicV2EvaluationUnlock
    evaluation_unlock_fingerprint: Sha256
    decision_barrier_evidence_fingerprint: Sha256
    all_label_free_modes_frozen: Literal[True] = True
    individual_sample_grade_count: Literal[0] = 0
    source_event_count: Literal[330] = 330

    @model_validator(mode="after")
    def _bind_complete_batch(self) -> Self:
        unlock = self.evaluation_unlock
        if (
            self.evaluation_unlock_fingerprint != unlock.fingerprint
            or self.decision_barrier_evidence_fingerprint != unlock.barrier_evidence_fingerprint
            or unlock.barrier_evidence.semantic_release_fingerprint
            != self.semantic_release_fingerprint
        ):
            raise ValueError("PURE@B batch authority lineage changed")
        expected_coordinates = tuple(
            (outer_seed, allocation)
            for outer_seed in BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64
            for allocation in expected_bfcl_v4_public_development_v2_pure_at_b_allocation()
        )
        actual_coordinates = tuple((cell.outer_seed_u64, cell.allocation) for cell in self.cells)
        if actual_coordinates != expected_coordinates:
            raise ValueError("PURE@B cells differ from the frozen seed/task order")
        for cell in self.cells:
            if (
                cell.campaign_plan_fingerprint != self.campaign_plan_fingerprint
                or cell.node_schedule_content_sha256 != self.node_schedule_content_sha256
                or cell.semantic_release_fingerprint != self.semantic_release_fingerprint
                or cell.decision_barrier_evidence_fingerprint
                != self.decision_barrier_evidence_fingerprint
                or cell.evaluation_unlock_fingerprint != self.evaluation_unlock_fingerprint
            ):
                raise ValueError("PURE@B cells do not share one frozen release and unlock")
        node_refs = tuple(ref for cell in self.cells for ref in cell.source_node_references)
        event_refs = tuple(ref for cell in self.cells for ref in cell.source_event_sha256)
        if len(node_refs) != self.source_event_count or len(event_refs) != self.source_event_count:
            raise ValueError("PURE@B batch does not contain all 330 source samples")
        if len(set(node_refs)) != self.source_event_count or len(set(event_refs)) != (
            self.source_event_count
        ):
            raise ValueError("PURE@B batch repeats a source node or event")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def build_bfcl_v4_public_v2_pure_at_b_batch_grade_request(
    *,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    events: tuple[BfclV4PublicV2JournalEvent, ...],
    aggregations: tuple[BfclV4PublicV2PureAtBAggregationRecord, ...],
    evaluation_unlock: BfclV4PublicV2EvaluationUnlock,
    semantic_release_fingerprint: Sha256,
) -> BfclV4PublicV2PureAtBBatchGradeRequest:
    """Build the exact 48-cell request from replay-derived, label-free state."""

    checked_campaign = BfclV4PublicDevelopmentV2CampaignPlan.model_validate(
        campaign,
        strict=True,
    )
    checked_events = tuple(
        BfclV4PublicV2JournalEvent.model_validate(event, strict=True) for event in events
    )
    checked_aggregations = tuple(
        BfclV4PublicV2PureAtBAggregationRecord.model_validate(record, strict=True)
        for record in aggregations
    )
    by_node = {event.node_id: event for event in checked_events}
    by_coordinate = {
        (record.outer_seed_u64, record.task_ref): record for record in checked_aggregations
    }
    if len(by_node) != len(checked_events):
        raise ValueError("PURE@B request builder received duplicate event nodes")
    if len(by_coordinate) != 48 or len(checked_aggregations) != 48:
        raise ValueError("PURE@B request builder requires 48 unique aggregations")

    cells: list[BfclV4PublicV2PureAtBCellGradeRequest] = []
    for outer_seed in BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64:
        for allocation in expected_bfcl_v4_public_development_v2_pure_at_b_allocation():
            nodes = tuple(
                node
                for node in checked_campaign.nodes
                if node.kind is BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE
                and node.outer_seed_u64 == outer_seed
                and node.task_ref == allocation.task_ref
            )
            try:
                source_events = tuple(by_node[node.node_id] for node in nodes)
                aggregation = by_coordinate[(outer_seed, allocation.task_ref)]
            except KeyError as error:
                raise ValueError("PURE@B request builder lacks a frozen source or mode") from error
            source_event_sha256 = tuple(event.fingerprint for event in source_events)
            if aggregation.source_event_sha256 != source_event_sha256:
                raise ValueError("PURE@B replay aggregation points to another source set")
            cells.append(
                BfclV4PublicV2PureAtBCellGradeRequest(
                    semantic_release_fingerprint=semantic_release_fingerprint,
                    task_ref=allocation.task_ref,
                    outer_seed_u64=outer_seed,
                    allocation=allocation,
                    source_nodes=nodes,
                    source_node_references=tuple(canonical_sha256(node) for node in nodes),
                    source_events=source_events,
                    source_event_sha256=source_event_sha256,
                    aggregation_spec=checked_campaign.pure_at_b_aggregation,
                    aggregation_spec_fingerprint=(
                        checked_campaign.pure_at_b_aggregation_fingerprint
                    ),
                    aggregation_result=aggregation.result,
                    aggregation_result_fingerprint=canonical_sha256(aggregation.result),
                    selected_canonical_response_fingerprint=(
                        bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
                            aggregation.result.selected_canonical_response
                        )
                    ),
                    decision_barrier_evidence_fingerprint=(
                        evaluation_unlock.barrier_evidence_fingerprint
                    ),
                    evaluation_unlock_fingerprint=evaluation_unlock.fingerprint,
                )
            )
    return BfclV4PublicV2PureAtBBatchGradeRequest(
        semantic_release_fingerprint=semantic_release_fingerprint,
        cells=tuple(cells),
        evaluation_unlock=evaluation_unlock,
        evaluation_unlock_fingerprint=evaluation_unlock.fingerprint,
        decision_barrier_evidence_fingerprint=(evaluation_unlock.barrier_evidence_fingerprint),
    )


class BfclV4PublicV2PureAtBCellGradeReceipt(ImmutableModel):
    """Minimal final-grade receipt for one modal response, never one sample."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_fingerprint: Sha256
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed
    task_ref: Annotated[str, Field(pattern=r"^holdout-[0-9]{2}$")]
    cell_grade_request_fingerprint: Sha256
    allocation: BfclV4PublicDevelopmentV2PureAtBAllocation
    allocation_fingerprint: Sha256
    source_event_sha256: Annotated[tuple[Sha256, ...], Field(min_length=6, max_length=7)]
    source_set_fingerprint: Sha256
    aggregation_spec_fingerprint: Sha256
    aggregation_result: BfclV4PublicDevelopmentV2PureAtBAggregationResult
    aggregation_result_fingerprint: Sha256
    selected_canonical_response: str | None
    selected_canonical_response_fingerprint: Sha256
    decision_barrier_evidence_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256
    loaded_question_bundle_fingerprint: Sha256
    trusted_authority_fingerprint: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT
    )
    trusted_grader_implementation_fingerprint: Literal[
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    ] = BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    pure_at_b_grader_implementation_fingerprint: Literal[
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    ] = BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    grader_source_sha256: Sha256
    pure_at_b_grader_source_sha256: Sha256
    worker_source_sha256: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256
    )
    worker_primitives_source_sha256: Literal[
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256
    ] = BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256
    checker_source_bundle_sha256: Literal[BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256] = (
        BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256
    )
    correct: bool
    trusted_grade_attempts_consumed: Literal[1] = 1
    isolated_worker_executed: bool
    exact_upstream_ast_checker_executed: bool
    possible_answers_read_in_isolated_trusted_worker: bool
    individual_sample_receipt_used: Literal[False] = False
    trusted_control_plane_only: Literal[True] = True
    candidate_visible: Literal[False] = False
    task_id_present: Literal[False] = False
    answers_present: Literal[False] = False
    answer_derived_identities_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_receipt(self) -> Self:
        if self.allocation.task_ref != self.task_ref or self.allocation_fingerprint != (
            canonical_sha256(self.allocation)
        ):
            raise ValueError("PURE@B receipt allocation lineage changed")
        if (
            len(self.source_event_sha256) != self.allocation.sample_count
            or len(set(self.source_event_sha256)) != self.allocation.sample_count
        ):
            raise ValueError("PURE@B receipt source count or uniqueness changed")
        if self.source_set_fingerprint != (
            bfcl_v4_public_v2_pure_at_b_source_set_fingerprint(self.source_event_sha256)
        ):
            raise ValueError("PURE@B receipt source-set lineage changed")
        if (
            self.aggregation_spec_fingerprint
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
            or self.aggregation_result_fingerprint != canonical_sha256(self.aggregation_result)
            or self.selected_canonical_response
            != self.aggregation_result.selected_canonical_response
            or self.selected_canonical_response_fingerprint
            != bfcl_v4_public_v2_pure_at_b_selected_response_fingerprint(
                self.selected_canonical_response
            )
        ):
            raise ValueError("PURE@B receipt modal lineage changed")
        worker_flags = (
            self.isolated_worker_executed,
            self.exact_upstream_ast_checker_executed,
            self.possible_answers_read_in_isolated_trusted_worker,
        )
        expected_worker = not self.aggregation_result.selected_no_response
        if worker_flags != (expected_worker, expected_worker, expected_worker):
            raise ValueError("PURE@B receipt worker flags differ from its selected response")
        if self.aggregation_result.selected_no_response and self.correct:
            raise ValueError("PURE@B no-response cell must deterministically grade false")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2PureAtBBatchGradeReceipt(ImmutableModel):
    """One atomic receipt for exactly 48 final modal-grade attempts."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_fingerprint: Sha256
    batch_grade_request_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256
    decision_barrier_evidence_fingerprint: Sha256
    cell_receipts: Annotated[
        tuple[BfclV4PublicV2PureAtBCellGradeReceipt, ...],
        Field(min_length=48, max_length=48),
    ]
    cell_receipt_fingerprints: Annotated[tuple[Sha256, ...], Field(min_length=48, max_length=48)]
    correct_count: Annotated[int, Field(ge=0, le=48, strict=True)]
    cell_count: Literal[48] = 48
    source_event_count: Literal[330] = 330
    trusted_grade_attempt_count: Literal[48] = 48
    individual_sample_grade_count: Literal[0] = 0
    isolated_worker_execution_count: Annotated[int, Field(ge=0, le=48, strict=True)]
    loaded_question_bundle_fingerprint: Sha256
    trusted_authority_fingerprint: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT
    )
    trusted_grader_implementation_fingerprint: Literal[
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    ] = BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    pure_at_b_grader_implementation_fingerprint: Literal[
        BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    ] = BFCL_V4_PUBLIC_V2_PURE_AT_B_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    grader_source_sha256: Sha256
    pure_at_b_grader_source_sha256: Sha256
    checker_source_bundle_sha256: Literal[BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256] = (
        BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256
    )
    all_label_free_modes_frozen_before_grading: Literal[True] = True
    source_sample_grades_present: Literal[False] = False
    trusted_control_plane_only: Literal[True] = True
    candidate_visible: Literal[False] = False
    task_ids_present: Literal[False] = False
    answers_present: Literal[False] = False
    answer_derived_identities_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_batch_receipt(self) -> Self:
        if self.cell_receipt_fingerprints != tuple(
            receipt.fingerprint for receipt in self.cell_receipts
        ):
            raise ValueError("PURE@B batch receipt cell fingerprints changed")
        if self.correct_count != sum(receipt.correct for receipt in self.cell_receipts):
            raise ValueError("PURE@B batch receipt correct count changed")
        if self.isolated_worker_execution_count != sum(
            receipt.isolated_worker_executed for receipt in self.cell_receipts
        ):
            raise ValueError("PURE@B batch receipt worker count changed")
        expected_coordinates = tuple(
            (outer_seed, allocation)
            for outer_seed in BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64
            for allocation in expected_bfcl_v4_public_development_v2_pure_at_b_allocation()
        )
        actual_coordinates = tuple(
            (receipt.outer_seed_u64, receipt.allocation) for receipt in self.cell_receipts
        )
        source_refs = tuple(
            source for receipt in self.cell_receipts for source in receipt.source_event_sha256
        )
        if actual_coordinates != expected_coordinates:
            raise ValueError("PURE@B batch receipt cell order changed")
        if len(source_refs) != self.source_event_count or len(set(source_refs)) != (
            self.source_event_count
        ):
            raise ValueError("PURE@B batch receipt lacks 330 unique source events")
        if (
            len({receipt.cell_grade_request_fingerprint for receipt in self.cell_receipts})
            != self.cell_count
            or len(set(self.cell_receipt_fingerprints)) != self.cell_count
        ):
            raise ValueError("PURE@B batch receipt repeats a request or receipt")
        shared = {
            (
                receipt.campaign_plan_fingerprint,
                receipt.node_schedule_content_sha256,
                receipt.semantic_release_fingerprint,
                receipt.decision_barrier_evidence_fingerprint,
                receipt.evaluation_unlock_fingerprint,
                receipt.loaded_question_bundle_fingerprint,
                receipt.grader_source_sha256,
                receipt.pure_at_b_grader_source_sha256,
            )
            for receipt in self.cell_receipts
        }
        expected = {
            (
                self.campaign_plan_fingerprint,
                self.node_schedule_content_sha256,
                self.semantic_release_fingerprint,
                self.decision_barrier_evidence_fingerprint,
                self.evaluation_unlock_fingerprint,
                self.loaded_question_bundle_fingerprint,
                self.grader_source_sha256,
                self.pure_at_b_grader_source_sha256,
            )
        }
        if shared != expected:
            raise ValueError("PURE@B batch receipt cells do not share one trusted lineage")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("bfcl_v4_public_v2_pure_at_b")
    or name.startswith("build_bfcl_v4_public_v2_pure_at_b")
]
