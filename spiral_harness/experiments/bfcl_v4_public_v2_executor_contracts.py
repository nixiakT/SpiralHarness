"""Provider-neutral journal contracts for the BFCL V4 public v2 rehearsal."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
    BfclV4PublicDevelopmentV2NodeKind,
    BfclV4PublicDevelopmentV2PureAtBAggregationResult,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256


class BfclV4PublicV2AttemptDisposition(StrEnum):
    """Terminal result of the sole provider attempt for one call slot."""

    SUCCEEDED = "succeeded"
    PROVIDER_FAILURE = "provider-failure"
    CRASH_RECOVERY_BURN = "crash-recovery-burn"


class BfclV4PublicV2ProposalDisposition(StrEnum):
    """Typed candidate validation outcome for one proposal slot."""

    VALID = "valid"
    DUPLICATE = "duplicate"
    NO_OP = "no-op"
    INVALID = "invalid"
    PROVIDER_FAILURE = "provider-failure"


class BfclV4PublicV2EventKind(StrEnum):
    """Journal transition type; every transition consumes one DAG node."""

    CALL = "call"
    NOMINATION = "nomination"
    DECISION = "decision"


class BfclV4PublicV2ControlValue(StrEnum):
    """Only legal deterministic nomination and promotion outputs."""

    PARENT_FALLBACK = "parent-fallback"
    CANDIDATE_0 = "candidate-0"
    CANDIDATE_1 = "candidate-1"
    CANDIDATE_2 = "candidate-2"
    PROMOTE = "promote"


class BfclV4PublicV2ProviderRequest(ImmutableModel):
    """Payload-free request identity passed to an injected transport."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    mutation_catalog_fingerprint: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    node_id: NonEmptyStr
    node_reference_sha256: Sha256
    campaign_call_slot: Annotated[int, Field(ge=0, lt=1_086, strict=True)]
    provider_seed_u63: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
    request_payload_sha256: Sha256
    decision_barrier_evidence_fingerprint: Sha256 | None = None
    evaluation_unlock_fingerprint: Sha256 | None = None
    request_payload_present: Literal[False] = False
    provider_free_contract_only: Literal[True] = True

    @model_validator(mode="after")
    def _bind_evaluation_authority_pair(self) -> Self:
        if (self.decision_barrier_evidence_fingerprint is None) is not (
            self.evaluation_unlock_fingerprint is None
        ):
            raise ValueError("decision barrier and evaluation unlock must be bound together")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ProviderAttempt(ImmutableModel):
    """Sanitized result returned by an injected one-shot provider transport."""

    schema_version: Literal["1"] = "1"
    disposition: BfclV4PublicV2AttemptDisposition
    canonical_response: str | None = None
    provider_response_fingerprint: Sha256 | None = None
    proposal_disposition: BfclV4PublicV2ProposalDisposition | None = None
    candidate_artifact_sha256: Sha256 | None = None
    provider_attempts_consumed: Annotated[int, Field(ge=0, le=1, strict=True)] = 1
    retry_count: Literal[0] = 0

    @model_validator(mode="after")
    def _close_attempt(self) -> Self:
        if self.disposition in {
            BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE,
            BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN,
        }:
            if any(
                value is not None
                for value in (
                    self.canonical_response,
                    self.provider_response_fingerprint,
                    self.candidate_artifact_sha256,
                )
            ):
                raise ValueError("provider failure cannot carry a response or candidate artifact")
            if self.proposal_disposition not in {
                None,
                BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE,
            }:
                raise ValueError("provider failure cannot claim a parsed proposal disposition")
        elif self.canonical_response is None or self.provider_response_fingerprint is None:
            raise ValueError("successful provider attempt requires a bound canonical response")
        if (self.proposal_disposition is BfclV4PublicV2ProposalDisposition.VALID) is not (
            self.candidate_artifact_sha256 is not None
        ):
            raise ValueError("only a valid proposal binds a candidate artifact")
        expected_attempts = int(
            self.disposition is not BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN
        )
        if self.provider_attempts_consumed != expected_attempts:
            raise ValueError("provider-attempt count differs from terminal disposition")
        return self


class BfclV4PublicV2TrustedGradeProjection(ImmutableModel):
    """Boolean plus opaque request/receipt lineage from the trusted grader."""

    schema_version: Literal["1"] = "1"
    correct: bool
    request_payload_sha256: Sha256
    provider_response_fingerprint: Sha256
    trusted_grade_request_fingerprint: Sha256
    trusted_grader_receipt_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256 | None = None
    trusted_plane_only: Literal[True] = True
    task_id_present: Literal[False] = False
    answers_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False


class BfclV4PublicV2JournalEvent(ImmutableModel):
    """One hash-chained terminal transition for the next exact DAG node."""

    schema_version: Literal["1"] = "1"
    sequence: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    previous_event_sha256: Sha256 | None
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    mutation_catalog_fingerprint: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    semantic_release_authenticity_attested: Literal[False] = False
    node_id: NonEmptyStr
    node_slot: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    node_reference_sha256: Sha256
    event_kind: BfclV4PublicV2EventKind
    request_fingerprint: Sha256
    request_payload_sha256: Sha256
    provider_attempt_disposition: BfclV4PublicV2AttemptDisposition | None = None
    provider_attempts_consumed: Annotated[int, Field(ge=0, le=1, strict=True)]
    executed_harness_variant: NonEmptyStr | None = None
    canonical_response: str | None = None
    provider_response_fingerprint: Sha256 | None = None
    binary_grade: bool | None = None
    trusted_grade_request_fingerprint: Sha256 | None = None
    trusted_grader_receipt_fingerprint: Sha256 | None = None
    trusted_grade_attempts_consumed: Annotated[int, Field(ge=0, le=1, strict=True)] = 0
    decision_barrier_evidence_fingerprint: Sha256 | None = None
    evaluation_unlock_fingerprint: Sha256 | None = None
    proposal_disposition: BfclV4PublicV2ProposalDisposition | None = None
    candidate_artifact_sha256: Sha256 | None = None
    control_value: BfclV4PublicV2ControlValue | None = None
    retry_count: Literal[0] = 0
    backfill_used: Literal[False] = False
    adaptive_stop_used: Literal[False] = False
    task_payload_present: Literal[False] = False
    possible_answer_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False

    @model_validator(mode="after")
    def _close_event_shape(self) -> Self:
        call_fields_present = (
            self.provider_attempt_disposition is not None,
            self.executed_harness_variant is not None,
        )
        if self.event_kind is BfclV4PublicV2EventKind.CALL:
            if call_fields_present != (True, True):
                raise ValueError("call event requires one terminal slot disposition")
            expected_attempts = int(
                self.provider_attempt_disposition
                is not BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN
            )
            if self.provider_attempts_consumed != expected_attempts:
                raise ValueError("call event provider-attempt count differs from disposition")
            if self.control_value is not None:
                raise ValueError("call event cannot carry a deterministic control value")
            if self.provider_attempt_disposition in {
                BfclV4PublicV2AttemptDisposition.PROVIDER_FAILURE,
                BfclV4PublicV2AttemptDisposition.CRASH_RECOVERY_BURN,
            } and (
                self.canonical_response is not None
                or self.provider_response_fingerprint is not None
                or self.binary_grade not in {None, False}
            ):
                raise ValueError("failed provider call must have no response and no true grade")
            if (
                self.provider_attempt_disposition is BfclV4PublicV2AttemptDisposition.SUCCEEDED
            ) is not (self.provider_response_fingerprint is not None):
                raise ValueError("successful call must bind one provider response fingerprint")
            if (self.trusted_grade_request_fingerprint is None) is not (
                self.trusted_grader_receipt_fingerprint is None
            ):
                raise ValueError("trusted grade request and receipt must be bound together")
            if (self.trusted_grader_receipt_fingerprint is not None) is not (
                self.trusted_grade_attempts_consumed == 1
            ):
                raise ValueError("trusted grade receipt must consume exactly one grader attempt")
            if (self.decision_barrier_evidence_fingerprint is None) is not (
                self.evaluation_unlock_fingerprint is None
            ):
                raise ValueError("evaluation barrier and unlock must be bound together")
        else:
            if call_fields_present != (False, False) or self.provider_attempts_consumed != 0:
                raise ValueError("control event cannot consume a provider attempt")
            if any(
                value is not None
                for value in (
                    self.canonical_response,
                    self.provider_response_fingerprint,
                    self.binary_grade,
                    self.trusted_grade_request_fingerprint,
                    self.trusted_grader_receipt_fingerprint,
                    self.decision_barrier_evidence_fingerprint,
                    self.evaluation_unlock_fingerprint,
                    self.proposal_disposition,
                    self.candidate_artifact_sha256,
                )
            ):
                raise ValueError("control event cannot carry provider or grader output")
            if self.trusted_grade_attempts_consumed != 0:
                raise ValueError("control event cannot consume a trusted grader attempt")
            if self.control_value is None:
                raise ValueError("control event requires its deterministic value")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2CallReservation(ImmutableModel):
    """Durable pre-call claim; recovery burns it instead of calling again."""

    schema_version: Literal["1"] = "1"
    sequence: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    expected_tail_event_sha256: Sha256 | None
    node: BfclV4PublicDevelopmentV2DagNode
    node_reference_sha256: Sha256
    request: BfclV4PublicV2ProviderRequest
    request_fingerprint: Sha256
    slot_reserved_before_provider_call: Literal[True] = True
    provider_call_started_attested: Literal[False] = False
    recovery_action: Literal["burn-without-provider-retry"] = "burn-without-provider-retry"

    @model_validator(mode="after")
    def _bind_reserved_call(self) -> Self:
        node = self.node
        request = self.request
        if (
            not node.consumes_model_call
            or node.node_slot != self.sequence
            or self.node_reference_sha256 != canonical_sha256(node)
            or self.request_fingerprint != request.fingerprint
            or request.node_id != node.node_id
            or request.node_reference_sha256 != self.node_reference_sha256
            or request.campaign_call_slot != node.campaign_call_slot
            or request.provider_seed_u63 != node.provider_seed_u63
        ):
            raise ValueError("call reservation differs from its frozen node or request")
        if (self.expected_tail_event_sha256 is None) is not (self.sequence == 0):
            raise ValueError("call reservation expected tail differs from its sequence")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ControlRecord(ImmutableModel):
    """Replay-derived nomination or decision."""

    schema_version: Literal["1"] = "1"
    node_id: NonEmptyStr
    outer_seed_u64: Annotated[int, Field(ge=0, strict=True)]
    arm: BfclV4PublicDevelopmentV2Arm
    kind: Literal[
        BfclV4PublicDevelopmentV2NodeKind.NOMINATION,
        BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION,
    ]
    value: BfclV4PublicV2ControlValue


class BfclV4PublicV2PureAtBAggregationRecord(ImmutableModel):
    """Replay-derived label-free aggregation for one seed/task cell."""

    schema_version: Literal["1"] = "1"
    outer_seed_u64: Annotated[int, Field(ge=0, strict=True)]
    task_ref: NonEmptyStr
    source_event_sha256: Annotated[tuple[Sha256, ...], Field(min_length=6, max_length=7)]
    result: BfclV4PublicDevelopmentV2PureAtBAggregationResult

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2PureAtBCellGradeProjection(ImmutableModel):
    """Opaque trusted receipt identity and Boolean for one final modal cell."""

    schema_version: Literal["1"] = "1"
    outer_seed_u64: Annotated[int, Field(ge=0, strict=True)]
    task_ref: NonEmptyStr
    aggregation_record_fingerprint: Sha256
    cell_grade_request_fingerprint: Sha256
    cell_grade_receipt_fingerprint: Sha256
    correct: bool


class BfclV4PublicV2PureAtBBatchGradeProjection(ImmutableModel):
    """Minimal 48-cell projection of one trusted aggregate batch receipt."""

    schema_version: Literal["1"] = "1"
    batch_grade_request_fingerprint: Sha256
    batch_grade_receipt_fingerprint: Sha256
    decision_barrier_evidence_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256
    cells: Annotated[
        tuple[BfclV4PublicV2PureAtBCellGradeProjection, ...],
        Field(min_length=48, max_length=48),
    ]
    correct_count: Annotated[int, Field(ge=0, le=48, strict=True)]
    source_event_count: Literal[330] = 330
    trusted_grade_attempt_count: Literal[48] = 48
    individual_sample_grade_count: Literal[0] = 0

    @model_validator(mode="after")
    def _close_batch_projection(self) -> Self:
        if self.correct_count != sum(cell.correct for cell in self.cells):
            raise ValueError("PURE@B projected correct count changed")
        coordinates = tuple((cell.outer_seed_u64, cell.task_ref) for cell in self.cells)
        if len(set(coordinates)) != 48:
            raise ValueError("PURE@B projected cells repeat a seed/task coordinate")
        return self


class BfclV4PublicV2ReplayState(ImmutableModel):
    """State reconstructed from events rather than trusted journal counters."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    next_node_slot: Annotated[int, Field(ge=0, le=1_098, strict=True)]
    tail_event_sha256: Sha256 | None
    completed_node_count: Annotated[int, Field(ge=0, le=1_098, strict=True)]
    burned_call_slot_count: Annotated[int, Field(ge=0, le=1_086, strict=True)]
    succeeded_call_count: Annotated[int, Field(ge=0, le=1_086, strict=True)]
    failed_call_count: Annotated[int, Field(ge=0, le=1_086, strict=True)]
    provider_attempt_count: Annotated[int, Field(ge=0, le=1_086, strict=True)]
    crash_recovery_burn_count: Annotated[int, Field(ge=0, le=1_086, strict=True)]
    nominations: Annotated[tuple[BfclV4PublicV2ControlRecord, ...], Field(max_length=6)]
    decisions: Annotated[tuple[BfclV4PublicV2ControlRecord, ...], Field(max_length=6)]
    pure_at_b_aggregations: Annotated[
        tuple[BfclV4PublicV2PureAtBAggregationRecord, ...],
        Field(max_length=48),
    ]
    complete: bool
    provider_attempts_per_call: Literal[1] = 1
    retries_used: Literal[0] = 0
    backfills_used: Literal[0] = 0
    adaptive_stops_used: Literal[0] = 0
    provider_free_rehearsal_only: Literal[True] = True
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_counts(self) -> Self:
        if self.completed_node_count != self.next_node_slot:
            raise ValueError("replayed completed-node count differs from next node slot")
        if self.burned_call_slot_count != self.succeeded_call_count + self.failed_call_count:
            raise ValueError("replayed burned-call count differs from terminal dispositions")
        if self.provider_attempt_count + self.crash_recovery_burn_count != (
            self.burned_call_slot_count
        ):
            raise ValueError("provider attempts plus crash burns differ from consumed call slots")
        expected_complete = self.next_node_slot == 1_098
        if self.complete is not expected_complete:
            raise ValueError("replayed completion flag differs from exact node count")
        if self.complete and (
            self.burned_call_slot_count != 1_086
            or len(self.nominations) != 6
            or len(self.decisions) != 6
            or len(self.pure_at_b_aggregations) != 48
        ):
            raise ValueError("complete v2 replay lacks exact calls, controls, or aggregations")
        return self


class BfclV4PublicV2JournalSnapshot(ImmutableModel):
    """Portable append-only event chain."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    events: Annotated[tuple[BfclV4PublicV2JournalEvent, ...], Field(max_length=1_098)]
    tail_event_sha256: Sha256 | None
    pending_call_reservation: BfclV4PublicV2CallReservation | None = None
    append_only: Literal[True] = True
    event_count_ceiling: Literal[1_098] = 1_098

    @model_validator(mode="after")
    def _bind_tail(self) -> Self:
        expected = self.events[-1].fingerprint if self.events else None
        if self.tail_event_sha256 != expected:
            raise ValueError("journal snapshot tail differs from its final event")
        pending = self.pending_call_reservation
        if pending is not None and (
            len(self.events) == 1_098
            or pending.sequence != len(self.events)
            or pending.expected_tail_event_sha256 != self.tail_event_sha256
            or pending.request.campaign_plan_fingerprint != self.campaign_plan_fingerprint
            or pending.request.node_schedule_content_sha256 != self.node_schedule_content_sha256
            or pending.request.runtime_fingerprint != self.runtime_fingerprint
            or pending.request.semantic_release_fingerprint != self.semantic_release_fingerprint
        ):
            raise ValueError("pending call reservation differs from its journal prefix")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ExecutionReceipt(ImmutableModel):
    """Terminal provider-free rehearsal receipt."""

    schema_version: Literal["1"] = "1"
    journal: BfclV4PublicV2JournalSnapshot
    replayed_state: BfclV4PublicV2ReplayState
    pure_at_b_batch_grade: BfclV4PublicV2PureAtBBatchGradeProjection | None = None
    replayed_independently: Literal[True] = True
    provider_transport_injected: Literal[True] = True
    real_api_called_by_core: Literal[False] = False
    score_bearing_execution: Literal[False] = False
    public_development_only: Literal[True] = True
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_terminal_state(self) -> Self:
        if not self.replayed_state.complete:
            raise ValueError("terminal execution receipt requires a complete replay")
        if self.journal.pending_call_reservation is not None:
            raise ValueError("terminal execution receipt cannot retain a pending reservation")
        if (
            self.journal.campaign_plan_fingerprint != self.replayed_state.campaign_plan_fingerprint
            or self.journal.node_schedule_content_sha256
            != self.replayed_state.node_schedule_content_sha256
            or self.journal.runtime_fingerprint != self.replayed_state.runtime_fingerprint
            or self.journal.semantic_release_fingerprint
            != self.replayed_state.semantic_release_fingerprint
            or self.journal.tail_event_sha256 != self.replayed_state.tail_event_sha256
        ):
            raise ValueError("terminal journal and independently replayed state differ")
        batch = self.pure_at_b_batch_grade
        if batch is not None:
            expected = tuple(
                (record.outer_seed_u64, record.task_ref, record.fingerprint)
                for record in self.replayed_state.pure_at_b_aggregations
            )
            observed = tuple(
                (cell.outer_seed_u64, cell.task_ref, cell.aggregation_record_fingerprint)
                for cell in batch.cells
            )
            if observed != expected:
                raise ValueError("PURE@B batch projection differs from replayed aggregations")
        return self


__all__ = [name for name in globals() if name.startswith("Bfcl")]
