"""Contracts for one trusted-runner four-context HarnessFault development invocation.

The represented study is permanently nonreportable development evidence.  It closes
SS/MS/SM/MM in one ledger over a shared exploration-fit block and a disjoint
shared gate block.  No value admitted here can be promoted to sealed or
confirmatory evidence, or used to attest the revision actually served by a
provider, independent freeze timing, or one-time gate-block consumption.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.benchmark._harness_fault_cases import PartitionEvaluationGrant
from spiral_harness.benchmark.harness_fault_compiler import (
    HARNESS_FAULT_COMPILATION_MEDIA_TYPE,
    HarnessRole,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    AttemptBudget,
    FrozenModelSpec,
)
from spiral_harness.experiments.confirmatory_arms import (
    FaultFactorialCell,
    OptimizerFeedbackMode,
    PromotionRule,
    make_fault_factorial_profile,
)
from spiral_harness.experiments.harness_fault_development_contracts import (
    HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE,
    DevelopmentInvocationMode,
    FaultDevelopmentGate,
    FaultDevelopmentObservation,
    FeedbackEvidenceLinkage,
    ModelAuthoredFaultProposal,
)

HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.fault-four-context-candidate-freeze.v1+json"
)
HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE = (
    "application/vnd.spiral-harness.fault-four-context-gate-batch.v1+json"
)
HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.fault-four-context-development-result.v1+json"
)

FAULT_CONTEXT_ORDER = (
    FaultFactorialCell.SS,
    FaultFactorialCell.MS,
    FaultFactorialCell.SM,
    FaultFactorialCell.MM,
)
_LINKAGE_BY_CELL = {
    FaultFactorialCell.SS: FeedbackEvidenceLinkage.SCORE_ONLY,
    FaultFactorialCell.MS: FeedbackEvidenceLinkage.SOURCE_LINKED,
    FaultFactorialCell.SM: FeedbackEvidenceLinkage.SCORE_ONLY,
    FaultFactorialCell.MM: FeedbackEvidenceLinkage.SOURCE_LINKED,
}


class HarnessFaultFourContextBlock(ImmutableModel):
    """One treatment namespace over disjoint shared fit and gate blocks."""

    schema_version: Literal["1"] = "1"
    cell: FaultFactorialCell
    feedback_linkage: FeedbackEvidenceLinkage
    invocation_mode: DevelopmentInvocationMode
    model_spec: FrozenModelSpec
    fit_task_count: Annotated[int, Field(ge=4, strict=True)]
    gate_task_count: Annotated[int, Field(ge=4, strict=True)]
    proposal_seed: Annotated[int, Field(ge=0, strict=True)]
    solver_master_seed: Annotated[int, Field(ge=0, strict=True)]
    max_tokens_per_call: Annotated[int, Field(ge=1, strict=True)]
    fit_partition_grant: PartitionEvaluationGrant
    gate_partition_grant: PartitionEvaluationGrant

    @model_validator(mode="after")
    def exact_cell_and_public_blocks(self) -> Self:
        profile = make_fault_factorial_profile(self.cell)
        if self.feedback_linkage is not _LINKAGE_BY_CELL[self.cell]:
            raise ValueError("context does not use the canonical SS/MS/SM/MM feedback linkage")
        if (
            profile.optimizer_feedback is OptimizerFeedbackMode.SCORE_ONLY
            and self.feedback_linkage is not FeedbackEvidenceLinkage.SCORE_ONLY
        ) or (
            profile.optimizer_feedback is OptimizerFeedbackMode.MECHANISM_VISIBLE
            and self.feedback_linkage is not FeedbackEvidenceLinkage.SOURCE_LINKED
        ):
            raise ValueError("feedback linkage differs from the factorial optimizer profile")
        if self.fit_partition_grant.partition is not ProtocolPartition.EXPLORATION:
            raise ValueError("four-context fit feedback requires the exploration partition")
        if self.gate_partition_grant.partition is not ProtocolPartition.GATE:
            raise ValueError("four-context attribution requires the gate partition")
        if (
            self.fit_partition_grant.public_commitment
            != self.gate_partition_grant.public_commitment
        ):
            raise ValueError("fit and gate grants must come from one exact authority commitment")
        fit_commitment = self.fit_partition_grant.public_commitment.partition(
            ProtocolPartition.EXPLORATION
        )
        gate_commitment = self.gate_partition_grant.public_commitment.partition(
            ProtocolPartition.GATE
        )
        if fit_commitment.scenario_count != self.fit_task_count:
            raise ValueError("fit authority block size differs from fit_task_count")
        if gate_commitment.scenario_count != self.gate_task_count:
            raise ValueError("gate authority block size differs from gate_task_count")
        return self

    @property
    def optimizer_feedback(self) -> OptimizerFeedbackMode:
        return make_fault_factorial_profile(self.cell).optimizer_feedback

    @property
    def promotion_rule(self) -> PromotionRule:
        return make_fault_factorial_profile(self.cell).promotion_rule

    @property
    def fit_stage_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-fit-stage/v1",
                "cell": self.cell,
                "feedback_linkage": self.feedback_linkage,
                "invocation_mode": self.invocation_mode,
                "model_spec": self.model_spec,
                "fit_task_count": self.fit_task_count,
                "proposal_seed": self.proposal_seed,
                "solver_master_seed": self.solver_master_seed,
                "max_tokens_per_call": self.max_tokens_per_call,
                "fit_partition_grant": self.fit_partition_grant,
                "call_count": self.candidate_definition_call_count,
            }
        )

    @property
    def fit_latent_block_id(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-fit-block/v1",
                "partition_grant": self.fit_partition_grant,
            }
        )

    @property
    def gate_latent_block_id(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-gate-block/v1",
                "partition_grant": self.gate_partition_grant,
            }
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def candidate_definition_call_count(self) -> int:
        return self.fit_task_count + 1

    @property
    def gate_attribution_call_count(self) -> int:
        return 4 * self.gate_task_count

    @property
    def model_call_count(self) -> int:
        return self.candidate_definition_call_count + self.gate_attribution_call_count

    @property
    def block_id(self) -> str:
        """Condition namespace ID; fit and gate blocks are shared, not state."""

        return canonical_sha256(
            {
                "schema": "spiral-harness/fault-four-context-namespace/v1",
                "cell": self.cell,
                "context_fingerprint": self.fingerprint,
                "fit_latent_block_id": self.fit_latent_block_id,
                "gate_latent_block_id": self.gate_latent_block_id,
            }
        )


class HarnessFaultFourContextPlan(ImmutableModel):
    """Exact matched SS/MS/SM/MM topology for one trusted runner invocation."""

    schema_version: Literal["1"] = "1"
    blocks: Annotated[
        tuple[HarnessFaultFourContextBlock, ...],
        Field(min_length=4, max_length=4),
    ]
    model_spec: FrozenModelSpec
    max_model_calls: Annotated[int, Field(ge=84, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=84, strict=True)]
    context_count: Literal[4] = 4
    one_runner_invocation_required: Literal[True] = True
    one_fit_block_shared_across_contexts: Literal[True] = True
    one_gate_block_shared_across_contexts: Literal[True] = True
    fit_gate_task_source_group_units_disjoint_required: Literal[True] = True
    paired_task_rosters_and_solver_seeds: Literal[True] = True
    shared_model_visible_mutable_state_between_contexts_permitted: Literal[False] = False
    context_local_feedback_required: Literal[True] = True
    distinct_backend_client_per_context_required: Literal[True] = True
    candidate_defining_calls_required_before_gate_attribution: Literal[True] = True
    trusted_runner_opens_gate_after_freeze: Literal[True] = True
    gate_opening_timing_independently_attested: Literal[False] = False
    gate_block_one_time_consumption_attested: Literal[False] = False
    cross_process_atomicity_attested: Literal[False] = False
    gate_dispatch_after_complete_cross_context_batch: Literal[True] = True

    @field_validator("blocks")
    @classmethod
    def canonical_blocks(
        cls,
        values: tuple[HarnessFaultFourContextBlock, ...],
    ) -> tuple[HarnessFaultFourContextBlock, ...]:
        by_cell = {value.cell: value for value in values}
        if len(by_cell) != len(values) or frozenset(by_cell) != frozenset(FAULT_CONTEXT_ORDER):
            raise ValueError("four-context plan requires exactly SS, MS, SM, and MM")
        return tuple(by_cell[cell] for cell in FAULT_CONTEXT_ORDER)

    @model_validator(mode="after")
    def exact_matched_coordinates_and_budget(self) -> Self:
        anchor = self.blocks[0]
        excluded = {"cell", "feedback_linkage"}
        anchor_coordinates = anchor.model_dump(
            mode="python",
            exclude=excluded,
            round_trip=True,
            warnings="none",
        )
        if anchor.model_spec != self.model_spec:
            raise ValueError("top-level model spec differs from the context plans")
        for block in self.blocks[1:]:
            coordinates = block.model_dump(
                mode="python",
                exclude=excluded,
                round_trip=True,
                warnings="none",
            )
            if coordinates != anchor_coordinates:
                raise ValueError("factorial contexts differ outside feedback and gate treatment")
            if block.model_spec != self.model_spec:
                raise ValueError("every context must share one exact FrozenModelSpec")

        fit_grant = self.blocks[0].fit_partition_grant
        gate_grant = self.blocks[0].gate_partition_grant
        if any(block.fit_partition_grant != fit_grant for block in self.blocks[1:]):
            raise ValueError("all contexts must share one exact fit authority block")
        if any(block.gate_partition_grant != gate_grant for block in self.blocks[1:]):
            raise ValueError("all contexts must share one exact gate authority block")
        if len({block.fit_latent_block_id for block in self.blocks}) != 1:
            raise ValueError("factorial contexts do not bind one paired fit block")
        if len({block.gate_latent_block_id for block in self.blocks}) != 1:
            raise ValueError("factorial contexts do not bind one paired gate block")
        if len({block.block_id for block in self.blocks}) != self.context_count:
            raise ValueError("factorial contexts require four distinct namespace identities")

        expected_calls = sum(block.model_call_count for block in self.blocks)
        expected_tokens = expected_calls * anchor.max_tokens_per_call
        if self.max_model_calls != expected_calls:
            raise ValueError("max_model_calls must equal the exact four-context call topology")
        if self.max_total_tokens != expected_tokens:
            raise ValueError("max_total_tokens must reserve every four-context call slot")
        return self

    @property
    def candidate_definition_call_count(self) -> int:
        return sum(block.candidate_definition_call_count for block in self.blocks)

    @property
    def gate_attribution_call_count(self) -> int:
        return sum(block.gate_attribution_call_count for block in self.blocks)

    @property
    def attempt_budget(self) -> AttemptBudget:
        return AttemptBudget(
            max_attempts=self.max_model_calls,
            max_total_tokens=self.max_total_tokens,
            max_tokens_per_attempt=self.blocks[0].max_tokens_per_call,
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_harness_fault_four_context_plan(
    *,
    fit_partition_grant: PartitionEvaluationGrant,
    gate_partition_grant: PartitionEvaluationGrant,
    invocation_mode: DevelopmentInvocationMode,
    model_spec: FrozenModelSpec,
    fit_task_count: int,
    gate_task_count: int,
    proposal_seed: int,
    solver_master_seed: int,
    max_tokens_per_call: int,
) -> HarnessFaultFourContextPlan:
    """Build the sole canonical four-context development plan."""

    checked_fit_grant = PartitionEvaluationGrant.model_validate(fit_partition_grant, strict=True)
    checked_gate_grant = PartitionEvaluationGrant.model_validate(gate_partition_grant, strict=True)
    checked_spec = FrozenModelSpec.model_validate(model_spec, strict=True)
    blocks = []
    for cell in FAULT_CONTEXT_ORDER:
        blocks.append(
            HarnessFaultFourContextBlock(
                cell=cell,
                feedback_linkage=_LINKAGE_BY_CELL[cell],
                invocation_mode=invocation_mode,
                model_spec=checked_spec,
                fit_task_count=fit_task_count,
                gate_task_count=gate_task_count,
                proposal_seed=proposal_seed,
                solver_master_seed=solver_master_seed,
                max_tokens_per_call=max_tokens_per_call,
                fit_partition_grant=checked_fit_grant,
                gate_partition_grant=checked_gate_grant,
            )
        )
    total_calls = sum(block.model_call_count for block in blocks)
    return HarnessFaultFourContextPlan(
        blocks=tuple(blocks),
        model_spec=checked_spec,
        max_model_calls=total_calls,
        max_total_tokens=total_calls * max_tokens_per_call,
    )


class FrozenFaultContextCandidate(ImmutableModel):
    """One model-authored candidate committed inside the joint freeze."""

    schema_version: Literal["1"] = "1"
    cell: FaultFactorialCell
    block_id: Sha256
    plan_fingerprint: Sha256
    feedback_ref: ArtifactRef
    proposer_harness_ref: ArtifactRef
    proposal: ModelAuthoredFaultProposal
    runtime_producer_id: Sha256
    parent_harness_ref: ArtifactRef
    compilation_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef

    @model_validator(mode="after")
    def exact_refs(self) -> Self:
        if self.feedback_ref.media_type != HARNESS_FAULT_DEVELOPMENT_FEEDBACK_MEDIA_TYPE:
            raise ValueError("candidate feedback_ref declares the wrong media type")
        if self.proposer_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("candidate proposer_harness_ref declares the wrong media type")
        if self.parent_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("candidate parent_harness_ref declares the wrong media type")
        if self.candidate_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("candidate harness_ref declares the wrong media type")
        if self.compilation_ref.media_type != HARNESS_FAULT_COMPILATION_MEDIA_TYPE:
            raise ValueError("candidate compilation_ref declares the wrong media type")
        if self.proposal.feedback_ref != self.feedback_ref:
            raise ValueError("candidate proposal differs from its frozen feedback ref")
        if self.proposal.proposer_harness_ref != self.proposer_harness_ref:
            raise ValueError("candidate proposal differs from its frozen proposer harness")
        return self


def fault_context_candidate_freeze_id(
    *,
    plan_fingerprint: str,
    candidates: tuple[FrozenFaultContextCandidate, ...],
) -> str:
    ordered = tuple(sorted(candidates, key=lambda item: FAULT_CONTEXT_ORDER.index(item.cell)))
    return canonical_sha256(
        {
            "schema": "spiral-harness/fault-four-context-joint-candidate-freeze/v1",
            "plan_fingerprint": plan_fingerprint,
            "candidates": ordered,
        }
    )


class JointFaultContextCandidateFreeze(ImmutableModel):
    """Joint candidate content and the trusted runner's declared call boundary."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    freeze_id: Sha256
    candidates: Annotated[
        tuple[FrozenFaultContextCandidate, ...], Field(min_length=4, max_length=4)
    ]
    candidate_definition_call_count: Annotated[int, Field(ge=20, strict=True)]
    last_candidate_definition_sequence: Annotated[int, Field(ge=19, strict=True)]
    first_attribution_sequence: Annotated[int, Field(ge=20, strict=True)]
    all_context_candidates_frozen_together: Literal[True] = True
    candidate_defining_calls_precede_gate_attribution_in_ledger: Literal[True] = True
    trusted_runner_persists_freeze_before_gate_opening: Literal[True] = True
    freeze_persistence_timing_independently_attested: Literal[False] = False
    frozen_content_replayed_before_cross_context_gate_batch: Literal[True] = True

    @field_validator("candidates")
    @classmethod
    def canonical_candidates(
        cls,
        values: tuple[FrozenFaultContextCandidate, ...],
    ) -> tuple[FrozenFaultContextCandidate, ...]:
        by_cell = {value.cell: value for value in values}
        if len(by_cell) != len(values) or frozenset(by_cell) != frozenset(FAULT_CONTEXT_ORDER):
            raise ValueError("joint freeze requires exactly one candidate per factorial cell")
        return tuple(by_cell[cell] for cell in FAULT_CONTEXT_ORDER)

    @model_validator(mode="after")
    def exact_freeze_boundary(self) -> Self:
        if self.last_candidate_definition_sequence != self.candidate_definition_call_count - 1:
            raise ValueError("last candidate definition sequence differs from its call count")
        if self.first_attribution_sequence != self.candidate_definition_call_count:
            raise ValueError("attribution must start immediately after the joint freeze boundary")
        expected = fault_context_candidate_freeze_id(
            plan_fingerprint=self.plan_fingerprint,
            candidates=self.candidates,
        )
        if self.freeze_id != expected:
            raise ValueError("freeze_id does not bind all four immutable candidates")
        return self


class HarnessFaultContextDevelopmentClosure(ImmutableModel):
    """Fit feedback and disjoint gate-attribution outputs for one cell."""

    schema_version: Literal["1"] = "1"
    cell: FaultFactorialCell
    block_id: Sha256
    plan_fingerprint: Sha256
    runtime_producer_id: Sha256
    parent_harness_ref: ArtifactRef
    compilation_ref: ArtifactRef
    feedback_ref: ArtifactRef
    proposer_harness_ref: ArtifactRef
    proposal: ModelAuthoredFaultProposal
    fit_parent_observations: Annotated[tuple[FaultDevelopmentObservation, ...], Field(min_length=4)]
    gate_observations: Annotated[tuple[FaultDevelopmentObservation, ...], Field(min_length=16)]
    gate: FaultDevelopmentGate

    @field_validator("fit_parent_observations", "gate_observations")
    @classmethod
    def canonical_observations(
        cls,
        values: tuple[FaultDevelopmentObservation, ...],
    ) -> tuple[FaultDevelopmentObservation, ...]:
        return tuple(sorted(values, key=lambda item: (item.harness_role.value, item.task_id)))


class FaultContextGateRecord(ImmutableModel):
    schema_version: Literal["1"] = "1"
    cell: FaultFactorialCell
    block_id: Sha256
    gate: FaultDevelopmentGate


class HarnessFaultFourContextGateBatch(ImmutableModel):
    """The four gates dispatched only after the complete attribution batch."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    candidate_freeze_ref: ArtifactRef
    candidate_freeze_id: Sha256
    records: Annotated[tuple[FaultContextGateRecord, ...], Field(min_length=4, max_length=4)]
    model_call_count_before_gate_batch: Annotated[int, Field(ge=84, strict=True)]
    all_four_attribution_quartets_complete: Literal[True] = True
    no_gate_feedback_released_during_batch: Literal[True] = True
    automatic_no_human_ranking: Literal[True] = True

    @field_validator("records")
    @classmethod
    def canonical_records(
        cls,
        values: tuple[FaultContextGateRecord, ...],
    ) -> tuple[FaultContextGateRecord, ...]:
        by_cell = {value.cell: value for value in values}
        if len(by_cell) != len(values) or frozenset(by_cell) != frozenset(FAULT_CONTEXT_ORDER):
            raise ValueError("gate batch requires exactly SS, MS, SM, and MM")
        return tuple(by_cell[cell] for cell in FAULT_CONTEXT_ORDER)

    @model_validator(mode="after")
    def exact_freeze_media(self) -> Self:
        if self.candidate_freeze_ref.media_type != HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE:
            raise ValueError("candidate_freeze_ref declares the wrong media type")
        return self


class HarnessFaultFourContextDevelopmentResult(ImmutableModel):
    """Offline-replayable, permanently non-reportable four-context closure."""

    schema_version: Literal["1"] = "1"
    kind: Literal["model-output-driven-harness-fault-four-context-development"] = (
        "model-output-driven-harness-fault-four-context-development"
    )
    plan: HarnessFaultFourContextPlan
    plan_fingerprint: Sha256
    candidate_freeze: JointFaultContextCandidateFreeze
    candidate_freeze_ref: ArtifactRef
    contexts: Annotated[
        tuple[HarnessFaultContextDevelopmentClosure, ...],
        Field(min_length=4, max_length=4),
    ]
    gate_batch: HarnessFaultFourContextGateBatch
    gate_batch_ref: ArtifactRef
    ledger_id: NonEmptyStr
    attempt_budget: AttemptBudget
    ledger_tail_ref: ArtifactRef
    model_call_count: Annotated[int, Field(ge=84, strict=True)]
    one_trusted_runner_invocation_completed: Literal[True] = True
    cross_process_atomicity_attested: Literal[False] = False
    exact_nominal_model_spec_shared: Literal[True] = True
    one_paired_fit_task_block_shared: Literal[True] = True
    one_paired_gate_task_block_shared: Literal[True] = True
    fit_gate_task_source_group_units_disjoint: Literal[True] = True
    paired_solver_seeds_across_contexts: Literal[True] = True
    runner_serialized_cross_context_feedback: Literal[False] = False
    distinct_backend_client_instances_runtime_checked: Literal[True] = True
    provider_transport_or_session_isolation_attested: Literal[False] = False
    raw_outputs_and_ledger_offline_replayable: Literal[True] = True
    candidate_definition_before_gate_attribution_calls_offline_replayable: Literal[True] = True
    trusted_runner_opened_gate_after_freeze: Literal[True] = True
    freeze_persistence_timing_independently_attested: Literal[False] = False
    gate_opening_timing_independently_attested: Literal[False] = False
    gate_block_one_time_consumption_attested: Literal[False] = False
    cross_context_feedback_disclosed: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    exact_served_revision_claim_allowed: Literal[False] = False
    development_data_only: Literal[True] = True
    gate_partition_consumed_for_development: Literal[True] = True
    sealed_evidence: Literal[False] = False
    reportable_benchmark_result: Literal[False] = False
    confirmatory_inference: Literal[False] = False
    permanently_nonreportable_development_artifact: Literal[True] = True

    @field_validator("contexts")
    @classmethod
    def canonical_contexts(
        cls,
        values: tuple[HarnessFaultContextDevelopmentClosure, ...],
    ) -> tuple[HarnessFaultContextDevelopmentClosure, ...]:
        by_cell = {value.cell: value for value in values}
        if len(by_cell) != len(values) or frozenset(by_cell) != frozenset(FAULT_CONTEXT_ORDER):
            raise ValueError("result requires exactly one closure per factorial cell")
        return tuple(by_cell[cell] for cell in FAULT_CONTEXT_ORDER)

    @model_validator(mode="after")
    def exact_atomic_closure(self) -> Self:
        if self.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("plan_fingerprint differs from the embedded four-context plan")
        if self.attempt_budget != self.plan.attempt_budget:
            raise ValueError("attempt budget differs from the frozen four-context plan")
        if self.model_call_count != self.plan.max_model_calls:
            raise ValueError("model_call_count differs from the exact four-context topology")
        if self.ledger_tail_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("ledger_tail_ref declares the wrong media type")
        if self.candidate_freeze_ref.media_type != HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE:
            raise ValueError("candidate_freeze_ref declares the wrong media type")
        if self.gate_batch_ref.media_type != HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE:
            raise ValueError("gate_batch_ref declares the wrong media type")
        if self.candidate_freeze.plan_fingerprint != self.plan_fingerprint:
            raise ValueError("candidate freeze belongs to another four-context plan")
        if (
            self.candidate_freeze.candidate_definition_call_count
            != self.plan.candidate_definition_call_count
        ):
            raise ValueError("candidate freeze occurs at the wrong exact call boundary")
        if self.gate_batch.plan_fingerprint != self.plan_fingerprint:
            raise ValueError("gate batch belongs to another four-context plan")
        if self.gate_batch.candidate_freeze_ref != self.candidate_freeze_ref:
            raise ValueError("gate batch differs from the joint candidate freeze")
        if self.gate_batch.candidate_freeze_id != self.candidate_freeze.freeze_id:
            raise ValueError("gate batch carries another candidate freeze identity")
        if self.gate_batch.model_call_count_before_gate_batch != self.model_call_count:
            raise ValueError("gate batch was not placed after all exact model calls")

        blocks = {block.cell: block for block in self.plan.blocks}
        candidates = {item.cell: item for item in self.candidate_freeze.candidates}
        gates = {item.cell: item for item in self.gate_batch.records}
        all_outcomes: set[str] = set()
        for context in self.contexts:
            block = blocks[context.cell]
            candidate = candidates[context.cell]
            gate = gates[context.cell]
            if (
                context.block_id != block.block_id
                or context.plan_fingerprint != block.fingerprint
                or candidate.block_id != block.block_id
                or candidate.plan_fingerprint != block.fingerprint
            ):
                raise ValueError("context closure or candidate belongs to another fresh block")
            if gate.block_id != block.block_id or gate.gate != context.gate:
                raise ValueError("gate batch differs from its context closure")
            if (
                candidate.feedback_ref != context.feedback_ref
                or candidate.proposer_harness_ref != context.proposer_harness_ref
                or candidate.proposal != context.proposal
                or candidate.runtime_producer_id != context.runtime_producer_id
                or candidate.parent_harness_ref != context.parent_harness_ref
                or candidate.compilation_ref != context.compilation_ref
            ):
                raise ValueError("joint freeze differs from a context's candidate definition")
            if len(context.fit_parent_observations) != block.fit_task_count:
                raise ValueError("context does not contain its complete fit parent pass")
            if any(
                item.harness_role is not HarnessRole.FAULTY_PARENT
                for item in context.fit_parent_observations
            ):
                raise ValueError("fit feedback admits faulty-parent observations only")
            if (
                len({item.task_id for item in context.fit_parent_observations})
                != block.fit_task_count
            ):
                raise ValueError("fit feedback repeats a task")
            expected_observations = 4 * block.gate_task_count
            if len(context.gate_observations) != expected_observations:
                raise ValueError("context does not contain its complete attribution quartet")
            for role in (
                HarnessRole.FAULTY_PARENT,
                HarnessRole.CANDIDATE,
                HarnessRole.REVERT,
                HarnessRole.PLACEBO,
            ):
                selected = tuple(
                    item for item in context.gate_observations if item.harness_role is role
                )
                if len(selected) != block.gate_task_count:
                    raise ValueError(f"{context.cell.value}/{role.value} has incomplete coverage")
                if len({item.task_id for item in selected}) != block.gate_task_count:
                    raise ValueError(f"{context.cell.value}/{role.value} repeats a task")
            fit_task_ids = {item.task_id for item in context.fit_parent_observations}
            gate_task_ids = {item.task_id for item in context.gate_observations}
            if fit_task_ids.intersection(gate_task_ids):
                raise ValueError("fit feedback and gate attribution reuse a task")
            all_observations = context.fit_parent_observations + context.gate_observations
            outcome_hashes = {context.proposal.outcome_ref.sha256} | {
                item.outcome_ref.sha256 for item in all_observations
            }
            if len(outcome_hashes) != block.model_call_count:
                raise ValueError("context model calls require unique outcome refs")
            if all_outcomes.intersection(outcome_hashes):
                raise ValueError("attempt outcomes cross two context ledgers")
            all_outcomes.update(outcome_hashes)
        if len(all_outcomes) != self.model_call_count:
            raise ValueError("declared calls do not close the global exact topology")
        return self


__all__ = [
    "FAULT_CONTEXT_ORDER",
    "HARNESS_FAULT_FOUR_CONTEXT_FREEZE_MEDIA_TYPE",
    "HARNESS_FAULT_FOUR_CONTEXT_GATE_BATCH_MEDIA_TYPE",
    "HARNESS_FAULT_FOUR_CONTEXT_RESULT_MEDIA_TYPE",
    "FaultContextGateRecord",
    "FrozenFaultContextCandidate",
    "HarnessFaultContextDevelopmentClosure",
    "HarnessFaultFourContextBlock",
    "HarnessFaultFourContextDevelopmentResult",
    "HarnessFaultFourContextGateBatch",
    "HarnessFaultFourContextPlan",
    "JointFaultContextCandidateFreeze",
    "fault_context_candidate_freeze_id",
    "make_harness_fault_four_context_plan",
]
