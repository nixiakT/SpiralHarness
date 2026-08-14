"""Task-bound prospective contracts for the budget-matched bare-model reference.

``PURE@B`` redirects FULL's complete ex-ante model-call, provider-attempt, and
token ceilings to independent bare-model samples.  A scalar total is not a
credible match: it would permit an implementation to concentrate the budget on
selected tasks or examples after observing outcomes.  These contracts therefore
freeze every sample inside a task-bound evaluation unit and join the task split,
grader, query DAG, and retry policy back to FULL.

The models remain prospective and non-attested.  They validate a frozen plan;
they do not prove that a provider executed it or that the declared samples were
independent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, NonEmptyStr, Sha256
from spiral_harness.experiments.confirmatory_arms import (
    ConfirmatoryArmProfileError,
    ConfirmatoryFourArmDesign,
    RealTaskArm,
)
from spiral_harness.experiments.confirmatory_resources import (
    ConfirmatoryTaskSplitManifest,
    ProspectiveConfirmatoryModel,
)

PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class AggregationMethod(StrEnum):
    """Frozen, grader-independent PURE@B aggregation families."""

    MAJORITY_BINARY = "majority-binary"
    NORMALIZED_ARTIFACT_PLURALITY = "normalized-artifact-plurality"
    TASK_NATIVE_FIXED = "task-native-fixed"


class PureAtBAggregationRule(ProspectiveConfirmatoryModel):
    """One total, nonadaptive aggregation rule bound to an exact task artifact."""

    schema_version: Literal["2"] = "2"
    task_id: NonEmptyStr
    task_manifest_ref: ArtifactRef
    rule_id: NonEmptyStr
    method: AggregationMethod
    implementation_fingerprint: Sha256
    normalizer_fingerprint: Sha256
    output_domain_fingerprint: Sha256
    tie_breaker: Literal["first-sample-in-frozen-order"] = "first-sample-in-frozen-order"
    failed_sample_policy: Literal["abstain-and-all-abstain-select-first"] = (
        "abstain-and-all-abstain-select-first"
    )
    adaptive: Literal[False] = False
    grader_feedback_used: Literal[False] = False
    sealed_feedback_used: Literal[False] = False
    total_for_every_output_multiset: Literal[True] = True

    @field_validator("task_id", "rule_id", mode="before")
    @classmethod
    def _identifiers_are_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("PURE@B identifiers must be exact and non-empty")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class PureAtBSampleAllocation(ProspectiveConfirmatoryModel):
    """One precommitted independent call and all of its provider-attempt capacity."""

    schema_version: Literal["1"] = "1"
    sample_id: NonEmptyStr
    seed_fingerprint: Sha256
    logical_call_ceiling: Literal[1] = 1
    token_ceiling: PositiveInt
    max_provider_attempts: PositiveInt
    provider_attempt_token_ceiling: PositiveInt

    @field_validator("sample_id", mode="before")
    @classmethod
    def _sample_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("PURE@B sample_id must be exact and non-empty")
        return value

    @model_validator(mode="after")
    def _close_attempt_tokens(self) -> Self:
        expected = self.token_ceiling * self.max_provider_attempts
        if self.provider_attempt_token_ceiling != expected:
            raise ValueError("sample attempt-token ceiling must cover every permitted attempt")
        return self


class PureAtBEvaluationUnitAllocation(ProspectiveConfirmatoryModel):
    """All PURE@B samples assigned to one frozen task/evaluation coordinate."""

    schema_version: Literal["1"] = "1"
    evaluation_unit_id: NonEmptyStr
    evaluation_unit_ref: ArtifactRef
    task_id: NonEmptyStr
    task_manifest_ref: ArtifactRef
    samples: Annotated[tuple[PureAtBSampleAllocation, ...], Field(min_length=1)]
    sample_count_ceiling: PositiveInt
    model_call_ceiling: PositiveInt
    token_ceiling: PositiveInt
    provider_attempt_ceiling: PositiveInt
    provider_attempt_token_ceiling: PositiveInt
    one_logical_call_per_sample: Literal[True] = True

    @field_validator("evaluation_unit_id", "task_id", mode="before")
    @classmethod
    def _identifiers_are_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("PURE@B identifiers must be exact and non-empty")
        return value

    @field_validator("samples")
    @classmethod
    def _canonicalize_samples(
        cls,
        values: tuple[PureAtBSampleAllocation, ...],
    ) -> tuple[PureAtBSampleAllocation, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.sample_id))
        sample_ids = tuple(item.sample_id for item in ordered)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must not repeat inside an evaluation unit")
        return ordered

    @model_validator(mode="after")
    def _close_unit_allocation(self) -> Self:
        if self.sample_count_ceiling != len(self.samples):
            raise ValueError("evaluation-unit sample count differs from its frozen samples")
        if self.model_call_ceiling != sum(item.logical_call_ceiling for item in self.samples):
            raise ValueError("evaluation-unit model-call ceiling differs from its samples")
        if self.sample_count_ceiling != self.model_call_ceiling:
            raise ValueError("every evaluation-unit sample must consume one model-call slot")
        if self.token_ceiling != sum(item.token_ceiling for item in self.samples):
            raise ValueError("evaluation-unit token ceiling differs from its samples")
        if self.provider_attempt_ceiling != sum(
            item.max_provider_attempts for item in self.samples
        ):
            raise ValueError("evaluation-unit provider-attempt ceiling differs from its samples")
        if self.provider_attempt_token_ceiling != sum(
            item.provider_attempt_token_ceiling for item in self.samples
        ):
            raise ValueError("evaluation-unit attempt-token ceiling differs from its samples")
        return self


def pure_at_b_seed_schedule_fingerprint(
    evaluation_units: tuple[PureAtBEvaluationUnitAllocation, ...],
) -> str:
    """Hash the exact canonical sample/seed schedule rather than a caller assertion."""

    ordered_units = tuple(sorted(evaluation_units, key=lambda item: item.evaluation_unit_id))
    return canonical_sha256(
        {
            "schema_version": "1",
            "schedule": tuple(
                {
                    "evaluation_unit_id": unit.evaluation_unit_id,
                    "samples": tuple(
                        {
                            "sample_id": sample.sample_id,
                            "seed_fingerprint": sample.seed_fingerprint,
                        }
                        for sample in unit.samples
                    ),
                }
                for unit in ordered_units
            ),
        }
    )


class PureAtBPlan(ProspectiveConfirmatoryModel):
    """FULL-total-matched, task-bound bare-model sampling reference."""

    schema_version: Literal["3"] = "3"
    four_arm_design: ConfirmatoryFourArmDesign
    four_arm_design_fingerprint: Sha256
    aggregations: Annotated[tuple[PureAtBAggregationRule, ...], Field(min_length=1)]
    evaluation_units: Annotated[
        tuple[PureAtBEvaluationUnitAllocation, ...],
        Field(min_length=1),
    ]
    model_spec_fingerprint: Sha256
    solver_config_fingerprint: Sha256
    task_split_fingerprint: Sha256
    task_split_manifest: ConfirmatoryTaskSplitManifest
    task_split_manifest_ref: ArtifactRef
    evaluation_seed_schedule_fingerprint: Sha256
    grader_fingerprint: Sha256
    query_dag_fingerprint: Sha256
    retry_policy_fingerprint: Sha256
    sample_seed_schedule_fingerprint: Sha256
    pure_sample_count_ceiling: PositiveInt
    pure_model_call_ceiling: PositiveInt
    pure_token_ceiling: PositiveInt
    max_attempts_per_sample: PositiveInt
    pure_model_attempt_ceiling: PositiveInt
    pure_attempt_token_ceiling: PositiveInt
    budget_match_scope: Literal[
        "FULL-total-with-precommitted-task-unit-sample-call-attempt-and-token-allocation"
    ] = "FULL-total-with-precommitted-task-unit-sample-call-attempt-and-token-allocation"
    same_model_solver_task_split_grader_query_dag_and_retry_required: Literal[True] = True
    exact_task_evaluation_roster_bijection_required: Literal[True] = True
    independent_samples_required: Literal[True] = True
    sample_independence_definition: Literal[
        "globally-distinct-precommitted-seeds-with-no-cross-sample-mutable-state"
    ] = "globally-distinct-precommitted-seeds-with-no-cross-sample-mutable-state"
    hidden_grader_best_of_k_permitted: Literal[False] = False
    same_solver_configuration_attested: Literal[False] = False
    sample_independence_attested: Literal[False] = False
    aggregation_execution_attested: Literal[False] = False
    execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @field_validator("aggregations")
    @classmethod
    def _canonicalize_aggregations(
        cls,
        values: tuple[PureAtBAggregationRule, ...],
    ) -> tuple[PureAtBAggregationRule, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.task_id))
        task_ids = tuple(item.task_id for item in ordered)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("PURE@B requires exactly one aggregation rule per task")
        return ordered

    @field_validator("evaluation_units")
    @classmethod
    def _canonicalize_evaluation_units(
        cls,
        values: tuple[PureAtBEvaluationUnitAllocation, ...],
    ) -> tuple[PureAtBEvaluationUnitAllocation, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.evaluation_unit_id))
        unit_ids = tuple(item.evaluation_unit_id for item in ordered)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("PURE@B evaluation-unit IDs must not repeat")
        unit_refs = tuple(
            (
                item.evaluation_unit_ref.sha256,
                item.evaluation_unit_ref.size,
                item.evaluation_unit_ref.media_type,
            )
            for item in ordered
        )
        if len(unit_refs) != len(set(unit_refs)):
            raise ValueError("PURE@B evaluation-unit artifact refs must not repeat")
        return ordered

    @model_validator(mode="after")
    def _bind_full_and_close_every_allocation(self) -> Self:
        if self.four_arm_design_fingerprint != self.four_arm_design.fingerprint:
            raise ValueError("PURE@B design fingerprint differs from its four-arm design")
        full = self.four_arm_design.arm(RealTaskArm.FULL)
        ceilings = full.adaptive_ceilings
        topology = full.adaptive_topology
        if ceilings is None or topology is None:  # pragma: no cover - four-arm invariant
            raise ValueError("FULL adaptive topology or ceilings are missing")
        commitments = full.evaluation_commitments
        commitment_fields = {
            "model_spec_fingerprint": commitments.model_spec_fingerprint,
            "solver_config_fingerprint": commitments.solver_config_fingerprint,
            "task_split_fingerprint": commitments.task_split_fingerprint,
            "evaluation_seed_schedule_fingerprint": commitments.seed_schedule_fingerprint,
            "grader_fingerprint": commitments.grader_fingerprint,
            "query_dag_fingerprint": commitments.query_dag_fingerprint,
            "retry_policy_fingerprint": commitments.retry_policy_fingerprint,
        }
        for field_name, expected in commitment_fields.items():
            if getattr(self, field_name) != expected:
                readable = field_name.replace("_", " ").removesuffix(" fingerprint")
                raise ValueError(f"PURE@B {readable} fingerprint differs from FULL")

        if self.task_split_manifest != commitments.task_split_manifest:
            raise ValueError("PURE@B task-split manifest differs from the four-arm roster")
        if self.task_split_manifest_ref != commitments.task_split_manifest_ref:
            raise ValueError("PURE@B task-split artifact ref differs from the four-arm roster")
        for arm in RealTaskArm:
            arm_manifest = self.four_arm_design.arm(arm).evaluation_commitments.task_split_manifest
            if arm_manifest != self.task_split_manifest:  # pragma: no cover - design invariant
                raise ValueError("PURE@B roster is not shared by every real-task arm")

        aggregations = {item.task_id: item for item in self.aggregations}
        expected_tasks = {
            item.task_id: item.task_manifest_ref for item in self.task_split_manifest.tasks
        }
        if set(aggregations) != set(expected_tasks):
            raise ValueError("PURE@B aggregation task roster differs from the canonical split")
        if any(
            aggregations[task_id].task_manifest_ref != task_ref
            for task_id, task_ref in expected_tasks.items()
        ):
            raise ValueError("PURE@B aggregation task artifact differs from the canonical split")

        expected_units = {
            (
                task.task_id,
                task.task_manifest_ref.sha256,
                task.task_manifest_ref.size,
                task.task_manifest_ref.media_type,
                unit.evaluation_unit_id,
                unit.evaluation_unit_ref.sha256,
                unit.evaluation_unit_ref.size,
                unit.evaluation_unit_ref.media_type,
            )
            for task in self.task_split_manifest.tasks
            for unit in task.evaluation_units
        }
        allocated_units = {
            (
                unit.task_id,
                unit.task_manifest_ref.sha256,
                unit.task_manifest_ref.size,
                unit.task_manifest_ref.media_type,
                unit.evaluation_unit_id,
                unit.evaluation_unit_ref.sha256,
                unit.evaluation_unit_ref.size,
                unit.evaluation_unit_ref.media_type,
            )
            for unit in self.evaluation_units
        }
        if allocated_units != expected_units or len(allocated_units) != len(self.evaluation_units):
            raise ValueError(
                "PURE@B task/evaluation-unit roster is not bijective with the canonical split"
            )

        samples = tuple(sample for unit in self.evaluation_units for sample in unit.samples)
        sample_ids = tuple(item.sample_id for item in samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("PURE@B sample IDs must be globally unique")
        sample_seeds = tuple(item.seed_fingerprint for item in samples)
        if len(sample_seeds) != len(set(sample_seeds)):
            raise ValueError("PURE@B sample seeds must be globally distinct")
        if self.sample_seed_schedule_fingerprint != pure_at_b_seed_schedule_fingerprint(
            self.evaluation_units
        ):
            raise ValueError("PURE@B seed schedule fingerprint differs from exact sample seeds")
        if any(sample.token_ceiling > ceilings.token_ceiling_per_model_call for sample in samples):
            raise ValueError("PURE@B per-sample token ceiling exceeds FULL per-call ceiling")
        if any(
            sample.max_provider_attempts != ceilings.max_attempts_per_model_call
            for sample in samples
        ):
            raise ValueError("PURE@B per-sample attempt ceiling differs from FULL")

        allocation_totals = {
            "pure_sample_count_ceiling": sum(
                item.sample_count_ceiling for item in self.evaluation_units
            ),
            "pure_model_call_ceiling": sum(
                item.model_call_ceiling for item in self.evaluation_units
            ),
            "pure_token_ceiling": sum(item.token_ceiling for item in self.evaluation_units),
            "pure_model_attempt_ceiling": sum(
                item.provider_attempt_ceiling for item in self.evaluation_units
            ),
            "pure_attempt_token_ceiling": sum(
                item.provider_attempt_token_ceiling for item in self.evaluation_units
            ),
        }
        for field_name, expected in allocation_totals.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"PURE@B {field_name} differs from evaluation-unit allocations")
        if self.pure_sample_count_ceiling != self.pure_model_call_ceiling:
            raise ValueError("each PURE@B sample must consume one model-call slot")
        if self.max_attempts_per_sample != ceilings.max_attempts_per_model_call:
            raise ValueError("PURE@B per-sample attempt ceiling differs from FULL")

        full_totals = {
            "pure_model_call_ceiling": ceilings.max_total_model_calls,
            "pure_token_ceiling": ceilings.max_total_tokens,
            "pure_model_attempt_ceiling": ceilings.max_total_model_attempts,
            "pure_attempt_token_ceiling": ceilings.max_total_attempt_tokens,
        }
        for field_name, expected in full_totals.items():
            if getattr(self, field_name) != expected:
                readable = field_name.removeprefix("pure_").replace("_", " ")
                raise ValueError(f"PURE@B {readable} differs from FULL")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_pure_at_b_plan(
    *,
    four_arm_design: ConfirmatoryFourArmDesign,
    aggregations: tuple[PureAtBAggregationRule, ...],
    evaluation_units: tuple[PureAtBEvaluationUnitAllocation, ...],
) -> PureAtBPlan:
    """Derive global ceilings and identities from FULL and exact unit allocations."""

    design = ConfirmatoryFourArmDesign.model_validate(four_arm_design, strict=True)
    rules = tuple(PureAtBAggregationRule.model_validate(item, strict=True) for item in aggregations)
    units = tuple(
        PureAtBEvaluationUnitAllocation.model_validate(item, strict=True)
        for item in evaluation_units
    )
    full = design.arm(RealTaskArm.FULL)
    ceilings = full.adaptive_ceilings
    topology = full.adaptive_topology
    if ceilings is None or topology is None:  # pragma: no cover - four-arm invariant
        raise ConfirmatoryArmProfileError("FULL adaptive topology or ceilings are missing")
    commitments = full.evaluation_commitments
    return PureAtBPlan(
        four_arm_design=design,
        four_arm_design_fingerprint=design.fingerprint,
        aggregations=rules,
        evaluation_units=units,
        model_spec_fingerprint=commitments.model_spec_fingerprint,
        solver_config_fingerprint=commitments.solver_config_fingerprint,
        task_split_fingerprint=commitments.task_split_fingerprint,
        task_split_manifest=commitments.task_split_manifest,
        task_split_manifest_ref=commitments.task_split_manifest_ref,
        evaluation_seed_schedule_fingerprint=commitments.seed_schedule_fingerprint,
        grader_fingerprint=commitments.grader_fingerprint,
        query_dag_fingerprint=commitments.query_dag_fingerprint,
        retry_policy_fingerprint=commitments.retry_policy_fingerprint,
        sample_seed_schedule_fingerprint=pure_at_b_seed_schedule_fingerprint(units),
        pure_sample_count_ceiling=sum(item.sample_count_ceiling for item in units),
        pure_model_call_ceiling=sum(item.model_call_ceiling for item in units),
        pure_token_ceiling=sum(item.token_ceiling for item in units),
        max_attempts_per_sample=ceilings.max_attempts_per_model_call,
        pure_model_attempt_ceiling=sum(item.provider_attempt_ceiling for item in units),
        pure_attempt_token_ceiling=sum(item.provider_attempt_token_ceiling for item in units),
    )


__all__ = [
    "AggregationMethod",
    "PureAtBAggregationRule",
    "PureAtBEvaluationUnitAllocation",
    "PureAtBPlan",
    "PureAtBSampleAllocation",
    "make_pure_at_b_plan",
    "pure_at_b_seed_schedule_fingerprint",
]
