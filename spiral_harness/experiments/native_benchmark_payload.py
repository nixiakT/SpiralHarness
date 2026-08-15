"""Prospective contracts for benchmark-required native capabilities.

Some benchmarks place tools or schemas inside the task itself.  Treating those
official inputs as a mutable harness component would change the benchmark;
removing them from PURE would instead cripple the baseline.  This module keeps
that immutable native payload separate from every harness-added capability.

The contracts are non-attested design values.  They do not prove that an
executor loaded the artifacts, honored a budget, or ran a reportable study.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, NonEmptyStr, Sha256
from spiral_harness.experiments.confirmatory_resources import ProspectiveConfirmatoryModel

NATIVE_BENCHMARK_PAYLOAD_MEDIA_TYPE = (
    "application/vnd.spiral-harness.native-benchmark-payload.v1+json"
)
NATIVE_BENCHMARK_FIVE_ARM_MEDIA_TYPE = (
    "application/vnd.spiral-harness.native-benchmark-five-arm.v1+json"
)

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class NativeBenchmarkArm(StrEnum):
    """Five same-model effectiveness coordinates used by a native benchmark."""

    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"
    PURE_AT_B = "pure-at-b"


_ARM_ORDER = (
    NativeBenchmarkArm.PURE,
    NativeBenchmarkArm.STATIC,
    NativeBenchmarkArm.SCORE,
    NativeBenchmarkArm.FULL,
    NativeBenchmarkArm.PURE_AT_B,
)


class BenchmarkRequiredCapabilityKind(StrEnum):
    """Kinds supplied by the frozen task, never invented by a candidate."""

    TOOL_SCHEMA_BUNDLE = "tool-schema-bundle"
    TOOL_EXECUTION_RUNTIME = "tool-execution-runtime"
    ATTACHMENT_BUNDLE = "attachment-bundle"
    TASK_ENVIRONMENT = "task-environment"


class BenchmarkRequiredCapability(ProspectiveConfirmatoryModel):
    """One content-addressed capability that is part of the official task."""

    schema_version: Literal["1"] = "1"
    capability_id: NonEmptyStr
    kind: BenchmarkRequiredCapabilityKind
    definition_ref: ArtifactRef
    runtime_implementation_ref: ArtifactRef | None = None
    source: Literal["official-benchmark-required"] = "official-benchmark-required"
    mutable_by_harness: Literal[False] = False
    candidate_may_replace: Literal[False] = False
    candidate_may_remove: Literal[False] = False
    charged_when_used: Literal[True] = True

    @field_validator("capability_id", mode="before")
    @classmethod
    def _identifier_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("native capability IDs must be exact and non-empty")
        return value

    @model_validator(mode="after")
    def _runtime_is_bound_when_execution_is_required(self) -> Self:
        requires_runtime = self.kind in {
            BenchmarkRequiredCapabilityKind.TOOL_EXECUTION_RUNTIME,
            BenchmarkRequiredCapabilityKind.TASK_ENVIRONMENT,
        }
        if requires_runtime != (self.runtime_implementation_ref is not None):
            raise ValueError("native executable capabilities require exactly one runtime ref")
        return self


class FrozenNativeBenchmarkPayload(ProspectiveConfirmatoryModel):
    """Task bytes, official capabilities, and provider serialization as one value."""

    schema_version: Literal["1"] = "1"
    benchmark_id: NonEmptyStr
    task_payload_ref: ArtifactRef
    provider_request_schema_ref: ArtifactRef
    provider_serialization_implementation_ref: ArtifactRef
    provider_serialization_config_ref: ArtifactRef
    required_capabilities: Annotated[
        tuple[BenchmarkRequiredCapability, ...],
        Field(min_length=1),
    ]
    official_task_payload: Literal[True] = True
    candidate_mutation_surface: Literal[False] = False

    @field_validator("benchmark_id", mode="before")
    @classmethod
    def _benchmark_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("native benchmark IDs must be exact and non-empty")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def _canonicalize_capabilities(
        cls,
        values: tuple[BenchmarkRequiredCapability, ...],
    ) -> tuple[BenchmarkRequiredCapability, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.capability_id))
        ids = tuple(item.capability_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("native capability IDs must not repeat")
        definitions = tuple(
            (item.definition_ref.sha256, item.definition_ref.size, item.definition_ref.media_type)
            for item in ordered
        )
        if len(definitions) != len(set(definitions)):
            raise ValueError("native capability definition refs must not repeat")
        return ordered

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=NATIVE_BENCHMARK_PAYLOAD_MEDIA_TYPE,
        )


class ProviderMinimalNativeTaskPlan(ProspectiveConfirmatoryModel):
    """PURE semantics when the benchmark itself requires tools or schemas."""

    schema_version: Literal["1"] = "1"
    native_payload: FrozenNativeBenchmarkPayload
    native_payload_fingerprint: Sha256
    native_payload_ref: ArtifactRef
    harness_system_prompt_ref: None = None
    harness_tool_bundle_ref: None = None
    harness_routing_ref: None = None
    harness_middleware_ref: None = None
    harness_memory_ref: None = None
    harness_retrieval_ref: None = None
    harness_few_shot_ref: None = None
    official_task_capabilities_preserved: Literal[True] = True
    benchmark_required_tools_are_task_payload: Literal[True] = True
    harness_added_capabilities_permitted: Literal[False] = False
    mutable_state_across_samples_permitted: Literal[False] = False
    execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_native_payload(self) -> Self:
        if self.native_payload_fingerprint != self.native_payload.fingerprint:
            raise ValueError("provider-minimal native payload fingerprint drifted")
        if self.native_payload_ref != self.native_payload.artifact_ref:
            raise ValueError("provider-minimal native payload ref drifted")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class NativeEvaluationResourceCeilings(ProspectiveConfirmatoryModel):
    """Per-sample official-run ceilings shared by all five native arms."""

    schema_version: Literal["1"] = "1"
    max_model_steps_per_turn: PositiveInt
    max_provider_attempts_per_model_step: PositiveInt
    token_ceiling_per_model_step: PositiveInt
    max_tool_executions_per_evaluation_unit: NonNegativeInt
    max_search_queries_per_evaluation_unit: NonNegativeInt
    max_http_fetches_per_evaluation_unit: NonNegativeInt
    max_downloaded_bytes_per_evaluation_unit: NonNegativeInt
    max_wall_time_seconds_per_evaluation_unit: Annotated[
        float,
        Field(gt=0, strict=True, allow_inf_nan=False),
    ]
    max_external_cost_usd_per_evaluation_unit: Annotated[
        float,
        Field(ge=0, strict=True, allow_inf_nan=False),
    ]
    retry_policy_ref: ArtifactRef
    price_table_ref: ArtifactRef
    sandbox_policy_ref: ArtifactRef
    failed_provider_attempts_charged: Literal[True] = True
    failed_tool_executions_charged: Literal[True] = True
    failed_search_and_http_calls_charged: Literal[True] = True

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class NativeTotalResourceCeilings(ProspectiveConfirmatoryModel):
    """Complete native-run capacity used to match FULL and PURE@B totals."""

    schema_version: Literal["1"] = "1"
    max_model_steps: PositiveInt
    max_provider_attempts: PositiveInt
    max_provider_attempt_tokens: PositiveInt
    max_tool_executions: NonNegativeInt
    max_search_queries: NonNegativeInt
    max_http_fetches: NonNegativeInt
    max_downloaded_bytes: NonNegativeInt
    max_wall_time_seconds: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
    max_external_cost_usd: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
    failed_operations_consume_capacity: Literal[True] = True

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class NativePureAtBTotalBudgetMatch(ProspectiveConfirmatoryModel):
    """Exact FULL-total match, including non-model native resources."""

    schema_version: Literal["1"] = "1"
    full_total_ceilings: NativeTotalResourceCeilings
    pure_at_b_total_ceilings: NativeTotalResourceCeilings
    model_budget_plan_ref: ArtifactRef
    external_resource_allocation_ref: ArtifactRef
    every_dimension_exact: Literal[True] = True
    execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _totals_are_exact(self) -> Self:
        if self.pure_at_b_total_ceilings != self.full_total_ceilings:
            raise ValueError("PURE@B native total resources differ from FULL")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class NativeBenchmarkArmBinding(ProspectiveConfirmatoryModel):
    """Native payload and official evaluation budget for one treatment arm."""

    schema_version: Literal["1"] = "1"
    arm: NativeBenchmarkArm
    native_payload_fingerprint: Sha256
    evaluation_resource_ceilings: NativeEvaluationResourceCeilings
    provider_minimal_plan: ProviderMinimalNativeTaskPlan | None = None

    @model_validator(mode="after")
    def _provider_minimal_plan_scope(self) -> Self:
        requires_minimal = self.arm in {
            NativeBenchmarkArm.PURE,
            NativeBenchmarkArm.PURE_AT_B,
        }
        if requires_minimal != (self.provider_minimal_plan is not None):
            raise ValueError("only PURE and PURE@B require provider-minimal native plans")
        if (
            self.provider_minimal_plan is not None
            and self.provider_minimal_plan.native_payload_fingerprint
            != self.native_payload_fingerprint
        ):
            raise ValueError("arm and provider-minimal native payloads differ")
        return self


class NativeBenchmarkFiveArmContract(ProspectiveConfirmatoryModel):
    """Join native payload semantics to the existing four-arm and PURE@B plans."""

    schema_version: Literal["1"] = "1"
    benchmark_id: NonEmptyStr
    public_snapshot_fingerprint: Sha256
    public_roster_fingerprint: Sha256
    confirmatory_four_arm_design_ref: ArtifactRef
    confirmatory_pure_at_b_plan_ref: ArtifactRef
    score_full_adaptive_budget_ref: ArtifactRef
    pure_at_b_total_budget_match: NativePureAtBTotalBudgetMatch
    arms: Annotated[tuple[NativeBenchmarkArmBinding, ...], Field(min_length=5, max_length=5)]
    official_all_scoring_roster_required: Literal[True] = True
    partial_evaluation_permitted: Literal[False] = False
    task_payload_and_official_capabilities_equal_across_arms: Literal[True] = True
    evaluation_resource_ceilings_equal_across_arms: Literal[True] = True
    score_full_adaptive_budget_equality_required: Literal[True] = True
    pure_static_search_usage_is_measured_not_padded: Literal[True] = True
    pure_at_b_matches_full_total_model_and_external_resource_ceilings: Literal[True] = True
    public_answers_are_hidden_evidence: Literal[False] = False
    execution_attested: Literal[False] = False
    dependency_environment_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    reportable_result: Literal[False] = False

    @field_validator("benchmark_id", mode="before")
    @classmethod
    def _benchmark_id_is_exact(cls, value: object) -> object:
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("native benchmark IDs must be exact and non-empty")
        return value

    @field_validator("arms")
    @classmethod
    def _canonicalize_arms(
        cls,
        values: tuple[NativeBenchmarkArmBinding, ...],
    ) -> tuple[NativeBenchmarkArmBinding, ...]:
        by_arm = {item.arm: item for item in values}
        if len(by_arm) != len(values) or frozenset(by_arm) != frozenset(_ARM_ORDER):
            raise ValueError("native benchmark contract requires exactly five arms")
        return tuple(by_arm[arm] for arm in _ARM_ORDER)

    @model_validator(mode="after")
    def _match_native_payload_and_official_budget(self) -> Self:
        payloads = {item.native_payload_fingerprint for item in self.arms}
        budgets = {item.evaluation_resource_ceilings.fingerprint for item in self.arms}
        if len(payloads) != 1:
            raise ValueError("five arms do not share one frozen native task payload")
        if len(budgets) != 1:
            raise ValueError("five arms do not share official evaluation resource ceilings")
        pure = self.arms[0].provider_minimal_plan
        pure_at_b = self.arms[-1].provider_minimal_plan
        if pure is None or pure_at_b is None:  # pragma: no cover - arm invariant
            raise ValueError("provider-minimal plans are missing")
        if pure != pure_at_b:
            raise ValueError("PURE and PURE@B provider-minimal semantics differ")
        if pure.native_payload.benchmark_id != self.benchmark_id:
            raise ValueError("contract and frozen native payload benchmark IDs differ")
        if (
            self.confirmatory_pure_at_b_plan_ref
            != self.pure_at_b_total_budget_match.model_budget_plan_ref
        ):
            raise ValueError("PURE@B contract and total-budget match bind different plans")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_provider_minimal_native_task_plan(
    payload: FrozenNativeBenchmarkPayload,
) -> ProviderMinimalNativeTaskPlan:
    """Build the only accepted PURE/PURE@B native-payload plan."""

    checked = FrozenNativeBenchmarkPayload.model_validate(payload, strict=True)
    return ProviderMinimalNativeTaskPlan(
        native_payload=checked,
        native_payload_fingerprint=checked.fingerprint,
        native_payload_ref=checked.artifact_ref,
    )


__all__ = [
    "NATIVE_BENCHMARK_FIVE_ARM_MEDIA_TYPE",
    "NATIVE_BENCHMARK_PAYLOAD_MEDIA_TYPE",
    "BenchmarkRequiredCapability",
    "BenchmarkRequiredCapabilityKind",
    "FrozenNativeBenchmarkPayload",
    "NativeBenchmarkArm",
    "NativeBenchmarkArmBinding",
    "NativeBenchmarkFiveArmContract",
    "NativeEvaluationResourceCeilings",
    "NativePureAtBTotalBudgetMatch",
    "NativeTotalResourceCeilings",
    "ProviderMinimalNativeTaskPlan",
    "make_provider_minimal_native_task_plan",
]
