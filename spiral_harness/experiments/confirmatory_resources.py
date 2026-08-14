"""Non-attested topology and resource contracts for confirmatory search.

These models freeze ex-ante call slots and condition isolation.  Equality of
two instances is a design property only; it does not prove that a runtime used
the declared topology, budget, model, or provider route.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, Sha256

NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class ProspectiveConfirmatoryModel(ImmutableModel):
    """A non-attested design value that revalidates at every publication boundary."""

    artifact_status: Literal["prospective-non-attested-design"] = "prospective-non-attested-design"
    runtime_proof_available: Literal[False] = False

    @classmethod
    def _raw_python_value(cls, value: Any) -> Any:
        if isinstance(value, ProspectiveConfirmatoryModel):
            return {
                name: cls._raw_python_value(getattr(value, name))
                for name in type(value).model_fields
            }
        if isinstance(value, Mapping):
            return {key: cls._raw_python_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._raw_python_value(item) for item in value)
        if isinstance(value, list):
            return [cls._raw_python_value(item) for item in value]
        return value

    def _strict_python_content(self) -> dict[str, Any]:
        return {
            name: self._raw_python_value(getattr(self, name)) for name in type(self).model_fields
        }

    def _strict_revalidate(self) -> None:
        type(self).model_validate(self._strict_python_content(), strict=True)

    @model_serializer(mode="wrap")
    def _strict_serializer(self, handler: SerializerFunctionWrapHandler) -> Any:
        """Guard instance, TypeAdapter, and direct core-serializer publication."""

        self._strict_revalidate()
        return handler(self)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Reject unchecked construction or mutation before returning a payload."""

        self._strict_revalidate()
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Reject unchecked construction or mutation before JSON publication."""

        self._strict_revalidate()
        return super().model_dump_json(**kwargs)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a newly validated copy; ``deep`` is implicit in reconstruction."""

        content = self._strict_python_content()
        if update is not None:
            content.update(update)
        return type(self).model_validate(content, strict=True)


class ModelMediatedRole(StrEnum):
    """Potential model-call roles whose ceilings must be declared explicitly."""

    SOLVER = "solver"
    DIAGNOSER = "diagnoser"
    PROPOSER = "proposer"
    MATERIALIZER = "materializer"
    RANKER = "ranker"
    NOMINATOR = "nominator"


_MODEL_ROLE_ORDER = tuple(ModelMediatedRole)
_ADAPTIVE_STAGES = (
    "observe",
    "diagnose",
    "propose",
    "materialize",
    "screen",
    "nominate",
    "execute-attribution-quartet",
    "produce-evidence",
    "gate",
)
_ATTRIBUTION_SIDES = ("parent", "candidate", "revert", "placebo")


class ModelRoleCeiling(ProspectiveConfirmatoryModel):
    """Ex-ante call and token capacity for one potentially model-mediated role."""

    schema_version: Literal["1"] = "1"
    role: ModelMediatedRole
    max_calls: NonNegativeInt
    max_tokens: NonNegativeInt

    @model_validator(mode="after")
    def _calls_and_tokens_are_jointly_enabled(self) -> Self:
        if (self.max_calls == 0) != (self.max_tokens == 0):
            raise ValueError("a role must enable or disable calls and tokens together")
        return self


class AdaptiveExecutionCeilings(ProspectiveConfirmatoryModel):
    """Complete ex-ante resource and stopping ceilings for adaptive conditions."""

    schema_version: Literal["1"] = "1"
    max_rounds: PositiveInt
    max_proposals_per_round: PositiveInt
    max_total_proposals: PositiveInt
    max_nominations_per_round: Literal[1] = 1
    max_total_nominations: PositiveInt
    max_gate_queries: PositiveInt
    max_feedback_releases: PositiveInt
    max_evaluations: PositiveInt
    max_attempts_per_model_call: PositiveInt
    token_ceiling_per_model_call: PositiveInt
    role_model_calls: Annotated[tuple[ModelRoleCeiling, ...], Field(min_length=6, max_length=6)]
    max_total_model_calls: PositiveInt
    max_total_tokens: PositiveInt
    max_total_model_attempts: PositiveInt
    max_total_attempt_tokens: PositiveInt
    retry_attempts_count_as_provider_calls: Literal[True] = True
    failed_attempts_consume_attempt_and_token_ceilings: Literal[True] = True
    max_wall_time_seconds: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
    max_cost_usd: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]

    @field_validator("role_model_calls")
    @classmethod
    def _canonicalize_complete_role_roster(
        cls,
        values: tuple[ModelRoleCeiling, ...],
    ) -> tuple[ModelRoleCeiling, ...]:
        by_role = {value.role: value for value in values}
        if len(by_role) != len(values):
            raise ValueError("role_model_calls must not contain duplicate roles")
        if frozenset(by_role) != frozenset(_MODEL_ROLE_ORDER):
            raise ValueError("role_model_calls must explicitly cover every model-mediated role")
        return tuple(by_role[role] for role in _MODEL_ROLE_ORDER)

    @model_validator(mode="after")
    def _ceilings_close_arithmetically(self) -> Self:
        if self.max_total_proposals > self.max_rounds * self.max_proposals_per_round:
            raise ValueError("max_total_proposals exceeds the round schedule")
        if self.max_total_nominations > self.max_rounds * self.max_nominations_per_round:
            raise ValueError("max_total_nominations exceeds the round schedule")
        if self.max_total_nominations > self.max_total_proposals:
            raise ValueError("nominations cannot exceed frozen proposals")
        if self.max_gate_queries != self.max_total_nominations:
            raise ValueError("every nomination requires exactly one fresh gate query")
        if self.max_feedback_releases != self.max_gate_queries:
            raise ValueError("every completed gate query has exactly one feedback release slot")
        total_calls = sum(item.max_calls for item in self.role_model_calls)
        total_tokens = sum(item.max_tokens for item in self.role_model_calls)
        if self.max_total_model_calls != total_calls:
            raise ValueError("max_total_model_calls differs from the role ceilings")
        if self.max_total_tokens != total_tokens:
            raise ValueError("max_total_tokens differs from the role ceilings")
        expected_attempts = total_calls * self.max_attempts_per_model_call
        if self.max_total_model_attempts != expected_attempts:
            raise ValueError("max_total_model_attempts must include every permitted retry attempt")
        expected_attempt_tokens = total_tokens * self.max_attempts_per_model_call
        if self.max_total_attempt_tokens != expected_attempt_tokens:
            raise ValueError("max_total_attempt_tokens must include every permitted retry attempt")
        for item in self.role_model_calls:
            if item.max_tokens > item.max_calls * self.token_ceiling_per_model_call:
                raise ValueError(f"{item.role.value} token ceiling exceeds its call capacity")
        solver = next(
            item for item in self.role_model_calls if item.role is ModelMediatedRole.SOLVER
        )
        proposer = next(
            item for item in self.role_model_calls if item.role is ModelMediatedRole.PROPOSER
        )
        nominator = next(
            item for item in self.role_model_calls if item.role is ModelMediatedRole.NOMINATOR
        )
        if solver.max_calls == 0 or proposer.max_calls == 0:
            raise ValueError("adaptive search requires positive solver and proposer call ceilings")
        if self.max_evaluations > solver.max_calls:
            raise ValueError("max_evaluations exceeds the solver call ceiling")
        if self.max_total_proposals > proposer.max_calls:
            raise ValueError("each frozen proposal requires one proposer call slot")
        if self.max_total_nominations > nominator.max_calls:
            raise ValueError("each nomination requires one nominator call slot")
        if self.max_evaluations < self.max_total_nominations * len(_ATTRIBUTION_SIDES):
            raise ValueError("solver evaluations cannot cover every attribution quartet")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


AdaptiveStage = Literal[
    "observe",
    "diagnose",
    "propose",
    "materialize",
    "screen",
    "nominate",
    "execute-attribution-quartet",
    "produce-evidence",
    "gate",
]
AttributionSide = Literal["parent", "candidate", "revert", "placebo"]


class AdaptiveProtocolCommitments(ProspectiveConfirmatoryModel):
    """Content identities held equal across adaptive treatment conditions.

    These digests make the ex-ante equality claim precise.  They remain design
    commitments, not proof that any runtime loaded the committed artifacts.
    """

    schema_version: Literal["1"] = "1"
    model_spec_fingerprint: Sha256
    solver_config_fingerprint: Sha256
    optimizer_config_fingerprint: Sha256
    task_split_fingerprint: Sha256
    seed_schedule_fingerprint: Sha256
    candidate_parser_fingerprint: Sha256
    grader_fingerprint: Sha256
    query_dag_fingerprint: Sha256
    retry_policy_fingerprint: Sha256
    runtime_binding_attested: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ExAnteAdaptiveTopology(ProspectiveConfirmatoryModel):
    """The call topology that controls must not shorten after treatment masking."""

    schema_version: Literal["1"] = "1"
    topology_version: Literal["confirmatory-adaptive-topology-v1"] = (
        "confirmatory-adaptive-topology-v1"
    )
    stages: tuple[AdaptiveStage, ...] = _ADAPTIVE_STAGES
    attribution_sides: tuple[AttributionSide, ...] = _ATTRIBUTION_SIDES
    protocol_commitments: AdaptiveProtocolCommitments
    condition_context_count: Annotated[int, Field(ge=2, strict=True)]
    condition_contexts_isolated: Literal[True] = True
    shared_mutable_state_between_conditions: Literal[False] = False
    full_evidence_computed_for_every_condition: Literal[True] = True
    candidates_frozen_before_cross_condition_gate_batch: Literal[True] = True
    feedback_released_after_complete_batch: Literal[True] = True
    cross_condition_feedback_disclosed: Literal[False] = False
    post_treatment_candidate_identity_required: Literal[False] = False
    realized_token_equality_claimed: Literal[False] = False
    execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _require_complete_topology(self) -> Self:
        if self.stages != _ADAPTIVE_STAGES:
            raise ValueError("adaptive topology must retain every frozen stage in order")
        if self.attribution_sides != _ATTRIBUTION_SIDES:
            raise ValueError("adaptive topology must retain parent/candidate/revert/placebo")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [
    "AdaptiveExecutionCeilings",
    "AdaptiveProtocolCommitments",
    "ExAnteAdaptiveTopology",
    "ModelMediatedRole",
    "ModelRoleCeiling",
    "ProspectiveConfirmatoryModel",
]
