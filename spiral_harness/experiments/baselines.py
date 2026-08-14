"""Frozen planning and validation for the legacy four-arm baseline protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import (
    ArtifactRef,
    ComponentKind,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)

BASELINE_STUDY_PLAN_MEDIA_TYPE = "application/vnd.spiral-harness.baseline-study-plan.v1+json"


class BaselineProtocolError(ValueError):
    """Raised when a baseline run cannot be compared under the frozen protocol."""


class BaselineKind(StrEnum):
    """Known study conditions across protocol versions."""

    STATIC = "static"
    RANDOM_VALID = "random-valid"
    PROMPT_ONLY = "prompt-only"
    SCORE_ONLY_MATCHED = "score-only-matched"
    EVIDENCE_TARGETED = "evidence-targeted"


LEGACY_BASELINE_KINDS = (
    BaselineKind.STATIC,
    BaselineKind.RANDOM_VALID,
    BaselineKind.PROMPT_ONLY,
    BaselineKind.EVIDENCE_TARGETED,
)
REQUIRED_BASELINES = frozenset(LEGACY_BASELINE_KINDS)


class FeedbackType(StrEnum):
    """Information a condition may consume while proposing candidates."""

    BENCHMARK_METADATA = "benchmark-metadata"
    EXPLORATION_INPUTS = "exploration-inputs"
    EXPLORATION_AGGREGATES = "exploration-aggregates"
    EXPLORATION_ITEM_FEEDBACK = "exploration-item-feedback"
    EXPLORATION_TRAJECTORIES = "exploration-trajectories"
    DIAGNOSTIC_EVIDENCE = "diagnostic-evidence"
    MECHANISM_EVIDENCE = "mechanism-evidence"
    GATE_AGGREGATES = "gate-aggregates"
    GATE_ITEM_CONTENT = "gate-item-content"
    SEALED_ITEM_CONTENT = "sealed-item-content"


FORBIDDEN_ITEM_FEEDBACK = frozenset(
    {FeedbackType.GATE_ITEM_CONTENT, FeedbackType.SEALED_ITEM_CONTENT}
)


class MutationMode(StrEnum):
    """How a condition is permitted to choose a mutation."""

    NONE = "none"
    UNIFORM_RANDOM_VALID = "uniform-random-valid"
    PROMPT_OPTIMIZATION = "prompt-optimization"
    EVIDENCE_TARGETED = "evidence-targeted"


class MutationOperation(StrEnum):
    """Atomic operations supported by the frozen mutation grammar."""

    REPLACE = "replace"


class FrozenMutationPolicy(ImmutableModel):
    """The exact mutation grammar shared by every search condition."""

    schema_version: Literal["1"] = "1"
    grammar_version: NonEmptyStr
    allowed_component_kinds: Annotated[
        tuple[ComponentKind, ...],
        Field(min_length=1),
    ]
    allowed_operations: Annotated[
        tuple[MutationOperation, ...],
        Field(min_length=1),
    ] = (MutationOperation.REPLACE,)
    max_components_per_proposal: Literal[1] = 1
    max_artifact_size_bytes: Annotated[int, Field(ge=1, strict=True)]

    @field_validator("allowed_component_kinds")
    @classmethod
    def canonicalize_component_kinds(
        cls,
        values: tuple[ComponentKind, ...],
    ) -> tuple[ComponentKind, ...]:
        return tuple(sorted(values, key=lambda value: value.value))

    @field_validator("allowed_operations")
    @classmethod
    def canonicalize_operations(
        cls,
        values: tuple[MutationOperation, ...],
    ) -> tuple[MutationOperation, ...]:
        return tuple(sorted(values, key=lambda value: value.value))

    @model_validator(mode="after")
    def reject_duplicates_and_require_prompt(self) -> FrozenMutationPolicy:
        if len(self.allowed_component_kinds) != len(set(self.allowed_component_kinds)):
            raise ValueError("allowed_component_kinds must not contain duplicates")
        if len(self.allowed_operations) != len(set(self.allowed_operations)):
            raise ValueError("allowed_operations must not contain duplicates")
        if ComponentKind.PROMPT not in self.allowed_component_kinds:
            raise ValueError("the four-baseline protocol requires prompt in the mutation grammar")
        return self


class FrozenRunContext(ImmutableModel):
    """Score-affecting context that must be byte-for-byte equal across arms."""

    schema_version: Literal["1"] = "1"
    benchmark_ref: ArtifactRef
    model_fingerprint: NonEmptyStr
    inference_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    seed_harness_ref: ArtifactRef
    mutation_policy: FrozenMutationPolicy
    proposal_random_seed: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def manifests_are_json(self) -> FrozenRunContext:
        for field_name in ("benchmark_ref", "seed_harness_ref"):
            ref = getattr(self, field_name)
            media_type = ref.media_type.partition(";")[0].strip().lower()
            if media_type != "application/json" and not media_type.endswith("+json"):
                raise ValueError(f"{field_name} must declare a JSON media type")
        return self


class PairedEvaluationPlan(ImmutableModel):
    """Independent search replications plus paired task-rollout repeats."""

    schema_version: Literal["1"] = "1"
    search_run_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=2),
    ]
    repeat_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=2),
    ]
    pairing_key: Literal["task-id-and-repeat-seed"] = "task-id-and-repeat-seed"
    require_complete_pairs: Literal[True] = True
    gate_feedback: Literal["aggregate-only"] = "aggregate-only"
    sealed_feedback: Literal["final-aggregate-only"] = "final-aggregate-only"
    expose_gate_item_content: Literal[False] = False
    expose_sealed_item_content: Literal[False] = False

    @field_validator("search_run_seeds", "repeat_seeds")
    @classmethod
    def canonicalize_seed_schedule(
        cls,
        values: tuple[int, ...],
        info: object,
    ) -> tuple[int, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            field_name = getattr(info, "field_name", "seed schedule")
            raise ValueError(f"{field_name} must be unique")
        return ordered

    @model_validator(mode="after")
    def require_disjoint_seed_domains(self) -> PairedEvaluationPlan:
        if set(self.search_run_seeds).intersection(self.repeat_seeds):
            raise ValueError("search_run_seeds and repeat_seeds must be disjoint")
        return self


class ResourceCeilings(ImmutableModel):
    """Identical available resources assigned to every condition."""

    max_evaluations: Annotated[int, Field(gt=0, strict=True)]
    max_feedback_queries: Annotated[int, Field(gt=0, strict=True)]
    max_proposals: Annotated[int, Field(gt=0, strict=True)]
    max_optimizer_model_calls: Annotated[int, Field(gt=0, strict=True)]
    max_tokens: Annotated[int, Field(gt=0, strict=True)]
    max_wall_time_seconds: Annotated[
        float,
        Field(gt=0, strict=True, allow_inf_nan=False),
    ]
    max_cost_usd: Annotated[
        float,
        Field(ge=0, strict=True, allow_inf_nan=False),
    ]


class ResourceUsage(ImmutableModel):
    """Caller-declared resource use; these claims are not implied by availability."""

    evaluations: Annotated[int, Field(ge=0, strict=True)] = 0
    feedback_queries: Annotated[int, Field(ge=0, strict=True)] = 0
    proposals: Annotated[int, Field(ge=0, strict=True)] = 0
    optimizer_model_calls: Annotated[int, Field(ge=0, strict=True)] = 0
    tokens: Annotated[int, Field(ge=0, strict=True)] = 0
    wall_time_seconds: Annotated[
        float,
        Field(ge=0, strict=True, allow_inf_nan=False),
    ] = 0.0
    cost_usd: Annotated[
        float,
        Field(ge=0, strict=True, allow_inf_nan=False),
    ] = 0.0


class MutationCapability(ImmutableModel):
    """Condition-specific mutation surface within the shared grammar."""

    mode: MutationMode
    mutable_component_kinds: tuple[ComponentKind, ...] = ()
    may_use_diagnostic_evidence: bool = False
    may_call_optimizer_model: bool = False

    @field_validator("mutable_component_kinds")
    @classmethod
    def canonicalize_mutable_kinds(
        cls,
        values: tuple[ComponentKind, ...],
    ) -> tuple[ComponentKind, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("mutable_component_kinds must not contain duplicates")
        return ordered


_COMMON_SEARCH_FEEDBACK = frozenset(
    {
        FeedbackType.BENCHMARK_METADATA,
        FeedbackType.EXPLORATION_INPUTS,
        FeedbackType.GATE_AGGREGATES,
    }
)


def _expected_feedback(kind: BaselineKind) -> frozenset[FeedbackType]:
    if kind is BaselineKind.STATIC:
        return frozenset({FeedbackType.BENCHMARK_METADATA})
    if kind is BaselineKind.RANDOM_VALID:
        return _COMMON_SEARCH_FEEDBACK
    if kind is BaselineKind.PROMPT_ONLY:
        return _COMMON_SEARCH_FEEDBACK | {
            FeedbackType.EXPLORATION_AGGREGATES,
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
        }
    if kind is BaselineKind.EVIDENCE_TARGETED:
        return _COMMON_SEARCH_FEEDBACK | {
            FeedbackType.EXPLORATION_AGGREGATES,
            FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            FeedbackType.EXPLORATION_TRAJECTORIES,
            FeedbackType.DIAGNOSTIC_EVIDENCE,
        }
    raise ValueError(f"{kind.value} is not supported by the legacy four-arm protocol")


def _expected_mutation_capability(
    kind: BaselineKind,
    policy: FrozenMutationPolicy,
) -> MutationCapability:
    if kind is BaselineKind.STATIC:
        return MutationCapability(mode=MutationMode.NONE)
    if kind is BaselineKind.RANDOM_VALID:
        return MutationCapability(
            mode=MutationMode.UNIFORM_RANDOM_VALID,
            mutable_component_kinds=policy.allowed_component_kinds,
        )
    if kind is BaselineKind.PROMPT_ONLY:
        return MutationCapability(
            mode=MutationMode.PROMPT_OPTIMIZATION,
            mutable_component_kinds=(ComponentKind.PROMPT,),
            may_call_optimizer_model=True,
        )
    if kind is BaselineKind.EVIDENCE_TARGETED:
        return MutationCapability(
            mode=MutationMode.EVIDENCE_TARGETED,
            mutable_component_kinds=policy.allowed_component_kinds,
            may_use_diagnostic_evidence=True,
            may_call_optimizer_model=True,
        )
    raise ValueError(f"{kind.value} is not supported by the legacy four-arm protocol")


class BaselineArmPlan(ImmutableModel):
    """One independently serializable condition within a matched study."""

    schema_version: Literal["1"] = "1"
    kind: BaselineKind
    context: FrozenRunContext
    evaluation: PairedEvaluationPlan
    ceilings: ResourceCeilings
    available_feedback: tuple[FeedbackType, ...]
    mutation: MutationCapability

    @field_validator("available_feedback")
    @classmethod
    def canonicalize_feedback(
        cls,
        values: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("available_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def enforce_condition_profile(self) -> BaselineArmPlan:
        supplied_feedback = frozenset(self.available_feedback)
        forbidden = supplied_feedback.intersection(FORBIDDEN_ITEM_FEEDBACK)
        if forbidden:
            joined = ", ".join(sorted(value.value for value in forbidden))
            raise ValueError(f"gate/sealed item feedback is forbidden: {joined}")
        expected_feedback = _expected_feedback(self.kind)
        if supplied_feedback != expected_feedback:
            raise ValueError(
                f"{self.kind.value} feedback permissions do not match the frozen profile"
            )
        expected_mutation = _expected_mutation_capability(
            self.kind,
            self.context.mutation_policy,
        )
        if self.mutation != expected_mutation:
            raise ValueError(
                f"{self.kind.value} mutation capability does not match the frozen profile"
            )
        return self


class BaselineStudyPlan(ImmutableModel):
    """A complete four-condition plan that rejects structural planning drift."""

    schema_version: Literal["1"] = "1"
    arms: tuple[BaselineArmPlan, ...]

    @field_validator("arms")
    @classmethod
    def canonicalize_arms(
        cls,
        values: tuple[BaselineArmPlan, ...],
    ) -> tuple[BaselineArmPlan, ...]:
        return tuple(sorted(values, key=lambda arm: arm.kind.value))

    @model_validator(mode="after")
    def require_complete_matched_study(self) -> BaselineStudyPlan:
        kinds = tuple(arm.kind for arm in self.arms)
        if len(kinds) != len(set(kinds)):
            raise ValueError("baseline arms must not contain duplicate conditions")
        supplied = frozenset(kinds)
        if supplied != REQUIRED_BASELINES:
            missing = sorted(kind.value for kind in REQUIRED_BASELINES.difference(supplied))
            unexpected = sorted(kind.value for kind in supplied.difference(REQUIRED_BASELINES))
            raise ValueError(
                f"baseline study must contain exactly four conditions; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        anchor = self.arms[0]
        for arm in self.arms[1:]:
            if arm.context != anchor.context:
                raise ValueError("baseline frozen context drifted between conditions")
            if arm.evaluation != anchor.evaluation:
                raise ValueError("baseline paired evaluation plan drifted between conditions")
            if arm.ceilings != anchor.ceilings:
                raise ValueError("baseline resource ceilings drifted between conditions")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the canonical identity bound into every usage report."""

        return canonical_sha256(self)

    def arm(self, kind: BaselineKind) -> BaselineArmPlan:
        """Return the exact plan for one required condition."""

        return next(arm for arm in self.arms if arm.kind is kind)


def plan_four_baselines(
    *,
    context: FrozenRunContext,
    evaluation: PairedEvaluationPlan,
    ceilings: ResourceCeilings,
) -> BaselineStudyPlan:
    """Build the only four condition profiles permitted by this protocol."""

    return BaselineStudyPlan(
        arms=tuple(
            BaselineArmPlan(
                kind=kind,
                context=context,
                evaluation=evaluation,
                ceilings=ceilings,
                available_feedback=tuple(_expected_feedback(kind)),
                mutation=_expected_mutation_capability(kind, context.mutation_policy),
            )
            for kind in LEGACY_BASELINE_KINDS
        )
    )


class BaselineUsageReport(ImmutableModel):
    """Unattested aggregate claim with availability separate from declared use.

    The search runner added in the next stage must derive this claim from
    controller-owned execution and resource events.  This model alone does not
    prove that the declared events occurred.
    """

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    kind: BaselineKind
    available: ResourceCeilings
    used: ResourceUsage
    executed_search_run_seeds: tuple[Annotated[int, Field(ge=0, strict=True)], ...]
    executed_repeat_seeds: tuple[Annotated[int, Field(ge=0, strict=True)], ...]
    feedback_used: tuple[FeedbackType, ...] = ()
    mutated_component_kinds: tuple[ComponentKind, ...] = ()

    @field_validator("executed_search_run_seeds", "executed_repeat_seeds")
    @classmethod
    def canonicalize_executed_seeds(
        cls,
        values: tuple[int, ...],
        info: object,
    ) -> tuple[int, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            field_name = getattr(info, "field_name", "executed seed schedule")
            raise ValueError(f"{field_name} must be unique")
        return ordered

    @field_validator("feedback_used")
    @classmethod
    def canonicalize_feedback_used(
        cls,
        values: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("feedback_used must not contain duplicates")
        return ordered

    @field_validator("mutated_component_kinds")
    @classmethod
    def canonicalize_mutated_kinds(
        cls,
        values: tuple[ComponentKind, ...],
    ) -> tuple[ComponentKind, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("mutated_component_kinds must not contain duplicates")
        return ordered


class BaselineProtocolConsistencyReport(ImmutableModel):
    """Structural success over a plan and its unattested aggregate claims.

    Deliberately do not call this an attested fairness result.  Until the
    automatic runner derives usage from controller-owned evidence, a caller can
    author ``BaselineUsageReport`` values directly.
    """

    schema_version: Literal["1"] = "1"
    reports_consistent: Literal[True] = True
    execution_attested: Literal[False] = False
    evidence_scope: Literal["self-reported-aggregate-claims"] = "self-reported-aggregate-claims"
    plan_fingerprint: Sha256
    baseline_kinds: tuple[BaselineKind, ...]
    checks: tuple[
        Literal[
            "complete-condition-set",
            "frozen-context-equal",
            "reported-search-seeds-match-plan",
            "reported-repeat-seeds-match-plan",
            "reported-available-ceilings-match-plan",
            "reported-usage-within-ceilings",
            "reported-information-within-permissions",
            "reported-mutations-within-permissions",
            "reported-static-search-usage-zero",
        ],
        ...,
    ] = (
        "complete-condition-set",
        "frozen-context-equal",
        "reported-search-seeds-match-plan",
        "reported-repeat-seeds-match-plan",
        "reported-available-ceilings-match-plan",
        "reported-usage-within-ceilings",
        "reported-information-within-permissions",
        "reported-mutations-within-permissions",
        "reported-static-search-usage-zero",
    )

    @field_validator("baseline_kinds")
    @classmethod
    def require_complete_canonical_conditions(
        cls,
        values: tuple[BaselineKind, ...],
    ) -> tuple[BaselineKind, ...]:
        ordered = tuple(sorted(values, key=lambda kind: kind.value))
        if frozenset(ordered) != REQUIRED_BASELINES or len(ordered) != len(REQUIRED_BASELINES):
            raise ValueError("protocol report requires exactly the four baseline conditions")
        return ordered

    @field_validator("checks")
    @classmethod
    def require_complete_canonical_checks(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "complete-condition-set",
            "frozen-context-equal",
            "reported-search-seeds-match-plan",
            "reported-repeat-seeds-match-plan",
            "reported-available-ceilings-match-plan",
            "reported-usage-within-ceilings",
            "reported-information-within-permissions",
            "reported-mutations-within-permissions",
            "reported-static-search-usage-zero",
        )
        if values != expected:
            raise ValueError("protocol report checks must be the complete canonical set")
        return values


class BaselineProtocolValidator:
    """Validate declared protocol structure without attesting execution."""

    @staticmethod
    def validate_plan(plan: BaselineStudyPlan | object) -> BaselineStudyPlan:
        """Revalidate even unchecked Pydantic copies before accepting a plan."""

        try:
            if not isinstance(plan, BaselineStudyPlan):
                raise TypeError("plan must be a BaselineStudyPlan")
            return BaselineStudyPlan.model_validate(
                plan.model_dump(mode="python", round_trip=True, warnings="none"),
                strict=True,
            )
        except (TypeError, ValueError) as exc:
            raise BaselineProtocolError(f"invalid baseline study plan: {exc}") from exc

    @classmethod
    def validate_usage(
        cls,
        plan: BaselineStudyPlan | object,
        reports: tuple[BaselineUsageReport, ...],
    ) -> BaselineProtocolConsistencyReport:
        """Reject inconsistent aggregate claims without treating them as evidence."""

        validated_plan = cls.validate_plan(plan)
        try:
            validated_reports = tuple(
                BaselineUsageReport.model_validate(
                    report.model_dump(mode="python", round_trip=True, warnings="none"),
                    strict=True,
                )
                for report in reports
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise BaselineProtocolError(f"invalid baseline usage report: {exc}") from exc

        kinds = tuple(report.kind for report in validated_reports)
        if len(kinds) != len(set(kinds)):
            raise BaselineProtocolError("baseline usage reports contain duplicate conditions")
        supplied = frozenset(kinds)
        if supplied != REQUIRED_BASELINES:
            missing = sorted(kind.value for kind in REQUIRED_BASELINES.difference(supplied))
            raise BaselineProtocolError(
                f"baseline usage reports are incomplete; missing={missing!r}"
            )

        for report in validated_reports:
            arm = validated_plan.arm(report.kind)
            if report.plan_fingerprint != validated_plan.fingerprint:
                raise BaselineProtocolError(
                    f"{report.kind.value} usage belongs to another baseline plan"
                )
            if report.available != arm.ceilings:
                raise BaselineProtocolError(
                    f"{report.kind.value} available ceilings do not match its frozen allocation"
                )
            if report.executed_search_run_seeds != arm.evaluation.search_run_seeds:
                raise BaselineProtocolError(
                    f"{report.kind.value} did not report the frozen independent search runs"
                )
            if report.executed_repeat_seeds != arm.evaluation.repeat_seeds:
                raise BaselineProtocolError(
                    f"{report.kind.value} did not report the frozen paired repeat seeds"
                )
            cls._validate_budget(report)
            cls._validate_permissions(arm, report)

        return BaselineProtocolConsistencyReport(
            plan_fingerprint=validated_plan.fingerprint,
            baseline_kinds=tuple(sorted(kinds, key=lambda kind: kind.value)),
        )

    @staticmethod
    def _validate_budget(report: BaselineUsageReport) -> None:
        checks = (
            ("evaluations", report.used.evaluations, report.available.max_evaluations),
            (
                "feedback_queries",
                report.used.feedback_queries,
                report.available.max_feedback_queries,
            ),
            ("proposals", report.used.proposals, report.available.max_proposals),
            (
                "optimizer_model_calls",
                report.used.optimizer_model_calls,
                report.available.max_optimizer_model_calls,
            ),
            ("tokens", report.used.tokens, report.available.max_tokens),
            (
                "wall_time_seconds",
                report.used.wall_time_seconds,
                report.available.max_wall_time_seconds,
            ),
            ("cost_usd", report.used.cost_usd, report.available.max_cost_usd),
        )
        for field_name, used, available in checks:
            if used > available:
                raise BaselineProtocolError(
                    f"{report.kind.value} exceeded {field_name}: used={used}, available={available}"
                )

    @staticmethod
    def _validate_permissions(
        arm: BaselineArmPlan,
        report: BaselineUsageReport,
    ) -> None:
        used_feedback = frozenset(report.feedback_used)
        forbidden = used_feedback.intersection(FORBIDDEN_ITEM_FEEDBACK)
        if forbidden:
            raise BaselineProtocolError(
                f"{report.kind.value} used forbidden gate/sealed item feedback"
            )
        if not used_feedback.issubset(arm.available_feedback):
            raise BaselineProtocolError(
                f"{report.kind.value} used feedback outside its information permissions"
            )

        mutated = frozenset(report.mutated_component_kinds)
        allowed = frozenset(arm.mutation.mutable_component_kinds)
        if not mutated.issubset(allowed):
            raise BaselineProtocolError(
                f"{report.kind.value} mutated a component outside its condition profile"
            )
        if report.used.proposals == 0 and mutated:
            raise BaselineProtocolError(
                f"{report.kind.value} reports mutations without any used proposal"
            )
        if report.used.proposals > 0 and not mutated:
            raise BaselineProtocolError(
                f"{report.kind.value} reports proposals without a mutated component kind"
            )
        if not arm.mutation.may_call_optimizer_model and report.used.optimizer_model_calls:
            raise BaselineProtocolError(f"{report.kind.value} may not use optimizer model calls")
        if report.kind is BaselineKind.STATIC:
            search_use = (
                report.used.feedback_queries,
                report.used.proposals,
                report.used.optimizer_model_calls,
            )
            if any(search_use) or report.mutated_component_kinds:
                raise BaselineProtocolError(
                    "static may leave search resources unused but may not claim search consumption"
                )


__all__ = [
    "BASELINE_STUDY_PLAN_MEDIA_TYPE",
    "FORBIDDEN_ITEM_FEEDBACK",
    "LEGACY_BASELINE_KINDS",
    "REQUIRED_BASELINES",
    "BaselineArmPlan",
    "BaselineKind",
    "BaselineProtocolConsistencyReport",
    "BaselineProtocolError",
    "BaselineProtocolValidator",
    "BaselineStudyPlan",
    "BaselineUsageReport",
    "FeedbackType",
    "FrozenMutationPolicy",
    "FrozenRunContext",
    "MutationCapability",
    "MutationMode",
    "MutationOperation",
    "PairedEvaluationPlan",
    "ResourceCeilings",
    "ResourceUsage",
    "plan_four_baselines",
]
