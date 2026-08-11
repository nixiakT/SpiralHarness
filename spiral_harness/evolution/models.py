"""Immutable contracts for diagnosis, proposal, screening, and nomination.

The substantive artifacts referenced by this module live in the content
addressed store.  Search events carry :class:`ArtifactRef` values rather than
paths, mutable objects, or caller-authored bare hashes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.experiment import EXPERIMENT_MANIFEST_MEDIA_TYPE
from spiral_harness.core.models import (
    ArtifactRef,
    ComponentKind,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.evolution.seeds import derive_strategy_seed
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineKind,
    FeedbackType,
    FrozenMutationPolicy,
    MutationCapability,
    MutationMode,
    MutationOperation,
)
from spiral_harness.verification.models import Decision

BoundedText = Annotated[str, Field(min_length=1, max_length=2_048)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]

GATE_AGGREGATE_VIEW_MEDIA_TYPE = "application/vnd.spiral-harness.gate-aggregate-view.v1+json"
STRATEGY_FEEDBACK_VIEW_MEDIA_TYPE = "application/vnd.spiral-harness.strategy-feedback-view.v1+json"
RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.run-bound-strategy-feedback.v1+json"
)
SEARCH_POLICY_MEDIA_TYPE = "application/vnd.spiral-harness.search-policy.v1+json"
STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.strategy-plugin-manifest.v1+json"
)
SEARCH_RUN_MANIFEST_MEDIA_TYPE = "application/vnd.spiral-harness.search-run-manifest.v1+json"
SEARCH_STOPPING_POLICY_MEDIA_TYPE = "application/vnd.spiral-harness.search-stopping-policy.v1+json"
PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.prompt-mutation-catalogue.v1+json"
)
DIAGNOSIS_MEDIA_TYPE = "application/vnd.spiral-harness.diagnosis.v1+json"
PROMPT_PROPOSAL_MEDIA_TYPE = "application/vnd.spiral-harness.prompt-proposal.v1+json"
PROPOSAL_BATCH_MEDIA_TYPE = "application/vnd.spiral-harness.proposal-batch.v1+json"
CANDIDATE_SCREEN_MEDIA_TYPE = "application/vnd.spiral-harness.candidate-screen.v1+json"
NOMINATION_MEDIA_TYPE = "application/vnd.spiral-harness.nomination.v1+json"


def _artifact_key(ref: ArtifactRef) -> tuple[str, int, str]:
    return ref.sha256, ref.size, ref.media_type


def _canonical_artifact_refs(
    refs: tuple[ArtifactRef, ...],
    *,
    field_name: str,
) -> tuple[ArtifactRef, ...]:
    hashes = tuple(ref.sha256 for ref in refs)
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"{field_name} must not contain duplicate artifacts")
    return tuple(sorted(refs, key=_artifact_key))


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


def _require_text_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if not media_type.startswith("text/"):
        raise ValueError(f"{field_name} must declare a text media type")


_COMMON_SEARCH_FEEDBACK = frozenset(
    {
        FeedbackType.BENCHMARK_METADATA,
        FeedbackType.EXPLORATION_INPUTS,
        FeedbackType.GATE_AGGREGATES,
    }
)


def expected_strategy_feedback(kind: BaselineKind) -> frozenset[FeedbackType]:
    """Return the exact information capability for one frozen condition."""

    if kind is BaselineKind.STATIC:
        return frozenset({FeedbackType.BENCHMARK_METADATA})
    if kind is BaselineKind.RANDOM_VALID:
        return _COMMON_SEARCH_FEEDBACK
    ordinary_optimizer_feedback = _COMMON_SEARCH_FEEDBACK | {
        FeedbackType.EXPLORATION_AGGREGATES,
        FeedbackType.EXPLORATION_ITEM_FEEDBACK,
        FeedbackType.EXPLORATION_TRAJECTORIES,
    }
    if kind is BaselineKind.PROMPT_ONLY:
        return ordinary_optimizer_feedback
    return ordinary_optimizer_feedback | {FeedbackType.DIAGNOSTIC_EVIDENCE}


def expected_strategy_mutation(kind: BaselineKind) -> MutationCapability:
    """Return the first-search-loop capability over the shared prompt grammar."""

    if kind is BaselineKind.STATIC:
        return MutationCapability(mode=MutationMode.NONE)
    if kind is BaselineKind.RANDOM_VALID:
        return MutationCapability(
            mode=MutationMode.UNIFORM_RANDOM_VALID,
            mutable_component_kinds=(ComponentKind.PROMPT,),
        )
    if kind is BaselineKind.PROMPT_ONLY:
        return MutationCapability(
            mode=MutationMode.PROMPT_OPTIMIZATION,
            mutable_component_kinds=(ComponentKind.PROMPT,),
            may_call_optimizer_model=True,
        )
    return MutationCapability(
        mode=MutationMode.EVIDENCE_TARGETED,
        mutable_component_kinds=(ComponentKind.PROMPT,),
        may_use_diagnostic_evidence=True,
        may_call_optimizer_model=True,
    )


class SearchStoppingPolicy(ImmutableModel):
    """Typed stopping boundary persisted independently for controller replay."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    max_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_gate_queries: Annotated[int, Field(ge=0, strict=True)]
    patience_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_consecutive_declines: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def _condition_has_an_exact_stopping_profile(self) -> Self:
        values = (
            self.max_rounds,
            self.max_gate_queries,
            self.patience_rounds,
            self.max_consecutive_declines,
        )
        if self.baseline_kind is BaselineKind.STATIC:
            if any(values):
                raise ValueError("static stopping counts must all be zero")
            return self
        if any(value == 0 for value in values):
            raise ValueError("non-static stopping counts must all be positive")
        return self


class SearchPolicy(ImmutableModel):
    """Frozen strategy boundary shared by all four first-version conditions."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    mutation_policy: FrozenMutationPolicy
    available_feedback: tuple[FeedbackType, ...]
    mutation: MutationCapability
    objective: Literal["benchmark-score"] = "benchmark-score"
    objective_direction: Literal["maximize"] = "maximize"
    selector: Literal["exploration-lcb-v1"] = "exploration-lcb-v1"
    max_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_gate_queries: Annotated[int, Field(ge=0, strict=True)]
    patience_rounds: Annotated[int, Field(ge=0, strict=True)]
    max_consecutive_declines: Annotated[int, Field(ge=0, strict=True)]
    max_diagnoses: Annotated[int, Field(ge=0, strict=True)]
    max_proposals: Annotated[int, Field(ge=0, strict=True)]
    max_screens: Annotated[int, Field(ge=0, strict=True)]
    max_proposals_per_round: Annotated[int, Field(ge=0, strict=True)]
    max_candidates_screened_per_round: Annotated[int, Field(ge=0, strict=True)]
    family_alpha: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    gate_confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    max_nominations_per_round: Literal[1] = 1
    allow_decline: Literal[True] = True
    nomination_rule: Literal[
        "eligible,confidence-lower-desc,mean-delta-desc,tokens-ratio-asc,"
        "latency-ratio-asc,candidate-ref-sha256-asc,proposal-ref-sha256-asc"
    ] = (
        "eligible,confidence-lower-desc,mean-delta-desc,tokens-ratio-asc,"
        "latency-ratio-asc,candidate-ref-sha256-asc,proposal-ref-sha256-asc"
    )

    @field_validator("available_feedback")
    @classmethod
    def _canonicalize_feedback(
        cls,
        value: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(value, key=lambda feedback: feedback.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("available_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _enforce_first_version_profile(self) -> Self:
        # Every condition receives the exact same prompt-only action grammar.
        # The condition capability, not a different grammar, creates treatment.
        if self.mutation_policy.allowed_component_kinds != (ComponentKind.PROMPT,):
            raise ValueError("the first automatic search requires a shared prompt-only grammar")
        if self.mutation_policy.allowed_operations != (MutationOperation.REPLACE,):
            raise ValueError("the first automatic search supports prompt replacement only")
        if frozenset(self.available_feedback) != expected_strategy_feedback(self.baseline_kind):
            raise ValueError(
                f"{self.baseline_kind.value} feedback permissions do not match its profile"
            )
        if self.mutation != expected_strategy_mutation(self.baseline_kind):
            raise ValueError(
                f"{self.baseline_kind.value} mutation permissions do not match its profile"
            )
        SearchStoppingPolicy(
            baseline_kind=self.baseline_kind,
            max_rounds=self.max_rounds,
            max_gate_queries=self.max_gate_queries,
            patience_rounds=self.patience_rounds,
            max_consecutive_declines=self.max_consecutive_declines,
        )
        search_counts = (
            self.max_diagnoses,
            self.max_proposals,
            self.max_screens,
            self.max_proposals_per_round,
            self.max_candidates_screened_per_round,
        )
        if self.baseline_kind is BaselineKind.STATIC:
            if any(search_counts):
                raise ValueError("static diagnosis/proposal/screen counts must all be zero")
            if self.gate_confidence_level is not None:
                raise ValueError("static does not have a gate confidence level")
            return self
        if any(
            value == 0
            for value in (
                self.max_proposals,
                self.max_screens,
                self.max_proposals_per_round,
                self.max_candidates_screened_per_round,
            )
        ):
            raise ValueError("non-static proposal and screen counts must all be positive")
        if self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            if self.max_diagnoses == 0:
                raise ValueError("evidence-targeted requires a positive diagnosis limit")
        elif self.max_diagnoses != 0:
            raise ValueError("only evidence-targeted may have a diagnosis limit")
        if self.max_proposals_per_round > self.max_candidates_screened_per_round:
            raise ValueError(
                "max_candidates_screened_per_round must cover every proposal in a round"
            )
        if self.max_proposals_per_round > self.max_proposals:
            raise ValueError("per-round proposals must not exceed max_proposals")
        if self.max_candidates_screened_per_round > self.max_screens:
            raise ValueError("per-round screens must not exceed max_screens")
        if self.max_gate_queries > self.max_proposals:
            raise ValueError("max_gate_queries must not exceed max_proposals")
        expected_confidence = 1.0 - (self.family_alpha / self.max_gate_queries)
        if self.gate_confidence_level != expected_confidence:
            raise ValueError("gate_confidence_level must equal 1 - family_alpha / max_gate_queries")
        return self

    @property
    def stopping_policy(self) -> SearchStoppingPolicy:
        """Project the exact independently persisted stopping artifact."""

        return SearchStoppingPolicy(
            baseline_kind=self.baseline_kind,
            max_rounds=self.max_rounds,
            max_gate_queries=self.max_gate_queries,
            patience_rounds=self.patience_rounds,
            max_consecutive_declines=self.max_consecutive_declines,
        )


class StrategyPluginManifest(ImmutableModel):
    """Auditable declarations for a strategy implementation plugin.

    Declaring a permission does not grant it.  Both this manifest and the
    frozen :class:`SearchPolicy` are rechecked by the strategy boundary.
    """

    schema_version: Literal["1"] = "1"
    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    implementation_ref: ArtifactRef
    baseline_kind: BaselineKind
    consumes_feedback: tuple[FeedbackType, ...]
    mutation: MutationCapability
    uses_finite_prompt_catalogue: bool
    emits_prompt_proposals: bool

    @field_validator("consumes_feedback")
    @classmethod
    def _canonicalize_feedback(
        cls,
        value: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(value, key=lambda feedback: feedback.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("consumes_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _enforce_declared_profile(self) -> Self:
        if frozenset(self.consumes_feedback) != expected_strategy_feedback(self.baseline_kind):
            raise ValueError(
                f"{self.baseline_kind.value} plugin feedback declarations do not match its profile"
            )
        if self.mutation != expected_strategy_mutation(self.baseline_kind):
            raise ValueError(
                f"{self.baseline_kind.value} plugin mutation declarations do not match its profile"
            )
        should_use_catalogue = self.baseline_kind is BaselineKind.RANDOM_VALID
        if self.uses_finite_prompt_catalogue is not should_use_catalogue:
            raise ValueError("uses_finite_prompt_catalogue must be true only for random-valid")
        should_emit = self.baseline_kind is not BaselineKind.STATIC
        if self.emits_prompt_proposals is not should_emit:
            raise ValueError("static cannot emit proposals and every search arm must be able to")
        return self


class SearchRunManifest(ImmutableModel):
    """Freeze one baseline arm and one independent search-run seed."""

    schema_version: Literal["1"] = "1"
    baseline_study_plan_ref: ArtifactRef
    experiment_ref: ArtifactRef
    search_policy_ref: ArtifactRef
    search_policy_fingerprint: Sha256
    strategy_plugin_ref: ArtifactRef
    strategy_plugin_fingerprint: Sha256
    analysis_plan_ref: ArtifactRef
    stopping_policy_ref: ArtifactRef
    stopping_policy_fingerprint: Sha256
    seed_harness_ref: ArtifactRef
    baseline_kind: BaselineKind
    baseline_plan_fingerprint: Sha256
    proposal_master_seed: NonNegativeInt
    search_run_seed: NonNegativeInt
    repeat_seeds: Annotated[tuple[NonNegativeInt, ...], Field(min_length=2)]
    strategy_seed: NonNegativeInt
    seed_derivation_domain: Literal["spiral-harness/evolution/strategy-seed/v1"] = (
        "spiral-harness/evolution/strategy-seed/v1"
    )
    prompt_mutation_catalogue_ref: ArtifactRef | None = None

    @field_validator("repeat_seeds")
    @classmethod
    def _canonicalize_repeat_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("repeat_seeds must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        for field_name in (
            "baseline_study_plan_ref",
            "experiment_ref",
            "search_policy_ref",
            "strategy_plugin_ref",
            "analysis_plan_ref",
            "stopping_policy_ref",
            "seed_harness_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)
        if self.baseline_study_plan_ref.media_type != BASELINE_STUDY_PLAN_MEDIA_TYPE:
            raise ValueError("baseline_study_plan_ref declares the wrong media type")
        if self.experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise ValueError("experiment_ref declares the wrong media type")
        if self.search_policy_ref.media_type != SEARCH_POLICY_MEDIA_TYPE:
            raise ValueError("search_policy_ref declares the wrong media type")
        if self.strategy_plugin_ref.media_type != STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE:
            raise ValueError("strategy_plugin_ref declares the wrong media type")
        if self.stopping_policy_ref.media_type != SEARCH_STOPPING_POLICY_MEDIA_TYPE:
            raise ValueError("stopping_policy_ref declares the wrong media type")
        if self.search_policy_fingerprint != self.search_policy_ref.sha256:
            raise ValueError("search_policy_fingerprint does not match search_policy_ref")
        if self.strategy_plugin_fingerprint != self.strategy_plugin_ref.sha256:
            raise ValueError("strategy_plugin_fingerprint does not match strategy_plugin_ref")
        if self.stopping_policy_fingerprint != self.stopping_policy_ref.sha256:
            raise ValueError("stopping_policy_fingerprint does not match stopping_policy_ref")
        if self.baseline_plan_fingerprint != self.baseline_study_plan_ref.sha256:
            raise ValueError("baseline_plan_fingerprint does not match baseline_study_plan_ref")
        if self.search_run_seed in self.repeat_seeds:
            raise ValueError("search_run_seed must be disjoint from repeat_seeds")
        expected_seed = derive_strategy_seed(
            proposal_master_seed=self.proposal_master_seed,
            search_run_seed=self.search_run_seed,
            baseline_kind=self.baseline_kind,
        )
        if self.strategy_seed != expected_seed:
            raise ValueError("strategy_seed does not match the domain-separated derivation")
        if self.baseline_kind is BaselineKind.RANDOM_VALID:
            if self.prompt_mutation_catalogue_ref is None:
                raise ValueError("random-valid requires a frozen prompt mutation catalogue")
            _require_json_ref(
                self.prompt_mutation_catalogue_ref,
                field_name="prompt_mutation_catalogue_ref",
            )
            if (
                self.prompt_mutation_catalogue_ref.media_type
                != PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE
            ):
                raise ValueError("prompt_mutation_catalogue_ref declares the wrong media type")
        elif self.prompt_mutation_catalogue_ref is not None:
            raise ValueError("only random-valid may bind a prompt mutation catalogue")
        return self


class GateAggregateMetrics(ImmutableModel):
    """Pre-registered aggregate gate statistics with no item coordinates."""

    schema_version: Literal["1"] = "1"
    n_valid_pairs: NonNegativeInt
    n_tasks: NonNegativeInt
    mean_delta: FiniteScore
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    confidence_lower: FiniteScore
    confidence_upper: FiniteScore
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tokens_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    latency_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    tool_calls_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    parent_score_mean: FiniteScore | None = None
    candidate_score_mean: FiniteScore | None = None
    wins: NonNegativeInt | None = None
    ties: NonNegativeInt | None = None
    losses: NonNegativeInt | None = None
    worst_task_delta: FiniteScore | None = None
    parent_tokens_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    candidate_tokens_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    parent_latency_ms_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    candidate_latency_ms_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    parent_tool_calls_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    candidate_tool_calls_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def _aggregate_counts_are_consistent(self) -> Self:
        outcome_counts = (self.wins, self.ties, self.losses)
        if any(value is None for value in outcome_counts) != all(
            value is None for value in outcome_counts
        ):
            raise ValueError("wins, ties, and losses must be present together")
        if all(value is not None for value in outcome_counts) and (
            sum(outcome_counts) != self.n_tasks  # type: ignore[arg-type]
        ):
            raise ValueError("wins, ties, and losses must sum to n_tasks")
        if self.n_valid_pairs < self.n_tasks:
            raise ValueError("n_valid_pairs must be greater than or equal to n_tasks")
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("confidence_lower must not exceed confidence_upper")
        return self


class GateAggregateView(ImmutableModel):
    """The only gate feedback shape visible to a search strategy.

    There are intentionally no task IDs, comparisons, item outcomes,
    trajectories, slice names, or arbitrary reason strings in this schema.
    """

    schema_version: Literal["1"] = "1"
    candidate_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    query_index: NonNegativeInt
    decision: Decision
    gate_version: NonEmptyStr
    gate_config_sha256: Sha256
    redaction_schema: Literal["search-controller-aggregate-redaction-v1"] = (
        "search-controller-aggregate-redaction-v1"
    )
    resolution: Literal["gate-decision", "superseded-promotion"] = "gate-decision"
    metrics: GateAggregateMetrics | None
    passed_check_count: NonNegativeInt
    failed_check_count: NonNegativeInt
    inconclusive_check_count: NonNegativeInt

    @model_validator(mode="after")
    def _decision_matches_aggregate_checks(self) -> Self:
        _require_json_ref(self.candidate_ref, field_name="candidate_ref")
        _require_json_ref(self.analysis_plan_ref, field_name="analysis_plan_ref")
        if self.passed_check_count + self.failed_check_count + self.inconclusive_check_count == 0:
            raise ValueError("gate aggregate must contain at least one check outcome")
        if self.decision is Decision.PROMOTE:
            if self.resolution != "gate-decision":
                raise ValueError("a promotion cannot use a selection-level resolution override")
            if self.metrics is None:
                raise ValueError("a promotion aggregate requires metrics")
            if self.failed_check_count or self.inconclusive_check_count:
                raise ValueError("a promotion aggregate cannot contain failed checks")
        elif self.decision is Decision.REJECT:
            if self.resolution != "gate-decision":
                raise ValueError("a rejection cannot use a selection-level resolution override")
            if self.failed_check_count == 0:
                raise ValueError("a rejection aggregate requires at least one failed check")
        else:
            if self.failed_check_count:
                raise ValueError("a gate with failed checks must be rejected")
            if self.resolution == "gate-decision":
                if self.inconclusive_check_count == 0:
                    raise ValueError("an inconclusive gate requires an inconclusive check")
            elif self.inconclusive_check_count or not self.passed_check_count:
                raise ValueError(
                    "a superseded promotion must preserve its all-pass gate check counts"
                )
        return self


class StrategyFeedbackView(ImmutableModel):
    """A structural information-flow view for one strategy invocation.

    ``exposed_feedback`` is audited against the fields that actually exist; it
    cannot authorize a hidden ref merely by omitting that ref from the claim.
    Gate feedback is embedded only through :class:`GateAggregateView`, whose
    schema cannot express item-level content.
    """

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    exposed_feedback: tuple[FeedbackType, ...]
    benchmark_metadata_ref: ArtifactRef
    exploration_inputs_ref: ArtifactRef | None = None
    exploration_aggregates_ref: ArtifactRef | None = None
    exploration_item_feedback_ref: ArtifactRef | None = None
    exploration_trajectories_ref: ArtifactRef | None = None
    diagnostic_evidence_ref: ArtifactRef | None = None
    gate_aggregate: GateAggregateView | None = None

    @field_validator("exposed_feedback")
    @classmethod
    def _canonicalize_exposed_feedback(
        cls,
        value: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(value, key=lambda feedback: feedback.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("exposed_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _validate_actual_disclosure(self) -> Self:
        actual: set[FeedbackType] = {FeedbackType.BENCHMARK_METADATA}
        optional_ref_fields = (
            ("exploration_inputs_ref", FeedbackType.EXPLORATION_INPUTS),
            ("exploration_aggregates_ref", FeedbackType.EXPLORATION_AGGREGATES),
            (
                "exploration_item_feedback_ref",
                FeedbackType.EXPLORATION_ITEM_FEEDBACK,
            ),
            (
                "exploration_trajectories_ref",
                FeedbackType.EXPLORATION_TRAJECTORIES,
            ),
            ("diagnostic_evidence_ref", FeedbackType.DIAGNOSTIC_EVIDENCE),
        )
        _require_json_ref(self.benchmark_metadata_ref, field_name="benchmark_metadata_ref")
        for field_name, feedback_type in optional_ref_fields:
            ref = getattr(self, field_name)
            if ref is not None:
                _require_json_ref(ref, field_name=field_name)
                actual.add(feedback_type)
        if self.gate_aggregate is not None:
            actual.add(FeedbackType.GATE_AGGREGATES)

        if frozenset(self.exposed_feedback) != frozenset(actual):
            raise ValueError("exposed_feedback must exactly describe the fields actually present")
        permitted = expected_strategy_feedback(self.baseline_kind)
        if not actual.issubset(permitted):
            forbidden = ", ".join(sorted(value.value for value in actual.difference(permitted)))
            raise ValueError(
                f"{self.baseline_kind.value} feedback view contains forbidden fields: {forbidden}"
            )

        required = {FeedbackType.BENCHMARK_METADATA}
        if self.baseline_kind is not BaselineKind.STATIC:
            required.add(FeedbackType.EXPLORATION_INPUTS)
        if self.baseline_kind in {
            BaselineKind.PROMPT_ONLY,
            BaselineKind.EVIDENCE_TARGETED,
        }:
            required.update(
                {
                    FeedbackType.EXPLORATION_AGGREGATES,
                    FeedbackType.EXPLORATION_ITEM_FEEDBACK,
                    FeedbackType.EXPLORATION_TRAJECTORIES,
                }
            )
        if self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            required.add(FeedbackType.DIAGNOSTIC_EVIDENCE)
        missing = required.difference(actual)
        if missing:
            joined = ", ".join(sorted(value.value for value in missing))
            raise ValueError(
                f"{self.baseline_kind.value} feedback view is missing required fields: {joined}"
            )
        return self


# Short public name retained for orchestration code that does not need the
# strategy qualifier.  It is the same strict model, not a weaker base class.
FeedbackView = StrategyFeedbackView


class Diagnosis(ImmutableModel):
    """A source-linked causal diagnosis available only to evidence-targeted search."""

    schema_version: Literal["1"] = "1"
    diagnosis_id: NonEmptyStr
    baseline_kind: Literal[BaselineKind.EVIDENCE_TARGETED] = BaselineKind.EVIDENCE_TARGETED
    source_feedback_ref: ArtifactRef
    failure_signature_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    evidence_packet_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    protected_anchor_refs: tuple[ArtifactRef, ...] = ()
    target_component_name: NonEmptyStr
    target_component_kind: Literal[ComponentKind.PROMPT] = ComponentKind.PROMPT
    observed_failure: BoundedText
    root_cause: BoundedText
    mechanism: BoundedText
    predicted_benefit: BoundedText
    predicted_risk: BoundedText
    falsifier: BoundedText

    @field_validator(
        "failure_signature_refs",
        "evidence_packet_refs",
        "protected_anchor_refs",
    )
    @classmethod
    def _canonicalize_diagnostic_refs(
        cls,
        value: tuple[ArtifactRef, ...],
        info: object,
    ) -> tuple[ArtifactRef, ...]:
        field_name = getattr(info, "field_name", "diagnostic_refs")
        return _canonical_artifact_refs(value, field_name=field_name)

    @model_validator(mode="after")
    def _refs_are_typed_json_artifacts(self) -> Self:
        if self.source_feedback_ref.media_type != RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE:
            raise ValueError("source_feedback_ref declares the wrong media type")
        for field_name in (
            "failure_signature_refs",
            "evidence_packet_refs",
            "protected_anchor_refs",
        ):
            for ref in getattr(self, field_name):
                _require_json_ref(ref, field_name=field_name)
        return self


class DeclineReason(StrEnum):
    """Bounded reasons for producing no candidate in a search round."""

    STATIC_CONDITION = "static-condition"
    NO_ACTIONABLE_DIAGNOSIS = "no-actionable-diagnosis"
    NO_ELIGIBLE_CATALOGUE_ENTRY = "no-eligible-catalogue-entry"
    OPTIMIZER_DECLINED = "optimizer-declined"
    POLICY_FILTERED_ALL = "policy-filtered-all"
    BUDGET_EXHAUSTED = "budget-exhausted"


class ProposalDecline(ImmutableModel):
    """An explicit, auditable choice to emit no proposal."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    round_index: NonNegativeInt
    source_feedback_ref: ArtifactRef
    reason: DeclineReason
    rationale: BoundedText
    diagnosis_refs: tuple[ArtifactRef, ...] = ()

    @field_validator("diagnosis_refs")
    @classmethod
    def _canonicalize_diagnosis_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        return _canonical_artifact_refs(value, field_name="diagnosis_refs")

    @model_validator(mode="after")
    def _validate_decline(self) -> Self:
        if self.source_feedback_ref.media_type != RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE:
            raise ValueError("source_feedback_ref declares the wrong media type")
        for ref in self.diagnosis_refs:
            if ref.media_type != DIAGNOSIS_MEDIA_TYPE:
                raise ValueError("diagnosis_refs declare the wrong media type")
        if self.baseline_kind is not BaselineKind.EVIDENCE_TARGETED and self.diagnosis_refs:
            raise ValueError("only evidence-targeted declines may cite diagnoses")
        if (
            self.baseline_kind is BaselineKind.STATIC
            and self.reason is not DeclineReason.STATIC_CONDITION
        ):
            raise ValueError("static must decline with reason static-condition")
        if (
            self.baseline_kind is not BaselineKind.STATIC
            and self.reason is DeclineReason.STATIC_CONDITION
        ):
            raise ValueError("only static may use reason static-condition")
        if (
            self.reason is DeclineReason.NO_ELIGIBLE_CATALOGUE_ENTRY
            and self.baseline_kind is not BaselineKind.RANDOM_VALID
        ):
            raise ValueError("only random-valid may exhaust the prompt mutation catalogue")
        return self


class PromptProposal(ImmutableModel):
    """One atomic non-noop prompt replacement under the shared grammar."""

    schema_version: Literal["1"] = "1"
    proposal_id: NonEmptyStr
    baseline_kind: BaselineKind
    round_index: NonNegativeInt
    parent_harness_ref: ArtifactRef
    target_component_name: NonEmptyStr
    target_component_kind: Literal[ComponentKind.PROMPT] = ComponentKind.PROMPT
    before_prompt_ref: ArtifactRef
    after_prompt_ref: ArtifactRef
    hypothesis_ref: ArtifactRef
    mechanism_family: NonEmptyStr
    diagnosis_ref: ArtifactRef | None = None
    catalogue_ref: ArtifactRef | None = None
    catalogue_entry_id: NonEmptyStr | None = None
    proposer_confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def _validate_prompt_proposal(self) -> Self:
        if self.baseline_kind is BaselineKind.STATIC:
            raise ValueError("static cannot emit a prompt proposal")
        _require_json_ref(self.parent_harness_ref, field_name="parent_harness_ref")
        _require_text_ref(self.before_prompt_ref, field_name="before_prompt_ref")
        _require_text_ref(self.after_prompt_ref, field_name="after_prompt_ref")
        _require_json_ref(self.hypothesis_ref, field_name="hypothesis_ref")
        if self.before_prompt_ref.sha256 == self.after_prompt_ref.sha256:
            raise ValueError("prompt proposals must be non-noop replacements")

        if self.baseline_kind is BaselineKind.RANDOM_VALID:
            if self.catalogue_ref is None or self.catalogue_entry_id is None:
                raise ValueError("random-valid proposals must cite a finite catalogue entry")
            if self.catalogue_ref.media_type != PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE:
                raise ValueError("catalogue_ref declares the wrong media type")
            if self.diagnosis_ref is not None or self.proposer_confidence is not None:
                raise ValueError("random-valid cannot use diagnosis or proposer confidence")
        elif self.catalogue_ref is not None or self.catalogue_entry_id is not None:
            raise ValueError("optimizer proposals cannot claim finite-catalogue provenance")

        if self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            if self.diagnosis_ref is None:
                raise ValueError("evidence-targeted proposals must cite a typed diagnosis")
            if self.diagnosis_ref.media_type != DIAGNOSIS_MEDIA_TYPE:
                raise ValueError("diagnosis_ref declares the wrong media type")
        elif self.diagnosis_ref is not None:
            raise ValueError("only evidence-targeted proposals may cite typed diagnoses")
        return self


class ProposalBatch(ImmutableModel):
    """Exactly one of a non-empty proposal set or an explicit decline."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    round_index: NonNegativeInt
    source_feedback_ref: ArtifactRef
    diagnosis_refs: tuple[ArtifactRef, ...] = ()
    proposals: tuple[PromptProposal, ...] = ()
    decline: ProposalDecline | None = None

    @field_validator("diagnosis_refs")
    @classmethod
    def _canonicalize_diagnosis_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        return _canonical_artifact_refs(value, field_name="diagnosis_refs")

    @field_validator("proposals")
    @classmethod
    def _canonicalize_proposals(
        cls,
        value: tuple[PromptProposal, ...],
    ) -> tuple[PromptProposal, ...]:
        return tuple(sorted(value, key=lambda proposal: proposal.proposal_id))

    @model_validator(mode="after")
    def _enforce_proposal_decline_xor(self) -> Self:
        if self.source_feedback_ref.media_type != RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE:
            raise ValueError("source_feedback_ref declares the wrong media type")
        has_proposals = bool(self.proposals)
        has_decline = self.decline is not None
        if has_proposals == has_decline:
            raise ValueError("exactly one of proposals or decline must be present")

        proposal_ids = tuple(proposal.proposal_id for proposal in self.proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("proposal IDs must be unique within a batch")
        mutation_coordinates = tuple(
            (
                proposal.target_component_name,
                proposal.before_prompt_ref.sha256,
                proposal.after_prompt_ref.sha256,
            )
            for proposal in self.proposals
        )
        if len(mutation_coordinates) != len(set(mutation_coordinates)):
            raise ValueError("a proposal batch must not duplicate a prompt mutation")

        for ref in self.diagnosis_refs:
            if ref.media_type != DIAGNOSIS_MEDIA_TYPE:
                raise ValueError("diagnosis_refs declare the wrong media type")
        diagnosis_hashes = frozenset(ref.sha256 for ref in self.diagnosis_refs)
        for proposal in self.proposals:
            if proposal.baseline_kind is not self.baseline_kind:
                raise ValueError("every proposal must belong to the batch baseline")
            if proposal.round_index != self.round_index:
                raise ValueError("every proposal must belong to the batch round")
            if (
                proposal.diagnosis_ref is not None
                and proposal.diagnosis_ref.sha256 not in diagnosis_hashes
            ):
                raise ValueError("proposal diagnosis_ref must be declared by the batch")

        if self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            if has_proposals and not self.diagnosis_refs:
                raise ValueError("evidence-targeted proposals require diagnosis_refs")
        elif self.diagnosis_refs:
            raise ValueError("only evidence-targeted batches may contain diagnosis_refs")

        if self.decline is not None:
            if self.decline.baseline_kind is not self.baseline_kind:
                raise ValueError("decline must belong to the batch baseline")
            if self.decline.round_index != self.round_index:
                raise ValueError("decline must belong to the batch round")
            if self.decline.source_feedback_ref != self.source_feedback_ref:
                raise ValueError("decline must cite the batch feedback view")
            if self.decline.diagnosis_refs != self.diagnosis_refs:
                raise ValueError("decline diagnosis_refs must equal the batch diagnosis_refs")
        if self.baseline_kind is BaselineKind.STATIC and self.decline is None:
            raise ValueError("static must take the explicit decline branch")
        return self


# More explicit synonym for callers that keep other component proposal batches.
PromptProposalBatch = ProposalBatch
Decline = ProposalDecline


class CandidateScreenStatus(StrEnum):
    """Trusted exploration-screen disposition before gate nomination."""

    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class CandidateScreenFailure(StrEnum):
    """Bounded failure vocabulary; item-level feedback remains out of line."""

    STRUCTURALLY_INVALID = "structurally-invalid"
    POLICY_VIOLATION = "policy-violation"
    DUPLICATE_CANDIDATE = "duplicate-candidate"
    EVALUATION_INCOMPLETE = "evaluation-incomplete"
    OBJECTIVE_UNAVAILABLE = "objective-unavailable"
    CONSTRAINT_FAILED = "constraint-failed"
    BUDGET_UNAVAILABLE = "budget-unavailable"


class CandidateScreen(ImmutableModel):
    """One trusted structural/exploration screen result."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    round_index: NonNegativeInt
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef | None = None
    candidate_harness_ref: ArtifactRef | None = None
    evaluation_ref: ArtifactRef | None = None
    status: CandidateScreenStatus
    primary_score: FiniteScore | None = None
    mean_delta: FiniteScore | None = None
    confidence_lower: FiniteScore | None = None
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    failure_codes: tuple[CandidateScreenFailure, ...] = ()

    @field_validator("failure_codes")
    @classmethod
    def _canonicalize_failure_codes(
        cls,
        value: tuple[CandidateScreenFailure, ...],
    ) -> tuple[CandidateScreenFailure, ...]:
        ordered = tuple(sorted(value, key=lambda code: code.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("failure_codes must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _validate_screen(self) -> Self:
        if self.baseline_kind is BaselineKind.STATIC:
            raise ValueError("static has no candidate screens")
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        for field_name in ("candidate_ref", "candidate_harness_ref", "evaluation_ref"):
            ref = getattr(self, field_name)
            if ref is not None:
                _require_json_ref(ref, field_name=field_name)
        candidate_values = (self.candidate_ref, self.candidate_harness_ref)
        if any(value is None for value in candidate_values) != all(
            value is None for value in candidate_values
        ):
            raise ValueError("candidate_ref and candidate_harness_ref must be present together")
        evaluation_values = (
            self.evaluation_ref,
            self.primary_score,
            self.mean_delta,
            self.confidence_lower,
            self.regression_rate,
            self.tokens_ratio,
            self.latency_ratio,
        )
        if any(value is None for value in evaluation_values) != all(
            value is None for value in evaluation_values
        ):
            raise ValueError("screen evaluation ref and aggregate metrics must be present together")
        has_candidate = self.candidate_ref is not None
        has_evaluation = self.evaluation_ref is not None
        if has_evaluation and not has_candidate:
            raise ValueError("a screen evaluation requires a materialized candidate")
        if self.status is CandidateScreenStatus.ELIGIBLE:
            if not has_candidate or not has_evaluation:
                raise ValueError("an eligible screen requires candidate, evaluation, and score")
            if self.failure_codes:
                raise ValueError("an eligible screen cannot contain failure_codes")
        elif not self.failure_codes:
            raise ValueError("a non-eligible screen requires at least one failure code")
        return self


class Nomination(ImmutableModel):
    """The sole candidate admitted from one screen batch to the gate."""

    schema_version: Literal["1"] = "1"
    baseline_kind: BaselineKind
    round_index: NonNegativeInt
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    screen_evaluation_ref: ArtifactRef
    primary_score: FiniteScore
    mean_delta: FiniteScore
    confidence_lower: FiniteScore
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    eligible_candidates_considered: Annotated[int, Field(gt=0, strict=True)]
    ranking_rule: Literal[
        "eligible,confidence-lower-desc,mean-delta-desc,tokens-ratio-asc,"
        "latency-ratio-asc,candidate-ref-sha256-asc,proposal-ref-sha256-asc"
    ] = (
        "eligible,confidence-lower-desc,mean-delta-desc,tokens-ratio-asc,"
        "latency-ratio-asc,candidate-ref-sha256-asc,proposal-ref-sha256-asc"
    )

    @model_validator(mode="after")
    def _validate_nomination(self) -> Self:
        if self.baseline_kind is BaselineKind.STATIC:
            raise ValueError("static cannot nominate a candidate")
        for field_name in (
            "proposal_ref",
            "candidate_ref",
            "candidate_harness_ref",
            "screen_evaluation_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        return self


__all__ = [
    "BASELINE_STUDY_PLAN_MEDIA_TYPE",
    "CANDIDATE_SCREEN_MEDIA_TYPE",
    "DIAGNOSIS_MEDIA_TYPE",
    "GATE_AGGREGATE_VIEW_MEDIA_TYPE",
    "NOMINATION_MEDIA_TYPE",
    "PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE",
    "PROMPT_PROPOSAL_MEDIA_TYPE",
    "PROPOSAL_BATCH_MEDIA_TYPE",
    "RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE",
    "SEARCH_POLICY_MEDIA_TYPE",
    "SEARCH_RUN_MANIFEST_MEDIA_TYPE",
    "SEARCH_STOPPING_POLICY_MEDIA_TYPE",
    "STRATEGY_FEEDBACK_VIEW_MEDIA_TYPE",
    "STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE",
    "CandidateScreen",
    "CandidateScreenFailure",
    "CandidateScreenStatus",
    "Decline",
    "DeclineReason",
    "Diagnosis",
    "FeedbackView",
    "GateAggregateMetrics",
    "GateAggregateView",
    "Nomination",
    "PromptProposal",
    "PromptProposalBatch",
    "ProposalBatch",
    "ProposalDecline",
    "SearchPolicy",
    "SearchRunManifest",
    "SearchStoppingPolicy",
    "StrategyFeedbackView",
    "StrategyPluginManifest",
    "expected_strategy_feedback",
    "expected_strategy_mutation",
]
