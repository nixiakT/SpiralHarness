"""Manifest-bound admission for the matched SCORE-versus-FULL contrast.

This module freezes the coordinates that must be identical between the two
conditions and validates them against an independently supplied expectation.
It proves manifest equality only.  Planned proposer topology, a persisted
paired seed, and successful admission do not attest that either condition was
executed or that runtime calls followed the plan.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    HarnessManifest,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.evolution.matched_media_types import MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE
from spiral_harness.evolution.orchestrator import (
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    SearchBenchmarkBinding,
)
from spiral_harness.evolution.seeds import (
    MANIFEST_BOUND_PAIRED_PROPOSER_SEED_DOMAIN,
    derive_manifest_bound_paired_proposer_seed,
)
from spiral_harness.experiments.baseline_profiles import MatchedContrastProfile
from spiral_harness.experiments.baselines import BaselineKind, FeedbackType
from spiral_harness.storage.protocol import ArtifactRepository

MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-v2-study-manifest.v1+json"
)
MATCHED_V2_ADMISSION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-v2-admission-report.v1+json"
)
MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.matched-v2-gate-query-block.v1+json"
)

NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class MatchedV2AdmissionError(ValueError):
    """Raised when persisted SCORE/FULL manifests do not match expectation."""


class MatchedV2GateTask(ImmutableModel):
    """One fresh-task identity; rollout repeats never create another identity."""

    schema_version: Literal["1"] = "1"
    task_id: NonEmptyStr
    source_id: NonEmptyStr
    family_id: NonEmptyStr


class MatchedV2GateQueryBlock(ImmutableModel):
    """One fresh block with a frozen, but not yet attested, execution plan."""

    schema_version: Literal["1"] = "1"
    query_index: NonNegativeInt
    nomination_index: NonNegativeInt
    tasks: Annotated[tuple[MatchedV2GateTask, ...], Field(min_length=1)]
    attribution_arms: tuple[
        Literal["parent", "candidate", "revert", "placebo"],
        Literal["parent", "candidate", "revert", "placebo"],
        Literal["parent", "candidate", "revert", "placebo"],
        Literal["parent", "candidate", "revert", "placebo"],
    ] = ("parent", "candidate", "revert", "placebo")
    experimental_conditions: tuple[
        Literal["score-only-matched"],
        Literal["evidence-targeted"],
    ] = ("score-only-matched", "evidence-targeted")
    maximum_campaign_consumptions: Literal[1] = 1
    candidate_freeze_boundary: Literal[
        "freeze-both-condition-candidates-before-any-gate-execution"
    ] = "freeze-both-condition-candidates-before-any-gate-execution"
    planned_cross_condition_batch: Literal["score-full-cross-condition-atomic-batch"] = (
        "score-full-cross-condition-atomic-batch"
    )
    feedback_release_boundary: Literal["release-after-complete-cross-condition-batch"] = (
        "release-after-complete-cross-condition-batch"
    )
    feedback_isolation_boundary: Literal["condition-local-no-cross-condition-disclosure"] = (
        "condition-local-no-cross-condition-disclosure"
    )
    freshness_unit: Literal["task-source-family"] = "task-source-family"
    rollout_seed_counts_as_fresh_task: Literal[False] = False

    @field_validator("tasks")
    @classmethod
    def _canonicalize_tasks(
        cls,
        values: tuple[MatchedV2GateTask, ...],
    ) -> tuple[MatchedV2GateTask, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.task_id))
        if len(ordered) != len({value.task_id for value in ordered}):
            raise ValueError("gate query block task_ids must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _require_exact_attribution_quartet(self) -> Self:
        if self.nomination_index != self.query_index:
            raise ValueError("gate query and nomination indexes must be identical")
        if self.attribution_arms != ("parent", "candidate", "revert", "placebo"):
            raise ValueError("gate query block must serve the exact matched quartet")
        if self.experimental_conditions != (
            "score-only-matched",
            "evidence-targeted",
        ):
            raise ValueError("gate query block must freeze the exact SCORE/FULL pair")
        return self


class MatchedV2ExecutionCeilings(ImmutableModel):
    """Every stopping and resource coordinate shared by SCORE and FULL."""

    schema_version: Literal["1"] = "1"
    max_rounds: PositiveInt
    max_proposals_per_round: PositiveInt
    max_total_proposals: PositiveInt
    max_nominations_per_round: Literal[1] = 1
    max_total_nominations: PositiveInt
    max_optimizer_model_calls: PositiveInt
    max_solver_model_calls: PositiveInt
    max_gate_queries: PositiveInt
    max_evaluations: PositiveInt
    max_feedback_queries: PositiveInt
    max_attempts_per_evaluation: PositiveInt
    token_ceiling_per_attempt: PositiveInt
    max_tokens: PositiveInt
    max_wall_time_seconds: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
    max_cost_usd: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]

    @model_validator(mode="after")
    def _derived_limits_fit_round_schedule(self) -> Self:
        if self.max_total_proposals > self.max_rounds * self.max_proposals_per_round:
            raise ValueError("max_total_proposals exceeds the round/proposal ceiling")
        if self.max_total_nominations > self.max_rounds * self.max_nominations_per_round:
            raise ValueError("max_total_nominations exceeds the round/nomination ceiling")
        if self.max_gate_queries > self.max_total_nominations:
            raise ValueError("max_gate_queries exceeds the nomination ceiling")
        if self.max_gate_queries != self.max_total_nominations:
            raise ValueError("every nomination requires exactly one fresh gate query block")
        return self


class MatchedV2PolicyBindings(ImmutableModel):
    """Exact proposal, selection, execution, and gate implementations."""

    schema_version: Literal["1"] = "1"
    proposer_policy_fingerprint: Sha256
    nomination_policy_fingerprint: Sha256
    optimizer_config_fingerprint: Sha256
    solver_config_fingerprint: Sha256
    grader_fingerprint: NonEmptyStr
    gate_policy_fingerprint: Sha256
    performance_policy_fingerprint: Sha256
    price_table_fingerprint: Sha256


class MatchedV2PlannedTopology(ImmutableModel):
    """A frozen runtime plan whose actual use remains unverified in phase 2a."""

    schema_version: Literal["1"] = "1"
    topology_id: Literal["shared-proposer-isolated-contexts-v1"] = (
        "shared-proposer-isolated-contexts-v1"
    )
    proposer_implementation_fingerprint: Sha256
    proposer_call_graph_fingerprint: Sha256
    proposer_worker_count: Literal[1] = 1
    condition_context_count: Literal[2] = 2
    shared_mutable_state_between_conditions: Literal[False] = False
    round_dispatch: Literal["paired-seed-counterbalanced-v1"] = "paired-seed-counterbalanced-v1"


class MatchedV2SharedCoordinates(ImmutableModel):
    """Complete non-treatment coordinates of one paired independent run."""

    schema_version: Literal["1"] = "1"
    contrast: MatchedContrastProfile
    contrast_fingerprint: Sha256
    study_id: NonEmptyStr
    benchmark_binding_ref: ArtifactRef
    model_fingerprint: NonEmptyStr
    inference_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    seed_harness_ref: ArtifactRef
    proposal_master_seed: NonNegativeInt
    rollout_master_seed: NonNegativeInt
    search_run_seed: NonNegativeInt
    repeat_seeds: Annotated[tuple[NonNegativeInt, ...], Field(min_length=1)]
    gate_query_block_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    mutation_policy_fingerprint: Sha256
    action_capability_fingerprint: Sha256
    policies: MatchedV2PolicyBindings
    planned_topology: MatchedV2PlannedTopology
    ceilings: MatchedV2ExecutionCeilings

    @field_validator("repeat_seeds")
    @classmethod
    def _canonicalize_repeat_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("repeat_seeds must not contain duplicates")
        return ordered

    @field_validator("gate_query_block_refs")
    @classmethod
    def _require_ordered_unique_gate_blocks(
        cls,
        values: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        if any(ref.media_type != MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE for ref in values):
            raise ValueError("gate_query_block_refs contains the wrong media type")
        if len(values) != len({ref.sha256 for ref in values}):
            raise ValueError("gate_query_block_refs must not reuse an artifact")
        return values

    @model_validator(mode="after")
    def _bind_profile_and_seed_domains(self) -> Self:
        if self.contrast_fingerprint != self.contrast.fingerprint:
            raise ValueError("contrast_fingerprint does not match contrast")
        if self.search_run_seed in self.repeat_seeds:
            raise ValueError("search_run_seed must be disjoint from repeat_seeds")
        if len(self.gate_query_block_refs) != self.ceilings.max_total_nominations:
            raise ValueError("gate query block count must equal the nomination ceiling")
        if self.benchmark_binding_ref.media_type != SEARCH_BENCHMARK_BINDING_MEDIA_TYPE:
            raise ValueError("benchmark_binding_ref declares the wrong media type")
        if self.seed_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("seed_harness_ref declares the wrong media type")
        if self.mutation_policy_fingerprint != canonical_sha256(
            self.contrast.score.mutation_policy
        ):
            raise ValueError("mutation_policy_fingerprint differs from the shared grammar")
        if self.action_capability_fingerprint != canonical_sha256(
            self.contrast.score.action_capability
        ):
            raise ValueError("action_capability_fingerprint differs from shared capability")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class MatchedV2RunManifest(ImmutableModel):
    """One SCORE or FULL arm with all non-treatment values inline and shared."""

    schema_version: Literal["1"] = "1"
    shared: MatchedV2SharedCoordinates
    baseline_kind: BaselineKind
    available_feedback: tuple[FeedbackType, ...]
    paired_proposer_seed: NonNegativeInt
    seed_derivation_domain: Literal[
        "spiral-harness/evolution/manifest-bound-paired-proposer-seed/v1"
    ] = MANIFEST_BOUND_PAIRED_PROPOSER_SEED_DOMAIN

    @field_validator("available_feedback")
    @classmethod
    def _canonicalize_feedback(
        cls,
        values: tuple[FeedbackType, ...],
    ) -> tuple[FeedbackType, ...]:
        ordered = tuple(sorted(values, key=lambda value: value.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("available_feedback must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _bind_condition_profile_and_seed(self) -> Self:
        if self.baseline_kind is BaselineKind.SCORE_ONLY_MATCHED:
            profile = self.shared.contrast.score
        elif self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            profile = self.shared.contrast.full
        else:
            raise ValueError("matched-v2 run kind must be SCORE or FULL")
        if self.available_feedback != profile.available_feedback:
            raise ValueError("run feedback differs from its exact contrast profile")
        expected_seed = derive_manifest_bound_paired_proposer_seed(
            proposal_master_seed=self.shared.proposal_master_seed,
            search_run_seed=self.shared.search_run_seed,
            baseline_kind=self.baseline_kind,
            shared_coordinate_fingerprint=self.shared.fingerprint,
        )
        if self.paired_proposer_seed != expected_seed:
            raise ValueError("paired_proposer_seed differs from manifest-bound derivation")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class MatchedV2AdmissionExpectation(ImmutableModel):
    """Out-of-band expected coordinates fixed independently of produced runs."""

    schema_version: Literal["1"] = "1"
    shared: MatchedV2SharedCoordinates
    shared_coordinate_fingerprint: Sha256
    contrast_fingerprint: Sha256
    expected_paired_proposer_seed: NonNegativeInt
    allowed_treatment_difference: Literal["kind-and-feedback-grant-only"] = (
        "kind-and-feedback-grant-only"
    )

    @model_validator(mode="after")
    def _bind_independent_expectation(self) -> Self:
        if self.shared_coordinate_fingerprint != self.shared.fingerprint:
            raise ValueError("expectation shared fingerprint differs from shared coordinates")
        if self.contrast_fingerprint != self.shared.contrast.fingerprint:
            raise ValueError("expectation contrast fingerprint differs from contrast")
        expected_seed = derive_manifest_bound_paired_proposer_seed(
            proposal_master_seed=self.shared.proposal_master_seed,
            search_run_seed=self.shared.search_run_seed,
            baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
            shared_coordinate_fingerprint=self.shared.fingerprint,
        )
        if self.expected_paired_proposer_seed != expected_seed:
            raise ValueError("expectation paired seed differs from manifest-bound derivation")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class MatchedV2StudyManifest(ImmutableModel):
    """Content-addressed pair awaiting independent admission."""

    schema_version: Literal["1"] = "1"
    score_run_ref: ArtifactRef
    full_run_ref: ArtifactRef
    expectation_fingerprint: Sha256
    shared_coordinate_fingerprint: Sha256
    contrast_fingerprint: Sha256
    allowed_treatment_difference: Literal["kind-and-feedback-grant-only"] = (
        "kind-and-feedback-grant-only"
    )

    @model_validator(mode="after")
    def _run_refs_are_distinct_and_exact(self) -> Self:
        for field_name in ("score_run_ref", "full_run_ref"):
            if getattr(self, field_name).media_type != MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE:
                raise ValueError(f"{field_name} declares the wrong media type")
        if self.score_run_ref.sha256 == self.full_run_ref.sha256:
            raise ValueError("SCORE and FULL run refs must be distinct")
        return self


_ADMISSION_CHECKS = (
    "independent-expectation-equal",
    "benchmark-binding-canonical",
    "seed-harness-canonical",
    "model-runtime-boundary-equal",
    "policy-and-ceilings-equal",
    "kind-and-feedback-only-treatment",
    "paired-seed-recomputed",
    "fresh-gate-block-roster-disjoint",
    "cross-condition-batch-plan-frozen",
)


class MatchedV2AdmissionReport(ImmutableModel):
    """Admission result that deliberately makes no execution claim."""

    schema_version: Literal["1"] = "1"
    study_ref: ArtifactRef
    score_run_ref: ArtifactRef
    full_run_ref: ArtifactRef
    expectation_fingerprint: Sha256
    shared_coordinate_fingerprint: Sha256
    contrast_fingerprint: Sha256
    paired_proposer_seed: NonNegativeInt
    checks: tuple[
        Literal[
            "independent-expectation-equal",
            "benchmark-binding-canonical",
            "seed-harness-canonical",
            "model-runtime-boundary-equal",
            "policy-and-ceilings-equal",
            "kind-and-feedback-only-treatment",
            "paired-seed-recomputed",
            "fresh-gate-block-roster-disjoint",
            "cross-condition-batch-plan-frozen",
        ],
        ...,
    ] = _ADMISSION_CHECKS
    manifest_pair_admitted: Literal[True] = True
    paired_proposer_seed_manifest_bound: Literal[True] = True
    paired_proposer_seed_runtime_attested: Literal[False] = False
    execution_attested: Literal[False] = False
    runtime_topology_matched: Literal[False] = False
    fresh_gate_blocks_runtime_attested: Literal[False] = False
    campaign_validity_attested: Literal[False] = False

    @model_validator(mode="after")
    def _report_shape_is_complete(self) -> Self:
        if self.study_ref.media_type != MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE:
            raise ValueError("study_ref declares the wrong media type")
        if self.checks != _ADMISSION_CHECKS:
            raise ValueError("admission checks must be the complete canonical set")
        return self


def make_matched_v2_run_manifest(
    *,
    shared: MatchedV2SharedCoordinates,
    baseline_kind: BaselineKind,
) -> MatchedV2RunManifest:
    """Construct the sole valid run shape for one member of a matched pair."""

    checked = MatchedV2SharedCoordinates.model_validate(shared, strict=True)
    if baseline_kind is BaselineKind.SCORE_ONLY_MATCHED:
        feedback = checked.contrast.score.available_feedback
    elif baseline_kind is BaselineKind.EVIDENCE_TARGETED:
        feedback = checked.contrast.full.available_feedback
    else:
        raise ValueError("matched-v2 run kind must be SCORE or FULL")
    seed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=checked.proposal_master_seed,
        search_run_seed=checked.search_run_seed,
        baseline_kind=baseline_kind,
        shared_coordinate_fingerprint=checked.fingerprint,
    )
    return MatchedV2RunManifest(
        shared=checked,
        baseline_kind=baseline_kind,
        available_feedback=feedback,
        paired_proposer_seed=seed,
    )


def make_matched_v2_expectation(
    *,
    shared: MatchedV2SharedCoordinates,
) -> MatchedV2AdmissionExpectation:
    """Freeze the independent coordinates before accepting produced manifests."""

    checked = MatchedV2SharedCoordinates.model_validate(shared, strict=True)
    seed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=checked.proposal_master_seed,
        search_run_seed=checked.search_run_seed,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        shared_coordinate_fingerprint=checked.fingerprint,
    )
    return MatchedV2AdmissionExpectation(
        shared=checked,
        shared_coordinate_fingerprint=checked.fingerprint,
        contrast_fingerprint=checked.contrast.fingerprint,
        expected_paired_proposer_seed=seed,
    )


def make_matched_v2_study_manifest(
    *,
    score_run_ref: ArtifactRef,
    full_run_ref: ArtifactRef,
    expectation: MatchedV2AdmissionExpectation,
) -> MatchedV2StudyManifest:
    """Bind persisted run refs to the independently frozen expectation."""

    checked = MatchedV2AdmissionExpectation.model_validate(expectation, strict=True)
    return MatchedV2StudyManifest(
        score_run_ref=score_run_ref,
        full_run_ref=full_run_ref,
        expectation_fingerprint=checked.fingerprint,
        shared_coordinate_fingerprint=checked.shared_coordinate_fingerprint,
        contrast_fingerprint=checked.contrast_fingerprint,
    )


def _load_canonical_model[ModelT: BaseModel](
    store: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
) -> ModelT:
    if ref.media_type != media_type:
        raise MatchedV2AdmissionError(f"{model_type.__name__} declares the wrong media type")
    try:
        payload = store.get_bytes(ref)
        loaded = store.get_json(ref, model_type)
    except Exception as exc:
        raise MatchedV2AdmissionError(
            f"{model_type.__name__} cannot be loaded as canonical content"
        ) from exc
    if payload != canonical_json_bytes(loaded):
        raise MatchedV2AdmissionError(f"{model_type.__name__} is not canonical")
    return loaded


def admit_matched_v2_study(
    store: ArtifactRepository,
    *,
    study_ref: ArtifactRef,
    expectation: MatchedV2AdmissionExpectation,
) -> MatchedV2AdmissionReport:
    """Fail closed unless both manifests equal one independent expected pair."""

    if not isinstance(store, ArtifactRepository):
        raise TypeError("store must implement ArtifactRepository")
    expected = MatchedV2AdmissionExpectation.model_validate(expectation, strict=True)
    study = _load_canonical_model(
        store,
        study_ref,
        MatchedV2StudyManifest,
        MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE,
    )
    if (
        study.expectation_fingerprint != expected.fingerprint
        or study.shared_coordinate_fingerprint != expected.shared_coordinate_fingerprint
        or study.contrast_fingerprint != expected.contrast_fingerprint
    ):
        raise MatchedV2AdmissionError("study differs from the independent expectation")
    score = _load_canonical_model(
        store,
        study.score_run_ref,
        MatchedV2RunManifest,
        MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    )
    full = _load_canonical_model(
        store,
        study.full_run_ref,
        MatchedV2RunManifest,
        MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    )
    if score.baseline_kind is not BaselineKind.SCORE_ONLY_MATCHED:
        raise MatchedV2AdmissionError("score_run_ref does not contain SCORE")
    if full.baseline_kind is not BaselineKind.EVIDENCE_TARGETED:
        raise MatchedV2AdmissionError("full_run_ref does not contain FULL")
    if score.shared != expected.shared or full.shared != expected.shared:
        raise MatchedV2AdmissionError("run shared coordinates differ from expectation")

    score_non_treatment = score.model_dump(
        mode="python",
        exclude={"baseline_kind", "available_feedback"},
        round_trip=True,
        warnings="none",
    )
    full_non_treatment = full.model_dump(
        mode="python",
        exclude={"baseline_kind", "available_feedback"},
        round_trip=True,
        warnings="none",
    )
    if score_non_treatment != full_non_treatment:
        raise MatchedV2AdmissionError("SCORE/FULL differ outside kind and feedback grant")

    benchmark = _load_canonical_model(
        store,
        expected.shared.benchmark_binding_ref,
        SearchBenchmarkBinding,
        SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    seed_harness = _load_canonical_model(
        store,
        expected.shared.seed_harness_ref,
        HarnessManifest,
        HARNESS_MANIFEST_MEDIA_TYPE,
    )
    if (
        seed_harness.model_fingerprint != expected.shared.model_fingerprint
        or seed_harness.runtime_fingerprint != expected.shared.runtime_fingerprint
    ):
        raise MatchedV2AdmissionError("seed harness differs from frozen model/runtime boundary")

    seen_tasks: set[str] = set()
    seen_sources: set[str] = set()
    seen_families: set[str] = set()
    reserved_digests = {
        benchmark.safe_benchmark_metadata_ref.sha256,
        benchmark.exploration_inputs_ref.sha256,
        benchmark.exploration_aggregates_ref.sha256,
        benchmark.exploration_item_feedback_ref.sha256,
        benchmark.exploration_trajectories_ref.sha256,
        benchmark.diagnostic_evidence_ref.sha256,
        *(split.manifest_ref.sha256 for split in benchmark.protocol_splits),
    }
    for query_index, block_ref in enumerate(expected.shared.gate_query_block_refs):
        if block_ref.sha256 in reserved_digests:
            raise MatchedV2AdmissionError("gate query block aliases a reserved benchmark artifact")
        block = _load_canonical_model(
            store,
            block_ref,
            MatchedV2GateQueryBlock,
            MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
        )
        if block.query_index != query_index:
            raise MatchedV2AdmissionError("gate query block order differs from query_index")
        task_ids = {task.task_id for task in block.tasks}
        source_ids = {task.source_id for task in block.tasks}
        family_ids = {task.family_id for task in block.tasks}
        if task_ids.intersection(benchmark.exploration_task_ids):
            raise MatchedV2AdmissionError("gate query block reuses an exploration task")
        if (
            seen_tasks.intersection(task_ids)
            or seen_sources.intersection(source_ids)
            or seen_families.intersection(family_ids)
        ):
            raise MatchedV2AdmissionError("gate query blocks reuse task/source/family identity")
        seen_tasks.update(task_ids)
        seen_sources.update(source_ids)
        seen_families.update(family_ids)

    recomputed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=expected.shared.proposal_master_seed,
        search_run_seed=expected.shared.search_run_seed,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        shared_coordinate_fingerprint=expected.shared.fingerprint,
    )
    full_recomputed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=expected.shared.proposal_master_seed,
        search_run_seed=expected.shared.search_run_seed,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
        shared_coordinate_fingerprint=expected.shared.fingerprint,
    )
    if (
        recomputed != full_recomputed
        or score.paired_proposer_seed != recomputed
        or full.paired_proposer_seed != recomputed
        or expected.expected_paired_proposer_seed != recomputed
    ):
        raise MatchedV2AdmissionError("paired proposer seed does not close over the manifests")

    return MatchedV2AdmissionReport(
        study_ref=study_ref,
        score_run_ref=study.score_run_ref,
        full_run_ref=study.full_run_ref,
        expectation_fingerprint=expected.fingerprint,
        shared_coordinate_fingerprint=expected.shared_coordinate_fingerprint,
        contrast_fingerprint=expected.contrast_fingerprint,
        paired_proposer_seed=recomputed,
    )


__all__ = [
    "MATCHED_V2_ADMISSION_REPORT_MEDIA_TYPE",
    "MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE",
    "MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE",
    "MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE",
    "MatchedV2AdmissionError",
    "MatchedV2AdmissionExpectation",
    "MatchedV2AdmissionReport",
    "MatchedV2ExecutionCeilings",
    "MatchedV2GateQueryBlock",
    "MatchedV2GateTask",
    "MatchedV2PlannedTopology",
    "MatchedV2PolicyBindings",
    "MatchedV2RunManifest",
    "MatchedV2SharedCoordinates",
    "MatchedV2StudyManifest",
    "admit_matched_v2_study",
    "make_matched_v2_expectation",
    "make_matched_v2_run_manifest",
    "make_matched_v2_study_manifest",
]
