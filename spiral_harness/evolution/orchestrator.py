"""Strict trusted admission and automatic orchestration for one search arm/run."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

import spiral_harness.execution.contracts as _contracts
import spiral_harness.execution.receipts as _receipts
from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.lifecycle import CandidateState
from spiral_harness.core.models import (
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessManifest,
    ImmutableModel,
    MutationHypothesis,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.evidence.models import (
    DiagnosticCluster,
    EvidencePacket,
    FailureSignature,
    Trajectory,
    resolve_evidence_span,
)
from spiral_harness.evolution.models import (
    CANDIDATE_SCREEN_MEDIA_TYPE,
    DIAGNOSIS_MEDIA_TYPE,
    GATE_AGGREGATE_VIEW_MEDIA_TYPE,
    NOMINATION_MEDIA_TYPE,
    PROMPT_PROPOSAL_MEDIA_TYPE,
    PROPOSAL_BATCH_MEDIA_TYPE,
    RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE,
    SEARCH_POLICY_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    SEARCH_STOPPING_POLICY_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    CandidateScreen,
    CandidateScreenFailure,
    CandidateScreenStatus,
    DeclineReason,
    Diagnosis,
    GateAggregateMetrics,
    GateAggregateView,
    Nomination,
    PromptProposal,
    ProposalBatch,
    ProposalDecline,
    SearchPolicy,
    SearchRunManifest,
    SearchStoppingPolicy,
    StrategyFeedbackView,
    StrategyPluginManifest,
)
from spiral_harness.evolution.strategies import (
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    RANDOM_VALID_SELECTION_MEDIA_TYPE,
    PromptMutationCatalogue,
    nominate_candidate,
    proposals_from_random_selection,
    sample_random_valid,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.receipts import TrustedExecutionUsage, replay_trusted_usage
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationPhase,
    SchedulePreflightCertificate,
)
from spiral_harness.experiments.admission import (
    CandidateAdmissionError,
    CandidateAdmissionService,
)
from spiral_harness.experiments.baselines import (
    BASELINE_STUDY_PLAN_MEDIA_TYPE,
    BaselineArmPlan,
    BaselineKind,
    BaselineStudyPlan,
)
from spiral_harness.experiments.lifecycle import SelectionClosure
from spiral_harness.experiments.search import (
    AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE,
    SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
    AggregateFeedbackDisclosure,
    SearchController,
    SearchControllerManifest,
    SearchRunSnapshot,
    SearchState,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.models import Decision, GateCheckOutcome, GateConfig

SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.search-run-admission-report.v1+json"
)
STRATEGY_OUTPUT_REJECTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.strategy-output-rejection.v1+json"
)
CANDIDATE_SCREEN_BATCH_MEDIA_TYPE = "application/vnd.spiral-harness.candidate-screen-batch.v1+json"
CANDIDATE_MATERIALIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.candidate-materialization.v1+json"
)
CANDIDATE_MATERIALIZATION_BATCH_MEDIA_TYPE = (
    "application/vnd.spiral-harness.candidate-materialization-batch.v1+json"
)
AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.automatic-search-loop-result.v1+json"
)
SEARCH_ANALYSIS_PLAN_MEDIA_TYPE = "application/vnd.spiral-harness.search-analysis-plan.v1+json"
SEARCH_BENCHMARK_BINDING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.search-benchmark-binding.v1+json"
)
SAFE_BENCHMARK_METADATA_MEDIA_TYPE = (
    "application/vnd.spiral-harness.safe-benchmark-metadata.v1+json"
)
EXPLORATION_INPUTS_MEDIA_TYPE = "application/vnd.spiral-harness.exploration-inputs.v1+json"
EXPLORATION_AGGREGATES_MEDIA_TYPE = "application/vnd.spiral-harness.exploration-aggregates.v1+json"
EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.exploration-item-feedback.v1+json"
)
EXPLORATION_TRAJECTORIES_MEDIA_TYPE = (
    "application/vnd.spiral-harness.exploration-trajectories.v1+json"
)
DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE = "application/vnd.spiral-harness.trajectory.v1+json"
DIAGNOSTIC_GRADER_VERDICT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.diagnostic-grader-verdict.v1+json"
)
TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.trusted-screen-evaluation.v3+json"
)
TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.trusted-objective-aggregate.v3+json"
)
TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.trusted-strategy-feedback.v1+json"
)
DIAGNOSTIC_CLUSTER_MEDIA_TYPE = "application/vnd.spiral-harness.diagnostic-cluster.v1+json"
FAILURE_SIGNATURE_MEDIA_TYPE = "application/vnd.spiral-harness.failure-signature.v1+json"
EVIDENCE_PACKET_MEDIA_TYPE = "application/vnd.spiral-harness.evidence-packet.v1+json"
MUTATION_HYPOTHESIS_MEDIA_TYPE = "application/vnd.spiral-harness.mutation-hypothesis.v1+json"
STRATEGY_ARTIFACT_ACCESS_LOG_MEDIA_TYPE = (
    "application/vnd.spiral-harness.strategy-artifact-access-log.v1+json"
)

_ADMISSION_CHECKS = (
    "external-trust-root-joined",
    "canonical-artifacts-loaded",
    "typed-analysis-and-benchmark-joined",
    "study-plan-fingerprint-joined",
    "arm-run-and-repeat-seeds-joined",
    "experiment-protocol-and-seed-joined",
    "prompt-only-capability-joined",
    "model-inference-runtime-joined",
    "policy-plugin-stopping-fingerprints-joined",
    "gate-confidence-correction-joined",
    "controller-coordinates-and-limits-joined",
    "artifact-closure-and-optimizer-budget-joined",
)

_EXPLORATION_FEEDBACK_REF_FIELDS = (
    "exploration_inputs_ref",
    "exploration_aggregates_ref",
    "exploration_item_feedback_ref",
    "exploration_trajectories_ref",
)

_EXPLORATION_FEEDBACK_REF_MEDIA_TYPES = {
    "exploration_inputs_ref": EXPLORATION_INPUTS_MEDIA_TYPE,
    "exploration_aggregates_ref": EXPLORATION_AGGREGATES_MEDIA_TYPE,
    "exploration_item_feedback_ref": EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    "exploration_trajectories_ref": EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
}


class SearchRunAdmissionError(ValueError):
    """Raised before execution when a frozen search run does not join."""


class AutomaticSearchLoopError(RuntimeError):
    """Raised after fail-closed archival of an invalid runtime output."""

    def __init__(
        self,
        message: str,
        *,
        rejection_ref: ArtifactRef | None = None,
        invalidated_tail_ref: ArtifactRef | None = None,
    ) -> None:
        super().__init__(message)
        self.rejection_ref = rejection_ref
        self.invalidated_tail_ref = invalidated_tail_ref


def _load_canonical_model[ModelT: BaseModel](
    store: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
) -> ModelT:
    """Load one exact typed artifact without trusting caller-declared metadata."""

    if ref.media_type != media_type:
        raise ValueError(f"{model_type.__name__} artifact declares the wrong media type")
    payload = store.get_bytes(ref)
    loaded = store.get_json(ref, model_type)
    if payload != canonical_json_bytes(loaded):
        raise ValueError(f"{model_type.__name__} artifact is not canonical")
    return loaded


def _verify_canonical_json(
    store: ArtifactRepository,
    ref: ArtifactRef,
    *,
    media_type: str | None = None,
) -> None:
    """Verify one exact canonical JSON ref, including media when role-bound."""

    if media_type is not None and ref.media_type != media_type:
        raise ValueError("JSON artifact declares the wrong media type")
    payload = store.get_bytes(ref)
    value = store.get_json(ref)
    if payload != canonical_json_bytes(value):
        raise ValueError("artifact is not canonical JSON")


def _load_exact_diagnostic_closure(
    store: ArtifactRepository,
    cluster_ref: ArtifactRef,
    *,
    exploration_task_ids: tuple[str, ...] | None = None,
    allowed_trajectory_refs: frozenset[ArtifactRef] | None = None,
) -> tuple[DiagnosticCluster, frozenset[ArtifactRef]]:
    """Strictly traverse the complete diagnostic graph rooted at ``cluster_ref``."""

    refs_by_digest: dict[str, ArtifactRef] = {}
    roles_by_digest: dict[str, set[str]] = {}
    exclusive_document_roles = {
        "diagnostic-cluster",
        "failure-signature",
        "evidence-packet",
        "trajectory",
        "grader-verdict",
    }

    def register(ref: ArtifactRef, *, role: str) -> None:
        checked = ArtifactRef.model_validate(ref, strict=True)
        prior_ref = refs_by_digest.get(checked.sha256)
        prior_roles = roles_by_digest.get(checked.sha256, set())
        if prior_ref is not None and prior_ref != checked:
            raise ValueError("diagnostic closure contains conflicting reference metadata")
        if (
            prior_roles
            and role not in prior_roles
            and (
                role in exclusive_document_roles
                or bool(prior_roles.intersection(exclusive_document_roles))
            )
        ):
            raise ValueError("diagnostic closure reuses one digest across semantic roles")
        refs_by_digest[checked.sha256] = checked
        roles_by_digest.setdefault(checked.sha256, set()).add(role)

    register(cluster_ref, role="diagnostic-cluster")
    cluster = _load_canonical_model(
        store,
        cluster_ref,
        DiagnosticCluster,
        DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    )
    exploration_tasks = (
        frozenset(exploration_task_ids) if exploration_task_ids is not None else None
    )
    if exploration_tasks is not None and not set(cluster.task_ids).issubset(exploration_tasks):
        raise ValueError("diagnostic cluster contains a non-exploration task")

    for signature_ref in cluster.failure_signature_refs:
        register(signature_ref, role="failure-signature")
        signature = _load_canonical_model(
            store,
            signature_ref,
            FailureSignature,
            FAILURE_SIGNATURE_MEDIA_TYPE,
        )
        if exploration_tasks is not None and not set(signature.affected_task_ids).issubset(
            exploration_tasks
        ):
            raise ValueError("failure signature contains a non-exploration task")
        for component_ref in signature.affected_component_refs:
            register(component_ref, role="affected-component")
            store.get_bytes(component_ref)
    for packet_ref in cluster.evidence_packet_refs:
        register(packet_ref, role="evidence-packet")
        packet = _load_canonical_model(
            store,
            packet_ref,
            EvidencePacket,
            EVIDENCE_PACKET_MEDIA_TYPE,
        )
        for span in (*packet.source_spans, *packet.positive_anchors):
            register(span.trajectory_ref, role="trajectory")
            if (
                allowed_trajectory_refs is not None
                and span.trajectory_ref not in allowed_trajectory_refs
            ):
                raise ValueError("diagnostic span cites a trajectory outside the frozen index")
            trajectory = _load_canonical_model(
                store,
                span.trajectory_ref,
                Trajectory,
                DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE,
            )
            if exploration_tasks is not None and trajectory.task_id not in exploration_tasks:
                raise ValueError("diagnostic trajectory belongs to a non-exploration task")
            resolve_evidence_span(span, trajectory, span.trajectory_ref)
            register(trajectory.harness_ref, role="trajectory-harness")
            store.get_bytes(trajectory.harness_ref)
            for event in trajectory.events:
                register(event.payload_ref, role="trajectory-event-payload")
                store.get_bytes(event.payload_ref)
                if event.component_ref is not None:
                    register(event.component_ref, role="trajectory-event-component")
                    store.get_bytes(event.component_ref)
        if packet.failure_signature_ref is not None:
            if packet.failure_signature_ref not in cluster.failure_signature_refs:
                raise ValueError("diagnostic packet cites a signature outside its cluster")
            register(packet.failure_signature_ref, role="failure-signature")
            _load_canonical_model(
                store,
                packet.failure_signature_ref,
                FailureSignature,
                FAILURE_SIGNATURE_MEDIA_TYPE,
            )
        if packet.grader_verdict_ref is not None:
            register(packet.grader_verdict_ref, role="grader-verdict")
            _verify_canonical_json(
                store,
                packet.grader_verdict_ref,
                media_type=DIAGNOSTIC_GRADER_VERDICT_MEDIA_TYPE,
            )
    return cluster, frozenset(refs_by_digest.values())


class SearchRunAdmissionExpectation(ImmutableModel):
    """Caller-owned trust root that a self-consistent foreign run cannot choose."""

    schema_version: Literal["1"] = "1"
    baseline_study_plan_ref: ArtifactRef
    experiment_ref: ArtifactRef
    baseline_kind: BaselineKind
    search_run_seed: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def _refs_have_exact_manifest_media_types(self) -> Self:
        if self.baseline_study_plan_ref.media_type != BASELINE_STUDY_PLAN_MEDIA_TYPE:
            raise ValueError("baseline_study_plan_ref declares the wrong media type")
        if self.experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise ValueError("experiment_ref declares the wrong media type")
        return self


class SearchAnalysisPlan(ImmutableModel):
    """Typed preregistration joined to both the study schedule and run policy."""

    schema_version: Literal["1"] = "1"
    objective: Literal["benchmark-score"] = "benchmark-score"
    selector: Literal["exploration-lcb-v1"] = "exploration-lcb-v1"
    family_alpha: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    max_gate_queries: Annotated[int, Field(gt=0, strict=True)]
    gate_confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    search_run_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=2),
    ]
    repeat_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...],
        Field(min_length=2),
    ]
    baseline_kinds: tuple[BaselineKind, ...] = tuple(BaselineKind)

    @field_validator("search_run_seeds", "repeat_seeds")
    @classmethod
    def _canonicalize_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("analysis seed schedules must not contain duplicates")
        return ordered

    @field_validator("baseline_kinds")
    @classmethod
    def _require_four_baselines(
        cls,
        value: tuple[BaselineKind, ...],
    ) -> tuple[BaselineKind, ...]:
        ordered = tuple(sorted(value, key=lambda kind: kind.value))
        if frozenset(ordered) != frozenset(BaselineKind) or len(ordered) != len(BaselineKind):
            raise ValueError("analysis plan must name exactly the four frozen baselines")
        return ordered

    @model_validator(mode="after")
    def _confidence_is_familywise_corrected(self) -> Self:
        if set(self.search_run_seeds).intersection(self.repeat_seeds):
            raise ValueError("analysis search and repeat seed schedules must be disjoint")
        expected = 1.0 - (self.family_alpha / self.max_gate_queries)
        if self.gate_confidence_level != expected:
            raise ValueError("analysis gate confidence must use the frozen correction")
        return self


class SafeBenchmarkMetadata(ImmutableModel):
    """Optimizer-readable benchmark metadata with no partition references."""

    schema_version: Literal["1"] = "1"
    benchmark_fingerprint: NonEmptyStr
    objective: Literal["benchmark-score"] = "benchmark-score"
    exploration_task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    redaction_schema: Literal["exploration-only-no-partition-refs-v1"] = (
        "exploration-only-no-partition-refs-v1"
    )

    @field_validator("exploration_task_ids")
    @classmethod
    def _canonicalize_exploration_tasks(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("safe benchmark exploration_task_ids must not contain duplicates")
        return ordered


class ExplorationTrajectoryIndex(ImmutableModel):
    """Exact exploration trajectory membership disclosed to the optimizer."""

    schema_version: Literal["1"] = "1"
    exploration_task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    trajectory_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]

    @field_validator("exploration_task_ids")
    @classmethod
    def _canonicalize_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("trajectory index task ids must not contain duplicates")
        return ordered

    @field_validator("trajectory_refs")
    @classmethod
    def _canonicalize_trajectory_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("trajectory index refs must have distinct artifact digests")
        if any(ref.media_type != DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE for ref in ordered):
            raise ValueError("trajectory index ref declares the wrong media type")
        return ordered


class SearchBenchmarkBinding(ImmutableModel):
    """Trusted binding for splits and the exact optimizer-readable closure."""

    schema_version: Literal["1"] = "1"
    benchmark_fingerprint: NonEmptyStr
    objective_aggregate_attestor_id: Sha256
    strategy_feedback_attestor_id: Sha256
    protocol_splits: Annotated[tuple[ProtocolSplit, ...], Field(min_length=2, max_length=3)]
    exploration_task_ids: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    safe_benchmark_metadata_ref: ArtifactRef
    exploration_inputs_ref: ArtifactRef
    exploration_aggregates_ref: ArtifactRef
    exploration_item_feedback_ref: ArtifactRef
    exploration_trajectories_ref: ArtifactRef
    diagnostic_evidence_ref: ArtifactRef
    diagnostic_closure_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=4)]

    @field_validator("exploration_task_ids")
    @classmethod
    def _canonicalize_exploration_tasks(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        ordered = tuple(sorted(value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("exploration_task_ids must not contain duplicates")
        return ordered

    @field_validator("protocol_splits")
    @classmethod
    def _canonicalize_splits(
        cls,
        value: tuple[ProtocolSplit, ...],
    ) -> tuple[ProtocolSplit, ...]:
        ordered = tuple(sorted(value, key=lambda split: split.partition.value))
        partitions = tuple(split.partition for split in ordered)
        if len(partitions) != len(set(partitions)):
            raise ValueError("benchmark binding contains duplicate protocol partitions")
        required = {ProtocolPartition.EXPLORATION, ProtocolPartition.GATE}
        if not required.issubset(partitions):
            raise ValueError("benchmark binding must contain exploration and gate splits")
        return ordered

    @field_validator("diagnostic_closure_refs")
    @classmethod
    def _canonicalize_diagnostic_closure(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("diagnostic closure must not contain duplicate artifact digests")
        return ordered

    @model_validator(mode="after")
    def _optimizer_refs_are_exact_and_disjoint(self) -> Self:
        role_ref_media_types = (
            (self.exploration_inputs_ref, EXPLORATION_INPUTS_MEDIA_TYPE),
            (self.exploration_aggregates_ref, EXPLORATION_AGGREGATES_MEDIA_TYPE),
            (self.exploration_item_feedback_ref, EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE),
            (self.exploration_trajectories_ref, EXPLORATION_TRAJECTORIES_MEDIA_TYPE),
        )
        for ref, expected_media_type in role_ref_media_types:
            if ref.media_type != expected_media_type:
                raise ValueError("exploration role ref declares the wrong media type")
        role_refs = tuple(ref for ref, _ in role_ref_media_types)
        role_digests = {ref.sha256 for ref in role_refs}
        if len(role_digests) != len(role_refs):
            raise ValueError("exploration role refs must have distinct artifact digests")
        if self.safe_benchmark_metadata_ref.media_type != SAFE_BENCHMARK_METADATA_MEDIA_TYPE:
            raise ValueError("safe benchmark metadata ref declares the wrong media type")
        if self.diagnostic_evidence_ref.media_type != DIAGNOSTIC_CLUSTER_MEDIA_TYPE:
            raise ValueError("diagnostic evidence ref declares the wrong media type")
        if self.diagnostic_evidence_ref not in self.diagnostic_closure_refs:
            raise ValueError("diagnostic closure must contain its root cluster ref")

        split_digests = {split.manifest_ref.sha256 for split in self.protocol_splits}
        if len(split_digests) != len(self.protocol_splits):
            raise ValueError("protocol split refs must have distinct artifact digests")
        if role_digests.intersection(split_digests):
            raise ValueError("exploration role refs must not alias protocol split artifacts")

        readable_refs = (
            self.safe_benchmark_metadata_ref,
            *role_refs,
            *self.diagnostic_closure_refs,
        )
        readable_digests = tuple(ref.sha256 for ref in readable_refs)
        if len(readable_digests) != len(set(readable_digests)):
            raise ValueError("optimizer-readable frozen refs must have distinct digests")
        if split_digests.intersection(readable_digests):
            raise ValueError("optimizer-readable refs must not alias protocol split artifacts")
        return self

    @property
    def exploration_split_ref(self) -> ArtifactRef:
        return next(
            split.manifest_ref
            for split in self.protocol_splits
            if split.partition is ProtocolPartition.EXPLORATION
        )


def _load_exploration_trajectory_index(
    store: ArtifactRepository,
    benchmark: SearchBenchmarkBinding,
) -> ExplorationTrajectoryIndex:
    """Load every frozen exploration trajectory without conflating it with diagnostics."""

    index = _load_canonical_model(
        store,
        benchmark.exploration_trajectories_ref,
        ExplorationTrajectoryIndex,
        EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    )
    if index.exploration_task_ids != benchmark.exploration_task_ids:
        raise ValueError("exploration trajectory index differs from the frozen task set")
    reserved_digests = {
        benchmark.safe_benchmark_metadata_ref.sha256,
        benchmark.exploration_inputs_ref.sha256,
        benchmark.exploration_aggregates_ref.sha256,
        benchmark.exploration_item_feedback_ref.sha256,
        benchmark.exploration_trajectories_ref.sha256,
        *(split.manifest_ref.sha256 for split in benchmark.protocol_splits),
    }
    for trajectory_ref in index.trajectory_refs:
        if trajectory_ref.sha256 in reserved_digests:
            raise ValueError("exploration trajectory aliases a reserved benchmark artifact")
        trajectory = _load_canonical_model(
            store,
            trajectory_ref,
            Trajectory,
            DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE,
        )
        if trajectory.task_id not in benchmark.exploration_task_ids:
            raise ValueError("exploration trajectory belongs to a non-exploration task")
        store.get_bytes(trajectory.harness_ref)
        for event in trajectory.events:
            store.get_bytes(event.payload_ref)
            if event.component_ref is not None:
                store.get_bytes(event.component_ref)
    return index


class RunBoundStrategyFeedback(ImmutableModel):
    """Current-run envelope preventing a valid feedback view from being replayed."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    experiment_ref: ArtifactRef
    baseline_kind: BaselineKind
    search_run_seed: Annotated[int, Field(ge=0, strict=True)]
    round_index: Annotated[int, Field(ge=0, strict=True)]
    champion_harness_ref: ArtifactRef
    trusted_feedback_ref: ArtifactRef
    view: StrategyFeedbackView

    @model_validator(mode="after")
    def _exact_reference_media_types(self) -> Self:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise ValueError("experiment_ref declares the wrong media type")
        if self.trusted_feedback_ref.media_type != TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE:
            raise ValueError("trusted_feedback_ref declares the wrong media type")
        if self.view.baseline_kind is not self.baseline_kind:
            raise ValueError("feedback view belongs to another baseline")
        return self


class TrustedStrategyFeedbackContent(ImmutableModel):
    """Run/round-bound feedback issued by the independent feedback authority."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    experiment_ref: ArtifactRef
    benchmark_binding_ref: ArtifactRef
    exploration_split_ref: ArtifactRef
    baseline_kind: BaselineKind
    search_run_seed: Annotated[int, Field(ge=0, strict=True)]
    round_index: Annotated[int, Field(ge=0, strict=True)]
    champion_harness_ref: ArtifactRef
    prior_gate_aggregate: GateAggregateView | None
    view: StrategyFeedbackView

    @model_validator(mode="after")
    def _exact_run_and_view_types(self) -> Self:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise ValueError("experiment_ref declares the wrong media type")
        if self.benchmark_binding_ref.media_type != SEARCH_BENCHMARK_BINDING_MEDIA_TYPE:
            raise ValueError("benchmark_binding_ref declares the wrong media type")
        exploration_media_type = self.exploration_split_ref.media_type.partition(";")[0].lower()
        if exploration_media_type != "application/json" and not exploration_media_type.endswith(
            "+json"
        ):
            raise ValueError("exploration_split_ref must declare JSON")
        if self.view.baseline_kind is not self.baseline_kind:
            raise ValueError("trusted feedback view belongs to another baseline")
        if self.view.gate_aggregate != self.prior_gate_aggregate:
            raise ValueError("trusted feedback gate view differs from its disclosed aggregate")
        return self


class TrustedStrategyFeedback(ImmutableModel):
    """HMAC-attested feedback bundle; arbitrary runtime refs are not authority."""

    schema_version: Literal["1"] = "1"
    content: TrustedStrategyFeedbackContent
    attestor_id: Sha256
    authentication_tag: Sha256


class StrategyFeedbackVerificationCapability:
    """Exact verify-only capability for one frozen feedback producer."""

    __slots__ = ("__attestor_id", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("strategy feedback verification capability cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        if type(secret) is not bytes or len(secret) < 32:
            raise ValueError("strategy feedback attestor secret must contain at least 32 bytes")
        self.__store = store
        self.__secret = secret
        self.__attestor_id = sha256_bytes(
            b"spiral-harness/strategy-feedback-attestor/v1\x00" + secret
        )

    @property
    def attestor_id(self) -> str:
        return self.__attestor_id

    def verify(self, feedback_ref: ArtifactRef) -> TrustedStrategyFeedbackContent:
        if feedback_ref.media_type != TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE:
            raise AutomaticSearchLoopError("trusted feedback declares the wrong media type")
        payload = self.__store.get_bytes(feedback_ref)
        envelope = self.__store.get_json(feedback_ref, TrustedStrategyFeedback)
        if payload != canonical_json_bytes(envelope):
            raise AutomaticSearchLoopError("trusted feedback artifact is not canonical")
        if envelope.attestor_id != self.__attestor_id:
            raise AutomaticSearchLoopError("trusted feedback uses another attestor")
        expected = hmac.new(
            self.__secret,
            b"spiral-harness/strategy-feedback/v1\x00" + canonical_json_bytes(envelope.content),
            sha256,
        ).hexdigest()
        if not hmac.compare_digest(envelope.authentication_tag, expected):
            raise AutomaticSearchLoopError("trusted feedback authentication failed")
        return envelope.content


class TrustedStrategyFeedbackService:
    """Trusted producer kept outside the optimizer and general runtime boundary."""

    __slots__ = ("__capability", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("trusted strategy feedback service cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        self.__store = store
        self.__secret = secret
        self.__capability = StrategyFeedbackVerificationCapability(store, secret=secret)

    @property
    def verification_capability(self) -> StrategyFeedbackVerificationCapability:
        return self.__capability

    def attest(self, content: TrustedStrategyFeedbackContent) -> ArtifactRef:
        checked = TrustedStrategyFeedbackContent.model_validate(
            content.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        self._validate_frozen_content(checked)
        envelope = TrustedStrategyFeedback(
            content=checked,
            attestor_id=self.__capability.attestor_id,
            authentication_tag=hmac.new(
                self.__secret,
                b"spiral-harness/strategy-feedback/v1\x00" + canonical_json_bytes(checked),
                sha256,
            ).hexdigest(),
        )
        return self.__store.put_json(
            envelope,
            media_type=TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE,
        )

    def _validate_frozen_content(self, content: TrustedStrategyFeedbackContent) -> None:
        """Authorize only the exact exploration-only closure frozen in the binding."""

        benchmark = _load_canonical_model(
            self.__store,
            content.benchmark_binding_ref,
            SearchBenchmarkBinding,
            SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
        )
        if content.exploration_split_ref != benchmark.exploration_split_ref:
            raise ValueError("trusted feedback uses another exploration split")

        metadata = _load_canonical_model(
            self.__store,
            benchmark.safe_benchmark_metadata_ref,
            SafeBenchmarkMetadata,
            SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
        )
        if (
            metadata.benchmark_fingerprint != benchmark.benchmark_fingerprint
            or metadata.exploration_task_ids != benchmark.exploration_task_ids
        ):
            raise ValueError("safe benchmark metadata differs from its frozen binding")
        if content.view.benchmark_metadata_ref != benchmark.safe_benchmark_metadata_ref:
            raise ValueError("feedback does not use the frozen safe benchmark metadata")

        for field_name, media_type in _EXPLORATION_FEEDBACK_REF_MEDIA_TYPES.items():
            frozen_ref = getattr(benchmark, field_name)
            _verify_canonical_json(self.__store, frozen_ref, media_type=media_type)
            disclosed_ref = getattr(content.view, field_name)
            if disclosed_ref is not None and disclosed_ref != frozen_ref:
                raise ValueError(f"feedback {field_name} differs from the frozen role ref")

        trajectory_index = _load_exploration_trajectory_index(self.__store, benchmark)

        _, actual_closure = _load_exact_diagnostic_closure(
            self.__store,
            benchmark.diagnostic_evidence_ref,
            exploration_task_ids=benchmark.exploration_task_ids,
            allowed_trajectory_refs=frozenset(trajectory_index.trajectory_refs),
        )
        if actual_closure != frozenset(benchmark.diagnostic_closure_refs):
            raise ValueError("diagnostic evidence differs from the frozen exact closure")
        if (
            content.view.diagnostic_evidence_ref is not None
            and content.view.diagnostic_evidence_ref != benchmark.diagnostic_evidence_ref
        ):
            raise ValueError("feedback diagnostic root differs from the frozen binding")


class TrustedObjectiveAggregateContent(ImmutableModel):
    """Trusted grader authorization for screen scores over one receipt batch."""

    schema_version: Literal["3"] = "3"
    search_run_ref: ArtifactRef
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    benchmark_binding_ref: ArtifactRef
    grader_fingerprint: NonEmptyStr
    schedule_fingerprint: Sha256
    receipt_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    primary_score: Annotated[float, Field(allow_inf_nan=False)]
    mean_delta: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @field_validator("receipt_refs")
    @classmethod
    def _canonicalize_objective_receipts(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if {ref.media_type for ref in ordered} != {_receipts.EXECUTION_RECEIPT_MEDIA_TYPE}:
            raise ValueError("objective receipt refs must be execution receipts")
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("objective receipt refs must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _objective_refs_have_exact_media_types(self) -> Self:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        if self.benchmark_binding_ref.media_type != SEARCH_BENCHMARK_BINDING_MEDIA_TYPE:
            raise ValueError("benchmark_binding_ref declares the wrong media type")
        return self


class TrustedObjectiveAggregate(ImmutableModel):
    """HMAC-attested score aggregate issued by the independent grader plane."""

    schema_version: Literal["3"] = "3"
    content: TrustedObjectiveAggregateContent
    attestor_id: Sha256
    authentication_tag: Sha256


class ObjectiveAggregateVerificationCapability:
    """Exact concrete, verify-only capability for independent score attestations."""

    __slots__ = ("__attestor_id", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover - definition guard
        raise TypeError("objective aggregate verification capability cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        if type(secret) is not bytes or len(secret) < 32:
            raise ValueError("objective aggregate attestor secret must contain at least 32 bytes")
        self.__store = store
        self.__secret = secret
        attestor_domain = b"spiral-harness/objective-aggregate-attestor/v3\x00"
        self.__attestor_id = sha256_bytes(attestor_domain + secret)

    @property
    def attestor_id(self) -> str:
        return self.__attestor_id

    def verify(self, aggregate_ref: ArtifactRef) -> TrustedObjectiveAggregateContent:
        if aggregate_ref.media_type != TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE:
            raise AutomaticSearchLoopError("objective aggregate declares the wrong media type")
        payload = self.__store.get_bytes(aggregate_ref)
        aggregate = self.__store.get_json(aggregate_ref, TrustedObjectiveAggregate)
        if payload != canonical_json_bytes(aggregate):
            raise AutomaticSearchLoopError("objective aggregate artifact is not canonical")
        if aggregate.attestor_id != self.__attestor_id:
            raise AutomaticSearchLoopError("objective aggregate uses another attestor")
        expected = hmac.new(self.__secret, b"spiral-harness/objective-aggregate/v3\x00", sha256)
        expected.update(self.__attestor_id.encode("ascii") + b"\x00")
        expected.update(canonical_json_bytes(aggregate.content))
        if not hmac.compare_digest(aggregate.authentication_tag, expected.hexdigest()):
            raise AutomaticSearchLoopError("objective aggregate authentication failed")
        return aggregate.content


class TrustedObjectiveAggregateService:
    """Trusted setup authority kept outside the general search runtime."""

    __slots__ = ("__capability", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover - definition guard
        raise TypeError("objective aggregate service cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        self.__store = store
        self.__secret = secret
        self.__capability = ObjectiveAggregateVerificationCapability(store, secret=secret)

    @property
    def verification_capability(self) -> ObjectiveAggregateVerificationCapability:
        return self.__capability

    def attest(self, content: TrustedObjectiveAggregateContent) -> ArtifactRef:
        checked = TrustedObjectiveAggregateContent.model_validate(
            content.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        authentication = hmac.new(
            self.__secret, b"spiral-harness/objective-aggregate/v3\x00", sha256
        )
        authentication.update(self.__capability.attestor_id.encode("ascii") + b"\x00")
        authentication.update(canonical_json_bytes(checked))
        aggregate = TrustedObjectiveAggregate(
            content=checked,
            attestor_id=self.__capability.attestor_id,
            authentication_tag=authentication.hexdigest(),
        )
        return self.__store.put_json(aggregate, media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE)


class TrustedScreenEvaluation(ImmutableModel):
    """Receipt-replay proof and trusted aggregate used by a candidate screen."""

    schema_version: Literal["3"] = "3"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    schedule: EvaluationBatchSchedule
    preflight_ref: ArtifactRef
    receipt_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    final_ledger_tail_ref: ArtifactRef
    trusted_usage: TrustedExecutionUsage
    objective_aggregate_ref: ArtifactRef
    primary_score: Annotated[float, Field(allow_inf_nan=False)]
    mean_delta: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @field_validator("receipt_refs")
    @classmethod
    def _canonicalize_receipts(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if {ref.media_type for ref in ordered} != {_receipts.EXECUTION_RECEIPT_MEDIA_TYPE}:
            raise ValueError("receipt_refs must be execution receipts")
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("receipt_refs must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _declared_execution_bindings_are_exact(self) -> Self:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        if self.preflight_ref.media_type != SCHEDULE_PREFLIGHT_MEDIA_TYPE:
            raise ValueError("preflight_ref declares the wrong media type")
        if self.objective_aggregate_ref.media_type != TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE:
            raise ValueError("objective_aggregate_ref declares the wrong media type")
        if self.final_ledger_tail_ref.media_type != _contracts.ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("final_ledger_tail_ref declares the wrong media type")
        if self.trusted_usage.schedule_fingerprint != self.schedule.fingerprint:
            raise ValueError("trusted usage belongs to another schedule")
        canonical_usage_receipts = tuple(
            sorted(
                self.trusted_usage.receipt_refs,
                key=lambda ref: (ref.sha256, ref.size, ref.media_type),
            )
        )
        if canonical_usage_receipts != self.receipt_refs:
            raise ValueError("trusted usage does not bind the declared receipt set")
        if self.trusted_usage.ledger_tail_refs != (self.final_ledger_tail_ref,):
            raise ValueError("trusted usage does not bind the final ledger tail")
        return self


class StrategyArtifactOperation(StrEnum):
    READ_JSON = "read-json"
    READ_TEXT = "read-text"
    WRITE_PROMPT = "write-prompt"
    WRITE_HYPOTHESIS = "write-hypothesis"


class StrategyArtifactAccess(ImmutableModel):
    sequence: Annotated[int, Field(ge=0, strict=True)]
    operation: StrategyArtifactOperation
    artifact_ref: ArtifactRef


class StrategyArtifactAccessLog(ImmutableModel):
    """Controller-derived record of every scoped optimizer read and write."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    invocation: Literal["diagnosis", "proposal"]
    accesses: tuple[StrategyArtifactAccess, ...]
    read_bytes: Annotated[int, Field(ge=0, strict=True)]
    written_bytes: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def _access_sequence_is_contiguous(self) -> Self:
        if tuple(access.sequence for access in self.accesses) != tuple(range(len(self.accesses))):
            raise ValueError("strategy artifact access sequence must be contiguous")
        read_operations = {
            StrategyArtifactOperation.READ_JSON,
            StrategyArtifactOperation.READ_TEXT,
        }
        expected_read = sum(
            access.artifact_ref.size
            for access in self.accesses
            if access.operation in read_operations
        )
        expected_written = sum(
            access.artifact_ref.size
            for access in self.accesses
            if access.operation not in read_operations
        )
        if (self.read_bytes, self.written_bytes) != (expected_read, expected_written):
            raise ValueError("strategy artifact access byte totals are inconsistent")
        return self


class StrategyArtifactView:
    """No-list, allowlisted optimizer storage capability for one open round."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        search_run_ref: ArtifactRef,
        baseline_kind: BaselineKind,
        round_index: int,
        invocation: Literal["diagnosis", "proposal"],
        allowed_read_refs: tuple[ArtifactRef, ...],
        max_prompt_bytes: int,
    ) -> None:
        self.__store = store
        self.__search_run_ref = search_run_ref
        self.__baseline_kind = baseline_kind
        self.__round_index = round_index
        self.__invocation = invocation
        self.__allowed_reads: dict[str, ArtifactRef] = {}
        for ref in allowed_read_refs:
            self.__allow_read(ref)
        self.__max_prompt_bytes = max_prompt_bytes
        self.__accesses: list[StrategyArtifactAccess] = []
        self.__read_refs: set[ArtifactRef] = set()
        self.__written_prompt_refs: set[ArtifactRef] = set()
        self.__written_hypothesis_refs: set[ArtifactRef] = set()

    @property
    def read_refs(self) -> frozenset[ArtifactRef]:
        return frozenset(self.__read_refs)

    @property
    def written_prompt_refs(self) -> frozenset[ArtifactRef]:
        return frozenset(self.__written_prompt_refs)

    @property
    def written_hypothesis_refs(self) -> frozenset[ArtifactRef]:
        return frozenset(self.__written_hypothesis_refs)

    def read_json(self, ref: ArtifactRef) -> object:
        checked = self.__require_allowed(ref)
        value = self.__store.get_json(checked)
        if self.__store.get_bytes(checked) != canonical_json_bytes(value):
            raise AutomaticSearchLoopError("allowlisted JSON artifact is not canonical")
        self.__record(StrategyArtifactOperation.READ_JSON, checked)
        self.__read_refs.add(checked)
        return value

    def read_text(self, ref: ArtifactRef) -> str:
        checked = self.__require_allowed(ref)
        if not checked.media_type.partition(";")[0].strip().lower().startswith("text/"):
            raise AutomaticSearchLoopError("allowlisted artifact is not text")
        payload = self.__store.get_bytes(checked)
        try:
            value = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AutomaticSearchLoopError("prompt artifact is not UTF-8") from exc
        self.__record(StrategyArtifactOperation.READ_TEXT, checked)
        self.__read_refs.add(checked)
        return value

    def put_prompt(self, value: str) -> ArtifactRef:
        if self.__invocation != "proposal":
            raise AutomaticSearchLoopError("diagnosis artifact view cannot write prompts")
        if not isinstance(value, str):
            raise TypeError("prompt output must be a string")
        payload = value.encode("utf-8")
        if not payload:
            raise ValueError("prompt output must not be empty")
        if len(payload) > self.__max_prompt_bytes:
            raise ValueError("prompt output exceeds the frozen mutation size limit")
        ref = self.__store.put_bytes(payload, media_type="text/plain")
        self.__record(StrategyArtifactOperation.WRITE_PROMPT, ref)
        self.__written_prompt_refs.add(ref)
        self.__allow_read(ref)
        return ref

    def put_hypothesis(self, value: MutationHypothesis) -> ArtifactRef:
        if self.__invocation != "proposal":
            raise AutomaticSearchLoopError("diagnosis artifact view cannot write hypotheses")
        if not isinstance(value, MutationHypothesis):
            raise TypeError("hypothesis output must be a MutationHypothesis")
        checked = MutationHypothesis.model_validate(
            value.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        payload = canonical_json_bytes(checked)
        if len(payload) > self.__max_prompt_bytes:
            raise ValueError("hypothesis output exceeds the frozen artifact size limit")
        if not set(checked.evidence_refs).issubset(self.__read_refs):
            raise ValueError("hypothesis evidence must have been read through this round view")
        ref = self.__store.put_json(checked, media_type=MUTATION_HYPOTHESIS_MEDIA_TYPE)
        self.__record(StrategyArtifactOperation.WRITE_HYPOTHESIS, ref)
        self.__written_hypothesis_refs.add(ref)
        self.__allow_read(ref)
        return ref

    def access_log(self) -> StrategyArtifactAccessLog:
        return StrategyArtifactAccessLog(
            search_run_ref=self.__search_run_ref,
            baseline_kind=self.__baseline_kind,
            round_index=self.__round_index,
            invocation=self.__invocation,
            accesses=tuple(self.__accesses),
            read_bytes=sum(
                access.artifact_ref.size
                for access in self.__accesses
                if access.operation
                in {
                    StrategyArtifactOperation.READ_JSON,
                    StrategyArtifactOperation.READ_TEXT,
                }
            ),
            written_bytes=sum(
                access.artifact_ref.size
                for access in self.__accesses
                if access.operation
                in {
                    StrategyArtifactOperation.WRITE_PROMPT,
                    StrategyArtifactOperation.WRITE_HYPOTHESIS,
                }
            ),
        )

    def __require_allowed(self, ref: ArtifactRef) -> ArtifactRef:
        try:
            checked = ArtifactRef.model_validate(ref, strict=True)
        except Exception as exc:
            raise AutomaticSearchLoopError("strategy artifact ref is malformed") from exc
        if self.__allowed_reads.get(checked.sha256) != checked:
            raise AutomaticSearchLoopError("strategy attempted a non-allowlisted artifact read")
        return checked

    def __allow_read(self, ref: ArtifactRef) -> None:
        checked = ArtifactRef.model_validate(ref, strict=True)
        prior = self.__allowed_reads.get(checked.sha256)
        if prior is not None and prior != checked:
            raise AutomaticSearchLoopError(
                "strategy allowlist contains conflicting reference metadata"
            )
        self.__allowed_reads[checked.sha256] = checked

    def __record(
        self,
        operation: StrategyArtifactOperation,
        ref: ArtifactRef,
    ) -> None:
        self.__accesses.append(
            StrategyArtifactAccess(
                sequence=len(self.__accesses),
                operation=operation,
                artifact_ref=ref,
            )
        )


class SearchRunAdmissionReport(ImmutableModel):
    """Typed proof that one controller can execute one frozen arm/run."""

    schema_version: Literal["1"] = "1"
    admitted: Literal[True] = True
    search_run_ref: ArtifactRef
    controller_manifest_ref: ArtifactRef
    baseline_study_plan_ref: ArtifactRef
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    analysis_plan_ref: ArtifactRef
    benchmark_binding_ref: ArtifactRef
    objective_aggregate_attestor_id: Sha256
    strategy_feedback_attestor_id: Sha256
    search_policy_ref: ArtifactRef
    stopping_policy_ref: ArtifactRef
    strategy_plugin_ref: ArtifactRef
    strategy_implementation_ref: ArtifactRef
    seed_harness_ref: ArtifactRef
    baseline_kind: BaselineKind
    search_run_seed: Annotated[int, Field(ge=0, strict=True)]
    repeat_seeds: Annotated[
        tuple[Annotated[int, Field(ge=0, strict=True)], ...], Field(min_length=2)
    ]
    gate_confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None
    max_optimizer_model_calls_per_run: Annotated[int, Field(ge=0, strict=True)]
    checks: tuple[NonEmptyStr, ...] = _ADMISSION_CHECKS

    @field_validator("checks")
    @classmethod
    def _checks_are_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != _ADMISSION_CHECKS:
            raise ValueError("admission checks must be the complete canonical set")
        return value


class SearchRunAdmissionService:
    """Strictly replay and join every artifact that authorizes a search run."""

    def __init__(self, store: ArtifactRepository) -> None:
        self.store = store

    def admit(
        self,
        *,
        search_run_ref: ArtifactRef,
        controller_manifest_ref: ArtifactRef,
        expectation: SearchRunAdmissionExpectation,
    ) -> SearchRunAdmissionReport:
        try:
            checked_expectation = self._revalidate_expectation(expectation)
            run = self._load(
                search_run_ref,
                SearchRunManifest,
                label="search run",
                expected_media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
            )
            self._join_expectation(
                run=run,
                expectation=checked_expectation,
            )
            study = self._load(
                run.baseline_study_plan_ref,
                BaselineStudyPlan,
                label="baseline study plan",
                expected_media_type=BASELINE_STUDY_PLAN_MEDIA_TYPE,
            )
            experiment = self._load(
                run.experiment_ref,
                ExperimentManifest,
                label="experiment",
                expected_media_type=EXPERIMENT_MANIFEST_MEDIA_TYPE,
            )
            protocol = self._load(
                experiment.protocol_ref,
                ProtocolManifest,
                label="protocol",
                expected_media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
            )
            gate_config = self._load(
                protocol.gate_config_ref,
                GateConfig,
                label="gate configuration",
            )
            policy = self._load(
                run.search_policy_ref,
                SearchPolicy,
                label="search policy",
                expected_media_type=SEARCH_POLICY_MEDIA_TYPE,
            )
            stopping = self._load(
                run.stopping_policy_ref,
                SearchStoppingPolicy,
                label="search stopping policy",
                expected_media_type=SEARCH_STOPPING_POLICY_MEDIA_TYPE,
            )
            plugin = self._load(
                run.strategy_plugin_ref,
                StrategyPluginManifest,
                label="strategy plugin manifest",
                expected_media_type=STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
            )
            controller_manifest = self._load(
                controller_manifest_ref,
                SearchControllerManifest,
                label="search controller manifest",
                expected_media_type=SEARCH_CONTROLLER_MANIFEST_MEDIA_TYPE,
            )
            seed_harness = self._load(
                run.seed_harness_ref,
                HarnessManifest,
                label="seed harness",
            )
            analysis = self._load(
                run.analysis_plan_ref,
                SearchAnalysisPlan,
                label="analysis plan",
                expected_media_type=SEARCH_ANALYSIS_PLAN_MEDIA_TYPE,
            )
            arm = study.arm(run.baseline_kind)
            benchmark = self._load(
                arm.context.benchmark_ref,
                SearchBenchmarkBinding,
                label="benchmark binding",
                expected_media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
            )
            self.store.get_bytes(plugin.implementation_ref)
            for component in seed_harness.components:
                self.store.get_bytes(component.artifact)
            for split in benchmark.protocol_splits:
                self._verify_generic_json(split.manifest_ref, label="protocol split")
            safe_metadata = _load_canonical_model(
                self.store,
                benchmark.safe_benchmark_metadata_ref,
                SafeBenchmarkMetadata,
                SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
            )
            if (
                safe_metadata.benchmark_fingerprint != benchmark.benchmark_fingerprint
                or safe_metadata.exploration_task_ids != benchmark.exploration_task_ids
            ):
                raise SearchRunAdmissionError(
                    "safe benchmark metadata differs from its frozen binding"
                )
            for field_name, media_type in _EXPLORATION_FEEDBACK_REF_MEDIA_TYPES.items():
                _verify_canonical_json(
                    self.store,
                    getattr(benchmark, field_name),
                    media_type=media_type,
                )
            trajectory_index = _load_exploration_trajectory_index(self.store, benchmark)
            _, diagnostic_closure = _load_exact_diagnostic_closure(
                self.store,
                benchmark.diagnostic_evidence_ref,
                exploration_task_ids=benchmark.exploration_task_ids,
                allowed_trajectory_refs=frozenset(trajectory_index.trajectory_refs),
            )
            if diagnostic_closure != frozenset(benchmark.diagnostic_closure_refs):
                raise SearchRunAdmissionError(
                    "diagnostic evidence differs from the frozen exact closure"
                )
        except SearchRunAdmissionError:
            raise
        except Exception as exc:
            raise SearchRunAdmissionError(
                f"search run artifacts could not be strictly loaded: {exc}"
            ) from exc

        self._join_study_and_run(run=run, study=study, arm=arm)
        self._join_experiment_and_protocol(
            run=run,
            study=study,
            experiment=experiment,
            protocol=protocol,
            seed_harness=seed_harness,
        )
        self._join_analysis_and_benchmark(
            run=run,
            study=study,
            experiment=experiment,
            protocol=protocol,
            policy=policy,
            analysis=analysis,
            benchmark=benchmark,
        )
        self._join_strategy(
            run=run,
            arm=arm,
            experiment=experiment,
            policy=policy,
            stopping=stopping,
            plugin=plugin,
        )
        if (
            run.baseline_kind is not BaselineKind.STATIC
            and gate_config.confidence_level != policy.gate_confidence_level
        ):
            raise SearchRunAdmissionError(
                "gate configuration confidence does not match the corrected search policy"
            )
        self._join_controller(
            run_ref=search_run_ref,
            run=run,
            policy=policy,
            controller_ref=controller_manifest_ref,
            controller=controller_manifest,
        )

        if run.baseline_kind is BaselineKind.RANDOM_VALID:
            try:
                catalogue_ref = run.prompt_mutation_catalogue_ref
                assert catalogue_ref is not None
                catalogue = self._load(
                    catalogue_ref,
                    PromptMutationCatalogue,
                    label="prompt mutation catalogue",
                    expected_media_type=PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
                )
                if catalogue.parent_harness_ref != run.seed_harness_ref:
                    raise SearchRunAdmissionError(
                        "prompt mutation catalogue belongs to another seed harness"
                    )
                if catalogue.grammar_version != policy.mutation_policy.grammar_version:
                    raise SearchRunAdmissionError(
                        "prompt mutation catalogue grammar differs from search policy"
                    )
                target_names = {entry.target_component_name for entry in catalogue.entries}
                if len(target_names) != 1:
                    raise SearchRunAdmissionError(
                        "first-version random-valid catalogue must target exactly one "
                        "prompt component"
                    )
                target_name = next(iter(target_names))
                prompt_components = {
                    component.name: component
                    for component in seed_harness.components
                    if component.kind is ComponentKind.PROMPT
                }
                if target_name not in prompt_components:
                    raise SearchRunAdmissionError(
                        "prompt mutation catalogue target is absent from the seed harness"
                    )
                for entry in catalogue.entries:
                    self.store.get_bytes(entry.expected_before_prompt_ref)
                    self.store.get_bytes(entry.after_prompt_ref)
                    hypothesis = self._load(
                        entry.hypothesis_ref,
                        MutationHypothesis,
                        label="catalogue mutation hypothesis",
                    )
                    for evidence_ref in hypothesis.evidence_refs:
                        self.store.get_bytes(evidence_ref)
            except SearchRunAdmissionError:
                raise
            except Exception as exc:
                raise SearchRunAdmissionError(
                    f"random-valid catalogue artifacts could not be strictly loaded: {exc}"
                ) from exc

        calls_per_run = self._max_optimizer_calls_per_run(policy)

        return SearchRunAdmissionReport(
            search_run_ref=search_run_ref,
            controller_manifest_ref=controller_manifest_ref,
            baseline_study_plan_ref=run.baseline_study_plan_ref,
            experiment_ref=run.experiment_ref,
            protocol_ref=experiment.protocol_ref,
            gate_config_ref=protocol.gate_config_ref,
            analysis_plan_ref=run.analysis_plan_ref,
            benchmark_binding_ref=arm.context.benchmark_ref,
            objective_aggregate_attestor_id=benchmark.objective_aggregate_attestor_id,
            strategy_feedback_attestor_id=benchmark.strategy_feedback_attestor_id,
            search_policy_ref=run.search_policy_ref,
            stopping_policy_ref=run.stopping_policy_ref,
            strategy_plugin_ref=run.strategy_plugin_ref,
            strategy_implementation_ref=plugin.implementation_ref,
            seed_harness_ref=run.seed_harness_ref,
            baseline_kind=run.baseline_kind,
            search_run_seed=run.search_run_seed,
            repeat_seeds=run.repeat_seeds,
            gate_confidence_level=policy.gate_confidence_level,
            max_optimizer_model_calls_per_run=calls_per_run,
        )

    def verify_report(
        self,
        *,
        report_ref: ArtifactRef,
        search_run_ref: ArtifactRef,
        controller_manifest_ref: ArtifactRef,
        expectation: SearchRunAdmissionExpectation,
    ) -> SearchRunAdmissionReport:
        report = self._load(
            report_ref,
            SearchRunAdmissionReport,
            label="search run admission report",
            expected_media_type=SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE,
        )
        expected = self.admit(
            search_run_ref=search_run_ref,
            controller_manifest_ref=controller_manifest_ref,
            expectation=expectation,
        )
        if report != expected:
            raise SearchRunAdmissionError("search run admission report is stale or foreign")
        return report

    def _load[ModelT: BaseModel](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        *,
        label: str,
        expected_media_type: str | None = None,
    ) -> ModelT:
        if not isinstance(ref, ArtifactRef):
            raise SearchRunAdmissionError(f"{label} ref must be an ArtifactRef")
        if expected_media_type is not None and ref.media_type != expected_media_type:
            raise SearchRunAdmissionError(f"{label} declares the wrong media type")
        try:
            payload = self.store.get_bytes(ref)
            loaded = self.store.get_json(ref, model_type)
            if payload != canonical_json_bytes(loaded):
                raise ValueError("typed bytes are not canonical")
        except Exception as exc:
            raise SearchRunAdmissionError(f"{label} could not be strictly loaded: {exc}") from exc
        return loaded

    def _verify_generic_json(self, ref: ArtifactRef, *, label: str) -> None:
        try:
            payload = self.store.get_bytes(ref)
            value = self.store.get_json(ref)
            if payload != canonical_json_bytes(value):
                raise ValueError("JSON bytes are not canonical")
        except Exception as exc:
            raise SearchRunAdmissionError(f"{label} could not be verified: {exc}") from exc

    @staticmethod
    def _revalidate_expectation(
        expectation: SearchRunAdmissionExpectation,
    ) -> SearchRunAdmissionExpectation:
        if not isinstance(expectation, SearchRunAdmissionExpectation):
            raise SearchRunAdmissionError(
                "search run admission requires a typed external expectation"
            )
        try:
            return SearchRunAdmissionExpectation.model_validate(
                expectation.model_dump(mode="python", round_trip=True, warnings="none"),
                strict=True,
            )
        except Exception as exc:
            raise SearchRunAdmissionError("search run expectation is invalid") from exc

    @staticmethod
    def _join_expectation(
        *,
        run: SearchRunManifest,
        expectation: SearchRunAdmissionExpectation,
    ) -> None:
        actual = (
            run.baseline_study_plan_ref,
            run.experiment_ref,
            run.baseline_kind,
            run.search_run_seed,
        )
        expected = (
            expectation.baseline_study_plan_ref,
            expectation.experiment_ref,
            expectation.baseline_kind,
            expectation.search_run_seed,
        )
        if actual != expected:
            raise SearchRunAdmissionError(
                "search run differs from the caller-owned admission expectation"
            )

    @staticmethod
    def _join_study_and_run(
        *,
        run: SearchRunManifest,
        study: BaselineStudyPlan,
        arm: BaselineArmPlan,
    ) -> None:
        typed_arm = study.arm(run.baseline_kind)
        if arm != typed_arm:
            raise SearchRunAdmissionError("selected baseline arm changed during admission")
        if study.fingerprint != run.baseline_plan_fingerprint:
            raise SearchRunAdmissionError("baseline study fingerprint differs from search run")
        if run.baseline_study_plan_ref.sha256 != study.fingerprint:
            raise SearchRunAdmissionError("baseline study artifact does not match its fingerprint")
        if run.search_run_seed not in typed_arm.evaluation.search_run_seeds:
            raise SearchRunAdmissionError("search run seed is absent from the frozen arm schedule")
        if run.repeat_seeds != typed_arm.evaluation.repeat_seeds:
            raise SearchRunAdmissionError("repeat seeds differ from the frozen arm schedule")
        if run.proposal_master_seed != typed_arm.context.proposal_random_seed:
            raise SearchRunAdmissionError("proposal master seed differs from the frozen arm")
        if run.seed_harness_ref != typed_arm.context.seed_harness_ref:
            raise SearchRunAdmissionError("search seed harness differs from the frozen arm")

    @staticmethod
    def _join_analysis_and_benchmark(
        *,
        run: SearchRunManifest,
        study: BaselineStudyPlan,
        experiment: ExperimentManifest,
        protocol: ProtocolManifest,
        policy: SearchPolicy,
        analysis: SearchAnalysisPlan,
        benchmark: SearchBenchmarkBinding,
    ) -> None:
        arm = study.arm(run.baseline_kind)
        if experiment.objective != policy.objective or analysis.objective != policy.objective:
            raise SearchRunAdmissionError(
                "experiment, analysis plan, and search policy objectives differ"
            )
        if analysis.selector != policy.selector:
            raise SearchRunAdmissionError("analysis selector differs from search policy")
        if analysis.family_alpha != policy.family_alpha:
            raise SearchRunAdmissionError("analysis family alpha differs from search policy")
        if run.baseline_kind is not BaselineKind.STATIC:
            if analysis.max_gate_queries != policy.max_gate_queries:
                raise SearchRunAdmissionError(
                    "analysis gate-query limit differs from non-static search policy"
                )
            if analysis.gate_confidence_level != policy.gate_confidence_level:
                raise SearchRunAdmissionError(
                    "analysis gate confidence differs from non-static search policy"
                )
        elif policy.max_gate_queries != 0 or policy.gate_confidence_level is not None:
            raise SearchRunAdmissionError("static search policy must have no gate queries")
        if analysis.search_run_seeds != arm.evaluation.search_run_seeds:
            raise SearchRunAdmissionError("analysis search seeds differ from baseline study")
        if analysis.repeat_seeds != arm.evaluation.repeat_seeds:
            raise SearchRunAdmissionError("analysis repeat seeds differ from baseline study")
        if run.search_run_seed not in analysis.search_run_seeds:
            raise SearchRunAdmissionError("current search seed is absent from the analysis plan")
        if benchmark.benchmark_fingerprint != protocol.benchmark_fingerprint:
            raise SearchRunAdmissionError("benchmark fingerprint differs from protocol")
        if benchmark.protocol_splits != protocol.splits:
            raise SearchRunAdmissionError("benchmark split binding differs from protocol")

    @staticmethod
    def _join_experiment_and_protocol(
        *,
        run: SearchRunManifest,
        study: BaselineStudyPlan,
        experiment: ExperimentManifest,
        protocol: ProtocolManifest,
        seed_harness: HarnessManifest,
    ) -> None:
        arm = study.arm(run.baseline_kind)
        if experiment.seed_harness_ref != run.seed_harness_ref:
            raise SearchRunAdmissionError("experiment and search run use different seed harnesses")
        required_baselines = frozenset(kind.value for kind in BaselineKind)
        if frozenset(experiment.baselines) != required_baselines:
            raise SearchRunAdmissionError(
                "automatic four-arm experiment must freeze exactly the four baseline names"
            )
        frozen_planes = (
            arm.context.model_fingerprint,
            arm.context.inference_fingerprint,
            arm.context.runtime_fingerprint,
        )
        protocol_planes = (
            protocol.model_fingerprint,
            protocol.inference_fingerprint,
            protocol.runtime_fingerprint,
        )
        if frozen_planes != protocol_planes:
            raise SearchRunAdmissionError("baseline model/inference/runtime differs from protocol")
        harness_planes = (
            seed_harness.model_fingerprint,
            seed_harness.runtime_fingerprint,
            seed_harness.trusted_plane_version,
        )
        expected_harness_planes = (
            protocol.model_fingerprint,
            protocol.runtime_fingerprint,
            protocol.trusted_plane_version,
        )
        if harness_planes != expected_harness_planes:
            raise SearchRunAdmissionError("seed harness differs from protocol frozen planes")

    @staticmethod
    def _join_strategy(
        *,
        run: SearchRunManifest,
        arm: BaselineArmPlan,
        experiment: ExperimentManifest,
        policy: SearchPolicy,
        stopping: SearchStoppingPolicy,
        plugin: StrategyPluginManifest,
    ) -> None:
        typed_arm = arm
        if run.search_policy_fingerprint != run.search_policy_ref.sha256:
            raise SearchRunAdmissionError("search policy fingerprint is misbound")
        if run.strategy_plugin_fingerprint != run.strategy_plugin_ref.sha256:
            raise SearchRunAdmissionError("strategy plugin fingerprint is misbound")
        if run.stopping_policy_fingerprint != run.stopping_policy_ref.sha256:
            raise SearchRunAdmissionError("stopping policy fingerprint is misbound")
        if stopping != policy.stopping_policy:
            raise SearchRunAdmissionError("stopping policy differs from search policy")
        if policy.baseline_kind is not run.baseline_kind:
            raise SearchRunAdmissionError("search policy belongs to another baseline")
        if plugin.baseline_kind is not run.baseline_kind:
            raise SearchRunAdmissionError("strategy plugin belongs to another baseline")
        if policy.mutation_policy != typed_arm.context.mutation_policy:
            raise SearchRunAdmissionError("search policy mutation grammar differs from arm")
        if policy.available_feedback != typed_arm.available_feedback:
            raise SearchRunAdmissionError("search policy feedback differs from arm capability")
        if policy.mutation != typed_arm.mutation:
            raise SearchRunAdmissionError("search policy mutation differs from arm capability")
        if plugin.consumes_feedback != policy.available_feedback:
            raise SearchRunAdmissionError("plugin feedback declaration differs from policy")
        if plugin.mutation != policy.mutation:
            raise SearchRunAdmissionError("plugin mutation declaration differs from policy")
        if policy.mutation_policy.allowed_component_kinds != (ComponentKind.PROMPT,):
            raise SearchRunAdmissionError("automatic search requires prompt-only grammar")
        if experiment.mutation_policy.allowed_kinds != (ComponentKind.PROMPT,):
            raise SearchRunAdmissionError("experiment mutation policy is not prompt-only")
        if (
            experiment.mutation_policy.max_artifact_size_bytes
            != policy.mutation_policy.max_artifact_size_bytes
        ):
            raise SearchRunAdmissionError("experiment and search prompt size limits are different")
        search_run_count = len(typed_arm.evaluation.search_run_seeds)
        if policy.max_proposals * search_run_count > typed_arm.ceilings.max_proposals:
            raise SearchRunAdmissionError(
                "aggregate search proposal limits exceed the baseline study ceiling"
            )
        if policy.max_gate_queries * search_run_count > typed_arm.ceilings.max_feedback_queries:
            raise SearchRunAdmissionError(
                "aggregate gate-query limits exceed the baseline study ceiling"
            )
        optimizer_calls = SearchRunAdmissionService._max_optimizer_calls_per_run(policy)
        if optimizer_calls * search_run_count > typed_arm.ceilings.max_optimizer_model_calls:
            raise SearchRunAdmissionError(
                "aggregate optimizer call preflight exceeds the baseline study ceiling"
            )

    @staticmethod
    def _max_optimizer_calls_per_run(policy: SearchPolicy) -> int:
        if policy.baseline_kind in {BaselineKind.STATIC, BaselineKind.RANDOM_VALID}:
            return 0
        calls_per_round = 2 if policy.baseline_kind is BaselineKind.EVIDENCE_TARGETED else 1
        return calls_per_round * policy.max_rounds

    @staticmethod
    def _join_controller(
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        policy: SearchPolicy,
        controller_ref: ArtifactRef,
        controller: SearchControllerManifest,
    ) -> None:
        expected_coordinates = (
            run_ref,
            run.baseline_study_plan_ref,
            run.experiment_ref,
            run.baseline_kind,
            run.search_run_seed,
            run.seed_harness_ref,
            run.analysis_plan_ref,
        )
        actual_coordinates = (
            controller.search_run_ref,
            controller.study_ref,
            controller.experiment_ref,
            controller.baseline_kind,
            controller.search_seed,
            controller.initial_champion_harness_ref,
            controller.analysis_plan_ref,
        )
        if actual_coordinates != expected_coordinates:
            raise SearchRunAdmissionError(
                "search controller repeated coordinates differ from search run"
            )
        expected_limits = (
            policy.max_rounds,
            policy.max_gate_queries,
            policy.patience_rounds,
            policy.max_consecutive_declines,
        )
        actual_limits = (
            controller.max_rounds,
            controller.max_gate_nominations,
            controller.patience_rounds,
            controller.max_consecutive_declines,
        )
        if actual_limits != expected_limits:
            raise SearchRunAdmissionError(
                "search controller repeated limits differ from search policy"
            )


class StrategyOutputStage(StrEnum):
    FEEDBACK = "feedback"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    SCREEN = "screen"
    GATE = "gate"


class StrategyOutputRejectionCode(StrEnum):
    WRONG_PYTHON_TYPE = "wrong-python-type"
    SCHEMA_INVALID = "schema-invalid"
    PROVENANCE_MISMATCH = "provenance-mismatch"
    COUNT_LIMIT_EXCEEDED = "count-limit-exceeded"
    DUPLICATE_OUTPUT = "duplicate-output"
    RUNTIME_EXCEPTION = "runtime-exception"


class StrategyOutputRejection(ImmutableModel):
    """Safe archival metadata for an output that never crossed admission."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    stage: StrategyOutputStage
    code: StrategyOutputRejectionCode
    received_type: NonEmptyStr


class CandidateMaterialization(ImmutableModel):
    """Trusted proposal materialization before any exploration evaluation."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef | None = None
    candidate_harness_ref: ArtifactRef | None = None
    failure_codes: tuple[CandidateScreenFailure, ...] = ()

    @field_validator("failure_codes")
    @classmethod
    def _canonicalize_failures(
        cls,
        value: tuple[CandidateScreenFailure, ...],
    ) -> tuple[CandidateScreenFailure, ...]:
        ordered = tuple(sorted(value, key=lambda code: code.value))
        if len(ordered) != len(set(ordered)):
            raise ValueError("materialization failure codes must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _candidate_or_failure_is_exclusive(self) -> Self:
        if self.search_run_ref.media_type != SEARCH_RUN_MANIFEST_MEDIA_TYPE:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        has_candidate = self.candidate_ref is not None
        if has_candidate != (self.candidate_harness_ref is not None):
            raise ValueError("candidate and candidate harness refs must be present together")
        if has_candidate == bool(self.failure_codes):
            raise ValueError("materialization requires exactly a candidate pair or failures")
        return self


class CandidateMaterializationBatch(ImmutableModel):
    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    proposal_batch_ref: ArtifactRef
    materialization_refs: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def _proposal_batch_media_type_is_exact(self) -> Self:
        if self.proposal_batch_ref.media_type != PROPOSAL_BATCH_MEDIA_TYPE:
            raise ValueError("proposal_batch_ref declares the wrong media type")
        return self

    @field_validator("materialization_refs")
    @classmethod
    def _canonicalize_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("materialization refs must not contain duplicates")
        if any(ref.media_type != CANDIDATE_MATERIALIZATION_MEDIA_TYPE for ref in ordered):
            raise ValueError("materialization refs contain the wrong media type")
        return ordered


class CandidateScreenArchiveEntry(ImmutableModel):
    proposal_ref: ArtifactRef
    screen_ref: ArtifactRef
    candidate_ref: ArtifactRef | None
    status: CandidateScreenStatus


class CandidateScreenBatch(ImmutableModel):
    """One exact screen result for each frozen proposal in a round."""

    schema_version: Literal["1"] = "1"
    search_run_ref: ArtifactRef
    baseline_kind: BaselineKind
    round_index: Annotated[int, Field(ge=0, strict=True)]
    entries: tuple[CandidateScreenArchiveEntry, ...]

    @field_validator("entries")
    @classmethod
    def _canonicalize_entries(
        cls,
        value: tuple[CandidateScreenArchiveEntry, ...],
    ) -> tuple[CandidateScreenArchiveEntry, ...]:
        return tuple(sorted(value, key=lambda entry: entry.proposal_ref.sha256))

    @model_validator(mode="after")
    def _entries_are_unique(self) -> Self:
        proposal_hashes = tuple(entry.proposal_ref.sha256 for entry in self.entries)
        screen_hashes = tuple(entry.screen_ref.sha256 for entry in self.entries)
        if len(proposal_hashes) != len(set(proposal_hashes)):
            raise ValueError("screen batch contains duplicate proposal refs")
        if len(screen_hashes) != len(set(screen_hashes)):
            raise ValueError("screen batch contains duplicate screen refs")
        return self


class AutomaticSearchLoopResult(ImmutableModel):
    """Persisted result of a completely closed automatic search run."""

    schema_version: Literal["1"] = "1"
    non_reportable_fixture: bool = False
    search_run_ref: ArtifactRef
    admission_report_ref: ArtifactRef
    final_search_tail_ref: ArtifactRef
    search_selection_closure_ref: ArtifactRef
    final_experiment_tail_ref: ArtifactRef
    final_champion_harness_ref: ArtifactRef
    final_champion_candidate_ref: ArtifactRef | None
    completed_rounds: Annotated[int, Field(ge=0, strict=True)]
    gate_nominations: Annotated[int, Field(ge=0, strict=True)]
    diagnosis_count: Annotated[int, Field(ge=0, strict=True)]
    diagnosis_invocation_count: Annotated[int, Field(ge=0, strict=True)]
    proposal_count: Annotated[int, Field(ge=0, strict=True)]
    proposal_invocation_count: Annotated[int, Field(ge=0, strict=True)]
    screen_count: Annotated[int, Field(ge=0, strict=True)]
    optimizer_model_call_count: Annotated[int, Field(ge=0, strict=True)]
    optimizer_usage_attestation: Literal["call-count-only"] = "call-count-only"
    optimizer_tokens_attested: Literal[False] = False
    optimizer_cost_attested: Literal[False] = False
    trusted_screen_cell_count: Annotated[int, Field(ge=0, strict=True)]
    trusted_screen_charged_tokens: Annotated[int, Field(ge=0, strict=True)]
    strategy_artifact_written_bytes: Annotated[int, Field(ge=0, strict=True)]
    strategy_access_log_refs: tuple[ArtifactRef, ...]
    archived_artifact_refs: tuple[ArtifactRef, ...]

    @field_validator("strategy_access_log_refs")
    @classmethod
    def _canonicalize_access_logs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("strategy_access_log_refs must not contain duplicates")
        if any(ref.media_type != STRATEGY_ARTIFACT_ACCESS_LOG_MEDIA_TYPE for ref in ordered):
            raise ValueError("strategy_access_log_refs contains the wrong media type")
        return ordered

    @field_validator("archived_artifact_refs")
    @classmethod
    def _canonicalize_archived_refs(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("archived_artifact_refs must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _derived_counts_and_access_logs_join(self) -> Self:
        if self.optimizer_model_call_count != (
            self.diagnosis_invocation_count + self.proposal_invocation_count
        ):
            raise ValueError("optimizer call count must equal diagnosis plus proposal calls")
        archived = frozenset(self.archived_artifact_refs)
        if not set(self.strategy_access_log_refs).issubset(archived):
            raise ValueError("strategy access logs must be present in archived artifacts")
        return self


class AutomaticSearchLoopExecution(ImmutableModel):
    result_ref: ArtifactRef
    result: AutomaticSearchLoopResult


@runtime_checkable
class StrategyPluginRuntime(Protocol):
    """Untrusted optimizer surface: typed views and immutable refs only."""

    @property
    def manifest_ref(self) -> ArtifactRef: ...

    @property
    def implementation_ref(self) -> ArtifactRef: ...

    def diagnose(
        self,
        *,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        search_run_ref: ArtifactRef,
        round_index: int,
        parent_harness_ref: ArtifactRef,
        artifacts: StrategyArtifactView,
    ) -> object: ...

    def propose(
        self,
        *,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        search_run_ref: ArtifactRef,
        round_index: int,
        parent_harness_ref: ArtifactRef,
        diagnosis_refs: tuple[ArtifactRef, ...],
        artifacts: StrategyArtifactView,
    ) -> object: ...


@runtime_checkable
class AutomaticSearchRuntime(Protocol):
    """Trusted benchmark/materialization/gate adapter owned outside plugins."""

    def collect_feedback(
        self,
        *,
        search_run_ref: ArtifactRef,
        baseline_kind: BaselineKind,
        round_index: int,
        champion_harness_ref: ArtifactRef,
        prior_gate_aggregate: GateAggregateView | None,
    ) -> object: ...

    def materialize_proposal(
        self,
        *,
        search_run_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposal: PromptProposal,
        proposal_ref: ArtifactRef,
        champion_harness_ref: ArtifactRef,
    ) -> object: ...

    def screen_candidate(
        self,
        *,
        search_run_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposal: PromptProposal,
        proposal_ref: ArtifactRef,
        materialization: CandidateMaterialization,
        champion_harness_ref: ArtifactRef,
    ) -> object: ...

    def run_gate(
        self,
        *,
        search_run_ref: ArtifactRef,
        nomination: Nomination,
        nomination_ref: ArtifactRef,
        search_tail_ref: ArtifactRef,
    ) -> object: ...

    def attempt_ledger_for(
        self,
        evaluation_ref: ArtifactRef,
    ) -> AttemptLedger: ...


@runtime_checkable
class ExperimentLifecycleCoordinator(Protocol):
    """Minimum experiment lifecycle authority needed after search closure."""

    @property
    def experiment_tail_ref(self) -> ArtifactRef | None: ...

    def close_current_selection(
        self,
        *,
        previous_tail_ref: ArtifactRef,
        analysis_plan_ref: ArtifactRef,
    ) -> ArtifactRef: ...

    def verify_experiment_selection_closure(
        self,
        tail_ref: ArtifactRef,
    ) -> SelectionClosure: ...


@dataclass(frozen=True, slots=True)
class _IssuedAutomaticSearchExecution:
    """Process-local proof that one exact loop capability produced a result."""

    result_ref: ArtifactRef
    result: AutomaticSearchLoopResult
    search_run_ref: ArtifactRef
    controller: SearchController
    lifecycle: ExperimentLifecycleCoordinator
    expectation: SearchRunAdmissionExpectation


class AutomaticSearchLoop:
    """Drive one admitted SearchController until immutable selection closure."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        runtime: AutomaticSearchRuntime,
        lifecycle: ExperimentLifecycleCoordinator,
        objective_aggregate_verifier: ObjectiveAggregateVerificationCapability,
        strategy_feedback_verifier: StrategyFeedbackVerificationCapability,
        plugin: StrategyPluginRuntime | None = None,
        non_reportable_fixture: bool = False,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.lifecycle = lifecycle
        if type(objective_aggregate_verifier) is not ObjectiveAggregateVerificationCapability:
            raise TypeError(
                "objective_aggregate_verifier must be the exact trusted capability type"
            )
        self.objective_aggregate_verifier = objective_aggregate_verifier
        if type(strategy_feedback_verifier) is not StrategyFeedbackVerificationCapability:
            raise TypeError("strategy_feedback_verifier must be the exact trusted capability type")
        self.strategy_feedback_verifier = strategy_feedback_verifier
        self.plugin = plugin
        self.non_reportable_fixture = non_reportable_fixture
        self.admission = SearchRunAdmissionService(store)
        self.__issued_execution: _IssuedAutomaticSearchExecution | None = None

    def run(
        self,
        *,
        search_run_ref: ArtifactRef,
        controller: SearchController,
        expectation: SearchRunAdmissionExpectation,
    ) -> AutomaticSearchLoopExecution:
        if self.__issued_execution is not None:
            raise AutomaticSearchLoopError(
                "one automatic-search loop capability may issue only one run result"
            )
        report = self.admission.admit(
            search_run_ref=search_run_ref,
            controller_manifest_ref=controller.controller_manifest_ref,
            expectation=expectation,
        )
        self._verify_objective_attestor(report.objective_aggregate_attestor_id)
        self._verify_feedback_attestor(report.strategy_feedback_attestor_id)
        admission_ref = self.store.put_json(
            report,
            media_type=SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE,
        )
        run = self._load_typed(search_run_ref, SearchRunManifest)
        policy = self._load_typed(run.search_policy_ref, SearchPolicy)
        self._verify_plugin_binding(
            run,
            implementation_ref=report.strategy_implementation_ref,
        )

        archived: dict[str, ArtifactRef] = {admission_ref.sha256: admission_ref}
        access_log_refs: list[ArtifactRef] = []
        diagnosis_count = 0
        diagnosis_invocation_count = 0
        proposal_count = 0
        proposal_invocation_count = 0
        screen_count = 0
        trusted_screen_cell_count = 0
        trusted_screen_charged_tokens = 0
        strategy_artifact_written_bytes = 0
        seen_proposal_ids: set[str] = set()
        seen_mutation_coordinates: set[tuple[str, str, str]] = set()
        seen_catalogue_entry_ids: set[str] = set()

        def archive(ref: ArtifactRef) -> None:
            prior = archived.get(ref.sha256)
            if prior is not None and prior != ref:
                raise AutomaticSearchLoopError(
                    "one artifact digest was presented with conflicting reference metadata"
                )
            archived[ref.sha256] = ref

        tail = controller.freeze() if controller.tail_ref is None else controller.tail_ref
        assert tail is not None
        if controller.state is SearchState.FROZEN:
            tail = controller.start_search(previous_tail_ref=tail)
        snapshot = controller.snapshot
        if snapshot is None or snapshot.state is not SearchState.SEARCHING:
            raise AutomaticSearchLoopError("search controller is not in SEARCHING state")
        if snapshot.active_round_index is not None:
            raise AutomaticSearchLoopError("automatic loop cannot resume a partially open round")
        if snapshot.completed_rounds:
            raise AutomaticSearchLoopError(
                "automatic loop does not support resume after completed rounds"
            )

        while not snapshot.stop_reasons:
            tail = controller.open_round(previous_tail_ref=tail)
            snapshot = self._snapshot(controller)
            round_index = snapshot.active_round_index
            assert round_index is not None
            champion_ref = snapshot.champion_harness_ref

            prior_gate, prior_gate_ref = self._last_gate_view(
                run=run,
                controller=controller,
            )
            if prior_gate_ref is not None:
                archive(prior_gate_ref)
            trusted_feedback, trusted_feedback_ref = self._call_feedback(
                run_ref=search_run_ref,
                run=run,
                round_index=round_index,
                champion_ref=champion_ref,
                prior_gate=prior_gate,
                controller=controller,
                tail=tail,
            )
            feedback = trusted_feedback.view
            archive(trusted_feedback_ref)
            bound_feedback = RunBoundStrategyFeedback(
                search_run_ref=search_run_ref,
                experiment_ref=run.experiment_ref,
                baseline_kind=run.baseline_kind,
                search_run_seed=run.search_run_seed,
                round_index=round_index,
                champion_harness_ref=champion_ref,
                trusted_feedback_ref=trusted_feedback_ref,
                view=feedback,
            )
            feedback_ref = self.store.put_json(
                bound_feedback,
                media_type=RUN_BOUND_STRATEGY_FEEDBACK_MEDIA_TYPE,
            )
            archive(feedback_ref)
            tail = controller.record_evidence(
                previous_tail_ref=tail,
                evidence_packet_ref=feedback_ref,
            )

            diagnosis_artifacts: StrategyArtifactView | None = None
            diagnosis_refs: tuple[ArtifactRef, ...] = ()
            if run.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
                remaining = policy.max_diagnoses - diagnosis_count
                if remaining > 0:
                    diagnosis_artifacts = self._make_strategy_artifact_view(
                        run_ref=search_run_ref,
                        run=run,
                        policy=policy,
                        round_index=round_index,
                        champion_ref=champion_ref,
                        trusted_feedback=trusted_feedback,
                        feedback_ref=feedback_ref,
                        invocation="diagnosis",
                        additional_read_refs=(),
                    )
                    diagnosis_invocation_count += 1
                    diagnoses = self._call_diagnose(
                        run_ref=search_run_ref,
                        run=run,
                        round_index=round_index,
                        champion_ref=champion_ref,
                        feedback=feedback,
                        feedback_ref=feedback_ref,
                        remaining=remaining,
                        controller=controller,
                        tail=tail,
                        artifacts=diagnosis_artifacts,
                    )
                    diagnosis_refs = tuple(
                        self.store.put_json(diagnosis, media_type=DIAGNOSIS_MEDIA_TYPE)
                        for diagnosis in diagnoses
                    )
                    for diagnosis_ref in diagnosis_refs:
                        archive(diagnosis_ref)
                    diagnosis_count += len(diagnosis_refs)

            proposal_artifacts = self._make_strategy_artifact_view(
                run_ref=search_run_ref,
                run=run,
                policy=policy,
                round_index=round_index,
                champion_ref=champion_ref,
                trusted_feedback=trusted_feedback,
                feedback_ref=feedback_ref,
                invocation="proposal",
                additional_read_refs=diagnosis_refs,
            )

            batch, random_selection_ref, proposal_invoked = self._proposal_batch(
                run_ref=search_run_ref,
                run=run,
                policy=policy,
                round_index=round_index,
                champion_ref=champion_ref,
                feedback=feedback,
                feedback_ref=feedback_ref,
                diagnosis_refs=diagnosis_refs,
                remaining_proposals=policy.max_proposals - proposal_count,
                controller=controller,
                tail=tail,
                artifacts=proposal_artifacts,
                excluded_catalogue_entry_ids=frozenset(seen_catalogue_entry_ids),
            )
            proposal_invocation_count += int(proposal_invoked)
            optimizer_call_count = diagnosis_invocation_count + proposal_invocation_count
            if optimizer_call_count > report.max_optimizer_model_calls_per_run:
                self._reject_and_raise(
                    run=run,
                    run_ref=search_run_ref,
                    round_index=round_index,
                    stage=StrategyOutputStage.PROPOSAL,
                    raw=batch,
                    code=StrategyOutputRejectionCode.COUNT_LIMIT_EXCEEDED,
                    controller=controller,
                    tail=tail,
                )
            if random_selection_ref is not None:
                archive(random_selection_ref)
            proposal_ids = {proposal.proposal_id for proposal in batch.proposals}
            mutation_coordinates = {
                (
                    proposal.target_component_name,
                    proposal.before_prompt_ref.sha256,
                    proposal.after_prompt_ref.sha256,
                )
                for proposal in batch.proposals
            }
            if seen_proposal_ids.intersection(proposal_ids) or (
                seen_mutation_coordinates.intersection(mutation_coordinates)
            ):
                self._reject_and_raise(
                    run=run,
                    run_ref=search_run_ref,
                    round_index=round_index,
                    stage=StrategyOutputStage.PROPOSAL,
                    raw=batch,
                    code=StrategyOutputRejectionCode.DUPLICATE_OUTPUT,
                    controller=controller,
                    tail=tail,
                )
            seen_proposal_ids.update(proposal_ids)
            seen_mutation_coordinates.update(mutation_coordinates)
            seen_catalogue_entry_ids.update(
                proposal.catalogue_entry_id
                for proposal in batch.proposals
                if proposal.catalogue_entry_id is not None
            )
            batch_ref = self.store.put_json(batch, media_type=PROPOSAL_BATCH_MEDIA_TYPE)
            archive(batch_ref)
            proposal_refs = tuple(
                self.store.put_json(proposal, media_type=PROMPT_PROPOSAL_MEDIA_TYPE)
                for proposal in batch.proposals
            )
            for proposal_ref in proposal_refs:
                archive(proposal_ref)
            proposal_count += len(proposal_refs)

            views_to_archive = tuple(
                view
                for view in (
                    diagnosis_artifacts,
                    proposal_artifacts if proposal_invoked else None,
                )
                if view is not None
            )
            for strategy_view in views_to_archive:
                access_log = strategy_view.access_log()
                access_log_ref = self.store.put_json(
                    access_log,
                    media_type=STRATEGY_ARTIFACT_ACCESS_LOG_MEDIA_TYPE,
                )
                archive(access_log_ref)
                access_log_refs.append(access_log_ref)
                strategy_artifact_written_bytes += access_log.written_bytes

            materializations = self._materialize_proposals(
                run_ref=search_run_ref,
                run=run,
                round_index=round_index,
                champion_ref=champion_ref,
                feedback_ref=feedback_ref,
                proposals=batch.proposals,
                proposal_refs=proposal_refs,
                controller=controller,
                tail=tail,
            )
            materialization_refs = tuple(
                self.store.put_json(
                    materialization,
                    media_type=CANDIDATE_MATERIALIZATION_MEDIA_TYPE,
                )
                for materialization in materializations
            )
            for materialization_ref in materialization_refs:
                archive(materialization_ref)
            materialization_batch = CandidateMaterializationBatch(
                search_run_ref=search_run_ref,
                baseline_kind=run.baseline_kind,
                round_index=round_index,
                proposal_batch_ref=batch_ref,
                materialization_refs=materialization_refs,
            )
            materialization_batch_ref = self.store.put_json(
                materialization_batch,
                media_type=CANDIDATE_MATERIALIZATION_BATCH_MEDIA_TYPE,
            )
            archive(materialization_batch_ref)
            candidate_refs = tuple(
                materialization.candidate_ref
                for materialization in materializations
                if materialization.candidate_ref is not None
            )
            tail = controller.record_candidate_pool(
                previous_tail_ref=tail,
                candidate_pool_ref=materialization_batch_ref,
                candidate_refs=candidate_refs,
                declined=batch.decline is not None,
            )

            screens, trusted_evaluations = self._screen_materializations(
                run_ref=search_run_ref,
                run=run,
                policy=policy,
                round_index=round_index,
                champion_ref=champion_ref,
                feedback_ref=feedback_ref,
                proposals=batch.proposals,
                proposal_refs=proposal_refs,
                materializations=materializations,
                remaining_screens=policy.max_screens - screen_count,
                controller=controller,
                tail=tail,
            )
            screen_refs = tuple(
                self.store.put_json(screen, media_type=CANDIDATE_SCREEN_MEDIA_TYPE)
                for screen in screens
            )
            for screen_ref in screen_refs:
                archive(screen_ref)
            screen_count += len(screen_refs)
            trusted_screen_cell_count += sum(
                evaluation.trusted_usage.cell_count for evaluation in trusted_evaluations
            )
            trusted_screen_charged_tokens += sum(
                evaluation.trusted_usage.charged_tokens for evaluation in trusted_evaluations
            )
            for screen in screens:
                if screen.evaluation_ref is not None:
                    archive(screen.evaluation_ref)
            screen_batch = CandidateScreenBatch(
                search_run_ref=search_run_ref,
                baseline_kind=run.baseline_kind,
                round_index=round_index,
                entries=tuple(
                    CandidateScreenArchiveEntry(
                        proposal_ref=proposal_ref,
                        screen_ref=screen_ref,
                        candidate_ref=screen.candidate_ref,
                        status=screen.status,
                    )
                    for proposal_ref, screen_ref, screen in zip(
                        proposal_refs,
                        screen_refs,
                        screens,
                        strict=True,
                    )
                ),
            )
            screen_batch_ref = self.store.put_json(
                screen_batch,
                media_type=CANDIDATE_SCREEN_BATCH_MEDIA_TYPE,
            )
            archive(screen_batch_ref)
            eligible_candidate_refs = tuple(
                screen.candidate_ref
                for screen in screens
                if screen.status is CandidateScreenStatus.ELIGIBLE
                and screen.candidate_ref is not None
            )
            tail = controller.record_screen(
                previous_tail_ref=tail,
                screen_report_ref=screen_batch_ref,
                eligible_candidate_refs=eligible_candidate_refs,
            )

            nomination = nominate_candidate(policy=policy, screens=screens)
            if nomination is None:
                tail = controller.record_no_candidate(
                    previous_tail_ref=tail,
                    reason=(
                        "proposal batch explicitly declined"
                        if batch.decline is not None
                        else "no candidate passed the trusted exploration screen"
                    ),
                )
                tail = controller.complete_no_candidate_round(previous_tail_ref=tail)
            else:
                nomination_ref = self.store.put_json(
                    nomination,
                    media_type=NOMINATION_MEDIA_TYPE,
                )
                archive(nomination_ref)
                tail = controller.nominate(
                    previous_tail_ref=tail,
                    nomination_ref=nomination_ref,
                    candidate_ref=nomination.candidate_ref,
                )
                terminal_ref = self._call_gate(
                    run_ref=search_run_ref,
                    nomination=nomination,
                    nomination_ref=nomination_ref,
                    tail=tail,
                    controller=controller,
                )
                try:
                    tail = controller.complete_nominated_round(
                        previous_tail_ref=tail,
                        terminal_authorization_ref=terminal_ref,
                    )
                except Exception as exc:
                    self._reject_and_raise(
                        run=run,
                        run_ref=search_run_ref,
                        round_index=round_index,
                        stage=StrategyOutputStage.GATE,
                        raw=terminal_ref,
                        code=self._error_code(exc),
                        controller=controller,
                        tail=tail,
                    )
            snapshot = self._snapshot(controller)

        tail = controller.close_selection(previous_tail_ref=tail)
        snapshot = controller.verify_selection_closure(tail)
        if snapshot.selection_closure_ref is None:  # pragma: no cover - controller proves it
            raise AutomaticSearchLoopError("search selection closure ref is missing")
        experiment_tail = self.lifecycle.experiment_tail_ref
        if experiment_tail is None:
            raise AutomaticSearchLoopError(
                "experiment lifecycle must already be in SEARCHING state"
            )
        final_experiment_tail = self.lifecycle.close_current_selection(
            previous_tail_ref=experiment_tail,
            analysis_plan_ref=run.analysis_plan_ref,
        )
        try:
            experiment_closure = self.lifecycle.verify_experiment_selection_closure(
                final_experiment_tail
            )
        except Exception as exc:
            raise AutomaticSearchLoopError(
                "experiment selection closure could not be re-authenticated"
            ) from exc
        expected_experiment_closure = (
            run.experiment_ref,
            report.protocol_ref,
            run.analysis_plan_ref,
            snapshot.champion_candidate_ref,
            snapshot.champion_harness_ref,
        )
        actual_experiment_closure = (
            experiment_closure.experiment_ref,
            experiment_closure.protocol_ref,
            experiment_closure.analysis_plan_ref,
            experiment_closure.champion_candidate_ref,
            experiment_closure.champion_harness_ref,
        )
        if actual_experiment_closure != expected_experiment_closure:
            raise AutomaticSearchLoopError(
                "experiment closure differs from the exact search selection closure"
            )
        optimizer_call_count = diagnosis_invocation_count + proposal_invocation_count
        result = AutomaticSearchLoopResult(
            non_reportable_fixture=self.non_reportable_fixture,
            search_run_ref=search_run_ref,
            admission_report_ref=admission_ref,
            final_search_tail_ref=tail,
            search_selection_closure_ref=snapshot.selection_closure_ref,
            final_experiment_tail_ref=final_experiment_tail,
            final_champion_harness_ref=snapshot.champion_harness_ref,
            final_champion_candidate_ref=snapshot.champion_candidate_ref,
            completed_rounds=snapshot.completed_rounds,
            gate_nominations=snapshot.gate_nominations,
            diagnosis_count=diagnosis_count,
            diagnosis_invocation_count=diagnosis_invocation_count,
            proposal_count=proposal_count,
            proposal_invocation_count=proposal_invocation_count,
            screen_count=screen_count,
            optimizer_model_call_count=optimizer_call_count,
            trusted_screen_cell_count=trusted_screen_cell_count,
            trusted_screen_charged_tokens=trusted_screen_charged_tokens,
            strategy_access_log_refs=tuple(access_log_refs),
            strategy_artifact_written_bytes=strategy_artifact_written_bytes,
            archived_artifact_refs=tuple(archived.values()),
        )
        result_ref = self.store.put_json(
            result,
            media_type=AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
        )
        checked_expectation = SearchRunAdmissionExpectation.model_validate(
            expectation.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        self.__issued_execution = _IssuedAutomaticSearchExecution(
            result_ref=result_ref,
            result=result,
            search_run_ref=search_run_ref,
            controller=controller,
            lifecycle=self.lifecycle,
            expectation=checked_expectation,
        )
        return AutomaticSearchLoopExecution(result_ref=result_ref, result=result)

    def verify_execution(
        self,
        result_ref: ArtifactRef,
        *,
        search_run_ref: ArtifactRef,
        controller: SearchController,
        search_selection_tail_ref: ArtifactRef,
        lifecycle: ExperimentLifecycleCoordinator,
        experiment_selection_tail_ref: ArtifactRef,
    ) -> AutomaticSearchLoopResult:
        """Re-authenticate one result issued by this exact live loop capability."""

        issued = self.__issued_execution
        checked_result_ref = ArtifactRef.model_validate(result_ref)
        if issued is None or checked_result_ref != issued.result_ref:
            raise AutomaticSearchLoopError(
                "automatic search result was not issued by this loop capability"
            )
        if (
            search_run_ref != issued.search_run_ref
            or controller is not issued.controller
            or lifecycle is not issued.lifecycle
            or self.lifecycle is not issued.lifecycle
        ):
            raise AutomaticSearchLoopError(
                "automatic search result belongs to another live run capability"
            )
        persisted = self._load_exact(
            checked_result_ref,
            AutomaticSearchLoopResult,
            AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE,
        )
        if persisted != issued.result:
            raise AutomaticSearchLoopError(
                "automatic search result differs from the capability-issued result"
            )
        if (
            persisted.final_search_tail_ref != search_selection_tail_ref
            or persisted.final_experiment_tail_ref != experiment_selection_tail_ref
        ):
            raise AutomaticSearchLoopError(
                "automatic search result belongs to another selection tail"
            )
        if persisted.admission_report_ref not in persisted.archived_artifact_refs:
            raise AutomaticSearchLoopError(
                "automatic search result omitted its admission report from the archive"
            )
        self.admission.verify_report(
            report_ref=persisted.admission_report_ref,
            search_run_ref=issued.search_run_ref,
            controller_manifest_ref=controller.controller_manifest_ref,
            expectation=issued.expectation,
        )
        search_snapshot = controller.verify_selection_closure(search_selection_tail_ref)
        experiment_closure = lifecycle.verify_experiment_selection_closure(
            experiment_selection_tail_ref
        )
        actual = (
            persisted.search_run_ref,
            persisted.final_search_tail_ref,
            persisted.search_selection_closure_ref,
            persisted.final_experiment_tail_ref,
            persisted.final_champion_harness_ref,
            persisted.final_champion_candidate_ref,
            persisted.completed_rounds,
            persisted.gate_nominations,
        )
        expected = (
            issued.search_run_ref,
            search_snapshot.tail_ref,
            search_snapshot.selection_closure_ref,
            experiment_selection_tail_ref,
            search_snapshot.champion_harness_ref,
            search_snapshot.champion_candidate_ref,
            search_snapshot.completed_rounds,
            search_snapshot.gate_nominations,
        )
        if actual != expected:
            raise AutomaticSearchLoopError(
                "automatic search result differs from the replayed search closure"
            )
        experiment_coordinates = (
            experiment_closure.experiment_ref,
            experiment_closure.analysis_plan_ref,
            experiment_closure.champion_harness_ref,
            experiment_closure.champion_candidate_ref,
        )
        search_coordinates = (
            controller.manifest.experiment_ref,
            controller.manifest.analysis_plan_ref,
            search_snapshot.champion_harness_ref,
            search_snapshot.champion_candidate_ref,
        )
        if experiment_coordinates != search_coordinates:
            raise AutomaticSearchLoopError(
                "automatic search result has divergent search and experiment closures"
            )
        return persisted

    def _verify_objective_attestor(self, expected_attestor_id: str) -> None:
        if self.objective_aggregate_verifier.attestor_id != expected_attestor_id:
            raise AutomaticSearchLoopError(
                "objective aggregate verifier differs from the frozen benchmark authority"
            )

    def _verify_feedback_attestor(self, expected_attestor_id: str) -> None:
        if self.strategy_feedback_verifier.attestor_id != expected_attestor_id:
            raise AutomaticSearchLoopError(
                "strategy feedback verifier differs from the frozen benchmark authority"
            )

    def _verify_plugin_binding(
        self,
        run: SearchRunManifest,
        *,
        implementation_ref: ArtifactRef,
    ) -> None:
        optimizer_arm = run.baseline_kind in {
            BaselineKind.PROMPT_ONLY,
            BaselineKind.EVIDENCE_TARGETED,
        }
        if optimizer_arm:
            if self.plugin is None or not isinstance(self.plugin, StrategyPluginRuntime):
                raise AutomaticSearchLoopError("optimizer arm requires a strategy plugin runtime")
            if self.plugin.manifest_ref != run.strategy_plugin_ref:
                raise AutomaticSearchLoopError("strategy runtime belongs to another manifest")
            if self.plugin.implementation_ref != implementation_ref:
                raise AutomaticSearchLoopError(
                    "strategy runtime implementation differs from its frozen manifest"
                )
        elif self.plugin is not None:
            raise AutomaticSearchLoopError("static/random-valid must not receive an optimizer")

    def _make_strategy_artifact_view(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        policy: SearchPolicy,
        round_index: int,
        champion_ref: ArtifactRef,
        trusted_feedback: TrustedStrategyFeedbackContent,
        feedback_ref: ArtifactRef,
        invocation: Literal["diagnosis", "proposal"],
        additional_read_refs: tuple[ArtifactRef, ...],
    ) -> StrategyArtifactView | None:
        if run.baseline_kind not in {
            BaselineKind.PROMPT_ONLY,
            BaselineKind.EVIDENCE_TARGETED,
        }:
            return None
        feedback = trusted_feedback.view
        benchmark = self._load_exact(
            trusted_feedback.benchmark_binding_ref,
            SearchBenchmarkBinding,
            SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
        )
        allowed: set[ArtifactRef] = {
            run_ref,
            run.experiment_ref,
            run.analysis_plan_ref,
            champion_ref,
            feedback_ref,
            benchmark.safe_benchmark_metadata_ref,
            *additional_read_refs,
        }
        if run.baseline_kind in {BaselineKind.PROMPT_ONLY, BaselineKind.EVIDENCE_TARGETED}:
            allowed.update(
                {
                    benchmark.exploration_inputs_ref,
                    benchmark.exploration_aggregates_ref,
                    benchmark.exploration_item_feedback_ref,
                    benchmark.exploration_trajectories_ref,
                }
            )
            trajectory_index = _load_exploration_trajectory_index(self.store, benchmark)
            allowed.update(trajectory_index.trajectory_refs)
        champion = self._load_typed(champion_ref, HarnessManifest)
        allowed.update(component.artifact for component in champion.components)
        if feedback.gate_aggregate is not None:
            allowed.add(feedback.gate_aggregate.candidate_ref)
            allowed.add(feedback.gate_aggregate.analysis_plan_ref)
        if run.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            allowed.update(benchmark.diagnostic_closure_refs)
        return StrategyArtifactView(
            self.store,
            search_run_ref=run_ref,
            baseline_kind=run.baseline_kind,
            round_index=round_index,
            invocation=invocation,
            allowed_read_refs=tuple(allowed),
            max_prompt_bytes=policy.mutation_policy.max_artifact_size_bytes,
        )

    def _call_feedback(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        round_index: int,
        champion_ref: ArtifactRef,
        prior_gate: GateAggregateView | None,
        controller: SearchController,
        tail: ArtifactRef,
    ) -> tuple[TrustedStrategyFeedbackContent, ArtifactRef]:
        try:
            study = self._load_exact(
                run.baseline_study_plan_ref,
                BaselineStudyPlan,
                BASELINE_STUDY_PLAN_MEDIA_TYPE,
            )
            expected_benchmark_ref = study.arm(run.baseline_kind).context.benchmark_ref
            benchmark = self._load_exact(
                expected_benchmark_ref,
                SearchBenchmarkBinding,
                SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
            )
            raw = self.runtime.collect_feedback(
                search_run_ref=run_ref,
                baseline_kind=run.baseline_kind,
                round_index=round_index,
                champion_harness_ref=champion_ref,
                prior_gate_aggregate=prior_gate,
            )
            if not isinstance(raw, ArtifactRef):
                raise TypeError("feedback runtime must return an attested ArtifactRef")
            trusted_feedback_ref = ArtifactRef.model_validate(raw, strict=True)
            trusted = StrategyFeedbackVerificationCapability.verify(
                self.strategy_feedback_verifier,
                trusted_feedback_ref,
            )
            actual_coordinates = (
                trusted.search_run_ref,
                trusted.experiment_ref,
                trusted.benchmark_binding_ref,
                trusted.exploration_split_ref,
                trusted.baseline_kind,
                trusted.search_run_seed,
                trusted.round_index,
                trusted.champion_harness_ref,
                trusted.prior_gate_aggregate,
            )
            expected_coordinates = (
                run_ref,
                run.experiment_ref,
                expected_benchmark_ref,
                benchmark.exploration_split_ref,
                run.baseline_kind,
                run.search_run_seed,
                round_index,
                champion_ref,
                prior_gate,
            )
            if actual_coordinates != expected_coordinates:
                raise ValueError("trusted feedback belongs to another run or round")
            feedback = self._revalidate(StrategyFeedbackView, trusted.view)
        except Exception as exc:
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.FEEDBACK,
                raw=locals().get("raw"),
                code=self._error_code(exc),
                controller=controller,
                tail=tail,
            )
        assert isinstance(feedback, StrategyFeedbackView)
        assert isinstance(trusted, TrustedStrategyFeedbackContent)
        if feedback.baseline_kind is not run.baseline_kind:
            self._provenance_failure(
                "feedback belongs to another baseline",
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.FEEDBACK,
                raw=feedback,
                controller=controller,
                tail=tail,
            )
        if feedback.gate_aggregate != prior_gate:
            self._provenance_failure(
                "feedback gate aggregate was not derived from the controller disclosure",
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.FEEDBACK,
                raw=feedback,
                controller=controller,
                tail=tail,
            )
        try:
            if feedback.benchmark_metadata_ref != benchmark.safe_benchmark_metadata_ref:
                raise ValueError("feedback metadata differs from the frozen safe view")
            trajectory_index = _load_exploration_trajectory_index(self.store, benchmark)
            for field_name, media_type in _EXPLORATION_FEEDBACK_REF_MEDIA_TYPES.items():
                value = getattr(feedback, field_name)
                if value is None:
                    continue
                frozen_ref = getattr(benchmark, field_name)
                if value != frozen_ref:
                    raise ValueError(f"{field_name} differs from its frozen role ref")
                _verify_canonical_json(self.store, frozen_ref, media_type=media_type)
            if feedback.diagnostic_evidence_ref is not None:
                if feedback.diagnostic_evidence_ref != benchmark.diagnostic_evidence_ref:
                    raise ValueError("diagnostic evidence differs from its frozen root")
                _, diagnostic_refs = self._load_diagnostic_closure(
                    benchmark.diagnostic_evidence_ref,
                    exploration_task_ids=benchmark.exploration_task_ids,
                    allowed_trajectory_refs=frozenset(trajectory_index.trajectory_refs),
                )
                if diagnostic_refs != frozenset(benchmark.diagnostic_closure_refs):
                    raise ValueError("diagnostic evidence differs from its frozen closure")
        except Exception:
            self._provenance_failure(
                "feedback contains a foreign, missing, or malformed artifact",
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.FEEDBACK,
                raw=feedback,
                controller=controller,
                tail=tail,
            )
        return trusted, trusted_feedback_ref

    def _load_diagnostic_closure(
        self,
        cluster_ref: ArtifactRef,
        *,
        exploration_task_ids: tuple[str, ...] | None = None,
        allowed_trajectory_refs: frozenset[ArtifactRef] | None = None,
    ) -> tuple[DiagnosticCluster, frozenset[ArtifactRef]]:
        try:
            return _load_exact_diagnostic_closure(
                self.store,
                cluster_ref,
                exploration_task_ids=exploration_task_ids,
                allowed_trajectory_refs=allowed_trajectory_refs,
            )
        except AutomaticSearchLoopError:
            raise
        except Exception as exc:
            raise AutomaticSearchLoopError(
                f"diagnostic evidence could not be strictly loaded: {exc}"
            ) from exc

    def _call_diagnose(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        round_index: int,
        champion_ref: ArtifactRef,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        remaining: int,
        controller: SearchController,
        tail: ArtifactRef,
        artifacts: StrategyArtifactView | None,
    ) -> tuple[Diagnosis, ...]:
        if self.plugin is None or artifacts is None:
            raise AutomaticSearchLoopError(
                "evidence-targeted diagnosis requires a scoped strategy artifact view"
            )
        try:
            raw = self.plugin.diagnose(
                feedback=feedback,
                feedback_ref=feedback_ref,
                search_run_ref=run_ref,
                round_index=round_index,
                parent_harness_ref=champion_ref,
                artifacts=artifacts,
            )
            if not isinstance(raw, tuple):
                raise TypeError("diagnoser output must be a tuple")
            diagnoses = tuple(self._revalidate(Diagnosis, value) for value in raw)
        except Exception as exc:
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.DIAGNOSIS,
                raw=locals().get("raw"),
                code=self._error_code(exc),
                controller=controller,
                tail=tail,
            )
        if len(diagnoses) > remaining:
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.DIAGNOSIS,
                raw=diagnoses,
                code=StrategyOutputRejectionCode.COUNT_LIMIT_EXCEEDED,
                controller=controller,
                tail=tail,
            )
        ids = tuple(diagnosis.diagnosis_id for diagnosis in diagnoses)
        if len(ids) != len(set(ids)):
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.DIAGNOSIS,
                raw=diagnoses,
                code=StrategyOutputRejectionCode.DUPLICATE_OUTPUT,
                controller=controller,
                tail=tail,
            )
        try:
            champion = self._load_typed(champion_ref, HarnessManifest)
            prompt_names = {
                component.name
                for component in champion.components
                if component.kind is ComponentKind.PROMPT
            }
            diagnostic_ref = feedback.diagnostic_evidence_ref
            if diagnostic_ref is None:
                raise ValueError("evidence-targeted feedback has no diagnostic cluster")
            cluster, _ = self._load_diagnostic_closure(diagnostic_ref)
            allowed_failures = frozenset(cluster.failure_signature_refs)
            allowed_evidence = frozenset(cluster.evidence_packet_refs)
            cited_refs: set[ArtifactRef] = set()
            valid = True
            for diagnosis in diagnoses:
                cited_refs.update(diagnosis.failure_signature_refs)
                cited_refs.update(diagnosis.evidence_packet_refs)
                cited_refs.update(diagnosis.protected_anchor_refs)
                valid = valid and (
                    diagnosis.source_feedback_ref == feedback_ref
                    and diagnosis.target_component_name in prompt_names
                    and set(diagnosis.failure_signature_refs).issubset(allowed_failures)
                    and set(diagnosis.evidence_packet_refs).issubset(allowed_evidence)
                    and set(diagnosis.protected_anchor_refs).issubset(allowed_evidence)
                )
            valid = valid and cited_refs.issubset(artifacts.read_refs)
        except Exception:
            valid = False
        if not valid:
            self._provenance_failure(
                "diagnosis evidence, source, target, or scoped reads differ from this round",
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.DIAGNOSIS,
                raw=diagnoses,
                controller=controller,
                tail=tail,
            )
        return diagnoses

    def _proposal_batch(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        policy: SearchPolicy,
        round_index: int,
        champion_ref: ArtifactRef,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        diagnosis_refs: tuple[ArtifactRef, ...],
        remaining_proposals: int,
        controller: SearchController,
        tail: ArtifactRef,
        artifacts: StrategyArtifactView | None,
        excluded_catalogue_entry_ids: frozenset[str],
    ) -> tuple[ProposalBatch, ArtifactRef | None, bool]:
        if remaining_proposals <= 0:
            return (
                self._decline_batch(
                    run=run,
                    round_index=round_index,
                    feedback_ref=feedback_ref,
                    diagnosis_refs=diagnosis_refs,
                    reason=DeclineReason.BUDGET_EXHAUSTED,
                ),
                None,
                False,
            )
        if run.baseline_kind is BaselineKind.EVIDENCE_TARGETED and not diagnosis_refs:
            return (
                self._decline_batch(
                    run=run,
                    round_index=round_index,
                    feedback_ref=feedback_ref,
                    diagnosis_refs=(),
                    reason=DeclineReason.NO_ACTIONABLE_DIAGNOSIS,
                ),
                None,
                False,
            )
        if run.baseline_kind is BaselineKind.RANDOM_VALID:
            catalogue_ref = run.prompt_mutation_catalogue_ref
            assert catalogue_ref is not None
            catalogue = self._load_typed(catalogue_ref, PromptMutationCatalogue)
            target_name = catalogue.entries[0].target_component_name
            champion = self._load_typed(champion_ref, HarnessManifest)
            prompt = next(
                component
                for component in champion.components
                if component.kind is ComponentKind.PROMPT and component.name == target_name
            )
            selection = sample_random_valid(
                catalogue=catalogue,
                catalogue_ref=catalogue_ref,
                policy=policy,
                seed_harness_ref=run.seed_harness_ref,
                parent_harness_ref=champion_ref,
                target_component_name=target_name,
                current_prompt_ref=prompt.artifact,
                strategy_seed=run.strategy_seed,
                round_index=round_index,
                requested_entry_count=min(
                    policy.max_proposals_per_round,
                    remaining_proposals,
                ),
                excluded_entry_ids=tuple(excluded_catalogue_entry_ids),
            )
            selection_ref = self.store.put_json(
                selection,
                media_type=RANDOM_VALID_SELECTION_MEDIA_TYPE,
            )
            proposals = proposals_from_random_selection(selection)
            if not proposals:
                batch = self._decline_batch(
                    run=run,
                    round_index=round_index,
                    feedback_ref=feedback_ref,
                    diagnosis_refs=(),
                    reason=DeclineReason.NO_ELIGIBLE_CATALOGUE_ENTRY,
                )
            else:
                batch = ProposalBatch(
                    baseline_kind=run.baseline_kind,
                    round_index=round_index,
                    source_feedback_ref=feedback_ref,
                    proposals=proposals,
                )
            return batch, selection_ref, False

        if self.plugin is None or artifacts is None:
            raise AutomaticSearchLoopError(
                "optimizer proposal requires a scoped strategy artifact view"
            )
        try:
            raw = self.plugin.propose(
                feedback=feedback,
                feedback_ref=feedback_ref,
                search_run_ref=run_ref,
                round_index=round_index,
                parent_harness_ref=champion_ref,
                diagnosis_refs=diagnosis_refs,
                artifacts=artifacts,
            )
            batch = self._revalidate(ProposalBatch, raw)
        except Exception as exc:
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.PROPOSAL,
                raw=locals().get("raw"),
                code=self._error_code(exc),
                controller=controller,
                tail=tail,
            )
        if len(batch.proposals) > min(policy.max_proposals_per_round, remaining_proposals):
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.PROPOSAL,
                raw=batch,
                code=StrategyOutputRejectionCode.COUNT_LIMIT_EXCEEDED,
                controller=controller,
                tail=tail,
            )
        self._validate_batch_provenance(
            batch=batch,
            run=run,
            run_ref=run_ref,
            round_index=round_index,
            champion_ref=champion_ref,
            feedback_ref=feedback_ref,
            diagnosis_refs=diagnosis_refs,
            controller=controller,
            tail=tail,
            artifacts=artifacts,
        )
        return batch, None, True

    def _validate_batch_provenance(
        self,
        *,
        batch: ProposalBatch,
        run: SearchRunManifest,
        run_ref: ArtifactRef,
        round_index: int,
        champion_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        diagnosis_refs: tuple[ArtifactRef, ...],
        controller: SearchController,
        tail: ArtifactRef,
        artifacts: StrategyArtifactView,
    ) -> None:
        try:
            champion = self._load_typed(champion_ref, HarnessManifest)
            prompts = {
                component.name: component
                for component in champion.components
                if component.kind is ComponentKind.PROMPT
            }
            valid = (
                batch.baseline_kind is run.baseline_kind
                and batch.round_index == round_index
                and batch.source_feedback_ref == feedback_ref
                and batch.diagnosis_refs == diagnosis_refs
            )
            for proposal in batch.proposals:
                current = prompts.get(proposal.target_component_name)
                valid = valid and (
                    proposal.parent_harness_ref == champion_ref
                    and current is not None
                    and proposal.before_prompt_ref == current.artifact
                    and proposal.after_prompt_ref in artifacts.written_prompt_refs
                    and proposal.hypothesis_ref in artifacts.written_hypothesis_refs
                )
                hypothesis = self._load_exact(
                    proposal.hypothesis_ref,
                    MutationHypothesis,
                    MUTATION_HYPOTHESIS_MEDIA_TYPE,
                )
                valid = valid and set(hypothesis.evidence_refs).issubset(artifacts.read_refs)
                if proposal.diagnosis_ref is not None:
                    valid = valid and proposal.diagnosis_ref in diagnosis_refs
        except Exception:
            valid = False
        if not valid:
            self._provenance_failure(
                "proposal batch differs from current feedback, diagnosis, or champion",
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.PROPOSAL,
                raw=batch,
                controller=controller,
                tail=tail,
            )

    def _materialize_proposals(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        round_index: int,
        champion_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposals: tuple[PromptProposal, ...],
        proposal_refs: tuple[ArtifactRef, ...],
        controller: SearchController,
        tail: ArtifactRef,
    ) -> tuple[CandidateMaterialization, ...]:
        materializations: list[CandidateMaterialization] = []
        for proposal, proposal_ref in zip(proposals, proposal_refs, strict=True):
            try:
                raw = self.runtime.materialize_proposal(
                    search_run_ref=run_ref,
                    feedback_ref=feedback_ref,
                    proposal=proposal,
                    proposal_ref=proposal_ref,
                    champion_harness_ref=champion_ref,
                )
                materialization = self._revalidate(CandidateMaterialization, raw)
                if (
                    materialization.search_run_ref != run_ref
                    or materialization.baseline_kind is not run.baseline_kind
                    or materialization.round_index != round_index
                    or materialization.proposal_ref != proposal_ref
                ):
                    raise ValueError("candidate materialization belongs to another round")
                if materialization.candidate_ref is not None:
                    self._join_materialized_candidate(
                        run=run,
                        champion_ref=champion_ref,
                        proposal=proposal,
                        materialization=materialization,
                    )
            except Exception as exc:
                if isinstance(exc, AutomaticSearchLoopError) and exc.rejection_ref is not None:
                    raise
                self._reject_and_raise(
                    run=run,
                    run_ref=run_ref,
                    round_index=round_index,
                    stage=StrategyOutputStage.SCREEN,
                    raw=locals().get("raw"),
                    code=self._error_code(exc),
                    controller=controller,
                    tail=tail,
                )
            materializations.append(materialization)
        candidate_hashes = tuple(
            materialization.candidate_ref.sha256
            for materialization in materializations
            if materialization.candidate_ref is not None
        )
        if len(candidate_hashes) != len(set(candidate_hashes)):
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.SCREEN,
                raw=tuple(materializations),
                code=StrategyOutputRejectionCode.DUPLICATE_OUTPUT,
                controller=controller,
                tail=tail,
            )
        return tuple(materializations)

    def _join_materialized_candidate(
        self,
        *,
        run: SearchRunManifest,
        champion_ref: ArtifactRef,
        proposal: PromptProposal,
        materialization: CandidateMaterialization,
    ) -> None:
        candidate_ref = materialization.candidate_ref
        candidate_harness_ref = materialization.candidate_harness_ref
        if candidate_ref is None or candidate_harness_ref is None:
            raise AutomaticSearchLoopError("materialized candidate pair is missing")
        candidate = self._load_typed(candidate_ref, CandidateManifest)
        if (
            candidate.experiment_ref != run.experiment_ref
            or candidate.parent_harness_ref != champion_ref
            or candidate.child_harness_ref != candidate_harness_ref
        ):
            raise AutomaticSearchLoopError(
                "materialized candidate lineage differs from proposal round"
            )
        try:
            CandidateAdmissionService(self.store).admit(
                candidate_ref=candidate_ref,
                experiment_ref=run.experiment_ref,
            )
        except CandidateAdmissionError as exc:
            raise AutomaticSearchLoopError(
                f"materialized candidate failed trusted admission: {exc}"
            ) from exc
        mutation = self._load_typed(candidate.mutation_ref, CandidateMutation)
        if run.baseline_kind in {
            BaselineKind.PROMPT_ONLY,
            BaselineKind.EVIDENCE_TARGETED,
        }:
            hypothesis = self._load_exact(
                proposal.hypothesis_ref,
                MutationHypothesis,
                MUTATION_HYPOTHESIS_MEDIA_TYPE,
            )
        else:
            hypothesis = self._load_typed(proposal.hypothesis_ref, MutationHypothesis)
        if (
            mutation.target_component != proposal.target_component_name
            or mutation.before.artifact != proposal.before_prompt_ref
            or mutation.after.artifact != proposal.after_prompt_ref
            or mutation.before.kind is not ComponentKind.PROMPT
            or mutation.after.kind is not ComponentKind.PROMPT
            or mutation.hypothesis != hypothesis
        ):
            raise AutomaticSearchLoopError(
                "materialized candidate mutation does not implement its prompt proposal"
            )

    def _screen_materializations(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        policy: SearchPolicy,
        round_index: int,
        champion_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposals: tuple[PromptProposal, ...],
        proposal_refs: tuple[ArtifactRef, ...],
        materializations: tuple[CandidateMaterialization, ...],
        remaining_screens: int,
        controller: SearchController,
        tail: ArtifactRef,
    ) -> tuple[tuple[CandidateScreen, ...], tuple[TrustedScreenEvaluation, ...]]:
        allowed = min(policy.max_candidates_screened_per_round, remaining_screens)
        if len(proposals) > allowed:
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=round_index,
                stage=StrategyOutputStage.SCREEN,
                raw=proposals,
                code=StrategyOutputRejectionCode.COUNT_LIMIT_EXCEEDED,
                controller=controller,
                tail=tail,
            )
        screens: list[CandidateScreen] = []
        evaluations: list[TrustedScreenEvaluation] = []
        joined = zip(proposals, proposal_refs, materializations, strict=True)
        for proposal, proposal_ref, materialization in joined:
            try:
                raw: object | None = None
                if materialization.candidate_ref is None:
                    screen = CandidateScreen(
                        baseline_kind=run.baseline_kind,
                        round_index=round_index,
                        proposal_ref=proposal_ref,
                        status=CandidateScreenStatus.REJECTED,
                        failure_codes=materialization.failure_codes,
                    )
                else:
                    raw = self.runtime.screen_candidate(
                        search_run_ref=run_ref,
                        feedback_ref=feedback_ref,
                        proposal=proposal,
                        proposal_ref=proposal_ref,
                        materialization=materialization,
                        champion_harness_ref=champion_ref,
                    )
                    screen = self._revalidate(CandidateScreen, raw)
                if (
                    screen.baseline_kind is not run.baseline_kind
                    or screen.round_index != round_index
                    or screen.proposal_ref != proposal_ref
                    or screen.candidate_ref != materialization.candidate_ref
                    or screen.candidate_harness_ref != materialization.candidate_harness_ref
                ):
                    raise ValueError("candidate screen belongs to another materialization")
                if screen.evaluation_ref is not None:
                    evaluations.append(
                        self._verify_screen_evaluation(
                            run_ref=run_ref,
                            run=run,
                            round_index=round_index,
                            champion_ref=champion_ref,
                            proposal_ref=proposal_ref,
                            screen=screen,
                        )
                    )
            except Exception as exc:
                if isinstance(exc, AutomaticSearchLoopError) and exc.rejection_ref is not None:
                    raise
                self._reject_and_raise(
                    run=run,
                    run_ref=run_ref,
                    round_index=round_index,
                    stage=StrategyOutputStage.SCREEN,
                    raw=raw,
                    code=self._error_code(exc),
                    controller=controller,
                    tail=tail,
                )
            screens.append(screen)
        return tuple(screens), tuple(evaluations)

    def _verify_screen_evaluation(
        self,
        *,
        run_ref: ArtifactRef,
        run: SearchRunManifest,
        round_index: int,
        champion_ref: ArtifactRef,
        proposal_ref: ArtifactRef,
        screen: CandidateScreen,
    ) -> TrustedScreenEvaluation:
        evaluation_ref = screen.evaluation_ref
        candidate_ref = screen.candidate_ref
        candidate_harness_ref = screen.candidate_harness_ref
        if evaluation_ref is None or candidate_ref is None or candidate_harness_ref is None:
            raise AutomaticSearchLoopError("screen evaluation candidate binding is incomplete")
        exact = self._load_exact
        evaluation = exact(
            evaluation_ref, TrustedScreenEvaluation, TRUSTED_SCREEN_EVALUATION_MEDIA_TYPE
        )
        study = exact(
            run.baseline_study_plan_ref, BaselineStudyPlan, BASELINE_STUDY_PLAN_MEDIA_TYPE
        )
        arm = study.arm(run.baseline_kind)
        benchmark = exact(
            arm.context.benchmark_ref, SearchBenchmarkBinding, SEARCH_BENCHMARK_BINDING_MEDIA_TYPE
        )
        experiment = exact(run.experiment_ref, ExperimentManifest, EXPERIMENT_MANIFEST_MEDIA_TYPE)
        protocol = exact(experiment.protocol_ref, ProtocolManifest, PROTOCOL_MANIFEST_MEDIA_TYPE)
        preflight = exact(
            evaluation.preflight_ref, SchedulePreflightCertificate, SCHEDULE_PREFLIGHT_MEDIA_TYPE
        )
        fingerprint_fields = ("model_fingerprint", "inference_fingerprint", "runtime_fingerprint")
        model_boundary = tuple(getattr(preflight.model_spec, field) for field in fingerprint_fields)
        foreign_spec = preflight.model_spec.fingerprint != protocol.model_spec_fingerprint
        if foreign_spec or any(
            tuple(getattr(frozen, field) for field in fingerprint_fields) != model_boundary
            for frozen in (protocol, arm.context)
        ):
            raise AutomaticSearchLoopError("screen preflight violates frozen model boundary")
        actual_coordinates = (
            evaluation.search_run_ref,
            evaluation.baseline_kind,
            evaluation.round_index,
            evaluation.proposal_ref,
            evaluation.candidate_ref,
            evaluation.parent_harness_ref,
            evaluation.candidate_harness_ref,
        )
        expected_coordinates = (
            run_ref,
            run.baseline_kind,
            round_index,
            proposal_ref,
            candidate_ref,
            champion_ref,
            candidate_harness_ref,
        )
        schedule = evaluation.schedule
        actual_schedule = (
            schedule.study,
            schedule.kind,
            schedule.phase,
            schedule.query,
            schedule.master_seed,
            schedule.parent_harness_id,
            schedule.candidate_harness_id,
            schedule.task_ids,
            schedule.search_runs,
            schedule.repeat_seeds,
        )
        expected_schedule = (
            run.baseline_study_plan_ref.sha256,
            run.baseline_kind.value,
            EvaluationPhase.EXPLORATION,
            round_index,
            run.search_run_seed,
            champion_ref.sha256,
            candidate_harness_ref.sha256,
            benchmark.exploration_task_ids,
            (run.search_run_seed,),
            run.repeat_seeds,
        )
        if actual_coordinates != expected_coordinates or actual_schedule != expected_schedule:
            raise AutomaticSearchLoopError("screen evaluation uses foreign schedule coordinates")
        attempt_ledger = self.runtime.attempt_ledger_for(evaluation_ref)
        replayed_usage = replay_trusted_usage(
            self.store,
            schedule=schedule,
            preflight_ref=evaluation.preflight_ref,
            attempt_ledger=attempt_ledger,
            receipt_refs=evaluation.receipt_refs,
        )
        verified_ledger_state = attempt_ledger.state()
        if verified_ledger_state.tail_ref != evaluation.final_ledger_tail_ref:
            raise AutomaticSearchLoopError("screen ledger changed after usage replay")
        if replayed_usage != evaluation.trusted_usage:
            raise AutomaticSearchLoopError("screen trusted usage differs from receipt replay")
        persisted_envelope = self._load_exact(
            evaluation.objective_aggregate_ref,
            TrustedObjectiveAggregate,
            TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
        )
        persisted_aggregate = persisted_envelope.content
        verified_aggregate = ObjectiveAggregateVerificationCapability.verify(
            self.objective_aggregate_verifier,
            evaluation.objective_aggregate_ref,
        )
        if verified_aggregate != persisted_aggregate:
            raise AutomaticSearchLoopError("trusted grader aggregate changed after persistence")
        actual_aggregate = (
            persisted_aggregate.search_run_ref,
            persisted_aggregate.proposal_ref,
            persisted_aggregate.candidate_ref,
            persisted_aggregate.parent_harness_ref,
            persisted_aggregate.candidate_harness_ref,
            persisted_aggregate.benchmark_binding_ref,
            persisted_aggregate.grader_fingerprint,
            persisted_aggregate.schedule_fingerprint,
            persisted_aggregate.receipt_refs,
        )
        expected_aggregate = (
            run_ref,
            proposal_ref,
            candidate_ref,
            champion_ref,
            candidate_harness_ref,
            arm.context.benchmark_ref,
            protocol.grader_fingerprint,
            schedule.fingerprint,
            evaluation.receipt_refs,
        )
        evaluation_metrics = (
            evaluation.primary_score,
            evaluation.mean_delta,
            evaluation.confidence_lower,
            evaluation.regression_rate,
            evaluation.tokens_ratio,
            evaluation.latency_ratio,
        )
        screen_metrics = (
            screen.primary_score,
            screen.mean_delta,
            screen.confidence_lower,
            screen.regression_rate,
            screen.tokens_ratio,
            screen.latency_ratio,
        )
        aggregate_metrics = (
            persisted_aggregate.primary_score,
            persisted_aggregate.mean_delta,
            persisted_aggregate.confidence_lower,
            persisted_aggregate.regression_rate,
            persisted_aggregate.tokens_ratio,
            persisted_aggregate.latency_ratio,
        )
        if (
            actual_aggregate != expected_aggregate
            or evaluation_metrics != screen_metrics
            or aggregate_metrics != screen_metrics
        ):
            raise AutomaticSearchLoopError(
                "screen scores differ from the trusted grader authorization"
            )
        if attempt_ledger.state() != verified_ledger_state:
            raise AutomaticSearchLoopError("screen ledger changed during verification")
        return evaluation

    def _call_gate(
        self,
        *,
        run_ref: ArtifactRef,
        nomination: Nomination,
        nomination_ref: ArtifactRef,
        tail: ArtifactRef,
        controller: SearchController,
    ) -> ArtifactRef:
        try:
            raw = self.runtime.run_gate(
                search_run_ref=run_ref,
                nomination=nomination,
                nomination_ref=nomination_ref,
                search_tail_ref=tail,
            )
            if not isinstance(raw, ArtifactRef):
                raise TypeError("gate runtime must return an ArtifactRef")
            terminal_ref = ArtifactRef.model_validate(raw)
        except Exception as exc:
            run = self._load_typed(run_ref, SearchRunManifest)
            self._reject_and_raise(
                run=run,
                run_ref=run_ref,
                round_index=nomination.round_index,
                stage=StrategyOutputStage.GATE,
                raw=locals().get("raw"),
                code=self._error_code(exc),
                controller=controller,
                tail=tail,
            )
        return terminal_ref

    def _last_gate_view(
        self,
        *,
        run: SearchRunManifest,
        controller: SearchController,
    ) -> tuple[GateAggregateView | None, ArtifactRef | None]:
        snapshot = self._snapshot(controller)
        disclosure_ref = snapshot.last_feedback_disclosure_ref
        if disclosure_ref is None:
            return None, None
        disclosure = self._load_exact(
            disclosure_ref,
            AggregateFeedbackDisclosure,
            AGGREGATE_FEEDBACK_DISCLOSURE_MEDIA_TYPE,
        )
        summary = disclosure.summary
        expected_decision = {
            CandidateState.PROMOTED: Decision.PROMOTE,
            CandidateState.REJECTED: Decision.REJECT,
            CandidateState.INCONCLUSIVE: Decision.INCONCLUSIVE,
        }[disclosure.terminal_state]
        expected_disclosure_coordinates = (
            controller.controller_manifest_ref,
            controller.manifest.search_run_ref,
            run.baseline_study_plan_ref,
            run.experiment_ref,
            run.baseline_kind,
            run.search_run_seed,
            snapshot.completed_rounds - 1,
            snapshot.champion_harness_ref,
        )
        actual_disclosure_coordinates = (
            disclosure.controller_manifest_ref,
            disclosure.search_run_ref,
            disclosure.study_ref,
            disclosure.experiment_ref,
            disclosure.baseline_kind,
            disclosure.search_seed,
            disclosure.round_index,
            disclosure.champion_harness_ref,
        )
        experiment = self._load_exact(
            run.experiment_ref,
            ExperimentManifest,
            EXPERIMENT_MANIFEST_MEDIA_TYPE,
        )
        protocol = self._load_exact(
            experiment.protocol_ref,
            ProtocolManifest,
            PROTOCOL_MANIFEST_MEDIA_TYPE,
        )
        if (
            actual_disclosure_coordinates != expected_disclosure_coordinates
            or summary.decision is not expected_decision
            or disclosure.gate_version != summary.gate_version
            or disclosure.gate_config_sha256 != summary.gate_config_sha256
            or summary.gate_config_sha256 != protocol.gate_config_ref.sha256
        ):
            raise AutomaticSearchLoopError(
                "controller aggregate disclosure has inconsistent provenance or terminal state"
            )
        metrics: GateAggregateMetrics | None = None
        if summary.mean_score_delta is not None:
            if any(
                value is None
                for value in (
                    summary.confidence_level,
                    summary.confidence_lower_bound,
                    summary.confidence_upper_bound,
                    summary.regression_rate,
                )
            ):
                raise AutomaticSearchLoopError(
                    "controller aggregate disclosure has incomplete metrics"
                )
            metrics = GateAggregateMetrics(
                n_valid_pairs=summary.n_valid_pairs,
                n_tasks=summary.n_tasks,
                mean_delta=summary.mean_score_delta,
                confidence_level=summary.confidence_level,
                confidence_lower=summary.confidence_lower_bound,
                confidence_upper=summary.confidence_upper_bound,
                regression_rate=summary.regression_rate,
                tokens_ratio=summary.tokens_ratio,
                latency_ratio=summary.latency_ratio,
                tool_calls_ratio=summary.tool_calls_ratio,
            )
        passed_check_count = sum(check.outcome is GateCheckOutcome.PASS for check in summary.checks)
        failed_check_count = sum(check.outcome is GateCheckOutcome.FAIL for check in summary.checks)
        inconclusive_check_count = sum(
            check.outcome is GateCheckOutcome.INCONCLUSIVE for check in summary.checks
        )
        resolution: Literal["gate-decision", "superseded-promotion"] = (
            "superseded-promotion"
            if summary.decision is Decision.INCONCLUSIVE
            and failed_check_count == 0
            and inconclusive_check_count == 0
            and passed_check_count > 0
            else "gate-decision"
        )
        view = GateAggregateView(
            candidate_ref=disclosure.candidate_ref,
            analysis_plan_ref=run.analysis_plan_ref,
            query_index=len(snapshot.feedback_disclosure_refs) - 1,
            decision=summary.decision,
            gate_version=summary.gate_version,
            gate_config_sha256=summary.gate_config_sha256,
            resolution=resolution,
            metrics=metrics,
            passed_check_count=passed_check_count,
            failed_check_count=failed_check_count,
            inconclusive_check_count=inconclusive_check_count,
        )
        view_ref = self.store.put_json(view, media_type=GATE_AGGREGATE_VIEW_MEDIA_TYPE)
        return view, view_ref

    @staticmethod
    def _decline_batch(
        *,
        run: SearchRunManifest,
        round_index: int,
        feedback_ref: ArtifactRef,
        diagnosis_refs: tuple[ArtifactRef, ...],
        reason: DeclineReason,
    ) -> ProposalBatch:
        decline = ProposalDecline(
            baseline_kind=run.baseline_kind,
            round_index=round_index,
            source_feedback_ref=feedback_ref,
            reason=reason,
            rationale=f"trusted automatic loop recorded {reason.value}",
            diagnosis_refs=diagnosis_refs,
        )
        return ProposalBatch(
            baseline_kind=run.baseline_kind,
            round_index=round_index,
            source_feedback_ref=feedback_ref,
            diagnosis_refs=diagnosis_refs,
            decline=decline,
        )

    def _reject_and_raise(
        self,
        *,
        run: SearchRunManifest,
        run_ref: ArtifactRef,
        round_index: int,
        stage: StrategyOutputStage,
        raw: object,
        code: StrategyOutputRejectionCode,
        controller: SearchController,
        tail: ArtifactRef,
    ) -> None:
        rejection = StrategyOutputRejection(
            search_run_ref=run_ref,
            baseline_kind=run.baseline_kind,
            round_index=round_index,
            stage=stage,
            code=code,
            received_type=type(raw).__qualname__,
        )
        rejection_ref = self.store.put_json(
            rejection,
            media_type=STRATEGY_OUTPUT_REJECTION_MEDIA_TYPE,
        )
        invalidated_tail = controller.invalidate(
            previous_tail_ref=tail,
            invalidation_ref=rejection_ref,
            reason=f"invalid {stage.value} output rejected",
        )
        raise AutomaticSearchLoopError(
            f"invalid {stage.value} output was archived and rejected",
            rejection_ref=rejection_ref,
            invalidated_tail_ref=invalidated_tail,
        )

    def _provenance_failure(self, message: str, **kwargs: object) -> None:
        self._reject_and_raise(
            **kwargs,  # type: ignore[arg-type]
            code=StrategyOutputRejectionCode.PROVENANCE_MISMATCH,
        )

    @staticmethod
    def _error_code(exc: Exception) -> StrategyOutputRejectionCode:
        if isinstance(exc, TypeError):
            return StrategyOutputRejectionCode.WRONG_PYTHON_TYPE
        if isinstance(exc, ValueError | AttributeError):
            return StrategyOutputRejectionCode.SCHEMA_INVALID
        return StrategyOutputRejectionCode.RUNTIME_EXCEPTION

    @staticmethod
    def _revalidate[ModelT: BaseModel](model_type: type[ModelT], raw: object) -> ModelT:
        if not isinstance(raw, model_type):
            raise TypeError(f"expected typed {model_type.__name__}")
        return model_type.model_validate(
            raw.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )

    def _load_typed[ModelT: BaseModel](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
    ) -> ModelT:
        return self._load_exact(ref, model_type, None)

    def _load_exact[ModelT: BaseModel](
        self,
        ref: ArtifactRef,
        model_type: type[ModelT],
        media_type: str | None,
    ) -> ModelT:
        if media_type is not None and ref.media_type != media_type:
            raise AutomaticSearchLoopError(
                f"{model_type.__name__} artifact declares the wrong media type"
            )
        payload = self.store.get_bytes(ref)
        loaded = self.store.get_json(ref, model_type)
        if payload != canonical_json_bytes(loaded):
            raise AutomaticSearchLoopError(f"{model_type.__name__} artifact is not canonical")
        return loaded

    def _verify_generic_json(self, ref: ArtifactRef, *, label: str) -> None:
        payload = self.store.get_bytes(ref)
        value = self.store.get_json(ref)
        if payload != canonical_json_bytes(value):
            raise AutomaticSearchLoopError(f"{label} artifact is not canonical JSON")

    @staticmethod
    def _snapshot(controller: SearchController) -> SearchRunSnapshot:
        snapshot = controller.snapshot
        if snapshot is None:
            raise AutomaticSearchLoopError("search controller has no replayed snapshot")
        return snapshot


__all__ = [
    "AUTOMATIC_SEARCH_LOOP_RESULT_MEDIA_TYPE",
    "CANDIDATE_SCREEN_BATCH_MEDIA_TYPE",
    "DIAGNOSTIC_TRAJECTORY_MEDIA_TYPE",
    "EXPLORATION_AGGREGATES_MEDIA_TYPE",
    "EXPLORATION_INPUTS_MEDIA_TYPE",
    "EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE",
    "EXPLORATION_TRAJECTORIES_MEDIA_TYPE",
    "SAFE_BENCHMARK_METADATA_MEDIA_TYPE",
    "SEARCH_BENCHMARK_BINDING_MEDIA_TYPE",
    "SEARCH_RUN_ADMISSION_REPORT_MEDIA_TYPE",
    "STRATEGY_OUTPUT_REJECTION_MEDIA_TYPE",
    "TRUSTED_STRATEGY_FEEDBACK_MEDIA_TYPE",
    "AutomaticSearchLoop",
    "AutomaticSearchLoopError",
    "AutomaticSearchLoopExecution",
    "AutomaticSearchLoopResult",
    "AutomaticSearchRuntime",
    "CandidateScreenArchiveEntry",
    "CandidateScreenBatch",
    "ExperimentLifecycleCoordinator",
    "ExplorationTrajectoryIndex",
    "SafeBenchmarkMetadata",
    "SearchBenchmarkBinding",
    "SearchRunAdmissionError",
    "SearchRunAdmissionReport",
    "SearchRunAdmissionService",
    "StrategyFeedbackVerificationCapability",
    "StrategyOutputRejection",
    "StrategyOutputRejectionCode",
    "StrategyOutputStage",
    "StrategyPluginRuntime",
    "TrustedStrategyFeedback",
    "TrustedStrategyFeedbackContent",
    "TrustedStrategyFeedbackService",
]
