"""Project authenticated SCORE-only feedback without disclosing private
receipts, gate blocks, HMAC material, or full mechanism-gate outcomes.
"""

from __future__ import annotations

import hmac
import math
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    ArtifactRef,
    ImmutableModel,
    Sha256,
)
from spiral_harness.evolution.feedback_media_types import (
    EXPLORATION_INPUTS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
)
from spiral_harness.evolution.feedback_views import (
    ScoreAggregateView,
    ScoreGateDecisionView,
    ScoreOnlyFeedbackView,
)
from spiral_harness.evolution.objective_evidence import (
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    ObjectiveAggregateVerificationCapability,
    TrustedObjectiveAggregate,
    TrustedObjectiveAggregateContent,
)
from spiral_harness.evolution.orchestrator import SafeBenchmarkMetadata, SearchBenchmarkBinding
from spiral_harness.evolution.score_candidate_lineage import (
    ScoreCandidateLineageError,
    verify_score_candidate_lineage,
)
from spiral_harness.evolution.score_receipt_closure import (
    ScoreReceiptClosure,
    ScoreReceiptClosureError,
    ScoreReceiptReplayCapability,
    verify_score_receipt_closure,
)
from spiral_harness.execution.schedule import EvaluationPhase
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.matched_v2 import (
    MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
    MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    MatchedV2GateQueryBlock,
    MatchedV2RunManifest,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.models import Decision

ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE = (
    "application/vnd.spiral-harness.attested-score-feedback.v2+json"
)

NormalizedDelta = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
PositiveRatio = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ScoreFeedbackAttestationError(ValueError):
    """Raised when a SCORE projection cannot close over trusted sources."""


class ScoreExplorationInputs(ImmutableModel):
    """The only accepted exploration-input role content."""

    partition: Literal["exploration"] = "exploration"
    task_ids: Annotated[tuple[str, ...], Field(min_length=1)]

    @field_validator("task_ids")
    @classmethod
    def _canonicalize_task_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("exploration task_ids must be exact and non-empty")
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("exploration task_ids must not contain duplicates")
        return ordered


class ScorePerformancePolicy(ImmutableModel):
    """Deterministic performance-only policy, separate from mechanism gates."""

    schema_version: Literal["2"] = "2"
    policy_version: Literal["utility-protected-cost-v2"] = "utility-protected-cost-v2"
    interval_estimator_version: Literal["paired-task-percentile-bootstrap-v1"] = (
        "paired-task-percentile-bootstrap-v1"
    )
    interval_statistical_unit: Literal["paired-task-mean-delta"] = "paired-task-mean-delta"
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    bootstrap_samples: Annotated[int, Field(ge=1_000, strict=True)]
    minimum_mean_delta: NormalizedDelta
    minimum_confidence_lower: NormalizedDelta
    maximum_regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_tokens_ratio: PositiveRatio
    maximum_latency_ratio: PositiveRatio
    reject_on_protected_or_cost_violation: Literal[True] = True
    implementation: Literal["deterministic-performance-projector-v2"] = (
        "deterministic-performance-projector-v2"
    )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class ScoreFeedbackProjectionRequest(ImmutableModel):
    """Trusted projector input; no caller-supplied gate decision is representable."""

    schema_version: Literal["2"] = "2"
    matched_run_ref: ArtifactRef
    exploration_objective_aggregate_ref: ArtifactRef
    gate_objective_aggregate_ref: ArtifactRef
    gate_query_block_ref: ArtifactRef
    round_index: Annotated[int, Field(gt=0, strict=True)]
    prior_parent_attestation_ref: ArtifactRef | None = None
    performance_policy: ScorePerformancePolicy

    @model_validator(mode="after")
    def _source_refs_have_exact_media_types(self) -> Self:
        expected = (
            (self.matched_run_ref, MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE),
            (
                self.exploration_objective_aggregate_ref,
                TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
            ),
            (self.gate_objective_aggregate_ref, TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE),
            (self.gate_query_block_ref, MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE),
        )
        if any(ref.media_type != media_type for ref, media_type in expected):
            raise ValueError("score projection source declares the wrong media type")
        if (
            self.exploration_objective_aggregate_ref.sha256
            == self.gate_objective_aggregate_ref.sha256
        ):
            raise ValueError("exploration and gate objectives must be distinct artifacts")
        if self.round_index == 1:
            if self.prior_parent_attestation_ref is not None:
                raise ValueError("the first SCORE round cannot cite a prior parent attestation")
        elif self.prior_parent_attestation_ref is None:
            raise ValueError("later SCORE rounds require a prior parent attestation")
        elif self.prior_parent_attestation_ref.media_type != ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE:
            raise ValueError("prior parent attestation declares the wrong media type")
        return self


class AttestedScoreFeedbackContent(ImmutableModel):
    """Private verifier closure; this object is not an optimizer disclosure."""

    schema_version: Literal["2"] = "2"
    request: ScoreFeedbackProjectionRequest
    view: ScoreOnlyFeedbackView
    input_parent_harness_sha256: Sha256
    output_champion_harness_sha256: Sha256
    exploration_receipt_closure_fingerprint: Sha256
    gate_receipt_closure_fingerprint: Sha256
    source_role_binding_attested: Literal[True] = True
    performance_projection_attested: Literal[True] = True
    performance_source: Literal["objective-aggregate-and-live-ledger-replay-v2"] = (
        "objective-aggregate-and-live-ledger-replay-v2"
    )
    full_mechanism_gate_outcome_consumed: Literal[False] = False

    @model_validator(mode="after")
    def _preserve_phase_one_claim_boundary(self) -> Self:
        if self.view.runtime_role_binding_attested is not False:
            raise ValueError("nested phase-one role flag must remain false")
        if self.view.performance_projection_attested is not False:
            raise ValueError("nested phase-one projection flag must remain false")
        if self.view.round_index != self.request.round_index:
            raise ValueError("feedback round differs from projection request")
        return self


class AttestedScoreFeedbackEnvelope(ImmutableModel):
    """HMAC-authenticated private closure persisted by the trusted projector."""

    schema_version: Literal["2"] = "2"
    content: AttestedScoreFeedbackContent
    attestor_id: Sha256
    authentication_tag: Sha256


class AttestedScoreFeedback(ImmutableModel):
    """Safe optimizer-facing wrapper returned only after exact verification."""

    schema_version: Literal["2"] = "2"
    attestation_ref: ArtifactRef
    attestor_id: Sha256
    matched_run_sha256: Sha256
    champion_harness_sha256: Sha256
    view: ScoreOnlyFeedbackView
    source_role_binding_attested: Literal[True] = True
    performance_projection_attested: Literal[True] = True
    full_mechanism_gate_outcome_consumed: Literal[False] = False

    @model_validator(mode="after")
    def _attestation_ref_is_exact(self) -> Self:
        if self.attestation_ref.media_type != ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE:
            raise ValueError("attestation_ref declares the wrong media type")
        if self.view.runtime_role_binding_attested is not False:
            raise ValueError("nested phase-one role flag must remain false")
        if self.view.performance_projection_attested is not False:
            raise ValueError("nested phase-one projection flag must remain false")
        return self


def _load_exact[ModelT: BaseModel](
    store: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
) -> ModelT:
    if ref.media_type != media_type:
        raise ScoreFeedbackAttestationError(f"{model_type.__name__} has wrong media type")
    try:
        payload = store.get_bytes(ref)
        loaded = store.get_json(ref, model_type)
    except Exception as exc:
        raise ScoreFeedbackAttestationError(
            f"{model_type.__name__} cannot be loaded as canonical content"
        ) from exc
    if payload != canonical_json_bytes(loaded):
        raise ScoreFeedbackAttestationError(f"{model_type.__name__} is not canonical")
    return loaded


def _load_objective(
    store: ArtifactRepository,
    verifier: ObjectiveAggregateVerificationCapability,
    ref: ArtifactRef,
) -> TrustedObjectiveAggregateContent:
    persisted = _load_exact(
        store,
        ref,
        TrustedObjectiveAggregate,
        TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
    )
    try:
        verified = ObjectiveAggregateVerificationCapability.verify(verifier, ref)
    except Exception as exc:
        raise ScoreFeedbackAttestationError("objective aggregate authentication failed") from exc
    if verified != persisted.content:
        raise ScoreFeedbackAttestationError("objective aggregate changed after verification")
    return verified


def _performance_decision(
    aggregate: TrustedObjectiveAggregateContent,
    *,
    policy: ScorePerformancePolicy,
) -> Decision:
    utility_passed = (
        aggregate.mean_delta >= policy.minimum_mean_delta
        and aggregate.confidence_lower >= policy.minimum_confidence_lower
    )
    protected_passed = aggregate.regression_rate <= policy.maximum_regression_rate
    cost_passed = (
        aggregate.tokens_ratio <= policy.maximum_tokens_ratio
        and aggregate.latency_ratio <= policy.maximum_latency_ratio
    )
    if utility_passed and protected_passed and cost_passed:
        return Decision.PROMOTE
    if (
        not protected_passed
        or not cost_passed
        or aggregate.confidence_upper < policy.minimum_mean_delta
    ):
        return Decision.REJECT
    return Decision.INCONCLUSIVE


def _aggregate_view(
    objective: TrustedObjectiveAggregateContent,
    closure: ScoreReceiptClosure,
) -> ScoreAggregateView:
    return ScoreAggregateView(
        candidate_sha256=objective.candidate_ref.sha256,
        n_valid_pairs=closure.n_valid_pairs,
        n_tasks=closure.n_tasks,
        parent_score_mean=objective.primary_score - objective.mean_delta,
        candidate_score_mean=objective.primary_score,
        mean_delta=objective.mean_delta,
        confidence_level=objective.confidence_level,
        confidence_lower=objective.confidence_lower,
        confidence_upper=objective.confidence_upper,
        resources=closure.resources,
    )


def _project_content(
    store: ArtifactRepository,
    objective_verifier: ObjectiveAggregateVerificationCapability,
    receipt_replay_capability: ScoreReceiptReplayCapability,
    request: ScoreFeedbackProjectionRequest,
    *,
    authorized_parent_harness_sha256: Sha256,
) -> AttestedScoreFeedbackContent:
    run = _load_exact(
        store,
        request.matched_run_ref,
        MatchedV2RunManifest,
        MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    )
    if run.baseline_kind is not BaselineKind.SCORE_ONLY_MATCHED:
        raise ScoreFeedbackAttestationError("score projector requires the SCORE run")
    if request.performance_policy.fingerprint != (
        run.shared.policies.performance_policy_fingerprint
    ):
        raise ScoreFeedbackAttestationError("performance policy differs from matched run")
    benchmark = _load_exact(
        store,
        run.shared.benchmark_binding_ref,
        SearchBenchmarkBinding,
        SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )
    if benchmark.objective_aggregate_attestor_id != objective_verifier.attestor_id:
        raise ScoreFeedbackAttestationError("benchmark binds another objective attestor")
    metadata = _load_exact(
        store,
        benchmark.safe_benchmark_metadata_ref,
        SafeBenchmarkMetadata,
        SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    inputs = _load_exact(
        store,
        benchmark.exploration_inputs_ref,
        ScoreExplorationInputs,
        EXPLORATION_INPUTS_MEDIA_TYPE,
    )
    if (
        metadata.benchmark_fingerprint != benchmark.benchmark_fingerprint
        or metadata.exploration_task_ids != benchmark.exploration_task_ids
        or inputs.task_ids != benchmark.exploration_task_ids
    ):
        raise ScoreFeedbackAttestationError("safe SCORE roles differ from benchmark binding")

    block_refs = run.shared.gate_query_block_refs
    if request.gate_query_block_ref not in block_refs:
        raise ScoreFeedbackAttestationError("gate query block is outside the matched run")
    block = _load_exact(
        store,
        request.gate_query_block_ref,
        MatchedV2GateQueryBlock,
        MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
    )
    if block.query_index >= len(block_refs):
        raise ScoreFeedbackAttestationError("gate query block index exceeds the matched run")
    if block_refs[block.query_index] != request.gate_query_block_ref:
        raise ScoreFeedbackAttestationError("gate query block order differs from the run")
    if request.round_index != block.query_index + 1:
        raise ScoreFeedbackAttestationError("gate query must immediately precede feedback round")

    exploration_objective = _load_objective(
        store,
        objective_verifier,
        request.exploration_objective_aggregate_ref,
    )
    gate_objective = _load_objective(
        store,
        objective_verifier,
        request.gate_objective_aggregate_ref,
    )
    if (
        exploration_objective.benchmark_binding_ref != run.shared.benchmark_binding_ref
        or gate_objective.benchmark_binding_ref != run.shared.benchmark_binding_ref
    ):
        raise ScoreFeedbackAttestationError("objective aggregate uses another benchmark binding")
    if (
        exploration_objective.grader_fingerprint != run.shared.policies.grader_fingerprint
        or gate_objective.grader_fingerprint != run.shared.policies.grader_fingerprint
    ):
        raise ScoreFeedbackAttestationError("objective aggregate uses another grader")
    shared_objective_coordinates = (
        exploration_objective.search_run_ref,
        exploration_objective.proposal_ref,
        exploration_objective.candidate_ref,
        exploration_objective.parent_harness_ref,
        exploration_objective.candidate_harness_ref,
    )
    gate_objective_coordinates = (
        gate_objective.search_run_ref,
        gate_objective.proposal_ref,
        gate_objective.candidate_ref,
        gate_objective.parent_harness_ref,
        gate_objective.candidate_harness_ref,
    )
    if shared_objective_coordinates != gate_objective_coordinates:
        raise ScoreFeedbackAttestationError(
            "exploration and gate objectives refer to different candidate coordinates"
        )
    if exploration_objective.search_run_ref != request.matched_run_ref:
        raise ScoreFeedbackAttestationError(
            "objective run ref differs from the matched SCORE manifest"
        )
    if exploration_objective.parent_harness_ref.sha256 != authorized_parent_harness_sha256:
        raise ScoreFeedbackAttestationError(
            "objective parent differs from the attested champion lineage"
        )
    try:
        verify_score_candidate_lineage(
            store,
            matched_run_ref=request.matched_run_ref,
            run=run,
            block=block,
            objective=exploration_objective,
        )
    except ScoreCandidateLineageError as exc:
        raise ScoreFeedbackAttestationError(str(exc)) from exc
    for objective in (exploration_objective, gate_objective):
        interval = objective.confidence_interval
        if (
            interval.estimator_version != request.performance_policy.interval_estimator_version
            or interval.statistical_unit != request.performance_policy.interval_statistical_unit
            or interval.confidence_level != request.performance_policy.confidence_level
            or interval.bootstrap_samples != request.performance_policy.bootstrap_samples
        ):
            raise ScoreFeedbackAttestationError(
                "objective interval differs from the frozen performance policy"
            )
    try:
        exploration_closure = verify_score_receipt_closure(
            store,
            replay_capability=receipt_replay_capability,
            objective=exploration_objective,
            run=run,
            phase=EvaluationPhase.EXPLORATION,
            allowed_task_ids=frozenset(benchmark.exploration_task_ids),
            expected_query_index=request.round_index,
        )
        gate_closure = verify_score_receipt_closure(
            store,
            replay_capability=receipt_replay_capability,
            objective=gate_objective,
            run=run,
            phase=EvaluationPhase.GATE,
            allowed_task_ids=frozenset(task.task_id for task in block.tasks),
            expected_query_index=block.query_index,
        )
    except ScoreReceiptClosureError as exc:
        raise ScoreFeedbackAttestationError("objective receipt closure is invalid") from exc
    for objective, closure in (
        (exploration_objective, exploration_closure),
        (gate_objective, gate_closure),
    ):
        if not math.isclose(
            objective.tokens_ratio,
            closure.tokens_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            objective.latency_ratio,
            closure.latency_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScoreFeedbackAttestationError(
                "objective resource ratios differ from receipt closure"
            )
    exploration_view = _aggregate_view(
        exploration_objective,
        exploration_closure,
    )
    gate_view = _aggregate_view(
        gate_objective,
        gate_closure,
    )
    decision = ScoreGateDecisionView(
        candidate_sha256=gate_view.candidate_sha256,
        query_index=block.query_index,
        decision=_performance_decision(
            gate_objective,
            policy=request.performance_policy,
        ),
        performance_policy_version=request.performance_policy.policy_version,
        performance_policy_config_sha256=request.performance_policy.fingerprint,
        aggregate=gate_view,
    )
    view = ScoreOnlyFeedbackView(
        round_index=request.round_index,
        benchmark_metadata_ref=benchmark.safe_benchmark_metadata_ref,
        exploration_inputs_ref=benchmark.exploration_inputs_ref,
        exploration_aggregate=exploration_view,
        prior_gate_decision=decision,
    )
    output_champion_harness_sha256 = (
        gate_objective.candidate_harness_ref.sha256
        if decision.decision is Decision.PROMOTE
        else gate_objective.parent_harness_ref.sha256
    )
    return AttestedScoreFeedbackContent(
        request=request,
        view=view,
        input_parent_harness_sha256=gate_objective.parent_harness_ref.sha256,
        output_champion_harness_sha256=output_champion_harness_sha256,
        exploration_receipt_closure_fingerprint=exploration_closure.fingerprint,
        gate_receipt_closure_fingerprint=gate_closure.fingerprint,
    )


class ScoreFeedbackVerificationCapability:
    """Exact verify-only capability for trusted SCORE feedback envelopes."""

    __slots__ = (
        "__attestor_id",
        "__objective_verifier",
        "__receipt_replay_capability",
        "__secret",
        "__store",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("score feedback verification capability cannot be subclassed")

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        secret: bytes,
        objective_verifier: ObjectiveAggregateVerificationCapability,
        receipt_replay_capability: ScoreReceiptReplayCapability,
    ) -> None:
        if type(secret) is not bytes or len(secret) < 32:
            raise ValueError("score feedback attestor secret must contain at least 32 bytes")
        if type(objective_verifier) is not ObjectiveAggregateVerificationCapability:
            raise TypeError("objective_verifier must be an exact verification capability")
        if type(receipt_replay_capability) is not ScoreReceiptReplayCapability:
            raise TypeError(
                "receipt_replay_capability must be an exact ScoreReceiptReplayCapability"
            )
        self.__store = store
        self.__secret = secret
        self.__objective_verifier = objective_verifier
        self.__receipt_replay_capability = receipt_replay_capability
        self.__attestor_id = sha256_bytes(b"spiral-harness/score-feedback-attestor/v2\x00" + secret)

    @property
    def attestor_id(self) -> str:
        return self.__attestor_id

    def authorized_parent_harness_sha256(
        self,
        request: ScoreFeedbackProjectionRequest,
    ) -> Sha256:
        """Resolve the first seed or the immediately prior attested champion."""

        checked = ScoreFeedbackProjectionRequest.model_validate(request, strict=True)
        run = _load_exact(
            self.__store,
            checked.matched_run_ref,
            MatchedV2RunManifest,
            MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
        )
        if checked.round_index == 1:
            return run.shared.seed_harness_ref.sha256
        prior_ref = checked.prior_parent_attestation_ref
        if prior_ref is None:  # pragma: no cover - model invariant
            raise ScoreFeedbackAttestationError("later SCORE round lacks prior attestation")
        prior = self.verify(prior_ref)
        if (
            prior.matched_run_sha256 != checked.matched_run_ref.sha256
            or prior.view.round_index != checked.round_index - 1
        ):
            raise ScoreFeedbackAttestationError(
                "prior parent attestation is not the immediately preceding SCORE round"
            )
        return prior.champion_harness_sha256

    def verify(self, attestation_ref: ArtifactRef) -> AttestedScoreFeedback:
        envelope = _load_exact(
            self.__store,
            attestation_ref,
            AttestedScoreFeedbackEnvelope,
            ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE,
        )
        if envelope.attestor_id != self.__attestor_id:
            raise ScoreFeedbackAttestationError("score feedback uses another attestor")
        authentication = hmac.new(
            self.__secret,
            b"spiral-harness/score-feedback/v2\x00",
            sha256,
        )
        authentication.update(self.__attestor_id.encode("ascii") + b"\x00")
        authentication.update(canonical_json_bytes(envelope.content))
        if not hmac.compare_digest(envelope.authentication_tag, authentication.hexdigest()):
            raise ScoreFeedbackAttestationError("score feedback authentication failed")
        authorized_parent = self.authorized_parent_harness_sha256(envelope.content.request)
        projected = _project_content(
            self.__store,
            self.__objective_verifier,
            self.__receipt_replay_capability,
            envelope.content.request,
            authorized_parent_harness_sha256=authorized_parent,
        )
        if projected != envelope.content:
            raise ScoreFeedbackAttestationError("score feedback differs from trusted projection")
        return AttestedScoreFeedback(
            attestation_ref=attestation_ref,
            attestor_id=self.__attestor_id,
            matched_run_sha256=projected.request.matched_run_ref.sha256,
            champion_harness_sha256=projected.output_champion_harness_sha256,
            view=projected.view,
        )


class ScoreFeedbackAttestationService:
    """Trusted producer kept outside the optimizer-visible runtime."""

    __slots__ = (
        "__capability",
        "__objective_verifier",
        "__receipt_replay_capability",
        "__secret",
        "__store",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("score feedback attestation service cannot be subclassed")

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        secret: bytes,
        objective_verifier: ObjectiveAggregateVerificationCapability,
        receipt_replay_capability: ScoreReceiptReplayCapability,
    ) -> None:
        self.__store = store
        self.__secret = secret
        self.__objective_verifier = objective_verifier
        if type(receipt_replay_capability) is not ScoreReceiptReplayCapability:
            raise TypeError(
                "receipt_replay_capability must be an exact ScoreReceiptReplayCapability"
            )
        self.__receipt_replay_capability = receipt_replay_capability
        self.__capability = ScoreFeedbackVerificationCapability(
            store,
            secret=secret,
            objective_verifier=objective_verifier,
            receipt_replay_capability=receipt_replay_capability,
        )

    @property
    def verification_capability(self) -> ScoreFeedbackVerificationCapability:
        return self.__capability

    def attest(self, request: ScoreFeedbackProjectionRequest) -> ArtifactRef:
        if type(request) is not ScoreFeedbackProjectionRequest:
            raise TypeError("request must be an exact ScoreFeedbackProjectionRequest")
        checked = ScoreFeedbackProjectionRequest.model_validate(request, strict=True)
        authorized_parent = self.__capability.authorized_parent_harness_sha256(checked)
        content = _project_content(
            self.__store,
            self.__objective_verifier,
            self.__receipt_replay_capability,
            checked,
            authorized_parent_harness_sha256=authorized_parent,
        )
        authentication = hmac.new(
            self.__secret,
            b"spiral-harness/score-feedback/v2\x00",
            sha256,
        )
        authentication.update(self.__capability.attestor_id.encode("ascii") + b"\x00")
        authentication.update(canonical_json_bytes(content))
        envelope = AttestedScoreFeedbackEnvelope(
            content=content,
            attestor_id=self.__capability.attestor_id,
            authentication_tag=authentication.hexdigest(),
        )
        ref = self.__store.put_json(envelope, media_type=ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE)
        verified = self.__capability.verify(ref)
        if (
            verified.view != content.view
            or verified.champion_harness_sha256 != content.output_champion_harness_sha256
        ):
            raise ScoreFeedbackAttestationError(
                "published score feedback changed during closing verification"
            )
        return ref


__all__ = [
    "ATTESTED_SCORE_FEEDBACK_MEDIA_TYPE",
    "AttestedScoreFeedback",
    "AttestedScoreFeedbackContent",
    "AttestedScoreFeedbackEnvelope",
    "ScoreExplorationInputs",
    "ScoreFeedbackAttestationError",
    "ScoreFeedbackAttestationService",
    "ScoreFeedbackProjectionRequest",
    "ScoreFeedbackVerificationCapability",
    "ScorePerformancePolicy",
]
