"""Live-ledger receipt replay required by trusted aggregate-only SCORE feedback."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.evolution.feedback_views import ScoreResourceTotals
from spiral_harness.evolution.objective_evidence import TrustedObjectiveAggregateContent
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    MODEL_EXECUTION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    ExecutionStatus,
    ModelExecution,
)
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    ExecutionReceiptIntegrityError,
    TrustedExecutionUsage,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    EVALUATION_BATCH_SCHEDULE_MEDIA_TYPE,
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    EvaluationPhase,
    EvaluationSide,
)
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.matched_v2 import MatchedV2RunManifest
from spiral_harness.storage.protocol import ArtifactRepository


class ScoreReceiptClosureError(ValueError):
    """Raised when aggregate receipts do not form one exact matched closure."""


class ScoreReceiptClosure(ImmutableModel):
    """Safe aggregate of replayed receipt facts with item coordinates removed."""

    schema_version: Literal["2"] = "2"
    n_tasks: Annotated[int, Field(gt=0, strict=True)]
    n_valid_pairs: Annotated[int, Field(gt=0, strict=True)]
    query_index: Annotated[int, Field(ge=0, strict=True)]
    schedule_ref: ArtifactRef
    schedule_fingerprint: Sha256
    preflight_ref: ArtifactRef
    trusted_usage_fingerprint: Sha256
    resources: ScoreResourceTotals
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    fingerprint: Sha256


class ScoreReceiptReplayCapability:
    """Private resolver for exact live ledgers behind authenticated objectives.

    The capability is deliberately constructed from already-frozen schedules,
    preflight certificates, and the exact process-local ledger writers.  A
    digest or a caller-authored ledger state is not accepted as a substitute.
    """

    __slots__ = ("__bindings", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("score receipt replay capability cannot be subclassed")

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        bindings: tuple[
            tuple[ArtifactRef, ArtifactRef, AttemptLedger],
            ...,
        ],
    ) -> None:
        if not isinstance(store, ArtifactRepository):
            raise TypeError("store must implement ArtifactRepository")
        if type(bindings) is not tuple or not bindings:
            raise ValueError("score receipt replay requires at least one binding")
        checked: dict[
            str,
            tuple[ArtifactRef, EvaluationBatchSchedule, ArtifactRef, AttemptLedger],
        ] = {}
        for binding in bindings:
            if type(binding) is not tuple or len(binding) != 3:
                raise TypeError("each replay binding must be a schedule/preflight/ledger tuple")
            raw_schedule_ref, raw_preflight_ref, ledger = binding
            schedule_ref = ArtifactRef.model_validate(raw_schedule_ref, strict=True)
            if schedule_ref.media_type != EVALUATION_BATCH_SCHEDULE_MEDIA_TYPE:
                raise ValueError("score replay schedule declares the wrong media type")
            schedule = _load_exact(
                store,
                schedule_ref,
                EvaluationBatchSchedule,
                EVALUATION_BATCH_SCHEDULE_MEDIA_TYPE,
            )
            if schedule_ref.sha256 != schedule.fingerprint:
                raise ValueError("score replay schedule ref differs from its fingerprint")
            preflight_ref = ArtifactRef.model_validate(raw_preflight_ref, strict=True)
            if preflight_ref.media_type != SCHEDULE_PREFLIGHT_MEDIA_TYPE:
                raise ValueError("score replay preflight declares the wrong media type")
            if type(ledger) is not AttemptLedger:
                raise TypeError("score replay ledger must be an exact AttemptLedger")
            if ledger.repository is not store:
                raise ValueError("score replay ledger uses another artifact repository")
            if schedule.fingerprint in checked:
                raise ValueError("score replay schedule fingerprints must be unique")
            checked[schedule.fingerprint] = (
                schedule_ref,
                schedule,
                preflight_ref,
                ledger,
            )
        self.__store = store
        self.__bindings = checked

    def replay(
        self,
        objective: TrustedObjectiveAggregateContent,
    ) -> tuple[ArtifactRef, EvaluationBatchSchedule, ArtifactRef, TrustedExecutionUsage]:
        """Replay the objective's exact batch against its still-live ledger."""

        binding = self.__bindings.get(objective.schedule_fingerprint)
        if binding is None:
            raise ScoreReceiptClosureError("objective schedule has no trusted replay binding")
        schedule_ref, schedule, preflight_ref, ledger = binding
        try:
            usage = replay_trusted_usage(
                self.__store,
                schedule=schedule,
                preflight_ref=preflight_ref,
                attempt_ledger=ledger,
                receipt_refs=objective.receipt_refs,
            )
        except (ExecutionReceiptIntegrityError, TypeError, ValueError) as exc:
            raise ScoreReceiptClosureError("trusted SCORE usage replay failed") from exc
        if usage.schedule_fingerprint != objective.schedule_fingerprint:
            raise ScoreReceiptClosureError("trusted usage belongs to another schedule")
        if {ref.sha256 for ref in usage.receipt_refs} != {
            ref.sha256 for ref in objective.receipt_refs
        }:
            raise ScoreReceiptClosureError("trusted usage differs from objective receipts")
        return schedule_ref, schedule, preflight_ref, usage


def _load_exact[ModelT: BaseModel](
    store: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    media_type: str,
) -> ModelT:
    if ref.media_type != media_type:
        raise ScoreReceiptClosureError(f"{model_type.__name__} has wrong media type")
    try:
        payload = store.get_bytes(ref)
        loaded = store.get_json(ref, model_type)
    except Exception as exc:
        raise ScoreReceiptClosureError(
            f"{model_type.__name__} cannot be loaded as canonical content"
        ) from exc
    if payload != canonical_json_bytes(loaded):
        raise ScoreReceiptClosureError(f"{model_type.__name__} is not canonical")
    return loaded


def verify_score_receipt_closure(
    store: ArtifactRepository,
    *,
    replay_capability: ScoreReceiptReplayCapability,
    objective: TrustedObjectiveAggregateContent,
    run: MatchedV2RunManifest,
    phase: EvaluationPhase,
    allowed_task_ids: frozenset[str],
    expected_query_index: int,
) -> ScoreReceiptClosure:
    """Replay exact schedule, preflight, ledger, pairing, and model boundaries."""

    if type(replay_capability) is not ScoreReceiptReplayCapability:
        raise TypeError("replay_capability must be an exact ScoreReceiptReplayCapability")
    schedule_ref, schedule, preflight_ref, usage = replay_capability.replay(objective)
    expected_tasks = tuple(sorted(allowed_task_ids))
    if not expected_tasks:
        raise ScoreReceiptClosureError("SCORE receipt roster must not be empty")
    expected_coordinates = (
        run.shared.study_id,
        BaselineKind.SCORE_ONLY_MATCHED.value,
        phase,
        expected_query_index,
        run.shared.rollout_master_seed,
        expected_tasks,
        (run.shared.search_run_seed,),
        run.shared.repeat_seeds,
        run.shared.ceilings.max_attempts_per_evaluation,
        run.shared.ceilings.token_ceiling_per_attempt,
    )
    actual_coordinates = (
        schedule.study,
        schedule.kind,
        schedule.phase,
        schedule.query,
        schedule.master_seed,
        schedule.task_ids,
        schedule.search_runs,
        schedule.repeat_seeds,
        schedule.max_attempts_per_cell,
        schedule.token_ceiling_per_attempt,
    )
    if actual_coordinates != expected_coordinates:
        raise ScoreReceiptClosureError("SCORE schedule differs from frozen run coordinates")
    if schedule.cell_count > run.shared.ceilings.max_evaluations:
        raise ScoreReceiptClosureError("SCORE schedule exceeds the evaluation ceiling")
    if (
        schedule.parent_harness_id != objective.parent_harness_ref.sha256
        or schedule.candidate_harness_id != objective.candidate_harness_ref.sha256
    ):
        raise ScoreReceiptClosureError("SCORE schedule differs from objective harness roles")

    attempts: dict[
        str,
        dict[EvaluationSide, list[tuple[int, ModelExecution, AttemptOutcome]]],
    ] = defaultdict(lambda: defaultdict(list))
    receipt_refs: list[ArtifactRef] = []
    total_tokens = 0
    total_reported_tokens = 0
    total_latency = 0.0
    total_tool_calls = 0
    failed_calls = 0
    retry_count = 0
    costs: list[float | None] = []
    side_tokens = {side: 0 for side in EvaluationSide}
    side_latency = {side: 0.0 for side in EvaluationSide}

    for receipt_ref in objective.receipt_refs:
        receipt = _load_exact(
            store,
            receipt_ref,
            ExecutionReceipt,
            EXECUTION_RECEIPT_MEDIA_TYPE,
        )
        outcome = _load_exact(
            store,
            receipt.outcome_ref,
            AttemptOutcome,
            ATTEMPT_OUTCOME_MEDIA_TYPE,
        )
        execution = _load_exact(
            store,
            receipt.execution_ref,
            ModelExecution,
            MODEL_EXECUTION_MEDIA_TYPE,
        )
        if receipt.preflight_ref != preflight_ref:
            raise ScoreReceiptClosureError("receipt uses another preflight boundary")
        expected_harness = (
            objective.parent_harness_ref
            if receipt.cell.side is EvaluationSide.PARENT
            else objective.candidate_harness_ref
        )
        if execution.request.harness_ref != expected_harness:
            raise ScoreReceiptClosureError("receipt execution uses a foreign harness")
        if (
            execution.model_fingerprint != run.shared.model_fingerprint
            or execution.inference_fingerprint != run.shared.inference_fingerprint
            or execution.runtime_fingerprint != run.shared.runtime_fingerprint
        ):
            raise ScoreReceiptClosureError("receipt execution crosses the model boundary")
        attempts[receipt.cell.pairing_fingerprint][receipt.cell.side].append(
            (receipt.attempt_index, execution, outcome)
        )
        receipt_refs.append(receipt_ref)
        total_tokens += receipt.charged_tokens
        total_reported_tokens += receipt.reported_tokens
        total_latency += execution.latency_ms
        total_tool_calls += execution.tool_calls
        side_tokens[receipt.cell.side] += receipt.charged_tokens
        side_latency[receipt.cell.side] += execution.latency_ms
        failed_calls += execution.status is ExecutionStatus.FAILED
        retry_count += receipt.attempt_index > 0
        costs.append(execution.cost_usd)

    valid_pairs = 0
    for sides in attempts.values():
        if set(sides) != {EvaluationSide.PARENT, EvaluationSide.CANDIDATE}:
            raise ScoreReceiptClosureError("receipt closure is missing a paired side")
        terminals: list[tuple[int, ModelExecution, AttemptOutcome]] = []
        for side in (EvaluationSide.PARENT, EvaluationSide.CANDIDATE):
            ordered = sorted(sides[side], key=lambda item: item[0])
            terminals.append(ordered[-1])
        if not all(
            execution.status is ExecutionStatus.COMPLETED
            and outcome.disposition is AttemptDisposition.SETTLED
            for _, execution, outcome in terminals
        ):
            raise ScoreReceiptClosureError("every scheduled SCORE pair must settle successfully")
        valid_pairs += 1

    expected_pair_count = (
        len(schedule.task_ids) * len(schedule.search_runs) * len(schedule.repeat_seeds)
    )
    if valid_pairs != expected_pair_count:
        raise ScoreReceiptClosureError("receipt closure does not cover every scheduled pair")
    interval = objective.confidence_interval
    if interval.n_tasks != len(schedule.task_ids) or interval.n_valid_pairs != valid_pairs:
        raise ScoreReceiptClosureError("objective statistical counts differ from receipt replay")
    if (
        usage.attempt_count != len(receipt_refs)
        or usage.charged_tokens != total_tokens
        or usage.reported_tokens != total_reported_tokens
        or usage.poisoned_attempts != 0
    ):
        raise ScoreReceiptClosureError("objective resources differ from trusted usage replay")

    parent_tokens = side_tokens[EvaluationSide.PARENT]
    parent_latency = side_latency[EvaluationSide.PARENT]
    if parent_tokens <= 0 or parent_latency <= 0:
        raise ScoreReceiptClosureError("parent resource denominator must be positive")
    tokens_ratio = side_tokens[EvaluationSide.CANDIDATE] / parent_tokens
    latency_ratio = side_latency[EvaluationSide.CANDIDATE] / parent_latency
    total_cost = (
        None
        if any(value is None for value in costs)
        else sum(value for value in costs if value is not None)
    )
    resources = ScoreResourceTotals(
        total_tokens=total_tokens,
        total_latency_ms=total_latency,
        total_tool_calls=total_tool_calls,
        total_model_calls=len(receipt_refs),
        failed_model_calls=failed_calls,
        retry_count=retry_count,
        total_cost_usd=total_cost,
    )
    fingerprint = canonical_sha256(
        {
            "schema": "spiral-harness/score-receipt-closure/v2",
            "phase": phase,
            "query_index": expected_query_index,
            "schedule_fingerprint": schedule.fingerprint,
            "schedule_ref": schedule_ref,
            "preflight_ref": preflight_ref,
            "trusted_usage_fingerprint": usage.fingerprint,
            "receipt_refs": tuple(sorted(receipt_refs, key=lambda ref: ref.sha256)),
            "n_tasks": len(schedule.task_ids),
            "n_valid_pairs": valid_pairs,
            "resources": resources,
            "tokens_ratio": tokens_ratio,
            "latency_ratio": latency_ratio,
        }
    )
    return ScoreReceiptClosure(
        n_tasks=len(schedule.task_ids),
        n_valid_pairs=valid_pairs,
        query_index=expected_query_index,
        schedule_ref=schedule_ref,
        schedule_fingerprint=schedule.fingerprint,
        preflight_ref=preflight_ref,
        trusted_usage_fingerprint=usage.fingerprint,
        resources=resources,
        tokens_ratio=tokens_ratio,
        latency_ratio=latency_ratio,
        fingerprint=fingerprint,
    )


__all__ = [
    "ScoreReceiptClosure",
    "ScoreReceiptClosureError",
    "ScoreReceiptReplayCapability",
    "verify_score_receipt_closure",
]
