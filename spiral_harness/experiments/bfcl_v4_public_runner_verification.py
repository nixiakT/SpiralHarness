"""Offline verification for a published BFCL V4 public-runner result."""

from __future__ import annotations

from dataclasses import dataclass

from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    adapt_bfcl_v4_public_pilot_task,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PublicPilotTask,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotCallKind,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
    BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4CallOutcome,
    BfclV4RunClosure,
)
from spiral_harness.benchmark.bfcl_v4_public_run_journal import (
    verify_bfcl_v4_public_run_closure,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    ATTEMPT_RESERVATION_MEDIA_TYPE,
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    ExecutionStatus,
)
from spiral_harness.execution.native_function_contracts import (
    NativeFunctionExecution,
    load_canonical_native_artifact,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_meta_native import (
    materialize_bfcl_v4_public_meta_native_request,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
    BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE,
    BfclV4PublicPilotRunResult,
    BfclV4PublicPilotRunVerification,
    BfclV4RunnerHarnessArtifact,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    load_canonical_model,
    materialize_solver_request,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4HoldoutEvidence,
    BfclV4JointSelectionDecision,
    BfclV4JointSelectionEvidence,
    BfclV4PublicDescriptiveMetrics,
)
from spiral_harness.storage.protocol import ArtifactRepository


@dataclass(frozen=True, slots=True)
class _ObservedBackendIdentity:
    fingerprint: str
    serializer_fingerprint: str
    parser_fingerprint: str
    transport_fingerprint: str


def _attempt_pair(
    repository: ArtifactRepository,
    *,
    outcome_ref: ArtifactRef,
) -> tuple[AttemptOutcome, AttemptReservation]:
    outcome = load_canonical_model(
        repository,
        outcome_ref,
        AttemptOutcome,
        media_type=ATTEMPT_OUTCOME_MEDIA_TYPE,
    )
    reservation = load_canonical_model(
        repository,
        outcome.reservation_ref,
        AttemptReservation,
        media_type=ATTEMPT_RESERVATION_MEDIA_TYPE,
    )
    return outcome, reservation


def _require_nonempty(repository: ArtifactRepository, ref: ArtifactRef, label: str) -> None:
    try:
        payload = repository.get_bytes(ref)
    except Exception as exc:
        raise BfclV4PublicRunnerError(f"{label} artifact is absent") from exc
    if not payload or ref.size != len(payload):
        raise BfclV4PublicRunnerError(f"{label} artifact is empty or has the wrong size")


def verify_bfcl_v4_public_pilot_result(
    repository: ArtifactRepository,
    result_ref: ArtifactRef,
) -> BfclV4PublicPilotRunVerification:
    """Rejoin all 100 request/execution/accounting/completion lineages offline."""

    result = load_canonical_model(
        repository,
        result_ref,
        BfclV4PublicPilotRunResult,
        media_type=BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    )
    plan = build_bfcl_v4_public_pilot_call_plan(result.outer_seed_u64)
    if (
        result.plan_fingerprint != plan.fingerprint
        or result.schedule_content_sha256 != plan.schedule_content_sha256
        or result.manifest_fingerprint != plan.manifest_fingerprint
    ):
        raise BfclV4PublicRunnerError("published runner result belongs to another frozen plan")
    public_tasks = tuple(
        load_canonical_model(
            repository,
            ref,
            BfclV4PublicPilotTask,
            media_type=BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE,
        )
        for ref in result.public_task_refs
    )
    if tuple(item.task_id for item in public_tasks) != tuple(
        item.task_id for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    ):
        raise BfclV4PublicRunnerError("published public tasks differ from the frozen roster")
    tasks_by_id = {item.task_id: item for item in public_tasks}
    closure_verification = verify_bfcl_v4_public_run_closure(
        repository,
        result.journal_closure_ref,
        plan=plan,
    )
    if closure_verification != result.closure_verification:
        raise BfclV4PublicRunnerError("stored closure verification differs from offline replay")
    closure = load_canonical_model(
        repository,
        result.journal_closure_ref,
        BfclV4RunClosure,
        media_type=BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    )

    previous_outcome_ref: ArtifactRef | None = None
    provider_successes = 0
    identity_coordinates: list[tuple[str | None, str | None]] = []
    charged_tokens = 0
    backend_coordinates: set[tuple[str, str, str, str]] = set()
    for index, slot in enumerate(plan.calls):
        outcome_ref = result.attempt_outcome_refs[index]
        execution_ref = result.native_execution_refs[index]
        completion_ref = closure.call_completion_refs[index]
        outcome, reservation = _attempt_pair(repository, outcome_ref=outcome_ref)
        try:
            execution = load_canonical_native_artifact(
                repository,
                execution_ref,
                NativeFunctionExecution,
            )
        except Exception as exc:
            raise BfclV4PublicRunnerError("native execution failed canonical loading") from exc
        completion = load_canonical_model(
            repository,
            completion_ref,
            BfclV4CallCompletion,
            media_type=BFCL_V4_CALL_COMPLETION_MEDIA_TYPE,
        )
        materialization = load_canonical_model(
            repository,
            completion.materialization_ref,
            BfclV4CallMaterialization,
            media_type=BFCL_V4_CALL_MATERIALIZATION_MEDIA_TYPE,
        )
        reservation_coordinates = (
            reservation.ledger_id,
            reservation.budget_fingerprint,
            reservation.sequence,
            reservation.previous_outcome_ref,
            reservation.reserved_tokens,
        )
        expected_reservation = (
            result.attempt_ledger_id,
            result.attempt_budget.fingerprint,
            index,
            previous_outcome_ref,
            result.attempt_budget.max_tokens_per_attempt,
        )
        if reservation_coordinates != expected_reservation:
            raise BfclV4PublicRunnerError("attempt reservation chain or budget changed")
        if (
            outcome.ledger_id != result.attempt_ledger_id
            or outcome.writer_epoch_id != reservation.writer_epoch_id
            or outcome.budget_fingerprint != result.attempt_budget.fingerprint
            or outcome.sequence != index
            or outcome.execution_ref != execution_ref
            or completion.attempt_outcome_ref != outcome_ref
        ):
            raise BfclV4PublicRunnerError("attempt outcome does not bind execution/completion")
        if (
            reservation.task_fingerprint != execution.task_fingerprint
            or reservation.execution_fingerprint != execution.execution_fingerprint
            or reservation.request_sha256 != execution.request_sha256
            or outcome.reported_tokens != execution.usage.total_tokens
        ):
            raise BfclV4PublicRunnerError("reservation/outcome differs from native execution")
        expected_execution = (
            result.model_spec,
            canonical_sha256(slot),
            slot.seed_u63,
            result.model_spec.model,
            result.model_spec.inference,
            result.model_spec.backend_fingerprint,
        )
        actual_execution = (
            execution.spec,
            execution.slot_fingerprint,
            execution.request.seed,
            execution.request.requested_model,
            execution.request.inference,
            execution.request.backend_fingerprint,
        )
        if actual_execution != expected_execution:
            raise BfclV4PublicRunnerError("native execution differs from frozen slot/model input")
        backend_coordinates.add(
            (
                execution.request.backend_fingerprint,
                execution.request.serializer_fingerprint,
                execution.request.parser_fingerprint,
                execution.request.transport_fingerprint,
            )
        )
        observed_backend = _ObservedBackendIdentity(
            fingerprint=execution.request.backend_fingerprint,
            serializer_fingerprint=execution.request.serializer_fingerprint,
            parser_fingerprint=execution.request.parser_fingerprint,
            transport_fingerprint=execution.request.transport_fingerprint,
        )
        if materialization.request_ref != execution.request_ref:
            raise BfclV4PublicRunnerError("semantic materialization used another native request")
        if slot.kind is BfclV4PilotCallKind.DIAGNOSIS:
            meta_prompt = load_canonical_model(
                repository,
                materialization.executed_harness_ref,
                BfclV4DiagnosisPrompt,
                media_type=BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
            )
            expected_task_fingerprint = meta_prompt.fingerprint
            expected_request = materialize_bfcl_v4_public_meta_native_request(
                prompt=meta_prompt,
                spec=result.model_spec,
                backend=observed_backend,
                seed=slot.seed_u63,
            )
        elif slot.kind is BfclV4PilotCallKind.PROPOSAL:
            meta_prompt = load_canonical_model(
                repository,
                materialization.executed_harness_ref,
                BfclV4ProposalPrompt,
                media_type=BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
            )
            expected_task_fingerprint = meta_prompt.fingerprint
            expected_request = materialize_bfcl_v4_public_meta_native_request(
                prompt=meta_prompt,
                spec=result.model_spec,
                backend=observed_backend,
                seed=slot.seed_u63,
            )
        else:
            if slot.task_id is None or slot.task_id not in tasks_by_id:
                raise BfclV4PublicRunnerError("solver execution is not bound to a public task")
            task = tasks_by_id[slot.task_id]
            harness = load_canonical_model(
                repository,
                materialization.executed_harness_ref,
                BfclV4RunnerHarnessArtifact,
                media_type=BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
            )
            expected_task_fingerprint = task.fingerprint
            expected_request = materialize_solver_request(
                task=task,
                adapter=adapt_bfcl_v4_public_pilot_task(task),
                harness=harness,
                spec=result.model_spec,
                backend=observed_backend,
                seed=slot.seed_u63,
            )
        if (
            execution.task_fingerprint != expected_task_fingerprint
            or execution.request != expected_request
        ):
            raise BfclV4PublicRunnerError("execution task/prompt/request binding changed")
        if completion.outcome is BfclV4CallOutcome.SUCCEEDED:
            if execution.status is not ExecutionStatus.COMPLETED:
                raise BfclV4PublicRunnerError("successful completion embeds a failed execution")
            if outcome.disposition is not AttemptDisposition.SETTLED:
                raise BfclV4PublicRunnerError("successful provider call was not settled")
            provider_successes += 1
        elif execution.status is not ExecutionStatus.FAILED or (
            outcome.disposition is AttemptDisposition.SETTLED
        ):
            raise BfclV4PublicRunnerError("provider failure was represented as successful")
        response = execution.response
        if (
            execution.status is ExecutionStatus.COMPLETED
            and response is not None
            and response.provider_identity_observation is not None
        ):
            identity = response.provider_identity_observation
            identity_coordinates.append((identity.response_model, identity.system_fingerprint))
        for label, ref in (
            ("request", execution.request_ref),
            ("execution", execution_ref),
            ("attempt outcome", outcome_ref),
            ("model output", completion.model_output_ref),
        ):
            _require_nonempty(repository, ref, label)
        charged_tokens += outcome.charged_tokens
        previous_outcome_ref = outcome_ref

    if previous_outcome_ref != result.attempt_ledger_tail_ref:
        raise BfclV4PublicRunnerError("result attempt tail differs from replayed final outcome")
    if charged_tokens > result.attempt_budget.max_total_tokens:
        raise BfclV4PublicRunnerError("replayed attempt charges exceed the frozen token budget")
    identity_consistent = bool(identity_coordinates) and len(set(identity_coordinates)) == 1
    if len(backend_coordinates) != 1:
        raise BfclV4PublicRunnerError("native backend protocol identities changed within the run")
    backend, serializer, parser, transport = next(iter(backend_coordinates))
    expected_runner_fingerprint = canonical_sha256(
        {
            "implementation": "spiral-harness/native-function-runner/v1",
            "spec_fingerprint": result.model_spec.fingerprint,
            "backend": {
                "fingerprint": backend,
                "serializer": serializer,
                "parser": parser,
                "transport": transport,
            },
        }
    )
    if (
        backend != result.model_spec.backend_fingerprint
        or expected_runner_fingerprint != result.native_runner_fingerprint
    ):
        raise BfclV4PublicRunnerError("native runner/backend identity binding changed")
    if (
        provider_successes != result.provider_attempts_succeeded
        or 100 - provider_successes != result.provider_attempts_failed
        or len(identity_coordinates) != result.provider_identity_observation_count
        or identity_consistent != result.provider_declared_identity_consistent
    ):
        raise BfclV4PublicRunnerError("provider outcome/identity summary differs from executions")

    evidence = load_canonical_model(
        repository,
        result.joint_selection_evidence_ref,
        BfclV4JointSelectionEvidence,
        media_type=BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE,
    )
    decision = load_canonical_model(
        repository,
        result.joint_selection_decision_ref,
        BfclV4JointSelectionDecision,
        media_type=BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
    )
    holdout = load_canonical_model(
        repository,
        result.holdout_evidence_ref,
        BfclV4HoldoutEvidence,
        media_type=BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    )
    metrics = load_canonical_model(
        repository,
        result.descriptive_metrics_ref,
        BfclV4PublicDescriptiveMetrics,
        media_type=BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    )
    if (
        evidence.plan_fingerprint != plan.fingerprint
        or decision.joint_evidence_fingerprint != evidence.fingerprint
        or holdout.joint_selection_decision_fingerprint != decision.fingerprint
        or metrics.holdout_evidence_fingerprint != holdout.fingerprint
    ):
        raise BfclV4PublicRunnerError("selection/HOLDOUT/metrics lineage changed")
    return BfclV4PublicPilotRunVerification(
        result_fingerprint=result.fingerprint,
        plan_fingerprint=plan.fingerprint,
        provider_identity_observation_count=len(identity_coordinates),
        provider_declared_identity_consistent=identity_consistent,
    )


__all__ = ["verify_bfcl_v4_public_pilot_result"]
