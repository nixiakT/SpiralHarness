"""Execute and replay controller-authorized, score-free matched skill probes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    MODEL_EXECUTION_MEDIA_TYPE,
    CandidateTask,
    FrozenModelSpec,
    ModelExecution,
)
from spiral_harness.execution.model import ModelBackend
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    TrustedExecutionUsage,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    EvaluationBatchSchedule,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.protocol import ArtifactRepository
from spiral_harness.verification.skill_inclusion import verify_settled_skill_request_inclusion
from spiral_harness.verification.skill_plan import (
    CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
    SKILL_PROBE_ROSTER_MEDIA_TYPE,
    ControlledSkillProbeTask,
    SkillMechanismPlan,
    SkillProbeRoster,
)

from .controller_artifacts import SkillProbeSettlementKind
from .skill_probe_arms import (
    _execute_prepared_skill_probe_arm,
    _prepare_skill_probe_arm,
    _SkillProbeArmError,
)
from .skill_probe_authorization import (
    SkillProbeExecutionAuthorization,
    SkillProbeExecutionAuthorizationCapability,
    TrustedSkillProbeExecutionAuthority,
    _ConsumedSkillProbeExecutionAuthorizationError,
)
from .skill_probe_closure import (
    MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
    SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE,
    MatchedSkillProbeClosure,
    MatchedSkillProbeExecutionResult,
    SkillProbeArmClosure,
    SkillProbeShadowReport,
    VerifiedMatchedSkillProbeResult,
)
from .skill_probe_shadow import _load_exact as _load_exact_artifact
from .skill_probe_shadow import _verify_shadow_for_closure
from .skill_probes import verify_skill_probe_preregistration

MATCHED_SKILL_PROBE_RESET_FINGERPRINT = sha256_bytes(
    b"spiral-harness/matched-skill-probe-new-runner-ledger-objects-per-arm/v1"
)
MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT = sha256_bytes(
    b"spiral-harness/matched-skill-probe-revert-then-placebo-first-attempt-only/v1"
)


class MatchedSkillProbeExecutionError(RuntimeError):
    """Raised when an execution or closure cannot be trusted exactly."""


@dataclass(frozen=True, slots=True)
class MatchedSkillProbeExecution:
    """Local live-ledger handles plus the immutable published result."""

    result: MatchedSkillProbeExecutionResult
    revert_attempt_ledger: AttemptLedger
    placebo_attempt_ledger: AttemptLedger


SkillProbeUsageSettlement = Callable[
    [
        ArtifactRef,
        ArtifactRef,
        ArtifactRef | None,
        ArtifactRef,
        ArtifactRef | None,
        SkillProbeSettlementKind,
        ArtifactRef | None,
    ],
    None,
]


def _load_exact[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    media_type: str,
    label: str,
) -> ModelT:
    try:
        return _load_exact_artifact(repository, ref, model_type, media_type=media_type, label=label)
    except Exception as exc:
        raise MatchedSkillProbeExecutionError(str(exc)) from exc


def _tasks(
    repository: ArtifactRepository,
    roster: SkillProbeRoster,
) -> dict[str, CandidateTask]:
    tasks: dict[str, CandidateTask] = {}
    for ref in roster.task_refs:
        frozen = _load_exact(
            repository,
            ref,
            ControlledSkillProbeTask,
            media_type=CONTROLLED_SKILL_PROBE_TASK_MEDIA_TYPE,
            label="controlled skill probe task",
        )
        if frozen.task_id in tasks:
            raise MatchedSkillProbeExecutionError("probe tasks contain a duplicate task ID")
        tasks[frozen.task_id] = frozen.candidate_task
    if tuple(sorted(tasks)) != roster.task_ids:
        raise MatchedSkillProbeExecutionError("probe tasks differ from the frozen roster")
    return tasks


def _verify_model_context(
    plan: SkillMechanismPlan,
    *,
    spec: FrozenModelSpec,
) -> None:
    if spec.fingerprint != plan.model_spec_fingerprint:
        raise MatchedSkillProbeExecutionError("model spec differs from the preregistered plan")
    if spec.runtime_fingerprint != plan.runtime_fingerprint:
        raise MatchedSkillProbeExecutionError("runtime differs from the preregistered plan")
    if plan.reset_fingerprint != MATCHED_SKILL_PROBE_RESET_FINGERPRINT:
        raise MatchedSkillProbeExecutionError(
            "plan does not freeze the new runner-and-ledger object declaration"
        )
    if plan.execution_order_fingerprint != MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT:
        raise MatchedSkillProbeExecutionError("plan does not freeze the trusted execution order")


def _execute_matched_skill_probes(
    repository: ArtifactRepository,
    *,
    authorization_ref: ArtifactRef,
    execution_authority: TrustedSkillProbeExecutionAuthority,
    authorization_capability: SkillProbeExecutionAuthorizationCapability,
    model_spec: FrozenModelSpec,
    revert_backend: ModelBackend,
    placebo_backend: ModelBackend,
    reserve_usage: Callable[
        [ArtifactRef, SkillProbeExecutionAuthorization, SkillMechanismPlan],
        None,
    ],
    settle_usage: SkillProbeUsageSettlement,
) -> MatchedSkillProbeExecution:
    """Preflight, reserve global usage, then consume and execute one live grant."""

    if not isinstance(repository, ArtifactRepository):
        raise MatchedSkillProbeExecutionError("repository must implement ArtifactRepository")
    if type(authorization_capability) is not SkillProbeExecutionAuthorizationCapability:
        raise MatchedSkillProbeExecutionError("authorization capability has the wrong type")
    if type(execution_authority) is not TrustedSkillProbeExecutionAuthority:
        raise MatchedSkillProbeExecutionError("execution authority has the wrong type")
    if authorization_capability.repository is not repository:
        raise MatchedSkillProbeExecutionError(
            "authorization capability uses a different repository object"
        )
    if execution_authority.repository is not repository:
        raise MatchedSkillProbeExecutionError(
            "execution authority uses a different repository object"
        )
    try:
        authorization = authorization_capability.verify_skill_probe_execution_authorization(
            authorization_ref
        )
    except Exception as exc:
        raise MatchedSkillProbeExecutionError(f"skill probe authorization failed: {exc}") from exc
    plan = verify_skill_probe_preregistration(
        repository,
        plan_ref=authorization.plan_ref,
        expected_experiment_ref=authorization.experiment_ref,
        expected_protocol_ref=authorization.protocol_ref,
        expected_candidate_ref=authorization.candidate_ref,
    )
    spec = FrozenModelSpec.model_validate(model_spec, strict=True)
    _verify_model_context(plan, spec=spec)
    if revert_backend is placebo_backend:
        raise MatchedSkillProbeExecutionError(
            "matched arms require independent backend session objects"
        )
    for control, backend in (("revert", revert_backend), ("placebo", placebo_backend)):
        if not isinstance(backend, ModelBackend):
            raise MatchedSkillProbeExecutionError(f"{control} backend must implement ModelBackend")
        try:
            fingerprint = backend.fingerprint
        except Exception as exc:
            raise MatchedSkillProbeExecutionError(
                f"{control} backend fingerprint cannot be read"
            ) from exc
        if fingerprint != spec.backend_fingerprint:
            raise MatchedSkillProbeExecutionError(
                f"{control} backend fingerprint differs from the frozen model spec"
            )
    roster = _load_exact(
        repository,
        plan.probe_roster_ref,
        SkillProbeRoster,
        media_type=SKILL_PROBE_ROSTER_MEDIA_TYPE,
        label="skill probe roster",
    )
    tasks = _tasks(repository, roster)
    try:
        revert_arm = _prepare_skill_probe_arm(
            repository,
            control="revert",
            schedule=plan.revert_schedule,
            parent_harness_ref=plan.parent_harness_ref,
            candidate_harness_ref=plan.candidate_harness_ref,
            tasks=tasks,
            spec=spec,
            backend=revert_backend,
            execution_nonce=authorization.execution_nonce,
        )
        placebo_arm = _prepare_skill_probe_arm(
            repository,
            control="placebo",
            schedule=plan.placebo_schedule,
            parent_harness_ref=plan.placebo_harness_ref,
            candidate_harness_ref=plan.candidate_harness_ref,
            tasks=tasks,
            spec=spec,
            backend=placebo_backend,
            execution_nonce=authorization.execution_nonce,
        )
    except _SkillProbeArmError as exc:
        raise MatchedSkillProbeExecutionError(str(exc)) from exc

    reserve_usage(authorization_ref, authorization, plan)
    try:
        begun = execution_authority.begin_skill_probe_execution(authorization_ref)
    except _ConsumedSkillProbeExecutionAuthorizationError as exc:
        try:
            settle_usage(
                authorization_ref,
                revert_arm.preflight_ref,
                revert_arm.terminal_tail_ref,
                placebo_arm.preflight_ref,
                placebo_arm.terminal_tail_ref,
                SkillProbeSettlementKind.FAILED,
                None,
            )
        except Exception as settlement_error:
            raise settlement_error from exc
        raise MatchedSkillProbeExecutionError(f"skill probe authorization failed: {exc}") from exc
    except Exception as exc:
        raise MatchedSkillProbeExecutionError(f"skill probe authorization failed: {exc}") from exc
    closure_ref: ArtifactRef | None = None
    terminal_kind = SkillProbeSettlementKind.FAILED
    execution_error: BaseException | None = None
    try:
        if begun != authorization:
            raise MatchedSkillProbeExecutionError(
                "skill probe authorization changed before execution"
            )
        try:
            revert = _execute_prepared_skill_probe_arm(revert_arm)
            placebo = _execute_prepared_skill_probe_arm(placebo_arm)
        except _SkillProbeArmError as exc:
            raise MatchedSkillProbeExecutionError(str(exc)) from exc
        closure = MatchedSkillProbeClosure(
            authorization_ref=authorization_ref,
            execution_nonce=authorization.execution_nonce,
            experiment_ref=authorization.experiment_ref,
            protocol_ref=authorization.protocol_ref,
            candidate_ref=authorization.candidate_ref,
            plan_ref=authorization.plan_ref,
            running_probes_tail_ref=authorization.running_probes_tail_ref,
            candidate_harness_ref=plan.candidate_harness_ref,
            probe_roster_ref=plan.probe_roster_ref,
            task_refs=roster.task_refs,
            model_spec_fingerprint=spec.fingerprint,
            runtime_fingerprint=spec.runtime_fingerprint,
            reset_fingerprint=MATCHED_SKILL_PROBE_RESET_FINGERPRINT,
            execution_order_fingerprint=MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT,
            revert=revert,
            placebo=placebo,
        )
        closure_ref = repository.put_json(
            closure,
            media_type=MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
        )
        try:
            execution_authority.register_skill_probe_execution_closure(
                authorization_ref,
                closure_ref,
            )
        except Exception as exc:
            raise MatchedSkillProbeExecutionError(
                f"skill probe closure registration failed: {exc}"
            ) from exc
        shadow = SkillProbeShadowReport(
            authorization_ref=authorization_ref,
            plan_ref=authorization.plan_ref,
            running_probes_tail_ref=authorization.running_probes_tail_ref,
            execution_closure_ref=closure_ref,
            request_inclusion_refs=(
                revert.request_inclusion_ref,
                placebo.request_inclusion_ref,
            ),
        )
        shadow_ref = repository.put_json(shadow, media_type=SKILL_PROBE_SHADOW_REPORT_MEDIA_TYPE)
        result = MatchedSkillProbeExecutionResult(
            closure_ref=closure_ref,
            shadow_report_ref=shadow_ref,
        )
        verified = _verify_matched_skill_probe_result(
            repository,
            result=result,
            authorization_capability=authorization_capability,
            revert_attempt_ledger=revert_arm.ledger,
            placebo_attempt_ledger=placebo_arm.ledger,
        )
        if verified.closure != closure or verified.shadow_report != shadow:
            raise MatchedSkillProbeExecutionError("published skill probe artifacts changed")
        terminal_kind = SkillProbeSettlementKind.COMPLETED
        return MatchedSkillProbeExecution(
            result=result,
            revert_attempt_ledger=revert_arm.ledger,
            placebo_attempt_ledger=placebo_arm.ledger,
        )
    except BaseException as exc:
        execution_error = exc
        raise
    finally:
        try:
            settle_usage(
                authorization_ref,
                revert_arm.preflight_ref,
                revert_arm.terminal_tail_ref,
                placebo_arm.preflight_ref,
                placebo_arm.terminal_tail_ref,
                terminal_kind,
                closure_ref,
            )
        except Exception as settlement_error:
            if execution_error is not None:
                raise settlement_error from execution_error
            raise


def _verify_arm(
    repository: ArtifactRepository,
    *,
    claimed: SkillProbeArmClosure,
    control: Literal["revert", "placebo"],
    schedule: EvaluationBatchSchedule,
    candidate_harness_ref: ArtifactRef,
    attempt_ledger: AttemptLedger,
) -> tuple[TrustedExecutionUsage, tuple[ExecutionReceipt, ...]]:
    if claimed.control != control or claimed.schedule_fingerprint != schedule.fingerprint:
        raise MatchedSkillProbeExecutionError(f"{control} closure belongs to another schedule")
    if attempt_ledger.repository is not repository:
        raise MatchedSkillProbeExecutionError(f"{control} ledger uses another repository object")
    usage = replay_trusted_usage(
        repository,
        schedule=schedule,
        preflight_ref=claimed.preflight_ref,
        attempt_ledger=attempt_ledger,
        receipt_refs=claimed.receipt_refs,
    )
    inclusion = verify_settled_skill_request_inclusion(
        repository,
        evidence_ref=claimed.request_inclusion_ref,
        schedule=schedule,
        preflight_ref=claimed.preflight_ref,
        attempt_ledger=attempt_ledger,
        candidate_harness_ref=candidate_harness_ref,
    )
    state = attempt_ledger.state()
    actual = (
        state.tail_ref,
        state.ledger_id,
        state.writer_epoch_id,
        state.budget.fingerprint,
        usage,
    )
    expected = (
        claimed.closing_ledger_tail_ref,
        claimed.ledger_id,
        claimed.writer_epoch_id,
        claimed.budget_fingerprint,
        claimed.usage,
    )
    if actual != expected or inclusion.usage != usage:
        raise MatchedSkillProbeExecutionError(f"{control} live closure changed")
    preflight = _load_exact(
        repository,
        claimed.preflight_ref,
        SchedulePreflightCertificate,
        media_type=SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        label=f"{control} preflight",
    )
    if preflight.ledger_tail_ref != claimed.opening_ledger_tail_ref:
        raise MatchedSkillProbeExecutionError(f"{control} opening ledger boundary changed")
    receipts = tuple(
        _load_exact(
            repository,
            ref,
            ExecutionReceipt,
            media_type=EXECUTION_RECEIPT_MEDIA_TYPE,
            label=f"{control} execution receipt",
        )
        for ref in usage.receipt_refs
    )
    return usage, receipts


def _verify_cross_arm_disjointness(
    revert_receipts: tuple[ExecutionReceipt, ...],
    placebo_receipts: tuple[ExecutionReceipt, ...],
) -> None:
    selectors = (
        ("receipt", lambda receipt: receipt.fingerprint),
        ("reservation", lambda receipt: receipt.reservation_ref.sha256),
        ("outcome", lambda receipt: receipt.outcome_ref.sha256),
    )
    for label, selector in selectors:
        left = {selector(receipt) for receipt in revert_receipts}
        right = {selector(receipt) for receipt in placebo_receipts}
        if left.intersection(right):
            raise MatchedSkillProbeExecutionError(f"matched arms reuse a {label}")


def _verify_task_bindings(
    repository: ArtifactRepository,
    *,
    roster: SkillProbeRoster,
    receipts: tuple[ExecutionReceipt, ...],
) -> None:
    expected = _tasks(repository, roster)
    for receipt in receipts:
        execution = _load_exact(
            repository,
            receipt.execution_ref,
            ModelExecution,
            media_type=MODEL_EXECUTION_MEDIA_TYPE,
            label="skill probe model execution",
        )
        if execution.task != expected[receipt.cell.task_id]:
            raise MatchedSkillProbeExecutionError(
                "skill probe execution task differs from the frozen task artifact"
            )


def _verify_matched_skill_probe_closure(
    repository: ArtifactRepository,
    *,
    closure_ref: ArtifactRef,
    authorization_capability: SkillProbeExecutionAuthorizationCapability,
    revert_attempt_ledger: AttemptLedger,
    placebo_attempt_ledger: AttemptLedger,
) -> MatchedSkillProbeClosure:
    """Re-derive a non-promoting closure from live, independent ledger writers."""

    if type(authorization_capability) is not SkillProbeExecutionAuthorizationCapability:
        raise MatchedSkillProbeExecutionError("authorization capability has the wrong type")
    if authorization_capability.repository is not repository:
        raise MatchedSkillProbeExecutionError(
            "authorization capability uses a different repository object"
        )
    closure = _load_exact(
        repository,
        closure_ref,
        MatchedSkillProbeClosure,
        media_type=MATCHED_SKILL_PROBE_CLOSURE_MEDIA_TYPE,
        label="matched skill probe closure",
    )
    try:
        authorization = authorization_capability.verify_registered_skill_probe_execution_closure(
            closure.authorization_ref,
            closure_ref,
        )
    except Exception as exc:
        raise MatchedSkillProbeExecutionError(f"skill probe authorization failed: {exc}") from exc
    if closure.execution_nonce != authorization.execution_nonce:
        raise MatchedSkillProbeExecutionError("closure uses another execution nonce")
    expected_context = (
        closure.experiment_ref,
        closure.protocol_ref,
        closure.candidate_ref,
        closure.plan_ref,
        closure.running_probes_tail_ref,
    )
    authorization_context = (
        authorization.experiment_ref,
        authorization.protocol_ref,
        authorization.candidate_ref,
        authorization.plan_ref,
        authorization.running_probes_tail_ref,
    )
    if expected_context != authorization_context:
        raise MatchedSkillProbeExecutionError("closure context differs from its authorization")
    plan = verify_skill_probe_preregistration(
        repository,
        plan_ref=closure.plan_ref,
        expected_experiment_ref=closure.experiment_ref,
        expected_protocol_ref=closure.protocol_ref,
        expected_candidate_ref=closure.candidate_ref,
    )
    if (
        closure.candidate_harness_ref != plan.candidate_harness_ref
        or closure.probe_roster_ref != plan.probe_roster_ref
        or closure.model_spec_fingerprint != plan.model_spec_fingerprint
        or closure.runtime_fingerprint != plan.runtime_fingerprint
        or closure.reset_fingerprint != plan.reset_fingerprint
        or closure.execution_order_fingerprint != plan.execution_order_fingerprint
    ):
        raise MatchedSkillProbeExecutionError("closure execution context differs from its plan")
    roster = _load_exact(
        repository,
        plan.probe_roster_ref,
        SkillProbeRoster,
        media_type=SKILL_PROBE_ROSTER_MEDIA_TYPE,
        label="skill probe roster",
    )
    if closure.task_refs != roster.task_refs:
        raise MatchedSkillProbeExecutionError("closure tasks differ from the frozen roster")
    if revert_attempt_ledger is placebo_attempt_ledger:
        raise MatchedSkillProbeExecutionError("matched arms reuse one live ledger")
    try:
        _, revert_receipts = _verify_arm(
            repository,
            claimed=closure.revert,
            control="revert",
            schedule=plan.revert_schedule,
            candidate_harness_ref=plan.candidate_harness_ref,
            attempt_ledger=revert_attempt_ledger,
        )
        _, placebo_receipts = _verify_arm(
            repository,
            claimed=closure.placebo,
            control="placebo",
            schedule=plan.placebo_schedule,
            candidate_harness_ref=plan.candidate_harness_ref,
            attempt_ledger=placebo_attempt_ledger,
        )
    except MatchedSkillProbeExecutionError:
        raise
    except Exception as exc:
        raise MatchedSkillProbeExecutionError("matched arm replay failed") from exc
    _verify_cross_arm_disjointness(revert_receipts, placebo_receipts)
    _verify_task_bindings(
        repository,
        roster=roster,
        receipts=(*revert_receipts, *placebo_receipts),
    )
    return closure


def _verify_matched_skill_probe_result(
    repository: ArtifactRepository,
    *,
    result: MatchedSkillProbeExecutionResult,
    authorization_capability: SkillProbeExecutionAuthorizationCapability,
    revert_attempt_ledger: AttemptLedger,
    placebo_attempt_ledger: AttemptLedger,
) -> VerifiedMatchedSkillProbeResult:
    """Verify the live closure before accepting its non-promoting shadow summary."""

    checked_result = MatchedSkillProbeExecutionResult.model_validate(result, strict=True)
    closure = _verify_matched_skill_probe_closure(
        repository,
        closure_ref=checked_result.closure_ref,
        authorization_capability=authorization_capability,
        revert_attempt_ledger=revert_attempt_ledger,
        placebo_attempt_ledger=placebo_attempt_ledger,
    )
    try:
        shadow = _verify_shadow_for_closure(
            repository,
            shadow_report_ref=checked_result.shadow_report_ref,
            verified_closure_ref=checked_result.closure_ref,
            verified_closure=closure,
        )
    except Exception as exc:
        raise MatchedSkillProbeExecutionError(str(exc)) from exc
    return VerifiedMatchedSkillProbeResult(closure=closure, shadow_report=shadow)


__all__ = [
    "MATCHED_SKILL_PROBE_EXECUTION_ORDER_FINGERPRINT",
    "MATCHED_SKILL_PROBE_RESET_FINGERPRINT",
    "MatchedSkillProbeExecution",
    "MatchedSkillProbeExecutionError",
]
