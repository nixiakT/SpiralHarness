from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import BfclV4PublicPilotTask
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
    BfclV4RunAction,
    BfclV4RunClosure,
)
from spiral_harness.benchmark.bfcl_v4_public_run_journal import (
    replay_bfcl_v4_public_run,
    verify_bfcl_v4_public_run_closure,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.execution.accounted_execution import load_accounted_execution
from spiral_harness.execution.contracts import (
    AttemptDisposition,
    AttemptOutcome,
    AttemptReservation,
    BackendTokenUsage,
    ExecutionStatus,
    FrozenModelSpec,
    ProviderIdentityObservation,
)
from spiral_harness.execution.native_function_contracts import NativeFunctionExecution
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET,
    BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
    BFCL_V4_PUBLIC_LIVE_INFERENCE,
)
from spiral_harness.experiments.bfcl_v4_public_runner import (
    run_bfcl_v4_public_pilot_replicate,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE,
    BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE,
)
from spiral_harness.experiments.bfcl_v4_public_runner_verification import (
    verify_bfcl_v4_public_pilot_result,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4HoldoutEvidence,
    BfclV4PublicDescriptiveMetrics,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BACKEND = "1" * 64
SERIALIZER = "2" * 64
PARSER = "3" * 64
TRANSPORT = "4" * 64
RESPONSE_MODEL = "offline-fixture/qwen36-35b-a3b-served-snapshot"
SYSTEM_FINGERPRINT = "fp_offline_replay_snapshot_2026_08_15"


@dataclass(frozen=True, slots=True)
class _ProviderIdentityAudit:
    successful_response_count: int
    observation_count: int
    response_model_observation_count: int
    system_fingerprint_observation_count: int
    response_models: tuple[str, ...]
    system_fingerprints: tuple[str, ...]
    requested_model_and_backend_match_verified: bool = True
    observations_are_provider_declared_only: bool = True
    same_provider_weights_attested: bool = False

    @property
    def complete_observation_coverage(self) -> bool:
        return self.observation_count == self.successful_response_count


def _audit_provider_identities(
    responses: tuple[NativeFunctionCallResponse, ...],
    spec: FrozenModelSpec,
) -> _ProviderIdentityAudit:
    observations = tuple(
        response.provider_identity_observation
        for response in responses
        if response.provider_identity_observation is not None
    )
    if any(
        item.requested_model != spec.model or item.backend_fingerprint != spec.backend_fingerprint
        for item in observations
    ):
        raise ValueError("provider identity observation differs from the frozen model request")
    response_models = tuple(
        sorted({item.response_model for item in observations if item.response_model})
    )
    system_fingerprints = tuple(
        sorted({item.system_fingerprint for item in observations if item.system_fingerprint})
    )
    if len(response_models) > 1 or len(system_fingerprints) > 1:
        raise ValueError("provider-declared response identities changed within the replicate")
    return _ProviderIdentityAudit(
        successful_response_count=len(responses),
        observation_count=len(observations),
        response_model_observation_count=sum(
            item.response_model is not None for item in observations
        ),
        system_fingerprint_observation_count=sum(
            item.system_fingerprint is not None for item in observations
        ),
        response_models=response_models,
        system_fingerprints=system_fingerprints,
    )


class _OfflineReplayBackend:
    """Deterministic request-bound backend; it has no transport or credential."""

    fingerprint = BACKEND
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(self) -> None:
        self.requests: list[NativeFunctionCallRequest] = []
        self.responses: list[NativeFunctionCallResponse] = []

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        checked = NativeFunctionCallRequest.model_validate(request, strict=True)
        self.requests.append(checked)
        tool = checked.task_required_tools[0]
        if tool.official_name == "submit_bfcl_diagnosis":
            calls = (
                NativeAssistantToolCall(
                    call_id=f"offline-call-{len(self.requests):03d}",
                    official_name=tool.official_name,
                    wire_name=tool.wire_name,
                    arguments_json=canonical_json(
                        {
                            "diagnosis": (
                                "Preserve correct calls; audit semantic function choice, exact "
                                "schema arguments, and independent call multiplicity."
                            )
                        }
                    ),
                ),
            )
        elif tool.official_name == "submit_bfcl_candidate":
            calls = (
                NativeAssistantToolCall(
                    call_id=f"offline-call-{len(self.requests):03d}",
                    official_name=tool.official_name,
                    wire_name=tool.wire_name,
                    arguments_json=canonical_json(
                        {
                            "strategy_appendix": (
                                "Silently decompose independent operations, choose tools by "
                                "documented semantics, then validate every required argument "
                                "and call count before emitting native calls."
                            )
                        }
                    ),
                ),
            )
        else:
            calls = ()
        response = NativeFunctionCallResponse(
            request_fingerprint=checked.fingerprint,
            serializer_fingerprint=checked.serializer_fingerprint,
            parser_fingerprint=checked.parser_fingerprint,
            transport_fingerprint=checked.transport_fingerprint,
            tools_fingerprint=checked.tools_fingerprint,
            tool_calls=calls,
            assistant_text=None if calls else "Offline replay produced no solver tool call.",
            finish_reason="tool_calls" if calls else "stop",
            usage=BackendTokenUsage(input_tokens=1, output_tokens=1),
            provider_identity_observation=ProviderIdentityObservation(
                requested_model=checked.requested_model,
                response_model=RESPONSE_MODEL,
                system_fingerprint=SYSTEM_FINGERPRINT,
                backend_fingerprint=checked.backend_fingerprint,
            ),
        )
        self.responses.append(response)
        return response


def _offline_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="offline-native-replay-fixture",
        backend_fingerprint=BACKEND,
        model=BFCL_V4_PUBLIC_LIVE_DIRECT_MODEL_ROUTE,
        revision="offline-integration-fixture-snapshot-2026-08-15",
        tokenizer="offline-integration-fixture-tokenizer",
        tokenizer_revision="offline-integration-fixture-snapshot-2026-08-15",
        runtime="offline-provider-free-replay@v1",
        inference=BFCL_V4_PUBLIC_LIVE_INFERENCE,
    )


def _pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else Path("/tmp/spiral-bfcl-upstream")
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for offline integration")
    return checkout


def _expected_actions() -> tuple[BfclV4RunAction, ...]:
    actions = [BfclV4RunAction.OPEN]
    for global_slot in range(100):
        if global_slot == 14:
            actions.append(BfclV4RunAction.FREEZE_CANDIDATES)
        if global_slot == 40:
            actions.append(BfclV4RunAction.FREEZE_SELECTIONS)
        actions.extend((BfclV4RunAction.MATERIALIZE_CALL, BfclV4RunAction.COMPLETE_CALL))
    actions.append(BfclV4RunAction.CLOSE)
    return tuple(actions)


def test_complete_provider_free_100_call_replicate_replays_and_scores_pure_at_b(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    backend = _OfflineReplayBackend()
    spec = _offline_spec()

    record = run_bfcl_v4_public_pilot_replicate(
        store,
        checkout=_pinned_checkout(),
        spec=spec,
        backend=backend,
        attempt_budget=BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET,
        attempt_ledger_id="bfcl-v4-public/offline-replicate-001",
    )
    result = record.result
    plan = build_bfcl_v4_public_pilot_call_plan(result.outer_seed_u64)

    assert len(backend.requests) == len(backend.responses) == 100
    assert result.total_model_calls == 100
    assert result.provider_attempts_succeeded == 100
    assert result.provider_attempts_failed == 0
    assert result.provider_identity_observation_count == 100
    assert result.provider_declared_identity_consistent is True
    assert result.provider_served_same_weights_attested is False
    assert result.automatic_retries_used is False
    assert result.model_spec == spec
    assert result.attempt_budget == BFCL_V4_PUBLIC_LIVE_ATTEMPT_BUDGET
    assert result.plan_fingerprint == plan.fingerprint
    assert result.closure_verification.plan_fingerprint == plan.fingerprint

    closure = store.get_json(result.journal_closure_ref, BfclV4RunClosure)
    closure_verification = verify_bfcl_v4_public_run_closure(
        store,
        result.journal_closure_ref,
        plan=plan,
    )
    events, state = replay_bfcl_v4_public_run(store, closure.journal_tail_ref, plan=plan)
    actions = tuple(event.action for event in events)
    assert actions == _expected_actions()
    assert len(events) == closure_verification.replayed_transition_count == 204
    assert Counter(actions) == Counter(
        {
            BfclV4RunAction.OPEN: 1,
            BfclV4RunAction.MATERIALIZE_CALL: 100,
            BfclV4RunAction.COMPLETE_CALL: 100,
            BfclV4RunAction.FREEZE_CANDIDATES: 1,
            BfclV4RunAction.FREEZE_SELECTIONS: 1,
            BfclV4RunAction.CLOSE: 1,
        }
    )
    assert state.closed is True
    assert state.next_global_slot == 100
    assert state.candidate_freeze_ref == result.candidate_freeze_ref
    assert state.joint_selection_freeze_ref == result.joint_selection_freeze_ref

    executions = tuple(load_accounted_execution(store, ref) for ref in result.native_execution_refs)
    assert all(isinstance(item, NativeFunctionExecution) for item in executions)
    native_executions: tuple[NativeFunctionExecution, ...] = executions  # type: ignore[assignment]
    outcomes = tuple(store.get_json(ref, AttemptOutcome) for ref in result.attempt_outcome_refs)
    reservations = tuple(
        store.get_json(outcome.reservation_ref, AttemptReservation) for outcome in outcomes
    )
    completions = tuple(
        store.get_json(ref, BfclV4CallCompletion) for ref in closure.call_completion_refs
    )
    materializations = tuple(
        store.get_json(completion.materialization_ref, BfclV4CallMaterialization)
        for completion in completions
    )
    public_tasks = tuple(
        store.get_json(ref, BfclV4PublicPilotTask) for ref in result.public_task_refs
    )
    tasks_by_id = {task.task_id: task for task in public_tasks}
    assert len(tasks_by_id) == 15
    assert all(
        ref.media_type == BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE for ref in result.public_task_refs
    )

    assert tuple(item.sequence for item in outcomes) == tuple(range(100))
    assert tuple(item.sequence for item in reservations) == tuple(range(100))
    assert tuple(item.previous_outcome_ref for item in reservations) == (
        None,
        *result.attempt_outcome_refs[:-1],
    )
    assert all(item.disposition is AttemptDisposition.SETTLED for item in outcomes)
    assert tuple(item.execution_ref for item in outcomes) == result.native_execution_refs
    assert tuple(item.attempt_outcome_ref for item in completions) == result.attempt_outcome_refs
    assert tuple(item.global_slot for item in completions) == tuple(range(100))

    for index, (slot, materialization, completion, execution, request) in enumerate(
        zip(
            plan.calls,
            materializations,
            completions,
            native_executions,
            backend.requests,
            strict=True,
        )
    ):
        assert slot.global_slot == index
        assert materialization.slot == slot
        assert completion.call_slot_reference_sha256 == canonical_sha256(slot)
        assert execution.slot_fingerprint == canonical_sha256(slot)
        assert execution.request_ref == materialization.request_ref
        assert execution.request == request
        assert execution.spec == spec
        assert execution.request.requested_model == spec.model
        assert execution.request.inference == spec.inference
        assert execution.request.seed == slot.seed_u63
        assert execution.request.backend_fingerprint == BACKEND
        assert execution.request.serializer_fingerprint == SERIALIZER
        assert execution.request.parser_fingerprint == PARSER
        assert execution.request.transport_fingerprint == TRANSPORT
        assert execution.status is ExecutionStatus.COMPLETED
        assert execution.provider_identity_attested is False
        assert execution.runtime_execution_attested is False
        if slot.kind is BfclV4PilotCallKind.DIAGNOSIS:
            assert materialization.executed_harness_ref.media_type == (
                BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE
            )
            prompt = store.get_json(
                materialization.executed_harness_ref,
                BfclV4DiagnosisPrompt,
            )
            assert execution.task_fingerprint == prompt.fingerprint
        elif slot.kind is BfclV4PilotCallKind.PROPOSAL:
            assert materialization.executed_harness_ref.media_type == (
                BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE
            )
            prompt = store.get_json(
                materialization.executed_harness_ref,
                BfclV4ProposalPrompt,
            )
            assert execution.task_fingerprint == prompt.fingerprint
        else:
            assert slot.task_id is not None
            assert execution.task_fingerprint == tasks_by_id[slot.task_id].fingerprint
        assert materialization.candidate_freeze_ref == (
            None if index < 14 else result.candidate_freeze_ref
        )
        assert materialization.joint_selection_freeze_ref == (
            None if index < 40 else result.joint_selection_freeze_ref
        )

    holdout = store.get_json(result.holdout_evidence_ref, BfclV4HoldoutEvidence)
    metrics = store.get_json(result.descriptive_metrics_ref, BfclV4PublicDescriptiveMetrics)
    pure_at_b = tuple(item for item in holdout.observations if item.arm is BfclV4PilotArm.PURE_AT_B)
    pure_metric = next(item for item in metrics.arms if item.arm is BfclV4PilotArm.PURE_AT_B)
    assert len(pure_at_b) == 8
    assert sum(len(item.source_call_ids) for item in pure_at_b) == 28
    assert (
        tuple(ref for item in pure_at_b for ref in item.source_completion_refs)
        == closure.call_completion_refs[72:]
    )
    assert all(item.pure_at_b_selection_fingerprint is not None for item in pure_at_b)
    assert pure_metric.correctness == tuple(item.accepted for item in pure_at_b)
    assert pure_metric.correct_count == sum(item.accepted for item in pure_at_b)
    assert pure_metric.accuracy_basis_points == pure_metric.correct_count * 1_250

    identity_audit = _audit_provider_identities(tuple(backend.responses), spec)
    assert identity_audit.successful_response_count == 100
    assert identity_audit.observation_count == 100
    assert identity_audit.response_model_observation_count == 100
    assert identity_audit.system_fingerprint_observation_count == 100
    assert identity_audit.response_models == (RESPONSE_MODEL,)
    assert identity_audit.system_fingerprints == (SYSTEM_FINGERPRINT,)
    assert identity_audit.complete_observation_coverage is True
    assert identity_audit.observations_are_provider_declared_only is True
    assert identity_audit.same_provider_weights_attested is False

    missing_identity = backend.responses[0].model_copy(
        update={"provider_identity_observation": None}
    )
    limited_audit = _audit_provider_identities(
        (missing_identity, *backend.responses[1:]),
        spec,
    )
    assert limited_audit.observation_count == 99
    assert limited_audit.complete_observation_coverage is False
    assert limited_audit.same_provider_weights_attested is False

    identity = backend.responses[0].provider_identity_observation
    assert identity is not None
    mismatched_identity = identity.model_copy(update={"requested_model": "foreign/model"})
    mismatched_response = backend.responses[0].model_copy(
        update={"provider_identity_observation": mismatched_identity}
    )
    with pytest.raises(ValueError, match="frozen model request"):
        _audit_provider_identities((mismatched_response,), spec)

    offline_verification = verify_bfcl_v4_public_pilot_result(store, record.result_ref)
    assert offline_verification.completed_model_calls == 100
    assert offline_verification.verified_attempt_outcomes == 100
    assert offline_verification.verified_native_executions == 100
    assert offline_verification.verified_journal_transitions == 204
    assert offline_verification.same_requested_model_and_inference_verified is True
    assert offline_verification.same_backend_protocol_identities_verified is True
    assert offline_verification.task_and_meta_prompt_fingerprints_verified is True
    assert offline_verification.provider_identity_observation_count == 100
    assert offline_verification.provider_declared_identity_consistent is True
    assert offline_verification.provider_served_same_weights_attested is False
    assert offline_verification.one_attempt_per_frozen_slot_verified is True
    assert offline_verification.request_execution_outcome_completion_join_verified is True
    assert offline_verification.semantic_closure_verified is True
    assert offline_verification.reportable_result is False
