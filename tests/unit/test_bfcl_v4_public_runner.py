from __future__ import annotations

import os
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicPrediction,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4PureAtBSample,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BfclV4CallCompletion,
    BfclV4RunClosure,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import (
    AttemptBudget,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
    ProviderIdentityObservation,
)
from spiral_harness.experiments.bfcl_v4_public_runner import (
    run_bfcl_v4_public_pilot_replicate,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BfclV4PublicPilotRunResult,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    BfclV4PublicRunnerError,
    load_canonical_model,
    publish_model,
)
from spiral_harness.experiments.bfcl_v4_public_runner_verification import (
    verify_bfcl_v4_public_pilot_result,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4HoldoutEvidence,
    BfclV4JointSelectionDecision,
)
from spiral_harness.experiments.bfcl_v4_public_selection_logic import (
    compute_bfcl_v4_public_descriptive_metrics,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeAssistantToolCall,
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.artifact_store import ArtifactStore

BACKEND = "a" * 64
SERIALIZER = "b" * 64
PARSER = "c" * 64
TRANSPORT = "d" * 64
_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


class _ScriptedBackend:
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(
        self,
        *,
        fail_indices: frozenset[int] = frozenset(),
        retained_response_failure_indices: frozenset[int] = frozenset(),
    ) -> None:
        self.fail_indices = fail_indices
        self.retained_response_failure_indices = retained_response_failure_indices
        self.requests: list[NativeFunctionCallRequest] = []
        self._mismatched_fingerprint_reads = 0

    @property
    def fingerprint(self) -> str:
        if self._mismatched_fingerprint_reads:
            self._mismatched_fingerprint_reads -= 1
            return "e" * 64
        return BACKEND

    def invoke(self, *, request: NativeFunctionCallRequest) -> NativeFunctionCallResponse:
        index = len(self.requests)
        self.requests.append(request)
        if index in self.fail_indices:
            raise TimeoutError("scripted provider timeout")
        submit_name = request.task_required_tools[0].official_name
        if submit_name == "submit_bfcl_diagnosis":
            arguments = {"diagnosis": "Preserve exact schema types and requested call counts."}
        elif submit_name == "submit_bfcl_candidate":
            arguments = {
                "strategy_appendix": (
                    "Silently map every requested operation to one documented function, then "
                    "check required argument types and independent-call multiplicity."
                )
            }
        elif index in self.retained_response_failure_indices:
            arguments = {}
        else:
            arguments = None
        tool_calls = ()
        assistant_text = "No native call selected by the deterministic replay fixture."
        finish_reason = "stop"
        if arguments is not None:
            tool_calls = (
                NativeAssistantToolCall(
                    call_id=f"fixture-call-{index}",
                    official_name=submit_name,
                    wire_name=request.task_required_tools[0].wire_name,
                    arguments_json=canonical_json(arguments),
                ),
            )
            assistant_text = None
            finish_reason = "tool_calls"
        response = NativeFunctionCallResponse(
            request_fingerprint=request.fingerprint,
            serializer_fingerprint=request.serializer_fingerprint,
            parser_fingerprint=request.parser_fingerprint,
            transport_fingerprint=request.transport_fingerprint,
            tools_fingerprint=request.tools_fingerprint,
            tool_calls=tool_calls,
            assistant_text=assistant_text,
            finish_reason=finish_reason,
            usage=BackendTokenUsage(input_tokens=1, output_tokens=1),
            provider_identity_observation=ProviderIdentityObservation(
                requested_model=request.requested_model,
                response_model="fixture/model-snapshot",
                system_fingerprint="fixture-system-v1",
                backend_fingerprint=BACKEND,
            ),
        )
        if index in self.retained_response_failure_indices:
            self._mismatched_fingerprint_reads = 1
        return response


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="scripted-native-replay",
        backend_fingerprint=BACKEND,
        model="fixture/model-snapshot",
        revision="snapshot-2026-08-15",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-15",
        runtime="fixture-runtime@sha256:bfcl-runner-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=8,
            timeout_seconds=5.0,
        ),
    )


def _budget() -> AttemptBudget:
    return AttemptBudget(
        max_attempts=100,
        max_total_tokens=1_000,
        max_tokens_per_attempt=10,
    )


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for runner integration")
    return checkout


def test_scripted_replicate_consumes_100_slots_and_offline_replays(
    tmp_path: Path,
    pinned_checkout: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    backend = _ScriptedBackend(
        fail_indices=frozenset({0}),
        retained_response_failure_indices=frozenset({40, 72}),
    )

    record = run_bfcl_v4_public_pilot_replicate(
        store,
        checkout=pinned_checkout,
        spec=_spec(),
        backend=backend,
        attempt_budget=_budget(),
        attempt_ledger_id="bfcl-v4-public-pilot/test-replicate",
    )
    verification = verify_bfcl_v4_public_pilot_result(store, record.result_ref)

    assert len(backend.requests) == 100
    assert record.result.provider_attempts_succeeded == 97
    assert record.result.provider_attempts_failed == 3
    assert record.result.provider_identity_observation_count == 97
    assert record.result.provider_declared_identity_consistent is True
    assert record.result.provider_served_same_weights_attested is False
    assert record.result.public_development_only is True
    assert record.result.hidden_test_evidence is False
    assert record.result.reportable_result is False
    assert verification.completed_model_calls == 100
    assert verification.verified_journal_transitions == 204
    assert verification.provider_identity_observation_count == 97

    closure = load_canonical_model(
        store,
        record.result.journal_closure_ref,
        BfclV4RunClosure,
    )
    failed_completion = load_canonical_model(
        store,
        closure.call_completion_refs[0],
        BfclV4CallCompletion,
    )
    assert failed_completion.prediction_ref is not None
    assert failed_completion.grader_receipt_ref is not None
    failed_prediction = load_canonical_model(
        store,
        failed_completion.prediction_ref,
        BfclV4PublicPrediction,
    )
    assert failed_prediction.calls == ()

    retained_failure_completion = load_canonical_model(
        store,
        closure.call_completion_refs[40],
        BfclV4CallCompletion,
    )
    assert retained_failure_completion.prediction_ref is not None
    retained_failure_prediction = load_canonical_model(
        store,
        retained_failure_completion.prediction_ref,
        BfclV4PublicPrediction,
    )
    assert retained_failure_prediction.calls == ()
    retained_pure_at_b_completion = load_canonical_model(
        store,
        closure.call_completion_refs[72],
        BfclV4CallCompletion,
    )
    retained_pure_at_b_sample = load_canonical_model(
        store,
        retained_pure_at_b_completion.model_output_ref,
        BfclV4PureAtBSample,
    )
    assert retained_pure_at_b_sample.calls is None

    holdout = load_canonical_model(
        store,
        record.result.holdout_evidence_ref,
        BfclV4HoldoutEvidence,
    )
    decision = load_canonical_model(
        store,
        record.result.joint_selection_decision_ref,
        BfclV4JointSelectionDecision,
    )
    target = holdout.observations[1]
    assert target.source_provider_attempt_succeeded == (True,)
    forged_target = target.model_copy(update={"accepted": not target.accepted})
    forged_holdout = BfclV4HoldoutEvidence(
        plan_fingerprint=holdout.plan_fingerprint,
        schedule_content_sha256=holdout.schedule_content_sha256,
        joint_selection_decision_fingerprint=holdout.joint_selection_decision_fingerprint,
        observations=(holdout.observations[0], forged_target, *holdout.observations[2:]),
    )
    forged_holdout_ref = publish_model(
        store,
        forged_holdout,
        media_type=BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    )
    forged_metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=build_bfcl_v4_public_pilot_call_plan(record.result.outer_seed_u64),
        joint_selection=decision,
        evidence=forged_holdout,
    )
    forged_metrics_ref = publish_model(
        store,
        forged_metrics,
        media_type=BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    )
    forged_result = BfclV4PublicPilotRunResult.model_validate(
        {
            **record.result.model_dump(mode="python"),
            "holdout_evidence_ref": forged_holdout_ref,
            "descriptive_metrics_ref": forged_metrics_ref,
        },
        strict=True,
    )
    forged_result_ref = publish_model(
        store,
        forged_result,
        media_type=BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    )
    with pytest.raises(BfclV4PublicRunnerError, match="HOLDOUT grade is detached"):
        verify_bfcl_v4_public_pilot_result(store, forged_result_ref)

    pure_at_b_index = 32
    pure_at_b = holdout.observations[pure_at_b_index]
    assert pure_at_b.pure_at_b_selection_fingerprint != "f" * 64
    forged_pure_at_b = pure_at_b.model_copy(
        update={"pure_at_b_selection_fingerprint": "f" * 64}
    )
    pure_at_b_holdout = BfclV4HoldoutEvidence(
        plan_fingerprint=holdout.plan_fingerprint,
        schedule_content_sha256=holdout.schedule_content_sha256,
        joint_selection_decision_fingerprint=holdout.joint_selection_decision_fingerprint,
        observations=(
            *holdout.observations[:pure_at_b_index],
            forged_pure_at_b,
            *holdout.observations[pure_at_b_index + 1 :],
        ),
    )
    pure_at_b_holdout_ref = publish_model(
        store,
        pure_at_b_holdout,
        media_type=BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE,
    )
    pure_at_b_metrics = compute_bfcl_v4_public_descriptive_metrics(
        plan=build_bfcl_v4_public_pilot_call_plan(record.result.outer_seed_u64),
        joint_selection=decision,
        evidence=pure_at_b_holdout,
    )
    pure_at_b_metrics_ref = publish_model(
        store,
        pure_at_b_metrics,
        media_type=BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    )
    pure_at_b_result = BfclV4PublicPilotRunResult.model_validate(
        {
            **record.result.model_dump(mode="python"),
            "holdout_evidence_ref": pure_at_b_holdout_ref,
            "descriptive_metrics_ref": pure_at_b_metrics_ref,
        },
        strict=True,
    )
    pure_at_b_result_ref = publish_model(
        store,
        pure_at_b_result,
        media_type=BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    )
    with pytest.raises(BfclV4PublicRunnerError, match="PURE@B selection or grade"):
        verify_bfcl_v4_public_pilot_result(store, pure_at_b_result_ref)


def test_runner_rejects_a_budget_that_cannot_burn_all_frozen_slots(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    backend = _ScriptedBackend()
    insufficient = AttemptBudget(
        max_attempts=100,
        max_total_tokens=999,
        max_tokens_per_attempt=10,
    )

    with pytest.raises(ValueError, match="100 fully burned"):
        run_bfcl_v4_public_pilot_replicate(
            store,
            checkout=tmp_path / "absent-checkout",
            spec=_spec(),
            backend=backend,
            attempt_budget=insufficient,
        )

    assert backend.requests == []


def test_runner_result_contract_cannot_claim_hidden_or_reportable_evidence() -> None:
    assert BfclV4PublicPilotRunResult.model_fields["hidden_test_evidence"].default is False
    assert BfclV4PublicPilotRunResult.model_fields["reportable_result"].default is False
