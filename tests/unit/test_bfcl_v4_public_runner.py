from __future__ import annotations

import os
from pathlib import Path

import pytest

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicPrediction,
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
    BfclV4PublicPilotRunResult,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    load_canonical_model,
)
from spiral_harness.experiments.bfcl_v4_public_runner_verification import (
    verify_bfcl_v4_public_pilot_result,
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
    fingerprint = BACKEND
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(self, *, fail_indices: frozenset[int] = frozenset()) -> None:
        self.fail_indices = fail_indices
        self.requests: list[NativeFunctionCallRequest] = []

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
        return NativeFunctionCallResponse(
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
    backend = _ScriptedBackend(fail_indices=frozenset({0}))

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
    assert record.result.provider_attempts_succeeded == 99
    assert record.result.provider_attempts_failed == 1
    assert record.result.provider_identity_observation_count == 99
    assert record.result.provider_declared_identity_consistent is True
    assert record.result.provider_served_same_weights_attested is False
    assert record.result.public_development_only is True
    assert record.result.hidden_test_evidence is False
    assert record.result.reportable_result is False
    assert verification.completed_model_calls == 100
    assert verification.verified_journal_transitions == 204
    assert verification.provider_identity_observation_count == 99

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
