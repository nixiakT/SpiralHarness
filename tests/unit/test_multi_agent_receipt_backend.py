from __future__ import annotations

import json

import pytest

from spiral_harness.agents.contracts import (
    AgentDependencyOutput,
    AgentSpec,
    AgentTurnRequest,
)
from spiral_harness.agents.receipt_backend import (
    ReceiptBackedAgentBackend,
    ReceiptBackedAgentBackendError,
    render_agent_turn_user_prompt,
)
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.contracts import (
    AttemptBudget,
    BackendResponse,
    BackendTokenUsage,
    FrozenModelSpec,
    InferenceConfig,
)
from spiral_harness.execution.model import FixedModelRunner, ReplayBackend
from spiral_harness.storage.artifact_store import ArtifactStore


def _spec(backend_fingerprint: str) -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="fixture",
        backend_fingerprint=backend_fingerprint,
        model="fixture/model",
        revision="snapshot-2026-08-20",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-20",
        runtime="fixture-runtime-v1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
            timeout_seconds=5.0,
        ),
    )


def _request(agent: AgentSpec) -> AgentTurnRequest:
    dependency_content = "candidate"
    return AgentTurnRequest(
        team_fingerprint="a" * 64,
        workflow_fingerprint="b" * 64,
        task_id="task-1",
        task_payload='{"question":"choose"}',
        turn_id="critique",
        agent_id=agent.agent_id,
        role=agent.role,
        instruction=agent.instruction,
        objective="Check the candidate.",
        dependency_outputs=(
            AgentDependencyOutput(
                turn_id="analysis",
                agent_id="analyst",
                content=dependency_content,
                content_sha256=sha256_bytes(dependency_content.encode("utf-8")),
            ),
        ),
    )


def _backend(tmp_path):
    fingerprint = "fixture-backend-v1"
    spec = _spec(fingerprint)
    agent = AgentSpec(
        agent_id="critic",
        role="critique",
        instruction="Find one concrete error.",
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
    )
    store = ArtifactStore(tmp_path / "cas")
    replay = ReplayBackend(
        fingerprint=fingerprint,
        default_response=BackendResponse(
            output="checked",
            usage=BackendTokenUsage(input_tokens=10, output_tokens=2),
        ),
    )
    ledger = AttemptLedger(
        store,
        ledger_id="multi-agent-adapter-test",
        budget=AttemptBudget(
            max_attempts=1,
            max_total_tokens=128,
            max_tokens_per_attempt=128,
        ),
    )
    runner = FixedModelRunner(spec=spec, backend=replay, attempt_ledger=ledger)
    return agent, ReceiptBackedAgentBackend(
        agent=agent,
        runner=runner,
        seed_resolver=lambda _request: 17,
    )


def test_rendered_turn_contains_only_declared_dependency() -> None:
    spec = _spec("fixture-backend-v1")
    agent = AgentSpec(
        agent_id="critic",
        role="critique",
        instruction="Find one concrete error.",
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
    )

    rendered = json.loads(render_agent_turn_user_prompt(_request(agent)))

    assert rendered["task"] == {
        "payload": '{"question":"choose"}',
        "task_id": "task-1",
    }
    assert rendered["dependency_outputs"] == [
        {
            "agent_id": "analyst",
            "content": "candidate",
            "content_sha256": sha256_bytes(b"candidate"),
            "turn_id": "analysis",
        }
    ]
    assert "score" not in rendered


def test_receipt_backend_persists_and_binds_execution(tmp_path) -> None:
    agent, backend = _backend(tmp_path)
    request = _request(agent)

    response = backend.invoke(request)

    assert response.content == "checked"
    assert response.request_fingerprint == request.fingerprint
    assert response.input_tokens == 10
    assert response.output_tokens == 2
    assert len(backend.records) == 1
    request_fingerprint, record = backend.records[0]
    assert request_fingerprint == request.fingerprint
    assert record.execution.request.seed == 17
    assert record.execution_ref.sha256
    assert record.outcome_ref.sha256


def test_receipt_backend_rejects_request_for_another_instruction(tmp_path) -> None:
    agent, backend = _backend(tmp_path)
    request = _request(agent).model_copy(update={"instruction": "changed"})

    with pytest.raises(ReceiptBackedAgentBackendError, match="bound agent"):
        backend.invoke(request)

    assert backend.records == ()
