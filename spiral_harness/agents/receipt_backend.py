"""Receipt-backed fixed-model adapter for multi-agent turns."""

from __future__ import annotations

from collections.abc import Callable

from spiral_harness.agents.contracts import AgentSpec, AgentTurnRequest, AgentTurnResponse
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.core.models import HARNESS_MANIFEST_MEDIA_TYPE
from spiral_harness.execution.contracts import (
    CandidateTask,
    ExecutionStatus,
    ModelExecutionRecord,
    ResolvedHarness,
)
from spiral_harness.execution.model import FixedModelRunner


class ReceiptBackedAgentBackendError(RuntimeError):
    """One receipt-backed agent turn failed closed."""


def render_agent_turn_user_prompt(request: AgentTurnRequest) -> str:
    """Render the exact score-free turn context sent as the user message."""

    checked = AgentTurnRequest.model_validate(request, strict=True)
    return canonical_json(
        {
            "dependency_outputs": [
                {
                    "agent_id": dependency.agent_id,
                    "content": dependency.content,
                    "content_sha256": dependency.content_sha256,
                    "turn_id": dependency.turn_id,
                }
                for dependency in checked.dependency_outputs
            ],
            "objective": checked.objective,
            "task": {"payload": checked.task_payload, "task_id": checked.task_id},
            "turn": {
                "agent_id": checked.agent_id,
                "role": checked.role,
                "turn_id": checked.turn_id,
            },
        }
    )


class ReceiptBackedAgentBackend:
    """Translate one agent turn into the existing receipt-backed execution plane."""

    def __init__(
        self,
        *,
        agent: AgentSpec,
        runner: FixedModelRunner,
        seed_resolver: Callable[[AgentTurnRequest], int],
    ) -> None:
        self._agent = AgentSpec.model_validate(agent, strict=True)
        if not isinstance(runner, FixedModelRunner):
            raise TypeError("runner must be a FixedModelRunner")
        if not callable(seed_resolver):
            raise TypeError("seed_resolver must be callable")
        if (
            self._agent.model_fingerprint != runner.spec.model_fingerprint
            or self._agent.runtime_fingerprint != runner.spec.runtime_fingerprint
        ):
            raise ValueError("agent identity differs from the frozen model runner")
        manifest_payload = {
            "agent_id": self._agent.agent_id,
            "domain": "spiral-harness/receipt-backed-agent-prompt/v1",
            "instruction": self._agent.instruction,
            "model_fingerprint": self._agent.model_fingerprint,
            "role": self._agent.role,
            "runtime_fingerprint": self._agent.runtime_fingerprint,
        }
        self._harness_ref = runner.repository.put_json(
            manifest_payload,
            media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        )
        self._harness = ResolvedHarness.from_prompt(
            harness_ref=self._harness_ref,
            system_prompt=self._agent.instruction,
        )
        self._runner = runner
        self._seed_resolver = seed_resolver
        self._records: list[tuple[str, ModelExecutionRecord]] = []

    @property
    def model_fingerprint(self) -> str:
        return self._agent.model_fingerprint

    @property
    def runtime_fingerprint(self) -> str:
        return self._agent.runtime_fingerprint

    @property
    def records(self) -> tuple[tuple[str, ModelExecutionRecord], ...]:
        """Return request-bound immutable execution and accounting references."""

        return tuple(self._records)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "agent": self._agent,
                "domain": "spiral-harness/receipt-backed-agent-backend/v1",
                "harness_ref": self._harness_ref,
                "runner_fingerprint": self._runner.fingerprint,
            }
        )

    def invoke(self, request: AgentTurnRequest) -> AgentTurnResponse:
        checked = AgentTurnRequest.model_validate(request, strict=True)
        if (
            checked.agent_id != self._agent.agent_id
            or checked.role != self._agent.role
            or checked.instruction != self._agent.instruction
        ):
            raise ReceiptBackedAgentBackendError("turn request differs from the bound agent")
        seed = self._seed_resolver(checked)
        if type(seed) is not int or seed < 0:
            raise ReceiptBackedAgentBackendError("seed resolver returned an invalid seed")
        record = self._runner.execute_record(
            CandidateTask(
                task_id=f"{checked.task_id}/{checked.turn_id}",
                question=render_agent_turn_user_prompt(checked),
            ),
            harness=self._harness,
            seed=seed,
        )
        self._records.append((checked.fingerprint, record))
        execution = record.execution
        if execution.status is not ExecutionStatus.COMPLETED or execution.output is None:
            raise ReceiptBackedAgentBackendError("receipt-backed model execution failed")
        return AgentTurnResponse(
            request_fingerprint=checked.fingerprint,
            turn_id=checked.turn_id,
            agent_id=checked.agent_id,
            model_fingerprint=self.model_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            content=execution.output,
            input_tokens=execution.usage.input_tokens,
            output_tokens=execution.usage.output_tokens,
        )


__all__ = [
    "ReceiptBackedAgentBackend",
    "ReceiptBackedAgentBackendError",
    "render_agent_turn_user_prompt",
]
