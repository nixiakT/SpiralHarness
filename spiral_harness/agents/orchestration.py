"""Deterministic, score-free execution of a frozen multi-agent workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from spiral_harness.agents.contracts import (
    AgentDependencyOutput,
    AgentSpec,
    AgentTask,
    AgentTeamManifest,
    AgentTurnRecord,
    AgentTurnRequest,
    AgentTurnResponse,
    AgentWorkflow,
    MultiAgentRun,
)
from spiral_harness.core.canonical import canonical_json_bytes


class MultiAgentRunError(RuntimeError):
    """Raised when a frozen team cannot complete without violating its contract."""

    def __init__(
        self,
        message: str,
        *,
        completed_records: tuple[AgentTurnRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.completed_records = completed_records


@runtime_checkable
class AgentBackend(Protocol):
    """One score-free model/runtime boundary assigned to one agent identity."""

    @property
    def model_fingerprint(self) -> str: ...

    @property
    def runtime_fingerprint(self) -> str: ...

    def invoke(self, request: AgentTurnRequest) -> AgentTurnResponse: ...


class MultiAgentHarnessRunner:
    """Run explicit agent turns without exposing grading or promotion authority."""

    def __init__(
        self,
        *,
        team: AgentTeamManifest,
        workflow: AgentWorkflow,
        backends: Mapping[str, AgentBackend],
    ) -> None:
        self.team = AgentTeamManifest.model_validate(team, strict=True)
        self.workflow = AgentWorkflow.model_validate(workflow, strict=True)
        self._agents = {agent.agent_id: agent for agent in self.team.agents}
        self._backends = dict(backends)
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        if self.workflow.team_fingerprint != self.team.fingerprint:
            raise MultiAgentRunError("workflow belongs to another team manifest")
        if len(self.workflow.turns) > self.team.max_turns:
            raise MultiAgentRunError("workflow exceeds the frozen turn budget")

        scheduled_agents = {turn.agent_id for turn in self.workflow.turns}
        unknown_agents = scheduled_agents - self._agents.keys()
        if unknown_agents:
            raise MultiAgentRunError(
                "workflow schedules unknown agents: " + ", ".join(sorted(unknown_agents))
            )
        unused_agents = self._agents.keys() - scheduled_agents
        if unused_agents:
            raise MultiAgentRunError(
                "team contains unscheduled agents: " + ", ".join(sorted(unused_agents))
            )
        if self.workflow.turns[-1].agent_id != self.team.coordinator_id:
            raise MultiAgentRunError("the coordinator must own the final workflow turn")
        if self._backends.keys() != self._agents.keys():
            missing = self._agents.keys() - self._backends.keys()
            extra = self._backends.keys() - self._agents.keys()
            detail = [
                *(f"missing={agent_id}" for agent_id in sorted(missing)),
                *(f"extra={agent_id}" for agent_id in sorted(extra)),
            ]
            raise MultiAgentRunError("backend roster differs from the team: " + ", ".join(detail))

        for agent_id, agent in self._agents.items():
            backend = self._backends[agent_id]
            if not isinstance(backend, AgentBackend):
                raise TypeError(f"backend for {agent_id!r} does not implement AgentBackend")
            if (
                backend.model_fingerprint != agent.model_fingerprint
                or backend.runtime_fingerprint != agent.runtime_fingerprint
            ):
                raise MultiAgentRunError(f"backend identity differs for agent {agent_id!r}")

    def execute(self, task: AgentTask) -> MultiAgentRun:
        """Execute every turn once in frozen order and return an unscored transcript."""

        checked_task = AgentTask.model_validate(task, strict=True)
        records: list[AgentTurnRecord] = []
        records_by_turn: dict[str, AgentTurnRecord] = {}
        total_tokens = 0
        total_output_bytes = 0

        for turn in self.workflow.turns:
            agent = self._agents[turn.agent_id]
            dependencies = tuple(
                self._dependency_output(records_by_turn[dependency])
                for dependency in turn.depends_on
            )
            request = AgentTurnRequest(
                team_fingerprint=self.team.fingerprint,
                workflow_fingerprint=self.workflow.fingerprint,
                task_id=checked_task.task_id,
                task_payload=checked_task.payload,
                turn_id=turn.turn_id,
                agent_id=agent.agent_id,
                role=agent.role,
                instruction=agent.instruction,
                objective=turn.objective,
                dependency_outputs=dependencies,
            )
            if len(canonical_json_bytes(request)) > self.team.max_context_bytes:
                raise MultiAgentRunError(
                    f"turn {turn.turn_id!r} exceeds the frozen context-byte budget",
                    completed_records=tuple(records),
                )
            response = self._invoke(agent, request, tuple(records))
            record = AgentTurnRecord(request=request, response=response)
            output_bytes = len(response.content.encode("utf-8"))
            if output_bytes > self.team.max_output_bytes_per_turn:
                raise MultiAgentRunError(
                    f"turn {turn.turn_id!r} exceeds the per-turn output budget",
                    completed_records=tuple(records),
                )
            total_output_bytes += output_bytes
            if total_output_bytes > self.team.max_total_output_bytes:
                raise MultiAgentRunError(
                    "run exceeds the frozen total output-byte budget",
                    completed_records=tuple(records),
                )
            total_tokens += response.input_tokens + response.output_tokens
            if total_tokens > self.team.max_total_tokens:
                raise MultiAgentRunError(
                    "run exceeds the frozen total token budget",
                    completed_records=tuple(records),
                )
            records.append(record)
            records_by_turn[turn.turn_id] = record

        final = records[-1]
        return MultiAgentRun(
            team_fingerprint=self.team.fingerprint,
            workflow_fingerprint=self.workflow.fingerprint,
            task_fingerprint=checked_task.fingerprint,
            records=tuple(records),
            final_turn_id=self.workflow.final_turn_id,
            final_output=final.response.content,
            total_input_tokens=sum(record.response.input_tokens for record in records),
            total_output_tokens=sum(record.response.output_tokens for record in records),
            total_output_bytes=total_output_bytes,
        )

    def _invoke(
        self,
        agent: AgentSpec,
        request: AgentTurnRequest,
        completed_records: tuple[AgentTurnRecord, ...],
    ) -> AgentTurnResponse:
        try:
            response = AgentTurnResponse.model_validate(
                self._backends[agent.agent_id].invoke(request),
                strict=True,
            )
        except Exception as exc:
            raise MultiAgentRunError(
                f"backend failed for turn {request.turn_id!r}",
                completed_records=completed_records,
            ) from exc
        expected = (
            request.fingerprint,
            request.turn_id,
            request.agent_id,
            agent.model_fingerprint,
            agent.runtime_fingerprint,
        )
        observed = (
            response.request_fingerprint,
            response.turn_id,
            response.agent_id,
            response.model_fingerprint,
            response.runtime_fingerprint,
        )
        if observed != expected:
            raise MultiAgentRunError(
                f"backend response changed the binding for turn {request.turn_id!r}",
                completed_records=completed_records,
            )
        return response

    @staticmethod
    def _dependency_output(record: AgentTurnRecord) -> AgentDependencyOutput:
        return AgentDependencyOutput(
            turn_id=record.request.turn_id,
            agent_id=record.request.agent_id,
            content=record.response.content,
            content_sha256=record.response.content_sha256,
        )


__all__ = ["AgentBackend", "MultiAgentHarnessRunner", "MultiAgentRunError"]
