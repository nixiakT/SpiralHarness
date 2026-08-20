"""Immutable contracts for deterministic multi-agent harness execution."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

AGENT_TEAM_MEDIA_TYPE = "application/vnd.spiral-harness.agent-team.v1+json"
AGENT_WORKFLOW_MEDIA_TYPE = "application/vnd.spiral-harness.agent-workflow.v1+json"
MULTI_AGENT_RUN_MEDIA_TYPE = "application/vnd.spiral-harness.multi-agent-run.v1+json"


class AgentSpec(ImmutableModel):
    """One least-authority role in a frozen multi-agent team."""

    agent_id: NonEmptyStr
    role: NonEmptyStr
    instruction: NonEmptyStr
    model_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    capabilities: tuple[NonEmptyStr, ...] = ()

    @field_validator("capabilities")
    @classmethod
    def canonicalize_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("agent capabilities must be unique")
        return ordered


class AgentTeamManifest(ImmutableModel):
    """Complete immutable configuration of a multi-agent harness."""

    schema_version: Literal["1"] = "1"
    team_id: NonEmptyStr
    coordinator_id: NonEmptyStr
    agents: Annotated[tuple[AgentSpec, ...], Field(min_length=2)]
    max_turns: Annotated[int, Field(ge=1, strict=True)]
    max_context_bytes: Annotated[int, Field(ge=1, strict=True)]
    max_output_bytes_per_turn: Annotated[int, Field(ge=1, strict=True)]
    max_total_output_bytes: Annotated[int, Field(ge=1, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=1, strict=True)]
    execution_mode: Literal["deterministic-sequential"] = "deterministic-sequential"

    @field_validator("agents")
    @classmethod
    def canonicalize_agents(cls, agents: tuple[AgentSpec, ...]) -> tuple[AgentSpec, ...]:
        ordered = tuple(sorted(agents, key=lambda agent: agent.agent_id))
        ids = tuple(agent.agent_id for agent in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("agent IDs must be unique")
        return ordered

    @model_validator(mode="after")
    def coordinator_belongs_to_team(self) -> AgentTeamManifest:
        if self.coordinator_id not in {agent.agent_id for agent in self.agents}:
            raise ValueError("coordinator_id must identify a member of the team")
        if self.max_total_output_bytes < self.max_output_bytes_per_turn:
            raise ValueError("total output budget must cover at least one turn")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AgentTurn(ImmutableModel):
    """One scheduled model call with an explicit upstream visibility set."""

    turn_id: NonEmptyStr
    agent_id: NonEmptyStr
    objective: NonEmptyStr
    depends_on: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def dependencies_are_unique_and_not_self(self) -> AgentTurn:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("turn dependencies must be unique")
        if self.turn_id in self.depends_on:
            raise ValueError("a turn cannot depend on itself")
        return self


class AgentWorkflow(ImmutableModel):
    """A deterministic DAG serialized in execution order."""

    schema_version: Literal["1"] = "1"
    workflow_id: NonEmptyStr
    team_fingerprint: Sha256
    turns: Annotated[tuple[AgentTurn, ...], Field(min_length=1)]
    final_turn_id: NonEmptyStr

    @model_validator(mode="after")
    def turns_form_an_ordered_dag(self) -> AgentWorkflow:
        seen: set[str] = set()
        for turn in self.turns:
            if turn.turn_id in seen:
                raise ValueError("turn IDs must be unique")
            missing = tuple(dependency for dependency in turn.depends_on if dependency not in seen)
            if missing:
                raise ValueError(
                    "turn dependencies must reference earlier turns: " + ", ".join(missing)
                )
            seen.add(turn.turn_id)
        if self.final_turn_id != self.turns[-1].turn_id:
            raise ValueError("final_turn_id must identify the last scheduled turn")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AgentTask(ImmutableModel):
    """Unscored input supplied to one frozen team and workflow."""

    task_id: NonEmptyStr
    payload: NonEmptyStr

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AgentDependencyOutput(ImmutableModel):
    """One explicitly disclosed upstream message."""

    turn_id: NonEmptyStr
    agent_id: NonEmptyStr
    content: NonEmptyStr
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_hash_matches(self) -> AgentDependencyOutput:
        if sha256_bytes(self.content.encode("utf-8")) != self.content_sha256:
            raise ValueError("dependency output hash does not match its content")
        return self


class AgentTurnRequest(ImmutableModel):
    """Exact score-free context disclosed to one agent call."""

    schema_version: Literal["1"] = "1"
    team_fingerprint: Sha256
    workflow_fingerprint: Sha256
    task_id: NonEmptyStr
    task_payload: NonEmptyStr
    turn_id: NonEmptyStr
    agent_id: NonEmptyStr
    role: NonEmptyStr
    instruction: NonEmptyStr
    objective: NonEmptyStr
    dependency_outputs: tuple[AgentDependencyOutput, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AgentTurnResponse(ImmutableModel):
    """Untrusted, score-free response returned by an injected backend."""

    schema_version: Literal["1"] = "1"
    request_fingerprint: Sha256
    turn_id: NonEmptyStr
    agent_id: NonEmptyStr
    model_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    content: NonEmptyStr
    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]

    @property
    def content_sha256(self) -> str:
        return sha256_bytes(self.content.encode("utf-8"))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AgentTurnRecord(ImmutableModel):
    """Request/response closure for one completed turn."""

    request: AgentTurnRequest
    response: AgentTurnResponse

    @model_validator(mode="after")
    def response_is_bound_to_request(self) -> AgentTurnRecord:
        if self.response.request_fingerprint != self.request.fingerprint:
            raise ValueError("response belongs to another request")
        if (self.response.turn_id, self.response.agent_id) != (
            self.request.turn_id,
            self.request.agent_id,
        ):
            raise ValueError("response belongs to another agent turn")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class MultiAgentRun(ImmutableModel):
    """Successful score-free transcript of one frozen workflow."""

    schema_version: Literal["1"] = "1"
    team_fingerprint: Sha256
    workflow_fingerprint: Sha256
    task_fingerprint: Sha256
    records: Annotated[tuple[AgentTurnRecord, ...], Field(min_length=1)]
    final_turn_id: NonEmptyStr
    final_output: NonEmptyStr
    total_input_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_output_tokens: Annotated[int, Field(ge=0, strict=True)]
    total_output_bytes: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def totals_and_final_output_match_records(self) -> MultiAgentRun:
        requests = tuple(record.request for record in self.records)
        if any(
            request.team_fingerprint != self.team_fingerprint
            or request.workflow_fingerprint != self.workflow_fingerprint
            for request in requests
        ):
            raise ValueError("run records belong to another team or workflow")
        task = AgentTask(task_id=requests[0].task_id, payload=requests[0].task_payload)
        if self.task_fingerprint != task.fingerprint or any(
            (request.task_id, request.task_payload) != (task.task_id, task.payload)
            for request in requests
        ):
            raise ValueError("run records belong to another task")

        records_by_turn: dict[str, AgentTurnRecord] = {}
        for record in self.records:
            turn_id = record.request.turn_id
            if turn_id in records_by_turn:
                raise ValueError("run turn IDs must be unique")
            dependency_ids = tuple(
                dependency.turn_id for dependency in record.request.dependency_outputs
            )
            if len(dependency_ids) != len(set(dependency_ids)):
                raise ValueError("run dependency outputs must be unique")
            for dependency in record.request.dependency_outputs:
                source = records_by_turn.get(dependency.turn_id)
                if source is None:
                    raise ValueError("run dependency output does not reference an earlier turn")
                if dependency != AgentDependencyOutput(
                    turn_id=source.request.turn_id,
                    agent_id=source.request.agent_id,
                    content=source.response.content,
                    content_sha256=source.response.content_sha256,
                ):
                    raise ValueError("run dependency output differs from its source turn")
            records_by_turn[turn_id] = record

        if self.records[-1].request.turn_id != self.final_turn_id:
            raise ValueError("final turn does not match the final record")
        if self.records[-1].response.content != self.final_output:
            raise ValueError("final output does not match the final response")
        if self.total_input_tokens != sum(record.response.input_tokens for record in self.records):
            raise ValueError("total input tokens do not match the transcript")
        if self.total_output_tokens != sum(
            record.response.output_tokens for record in self.records
        ):
            raise ValueError("total output tokens do not match the transcript")
        expected_bytes = sum(
            len(record.response.content.encode("utf-8")) for record in self.records
        )
        if self.total_output_bytes != expected_bytes:
            raise ValueError("total output bytes do not match the transcript")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [
    "AGENT_TEAM_MEDIA_TYPE",
    "AGENT_WORKFLOW_MEDIA_TYPE",
    "MULTI_AGENT_RUN_MEDIA_TYPE",
    "AgentDependencyOutput",
    "AgentSpec",
    "AgentTask",
    "AgentTeamManifest",
    "AgentTurn",
    "AgentTurnRecord",
    "AgentTurnRequest",
    "AgentTurnResponse",
    "AgentWorkflow",
    "MultiAgentRun",
]
