"""Bounded self-evolution helpers for multi-agent harness manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from spiral_harness.agents.contracts import (
    AGENT_TEAM_MEDIA_TYPE,
    AgentSpec,
    AgentTeamManifest,
    AgentWorkflow,
)
from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import (
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    ImmutableModel,
    MutationHypothesis,
    NonEmptyStr,
    Sha256,
)


class AgentInstructionMutation(ImmutableModel):
    """One hypothesis-bound replacement of one agent's instruction."""

    schema_version: Literal["1"] = "1"
    parent_team_fingerprint: Sha256
    target_agent_id: NonEmptyStr
    before_instruction_sha256: Sha256
    after_instruction: NonEmptyStr
    hypothesis: MutationHypothesis

    @model_validator(mode="after")
    def mutation_changes_instruction_bytes(self) -> AgentInstructionMutation:
        if sha256_bytes(self.after_instruction.encode("utf-8")) == self.before_instruction_sha256:
            raise ValueError("agent instruction mutation must not be a no-op")
        return self


def apply_agent_instruction_mutation(
    parent: AgentTeamManifest,
    mutation: AgentInstructionMutation,
) -> AgentTeamManifest:
    """Apply exactly one instruction change while freezing every other team field."""

    checked_parent = AgentTeamManifest.model_validate(parent, strict=True)
    checked_mutation = AgentInstructionMutation.model_validate(mutation, strict=True)
    if checked_mutation.parent_team_fingerprint != checked_parent.fingerprint:
        raise ValueError("instruction mutation belongs to another parent team")
    by_id = {agent.agent_id: agent for agent in checked_parent.agents}
    target = by_id.get(checked_mutation.target_agent_id)
    if target is None:
        raise ValueError("instruction mutation targets an unknown agent")
    if sha256_bytes(target.instruction.encode("utf-8")) != (
        checked_mutation.before_instruction_sha256
    ):
        raise ValueError("instruction mutation does not match the parent instruction")

    replacement = AgentSpec.model_validate(
        target.model_copy(update={"instruction": checked_mutation.after_instruction}),
        strict=True,
    )
    candidate = checked_parent.model_copy(
        update={
            "agents": tuple(
                replacement if agent.agent_id == target.agent_id else agent
                for agent in checked_parent.agents
            )
        }
    )
    return AgentTeamManifest.model_validate(candidate, strict=True)


def rebind_agent_workflow(
    workflow: AgentWorkflow,
    *,
    parent_team: AgentTeamManifest,
    candidate_team: AgentTeamManifest,
) -> AgentWorkflow:
    """Bind an unchanged workflow to an instruction-only child team."""

    checked_workflow = AgentWorkflow.model_validate(workflow, strict=True)
    checked_parent = AgentTeamManifest.model_validate(parent_team, strict=True)
    checked_candidate = AgentTeamManifest.model_validate(candidate_team, strict=True)
    if checked_workflow.team_fingerprint != checked_parent.fingerprint:
        raise ValueError("workflow belongs to another parent team")
    parent_team_fields = checked_parent.model_dump(mode="python", exclude={"agents"})
    candidate_team_fields = checked_candidate.model_dump(mode="python", exclude={"agents"})
    if parent_team_fields != candidate_team_fields:
        raise ValueError("instruction-only evolution cannot change team configuration")
    if tuple(agent.agent_id for agent in checked_parent.agents) != tuple(
        agent.agent_id for agent in checked_candidate.agents
    ):
        raise ValueError("instruction-only evolution cannot change the agent roster")
    changed_instructions = 0
    for parent_agent, candidate_agent in zip(
        checked_parent.agents,
        checked_candidate.agents,
        strict=True,
    ):
        if parent_agent.model_dump(mode="python", exclude={"instruction"}) != (
            candidate_agent.model_dump(mode="python", exclude={"instruction"})
        ):
            raise ValueError("instruction-only evolution cannot change agent configuration")
        changed_instructions += parent_agent.instruction != candidate_agent.instruction
    if changed_instructions != 1:
        raise ValueError("instruction-only evolution must change exactly one instruction")
    rebound = checked_workflow.model_copy(
        update={"team_fingerprint": checked_candidate.fingerprint}
    )
    return AgentWorkflow.model_validate(rebound, strict=True)


def agent_team_component(
    team: AgentTeamManifest,
    *,
    component_name: str = "agent-team",
) -> HarnessComponentRef:
    """Represent a complete team as one atomic control-flow harness component."""

    checked_team = AgentTeamManifest.model_validate(team, strict=True)
    payload = canonical_json_bytes(checked_team)
    return HarnessComponentRef(
        name=component_name,
        kind=ComponentKind.CONTROL_FLOW,
        artifact=ArtifactRef(
            sha256=sha256_bytes(payload),
            size=len(payload),
            media_type=AGENT_TEAM_MEDIA_TYPE,
        ),
    )


def materialize_agent_instruction_candidate(
    parent: AgentTeamManifest,
    mutation: AgentInstructionMutation,
    *,
    component_name: str = "agent-team",
) -> tuple[AgentTeamManifest, CandidateMutation, bytes]:
    """Build the child team, generic atomic mutation, and exact candidate bytes."""

    checked_mutation = AgentInstructionMutation.model_validate(mutation, strict=True)
    candidate = apply_agent_instruction_mutation(parent, checked_mutation)
    before = agent_team_component(parent, component_name=component_name)
    after = agent_team_component(candidate, component_name=component_name)
    payload = canonical_json_bytes(candidate)
    return (
        candidate,
        CandidateMutation(
            target_component=component_name,
            before=before,
            after=after,
            hypothesis=checked_mutation.hypothesis,
        ),
        payload,
    )


__all__ = [
    "AgentInstructionMutation",
    "agent_team_component",
    "apply_agent_instruction_mutation",
    "materialize_agent_instruction_candidate",
    "rebind_agent_workflow",
]
