from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.agents.contracts import (
    AGENT_TEAM_MEDIA_TYPE,
    AgentSpec,
    AgentTask,
    AgentTeamManifest,
    AgentTurn,
    AgentTurnRequest,
    AgentTurnResponse,
    AgentWorkflow,
    MultiAgentRun,
)
from spiral_harness.agents.evolution import (
    AgentInstructionMutation,
    materialize_agent_instruction_candidate,
    rebind_agent_workflow,
)
from spiral_harness.agents.orchestration import MultiAgentHarnessRunner, MultiAgentRunError
from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.experiment import MutationPolicy
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ComponentKind,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.harness.registry import HarnessRegistry


class ScriptedBackend:
    def __init__(self, agent_id: str, *, content: str | None = None) -> None:
        self.agent_id = agent_id
        self.model_fingerprint = "fixed-model-v1"
        self.runtime_fingerprint = "scripted-runtime-v1"
        self.content = content
        self.requests: list[AgentTurnRequest] = []

    def invoke(self, request: AgentTurnRequest) -> AgentTurnResponse:
        self.requests.append(request)
        dependencies = ",".join(item.turn_id for item in request.dependency_outputs) or "none"
        content = self.content or f"{self.agent_id}:{request.objective}:deps={dependencies}"
        return AgentTurnResponse(
            request_fingerprint=request.fingerprint,
            turn_id=request.turn_id,
            agent_id=request.agent_id,
            model_fingerprint=self.model_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            content=content,
            input_tokens=10,
            output_tokens=4,
        )


def team(**updates: object) -> AgentTeamManifest:
    values: dict[str, object] = {
        "team_id": "research-team",
        "coordinator_id": "coordinator",
        "agents": (
            AgentSpec(
                agent_id="analyst",
                role="analysis",
                instruction="Find a concrete solution.",
                model_fingerprint="fixed-model-v1",
                runtime_fingerprint="scripted-runtime-v1",
            ),
            AgentSpec(
                agent_id="critic",
                role="critique",
                instruction="Find unsupported assumptions.",
                model_fingerprint="fixed-model-v1",
                runtime_fingerprint="scripted-runtime-v1",
            ),
            AgentSpec(
                agent_id="coordinator",
                role="synthesis",
                instruction="Resolve disagreements and return the final answer.",
                model_fingerprint="fixed-model-v1",
                runtime_fingerprint="scripted-runtime-v1",
            ),
        ),
        "max_turns": 3,
        "max_context_bytes": 8_192,
        "max_output_bytes_per_turn": 1_024,
        "max_total_output_bytes": 3_072,
        "max_total_tokens": 100,
    }
    values.update(updates)
    return AgentTeamManifest(**values)


def workflow(agent_team: AgentTeamManifest) -> AgentWorkflow:
    return AgentWorkflow(
        workflow_id="analysis-critique-synthesis",
        team_fingerprint=agent_team.fingerprint,
        turns=(
            AgentTurn(turn_id="analysis", agent_id="analyst", objective="analyze"),
            AgentTurn(
                turn_id="critique",
                agent_id="critic",
                objective="critique",
                depends_on=("analysis",),
            ),
            AgentTurn(
                turn_id="final",
                agent_id="coordinator",
                objective="synthesize",
                depends_on=("analysis", "critique"),
            ),
        ),
        final_turn_id="final",
    )


def backends(**overrides: ScriptedBackend) -> dict[str, ScriptedBackend]:
    values = {
        agent_id: ScriptedBackend(agent_id) for agent_id in ("analyst", "critic", "coordinator")
    }
    values.update(overrides)
    return values


def hypothesis() -> MutationHypothesis:
    return MutationHypothesis(
        evidence_refs=(ArtifactRef(sha256="a" * 64, size=1, media_type="application/json"),),
        where="critic instruction",
        why="the critic accepted unsupported claims",
        expected_activation="the critic receives the analyst output",
        expected_adherence="the critic cites one concrete unsupported assumption",
        expected_behavior="the coordinator receives a more specific critique",
        expected_benefit="fewer unsupported final answers",
        protected_slices=("already-correct answers",),
        falsifier="paired gate accuracy does not improve",
        negative_control="retain the parent critic instruction",
        risks=("over-criticism",),
    )


def test_runner_exposes_only_declared_dependencies_and_returns_score_free_run() -> None:
    agent_team = team()
    agent_backends = backends()
    run = MultiAgentHarnessRunner(
        team=agent_team,
        workflow=workflow(agent_team),
        backends=agent_backends,
    ).execute(AgentTask(task_id="task-1", payload="Explain the result."))

    assert tuple(record.request.turn_id for record in run.records) == (
        "analysis",
        "critique",
        "final",
    )
    assert agent_backends["analyst"].requests[0].dependency_outputs == ()
    assert tuple(
        item.turn_id for item in agent_backends["critic"].requests[0].dependency_outputs
    ) == ("analysis",)
    assert tuple(
        item.turn_id for item in agent_backends["coordinator"].requests[0].dependency_outputs
    ) == ("analysis", "critique")
    assert run.final_output == "coordinator:synthesize:deps=analysis,critique"
    assert not hasattr(run, "score")
    assert run.total_input_tokens == 30
    assert run.total_output_tokens == 12


def test_workflow_rejects_forward_dependency() -> None:
    agent_team = team()
    with pytest.raises(ValidationError, match="earlier turns"):
        AgentWorkflow(
            workflow_id="invalid",
            team_fingerprint=agent_team.fingerprint,
            turns=(
                AgentTurn(
                    turn_id="analysis",
                    agent_id="analyst",
                    objective="analyze",
                    depends_on=("future",),
                ),
                AgentTurn(turn_id="future", agent_id="coordinator", objective="finish"),
            ),
            final_turn_id="future",
        )


def test_runner_fails_closed_on_backend_identity_or_budget_drift() -> None:
    agent_team = team()
    wrong = ScriptedBackend("critic")
    wrong.model_fingerprint = "other-model"
    with pytest.raises(MultiAgentRunError, match="identity differs"):
        MultiAgentHarnessRunner(
            team=agent_team,
            workflow=workflow(agent_team),
            backends=backends(critic=wrong),
        )

    oversized = ScriptedBackend("critic", content="x" * 1_025)
    runner = MultiAgentHarnessRunner(
        team=agent_team,
        workflow=workflow(agent_team),
        backends=backends(critic=oversized),
    )
    with pytest.raises(MultiAgentRunError, match="per-turn output budget") as error:
        runner.execute(AgentTask(task_id="task-1", payload="Explain the result."))
    assert len(error.value.completed_records) == 1


def test_instruction_evolution_is_atomic_and_uses_existing_harness_registry() -> None:
    parent_team = team()
    parent_workflow = workflow(parent_team)
    critic = next(agent for agent in parent_team.agents if agent.agent_id == "critic")
    instruction_mutation = AgentInstructionMutation(
        parent_team_fingerprint=parent_team.fingerprint,
        target_agent_id="critic",
        before_instruction_sha256=sha256_bytes(critic.instruction.encode("utf-8")),
        after_instruction="Identify and quote one unsupported assumption.",
        hypothesis=hypothesis(),
    )
    candidate_team, candidate_mutation, candidate_bytes = materialize_agent_instruction_candidate(
        parent_team, instruction_mutation
    )
    candidate_workflow = rebind_agent_workflow(
        parent_workflow,
        parent_team=parent_team,
        candidate_team=candidate_team,
    )

    assert candidate_workflow.turns == parent_workflow.turns
    assert candidate_workflow.team_fingerprint == candidate_team.fingerprint
    changed = [
        (before.agent_id, before.instruction, after.instruction)
        for before, after in zip(parent_team.agents, candidate_team.agents, strict=True)
        if before != after
    ]
    assert changed == [
        (
            "critic",
            "Find unsupported assumptions.",
            "Identify and quote one unsupported assumption.",
        )
    ]

    parent_manifest = HarnessManifest(
        model_fingerprint="fixed-model-v1",
        runtime_fingerprint="scripted-runtime-v1",
        trusted_plane_version="test-v1",
        components=(candidate_mutation.before,),
    )
    parent_bytes = canonical_json_bytes(parent_manifest)
    child = HarnessRegistry(
        MutationPolicy(
            allowed_kinds=(ComponentKind.CONTROL_FLOW,),
            allowed_component_names=("agent-team",),
            allowed_media_types=(AGENT_TEAM_MEDIA_TYPE,),
            max_artifact_size_bytes=16_384,
        )
    ).apply_mutation(
        parent=parent_manifest,
        parent_ref=ArtifactRef(
            sha256=sha256_bytes(parent_bytes),
            size=len(parent_bytes),
            media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        ),
        mutation=candidate_mutation,
        artifact_bytes=candidate_bytes,
        artifact_media_type=AGENT_TEAM_MEDIA_TYPE,
    )

    assert child.components == (candidate_mutation.after,)
    assert child.parent is not None


def test_instruction_mutation_rejects_parent_drift() -> None:
    parent_team = team()
    critic = next(agent for agent in parent_team.agents if agent.agent_id == "critic")
    mutation = AgentInstructionMutation(
        parent_team_fingerprint="f" * 64,
        target_agent_id="critic",
        before_instruction_sha256=sha256_bytes(critic.instruction.encode("utf-8")),
        after_instruction="Challenge every assumption.",
        hypothesis=hypothesis(),
    )
    with pytest.raises(ValueError, match="another parent team"):
        materialize_agent_instruction_candidate(parent_team, mutation)


def test_run_contract_rejects_cross_task_or_forged_dependency_transcript() -> None:
    agent_team = team()
    run = MultiAgentHarnessRunner(
        team=agent_team,
        workflow=workflow(agent_team),
        backends=backends(),
    ).execute(AgentTask(task_id="task-1", payload="Explain the result."))
    forged_request = run.records[1].request.model_copy(update={"task_id": "other-task"})
    forged_response = run.records[1].response.model_copy(
        update={"request_fingerprint": forged_request.fingerprint}
    )
    forged_record = run.records[1].model_copy(
        update={"request": forged_request, "response": forged_response}
    )
    forged = run.model_copy(update={"records": (run.records[0], forged_record, run.records[2])})
    with pytest.raises(ValidationError, match="another task"):
        MultiAgentRun.model_validate(forged.model_dump(mode="python"), strict=True)

    wrong_source = (
        run.records[1].request.dependency_outputs[0].model_copy(update={"agent_id": "coordinator"})
    )
    forged_request = run.records[1].request.model_copy(
        update={"dependency_outputs": (wrong_source,)}
    )
    forged_response = run.records[1].response.model_copy(
        update={"request_fingerprint": forged_request.fingerprint}
    )
    forged_record = run.records[1].model_copy(
        update={"request": forged_request, "response": forged_response}
    )
    forged = run.model_copy(update={"records": (run.records[0], forged_record, run.records[2])})
    with pytest.raises(ValidationError, match="differs from its source turn"):
        MultiAgentRun.model_validate(forged.model_dump(mode="python"), strict=True)


def test_workflow_rebind_rejects_non_instruction_team_drift() -> None:
    parent = team()
    changed_agents = tuple(
        agent.model_copy(update={"role": "untrusted-new-role"})
        if agent.agent_id == "critic"
        else agent
        for agent in parent.agents
    )
    candidate = AgentTeamManifest.model_validate(
        parent.model_copy(update={"agents": changed_agents}),
        strict=True,
    )
    with pytest.raises(ValueError, match="agent configuration"):
        rebind_agent_workflow(workflow(parent), parent_team=parent, candidate_team=candidate)
