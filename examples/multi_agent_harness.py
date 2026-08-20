"""Run a provider-free multi-agent harness and one instruction revision."""

from __future__ import annotations

import json

from spiral_harness.agents.contracts import (
    AgentSpec,
    AgentTask,
    AgentTeamManifest,
    AgentTurn,
    AgentTurnRequest,
    AgentTurnResponse,
    AgentWorkflow,
)
from spiral_harness.agents.evolution import (
    AgentInstructionMutation,
    apply_agent_instruction_mutation,
    rebind_agent_workflow,
)
from spiral_harness.agents.orchestration import MultiAgentHarnessRunner
from spiral_harness.core.canonical import sha256_bytes
from spiral_harness.core.models import ArtifactRef, MutationHypothesis


class ExampleBackend:
    """Deterministic stand-in; replace this boundary with a real model adapter."""

    model_fingerprint = "example-fixed-model-v1"
    runtime_fingerprint = "example-runtime-v1"

    def invoke(self, request: AgentTurnRequest) -> AgentTurnResponse:
        visible = ", ".join(output.content for output in request.dependency_outputs)
        content = f"[{request.role}] {request.objective}"
        if visible:
            content += f" | visible: {visible}"
        return AgentTurnResponse(
            request_fingerprint=request.fingerprint,
            turn_id=request.turn_id,
            agent_id=request.agent_id,
            model_fingerprint=self.model_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            content=content,
            input_tokens=8,
            output_tokens=4,
        )


def build_team() -> AgentTeamManifest:
    return AgentTeamManifest(
        team_id="example-team",
        coordinator_id="coordinator",
        agents=tuple(
            AgentSpec(
                agent_id=agent_id,
                role=role,
                instruction=instruction,
                model_fingerprint=ExampleBackend.model_fingerprint,
                runtime_fingerprint=ExampleBackend.runtime_fingerprint,
            )
            for agent_id, role, instruction in (
                ("analyst", "analysis", "Produce one candidate solution."),
                ("critic", "critique", "Find unsupported assumptions."),
                ("coordinator", "synthesis", "Resolve the critique and answer."),
            )
        ),
        max_turns=3,
        max_context_bytes=16_384,
        max_output_bytes_per_turn=2_048,
        max_total_output_bytes=6_144,
        max_total_tokens=256,
    )


def build_workflow(team: AgentTeamManifest) -> AgentWorkflow:
    return AgentWorkflow(
        workflow_id="analyze-critique-synthesize",
        team_fingerprint=team.fingerprint,
        turns=(
            AgentTurn(turn_id="analysis", agent_id="analyst", objective="Analyze the task."),
            AgentTurn(
                turn_id="critique",
                agent_id="critic",
                objective="Critique the analysis.",
                depends_on=("analysis",),
            ),
            AgentTurn(
                turn_id="final",
                agent_id="coordinator",
                objective="Return the final answer.",
                depends_on=("analysis", "critique"),
            ),
        ),
        final_turn_id="final",
    )


def main() -> None:
    parent = build_team()
    parent_workflow = build_workflow(parent)
    critic = next(agent for agent in parent.agents if agent.agent_id == "critic")
    evidence_ref = ArtifactRef(sha256="a" * 64, size=1, media_type="application/json")
    mutation = AgentInstructionMutation(
        parent_team_fingerprint=parent.fingerprint,
        target_agent_id="critic",
        before_instruction_sha256=sha256_bytes(critic.instruction.encode("utf-8")),
        after_instruction="Quote one unsupported assumption and explain the failure mode.",
        hypothesis=MutationHypothesis(
            evidence_refs=(evidence_ref,),
            where="critic instruction",
            why="generic criticism was not actionable",
            expected_activation="the critic reads the analysis",
            expected_adherence="the critic quotes one assumption",
            expected_behavior="the coordinator receives actionable criticism",
            expected_benefit="fewer unsupported answers",
            protected_slices=("correct answers",),
            falsifier="the paired gate does not improve",
            negative_control="retain the parent instruction",
            risks=("excessive criticism",),
        ),
    )
    candidate = apply_agent_instruction_mutation(parent, mutation)
    candidate_workflow = rebind_agent_workflow(
        parent_workflow,
        parent_team=parent,
        candidate_team=candidate,
    )
    runner = MultiAgentHarnessRunner(
        team=candidate,
        workflow=candidate_workflow,
        backends={agent.agent_id: ExampleBackend() for agent in candidate.agents},
    )
    run = runner.execute(AgentTask(task_id="example", payload="Should this candidate be promoted?"))
    print(
        json.dumps(
            {
                "candidate_team_fingerprint": candidate.fingerprint,
                "run_fingerprint": run.fingerprint,
                "final_output": run.final_output,
                "score": None,
                "next_step": "evaluate parent and candidate through the trusted paired gate",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
