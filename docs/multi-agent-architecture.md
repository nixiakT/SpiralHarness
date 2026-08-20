# Multi-agent self-evolution architecture

> **Status:** implemented score-free kernel with deterministic sequential
> execution, receipt-backed LiteLLM turns, atomic single-agent instruction
> evolution, and one non-reportable BFCL public-development comparison. It is
> not yet confirmatory evidence that a team generally improves benchmarks.

## Goal

SpiralHarness treats a multi-agent system as one versioned harness rather than
as a collection of unrestricted chat sessions. A complete team, its workflow,
its information flow, and its budgets are frozen before execution. The team may
produce a candidate revision, but the existing trusted verification plane—not
the team—grades and promotes it.

```text
                           frozen protocol
                                 |
                    +------------v-------------+
                    | AgentTeamManifest        |
                    | roles / instructions     |
                    | model + runtime identity |
                    | byte + token budgets     |
                    +------------+-------------+
                                 |
                    +------------v-------------+
                    | AgentWorkflow            |
                    | ordered DAG / visibility |
                    +------------+-------------+
                                 |
              +------------------v------------------+
              | MultiAgentHarnessRunner             |
              | score-free, deterministic, bounded  |
              +------------------+------------------+
                                 |
                       immutable transcript
                                 |
                 +---------------v---------------+
                 | trusted benchmark + verifier  |
                 | grade / compare / gate        |
                 +---------------+---------------+
                                 |
                    promote / reject / inconclusive
```

## Implemented contracts

`spiral_harness.agents.contracts` owns the immutable runtime values:

- `AgentSpec` freezes one role, instruction, model/runtime identity, and
  declarative capability labels;
- `AgentTeamManifest` requires at least two uniquely identified agents, one
  coordinator, and explicit turn/context/output/token ceilings;
- `AgentWorkflow` is an ordered DAG. Every dependency must identify an earlier
  turn, making both execution order and message visibility replayable;
- `AgentTurnRequest` contains only the task and declared dependency outputs for
  that turn. It does not expose a repository handle, grader, score, sibling
  output, or undeclared history;
- `MultiAgentRun` closes the complete score-free transcript and accounting.

`MultiAgentHarnessRunner` requires an exact backend binding for every agent.
The backend's model and runtime fingerprints must match the frozen `AgentSpec`.
Every response is bound to the exact request, turn, and agent. Context bytes,
per-turn output bytes, total output bytes, and reported tokens fail closed at
their frozen ceilings.

The first execution mode is deliberately `deterministic-sequential`. The DAG
can express independent turns, but the runner does not yet claim concurrent
provider sessions or parallel reset isolation. Parallel execution must add an
explicit scheduling and receipt contract before it becomes another mode.

## Self-evolution boundary

The first admitted multi-agent mutation changes one agent instruction and
nothing else:

```text
parent AgentTeamManifest
       |
       | AgentInstructionMutation
       | hypothesis + evidence + exact before hash
       v
candidate AgentTeamManifest
       |
       | represented as ComponentKind.CONTROL_FLOW
       v
existing HarnessRegistry -> candidate admission -> paired verification gate
```

`materialize_agent_instruction_candidate` produces the child team, exact
canonical bytes, and the existing generic `CandidateMutation`. Consequently,
the same content-addressed lineage and `HarnessRegistry` checks used by other
harness components also protect an agent team. Callers must explicitly enable
the `control_flow` component kind and the agent-team media type; the default
prompt-only mutation policy remains unchanged.

Team membership, roles, model identities, capabilities, coordinator, budgets,
and workflow turns remain frozen for this mutation. `rebind_agent_workflow`
only updates the child team fingerprint after verifying that the roster is
unchanged.

## Trust boundary

The current kernel establishes:

- canonical immutable team, workflow, request, response, and run identities;
- explicit per-turn information flow;
- score-free agent execution;
- backend identity and response-lineage checks;
- hard logical byte, turn, and reported-token limits;
- one-component, one-agent instruction mutation;
- compatibility with the existing evidence-gated candidate path;
- receipt-backed real-model turns through `FixedModelRunner`, the attempt
  ledger, and the content-addressed artifact repository;
- an exact-call-count baseline/Harness BFCL development comparison whose result
  directly closes all 54 execution references and nine team transcripts.

It does not yet establish:

- OS/process/network isolation between agents;
- true parallel sessions or provider-side cache reset;
- trusted provider token attestation;
- tool-call capability enforcement;
- topology, roster, role, model, memory, or tool self-mutation;
- multi-agent-specific causal credit assignment;
- benchmark improvement or a reportable scientific result.

## Planned increments

1. Add matched team-level parent/candidate schedules and per-agent mechanism
   probes for activation, information use, critique adherence, and synthesis.
2. Add token- and cost-matched controls alongside the implemented exact-call
   baseline comparison.
3. Add a bounded topology grammar: add/remove one edge, reorder one independent
   turn, or add/remove one role under an explicit call-budget equivalence rule.
4. Add process-isolated parallel execution with deterministic join ordering.
5. Only then permit tool, memory, or model-routing mutations, each with a new
   capability policy and negative-control study.

The promotion rule remains unchanged throughout: a team cannot grade or
promote its own mutation, and sealed evidence cannot flow back into evolution.
