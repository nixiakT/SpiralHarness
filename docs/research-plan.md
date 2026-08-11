# Research plan

## 1. Problem statement

Given a fixed base model, an initial harness \(H_0\), a task distribution
\(D\), and a bounded evaluation budget \(B\), automatically produce a harness
\(H^*\) that improves held-out downstream utility without violating regression,
cost, or safety constraints.

The harness state is broader than a system prompt. It can contain:

- prompts and context templates;
- skills and reusable procedures;
- memory policies and stored artifacts;
- tool descriptions, routing, and implementations;
- middleware and observability hooks;
- control flow and sub-agent orchestration.

The initial system does not update model weights.

## 2. Research position

Recent work has established full-harness search, observability-driven
modification, self-modifying harnesses, quality-diversity archives, continual
populations, and model–harness co-evolution. SpiralHarness will not compete by
adding another unconstrained edit loop. Its initial research bet is:

> Reliable self-evolution comes from making diagnosis, mutation, and promotion
> separate protocols connected by auditable evidence.

This leads to four design choices:

1. **Typed mutations.** A candidate declares exactly which harness components
   change. This enables attribution, validation, rollback, and ablation.
2. **Falsifiable diagnoses.** Each change records failure evidence, a root-cause
   hypothesis, expected impact, and explicit regression risks before evaluation.
3. **Independent promotion.** Benchmark workers and statistical gates, not the
   proposing model, determine whether a candidate is retained.
4. **Mechanism evidence.** Promotion should test the chain from patch activation
   and agent adherence to behavioral change and downstream benefit, rather than
   infer causality from an aggregate score alone.

## 3. Initial research questions

### RQ1 — Benefit

Does evidence-targeted harness evolution improve held-out benchmark score over
a static harness under fixed model, token, wall-clock, and evaluation budgets?

### RQ2 — Mechanism

Does verifying patch activation, agent adherence, and intended behavior change
predict downstream benefit better than aggregate benchmark score alone?

### RQ3 — Verification

At a fixed evaluation budget, how precisely does the verification gate accept
truly beneficial edits and reject apparent improvements that fail to replicate
or introduce regressions?

### RQ4 — Attribution

Which component mutations, alone or in interaction, produce transferable gains?

Continual adaptation, weak/no-ground-truth feedback, population routing, and
model–harness co-evolution are follow-on questions.

## 4. Unit of evolution

Every candidate is an immutable child of one parent harness and contains:

- a content-addressed harness snapshot;
- a typed patch over one or more components;
- evidence references into immutable trajectories;
- a diagnosis and root-cause hypothesis;
- predicted positive and negative effects;
- the evaluation plan and budget;
- resulting measurements and a promotion decision.

The first implementation supports atomic prompt and skill mutations. The state
schema will reserve explicit boundaries for memory, tools, middleware, and
control flow so they can be added without turning the harness into an opaque
code blob. Multi-component edits require an explicit interaction experiment;
otherwise they make credit assignment needlessly ambiguous.

## 5. Evolution loop

1. Run the current champion on exploration tasks and persist full trajectories.
2. Cluster or sample failures, then localize each failure to a component and
   behavioral cause.
3. Generate multiple small candidates with pre-registered hypotheses.
4. Reject structurally invalid candidates before expensive execution.
5. Probe whether each patch is activated, followed, and changes its intended
   behavior; use disable, revert, or placebo interventions when appropriate.
6. Evaluate candidates and their parent on identical task/seed pairs.
7. Apply the promotion protocol in `verification-protocol.md`.
8. Promote only a passing candidate; otherwise retain the current champion.
9. Archive all results, including rejected candidates, and use them as future
   search evidence without exposing sealed test content.

This is a champion–challenger loop first. A quality-diversity archive and
task-conditioned routing can be introduced once single-lineage attribution is
sound.

## 6. Minimum credible experiment

The first credible study has two complementary tracks:

1. a controlled repair suite with injected harness faults and known failure
   location/cause, used to measure localization, mechanism verification, and
   false-promotion rate;
2. open improvement of a seed harness on one real benchmark with deterministic
   or replayable grading, used to measure generalizable downstream gain.

Before search begins, freeze:

- benchmark version and split hashes;
- model and inference settings;
- tool/runtime/container versions;
- search and evaluation budgets;
- primary score and constraint definitions;
- baselines and stopping criteria;
- statistical thresholds.

Compare at least four conditions over multiple independent search runs:

1. static seed harness;
2. random valid mutation search;
3. prompt-only optimizer;
4. SpiralHarness evidence-targeted component optimizer.

Report search curves, final sealed-test score, accepted-edit precision,
false-promotion rate, cost, variance, regression rate, activation/adherence
rate, candidate acceptance rate, and component-level ablations. Search-time
best score alone is not a valid result.

## 7. Milestones

### M0 — Verification kernel

- versioned schemas for harnesses, trials, candidates, and decisions;
- content-addressed artifact storage and lineage;
- paired statistics and configurable promotion constraints;
- deterministic controlled-fault fixture and tests for accept/reject behavior.

### M1 — Benchmark-driven prompt/skill evolution

- one real benchmark adapter;
- fixed-model task runner with complete trajectory capture;
- failure diagnosis and bounded candidate generation;
- static, random-valid, prompt-only, and evidence-targeted conditions;
- end-to-end reproducible experiment command.

Implementation status: the pinned GSM8K adapter, fixed/replay runner,
pre-execution attempt ledger, and four-condition comparison protocol are in
place. The first bounded prompt-search kernel now adds typed diagnosis and
proposal contracts, finite-catalogue random-valid sampling, scoped strategy
artifact access, receipt-backed candidate screens, aggregate-only gate
feedback, immutable search journals, and an exact all-run study barrier. Its
multi-condition replay fixture is non-reportable. A credentialed fixed-model
provider and preregistered live study remain necessary for an empirical result;
skill evolution is the next component-level extension of the control loop.

### M2 — Multi-component evolution

- memory, tool, middleware, and control-flow mutations;
- interaction-aware ablations and change-risk policies;
- reusable mutation archive organized by failure location and cause.

### M3 — Sustained evolution

- non-stationary task streams and drift detection;
- harness populations and task-conditioned routing;
- delayed/weak feedback adapters;
- promotion policies for personalization, efficiency, and long-term utility.

## 8. Explicit non-goals for M0–M1

- training or fine-tuning model weights;
- letting the optimizer alter graders or evaluation policy;
- repeatedly tuning against the sealed test set;
- accepting a change solely through LLM self-judgment;
- optimizing an unbounded scalar that silently trades correctness for cost or
  safety.

## 9. Terminology boundary

`Training-free` means no target-model weight update; it does not mean
compute-free. Evaluation calls, tokens, latency, and monetary cost are reported.

If a stronger external optimizer performs the edits, the experiment demonstrates
training-free harness optimization. A self-evolution claim additionally requires
the same base model or agent family to close the execute–diagnose–propose–select
loop, preserve state across iterations, and choose changes without manual
cherry-picking. Both settings are useful, but they answer different questions.
