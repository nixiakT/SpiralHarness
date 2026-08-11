# SpiralHarness

SpiralHarness is a research framework for **training-free, causally informed,
evidence-gated self-evolution of agent harnesses**. It studies how a fixed model
can improve its surrounding system—prompts, skills, memory, tools, middleware,
and control flow—through a repeatable loop:

> execute → observe → diagnose → propose → verify → promote or reject

The first milestone is deliberately benchmark-driven. A benchmark provides a
clear optimization signal, while an independent verification gate decides
whether a proposed harness change is a real, generalizable improvement rather
than noise, overfitting, or reward hacking.

## Research thesis

Harness self-evolution should be treated as a sequence of falsifiable,
versioned experiments—not unrestricted self-editing. Every mutation must carry
an evidence-backed hypothesis, be evaluated against its exact parent under a
matched budget, and pass held-out regression checks before promotion.

SpiralHarness initially focuses on:

- fixed model weights (training-free evolution);
- typed, component-level harness mutations;
- trajectory-based failure localization;
- benchmark score as the primary objective;
- paired, reproducible evaluation and statistical promotion gates;
- complete lineage, rollback, and audit artifacts.

Model training, open-ended personalization, and unsupervised self-preference
are later extensions, not part of the initial claim.

## Status

The M0 verification kernel now includes:

- immutable schemas for harness manifests and atomic candidate mutations;
- canonical hashing and an integrity-checking content-addressed artifact store;
- strict task/seed pairing with task-level bootstrap statistics;
- a three-state promotion gate (`promote`, `reject`, `inconclusive`);
- hard checks for preregistered rosters, mechanism evidence, protected slices,
  policy violations, regressions, and resource budgets.
- a deterministic controlled-fault vertical slice with logically disjoint
  exploration and gate rosters and replayable artifact lineage.

The immediate plan is:

1. connect one reproducible benchmark and a fixed-model runner;
2. implement bounded diagnosis/proposal strategies over prompt mutations;
3. compare targeted evolution with static, random-mutation, and prompt-only
   baselines under the same evaluation budget;
4. expand the mutation space only after the core claim is measurable.

See [the research plan](docs/research-plan.md) and
[the verification protocol](docs/verification-protocol.md).

## Run the M0 vertical slice

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
uv run spiral demo --output runs/controlled-demo
uv run pytest
```

The demo injects a known prompt fault into a deterministic synthetic executor,
diagnoses it on an exploration roster, applies one atomic prompt repair, and
evaluates parent/candidate pairs on a disjoint gate roster. It prints the root
SHA-256 of a replay manifest; all prompts, manifests, trials, mechanism checks,
gate configuration, and the decision are stored beneath the output directory.

The fixture validates plumbing and adversarial gate behavior only. It is not a
real benchmark, does not call an LLM, and is not evidence of model or agent
capability improvement.

## Core principle

The optimizer may propose changes; it may not decide that its own change
worked. Promotion belongs to a separate, deterministic evaluation path with
sealed inputs and durable evidence. A credible promotion should show that a
patch was activated, was followed, changed the intended behavior, improved the
task outcome, and survived held-out regression checks.
