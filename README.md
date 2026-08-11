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

The project is at the foundation stage. The immediate plan is:

1. specify the experiment and verification contracts;
2. implement the typed state, artifact store, and promotion gate;
3. connect one reproducible benchmark and a fixed-model runner;
4. compare targeted evolution with static, random-mutation, and prompt-only
   baselines under the same evaluation budget;
5. expand the mutation space only after the core claim is measurable.

See [the research plan](docs/research-plan.md) and
[the verification protocol](docs/verification-protocol.md).

## Core principle

The optimizer may propose changes; it may not decide that its own change
worked. Promotion belongs to a separate, deterministic evaluation path with
sealed inputs and durable evidence. A credible promotion should show that a
patch was activated, was followed, changed the intended behavior, improved the
task outcome, and survived held-out regression checks.
