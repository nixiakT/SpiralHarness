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

The M0.2 verification and experiment-control kernel now includes:

- immutable schemas for harness manifests and atomic candidate mutations;
- an exact-media v2 protocol and frozen experiment manifests binding split,
  model, runtime, grader,
  mechanism-evidence and gate-batch attestors, gate, mutation policy, stopping
  rule, and evaluation budget;
- canonical typed-artifact reads that reject aliases, omitted defaults, and
  validator normalization when those would create a second content identity;
- a prompt-only mutation registry that verifies the actual parent component and
  candidate bytes, hash, media type, size, and immutable fields;
- trusted candidate admission that reloads and rejoins the experiment,
  protocol, parent lineage, mutation, evidence, child, and gate configuration;
- canonical hashing and an integrity-checking content-addressed artifact store;
- a strict evidence-bearing candidate lifecycle stored as a content-addressed,
  append-only linked journal with no overwriteable head;
- score-free candidate output traces authenticated by a parent-held capture
  capability, and an in-process trusted grader that owns sealed answers,
  scoring, policy checks, and captured mechanism evidence;
- controlled adversarial fixtures covering no-op, wrong-target, inactive,
  non-adherent, placebo, regression, timeout, malformed, tamper, budget, and
  leakage cases;
- a deny-by-default capability-policy schema, with exact grants bound into the
  frozen protocol and checked again during candidate admission;
- typed trajectory events, source spans, evidence packets, failure signatures,
  and diagnostic clusters;
- structural interfaces for benchmark, executor, diagnoser, proposer, artifact
  repository, and independent verifier implementations;
- strict task/seed pairing with task-level bootstrap statistics;
- a three-state promotion gate (`promote`, `reject`, `inconclusive`);
- producer-attested mechanism-evidence envelopes whose HMAC covers the
  protocol, candidate and child harness, exploration split and task-set
  identity, execution context, source artifacts, and complete mechanism checks;
  the controller verifies the protocol-pinned producer before either accepting
  or rejecting the probes, so a caller-authored `passed=true` cannot enter the
  gate;
- producer-attested gate trial batches whose HMAC covers the candidate, arm,
  harness, gate split, task-set identity, complete observations and resource
  fields, protocol execution context, source artifacts, and mechanism evidence;
  the accepted attestor identity is frozen in the protocol;
- an evaluation manifest that closes over admission, frozen gate config and
  implementation fingerprint, both attested batches, and mechanism evidence;
- terminal-decision validation that replays admission, reloads the frozen gate
  split/config and paired evidence, recomputes the complete gate decision, and
  binds it to the only matching lifecycle outcome;
- hard checks for preregistered rosters, mechanism evidence, protected slices,
  policy violations, regressions, and resource budgets;
- a semantic experiment controller that owns candidate admission, probe, gate,
  evidence-complete, and terminal transitions; maintains an immutable
  query/resource-usage ledger; rejects stale tails and duplicate charges; and
  recomputes frozen budget consumption from both trial arms;
- typed sibling-supersession handling: a later local `PROMOTE` against an old
  champion resolves to `INCONCLUSIVE` without advancing the champion or
  deadlocking selection;
- a typed experiment lifecycle from `FROZEN` through `SEARCHING`,
  `SELECTION_CLOSED`, `SEALED_RUNNING`, and `COMPLETE`, with integrity or
  leakage evidence able to move a nonterminal lineage to `INVALIDATED`;
- typed selection closure, sealed-run authorization, completion, and
  invalidation artifacts that bind each transition to one exact branch;
- a deterministic controlled-fault vertical slice with logically disjoint
  exploration and gate rosters, pre-gate mechanism probes, a frozen experiment
  contract, and content-addressed artifact/lifecycle lineage.

M0.2 establishes typed logical trust boundaries, not deployment isolation. Its
capture, mechanism-evidence, and gate-batch HMACs are process-local API
capabilities: the symmetric verification objects still contain key material,
and losing an ephemeral key prevents cryptographic replay in another process.
The signers, grader, and controller therefore remain trusted code in one
interpreter. The capability policy is a schema/admission contract rather than
an OS-enforced sandbox; there is no enforced network, filesystem, or process
isolation.

The controller uses caller-held content-addressed tails and one in-process
single writer. It rejects experiment-lifecycle resume entirely because M0.2
cannot semantically authenticate a recovered head, and it does not provide
cross-process durable compare-and-swap, leases, or transactions. Usage is
charged from submitted signed batches; a future runner must reserve attempts
before execution so omitted retries cannot escape accounting. Typed sealed
reports are joined to the exact selection branch, but their producer is not yet
cryptographically attested. The repository still has no real benchmark or
fixed-model production runner. Current producer HMACs authenticate assertions
and exact source hashes; they do not re-execute source artifacts to derive
mechanism booleans or scores. Typed capture/source replay and trusted regrading
belong in that runner stage.

The next implementation stage is:

1. connect one reproducible benchmark and a fixed-model runner;
2. implement bounded diagnosis/proposal strategies over prompt mutations;
3. compare targeted evolution with static, random-mutation, and prompt-only
   baselines under the same evaluation budget;
4. expand the mutation space only after the core claim is measurable.

See [the research plan](docs/research-plan.md),
[the verification protocol](docs/verification-protocol.md), and the
[pinned open-source architecture study](docs/reference-architecture-study.md).

## Run the M0 vertical slice

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
uv run spiral demo --output runs/controlled-demo
uv run pytest
```

The demo injects a known prompt fault into a deterministic synthetic executor,
diagnoses it on an exploration roster, applies one policy-checked atomic prompt
repair, verifies the proposed mechanism on exploration probes, and evaluates
parent/candidate pairs on a disjoint gate roster loaded back through the frozen
protocol. It prints the root SHA-256 of
a replay manifest; all protocol/experiment contracts, prompts, manifests,
trials, mechanism checks, candidate lifecycle events, gate configuration, and
the decision are stored beneath the output directory.

The fixture validates plumbing and adversarial gate behavior only. It is not a
real benchmark, does not call an LLM, and is not evidence of model or agent
capability improvement. Its trusted grader is an in-process boundary, not an
out-of-process service or a worker sandbox.

## Core principle

The optimizer may propose changes; it may not decide that its own change
worked. Promotion belongs to a separate, deterministic evaluation path with
sealed inputs and content-addressed evidence. A credible promotion should show that a
patch was activated, was followed, changed the intended behavior, improved the
task outcome, and survived held-out regression checks.
