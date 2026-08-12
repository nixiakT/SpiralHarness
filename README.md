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

## Repository layout

- `spiral_harness/`: the single importable package, grouped by domain; there is
  no duplicate `src/spiral_harness/` tree or forwarding package.
- `tests/`: unit, integration, and architecture contracts at one visible level.
- `docs/`: research protocols, architecture decisions, and the pinned
  open-source architecture study.
- `tools/`: small repository verification utilities.
- `.github/workflows/`: CI only; GitHub Actions requires this exact path.

The repository does not track placeholder directories or `.gitkeep` files.

## Status

The M0.2 verification and experiment-control kernel includes:

- immutable schemas for harness manifests and atomic candidate mutations;
- an exact-media v4 protocol and frozen experiment manifests binding split,
  model, full credential-free model-spec fingerprint, runtime, grader,
  mechanism-evidence and gate-batch attestors, optional skill-verification
  policy, gate, mutation policy, stopping rule, and evaluation budget;
- canonical typed-artifact reads that reject aliases, omitted defaults, and
  validator normalization when those would create a second content identity;
- an atomic mutation registry whose safe default remains prompt-only and that
  verifies the actual parent component and candidate bytes, hash, media type,
  size, and immutable fields; skill replacement requires an explicit kind,
  component-name, and media-type policy;
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

The first M1 benchmark/runner foundation now also includes:

- a byte-exact GSM8K registry pinned to an upstream commit, file sizes,
  SHA-256 values, MIT attribution, and a verify-before-publish downloader;
- a strict trusted GSM8K adapter with question-only worker tasks, the official
  compatible scorer, group-aware exploration/gate membership, official-test
  sealed membership, and frozen full-data split fingerprints;
- a provider-neutral fixed-model specification and score-free runner whose
  paired fingerprint separates treatment-arm request identity from the shared
  model/task/seed context;
- a v3 preflight certificate that freezes the exact model specification,
  ledger boundary, and process-local writer epoch before a paired batch, with
  receipt replay rejecting reconstructed historical writers, recomputing the
  model fingerprint, and requiring both arms to share the same task bytes;
- a trusted benchmark batch runner that executes a complete GATE schedule with
  the fixed runner, keeps grading inside the adapter, signs parent/candidate
  gate batches, and fails before backend calls if the protocol, split, roster,
  model spec, or attestor identity drifts;
- a baseline execution binding that derives plan-owned GATE schedules for each
  condition/search-run seed and converts receipt-backed runner usage into
  budget-checked baseline usage reports without treating them as final fairness
  attestations;
- a trusted baseline GATE runner that executes those plan-derived schedules
  with a fresh exact-budget attempt ledger per batch, publishes canonical usage
  report artifacts, and validates the four condition reports together while
  still marking them as structural usage evidence rather than reportable
  benchmark gains;
- an immutable v3 attempt reservation/outcome ledger whose records bind the
  originating writer epoch, reserve before a backend call, settle successes,
  burn uncertain failures, poison known token overruns, and make reopened tails
  audit-only;
- an exact four-condition protocol for static, random-valid, prompt-only, and
  evidence-targeted studies, including independent search-run seeds, paired
  rollout seeds, identical ceilings, information boundaries, and structural
  validation of declared aggregate usage.

The first declarative skill execution slice now adds:

- one canonical JSON `SkillPackage` artifact containing metadata, ordered
  rules, procedure, examples, fixed model/runtime compatibility, and no
  executable entry point or runtime permission grant;
- strictly parsed and normalized SPDX licence expressions, plus package-declared
  source class and immutable provenance, compliance-review, and third-party
  notice references whose exact content-addressed bytes the trusted loader
  requires before use; reference existence does not authenticate the declared
  source class or a reviewer, prove approval, or establish a legal conclusion;
- deterministic `metadata`, `rules`, and `full` disclosure levels with fixed
  framing, renderer identity, exact context bytes, hash, and size;
- a deliberately narrow skill mutation for a package declaring
  `source_kind=generated`: the next revision must point directly to its exact
  parent, increment by one, change rules, and preserve every other package field
  byte-for-byte in canonical form; this path rejects packages declaring
  first- or third-party source classes, but does not authenticate provenance or
  recursively validate older lineage;
- trusted candidate admission that reloads both package revisions and replays
  those semantic checks in addition to the generic atomic registry;
- a `HarnessMaterializer` that resolves exactly one prompt and at most one
  compatible skill, using the fixed `rules` disclosure for scheduled execution;
  the loader separately supports deterministic `metadata`, `rules`, and `full`
  projections. `ResolvedHarness` and `ModelRequest` v2 place the exact manifest
  reference and disclosure into the actual backend request; `ModelExecution` v2
  embeds the complete credential-free frozen model specification. The preflight
  certificate fixes that specification for the whole paired batch, and receipt
  publication/replay automatically reconstruct the request from the manifest,
  prompt, and package CAS artifacts;
- a protocol-bound `SkillVerificationPolicy` that freezes the evidence profile,
  model/runtime/grader coordinates, trusted request-inclusion verifier,
  claim-specific producer/verifier/config identities, independent claim and
  aggregate attestors, typed probe roster, deterministic neutral-rule builder,
  and future evidence thresholds;
- a candidate-bound `SkillMechanismPlan` that replays the exact experiment,
  protocol, mutation, package and harness lineage; reconstructs a matched sham
  package and harness; and freezes matched treatment-vs-revert and
  treatment-vs-placebo schedules before probe execution. Existing semantic
  rule order must be preserved, and every seed-affecting schedule coordinate is
  fixed by the policy roster;
- controller lifecycle binding and replay of that plan before entering probes,
  again before gate authorization, and again when gate history is consumed;
- a controller-owned, one-shot process-local execution authorization bound to
  the exact experiment, protocol, candidate, plan, and current
  `RUNNING_PROBES` tail. The authorized executor runs the frozen revert arm
  before the placebo arm, constructing a new process-local runner wrapper,
  preflight certificate, writer epoch, and attempt ledger for each arm. The two
  backend objects are caller-supplied; requiring different objects prevents
  direct reuse but does not attest backend/provider-session or runtime reset;
- score-free matched parent/candidate execution over exact typed probe tasks,
  whose schema contains only the frozen task ID and question. Live-ledger
  replay closes the exact receipts, usage, task bytes, and candidate request
  inclusion separately for both arms;
- a non-promoting matched-execution closure and request-inclusion shadow report.
  While the original controller capability and both live ledger writers remain
  available, verification re-derives the exact process-local authorization,
  receipt/accounting, task, and request-construction chain. These artifacts are
  not durable standalone proofs, and both hard-code `promotion_authority=false`.

The automatic-search control kernel now adds:

- one preregistered search-run manifest per baseline and independent search
  seed, with a common analysis plan and exact strategy/stopping bindings;
- scoped, no-list strategy artifact views that log optimizer reads and writes
  and require diagnoses, hypotheses, and prompt proposals to cite artifacts
  actually exposed in the current round;
- HMAC-attested strategy feedback bound to the exact run, round, champion,
  benchmark binding, and exploration split; optimizer grants come only from
  frozen safe metadata, role-specific exploration refs, and the exact
  diagnostic closure, never from the complete binding or gate/sealed splits;
- a finite-catalogue random-valid strategy with reproducible uniform sampling
  over the eligible entries and cross-round exclusion;
- all-or-nothing paired execution schedules, domain-separated rollout seeds,
  receipt replay, ledger-derived screen usage, and an independently frozen
  objective-aggregate attestor;
- deterministic multi-proposal screening and at most one gate nomination per
  round, followed by controller-derived aggregate-only feedback;
- HMAC-authenticated append-only search and study journals, exact champion and
  terminal-decision joins, frozen stopping rules, and a complete
  `4 conditions × search seeds` selection barrier before sealed authorization;
- a deterministic multi-condition replay fixture marked non-reportable.

M0.2 establishes typed logical trust boundaries, not deployment isolation. Its
capture, mechanism-evidence, and gate-batch HMACs are process-local API
capabilities: the symmetric verification objects still contain key material,
and losing an ephemeral key prevents cryptographic replay in another process.
The signers, grader, and controller therefore remain trusted code in one
interpreter. The capability policy is a schema/admission contract rather than
an OS-enforced sandbox; there is no enforced network, filesystem, or process
isolation. Python private attributes, name mangling, constructor guards, and
distinct in-process objects are composition checks, not hostile-code security
boundaries; the entire interpreter and controller composition belong to the
trusted computing base.

The controllers and attempt ledger use caller-held content-addressed tails and
one in-process single writer. Experiment-lifecycle resume is rejected because
M0.2 cannot semantically authenticate a recovered head. Automatic search also
rejects resume after a completed round because its process-local optimizer
capability and call counts cannot yet be re-authenticated. None of these
controllers provides cross-process durable compare-and-swap, leases, or
transactions. Usage is charged from submitted signed batches; the M1 runner
separately reserves each backend attempt before execution so an observed
failure cannot disappear from its local budget stream. Typed sealed
reports are joined to the exact selection branch, but their producer is not yet
cryptographically attested. Current producer HMACs authenticate assertions
and exact source hashes; they do not re-execute source artifacts to derive
mechanism booleans or scores. Typed capture/source replay and trusted regrading
are not yet wired from the new runner into automatic gate-batch production.

The fixed runner is provider-neutral and currently ships a deterministic replay
backend, not a credentialed live-model provider. It does not add an OS sandbox,
network/filesystem/process isolation, hard cancellation around a blocking SDK,
an authoritative repository-wide ledger head, or cross-process attempt leases.
Its local ledger hard-enforces attempts and tokens; backend-reported monetary
cost is captured but is not yet pre-reserved or hard-bounded. Execution
artifacts are strictly parsed and bound to reservations, but are assertions of
the trusted in-process runner rather than cryptographic producer attestations.

The skill slice establishes package integrity, direct/existence-only reference
closure, immediate revision admission, exact settled request inclusion/replay,
candidate-bound matched-control preregistration, and one controller-authorized
process-local execution of the revert arm followed by the placebo arm. Each arm
has a distinct process-local runner/ledger wrapper, preflight, and writer epoch;
the closure can replay the exact typed task and request-inclusion bindings only
through the original live capability and ledger writers. This is a score-free
plumbing and integrity result, not mechanism or efficacy evidence. It does not
prove remote-provider delivery, model attention or runtime
activation, adherence, behavior change, a revert/placebo causal effect,
downstream benefit, or generalization. The shadow report has no promotion
authority; dedicated claim evidence producers and verifiers are not yet wired,
and generic mechanism signatures for every reserved skill claim remain
quarantined, so no skill can reach promotion. The automatic strategy grammar,
finite catalogue, and proposal contracts remain prompt-only; there is no
general skill router or skill search proposal. Consequently this repository
does not yet demonstrate a skill gain.

The next implementation stages are:

1. run a new preregistered scored matched claim-capture study under the same
   frozen construction, with replayable provider-delivery, runtime-activation,
   adherence, behavior, and revert/placebo-effect capture, before
   unquarantining any skill gate check;
2. extend the bounded search grammar to skill proposals only after that
   verification path is fail-closed, then connect the replayable conditions to
   a credentialed fixed-model provider and preregistered reportable study;
3. expand into memory, tools, middleware, and control flow only after the
   prompt/skill claims are measurable.

See [the research plan](docs/research-plan.md),
[the verification protocol](docs/verification-protocol.md), and the
[pinned open-source architecture study](docs/reference-architecture-study.md).
The real-benchmark details are in the
[GSM8K protocol](docs/gsm8k-protocol.md), and baseline equality rules are in
the [four-condition protocol](docs/baseline-protocol.md). The automatic loop,
receipt-backed screen, and study barrier are specified in the
[trusted search protocol](docs/search-protocol.md).

## Materialize the real benchmark

```bash
uv sync --locked --all-groups
uv run spiral benchmark fetch gsm8k --output data/benchmarks/gsm8k
```

This verifies the registered upstream bytes and prints the frozen split and
adapter fingerprints. It does not run a model or expose a sealed-test score.
Keep the downloaded official test data in the trusted evaluation plane.

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
