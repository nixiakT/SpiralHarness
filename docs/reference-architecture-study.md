# Reference architecture study

> **Snapshot date:** 2026-08-11  
> **Scope:** the open-source repositories named in the initial SpiralHarness
> research notes. Paper claims are useful context, but the decisions below are
> based on pinned source snapshots and their observable contracts.  
> **Reuse rule:** this document records patterns and provenance. No third-party
> implementation has been copied into SpiralHarness.

## 1. Executive decision

The reference projects support the initial training-free direction, but they
also narrow the defensible contribution. Full-program Harness search,
trajectory-driven diagnosis, atomic hook proposals, held-out acceptance,
population routing, and product-scale agent runtimes already exist in some
form. SpiralHarness should therefore concentrate on a stricter substrate:

> **Causally attributable, evidence-gated, budget-aware training-free Harness
> self-evolution.**

Benchmark score is the first external objective because it is measurable and
replayable. Trajectories, reflection, self-consistency, and model preferences
are evidence for diagnosis and exploration; they are not sufficient authority
for promotion. The trusted evidence chain is:

```text
patch valid
  -> component activated
  -> agent adhered
  -> intended behavior changed
  -> paired task utility improved
  -> held-out gain replicated without protected-slice, policy, or cost regression
```

This framing is deliberately narrower than building an all-purpose agent
platform immediately. A platform can eventually reach hundreds of thousands
of lines through benchmark integrations, sandboxes, providers, gateways,
plugins, observability, security, and UI. The research kernel should remain
small enough to audit and replay.

## 2. Reproducible source and licence inventory

LOC is a rough scale signal, not a quality metric. The figures below count
common source languages and may include tests or vendored code; they exclude
datasets and most configuration. Licence findings apply to the pinned branch
and commit, not necessarily another branch or a future revision.

| Project | Pinned branch and commit | Approx. source LOC | Licence boundary | SpiralHarness reuse rule |
| --- | --- | ---: | --- | --- |
| [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) | `main` at `44b9942127847f7421db70d8c7e48407f09a3c70` | 10.2k | Root MIT | Patterns and attributed code may be reused under MIT; integrations stay behind adapters. |
| [Agentic Harness Engineering](https://github.com/china-qijizhifeng/agentic-harness-engineering) | `main` at `8b2a55d97590363fe50c3cc6b5e833b020a4bb4c` | 23.3k | Root MIT | Evidence and trace patterns are usable; do not copy its unpinned dependency practice. |
| [Self-Harness](https://github.com/qzzqzzb/Self-Harness) | `main` at `2720dbb3f52283684f4b85a1065d642df1779dd8` | 5.7k | **No licence found** | Ideas only. Do not copy source, prompts, configuration, tests, or assets. |
| [A-Evolve: Harness Evolution](https://github.com/A-EVO-Lab/a-evolve/tree/release/harness-evolution) | `release/harness-evolution` at `986d97d43b6313c94c7e72c0b0ab6181ed9edba0` | 34.4k | `pyproject.toml` says MIT, but this commit has no licence text | Clean-room reimplementation of ideas only unless upstream supplies an applicable licence file. |
| [A-Evolve: Adaptive Auto-Harness](https://github.com/A-EVO-Lab/a-evolve/tree/release/adaptive-auto-harness) | `release/adaptive-auto-harness` at `17bc9ebb7d4d142af1b109b43ef160031967cc9a` | 36.6k | Root MIT | Reusable with attribution; keep its large optional dependency surface outside the core. |
| [HarnessX](https://github.com/Darwin-Agent/HarnessX) | `main` at `bf5f199ee65034d55db0c536e582f1e7c8abf669` | 129.1k | Root MIT, **mixed repository** | Core patterns are usable. Do not treat restricted extension assets as MIT. |
| [Skill Self-Play](https://github.com/Qwen-Applications/skill-self-play) | `main` at `bb693c89fee66e1f824d6a777759a49b7a295a83` | 28.5k | Apache-2.0 text; GitHub reports `NOASSERTION`; dataset boundaries vary | First-party patterns may be used with a manual compliance record. Do not redistribute unclear datasets or its vendored training fork. |
| [Hermes Agent Self-Evolution](https://github.com/NousResearch/hermes-agent-self-evolution) | `main` at `0a929e3aa20e15cf04dc7c28492a7d41a5139125` | 3.9k | Package metadata says MIT, but no licence text found | Ideas only until the licence grant is made explicit. |
| [auto-harness](https://github.com/neosigmaai/auto-harness) | `main` at `d1e3260bc43d8b4e2decdf6a41789ae1587c9e80` | 2.9k | Root MIT | Reusable with attribution; keep Tau2 and its environment as an optional benchmark integration. |
| [Raven](https://github.com/EverMind-AI/Raven) | `main` at `9e2079bf09d85ff31f15d719ebcdcf98248bafc2` | 303.1k | Apache-2.0 plus `NOTICES.md` and nested notices | Use as an engineering-governance reference. Any copied material must preserve the relevant notices. |

Important repository-specific boundaries:

- HarnessX contains Anthropic-licensed skill directories under
  `extensions/skills/{docx,pdf,pptx,xlsx}` whose licence text is not the root
  MIT grant. Those assets must not be copied as though they were MIT. Its VERL
  submodule also needs an independent licence record.
- Skill Self-Play's BFCL benchmark declares Apache-2.0, while API-Bank and the
  `ZebraLogicBench-private` material do not present a sufficiently clear grant
  in this snapshot. They must not be vendored or redistributed. Its bundled
  VERL fork should be consumed as a separately reviewed dependency, not copied
  into the research kernel.
- Raven carries notices for inherited Hermes, nanobot, and Ink material. Root
  Apache-2.0 alone is not the complete attribution record.
- A package classifier such as `license = "MIT"` is not a substitute for an
  actual licence grant. Self-Harness, the Harness Evolution branch of A-Evolve,
  and Hermes are clean-room references for now.

## 3. What each project teaches us

### 3.1 Meta-Harness: a bounded outer search needs a trusted evaluator

The Harbor controller freezes tasks, model, attempts, evaluation budget, and
aggregation; hashes candidate source; stages it in a temporary directory; and
runs each task in a child environment without importing candidate code into
the controller. Its text-classification example lets search see validation
results and opens test only during finalization.

Adopt:

- a narrow evaluator capability rather than benchmark filesystem access;
- candidate hashes, remaining-budget accounting, and append-only evaluation
  history;
- a separate staging/validation step and short-lived candidate processes;
- one-way sealed final evaluation after selection closes.

Do not adopt:

- source-string denylists as the security boundary;
- arbitrary whole-program edits as the first mutation surface;
- “any positive frontier delta wins” without paired uncertainty, regression
  slices, or an `INCONCLUSIVE` outcome;
- overwriteable JSON/JSONL as the authoritative experiment ledger.

### 3.2 AHE: observability is useful only when evidence remains attributable

AHE normalizes long traces across providers, records tool/sub-agent/token/cost
signals, contrasts pass/fail behavior, asks for failure evidence and root cause,
and predicts which tasks a change should fix or endanger. Its iteration history
is operationally useful for resuming large runs.

Adopt:

- provider-neutral typed trajectory events with source-span references;
- an `EvidencePacket` that keeps compact diagnosis connected to raw artifacts;
- task transition matrices and same-task pass/fail contrast;
- `failure evidence -> root cause -> targeted change -> predicted impact` as
  the proposal contract;
- resumable experiment state and candidate workspace snapshots.

Do not adopt:

- committing a workspace before independent validation;
- selecting the best of a bad batch with no “decline/no candidate” option;
- free-form multi-component changes and post-hoc attribution;
- incompatible manifest shapes or file-existence-only validation;
- repeated optimization on the same tasks without query accounting.

### 3.3 Self-Harness: separate diagnosis, proposal, acceptance, and merge

Self-Harness organizes the public workflow into diagnosis, mechanism-diverse
single-hook proposals, evaluation, acceptance, and branch merge. Passing cases
become regression anchors. Independently accepted atomic candidates are merged
only after conflict checks, and the merged result is evaluated again.

Adopt by clean-room implementation:

- terminal-cause and agent-mechanism failure signatures;
- multiple atomic, mechanism-diverse proposals plus an explicit decline path;
- exact mutation-surface enforcement;
- positive examples as protected anchors;
- accept atomic candidates independently, then re-run the complete gate on a
  merge to measure interaction effects.

Do not adopt:

- two-repeat mean-sign rules as a statistical gate;
- “same denominator” as a substitute for exact task/seed/attempt pairing;
- exposing the same held-out set to every adaptive candidate;
- mutable path-based lineage;
- code or prompts from this unlicensed snapshot.

### 3.4 A-Evolve: typed action spaces help, but declared scope is not proof

The Harness Evolution branch separates benchmark adapters and
Reader/Operator/Verifier atoms, declares feedback capability separately from
the evidence observed in a cycle, assigns artifact read/write scopes, emits
mutation reports, and preserves rollback history. The Adaptive branch adds a
catalog, selection/materialization boundary, stable identities, temporal label
reveal, population branches, retrieval, and task-conditioned routing.

Adopt:

- small `BenchmarkAdapter`, `Executor`, `Diagnoser`, `Proposer`, `Verifier`, and
  `ArtifactRepository` protocols;
- static feedback capability versus runtime evidence availability;
- read-only evidence readers and tightly scoped mutation operators;
- persistent versus task-local artifact separation;
- observer sidecars and compatibility/conformance tests;
- later, task-only selection with an enforced capacity and explicit fallback,
  shadow, and routing-regret measurements.

Do not adopt:

- trusting an operator's declared scope without comparing the actual diff;
- validation functions that create or mutate missing workspace state;
- “mutation committed” as evidence that the mutation benefited execution;
- production recipes whose verifier is effectively `NoVerify`;
- silent materialization failure or unpersisted routing logs;
- population/routing before single-lineage attribution is reliable.

### 3.5 Skill Self-Play: skill packaging is relevant; its training reward is not a gate

This project uses skills during curriculum generation and training; its final
solver is prompt-only at inference. It therefore does not directly establish
that a fixed model benefits from runtime skill evolution.

Its strongest reusable idea is a package with metadata, routing summary,
rules, examples, generator hints, validation specification, tests, and stats.
It also supports progressive disclosure (`name -> summary -> rules -> full`),
explicit skill-guided versus no-skill exploration pools, performance buckets,
retirement, induction, and targeted refinement.

Adopt later for SpiralHarness skills:

- immutable `SkillPackageManifest` revisions with parent, provenance, licence,
  permissions, target-model compatibility, evidence, promotion state, and
  rollback reference;
- metadata-first routing and progressive context loading;
- explicit exploration and skill-routed pools;
- confidence-aware frontier/easy/hard/inconsistent buckets;
- retire by creating a terminal revision, never by deleting history.

Reject as promotion evidence:

- a proposer-generated reference answer;
- solver self-consistency with that answer;
- evaluation that provides the target skill's oracle solver rules;
- uncertainty rewards without an independent grader.

Generated `validator.py` or `checker.py` must never be imported into the trusted
orchestrator. Prefer declarative validation. If executable validation is
unavoidable, run it without network or secrets in a read-only, resource-limited
subprocess/container/WASM boundary and still subject its output to a fixed
trusted grader.

### 3.6 Hermes: long-term learning is a second signal regime

Hermes is useful for separating benchmark-driven improvement from experience
accumulation. Its current implementation is much narrower than its roadmap:
`tools`, `code`, and `monitor` are placeholders, while the implemented loop
loads a skill, builds synthetic/session/golden data, invokes GEPA or MIPRO,
checks constraints, compares holdout means, and writes a timestamped output
directory (`evolution/skills/evolve_skill.py:36-292`). It can import historical
Claude Code, Copilot, and Hermes sessions and derive candidate evaluation cases
(`core/external_importers.py:40-80,157-543`). Because the snapshot lacks an
explicit licence text and has a small history, it is an architectural reference
rather than a source dependency.

Adopt later:

- trace-to-skill and trace-to-memory as candidate generators;
- provenance and freshness on learned experience;
- context/caching efficiency as explicit constrained objectives.

Keep separate:

- model- or self-judged experience may enter exploration memory;
- it cannot silently alter the champion or trusted evaluation plane;
- promotion requires evidence appropriate to the claim, and weak-feedback
  updates should start in shadow/quarantine state.

Do not inherit the current validation shortcuts:

- the main fitness metric is expected-rubric/output word overlap even though an
  LLM judge exists (`core/fitness.py:107-136`);
- the skill body is handed to a structure validator that requires frontmatter,
  creating a blocking body/frontmatter contract mismatch
  (`evolution/skills/evolve_skill.py:118-132,186-204` and
  `constraints.py:150-174`);
- `--run-tests` is recorded but the main loop does not invoke the implemented
  test-suite guard (`evolve_skill.py:49-56,295-317`);
- unseeded shuffles and ungrouped session splits make replay and leakage control
  unreliable;
- any positive one-shot holdout mean is called an improvement, without paired
  seeds, uncertainty, minimum effect, protected slices, or `INCONCLUSIVE`.

Most importantly, passing `SkillModule.skill_text` as an ordinary runtime input
does not itself prove the optimizer actually treats that field as an evolvable
parameter. SpiralHarness must always verify that an artifact changed, was
loaded by the real runtime, and altered behavior before interpreting a score.

### 3.7 auto-harness: a small benchmark-specific baseline is valuable

auto-harness is closer to a compact task-specific optimization baseline than a
general platform. Its scale is a useful reminder that credible comparative
experiments do not require a massive runtime. SpiralHarness should reproduce
this class of baseline behind the same budget and evaluator interfaces rather
than compare only against a static seed.

Adopt:

- an optional benchmark integration with a commit-pinned external dependency;
- a simple search baseline exercising exactly the same trial and gate budget;
- minimal end-to-end scripts as smoke tests.

Useful concrete contracts include its `task_id -> reward | None` runner, where
missing requested tasks remain in the denominator (`benchmark.py:29-50`), and
the regression roster's use of expected task IDs (`gating.py:239-250`). It also
checks tracked and untracked Git changes (`gating.py:112-146`).

Do not inherit the experiment semantics:

- it reports the test split every iteration, so test is an adaptive development
  gate rather than a holdout (`gating.py:262-280`);
- validation only needs to match the historical best mean, with neither pairing
  nor uncertainty, and the suite may tolerate 20% regression;
- baseline timeouts are excluded while constructing splits, producing survivor
  bias (`prepare.py:367-375,421-426`);
- trace hiding is an environment-variable convention rather than a capability
  boundary;
- the candidate, workspace, runner, result files, and evaluation scripts share
  writable authority, while the Git guard can fail open;
- mutable JSON/TSV state lacks atomicity, schemas, hashes, and recovery, and raw
  Harbor evidence is deleted.

Benchmark-specific dependencies, credentials, or environment setup must remain
optional and outside the trusted core.

### 3.8 HarnessX: composability shows where platform scale comes from

HarnessX demonstrates a broad, processor-oriented harness foundry with
benchmark integrations, plugins, memory, trajectories, gateway, frontend, and
training recipes. Its nine architectural dimensions and eight lifecycle hooks
are implemented through processors that can transform, split, block, pass, or
raise (`docs/architecture.md:7-23`, `docs/concepts/processors.md:8-20`,
`harnessx/core/processor.py:342-354`). Its builder copies composition state,
aggregates conflicts, and topologically orders dependencies
(`harnessx/core/builder.py:92-188,275-334,669-747`). State preserves an
append-only raw track beside the effective context track
(`harnessx/core/state.py:119-159`). It is a useful target for extension
boundaries, not a module list to reproduce immediately.

Adopt:

- processor/plugin contracts and capability metadata;
- optional benchmark and provider packages;
- a stable event/trajectory spine shared by CLI, services, and UI;
- conformance tests for extensions and clear ownership boundaries.

Defer:

- gateway and frontend;
- a general model-provider matrix;
- SFT/RL training recipes;
- a broad plugin marketplace;
- large benchmark coverage before one real adapter is reproducible.

Avoid its failure modes:

- ordinary processor exceptions may be swallowed so execution continues
  (`harnessx/core/processor.py:681-714`); research evidence collection must
  fail closed;
- dynamic `_target_` snapshot restoration imports arbitrary classes and the
  snapshot is not a complete execution identity (`harnessx/core/state.py:15-89,
  205-284`);
- synthetic replay largely proves “did not crash,” not that the expected
  behavior occurred (`harnessx/meta_harness/replay.py:64-151`);
- GAIA optimization repeatedly queries a fixed task set and lacks a genuinely
  sealed, three-outcome statistical gate;
- shell write-scope detection is heuristic string parsing, not a sandbox
  (`processors/write_scope_gate.py:58-153,233-235`).

### 3.9 Raven: product engineering, security, and provenance are the lesson

Raven is the closest example of why this domain can become a 300k-line system:
channels, gateway, providers, sandboxing, memory/context pipelines, skills,
plugins, tracing, security, CLI/TUI, and hundreds of tests. Its evolver also has
the strongest verification skeleton in this set: a small benchmark bundle,
fixed-denominator `TaskEval`, explicit train/test backend roles, infra retry,
activation ledger, paired task lift, baseline policies, one-way unseal state,
append-only/fsynced round journal, and candidate Git DAG
(`raven/evolver/launch/contract.py:25-60`,
`raven/evolver/orchestrator/scoring.py:34-121,196-213`,
`raven/evolver/orchestrator/gates/pipeline.py:1-106`,
`raven/evolver/orchestrator/sealed/runner.py:38-104,163-270`). Its value to the
current project is engineering governance and a strong comparison baseline,
not a design to adopt uncritically.

Adopt progressively:

- security policy, approval, secret-redaction, and sandbox tests;
- atomic I/O and explicit registries;
- a stable tracing API and context assembly boundary;
- plugin/skill provenance and third-party notices;
- dense unit, contract, integration, and end-to-end testing.

Improve on its research gate:

- current paired promotion still uses `candidate_mean > control_mean`; a 2-sigma
  result changes attribution credit rather than promotion
  (`raven/evolver/orchestrator/gates/paired.py:61-102`);
- the post-hoc fired subset is useful for mechanism attribution but must never
  replace the full preregistered promotion denominator;
- a diff beacon proves only that a marker string exists, not that the mechanism
  activated (`raven/evolver/applier/beacon_guard.py:20-50`);
- missing activation instrumentation can fail open
  (`raven/evolver/activation/ledger.py:86-124`), whereas a required mechanism
  in SpiralHarness becomes `INCONCLUSIVE` or `REJECT`;
- sealed evaluation is separated by code flow, not yet by filesystem/network
  capability;
- path protection needs a mutable allowlist as well as an immutable denylist.

On the product side, Raven's manifest-first plugin discovery, narrow
`PluginContext`, five-method memory backend, two-phase context assembly,
progressive tool disclosure, session fsync/atomic replace/fork lineage, and
artifact-backed tracing are good long-term boundaries. Its 3,000-line agent
loop, implicit hook ordering, silent registry overwrite, permissive
`extra="ignore"`, and configuration fallback demonstrate why SpiralHarness
should impose module-size budgets and fail-closed research configuration.

Do not copy the whole product surface before the evolution experiment is
credible. Otherwise infrastructure complexity will hide whether the research
mechanism works.

The pinned HarnessX and Raven snapshots both expose their Python import package
at the repository root (`harnessx/` and `raven/`). SpiralHarness follows that
flat layout with `spiral_harness/` for direct navigation. Because flat layouts
can let checkout-local imports hide a broken distribution, the project keeps an
isolated wheel-install verification in CI rather than adding a second source
tree or a forwarding package.

## 4. Cross-project architecture matrix

| Concern | Strong reference pattern | Common gap | SpiralHarness decision |
| --- | --- | --- | --- |
| Optimization signal | Benchmark/ground-truth reward anchors search | Reflection and self-consistency are sometimes treated as correctness | Benchmark score is the initial primary objective; surrogate signals allocate exploration only. |
| Diagnosis | AHE evidence compression; Self-Harness causal clustering | Summaries lose source provenance | Typed evidence packets cite immutable trajectory spans. |
| Proposal | Multiple targeted atomic proposals | Best-of-N always returns something; actual diff can escape declared scope | Candidate batch supports decline; registry verifies the actual before/after mutation. |
| Execution | Meta-Harness child staging; Raven sandbox governance | Candidate code and grader often share a process | Target: trusted controller, ephemeral untrusted worker, out-of-process grader. M0.2 has only an in-process trusted-grading boundary and semantic controller. |
| Verification | Held-out acceptance and regression anchors | Mean-only rules, adaptive reuse, no uncertainty or cost parity | Paired task/seed gate, three outcomes, protected slices, policy/cost constraints. |
| Mechanism | AHE predicted impact; Harness-benefit distinction | A patch can exist without being used | Require applied/activated/adhered/behavior/benefit/generalization evidence. |
| Lineage | Git branches, histories, rollback commits | Mutable files and path identity | CAS blobs plus append-only typed event chains; SQLite is an index, not authority. |
| Skills | Skill Self-Play packages and progressive disclosure | In-place refinement, weak provenance, unsafe validators | Immutable package revisions, lifecycle, capability policy, sandboxed validation. |
| Continual adaptation | Adaptive A-Evolve catalog and routing | Population quality and router quality are conflated | Defer; later measure evolution loss and routing/adaptation loss separately. |
| Product scale | HarnessX composition; Raven governance/tests | Premature platform breadth | Stable small kernel with optional adapters/plugins/services. |

## 5. Current SpiralHarness gap analysis

| Capability | Current status | Next invariant |
| --- | --- | --- |
| Typed component state | Implemented for prompt, skill, memory, tool, middleware, control flow refs | Enforce per-kind contracts and actual-diff mutation policy. |
| Immutable artifacts | Local SHA-256 CAS, canonical typed reads, and linked lifecycle events implemented | Add a query index without making it authoritative. |
| Candidate hypothesis | Atomic mutation, falsifiable hypothesis, candidate manifest, trusted lineage admission, and controller-owned candidate transitions implemented | Add generated batch diversity, explicit decline, and search-strategy nomination. |
| Evaluation integrity | Protocol freezes independent process-local mechanism and batch attestors; signed envelopes bind protocol context, sources, candidate/harness/splits, checks, scores, and resources before terminal recomputation | Replace ephemeral symmetric capabilities with grader-integrated, durable asymmetric/KMS provenance and typed source replay in a real runner. |
| Statistics and accounting | Task-level paired bootstrap, regression/cost checks, and an experiment-wide query/resource ledger implemented | Add multiple-candidate correction, winner replication, and a preregistered sequential policy. |
| Mechanism evidence | Typed trajectories/spans plus a protocol-pinned producer envelope prevent caller-authored probe booleans; adversarial inactive/non-adherent/placebo fixtures are implemented | Add reusable trusted probes, typed source re-execution, and stronger disable/revert interventions on a real runtime. |
| Experiment control | Single-process semantic controller owns admission, probes, gate completion, terminal decisions, budgets, sibling supersession, experiment lifecycle, and typed selection/sealed/invalidation closure; lifecycle resume fails closed | Replace caller-held tails with authenticated recovery plus durable cross-process compare-and-swap/leases/transactions before concurrent writers. |
| Benchmark support | Pinned GSM8K adapter, provider-neutral fixed/replay runner, writer-epoch-bound attempt ledger, four-condition protocol, and complete study barrier | Execute credentialed fixed-model studies through the same frozen protocol. |
| Isolation | Trusted grading is logically separate but in-process; deny-by-default capability policy exists as a schema/admission constraint | Add short-lived worker sandbox, no network, read-only materialization, credential isolation, and an out-of-process grader. |
| Evolution engine | Bounded diagnosis, multi-proposal prompt search, four baselines, aggregate feedback, nomination, and stopping are implemented | Add skill proposals only after the skill-specific verification chain is fail-closed. |
| Skills | Immutable declarative packages, direct generated revisions, fixed rules disclosure, request materialization, and settled request-inclusion replay; no activation/adherence/benefit or promotion authority | Add independent provider-delivery, runtime-activation, adherence, behavior, revert, and placebo evidence before skill search. |
| Continual routing | Not implemented | Defer until single-lineage experiments pass. |
| Platform/API/UI | Not implemented | Add only after experiment artifacts and service boundaries stabilize. |

## 6. Target package boundaries

```text
spiral_harness/
  core/             immutable manifests, identities, states, events
  storage/          CAS, linked journal, lineage/query index
  evidence/         trajectory schema, spans, packets, diagnosis signatures
  harness/          component registry, mutation policy, snapshot application
  benchmark/        protocol and optional adapters
  execution/        trusted scheduler, worker protocol, sandbox, accounting
  evolution/        diagnosers, proposers, candidate batches, search baselines
  verification/     probes, paired statistics, adaptive-query and promotion gates
  experiments/      frozen manifests, lifecycle controller, orchestration
  access/           exploration/gate/sealed capabilities and redaction
  skills/           package registry, progressive loader, router, lifecycle
  adaptation/       later population/catalog/routing layer
  providers/        optional model/tool implementations
  service/          later API/gateway/UI support
```

Dependencies point inward. The core does not import providers, benchmark
implementations, or model SDKs. Candidate workers never receive repository,
grader, split, or champion-update authority. Training integrations consume
versioned events through a separate adapter and cannot become a hidden core
dependency.

## 7. Phased implementation backlog

### M0.1 — freeze and replay the experiment contract

1. Add `ProtocolManifest` and `ExperimentManifest`.
2. Add a prompt-only `MutationPolicy` and registry that verifies the actual
   before/after component and preserves all frozen fields.
3. Add a strict candidate lifecycle projected from append-only events.
4. Store journal nodes as content-addressed linked artifacts; a database may
   index them but cannot rewrite their history.
5. Add typed `TrajectoryEvent`, `Trajectory`, and `EvidenceSpanRef` contracts.
6. Declare protocols for artifact repository, benchmark adapter, executor,
   diagnoser, proposer, and verifier.

Exit criterion: the synthetic demo can be replayed from one frozen experiment
reference, and illegal state transitions or out-of-policy mutations fail
closed.

### M0.2 — make the controlled fixture adversarial

1. **Implemented:** scoring is owned by a trusted grader distinct from the
   candidate-output path. The boundary is testable but remains in-process.
2. **Implemented:** no-op, wrong-target, inactive, non-adherent, placebo,
   regression, timeout, malformed-event, tamper, budget, and leakage fixtures
   fail closed.
3. **Implemented:** `AttestedMechanismEvidence` uses an independent,
   protocol-pinned HMAC producer to bind the candidate/child harness,
   exploration split/task set, execution context, complete checks, and concrete
   CAS sources. The controller verifies it before either gate entry or probe
   rejection, and terminal replay verifies it again. Raw `passed=true`, old
   signatures, rogue signers, context/candidate substitution, and missing
   sources fail closed.
4. **Implemented:** v2 `GateTrialBatch` HMAC attests protocol context, complete
   score/resource observations, source refs, candidate/arm/harness/split/task
   set, and mechanism evidence. The protocol pins its attestor ID, and the
   controller and terminal verifier both fail closed on tampering or a rogue
   signer. `GateEvaluationManifest` and protocol artifacts separately pin the
   gate implementation before terminal recomputation.
5. **Implemented as a logical contract:** `CapabilityPolicy` defaults to no
   grants, uses exact resource allowlists, and is bound by protocol and
   admission. No launcher or OS layer enforces it yet.
6. **Implemented for one process:** the semantic controller owns candidate and
   experiment transitions, recomputed query/resource accounting, stale-tail and
   duplicate-charge checks, sibling supersession, and typed selection, sealed,
   completion, and invalidation artifacts. Experiment lifecycle resume is
   rejected until recovered semantic heads can be authenticated.
7. **Still open:** add broader property/fuzz coverage for schemas, mutation
   application, journal replay, pairing, and gate determinism.
8. **Still open:** add process-level isolation, network/filesystem/process
   controls, credential containment, and an out-of-process grader.

The logical exit criterion is now covered by the controlled adversarial tests:
injected failures cannot produce `PROMOTE`, including a caller-authored
mechanism `passed=true` and a forged-score attack that changes a genuine
`REJECT` into a matching recomputed `PROMOTE`. Frozen evidence replays to the
same decision while the original HMAC verification capabilities remain
available. The HMACs authenticate producer assertions and source hashes but do
not yet re-execute those sources. This does not establish malicious-code
containment or durable independent replay; typed source replay, process
isolation, protected or asymmetric keys, authenticated recovery, and
cross-process coordination remain open.

### M1 — one real benchmark, fixed-model runner, and prompt evolution

1. Select a deterministic or replayable benchmark with a clear data licence.
2. Implement a production runner that keeps the model and inference settings
   fixed, captures score-free candidate output, and hands it to the trusted
   grading interface.
3. Freeze exploration/gate/sealed hashes, grader, model settings, runtime image,
   budgets, primary metric, constraints, and stopping rule.
4. Implement evidence extraction, failure signatures, diagnostic clusters, and
   mechanism-diverse atomic prompt candidates.
5. Compare static, random-valid, prompt-only, and evidence-targeted strategies
   across independent search runs under the same total budget.
6. Add holdout-burn accounting, candidate-family correction, winner
   replication, and exactly one sealed final evaluation.

Exit criterion: report held-out gain with uncertainty, evaluation cost,
promotion/rejection errors on controlled faults, and full artifact lineage.

### M1.5 — skill evolution

1. Define immutable skill packages and provenance.
2. Implement metadata/rules/full progressive loading.
3. Probe selection, activation, adherence, behavior, task benefit, conflicts,
   and context cost.
4. Introduce shadow/canary/active/retired/quarantined states.
5. Allow only declarative validation initially; executable validators stay in
   an untrusted sandbox.

Exit criterion: a skill gain survives held-out paired evaluation and removal or
revert ablation without prompt+skill credit ambiguity.

### M2 — higher-risk Harness surfaces

Open memory, tools, middleware, and control flow one at a time. Each surface
requires a versioned capability policy, instrumentation, adversarial fixture,
rollback path, and surface-specific mechanism probe before it may enter search.
Multi-component candidates require an interaction design and a complete merged
gate.

### M3 — continual adaptation and platform scale

Add a capacity-bounded catalog, task-only selector, shadow routing, fallback,
and separate population-quality/routing-quality metrics. Gateway, provider
matrix, UI, remote workers, and training adapters can then grow as optional
packages around the stable research kernel.

## 8. Verification backlog beyond the current gate

The existing paired gate and semantic controller now provide a replayed
experiment-wide gate query/resource ledger, exact evidence-complete branches,
typed selection closure, and one-way sealed-run authorization. They are still
necessary rather than sufficient for adaptive search. The following controls
are required before making strong real-benchmark claims:

- extend accounting from synthetic gate batches to every production gate and
  holdout attempt, including pre-execution reservations for runner failures and
  retries that never emit a batch;
- family-wise or false-discovery control for multiple candidates;
- independent re-evaluation of the selected winner to reduce winner's curse;
- a preregistered sequential rule for spending more trials on `INCONCLUSIVE`;
- cluster-aware or hierarchical intervals when benchmark tasks are related;
- explicit minimum detectable effect and power simulation;
- temporal or source-group splits to expose memorization and contamination;
- execute exactly one sealed final batch through the typed authorization path
  with an authenticated sealed-evaluator producer and without allowing its
  result to cause another mutation;
- disable/revert/placebo interventions for mechanism claims;
- separate reporting for statistical evidence, mechanism evidence, safety,
  policy, and resource constraints.

Before concurrent or adversarial workers are in scope, the caller-held
single-writer tails must also be replaced or wrapped by durable cross-process
compare-and-swap, leases, or transactions, and the capability policy must be
enforced by an OS-level worker boundary. Ephemeral HMAC capabilities must be
replaced by protected versioned keys or asymmetric/KMS signatures so historical
evidence can be verified without retaining a signing secret in the verifier.

## 9. Supply-chain policy for a large project

Before importing any external implementation or dataset, record repository,
commit, path, licence, copyright holder, local owner, modification, and allowed
use. Git dependencies and images must be pinned by commit/digest, not a moving
branch or tag. Model weights, prompts, generated code, datasets, and benchmark
containers are provenance-bearing artifacts too.

Planned repository controls:

- `THIRD_PARTY.yml`, `NOTICE`, dataset/model registry, and source attribution;
- Python, frontend, and container lockfiles kept independently;
- SBOM generation, vulnerability audit, secret scan, licence scan, and
  container scan in CI;
- DCO or CLA for contributions and an explicit first-party project licence;
- generated code records its model, prompt/config, source evidence, and review;
- unlicensed reference repositories remain outside the source tree.

The codebase may eventually be large, but size is an outcome of independently
testable capabilities. The quality bar is that every module has an owner,
contract, threat boundary, replay story, and evidence for why it exists.
