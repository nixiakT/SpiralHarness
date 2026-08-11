# M0–M1 architecture

> **Status:** target architecture with an implemented M0.2 logical control
> plane and an M1 replayable automatic-search kernel. M0/M0.2 provides
> immutable schemas and local content-addressed
> artifacts, linked lifecycle journals, trusted mutation admission, typed
> trajectory and mechanism evidence, paired statistics, an adversarial
> controlled fixture, producer-attested mechanism/gate-evidence closure, and a
> semantic experiment controller with budget accounting and typed
> selection/sealed closure. The M1 foundation adds a pinned GSM8K adapter,
> provider-neutral fixed/replay runner, pre-execution attempt accounting, the
> frozen four-condition baseline protocol, receipt-backed exploration screens,
> bounded diagnosis/proposal/search orchestration, and a study-level all-run
> barrier. A first declarative skill slice adds canonical package loading,
> declared-generated rules-only revision admission, exact prompt/skill
> materialization, and backend-request/receipt replay. It does not yet add skill
> proposals, search,
> adherence/behavior verification, or a skill promotion result. These trust
> boundaries remain in-process: there is no OS sandbox,
> network/filesystem/process isolation, cross-process durable compare-and-swap
> or lease, or live provider. The multi-condition fixture is not a reportable
> benchmark result.

This document turns the contracts in the [research plan](research-plan.md) and
[verification protocol](verification-protocol.md) into implementable module and
data boundaries. Those documents remain authoritative for the research claims
and promotion policy.

## 1. Two planes and one execution boundary

```text
                         immutable, typed artifacts
  Mutable evolution plane ------------------------------> Trusted verification plane
  observe / diagnose / propose                            freeze protocol and splits
  search / archive / baselines                            schedule matched trials
             ^                                            grade / account / decide
             | redacted gate result                                  |
             +-------------------------------------------------------+
                                  |
                     untrusted candidate runtime
              (target: ephemeral sandbox per trial)
```

The **mutable evolution plane** may construct new harness component artifacts
and update its search working state. In M1 it consumes exploration evidence,
diagnoses failures, creates bounded typed patches, and nominates immutable
candidates. The search implementation and configuration are still frozen per
experiment so that they do not become an unrecorded confound. This plane never
writes trial results, scores, decisions, split manifests, or historical
artifacts.

In the target architecture, the **trusted verification plane** owns protocol
manifests, split loading, artifact persistence, the execution controller,
sandbox policy, graders, resource accounting, paired statistics, promotion
rules, access views, and the sealed-set controller. Its version and
configuration are frozen when an experiment starts. A change creates a new
protocol lineage; it does not update an experiment in place. “Immutable” here
means frozen inputs plus append-only outputs: the plane creates evidence and
transition events but never rewrites an existing artifact.

Target execution deliberately straddles the boundary. A trusted controller
resolves a snapshot, constructs a run fingerprint, provisions an isolated
worker, and captures events. The harness and any model-produced output inside
that worker are untrusted. They receive only the task inputs and component
artifacts needed for that trial and cannot write the artifact ledger or invoke
a grader.

M0.2 implements the typed logical boundary that this execution design will
consume. Candidate traces contain output and capture-owned telemetry but no
score or sealed answer; a distinct trusted grader turns those traces into
observations. A separate trusted capability signs complete gate batches, and
the protocol freezes the accepted signer identity. The grader, signer, and
verifiers still run in the same process. Likewise, the current semantic
controller authorizes state transitions and replays artifact joins, but it does
not launch or isolate candidate workers.

## 2. Module ownership and interfaces

The table describes target ownership. The implemented M0.2 subset and its
limits are stated immediately below it.

| Module | Plane | Owns | May emit |
| --- | --- | --- | --- |
| Domain schemas | Trusted/shared | versioned value objects and state enums; no I/O | validated canonical payloads |
| Artifact repository | Trusted | content-addressed blobs and append-only provenance edges | artifact references |
| Harness registry | Trusted | snapshot resolution, typed patch application, structural validation | child snapshot or validation failure |
| Evolution engine | Mutable | failure localization, hypotheses, mutation strategies, search archive | candidate drafts and nominations |
| Execution controller | Trusted | trial scheduling, fingerprints, limits, worker reset | run and trajectory records |
| Candidate runtime | Mutable/untrusted | resolved prompt/skill behavior for one trial | structured events and final output |
| Benchmark adapter | Trusted | task loading, split membership, grading contract | private measurements |
| Verifier | Trusted | probes, paired comparisons, constraints, statistical summaries | decision artifact |
| Experiment controller | Trusted | lifecycle, budgets, query counts, champion pointer | state-transition events |
| Access view | Trusted | role- and partition-specific disclosure | exploration evidence or redacted decision |

Target cross-plane calls use schema-versioned artifact references rather than
shared mutable Python objects. The evolution engine submits a frozen
`CandidateManifest`; the verifier returns a `DecisionArtifact` through an
access view. Benchmark adapters do not import evolution code, and graders do
not execute inside candidate workers.

M0/M0.2 now provides the domain, repository, atomic harness registry, lifecycle
projection/journals, verifier, frozen experiment contracts, trusted candidate
admission, gate and terminal-decision joins, interface protocols, typed
evidence schemas, trusted in-process grading, and a deterministic adversarial
controlled-fault adapter. The `ExperimentController` is the semantic owner of
candidate admission, probe, gate, evidence-complete, and terminal transitions;
it also owns experiment lifecycle transitions and an experiment-wide
query/resource ledger. The journals remain structural storage, while the
controller performs the semantic authorization.

The M1 foundation adds a byte-pinned GSM8K registry/downloader, a trusted
question-only adapter with frozen exploration/gate/sealed membership, a
provider-neutral score-free runner, and a pre-execution reservation/outcome
ledger. It also freezes the static, random-valid, prompt-only, and
evidence-targeted comparison contract. The automatic-search kernel now consumes
that contract through exact run manifests, scoped strategy views, typed
diagnosis/proposal artifacts, receipt-backed exploration screens, one-candidate
gate nominations, aggregate-only feedback, and immutable search/study journals.
The GSM8K runner path and the automatic loop are each replayable, but a
credentialed live provider and a reportable end-to-end study are not yet wired.

The declarative skill slice stores one complete canonical JSON `SkillPackage`
per revision. The package binds metadata, ordered rules, procedure, examples,
model/runtime compatibility, a strictly parsed and normalized SPDX licence
expression, a package-declared source class, provenance/compliance references,
and third-party notices; it grants no executable entry point or runtime
permission. `SkillPackageLoader` requires the exact content-addressed bytes for
every declared reference, but follows only the direct package reference and
does not recursively validate typed lineage. M1 does not authenticate source
classification or reference producers, interpret reference semantics, prove
review approval, or establish a legal conclusion. The loader can produce exact
`metadata`, `rules`, or `full` disclosures with fixed delimiters, renderer
identity, bytes, hash, and size. Its revision verifier permits only an immediate
revision whose parent declares `source_kind=generated` and whose rules changed
while every other canonical field stayed fixed. The path rejects packages
declaring first- or third-party source classes; the declaration itself is not a
provenance attestation.

`HarnessMaterializer` resolves exactly one prompt and at most one compatible
skill from a snapshot, with scheduled execution fixed to the `rules`
projection until disclosure policy is frozen into protocol/preflight. It emits
a `ResolvedHarness`; `ModelRequest` v2 carries the exact manifest reference,
base prompt, disclosure, resolved prompt, and their hashes into the backend
call. `ModelExecution` v2 embeds the complete
credential-free `FrozenModelSpec`. Receipt publication and replay use those
two exact inputs to call `HarnessMaterializer.verify_execution_request` and
reconstruct the manifest, prompt, package, and its directly declared reference
bytes from CAS. Before any scheduled backend call, a v2 preflight certificate
freezes that complete spec for the paired batch. Replay rejects a spec mismatch,
recomputes each paired execution fingerprint, and requires parent and candidate
to share the exact task and seed context. This proves package and request
identity, not that the model attended to the skill or that the skill improved
behavior.

These controllers are single-process, single-writer implementations with
caller-held content-addressed tails. They reject stale tails and duplicate
charges inside that process, but they do not provide durable cross-process
compare-and-swap, leases, or transactions. The experiment controller
deliberately rejects any lifecycle tail supplied to a new controller:
structural journal replay is not semantic authorization. Automatic search
likewise rejects a partially open or previously completed run when a fresh
loop instance cannot reconstruct process-local optimizer capabilities and call
counts. The next integration boundary is to derive trusted skill-activation
evidence from materialized request replay, then independently test adherence
and the predeclared behavior change with parent-revert and placebo
interventions. Only after that path feeds the existing mechanism-evidence and
promotion gates should the bounded grammar gain skill proposals. A
credentialed live fixed-model study remains a separate subsequent boundary.

## 3. Artifact model and lineage

The M0 local backend uses SHA-256 over canonical, schema-versioned bytes and
filesystem blobs created atomically without overwriting an existing object. A
transactional SQLite catalog for append-only lineage and lifecycle events is
planned. Storage metadata such as local paths and creation timestamps is not
part of the logical payload hash. A later remote backend must preserve the same
repository interface and immutability.

Core artifacts are:

| Artifact | Required references |
| --- | --- |
| `ProtocolManifest` v3 | split refs, gate-config and capability-policy refs, gate implementation and mechanism/batch-attestor identities, full model-spec fingerprint, grader/runtime/model settings, budgets |
| `ExperimentManifest` | protocol hash, seed harness, objectives, baselines, stopping rule |
| `HarnessSnapshot` | typed component name → component artifact hash |
| `HarnessPatch` | parent snapshot, changed component(s), canonical before/after hashes |
| `SkillPackage` | direct parent ref (existence only unless it is the immediate revision edge being checked), fixed model/runtime compatibility, normalized SPDX expression, package-declared source class, provenance/compliance refs (existence only), notices, and declarative content |
| `SkillDisclosure` | exact package, disclosure level, renderer, exposed sections, context bytes, hash, and size |
| `ModelRequest` v2 | task, exact harness-manifest ref, base prompt, optional skill disclosure, resolved prompt and hashes, user prompt, and seed |
| `ModelExecution` v2 | complete frozen model spec, exact request, score-free output/status/usage, paired execution identity, and request hash |
| `SchedulePreflightCertificate` v2 | exact schedule, complete model spec, ledger start boundary, budget, and worst-case paired-batch capacity |
| `CandidateManifest` | parent and child snapshots, patch, evidence IDs, hypothesis, risks, evaluation plan |
| `CapabilityPolicy` | default-deny policy and exact per-capability resource grants |
| `RunRecord` | task reference, snapshot, seed, execution fingerprint, resource usage, status |
| `Trajectory` | run ID and ordered, typed runtime events |
| `Measurement` | run ID, frozen grader, metric values and error status |
| `AttestedMechanismEvidence` | signed protocol/candidate/child-harness/exploration context, source artifacts, complete mechanism checks, producer and HMAC |
| `GateTrialBatch` | signed protocol/candidate/arm/harness/split context, task-set and mechanism refs, producer sources, non-empty observations, attestor and HMAC |
| `GateEvaluationManifest` | admission, gate config and implementation, gate split, both trial batches, mechanism evidence |
| `DecisionArtifact` | candidate, parent and release anchor, evidence set, gate version, decision and reasons |
| `ExperimentUsageEntry` | exact evaluated candidate/batches and cumulative query/resource charges |
| `SupersededCandidateReport` | stale local promotion, current champion, superseding candidate, and resolved outcome |
| `SelectionClosure` | frozen champion, candidate tail, analysis plan, and usage tail |
| `SealedRunAuthorization` | exact selection branch, champion, sealed split, analysis plan, and usage tail |
| `SealedEvaluationReport` | frozen execution planes, authorization, final result, and evaluator evidence |
| `ExperimentCompletionReport` | sealed authorization and immutable final report |
| `ExperimentInvalidationReport` | integrity/leakage evidence and invalidated source branch |

`HarnessSnapshot` is a manifest, not a copied directory. Applying a patch
creates new component blobs and a new snapshot; unchanged component hashes are
shared. A candidate is one immutable child of one parent. Correcting its
hypothesis, patch, or evaluation plan creates a new candidate linked by
`supersedes`, never an in-place edit.

Every persisted result has enough references to reconstruct this chain:

```text
protocol -> experiment -> epoch -> candidate -> child snapshot
                                  |             |
                       evidence trajectories   +-> run -> trajectory
                                  |                       -> measurement
                                  +---------------------------> decision
```

The catalog records relation type and producer identity. At minimum it supports
`derived_from`, `patches`, `uses_evidence`, `executed_as`, `measured_by`,
`decided_by`, and `supersedes`. Lifecycle state is a projection of append-only
events; changing a mutable database row is not sufficient evidence. Cache reuse
is legal only for an exact execution fingerprint match.

Artifact hashes are identifiers, not access tokens. In the target runtime,
mutable workers have no general `list` or `get` access to the repository; the
trusted controller materializes an explicit read-only view for each run. M0.2
does not yet enforce that worker view at a process or filesystem boundary.

## 4. Lifecycle state machines

Candidate transitions are controlled by the experiment controller:

```text
DRAFT --freeze manifest--> REGISTERED --schema/policy checks--> VALID
                               |                                  |
                               +--> INVALID                       +--> RUNNING_PROBES
                                                                    |
                                              probe failure --------+--> REJECTED
                                                                    |
                                                             RUNNING_GATE
                                                                    |
                                                           EVIDENCE_COMPLETE
                                                              /     |      \
                                                      PROMOTED  REJECTED  INCONCLUSIVE
```

`DRAFT` exists only in the mutable plane. From `REGISTERED` onward, manifests
and recorded evidence are immutable. Terminal outcomes remain archived. An
`INCONCLUSIVE` candidate receives more trials only when its preregistered
sequential policy permits it; otherwise further work is a new candidate or
experiment. Promotion atomically appends a decision and advances the champion
pointer while retaining the previous champion for rollback.

Multiple siblings may finish against the same parent. If one sibling advances
the champion first, a later valid local `PROMOTE` is preserved in a typed
`SupersededCandidateReport` but resolves to terminal `INCONCLUSIVE`; it cannot
replace the new champion, and it no longer leaves selection blocked by a
nonterminal candidate.

The current `CandidateJournal` enforces immutable links, sequence, stream,
candidate identity, and the legal transition graph. The implemented semantic
`ExperimentController` owns trusted admission, required-probe outcomes, gate
accounting, evidence completion, and terminal authorization. It replays the
referenced semantic joins before publishing `VALID`, `REJECTED`,
`EVIDENCE_COMPLETE`, `PROMOTED`, or `INCONCLUSIVE` events rather than treating
the structural journal as authority.

Probe results are stored as `AttestedMechanismEvidence`. A separately
domain-separated producer capability signs the protocol, candidate and child
harness, exploration split/task set, frozen execution context, complete checks,
and concrete source-artifact refs. The protocol pins that producer ID. The
controller verifies the envelope and source availability before a probe can
produce either `RUNNING_GATE` or `REJECTED`, and repeats the verification when
replaying candidate history. A raw caller-authored `passed=true` artifact has
the wrong media/schema and no accepted attestation.

`GateTrialBatch` v2 is an HMAC-attested envelope. Its signed content includes
the protocol, candidate, parent/candidate arm, harness, gate split and task-set
identity, complete observations and resource fields, frozen execution context,
source artifacts, and exact mechanism-evidence reference. The protocol freezes
the accepted attestor ID. Both the controller and terminal verifier require the
matching verification capability and recheck every join before scores or
resource claims are consumed. `GateEvaluationManifest` then closes the gate
input set over the verified admission report, frozen gate config, protocol gate
split, both arm batches, and mechanism evidence. The protocol, evaluation
manifest, and decision report also bind the promotion-gate implementation
fingerprint. Terminal authorization recomputes the decision from those frozen
inputs and binds it, the charged usage entry, and the exact
`EVIDENCE_COMPLETE` journal tail before a terminal event can be appended.

Implemented experiment transitions are:

```text
FROZEN -> SEARCHING -> SELECTION_CLOSED -> SEALED_RUNNING -> COMPLETE
   \          \                \                 \
    +----------+----------------+-----------------+--> INVALIDATED
                         integrity or leakage
```

`FROZEN` binds the protocol, splits, budgets, seed harness, and stopping rule.
`SELECTION_CLOSED` binds the final champion and analysis plan before any sealed
task is opened. A protocol modification forks a new experiment. Cancellation
may stop work, but it cannot erase completed records.

The lifecycle is recorded as typed linked events. `SelectionClosure`,
`SealedRunAuthorization`, and `ExperimentCompletionReport` close the normal
one-way path; `ExperimentInvalidationReport` records the evidence for an
integrity- or leakage-triggered transition to `INVALIDATED`. These are logical
artifact and state boundaries, not OS capability boundaries.

## 5. Exploration, gate, and sealed information flow

The target architecture enforces the three partitions as capabilities exposed
by `AccessView`, not as naming conventions:

1. **Exploration:** the split loader sends tasks to isolated runs. Full allowed
   feedback and trajectories may be released to the evolution engine and cited
   by candidates.
2. **Gate:** the engine can only nominate a registered candidate. The trusted
   execution service privately runs candidate/parent pairs, and the verifier
   records per-task evidence. The engine receives only the configured aggregate
   summary or `promote` / `reject` / `inconclusive`; task bodies and per-task
   gate trajectories never cross back. Every nomination consumes the frozen
   query budget.
3. **Sealed:** no search process receives a handle to this view. After
   `SELECTION_CLOSED`, a separate controller evaluates exactly the frozen
   champion and emits the final report. Its result cannot trigger another
   candidate in the same experiment lineage.

M0.2 and the automatic-search kernel enforce the logical portion of this flow.
Gate use is charged from the referenced parent and candidate batches; selection
freezes the champion, analysis plan, and usage tail. Sealed access requires an
authorization bound to that exact closed branch and frozen sealed split. The
strategy-facing access view is now an exact per-round allowlist with no
repository-list operation, and gate feedback has a fixed aggregate-only
schema. The current controller still
does not provide an isolated worker or prevent a caller process from holding
additional references. Its study barrier detects premature sealed transitions
but cannot revoke an older controller reference retained in the same
interpreter.

In the target runtime, gate and sealed workers must have no network egress or
persistent candidate-owned state. Otherwise a harness could copy private tasks
into memory and expose them on a later exploration run. Trial order and public
identifiers must also avoid revealing split membership.

## 6. Security and integrity boundary

For the target M0–M1 runtime, the following are invariants, not optimizer
preferences:

- Candidate patches cannot target split manifests, task data, graders, gate
  code/configuration, sandbox policy, accounting, the artifact repository, or
  experiment state.
- Candidate snapshots are read-only; each task receives a fresh temporary
  workspace and reset process state. Model/tool credentials remain in the
  trusted wrapper and are never persisted in trajectories.
- Network, filesystem, subprocess, tool, token, time, and monetary access are
  denied by default and granted through versioned allowlists and hard limits.
- The grader consumes captured output outside the candidate process. Candidate
  output is data; it cannot request grading actions or write a score.
- Every run fingerprint binds the snapshot, task/version, seed, model and
  inference settings, tool/runtime image, limits, and grader-facing output
  schema. Missing or mismatched fields fail closed.
- Only trusted services append artifacts and transitions. Malformed events,
  incomplete pairs, budget overruns, leakage, or integrity failures cannot be
  converted into a promotion by the evolution engine.

M0.2 currently enforces a narrower, typed logical subset:

- `CapabilityPolicy` is deny-by-default, supports only exact resource grants,
  and is bound through the protocol and candidate admission report. It is a
  schema/admission contract for a future trusted launcher, not an OS-enforced
  sandbox or system-call policy.
- Candidate-facing traces contain no score, sealed answer, or grader action.
  The trusted fixed-output grader validates captured digests, roster/task
  identity, policy limits, and mechanism telemetry before producing an
  observation. This separation is in-process, not out-of-process.
- Mechanism evidence is signed by its own protocol-pinned producer capability.
  Its HMAC covers the candidate, child harness, exploration split, execution
  context, complete checks, and their concrete CAS sources; the controller
  verifies it before the lifecycle can enter the gate and terminal replay
  verifies it again before promotion.
- Gate batches are signed by a process-local capability whose ID is frozen in
  the protocol. The HMAC covers observations, resource use, execution context,
  sources, and mechanism evidence; the controller verifies it before
  accounting, and the terminal service verifies it again before promotion.
- The evaluation manifest, gate implementation fingerprint, terminal
  recomputation, exact journal tails, and cumulative budget/query entries close
  the remaining evidence path used for promotion.
- The semantic controller rejects stale caller-held tails, duplicate charging,
  forged evidence branches, wrong signers, budget overruns, and illegal
  candidate or experiment lifecycle transitions within one process. A new
  controller rejects every supplied experiment lifecycle tail rather than
  mistaking structural replay for semantic recovery.

M0.2 does **not** provide fresh OS workspaces, process reset, network or
filesystem isolation, subprocess controls, credential isolation, or an
out-of-process grader. Its symmetric verifiers contain HMAC secrets, and
ephemeral key loss makes historical evidence unverifiable outside the original
process. Producer attestations authenticate complete assertions and source
hashes, but M0.2 does not replay those sources to re-derive mechanism booleans
or scores. M0.2 itself also has no durable multi-process compare-and-swap,
lease, transactional head update, pre-execution attempt reservation, or
authenticated sealed-evaluator producer. M1 now supplies a local trusted-runner
reservation stream, while the remaining isolation, attestation, and durable
coordination controls must still be implemented before those broader claims
apply.

The controlled adversarial suite now covers no-op, wrong-target, inactive,
non-adherent, placebo, regression, timeout, malformed, tamper, budget, and
leakage cases, all of which fail closed instead of producing `PROMOTE`. These
fixtures validate logical grading and control-path integrity; they are not
sandbox-escape tests.

## 7. Mutation rollout order

M1 automatic search first enables one-component **prompt mutations**. Prompts
are canonical text artifacts: diffs are easy to inspect and revert, activation
is observable at context construction, and there are no new execution
privileges. This is the narrowest surface on which to validate lineage, matched
evaluation, and the promotion gate.

M1 now also implements a deliberately narrower direct **skill revision**
slice. A candidate may replace one declarative package only when an explicit
skill mutation policy is selected; admission permits an immediate rules-only
revision when the parent package declares `source_kind=generated` and freezes
every other package field. It verifies that direct revision edge, not
recursively typed historical lineage or authenticated provenance. The loader
supports three projections, while scheduled materialization always uses
`rules`; selection, loading, and backend-request construction are replayable.
They do not establish adherence or benefit, and the automatic proposal
catalogue and strategy grammar remain prompt-only. Arbitrary Python,
tool implementation changes, package metadata rewrites, and combined
prompt+skill edits are not classified as this M1 skill operation.

Memory, tools, middleware, and control flow remain planned for M2. They add,
respectively, persistent/private state, side effects and credentials,
observation-path interference, and topology/concurrency changes. Each needs a
new policy validator, instrumentation contract, and threat-model fixture.
Prompt+skill edits also remain disabled by default until an explicit
interaction experiment can attribute their joint effect.

## 8. Python package layout

```text
src/spiral_harness/
  cli.py
  core/            # implemented: immutable state, experiment and lifecycle schemas
  storage/         # implemented: CAS objects and content-addressed linked journal
  evidence/        # implemented: trajectory, span, packet and diagnosis contracts
  verification/    # implemented: paired gate, bound trial batches, closure artifacts
  benchmark/       # controlled fixture + pinned GSM8K registry, split and trusted grader
  harness/         # implemented: atomic policy validation/snapshot application
  skills/          # canonical declarative packages, disclosure, and revision verification
  execution/       # harness materialization, fixed/replay runner, receipts, schedules, ledger
  evolution/       # typed strategies, scoped artifact views, automatic search orchestration
  experiments/     # experiment/search/study controllers, gate replay, four-condition protocol
```

`src/spiral_harness` is one package path, not two nested application layers:
`src` is the distribution build root and `spiral_harness` is the Python import
namespace. GitHub and VS Code compact single-child directories into the label
`src/spiral_harness`; that display does not indicate an empty directory. Keeping
the build root prevents a checkout-local package from masking a broken wheel.
Every domain directory above contains implemented code; future memory, tool,
service, and UI directories are added only with their first real module.

Domain `__init__.py` files do not aggregate implementation symbols. Internal
and test code imports each symbol from its owning leaf module, so importing a
small contract does not eagerly load controllers, replay fixtures, or workflow
composition. Architecture tests reject package-barrel imports, local dynamic
imports, dependency cycles, import-only forwarding modules, and accidental
second package roots. The root package remains the intentionally small public
entry point for package metadata; the CLI is exposed through the installed
`spiral` command.

The target dependency direction is inward: infrastructure modules implement
interfaces declared in the core/domain layer; `evolution` and benchmark
adapters do not own storage or gate policy; `experiments` composes services but
contains no scoring logic. The CLI invokes experiment use cases and never
bypasses state transitions.

The controlled end-to-end vertical slice exercises a prompt fault, one
typed repair, paired deterministic trials, trusted grading, immutable gate and
lifecycle closure, and replay from artifact hashes. The M1 runner/adapter
integration additionally exercises score-free GSM8K execution, pre-call
reservation, and trusted grading. The automatic-search fixture executes the
four matched strategy conditions across multiple independent run seeds through
the same bounded controller path and study barrier. It remains explicitly
non-reportable. Skill integration tests establish canonical package loading,
direct-edge revision admission, exact rules-only materialization, and
request/receipt replay;
they do not establish skill adherence, a causal behavior change, or benchmark
gain. A live provider, OS isolation, and durable multi-process coordination
remain future integration and hardening work.
