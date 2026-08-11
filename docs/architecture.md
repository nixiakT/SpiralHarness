# M0–M1 architecture

> **Status:** target architecture. M0 currently implements immutable core
> schemas, content-addressed local artifacts, paired statistics, and a promotion
> gate. Other modules and isolation guarantees below remain planned until they
> are explicitly marked as implemented.

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
                    (ephemeral sandbox per trial)
```

The **mutable evolution plane** may construct new harness component artifacts
and update its search working state. In M1 it consumes exploration evidence,
diagnoses failures, creates bounded typed patches, and nominates immutable
candidates. The search implementation and configuration are still frozen per
experiment so that they do not become an unrecorded confound. This plane never
writes trial results, scores, decisions, split manifests, or historical
artifacts.

The **trusted verification plane** owns protocol manifests, split loading,
artifact persistence, the execution controller, sandbox policy, graders,
resource accounting, paired statistics, promotion rules, access views, and the
sealed-set controller. Its version and configuration are frozen when an
experiment starts. A change creates a new protocol lineage; it does not update
an experiment in place. “Immutable” here means frozen inputs plus append-only
outputs: the plane creates evidence and transition events but never rewrites an
existing artifact.

Execution deliberately straddles the boundary. A trusted controller resolves a
snapshot, constructs a run fingerprint, provisions an isolated worker, and
captures events. The harness and any model-produced output inside that worker
are untrusted. They receive only the task inputs and component artifacts needed
for that trial and cannot write the artifact ledger or invoke a grader.

## 2. Module ownership and interfaces

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

Cross-plane calls use schema-versioned artifact references rather than shared
mutable Python objects. The evolution engine submits a frozen
`CandidateManifest`; the verifier returns a `DecisionArtifact` through an
access view. Benchmark adapters do not import evolution code, and graders do
not execute inside candidate workers.

M0 is scoped to the domain, repository, harness registry, verifier, experiment
controller, and a deterministic controlled-fault adapter. M1 is scoped to a
fixed-model runtime, one real benchmark adapter, trajectory-driven evolution,
and static, random-valid, and prompt-only experiment strategies.

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
| `ProtocolManifest` | split-manifest hashes, grader/gate versions, runtime policy, model settings, budgets |
| `ExperimentManifest` | protocol hash, seed harness, objectives, baselines, stopping rule |
| `HarnessSnapshot` | typed component name → component artifact hash |
| `HarnessPatch` | parent snapshot, changed component(s), canonical before/after hashes |
| `CandidateManifest` | parent and child snapshots, patch, evidence IDs, hypothesis, risks, evaluation plan |
| `RunRecord` | task reference, snapshot, seed, execution fingerprint, resource usage, status |
| `Trajectory` | run ID and ordered, typed runtime events |
| `Measurement` | run ID, frozen grader, metric values and error status |
| `DecisionArtifact` | candidate, parent and release anchor, evidence set, gate version, decision and reasons |
| `FinalReport` | frozen champion, analysis plan, one sealed evaluation batch |

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

Artifact hashes are identifiers, not access tokens. Mutable workers have no
general `list` or `get` access to the repository; the trusted controller
materializes an explicit read-only view for each run.

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

Experiment transitions are:

```text
DRAFT -> FROZEN -> SEARCHING -> SELECTION_CLOSED
                                  -> SEALED_RUNNING -> COMPLETE
       \---------------- integrity or leakage ----------------> INVALIDATED
```

`FROZEN` binds the protocol, splits, budgets, seed harness, and stopping rule.
`SELECTION_CLOSED` binds the final champion and analysis plan before any sealed
task is opened. A protocol modification forks a new experiment. Cancellation
may stop work, but it cannot erase completed records.

## 5. Exploration, gate, and sealed information flow

The three partitions are enforced as capabilities exposed by `AccessView`, not
as naming conventions:

1. **Exploration:** the split loader sends tasks to isolated runs. Full allowed
   feedback and trajectories may be released to the evolution engine and cited
   by candidates.
2. **Gate:** the engine can only nominate a registered candidate. The trusted
   controller privately runs candidate/parent pairs, and the verifier records
   per-task evidence. The engine receives only the configured aggregate summary
   or `promote` / `reject` / `inconclusive`; task bodies and per-task gate
   trajectories never cross back. Every nomination consumes the frozen query
   budget.
3. **Sealed:** no search process receives a handle to this view. After
   `SELECTION_CLOSED`, a separate controller evaluates exactly the frozen
   champion and emits the final report. Its result cannot trigger another
   candidate in the same experiment lineage.

Gate and sealed workers have no network egress or persistent candidate-owned
state. Otherwise a harness could copy private tasks into memory and expose them
on a later exploration run. Trial order and public identifiers must also avoid
revealing split membership.

## 6. Security and integrity boundary

For M0–M1 the following are invariants, not optimizer preferences:

- Candidate patches cannot target split manifests, task data, graders, gate
  code/configuration, sandbox policy, accounting, the artifact repository, or
  experiment state.
- Candidate snapshots are read-only; each task receives a fresh temporary
  workspace and reset process state. Model/tool credentials remain in the
  trusted wrapper and are never persisted in trajectories.
- Network, filesystem, subprocess, tool, token, time, and monetary access are
  denied by default and granted through versioned allowlists and hard limits.
- The grader consumes captured output out of process. Candidate output is data;
  it cannot request grading actions or write a score.
- Every run fingerprint binds the snapshot, task/version, seed, model and
  inference settings, tool/runtime image, limits, and grader-facing output
  schema. Missing or mismatched fields fail closed.
- Only trusted services append artifacts and transitions. Malformed events,
  incomplete pairs, budget overruns, leakage, or integrity failures cannot be
  converted into a promotion by the evolution engine.

The controlled-fault suite should exercise these boundaries before a real
benchmark is used. The concrete adversarial cases and statistical requirements
remain specified in the verification protocol.

## 7. Mutation rollout order

M1 first enables one-component **prompt mutations**. Prompts are canonical text
artifacts: diffs are easy to inspect and revert, activation is observable at
context construction, and there are no new execution privileges. This makes
them the narrowest surface on which to validate lineage, matched evaluation,
and the promotion gate.

After that path is stable, M1 enables one-component **skill mutations** limited
to declarative skill manifests and reusable text procedures. Skills add a
second mechanism to verify—selection and loading before adherence—and therefore
test whether the architecture handles conditional activation. Arbitrary Python
or tool implementation changes are not classified as M1 skill edits.

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
  core/            # implemented: immutable schemas and canonical hashing
  storage/         # implemented: content-addressed local artifact objects
  verification/    # implemented: pairing, statistics, constraints, promotion gate
  benchmark/       # planned: adapter protocol, controlled fixture, M1 adapter
  harness/         # planned: snapshots, patch application, component validators
  execution/       # planned: controller, sandbox, fingerprints, event capture
  evolution/       # planned: diagnosis, proposal, search strategies, baselines
  experiments/     # planned: budgets, state-machine controller, orchestration
  access/          # planned: exploration/gate/sealed views and disclosure policy
```

The target dependency direction is inward: infrastructure modules implement
interfaces declared in the core/domain layer; `evolution` and benchmark
adapters do not own storage or gate policy; `experiments` composes services but
contains no scoring logic. The CLI invokes experiment use cases and never
bypasses state transitions.

The first end-to-end vertical slice should be deliberately small: a controlled
prompt fault, one typed repair, paired deterministic trials, an immutable
decision, and complete replay from artifact hashes. The real benchmark and
automated search are added only after that slice can detect no-op, regression,
tampering, and inconclusive candidates correctly.
