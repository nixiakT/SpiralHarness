# Verification protocol

Self-verification is the trust boundary of SpiralHarness. It is not a second
prompt asking the proposing model whether its idea looks good. It is an
independent protocol that attempts to falsify every claimed improvement.

## 1. Separation of roles

The system has four logical roles:

- **Executor:** runs a frozen harness on a task and emits a trajectory.
- **Diagnoser/Proposer:** reads permitted evidence and creates candidates.
- **Evaluator:** computes task outcomes using a frozen grader.
- **Gate:** applies a deterministic, versioned promotion rule.

The proposer cannot edit benchmark data, split membership, graders, the gate,
or historical artifacts during an experiment. Artifacts exchanged across these
roles are content-addressed and record their provenance.

In the target architecture, the split loader, grader, worker
sandbox/permissions, resource accounting, statistics, gate, artifact ledger,
and sealed-set controls form a trusted verification plane. Its version is
frozen for an experiment and is outside the candidate's mutable surface.
Changing it starts a new protocol lineage.

M0.2 implements the typed, in-process portion of that boundary. Candidate
traces contain output and runner-captured telemetry but cannot contain scores;
the trusted grader alone holds sealed answers and creates gate observations.
Parent-held capabilities authenticate captures, mechanism-probe results, and
complete gate batches, and the protocol freezes independent accepted
mechanism-evidence and gate-batch attestor IDs. The grader, signers, and
symmetric verification capabilities nevertheless run in the same process as
the controlled fixture. A deny-by-default capability schema is bound into the
protocol and checked at admission, but it is not OS enforcement. M0.2 has no
enforced network, filesystem, or process isolation and no worker sandbox.
The protocol schema is version 4 and is accepted only under its exact v4 media
type. Version 4 adds an optional JSON reference that freezes the skill
verification policy; prompt-only protocols persist it as `null`. The full
model-spec fingerprint remains bound for scheduled preflight authorization,
and a payload under another version's media type fails closed.

## 2. Data boundaries

Use three conceptually distinct partitions:

1. **Exploration:** full tasks, feedback, and trajectories may guide diagnosis
   and candidate creation.
2. **Gate:** used for candidate selection. The optimizer receives only the
   information explicitly allowed by the protocol, and gate queries consume a
   fixed budget. Repeated use makes this validation data, never final test data.
3. **Sealed test:** inaccessible to the optimizer and untouched until a harness
   and analysis plan are frozen. It supports the final generalization claim,
   not iterative selection.

Split manifests contain stable task IDs and hashes. Related tasks should be
grouped by template, repository, generator, or time before splitting so that
near-duplicates cannot cross boundaries. Any task contamination or split change
invalidates the corresponding experiment lineage.

Repeated gate access creates adaptive leakage even when task contents stay
hidden. Each evolution epoch therefore nominates a bounded number of candidates,
consumes a declared gate/query budget, and receives only the configured summary
or `promote`/`reject`/`inconclusive` decision. Per-task gate trajectories are not
returned to the proposer. A viewed sealed set is no longer sealed.

The M0.2 semantic controller records each gate query in an immutable
experiment-wide usage ledger. It recomputes evaluation, token, tool-call,
wall-time, and cost usage from the referenced parent/candidate batches; rejects
duplicate candidate or evaluation claims, stale usage tails, and budget
overruns; and binds selection closure and sealed-run authorization to the exact
usage and lifecycle branches. These are logical one-way controls inside one
trusted process, not OS-level capability separation.

This gate ledger accounts for persisted signed batches after execution. The M1
fixed runner now adds a separate content-addressed single-use reservation before
each local backend launch and settles success or burns failure afterward. Its
budget fingerprint is bound into every link. The runner ledger is still one
in-process writer and is not yet atomically joined to gate-batch publication or
protected by a cross-process lease.

## 3. Candidate preregistration

Before seeing candidate results, persist a manifest containing:

- parent harness hash and candidate patch hash;
- changed components and evidence trajectory IDs;
- failure location and proposed root cause;
- expected activation, adherence, behavior change, downstream improvement, and
  protected capabilities;
- anticipated regressions and affected slices;
- primary metric, constraints, sample plan, seeds, and gate version.

This prevents post-hoc changes to the claimed target or acceptance criterion.

The frozen M0.2 protocol also references an exact capability policy. Its empty
grant set denies every declared capability by default, and candidate admission
rejoins that reference with the experiment and protocol. A future trusted
worker launcher must enforce those grants; the current schema does not itself
restrict network, filesystem, secrets, subprocesses, or tools.

For a skill candidate, admission additionally reloads the exact parent and
child `SkillPackage` artifacts and replays an immediate rules-only revision
when the parent declares `source_kind=generated`. That declaration is not an
attestation, and the check covers the direct edge rather than recursively typed
historical lineage. `HarnessMaterializer` then resolves exactly one prompt and
at most one compatible skill using a fixed `rules` disclosure for scheduled
execution; `ModelRequest` v2 binds the exact manifest ref, `SkillDisclosure`,
base prompt, resolved prompt, and hashes into the structured request passed to
the backend adapter. `ModelExecution` v2 embeds the credential-free frozen model
spec, allowing receipt publication and replay to reconstruct that same request
from the manifest, prompt, package, and its directly declared reference bytes
in CAS. For a settled attempt, replay rejoins the canonical skill-bearing
request to the completed execution, `SETTLED` attempt outcome, receipt, and
ledger. This establishes exact settled skill request inclusion/replay as a
prerequisite only. It does not prove remote-provider delivery, model attention
or activation, adherence, behavior change, downstream benefit, or
generalization, and it creates no skill mechanism check or promotion authority.

Protocol v4 may additionally bind a `SkillVerificationPolicy`. Before a skill
candidate enters `RUNNING_PROBES`, its `SkillMechanismPlan` must exactly rejoin
that policy with the experiment, candidate, atomic rules revision, parent and
child harnesses, typed exploration probe roster, deterministic neutral rules,
matched placebo package/harness, and both treatment-vs-revert and
treatment-vs-placebo schedules. The schedules share every coordinate except the
control harness, including `study`, `kind`, `query`, task/seed axes, retries,
and token ceilings. The controller records the plan in the lifecycle and
replays it again before gate authorization and when gate history is consumed.

The implemented executor requires a one-shot capability issued by the same
live controller that owns the current `RUNNING_PROBES` head. The grant binds
the exact experiment, protocol, candidate, plan, lifecycle tail, and repository
object; stale, foreign, reconstructed, or already consumed grants fail closed.
In one process, the executor runs the frozen revert schedule before the frozen
placebo schedule. Each arm receives a new process-local fixed-model runner
wrapper, a distinct writer epoch and attempt ledger, and its own ledger-bound
preflight certificate. Parent and candidate cells execute in preregistered
paired order over exact `ControlledSkillProbeTask` artifacts, whose schema
freezes only task ID and question and cannot represent an answer or gold label.
The caller supplies both backend objects. Their required object inequality
prevents direct reuse, but does not attest provider-session independence or
reset caches, connections, remote sessions, process globals, or runtime state.
The frozen reset fingerprint identifies the agreed wrapper-construction path;
it is not evidence that an external reset occurred.

The resulting `MatchedSkillProbeClosure` replays both live ledgers, receipts,
usage, task bytes, and candidate request-inclusion evidence and rejects
cross-arm reuse. This reverification requires the original controller
verification capability, exact registered closure ref, and both original live
ledger writers. A reconstructed controller/writer epoch or CAS artifacts alone
cannot reproduce the same semantic check, so this is not a durable standalone
certificate. Its companion `SkillProbeShadowReport` reports only exact request
inclusion. Both are explicitly non-promoting: process-local execution records
do not attest remote-provider delivery, runtime activation, adherence, behavior
change, a revert/placebo causal effect, downstream benefit, or generalization.

This layer freezes the exploration-split identity, roster and opaque
claim/probe configuration artifacts by hash; the generic controller does not
yet parse benchmark-specific split membership or derive claim outcomes from
those configs. Evidence thresholds, including token-distance limits, are also
preregistered but not yet consumed. Dedicated producers must later rejoin typed
task membership, apply every threshold, and emit replayable terminal evidence
before any reserved skill check can be unquarantined.

## 4. Matched evaluation

Evaluate the candidate and its exact parent on the same task instances, seeds,
model settings, tool state, and resource limits. Randomize or interleave run
order when environment drift is possible. Cache only when the complete execution
fingerprint matches.

For stochastic agents, use repeated runs or a predeclared seed set. Preserve
per-task paired outcomes; aggregate means alone erase the evidence needed for a
reliable comparison.

Each scheduled paired batch starts from a persisted v3 preflight certificate
that freezes the exact `FrozenModelSpec`, schedule, clean ledger boundary,
process-local writer epoch, and worst-case attempt/token capacity before a
backend call. Receipt publication and trusted replay require every execution to
match that spec, recompute its paired fingerprint, compare parent/candidate
fingerprints for each retry slot, and come from the writer epoch that issued the
preflight. Each v3 reservation and outcome also carries that epoch. Reopening
the same historical CAS tail creates another epoch, leaves the reopened ledger
audit-only, and cannot reuse or repackage the old batch under a new certificate.
The epoch is a public process-local freshness coordinate, not a durable lease:
independent fresh writers still require repository-wide coordination. The
preflight spec fingerprint is also joined to the protocol and frozen run
context, so a self-consistent batch cannot authorize a different backend or
model configuration.

M0.2 closes the gate inputs before a decision is accepted. Each v2 gate trial
batch is signed by a process-local HMAC capability. The signed content binds
the protocol, candidate, arm, harness, gate split and task-set hash, full
observations and resources, model/inference/runtime/sandbox/grader/capability
context, source artifacts, and exact mechanism-evidence reference. The protocol
freezes the accepted attestor ID. The controller verifies the batch before
charging it, and terminal validation verifies it again before recomputing the
decision. The gate evaluation manifest additionally binds admission, the
frozen gate config and split, both batches, mechanism evidence, and the gate
implementation fingerprint also frozen in the protocol.

HMAC here is an in-process capability, not a durable signature. The so-called
verify-only object still contains symmetric key material; arbitrary code in
that interpreter can inspect memory. Losing the ephemeral capability prevents
cryptographic replay in a later process. A production runner therefore needs
protected versioned keys or asymmetric/KMS attestation and must keep signing
authority outside the candidate worker.

The gate-batch HMAC authenticates the trusted producer's assertion. M0.2 checks
that each referenced source artifact exists at its declared hash, but it does
not replay those sources to re-derive the signed observations. End-to-end
grading replay requires typed capture/source bundles and trusted regrading in a
later runner stage.

The task (or a predeclared task group), not each stochastic rollout, is the
independent statistical unit. Multiple rollouts within one task must not be used
to inflate the apparent sample size. Stateful workers reset between tasks unless
persistent state is explicitly part of—and frozen into—the harness candidate.

## 5. Mechanism evidence

A score delta alone cannot show why a change worked. Where the component permits
instrumentation, collect a layered evidence chain:

1. **Patch validity:** the intended artifact is present and structurally valid.
2. **Activation:** the changed prompt, skill, tool, memory item, middleware, or
   control-flow branch is actually reached on the targeted tasks.
3. **Adherence:** the agent uses the activated artifact rather than ignoring or
   contradicting it.
4. **Behavior:** a predeclared observable changes in the expected direction.
5. **Benefit:** matched downstream task outcomes improve.
6. **Generalization:** the benefit survives broader regression and unseen tasks.

Disable, revert, and placebo interventions can distinguish a real mechanism
from coincidental score movement. Not every behavior is perfectly observable;
missing mechanism evidence must be surfaced as uncertainty, never silently
replaced by the proposer's explanation.

M0.2 does not accept a raw `MechanismEvidence` object as lifecycle authority.
The trusted probe producer signs an exact-media envelope covering the protocol,
candidate and child harness, exploration split/task set, frozen execution
context, complete checks, and concrete CAS source refs. Every cited check ref
must name one of those sources. The controller verifies the protocol-pinned
producer, all context joins, and source availability before publishing either a
gate-entry or probe-rejection transition; history and terminal replay repeat
the same verification. This closes the caller-authored `passed=true` path.

As with gate batches, this HMAC authenticates the trusted producer's complete
assertion but does not re-execute its sources to derive each check boolean.
Typed capture bundles and trusted probe replay remain runner-stage hardening.

The skill path now has fail-closed preregistration plus one
controller-authorized, process-local matched execution, but it has not entered
terminal mechanism-evidence production. The policy separates the generic
mechanism signer, aggregate skill signer, and six claim-specific attestors,
producers, verifiers, domains, and configs for provider delivery, runtime
activation, adherence, behavior, revert, and placebo. Request inclusion remains
live-ledger re-derived and cannot be assigned a claim attestor. The candidate
plan reconstructs its matched sham and freezes the two control schedules; the
executor consumes a one-shot grant, runs revert before placebo with distinct
process-local ledger writers and preflights, and publishes a non-promoting
closure over exact tasks, receipts, usage, and request inclusion. The shadow
report is only a request-inclusion audit summary and does not enter mechanism
evidence or the promotion gate. Neither successful execution nor this shadow
closure establishes any claim-specific outcome or control effect.

The next integration must:

1. retain exact settled skill request inclusion/replay, including
   `HarnessMaterializer.verify_execution_request`, as a prerequisite rather
   than relabeling it as activation;
2. independently establish provider delivery and a predeclared runtime
   activation observable;
3. obtain independent evidence for rule adherence and the predeclared behavior
   change rather than inferring either from request inclusion;
4. run a new preregistered scored matched claim-capture study under the same
   frozen construction, with provider-delivery, runtime/adherence/behavior
   capture and replayable revert/placebo scoring;
5. submit those concrete claim sources through
   `TrustedMechanismEvidenceService` and the existing promotion gate.

Until all five steps are present, request inclusion, preregistration, local
execution, and the shadow closure have no promotion authority. Generic signed
checks using any reserved skill ID are filtered, so a skill remains fail-closed
at the mechanism gate.

## 6. Promotion gate

A candidate is eligible for promotion only when all configured checks pass:

1. **Integrity:** manifests, artifact hashes, split boundaries, and run
   fingerprints are valid; required trials are complete.
2. **Primary benefit:** the paired effect on the primary metric exceeds a
   practical minimum, and its confidence lower bound clears the configured
   threshold.
3. **Regression:** protected task slices and critical cases satisfy declared
   non-inferiority floors. A catastrophic failure is a hard rejection even when
   mean score rises.
4. **Budget:** token use, latency, tool calls, and monetary cost remain within
   explicit constraints. The initial protocol does not hide these trade-offs in
   a single weighted reward.
5. **Multiplicity:** selection across many candidates is accounted for through
   a separate confirmation set, correction, or a predeclared sequential testing
   policy.

Small samples may be incapable of proving an improvement. In that case the
correct decision is `inconclusive`, not promotion.

For continued evolution, compare a candidate both with its direct parent and a
stable release anchor. This prevents many individually tolerated regressions
from accumulating unnoticed across generations.

## 7. Recommended initial statistics

The M0 implementation should retain raw paired observations and use a seeded
paired bootstrap confidence interval for the mean score difference. The gate
should expose, rather than hide:

- number of valid pairs;
- parent and candidate means;
- mean paired delta and confidence interval;
- win/tie/loss counts;
- per-slice deltas;
- cost and latency deltas;
- missing, errored, and timed-out trials.

Bootstrap inference is not universally optimal, especially for tiny or highly
discrete samples. Benchmark adapters may supply a stronger paired test, but may
not weaken artifact, leakage, regression, or budget checks.

## 8. Promotion lifecycle

Use staged evidence to control cost and risk:

1. schema and policy validation;
2. deterministic smoke tasks;
3. targeted activation/adherence/behavior probes;
4. targeted replay of the diagnosed failure class;
5. broader gate evaluation against the parent;
6. promotion to champion or rejection/inconclusive;
7. periodic regression suite and drift monitoring;
8. one-time sealed-test evaluation for a frozen experimental result.

Every promotion is reversible because its parent, patch, evidence, environment,
and decision are immutable. Rejected candidates remain useful negative evidence.

The M0.2 controller implements the experiment path `FROZEN` → `SEARCHING` →
`SELECTION_CLOSED` → `SEALED_RUNNING` → `COMPLETE`; proved integrity or leakage
violations can instead move any eligible nonterminal state to `INVALIDATED`.
Typed `SelectionClosure`, `SealedRunAuthorization`,
`ExperimentCompletionReport`, and `ExperimentInvalidationReport` artifacts bind
the champion, analysis plan, sealed split, usage tail, final report, or
violation evidence to the exact lifecycle branch.

This controller uses caller-held content-addressed tails and one in-process
single writer. Stale tails are rejected within that process, but there is no
cross-process durable compare-and-swap, lease, or transaction. Multi-worker
coordination remains future work. A new M0.2 controller rejects every supplied
experiment lifecycle tail, including a genuine one, because the structural
journal alone cannot prove that prior semantic transitions were controller
authorized. Usage-ledger replay remains read-only and requires the original
gate-batch verification capability.

When two siblings both pass against the same parent, only the first may advance
the champion. A later local promotion is retained in a typed supersession proof
and resolved to `INCONCLUSIVE`, so it neither replaces the current champion nor
leaves an active candidate that prevents selection closure.

Before trusting an evolver, test the verifier with adversarial fixtures: no-op
patches, wrong-target mutations, inactive or non-adherent changes, placebo
interventions, protected-slice regressions, timeouts, malformed traces, capture
tampering, budget overruns, and forbidden-resource leakage. The M0.2 controlled
fixture covers all eleven cases and prevents each from being promoted.
Separate control-path regressions also start from a genuine `REJECT`, tamper
both signed score arms, recompute a matching `PROMOTE`, and confirm that both an
old signature and an attacker-controlled signer fail before accounting or
terminal authorization. Mechanism-evidence swaps, resource-field tampering,
and lifecycle-resume injection likewise fail closed.

## 9. Threats this protocol must test

- **Benchmark overfitting:** gains disappear outside exploration tasks.
- **Reward hacking:** the score rises while task semantics or constraints fail.
- **Evaluator leakage:** the candidate acquires answers, hidden tests, or grader
  internals.
- **Selection bias:** many noisy candidates are tried and only the lucky maximum
  is reported.
- **Non-deterministic attribution:** model, tools, or environment change together
  with the harness.
- **Average-score masking:** improvements in common cases hide severe slice or
  critical-case regressions.
- **Harness bloat:** score gains rely on unacceptable context, latency, or cost.
- **Self-confirmation:** the same model proposes, interprets, and approves its
  own change without an external measurable check.

## 10. Audit rule

No score without a run record; no comparison without matched fingerprints; no
promotion unless the versioned decision can be deterministically recomputed
from producer-attested paired observations, its frozen gate config, and
separately producer-attested mechanism evidence; no final claim without a test
set that did not participate in search. The current typed sealed report proves
branch consistency, not the identity of an external sealed evaluator;
production completion must add an authenticated evaluator producer.

M0.2 validates the promotion protocol against deterministic synthetic tasks.
M1 now supplies a pinned real GSM8K adapter and provider-neutral score-free
runner with replay backend, plus a bounded prompt-only automatic-search kernel.
The declarative skill slice reaches admission, materialized backend requests,
receipt replay, matched-control preregistration, and one controller-authorized
process-local revert-then-placebo execution with distinct process-local ledgers and
preflights. While the original capability and live ledger writers remain
available, its closure re-derives the exact process-local receipts, accounting,
tasks, and request construction. The current tasks carry no gold labels and the
run captures no claim outcomes, so formal scored effects cannot be recovered
post hoc from these outputs. It does not establish provider delivery,
activation, adherence, behavior, control effect, benefit, or generalization,
and it cannot authorize promotion.
Dedicated skill evidence production, automatic skill search, a live-model
study, and a verified skill gain remain absent. The controlled fixture remains
evidence about verifier plumbing and fail-closed behavior, not evidence of
agent capability improvement.
