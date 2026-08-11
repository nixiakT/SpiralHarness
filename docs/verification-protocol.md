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

Together, the split loader, grader, sandbox/permissions, resource accounting,
statistics, gate, artifact ledger, and sealed-set controls form a trusted
verification plane. Its version is frozen for an experiment and is outside the
candidate's mutable surface. Changing it starts a new protocol lineage.

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

## 4. Matched evaluation

Evaluate the candidate and its exact parent on the same task instances, seeds,
model settings, tool state, and resource limits. Randomize or interleave run
order when environment drift is possible. Cache only when the complete execution
fingerprint matches.

For stochastic agents, use repeated runs or a predeclared seed set. Preserve
per-task paired outcomes; aggregate means alone erase the evidence needed for a
reliable comparison.

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

Before trusting an evolver, test the verifier with adversarial fixtures: no-op
patches, lucky random candidates, hard-coded search answers, hidden-file reads,
grader tampering, forged score output, cross-task answer caches, budget overruns,
and regressions confined to a small protected slice.

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
promotion without a versioned decision artifact; no final claim without a test
set that did not participate in search.
