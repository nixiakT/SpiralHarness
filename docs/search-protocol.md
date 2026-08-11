# Trusted automatic search protocol

## Scope and claim boundary

This protocol defines the first bounded, training-free automatic search loop in
SpiralHarness. It covers prompt replacement only and compares exactly four
conditions over independent search-run seeds:

1. `static`;
2. `random-valid`;
3. `prompt-only`;
4. `evidence-targeted`.

The implementation is a replayable control-loop kernel. Its deterministic
replay fixture exercises all four conditions and multiple search seeds, but it
does not call a live model and does not produce a reportable benchmark score.
The protocol must be connected to the pinned benchmark adapter, a fixed live
model provider, and a preregistered study before an empirical improvement claim
is valid.

## Frozen study boundary

Before any strategy is invoked, the trusted study plane persists:

- the exact `BaselineStudyPlan` and its four condition profiles;
- a common analysis plan;
- every `(baseline kind, search-run seed)` manifest;
- one experiment, search policy, stopping policy, and strategy implementation
  binding for each run;
- the seed harness, prompt mutation grammar, benchmark split fingerprints,
  model/inference/runtime fingerprints, repeat seeds, and resource ceilings.

The study controller derives the required run grid from the baseline plan. A
caller cannot replace it with a smaller roster. In the minimum configuration,
four conditions and two independent search seeds create eight required run
closures. A missing, duplicate, foreign, or drifted run invalidates the barrier
instead of being silently excluded.

Runs of the same condition must reuse the same policy and strategy plugin.
Comparable non-static runs must also retain the same search limits. Each cell
uses distinct live `SearchController` and `ExperimentController` instances so
one controller capability cannot satisfy multiple replications.

## One search run

One run follows this state machine:

```text
freeze -> start -> open round -> bind exploration feedback
       -> diagnose (evidence-targeted only)
       -> propose or decline
       -> materialize and screen every proposal
       -> nominate at most one eligible candidate
       -> authenticate the experiment terminal outcome
       -> disclose aggregate-only gate feedback
       -> repeat until a frozen stopping rule fires
       -> close search selection and experiment selection
```

The journal is content-addressed, append-only, sequence checked, and HMAC
authenticated by a process-scoped controller capability. Replay verifies every
link, run coordinate, round stage, champion transition, nomination, terminal
authorization, aggregate disclosure, and selection closure. A stale tail,
stage skip, second nomination in one round, or event from another run fails
closed.

The controller derives stopping from persisted events. The initial policy can
stop on maximum rounds, maximum gate nominations, patience since promotion, or
consecutive declined/empty rounds. Static closes without opening a search
round. A non-static controller with any completed round cannot be resumed by a
fresh automatic-loop invocation because the process-local optimizer call
capability and counters cannot yet be re-authenticated across process restart.

## Strategy information boundary

The optimizer receives a typed `StrategyFeedbackView`, never repository access.
A per-invocation artifact view has no list operation and permits reads only from
an exact allowlist assembled for the current run, round, champion, and feedback
closure. Prompt and hypothesis artifacts must be written through that same
view. The controller records every permitted read and write and rejects a
proposal that cites an artifact it did not read or an output it did not write.

Benchmark metadata in that view is a typed, redacted artifact with no protocol
split references. The frozen benchmark binding separately names that safe
artifact, four role-specific exploration artifacts with distinct media types,
and the exact diagnostic root and closure. Feedback is HMAC-authenticated over
the run, experiment, benchmark binding, exploration split, baseline, seed,
round, champion, prior controller disclosure, and view. The signer reloads the
binding and accepts only those exact refs; after verification, the loop derives
optimizer read grants from the binding rather than from caller-supplied refs.
The complete binding and protocol split manifests are never granted to the
optimizer.

The four profiles are intentionally different treatments over the same initial
prompt-replacement grammar:

| Condition | Candidate source | Search feedback |
| --- | --- | --- |
| `static` | no candidates | benchmark metadata only |
| `random-valid` | frozen finite catalogue | metadata, exploration inputs, aggregate gate feedback |
| `prompt-only` | optimizer prompt proposals | ordinary exploration aggregates, item feedback, trajectories, and aggregate gate feedback |
| `evidence-targeted` | diagnosis-linked optimizer proposals | prompt-only feedback plus typed diagnostic evidence |

`random-valid` samples uniformly without replacement from the currently
eligible entries in its frozen finite catalogue. The sample seed is derived
from the preregistered strategy seed and round. Entries already selected in an
earlier round are excluded. This is a finite-catalogue claim, not uniformity
over all possible prompt strings.

For optimizer arms, diagnoses and proposals are treated as untrusted data. The
controller strict-loads their typed content and rejoins it to the current
feedback, round, champion prompt, diagnostic cluster, cited evidence, mutation
policy, and scoped access log. Proposer confidence is not accepted as a ranking
input. Eligible screens are ordered deterministically by trusted exploration
statistics and resource ratios, followed by stable artifact hashes.

## Receipt-backed exploration screen

Candidate screening does not trust a runtime-returned score, resource total,
success flag, or receipt summary. Before execution, the controller freezes the
complete evaluation-cell schedule and performs an all-or-nothing attempt/token
preflight against the exact attempt-ledger boundary. Parent and candidate use
matched task/repeat/retry coordinates and a domain-separated seed derivation.

Each scheduled attempt reserves budget before the backend call and atomically
persists the execution and outcome references. Failures with uncertain usage
burn the reservation. Known token overruns poison the ledger. The trusted
screen then replays the complete receipt set against:

- the frozen schedule and preflight certificate;
- parent/candidate harness bindings;
- task, phase, repeat, side, retry, and derived-seed coordinates;
- reservation, outcome, execution, and ledger ancestry;
- exact paired retry coverage and the current final ledger tail.

Missing cells, asymmetric retries, side swaps, duplicate reservations, foreign
attempts, stale preflights, poisoned ledgers, or unreceipted outcomes after the
preflight boundary reject the screen. Usage is summed from replayed ledger
outcomes, not caller-authored totals. An independent process-scoped grader
service HMAC-attests the objective aggregate; the general search runtime cannot
verify its own score. The exact verification capability reloads that envelope
and binds the same proposal, candidate, harnesses, benchmark/grader identity,
schedule, and receipt set before metrics can enter deterministic screen
ranking. All proposals are archived with an eligible or rejected screen;
partial screen batches cannot proceed to nomination.

## Gate feedback and champion transition

The nominated candidate is evaluated through its paired
`ExperimentController`. The search controller accepts only an authenticated
terminal authorization for the exact experiment, candidate, current parent,
and prior champion. It reloads the referenced trusted gate decision and derives
a fixed aggregate disclosure containing:

- the authorized terminal result;
- the real gate implementation version and gate-configuration digest;
- the complete canonical eight check outcomes: integrity, sample size, policy,
  mechanism, primary effect, protected slices, task regressions, and resources;
- aggregate pair/task counts, confidence interval, regression rate, and
  resource ratios when available.

No gate task identifiers, task bodies, per-item scores, outputs, trajectories,
slice names, free-form verifier reasons, or arbitrary gate artifact references
fit in this disclosure schema. Promotion changes the run champion only when the
experiment controller authorizes that exact terminal transition. A locally
passing candidate that was superseded by another champion is not presented to
the strategy as a successful promotion.

## Study barrier and sealed access

Every run must close both its search selection and experiment selection on the
same champion, candidate lineage, analysis plan, and usage tail. Registration
replays both live controllers. After all required cells are registered, the
study controller writes a barrier closure and re-verifies every live run at its
exact tail before emitting the study-level sealed authorization.

This is a logical, in-process capability barrier. It detects if an experiment
was moved into sealed execution before study authorization, but it cannot
physically stop other code in the same interpreter from retaining and calling
an older `ExperimentController` reference. Deployment must place sealed data
and authorization in a separately isolated trusted service.

## Current limitations

- The current binding freezes one shared set of exploration role artifacts and
  a diagnostic closure for the one-round non-reportable replay fixture. A
  reportable multi-round producer must instead attest per-run/round/champion
  feedback and join its evaluation schedule and receipt lineage. That dynamic
  provenance is deliberately not claimed by this kernel.
- The replay adapter still holds the feedback signer in-process. Exact positive
  authorization prevents it from signing copied, wrapped, or nested gate
  artifacts outside the frozen binding, but deployment must isolate the signer
  from the general runtime capability plane.
- HMAC signers and verification capabilities are ephemeral and process-local;
  losing their keys prevents later cryptographic replay.
- Objective HMACs authenticate the preregistered grader authority and exact
  source bindings; they do not independently re-execute model outputs to derive
  the score inside the verifier.
- Content-addressed tails provide in-process single-writer compare-and-check,
  not durable cross-process compare-and-swap, leases, or transactions.
- Artifact views and capability manifests are API contracts, not an OS sandbox;
  there is no enforced filesystem, network, subprocess, or credential
  isolation.
- The bundled fixed runner uses a deterministic replay backend. There is no
  credentialed live-model provider or hard cancellation wrapper yet.
- Optimizer invocation counts are controller-derived, but an opaque plugin's
  internal model calls, tokens, and cost are not independently attested in this
  kernel.
- The multi-condition replay fixture is explicitly non-reportable. Sealed-test
  scores may be reported only after the complete preregistered study barrier,
  live fixed-model execution, and external experiment packaging are in place.
