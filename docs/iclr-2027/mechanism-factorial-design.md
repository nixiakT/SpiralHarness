# Mechanism evidence as guidance and verification

Status: design amendment draft; not preregistered and not evidence.

## Why the primary contrast needs decomposition

Mechanism evidence can affect a self-evolving harness in two different places:

1. **search guidance**: source-linked activation, adherence, behavior, revert,
   and placebo evidence can help the optimizer choose its next mutation;
2. **promotion verification**: the independent authority can require that same
   evidence before changing the champion.

A single full-system versus score-only contrast measures the deployable policy
difference, but it cannot identify which role of mechanism evidence caused the
change.  The proposed confirmatory design therefore retains that end-to-end
contrast and would preregister a 2-by-2 decomposition on a powered
controlled-fault subset before any relevant outcome is visible.

## Factorial conditions

| Condition | Optimizer feedback | Promotion rule |
|---|---|---|
| `SS` | binary-success score, interval, usage, performance decision | utility + protected behavior + cost |
| `MS` | `SS` plus source-bound mechanism packet | same performance-only rule as `SS` |
| `SM` | same score-only view as `SS` | performance rule plus activation, adherence, intended behavior, revert, and placebo |
| `MM` | mechanism-visible view | full mechanism gate; this is the complete MGV-Harness policy |

`MM - SS` is the end-to-end protocol effect.  `MS - SS` estimates the
conditional guidance simple effect when promotion is score-based.  `SM - SS`
estimates the conditional promotion-policy simple effect when detailed
mechanism evidence is withheld from the optimizer.  The
difference-in-differences

```text
(MM - SM) - (MS - SS)
```

measures interaction rather than being folded into either simple effect.  These
are not marginal factorial main effects unless the final analysis explicitly
defines and estimates the corresponding averages over the other factor.

The `SM` optimizer inevitably observes whether its champion changed.  That bit
is part of the verification-policy treatment and must not be described as a
perfectly blinded information-only comparison.  Detailed gate checks, item
outputs, hidden task identities, and mechanism packets remain unavailable.

## Call-topology and budget equality

All four cells must execute the same eventual frozen topology:

```text
observe -> diagnose -> propose -> materialize -> screen -> nominate
        -> child/parent/revert/placebo execution -> evidence production -> gate
```

The authority computes both the performance projection and mechanism checks in
every cell.  A condition controls only which typed projection reaches the
optimizer and which frozen rule may authorize promotion.  Withholding evidence
must never skip model calls, control arms, retries, ledger writes, or resource
charges.  Within each model-task-search-seed block, the four cells share:

- exact solver, optimizer and role model specifications;
- seed harness, mutation grammar and editable component kinds;
- proposer random stream and proposal/nomination ceilings;
- the rule generating task, group, rollout and randomized arm-order coordinates;
- the child, parent, exact-revert and matched-placebo *execution topology*;
- attempt, token, cost, wall-time and failure ceilings;
- stopping rules and sealed authorization barrier.

For the same-model claim, “exact solver, optimizer and role model
specifications” means one provider-resolved model identity and revision across
all model-mediated roles.  Role-specific decoding may differ only when frozen
before outcomes and identical across cells.  A shared family label, routing
alias, or nominal model string is not sufficient evidence.

After feedback or promotion first differs, the conditions generally have
different champions and therefore different candidate contents and model
outputs.  Those differences are consequences of the randomized treatment, not
matching failures.  The artifact must never claim that post-divergence
candidates or executions are identical.  It instead verifies equal ex-ante
coordinate rules and ceilings, and records every realized call and failure for
intention-to-treat analysis.  If a result claims equal *realized* token cost in
addition to equal ceilings, it needs a prospectively specified model-tokenizer
padding or reserve-and-charge rule; equal call counts alone do not establish
that claim.

The runtime study barrier must emit a matched-contrast report that replays the
model requests, receipts, retries, usage and candidate counts.  A configuration
profile or equal seed helper is not execution evidence.

The score-only projection needs the same standard.  Confidence level, both
interval endpoints, estimator version, analysis-plan identity, and the scored
observation closure must come from one independently authenticated objective;
the projection caller cannot supply an endpoint.  Receipt replay must exact-load
the content-addressed schedule and preflight, bind every cell seed and complete
task/repeat roster, and close against the live ledger's frozen start and final
tail.  Signing caller-declared interval or usage values does not establish their
provenance.

## Adaptive gate validity

The gate manifest distinguishes exact predicates from stochastic claims.
Activation, policy, provenance, leakage, roster, and accounting checks are
source-bound deterministic predicates and fail closed.  For every stochastic
adherence, intended-behavior, utility, protected-behavior, or cost link it
freezes: the observation unit; child--parent and required child--exact-revert/
matched-placebo contrasts; null; operationally justified superiority or non-
inferiority margin; missingness mapping; test; and assigned `(q,a,j)` alpha.
Promotion requires every declared contrast, not a selected favorable member.
A model-judged link must propagate the preregistered class-conditional
calibration-error bound into its worst-case interval; failed calibration or an
unsupported semantic-equivalence class yields `inconclusive`.

Alpha spending does not make a p-value valid when an optimizer repeatedly
adapts to aggregates from the same held-out items.  Before search, the gate
authority must partition source/template/generator families into `Q` disjoint,
unopened query blocks and commit to their order.  Nomination `q` consumes block
`q` exactly once as one matched-condition batch.  Every condition's nomination
at index `q` is frozen before any member of the batch is evaluated.  The
authority then evaluates the batch atomically and releases only the authorized
condition-specific projections; no condition can see another condition's
outcome.  Later nominations cannot reuse the block's tasks, targets, model
outputs, grader state or derived sufficient statistics.  A terminal or missing
nomination is recorded prospectively rather than replaced with a later,
outcome-selected candidate.

Conditional on the search history, a fresh block must remain independent of
the nominated candidate under grouping assumptions frozen before execution.
Only then may the campaign apply its frozen alpha allocation across nominations.
Rollout seeds do not make a reused task family fresh.  If a benchmark cannot
supply enough independent groups, the study must reduce `Q`, use a separately
proved reusable-holdout mechanism, or label the promotion statistics
exploratory.  A confidence sequence for a fixed data stream cannot be cited as
generic permission to adapt candidates against a reused leaderboard.

The sealed partition remains separate: it is never consumed by promotion and
is released once only after every search cell closes.

### Conditional campaign guarantee

Let `F_(q-1)` contain the joint, feedback-isolated history before nomination
batch `q`, so every condition's candidate and allocated thresholds are
predictable.  For every condition `a` and gate dimension `j` whose null is true,
require its fresh-block p-value to be conditionally super-uniform:

```text
P(p[q,a,j] <= x | F[q-1]) <= x  for every x in [0,1].
```

If a promotion rejects every active null required by the intersection gate and
the predictable thresholds satisfy
`sum(q,a,j) alpha[q,a,j] <= alpha = 0.05`, then the probability of at least one
**statistical false authorization**---a promotion for which at least one
prespecified fresh-block gate null is true---is at most `alpha` by conditional
super-uniformity and a union bound over the first rejected true null.  With
multiple matched conditions the sum and union range over condition
as well as nomination and gate dimension; pairing does not require independence
between conditions.  This guarantee does not hold merely because threshold
labels sum to `alpha`; conditional validity is an assumption stronger than
unique block identifiers.  Family/source disjointness, unopened targets,
candidate freezing, isolated feedback, grader integrity and complete query
accounting must be attested, and the statistical model must justify why they
imply the stated conditional super-uniformity.

This is deliberately narrower than the study's oracle false-promotion event.
It does not control harm visible only under sealed distribution shift, grader or
policy exploits outside the tested nulls, provenance violations, or judge/model
misspecification.  Those remain separate harmful or protocol-invalid promotion
outcomes in the family-level operating-characteristic study.

Accordingly, `alpha/Q` can only abbreviate the total envelope for one nomination
block.  It does not authorize every condition and gate dimension to spend
`alpha/Q` separately; the full `(q,a,j)` sum must remain at most `0.05`.

## Factorial estimands and replication

One primary fault block contains an independently generated family, one
independently sampled primary optimizer seed, and one complete end-to-end run
per condition.  The four condition runs share that seed and schedule.  Family
is the independent cluster and conditions are paired within it; neither the
four cells nor extra seed repeats are independent replicates.  Rollouts, control
arms and scenarios inside a search are measurements.  A secondary multi-seed
analysis must reduce each family to one prospectively fixed mean or any-over-
seeds event and use family-clustered inference.  Model-by-family cells receive
the preregistered fixed-roster and stratum weights.

Because later candidates are treatment-induced, the factorial estimates policy
effects over adaptive runs.  A separately corrected, secondary operating-
characteristic analysis applies both promotion rules in shadow mode to a
prospectively common candidate slate, without returning shadow decisions to any
optimizer.  This same-candidate analysis can measure decision disagreement and
false promotion, but it does not replace the end-to-end factorial or receive
promotion authority.

For an eventual frozen outcome such as verified repair or false promotion, fit
the prospectively frozen factorial contrast

```text
outcome ~ mechanism_feedback + mechanism_gate
          + mechanism_feedback:mechanism_gate
```

with search-run and fault-family clustering appropriate to the outcome.  Under
treatment coding, report the two conditional simple effects and interaction
with simultaneous intervals; if marginal main effects are desired, define them
separately before outcomes.  Family-level false-promotion control/reduction and
the real-task `MM - SS` and `MM - STATIC` endpoints form the two co-primary
families of the conjunctive credible-commit decision.  Component effects remain
a separately corrected secondary family unless a preregistration replaces an
endpoint before any relevant outcome is visible.  Scenario rollouts must never
be counted as independent `n` for a factorial claim.

## Information-volume placebo

On the controlled-fault subset, add `M~M`: its optimizer receives the same
packet schema, byte/token envelope and call budget as `MM`, but an explicitly
authority-labeled placebo projection permutes evidence within a frozen model,
surface, fault-family, severity and round stratum.  The proposer sees the same
common `mechanism-visible` view label in `M~M` and `MM`; placebo assignment and
the authority's source mapping are blinded, and deterministic padding preserves
the frozen envelope.  The independent promotion authority continues to consume
the genuine source-linked packet.  The placebo projection cannot enter the
mechanism gate.  Its permutation is authority-owned and fixed before outcomes.

This arm tests whether gains require correctly linked mechanism evidence rather
than merely more tokens, additional fields, or a generic signal that a failure
occurred.  Cross-stratum shuffling is forbidden because it can change marginal
difficulty and create an easier placebo.

## Controlled-fault component ablations

The following ablations are secondary and run on a prospectively frozen
controlled-fault roster, not selected examples.  They use paired family blocks
where scientifically appropriate and a separately declared multiplicity
family; calling them powered requires their own design calculation.  If the
same family contributes to several contrasts, the analysis must retain that
dependence rather than count it as new independent evidence:

- omit request/path activation evidence;
- omit adherence evidence;
- omit exact revert;
- omit matched placebo;
- omit protected hard negatives;
- replace campaign-level error spending with per-query nominal testing;
- allow a candidate-authored verifier instead of the independent authority;
- remove source/provenance binding while preserving packet contents.

Each ablation reports verified repair, accepted-edit precision, family-level
false promotion, harmful promotion, inconclusive rate, calls/tokens per verified
repair, and protected/cost violations.  It is not sufficient to show only final
task score.

## Real-task and fault-suite roles

Real tasks establish external utility; controlled faults establish operating
characteristics under known causal truth.  Neither substitutes for the other.

The real-task roster should cover at least four behaviors with one untouched
search/gate/sealed grouping rule per task family:

- mathematical generation;
- heterogeneous reasoning, macro-averaged by subtask;
- high-label classification with source/group separation;
- strict structured generation, coding, or tool use with source/time
  separation.

The current v4 executable has seven generated families across six declared
surfaces and 28 role scenarios per partition.  Those scenarios are not 28
independent families, and deterministic benchmark middleware rather than
base-model output drives behavior.  It is a multi-surface trust-closure fixture,
not live-model, optimizer-capability, self-evolution, or powered-benchmark
evidence.  The prospective HarnessFaultBench must contain enough repair rules
and families that no constant mutation is near the oracle.  It needs repairable,
no-fault,
unrepairable, correlated-distractor, protected-hard-negative and interacting
fault families under a frozen sampling target.
Because an author-built generator can favor the proposed gate, the powered
study must add a blinded naturally occurring fault subset, external adversarial
audit, semantic-equivalence classes for alternative valid repairs, blinded
adjudication of novel legal repairs, placebo behavioral-neutrality calibration,
and source/template clustering.  The optimizing model's uncalibrated preference
cannot validate its own mechanism claim.

The separate four-context development runner is a stronger plumbing check, not
the missing experiment.  In one process it uses a paired exploration block for
all four condition-local candidates, persists a joint freeze, and then runs
four condition-specific sets of 28 child/parent/revert/placebo gate quartets on
a source/group-disjoint fixture GATE block before automatic gate dispatch.  Its
default topology is `4 * (28 fit-parent + 1 proposer + 4 * 28 gate) = 116 + 448
= 564` calls.  It dispatches contexts in fixed `SS -> MS -> SM -> MM` order;
prospective counterbalancing is not implemented.  Offline code-order replay does not independently
attest freeze/gate time, one-time grant consumption, cross-process atomicity,
provider-session isolation, or an exact served revision.

## Self-evolution closure

A result is labeled **same-model self-evolution** only if one exact frozen,
provider-resolved model identity and revision closes execution, diagnosis,
proposal, materialization, model-mediated candidate ranking, and nomination with
no manual candidate choice.  Final statistical promotion belongs to the fixed
independent authority, not the model.  A family name, routing alias, or nominal
model string is insufficient.  Every role call, failed call, retry, token and
cost is charged.  Pure-model, static, random-valid and external optimizer
baselines remain separate:

- `PURE` uses only the task payload and provider-minimal wrapper, with no seed
  harness surfaces, under the same exact solver configuration and coordinates;
- `PURE@B` spends the same frozen ex-ante model-call/token ceiling as real-task
  `FULL` on
  independent PURE samples under a nonadaptive, task-native aggregation rule;
- `STATIC` measures the unmodified harness using the same solver model;
- `RANDOM` measures the finite mutation grammar without optimizer intelligence;
- `SS` is the matched score-only self-evolution control;
- `MM` is the complete MGV-Harness policy;
- official external systems are secondary and are never conflated with a
  paper-algorithm reconstruction; a stronger external proposer is a
  heterogeneous baseline, not a same-model or `>10 pp` comparison.

## Claim rule

The factorial design does not license a result until all cells close and the
one-time sealed authority releases the frozen aggregate.  Development tuning
may continue on exploration and pilot lineages, but failure to exceed a desired
margin on sealed data cannot trigger more tuning on that lineage.  A claim of
more than ten percentage points requires every applicable, preregistered
claim-family simultaneous multiplicity-adjusted one-sided lower confidence
bound on the prospectively binary success-probability scale to be strictly
greater than `0.10`; a favorable point estimate or arbitrary normalized-score
difference alone is insufficient.
