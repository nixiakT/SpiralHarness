# Prospective main-study protocol (draft before pilot)

Status: `pilot-pending`.  This document records the proposed scientific
structure.  All quantities explicitly delegated to a disjoint pilot must be
filled, reviewed, and committed in a new immutable preregistration before
confirmatory search starts.  Empty values or `TBD` are invalid in a `frozen`
manifest.

The status may change to `frozen` only after one commit binds nonempty exact
model identities/revisions, task and source-group manifests, search seeds `S`,
fresh nomination blocks `Q`, rollout repeats, all budgets and stopping limits,
the powered factorial/fault-family roster, retry and failure policy, validated
analysis and coverage-audit code hashes, environment/container hashes,
grader/price-table hashes, and the sealed-authority public key.  The freeze
audit must verify that none of these values remains delegated, inferred from a
default, or selected from a score-bearing outcome.

## 1. Estimands and primary family

The common primary scale is a prospectively specified **binary success per
evaluation unit**, not an arbitrary min--max normalization.  Before freezing,
each task must bind its source-group roster or sampling target, group and unit
weights, direction, deterministic programmatic success rubric, failure and
missingness mapping, and any secondary official metric.  Give every frozen
model-by-task cell equal macro weight regardless of task size.  For arm `a`,
model `m`, and task `t`, `mu(a,m,t)` is the expected binary success under the
declared source-group target and the frozen search/rollout randomization.  Models
and tasks are fixed rosters; no claim extends to an unspecified population of
models or tasks.

Consequently, `0.10` means ten percentage points of success probability.  A
task that cannot support this binary rubric is excluded from the cross-task
`>10 pp` family and its bounded continuous score is analyzed separately in its
own declared unit.  Its arbitrary normalized difference must never be called
percentage points.  Let `FULL` denote evidence-targeted mechanism-gated search,
`SCORE` its score-only-matched control, `STATIC` the unmodified seed harness,
and `PURE` the task payload executed through only the provider-minimal wrapper.

For three models and four real task families:

\[
\Delta_{policy}=\frac{1}{12}\sum_{m,t}
 [\mu(FULL,m,t)-\mu(SCORE,m,t)]
\]

\[
\Delta_{e2e}=\frac{1}{12}\sum_{m,t}
 [\mu(FULL,m,t)-\mu(STATIC,m,t)]
\]

The secondary bare-model reference is:

\[
\Delta_{pure}=\frac{1}{12}\sum_{m,t}
 [\mu(FULL,m,t)-\mu(PURE,m,t)].
\]

On HarnessFaultBench:

\[
\Delta_{repair}=P(verified\ repair\mid FULL)
-P(verified\ repair\mid SCORE).
\]

`Delta_policy` is the joint `MM - SS` deployable-policy effect: FULL changes
both the optimizer's feedback and the rule that can promote a candidate.  It is
not an isolated effect of feedback visibility or of the mechanism gate.

The three one-sided primary margin hypotheses are:

- H01: `Delta_policy <= 0.02` versus H11: `Delta_policy > 0.02` absolute
  binary-success probability (two percentage points).
- H02: `Delta_e2e <= 0.02` versus H12: `Delta_e2e > 0.02`.
- H03: `Delta_repair <= 0.15` versus H13: `Delta_repair > 0.15`.

Use one-sided Holm--Bonferroni across H01--H03 with family-wise `alpha=0.05`.
The primary statistical pass requires all three Holm-corrected one-sided margin
tests to reject; a point estimate above its SESOI is insufficient.  A full
headline conclusion additionally requires every protected-behavior, cost,
provenance, and leakage constraint to close.  The project preference for a
`>10` percentage-point improvement is a separate claim family: every applicable
preregistered simultaneous multiplicity-adjusted one-sided lower confidence
bound must be strictly greater than `0.10`.  This includes a `Delta_pure`, named
competitor, model, or benchmark claim rather than borrowing multiplicity control
from H01--H03.  It is not a rule for reopening sealed data.

## 2. Conditions and treatment contrast

The real-task grid contains six primary conditions and one secondary
model-call/token-budget-matched reference:

1. `pure`;
2. `static`;
3. `random-valid`;
4. `prompt-only`;
5. `score-only-matched` (`SCORE`);
6. `evidence-targeted-full` (`FULL`);
7. `pure-model-budget-matched` (`PURE@B`).

`PURE` receives the official task payload plus only provider-required message
serialization.  It receives no seed system prompt, few-shot harness examples,
skill, memory/retrieval content, tool definition, middleware transformation, or
control-flow/routing logic.  For every model--task--search-seed coordinate it
uses the same exact provider-resolved model revision, role inference settings,
task/group/rollout seeds, output contract, grader, retry policy, and evaluation
resource ceiling as the other evaluation arms.  PURE performs no search, so its
zero optimizer usage is reported rather than padded; all actual solver,
provider-wrapper, retry, and grading usage is retained.  The executable PURE
contract and trusted local runner now have engineering tests, but no reportable
live-model roster, paired receipt closure, or sealed PURE evaluation has run.
Those experimental obligations must close before this condition is reportable.

The single-call PURE contrast is an effectiveness reference, not a
model-call/token-budget-matched comparison: FULL also spends model calls on
diagnosis, proposal, nomination, and verification.  PURE@B therefore receives
FULL's frozen ex-ante total model-call and token ceiling as independent
provider-minimal samples.  A task-native deterministic aggregation rule (for
example exact-answer majority) is frozen without grader, gate, or sealed
feedback.  Every task in a cross-task `Delta_pure@B` claim must bind a total rule
from the complete sample multiset to one binary-scored output; if any task lacks
one, that full-roster estimand and claim are undefined rather than silently
dropping the task.  Its repeated-sampling distribution may still be descriptive
but cannot substitute for `mu(PURE@B,m,t)`.  PURE@B performs no adaptive harness
edit.  Both FULL-minus-PURE and FULL-minus-PURE@B have separately powered,
multiplicity-controlled claim families if either is intended for a headline.

SCORE and FULL must match exactly on seed harness, proposer model and sampling,
mutation grammar, editable surfaces, proposals per round, solver model,
task/rollout coordinates, search rounds, nomination policy, gate-query count,
and resource ceilings.  Both receive the same aggregate score, interval, cost,
and frozen performance-only decision over utility, protected behavior, and
cost.  This is not a redacted full-gate decision: it excludes activation,
adherence, and intended-behavior checks.  SCORE cannot read item outputs,
trajectories, localization, activation, adherence, behavior, or mechanism
packets.  FULL can read only typed, source-linked evidence permitted by the
frozen protocol.

Every score projection must derive its confidence level, lower and upper
endpoint, estimator version, and analysis-plan identity from the same
independently authenticated objective artifact.  A projection request may
reference that artifact but may not supply either interval endpoint.  Receipt
closure must load a content-addressed schedule and preflight, replay the exact
live-ledger boundary and final tail, bind execution seeds to complete cell
coordinates, and cover the entire frozen task/repeat roster.  A signed wrapper
around caller-declared interval or accounting values is not provenance.

`prompt-only` is not the score-only control if it can read item feedback or
trajectories.  HarnessFaultBench adds a mechanism-only
`shuffled-evidence-placebo` whose packet formatting and budget match FULL while
evidence is permuted within a preregistered factor stratum.

### 2.1 Controlled-fault feedback-by-promotion factorial

The powered controlled-fault subset crosses optimizer feedback with promotion
authority:

| Cell | Optimizer feedback | Promotion authority |
|---|---|---|
| `SS` | score/interval/resources/performance decision | performance-only |
| `MS` | `SS` plus source-linked mechanism packet | performance-only |
| `SM` | same score-only view as `SS` | full mechanism gate |
| `MM` | mechanism-visible | full mechanism gate |

`MM - SS` is the joint deployable-policy effect.  `MS - SS` is the conditional
feedback simple effect when promotion is performance-only; `SM - SS` is the
conditional promotion-policy simple effect under score-only feedback;
`(MM - SM) - (MS - SS)` is their interaction.
These are intention-to-treat effects of adaptive policies.  After the first
different feedback or promotion, candidates and champions may diverge; the
factorial must not be described as candidate-level mediation.

For nomination index `q`, all four candidate/no-nomination states are frozen
before one fresh source/family block is opened.  Isolated condition contexts are
dispatched in a preregistered counterbalanced order.  A terminal condition
contributes an absorbing no-nomination outcome; it is not replaced by a later
candidate.  The runtime campaign closure must bind the block, all four
schedules, preflights, receipt streams, final ledger tails, failures, and the
one-time consumption record.  The literal `atomic=true` in a manifest is only a
plan, not evidence of atomic execution.

As a secondary gate operating-characteristic analysis, the authority applies
both frozen promotion rules in shadow mode to a prospectively common candidate
slate without feeding the shadow result back to search.  This separates
same-candidate gate disagreement from the history-mediated end-to-end factorial
effect.  It has its own multiplicity family and no promotion authority.

## 3. Model and task roster

Select three non-OpenAI/non-Anthropic model configurations, spanning at least
two model families, using availability-only smoke tests.  In a same-model run,
one exact provider-resolved model identity and revision must fill solver,
diagnosis, proposal, materialization-planning, model-mediated candidate ranking,
and nomination roles; no stronger teacher or human may choose a candidate.
Final statistical promotion is not a model role and remains a deterministic,
authority-owned application of the frozen rule.  Freeze provider route, served
revision/fingerprint, inference settings by role, tokenizer accounting,
timeout, tools, and model role policy before any score-bearing pilot, and keep
them identical across conditions.  If exact served identity cannot be attested,
or a stronger proposer is used, label the result harness optimization rather
than same-model self-evolution.  Do not choose models by benchmark results.

An external method with a stronger proposer, diagnoser, materializer,
model-mediated candidate ranker, or nominator is an explicitly heterogeneous
optimizer baseline.  It may be
useful descriptively, but it cannot support a same-model self-evolution claim or
enter the `>10 pp` same-model claim family.

The intended real-task roster covers four distinct behaviors:

- mathematical generation (for example GSM8K);
- multi-type reasoning (for example BBH, macro-averaged by subtask);
- high-label classification (a new source/group-separated task);
- strict structured generation or coding (a new temporal/source-separated
  task).

S2D, LawBench and USPTO test data have already been inspected.  They are
development data.  They may enter confirmation only through an independently
constructed, never-opened source/time/group-separated split; otherwise a new
benchmark must replace them.

## 4. Independent unit and replication

One complete end-to-end search run is the algorithmic repeated unit.  Task
rollouts within that search are paired observations, not independent search
replicates.  With `S` search seeds, the five adaptive/static conditions contain
`3 * 4 * 5 * S = 60S` complete condition closures.  PURE and PURE@B add `24S`
matched evaluation references but no optimizer trajectory, for `84S` total
condition evaluations.  PURE@B may contain many calls inside one reference and
must reserve its full ceiling before execution.

For every `(model, task, search_seed)` retain matched item/group outcomes and
compute FULL-minus-control differences.  Template, repository, source, or
generator family is the item-side resampling unit.  All preregistered cells,
including failures, remain in the intention-to-treat analysis.

## 5. HarnessFaultBench

The current v4 executable is a deterministic multi-surface trust-closure
fixture: seven generated fault families span six declared surfaces, and each
partition contains four scenario roles per family.  Thus each partition has 28
scenarios, but not 28 independent families.  Deterministic benchmark
middleware, not the charged base-model output, determines final behavior.  The
v4 artifact is not a live-model, LLM-solver, optimizer-capability,
self-evolution, or powered-benchmark result and does not instantiate the full
prospective design below.

Every scenario contains the oracle-correct harness, injected faulty harness,
patch hash, true causal surface/stage/trigger, exploration/gate/sealed groups,
protected clean behavior, admissible repair criterion, exact revert, matched
placebo, repairability label, generator seed, and all artifact hashes.

Predeclare a compatibility-masked balanced fractional-factorial design over:

- mutable surface: prompt, memory/retrieval, tool/schema/routing, output
  middleware, control flow/skill routing;
- causal stage: selection/activation, adherence/conflict,
  behavior/transformation, output/verification contract;
- prevalence: sparse or dense;
- topology: single fault or interacting/masked double fault;
- distractor: absent or correlated non-causal distractor;
- shift: IID gate or held-out template/source sealed;
- repairability: repairable, oracle-optimal no-fault, or out-of-policy.

Hash the generator and design matrix before execution.  `verified repair`
requires sealed target improvement, protected-slice non-inferiority,
candidate-versus-revert and candidate-versus-placebo attribution, activation,
adherence, and no leakage, policy, grader-exploit, or resource violation.

At least 59 independent null/unrepairable *scenario families* are required if
zero observed false promotions is to yield a one-sided 95% exact binomial upper
bound below 5%.  This interpretation additionally requires exchangeable draws
from a prospectively frozen family-sampling target; otherwise only the finite
roster rate is reportable.  Search-seed repeats and multiple roles within one
family do not create new fault families, and the current seven-family fixture
does not satisfy this target.

### 5.1 Measurement and judge validity

Every activation, adherence, intended-behavior, protected-behavior, and repair
criterion must bind a canonical observation schema, source locations, a
programmatic decision rule where feasible, and its inconclusive state.  An
outcome score cannot be reused as proof of an upstream mechanism link.

If a link cannot be checked programmatically, the frozen protocol must bind the
judge provider-resolved identity/revision, prompt and rubric bytes, decoding,
input projection, order randomization, blindness fields, retry rule, and
software hash.  The judge may not see arm names, candidate lineage, promotion
decisions, aggregate utility, or sealed targets beyond the minimum blinded
evidence required for that link.  Before main execution it must pass a disjoint,
arm-balanced human-labeled calibration set whose sampling frame, size,
adjudication process, agreement statistic, class-conditional error metrics,
confidence bounds, and minimum thresholds are fixed after the pilot but before
freeze.  Human disagreements and judge abstentions follow a frozen adjudication
or inconclusive rule; they are never resolved after viewing which arm benefits.
Failure of any calibration threshold makes the corresponding mechanism endpoint
inconclusive and blocks verified-repair or promotion claims that require it.
Judge calls and human adjudications are isolated from the optimizer and fully
accounted.

## 6. Analysis

The declared estimand is the equal-weight macro mean of the frozen
model-by-task cells.  The current method-development prototype computes that
macro estimator and a basic nested bootstrap: resample complete search seeds
within each model-by-task cell, task groups within the selected run, and rollout
repeats within the selected group.  Formal configurations require at least
20,000 fixed-seed draws, but a draw-count floor is not a coverage guarantee.

The prototype has not completed finite-sample coverage calibration, been
independently audited against the margin nulls, or consumed an
authority-attested sealed input artifact.  It is neither a frozen confirmatory
analysis nor a formal sample-size recommendation.  The final preregistration
must bind one validated estimator, resampling hierarchy, interval/test
construction, software and environment hash, and deterministic seed.  There is
no outcome-selected estimator or resampling fallback.  Models and tasks are a
fixed roster, so conclusions apply to that roster rather than an imaginary
population of all models and tasks.  Individual model, task, and factor-slice
claims receive their own frozen multiplicity family or remain exploratory.

## 7. Pilot and power simulation

The pilot uses disjoint task groups and HarnessFaultBench generator seeds.  It
may estimate search variance, model-by-task heterogeneity, paired discordance,
task-family and rollout ICC, timeout/failure probability, token/cost
distributions, and dependence among the three primary estimators.
`Delta_policy` and `Delta_e2e` share FULL, so independence is inadmissible.

The repository contains both a bounded analytic sensitivity proxy and a newer
prospective simulator that passes synthetic fixed-roster data through the basic
nested-bootstrap prototype.  They are method-development tools, not evidence of
an observed harness effect.  Their calibration/audit streams and Monte Carlo
intervals do not authenticate a caller-declared pilot artifact, establish pilot
disjointness or closure, validate sealed blinding, or prove frequentist coverage
under the final data-generating process.  A dataset-source enum or digest over
caller-declared syntax is not a sealed-authority attestation.

Formal `S` remains blocked until the disjoint pilot closes and a frozen simulator
uses the exact validated analysis.  It must predeclare candidate `S`, audit the
three SESOI boundary nulls on an independent stream, retain H1/H2 shared-run
dependence, conservatively cover H3 dependence and failures, and incorporate
Monte Carlo uncertainty into the selection rule.  The current prototype's
candidate selector targets the conjunction of all three Holm-corrected
one-sided SESOI-margin rejections **and** simultaneous one-sided lower bounds
strictly above `0.10` for H1 and H2.  Its code-level
`combined_sizing_event` label refers to that five-part event.  Protected
behavior, cost, provenance, and leakage are required for the full headline
conclusion but are not part of the prototype.  Nor does it size
`Delta_pure`, `Delta_pure@B`, external-comparator, per-model, or per-benchmark
`>10 pp` claims;
the final frozen simulator must enumerate and power every claim intended for a
headline.  No current output is a required-seed recommendation.

More specifically, H1 and H2 share the declared hierarchical copula, whereas
the prototype generates H3 independently and treats its paired-binary leaves as
IID within the fixed hierarchy; it simulates no timeout, failure, or missingness
process.  Its null stream covers only the joint SESOI-boundary configuration
labelled “global least favourable.”  It does not audit partial nulls or the
H1/H2 `0.10` claim boundaries, does not prove its global configuration is least
favourable, and does not use null-FWER diagnostics to select a candidate.

Candidate sets are strict prefixes ordered by cardinality.  The prototype takes
the first candidate whose alternative-calibration combined-event power has a
95% one-sided Wilson lower bound at least a caller-declared `target_power`, then
applies that criterion once on a domain-separated validation stream.  Validation
failure yields no recommendation and no larger fallback.  This is an explicit
diagnostic rule, not a frozen 0.90-power design.  The final design must bind the
target, roster, replicate counts/seeds and no-fallback rule, simulate defensible
H3 dependence/ICC/failures, and make independent global, partial-null, and
`0.10`-boundary audits a fail-closed prerequisite.

At a true effect exactly equal to its SESOI, a valid one-sided margin test
rejects only at its calibrated type-I rate.  A 0.90 combined-event power
design therefore needs a distinct, scientifically justified alternative
with H1/H2 effects strictly above `0.10` and the H3 effect strictly above
`0.15`, with pilot-justified separation sufficient for the target power and
frozen before outcomes.  Each eventual `>10 pp` family is powered and analyzed
through its simultaneous
multiplicity-adjusted one-sided lower bounds, never a point estimate or a
null-derived proxy interval.  The threshold cannot enter sample-size reopening
or optional stopping; for verified repair it is weaker than, and must never be
relabeled as, the 0.15 primary SESOI.  `Delta_repair` will use equal weight over
the eventual frozen model-by-repair-stratum cells.

The pilot must also determine rollout repeats, repairable fault count, search
round/proposal/nomination ceilings, patience, retry limits, protected-slice
sample size, and token/wall-time/cost ceilings.  It cannot change the primary
contrasts, direction, SESOI definition, multiplicity, equal-cell weighting,
factor axes, failure policy, or one-time sealed release.

## 8. Search stopping and adaptive validity

Each condition stops at the first eventual frozen limit reached: rounds, gate
nominations, proposals, optimizer calls, resource ceiling, consecutive rounds
without promotion, or consecutive invalid/declined candidates.  There is at
most one gate nomination per round.  For thresholds `alpha[q,a,j]` indexed by
nomination block, condition, and gate dimension, the eventual frozen rule must
satisfy `sum(q,a,j) alpha[q,a,j] <= 0.05`.  A shorthand `alpha/Q` may denote only
one block's total envelope; it must still be validly divided or combined across
conditions and dimensions.  Sealed score never enters stopping.  With no
promotion, the final champion is the seed harness.

The full grid must reach selection closure before unsealing.  There is no
study-level efficacy/futility stop.  After sealed failure, further optimization
requires a new lineage, preregistration, and sealed set.

## 9. Sealed authority

The confirmatory evaluator must be process- and authority-separated from
candidate/search code.  Before search it publishes a split commitment,
task-grouping algorithm hash, grader/container digest, preregistration commit,
service image digest, and asymmetric public key.  The private key, targets, and
grader secrets never enter the candidate interpreter or researcher workspace.

Candidate execution uses no network or secrets, a read-only filesystem, and
resource limits.  The evaluator accepts only champion hashes authorized by the
complete study barrier.  One authorization triggers the entire grid; no scores
are released until all cells settle.  It publishes a signed aggregate report
and escrows item-level artifacts.  Any exposure of a sealed ID, target, score,
or trajectory to a proposer/author invalidates that lineage.

Process-local HMACs and Python capability objects in the current kernel are
useful composition checks but do not satisfy this external-authority claim.

## 10. Resources, failures, and missingness

Account separately for solver, proposer, diagnoser, grader, retries, and failed
reservations: uncached/cached input tokens, output/reasoning tokens, actual
billed price, normalized frozen-price cost, wall time, CPU/GPU time, peak
memory, and tool calls.  Unknown failed-call usage burns its reservation.
Report absolute score/cost, score per million tokens, and the Pareto frontier;
do not hide cost or safety inside the optimized scalar.

Every protected slice and cost constraint needs an exact direction, estimand,
non-inferiority or maximum-cost margin, confidence/test construction,
missingness rule, and multiplicity family in the frozen preregistration.  These
values are currently delegated freeze blockers; no default, optimized weighted
sum, or post-outcome choice may fill them.

Malformed output, refusal, candidate-caused timeout/tool failure, and crash are
method outcomes: score zero for the affected unit, cost retained, and the
search keeps its previous legal champion.  Before freeze, every study component
binds one indivisible infrastructure-retry unit: the complete six-condition
real-task block (plus its PURE@B reference) or the complete four-cell factorial
nomination block.  A provider or harness-independent infrastructure failure
reruns that entire declared unit on the next reserve coordinate; it never reruns
only the favorable pair or the arm where failure occurred.  Original attempts,
failure labels, reservations, and cost remain in the ledger, while the frozen
analysis specifies whether an infrastructure-aborted unit is excluded from the
effect estimator or makes the cell invalid after the retry ceiling.  The freeze
also distinguishes a same-block transport retry with no model response from a
new reserve task/block; this choice cannot depend on partial outcomes.  Only the
blinded evaluator may invalidate a grader-broken item symmetrically and replace
the entire affected retry unit from the reserve roster.  Model endpoint or
revision drift invalidates the full model--task--seed block across conditions.

Report intention-to-treat, complete-case sensitivity, failure-as-zero, and
worst-case bounds.  Provider-attributed missingness beyond the frozen threshold
makes the study inconclusive/invalid rather than authorizing a favorable
complete-case claim.

## 11. Amendments and invalidation

Every amendment records its reason, timestamp, old/new artifact hashes, whether
any outcomes were visible, and which claims remain confirmatory.  Once any
score-bearing main-study outcome exists, changing a primary endpoint, roster,
analysis, stopping rule, or sealed commitment creates a new study ID.
