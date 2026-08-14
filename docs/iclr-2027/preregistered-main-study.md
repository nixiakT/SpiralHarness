# Preregistered main-study protocol (draft before pilot)

Status: `pilot-pending`.  This document freezes the scientific structure.  All
quantities explicitly delegated to a disjoint pilot must be filled, reviewed,
and committed in a new immutable preregistration before confirmatory search
starts.  Empty values or `TBD` are invalid in a `frozen` manifest.

## 1. Estimands and primary family

Map each official task score to `[0,1]`.  Give every preregistered
model-by-task cell equal weight, regardless of its number of examples.  Let
`FULL` denote evidence-targeted mechanism-gated search, `SCORE` its
score-only-matched control, and `STATIC` the unmodified seed harness.

For three models and four real task families:

\[
\Delta_{causal}=\frac{1}{12}\sum_{m,t}
 [\mu(FULL,m,t)-\mu(SCORE,m,t)]
\]

\[
\Delta_{e2e}=\frac{1}{12}\sum_{m,t}
 [\mu(FULL,m,t)-\mu(STATIC,m,t)]
\]

On HarnessFaultBench:

\[
\Delta_{repair}=P(verified\ repair\mid FULL)
-P(verified\ repair\mid SCORE).
\]

The three one-sided primary hypotheses are:

- H1: `FULL > SCORE`, with a minimum scientifically meaningful effect (SESOI)
  of `+0.02` absolute normalized score.
- H2: `FULL > STATIC`, SESOI `+0.02`.
- H3: FULL improves verified repair probability over SCORE, SESOI `+0.15`.

Use one-sided Holm--Bonferroni across H1--H3 with family-wise `alpha=0.05`.
The headline conclusion requires all three tests to reject and all point
estimates to exceed their SESOI.  The project preference for a `>10` percentage
point improvement is reported only when its preregistered confidence bound
supports that stronger margin.  It is not a rule for reopening sealed data.

## 2. Conditions and treatment contrast

The real-task grid contains five conditions:

1. `static`;
2. `random-valid`;
3. `prompt-only`;
4. `score-only-matched` (`SCORE`);
5. `evidence-targeted-full` (`FULL`).

SCORE and FULL must match exactly on seed harness, proposer model and sampling,
mutation grammar, editable surfaces, proposals per round, solver model,
task/rollout coordinates, search rounds, nomination policy, gate-query count,
and resource ceilings.  Both receive the same aggregate score, interval, cost,
and decision.  SCORE cannot read item outputs, trajectories, localization,
activation, adherence, behavior, or mechanism packets.  FULL can read only
typed, source-linked evidence permitted by the frozen protocol.

`prompt-only` is not the score-only control if it can read item feedback or
trajectories.  HarnessFaultBench adds a mechanism-only
`shuffled-evidence-placebo` whose packet formatting and budget match FULL while
evidence is permuted within a preregistered factor stratum.

## 3. Model and task roster

Select three non-OpenAI/non-Anthropic model configurations, spanning at least
two model families, using availability-only smoke tests.  Freeze provider
route, served revision/fingerprint, inference settings, tokenizer accounting,
timeout, tools, and model role policy before any score-bearing pilot.  Do not
choose models by their benchmark results.

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
replicates.  With `S` search seeds, the five-condition real-task grid contains
`3 * 4 * 5 * S = 60S` complete searches.

For every `(model, task, search_seed)` retain matched item/group outcomes and
compute FULL-minus-control differences.  Template, repository, source, or
generator family is the item-side resampling unit.  All preregistered cells,
including failures, remain in the intention-to-treat analysis.

## 5. HarnessFaultBench

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
bound below 5%.  Search-seed repeats do not create new fault families.

## 6. Analysis

For cell-level search difference `d_mts`, fit the preregistered robust
hierarchical model:

\[
d_{mts}\sim t_\nu(\theta_{mt},
\sqrt{\sigma^2_{search,mt}+v_{mts}}),\quad
\theta_{mt}=\mu+a_m+b_t+u_{mt},
\]

where `v_mts` comes from a task-group-aware paired bootstrap.  Models and tasks
are a fixed roster, so conclusions generalize to that roster rather than an
imaginary population of all models and tasks.

Use at least 20,000 fixed-seed nested bootstrap draws: resample search seeds
within model-by-task cells, then task groups within runs, and refit the same
model.  If it fails to converge, use the preregistered fallback: an equal-weight
12-cell paired macro estimator with a studentized cluster bootstrap.  Do not
switch analysis after viewing which method is more significant.  Correct
individual model, task, and factor-slice claims with Holm; otherwise label them
exploratory.

## 7. Pilot and power simulation

The pilot uses disjoint task groups and HarnessFaultBench generator seeds.  It
may estimate search variance, model-by-task heterogeneity, paired discordance,
task-family and rollout ICC, timeout/failure probability, and token/cost
distributions.  It cannot select the most favorable model or task.

For candidate `S in {8,12,16,20,24}`, simulate at least 20,000 complete null
datasets and 20,000 datasets at each SESOI, injecting the upper-confidence
failure rate and running the exact hierarchical/Holm pipeline.  Select the
smallest `S` for which global-null FWER is at most 0.05, worst-case probability
that all three primary hypotheses pass is at least 0.90, and each individual
test has power at least 0.90.  If `S=24` fails, do not launch confirmation;
increase independent tasks/faults or narrow the scientific question.

The pilot must also determine rollout repeats, repairable fault count, search
round/proposal/nomination ceilings, patience, retry limits, protected-slice
sample size, and token/wall-time/cost ceilings.  It cannot change the primary
contrasts, direction, SESOI definition, multiplicity, equal-cell weighting,
factor axes, failure policy, or one-time sealed release.

## 8. Search stopping and adaptive validity

Each condition stops at the first frozen limit reached: rounds, gate
nominations, proposals, optimizer calls, resource ceiling, consecutive rounds
without promotion, or consecutive invalid/declined candidates.  There is at
most one gate nomination per round.  Gate alpha is spent by a frozen rule such
as `alpha/Q`, or a preregistered online error-rate procedure; sealed score never
enters stopping.  With no promotion, the final champion is the seed harness.

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

Malformed output, refusal, candidate-caused timeout/tool failure, and crash are
method outcomes: score zero for the affected unit, cost retained, and the
search keeps its previous legal champion.  A provider outage reruns the whole
matched block with the next preregistered retry seed; asymmetric infrastructure
failure also reruns both arms and retains original cost.  Only the blinded
evaluator may symmetrically invalidate a grader-broken item and replace it from
the reserve roster.  Model endpoint/revision drift invalidates its full block.

Report intention-to-treat, complete-case sensitivity, failure-as-zero, and
worst-case bounds.  Provider-attributed missingness beyond the frozen threshold
makes the study inconclusive/invalid rather than authorizing a favorable
complete-case claim.

## 11. Amendments and invalidation

Every amendment records its reason, timestamp, old/new artifact hashes, whether
any outcomes were visible, and which claims remain confirmatory.  Once any
score-bearing main-study outcome exists, changing a primary endpoint, roster,
analysis, stopping rule, or sealed commitment creates a new study ID.
