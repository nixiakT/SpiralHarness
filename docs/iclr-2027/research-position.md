# Research position: verification science for self-evolving harnesses

Status: claim-boundary freeze, 2026-08-14.  This is neither a completed
preregistration nor a result.

## Question

Given a fixed solver model, mutation space, and search budget, when is there
enough evidence to commit an adaptively selected harness edit credibly?

The working title is:

> **MGV-Harness: When to Commit Adaptive LLM Harness Edits**

The paper should not sell “an LLM edits its own prompt or harness” as the new
idea.  That capability is already established.  The proposed contribution is a
decision rule for when an adaptive edit can be committed: primary family-level
false-promotion control joined conjunctively to sealed utility superiority and
protected-behavior/cost non-inferiority.

## Main treatment

The main experiment compares two budget-matched adaptive policies:

- **SCORE:** sees only aggregate score, confidence bounds, resource totals,
  and a frozen performance-only decision over utility, protected behavior, and
  cost.  This projection excludes activation, adherence, and intended-behavior
  checks; it is not a redaction of the full mechanism gate.
- **FULL:** has the same model, candidate grammar, candidate count, task
  coordinates, scoring summaries, and budget, but additionally receives typed
  evidence linking activation, adherence, behavior, revert, and placebo
  interventions to the proposed edit.

The real-task `FULL - SCORE` contrast changes both information and promotion
authority.  It is the effect of the joint deployable policy, not an isolated
mechanism-feedback effect.  A powered controlled-fault `2 x 2` crosses
score/mechanism feedback with performance/mechanism promotion to identify the
two policy components and their interaction.

Two non-search evaluation references remain distinct.  `PURE` is the official
task payload through only the provider-minimal wrapper, with no seed prompt,
memory, skills, tools, middleware, or control-flow harness.  `STATIC` is the
unchanged seed harness.  Both use the same exact solver configuration and task
coordinates as FULL; `FULL - PURE` is a secondary effectiveness contrast and
currently evidence-blocked.  It is not model-call/token-budget matched.
`PURE@B` separately spends FULL's frozen ex-ante model-call/token ceiling on
independent PURE samples under a nonadaptive task-native aggregation rule; a
cross-task effect exists only when every primary task freezes a total rule.  Its
prospective contract now commits every task/evaluation-unit/sample allocation,
distinct seed, retry allowance, and token cap and joins the split, grader,
query-DAG, and retry identities to FULL; no live execution is thereby attested.

“Same model” means one exact provider-resolved model identity and revision for
solver, diagnosis, proposal, materialization, candidate ranking, and nomination,
with role-specific inference settings frozen and matched across conditions.
Final statistical promotion remains deterministic and authority-owned.  Same
family name is insufficient.  A stronger teacher, proposer, diagnoser, or
nominator makes the run a heterogeneous harness-optimization baseline, not
same-model self-evolution and not evidence for a same-model `>10 pp` claim.

The promotion authority is outside the mutable candidate surface.  A candidate
cannot modify the task loader, grader, gate, resource ledger, sealed-set
authority, or historical evidence.  Missing evidence yields `inconclusive` or
`reject`, never an inferred pass.

## Contributions that require evidence

1. **Campaign-level error control.** Measure experiment-wide false promotion
   across candidates, rounds, and objectives rather than reporting only a
   per-decision threshold.
2. **Randomized mechanism checks.** Compare the candidate with its exact
   parent/revert and a matched neutral placebo; measure activation, adherence,
   intended behavior, utility, and protected behavior in sequence.
3. **Multi-constraint promotion.** Require benefit plus preregistered
   non-inferiority for protected slices, safety, and cost.  These constraints
   are not hidden inside an arbitrarily weighted scalar.
4. **HarnessFaultBench.** Use controlled faults with known causal surfaces to
   measure localization, verified repair, accepted-edit precision, harmful
   commits, and false promotion, then test external validity on real tasks.

Items 1--4 remain proposed contributions until the claim ledger marks their
evidence complete.

## Closest known work and non-overclaim policy

The final literature search must be repeated before submission.  As of the
2026-08-14 literature-audit cutoff, the most relevant work includes:

- [STOP](https://arxiv.org/abs/2310.02304),
  [MIPRO](https://arxiv.org/abs/2406.11695),
  [Automated Design of Agentic Systems](https://openreview.net/forum?id=t9U3LW7JVX),
  [AFlow](https://openreview.net/forum?id=z5uVAKwmjf),
  [GEPA](https://openreview.net/forum?id=RQm2KQTM5r), and
  [ACE](https://openreview.net/forum?id=eC4ygDs02R) establish recursive
  improvement, program/prompt search, workflow search, reflective evolution,
  and context evolution.
- [Meta-Harness](https://arxiv.org/abs/2603.28052),
  [AHE](https://arxiv.org/abs/2604.25850),
  [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621),
  [HarnessFix](https://arxiv.org/abs/2606.06324),
  [PACE](https://arxiv.org/abs/2606.08106),
  [Self-Harness](https://arxiv.org/abs/2606.09498),
  [HarnessBank](https://arxiv.org/abs/2607.13683),
  [SEAL](https://arxiv.org/abs/2607.24300),
  [HarnessCompass](https://arxiv.org/abs/2608.01918),
  [HarnessOpt-Bench](https://arxiv.org/abs/2608.06301), and
  [Evo-Bench](https://arxiv.org/abs/2608.09096) directly narrow the available
  novelty around full-harness search, evidence-based updates, sequential
  testing, held-out acceptance, harness archives, external auditing, and
  evolution benchmarks.
- [Penguin Harness](https://github.com/Prism-Shadow/penguin-harness) exposes a
  recursive public harness-state example.  At the pinned development revision,
  its advertised multi-task suite and graders were not public, so only the
  repeated public example can be audited; it cannot serve as a confirmatory
  hidden-set baseline.

Consequently, the paper will not claim “first self-evolving harness,” “first
verification gate,” “first independent acceptor,” or “first sequentially valid
harness search.”  Most 2026 items above are recent preprints; numerical claims
about them must be phrased as author-reported until independently reproduced.

The closest benchmark boundaries are especially important:

| Work | Already established by the cited paper | Boundary for Spiral |
|---|---|---|
| PACE | Paired anytime-valid per-candidate commit testing under optional stopping | Do not claim the first sequentially valid acceptor; test the incremental value of mechanism-linked evidence and campaign-wide spending across all nomination dimensions. |
| HarnessFix | Trace-grounded failure localization and scoped regression-aware harness repair | Do not claim the first evidence-localized harness diagnosis or repair. |
| HarnessCompass | Constrained, proactive-feedback, component-wise harness evolution with held-out transfer | Do not claim the first proactive or component-wise evolution; identify feedback versus promotion policy instead. |
| HarnessBank v2 | Activation, paired-significance, and positive-gain screening before archive admission | Do not claim the first activation check, statistical gate, or gain gate. |
| HarnessOpt-Bench | OfficeQA, BrowseComp-Plus, Terminal-Bench 2, and GAIA; exact 20/40/40 split proportions; normalized held-out gain; trusted execution; 111 scored runs | Core optimizer configurations have two search replicates, and “resolution bands” are descriptive rather than formal significance tests. It evaluates optimizer capability, not a feedback-by-promotion treatment. |
| Evo-Bench | Fixed policy model; long-horizon executable-harness editing; 160 visible validation tasks and 448 evaluation tasks from BrowseComp, HLE, GDPval, APEX-Agents, and Claw-Eval | It isolates evolver capability and transfer, not campaign false-promotion or the contribution of promotion evidence. |

At this literature-audit cutoff we had not located a public executable repository for
HarnessOpt-Bench or Evo-Bench.  Unless an auditable official release appears
before the experiment freeze, both are paper-only comparators: their reported
numbers cannot be treated as locally reproduced baselines or compared directly
with Spiral scores from a different model, task, split, or budget.

Meta-Harness's public repository does not currently contain every final
classifier implementation described by its paper.  Any local comparison must
therefore be labeled **paper-algorithm reconstruction**, not official source
reproduction or a native baseline.

The distinction from these systems is deliberately narrow.  Meta-Harness asks
what full harness code an outer optimizer can discover; Penguin and Self-Harness
show recursive updating patterns; HarnessOpt-Bench and Evo-Bench standardize
optimizer-capability evaluation; HarnessBank already gates activation and
paired gain.  SpiralHarness asks whether child--parent--revert--placebo
attribution and an activation--adherence--behavior chain improve campaign-level
false promotion and sealed utility, and uses a feedback-by-promotion 2-by-2 to
separate policy components.  This is a prospective identification claim.
Without the frozen factorial and sealed experiment it is a design difference,
not an empirical novelty result.

The current HarnessFault v4 executable is a deterministic seven-family,
six-surface trust-closure fixture.  Its 28 scenarios per partition are four
roles within seven families, not 28 independent families, and deterministic
benchmark middleware rather than base-model output drives behavior.  It is not
a live-model, optimizer-capability, self-evolution, or powered benchmark result.
A separate raw-output development path now fits `SS/MS/SM/MM` on one paired
exploration block, jointly freezes the four candidates, and executes all
attribution quartets on a source/group-disjoint fixture GATE block in one
offline-replayable `4*(28+1+4*28)=116+448=564`-call process-local ledger.  Its
fixed `SS->MS->SM->MM` dispatch is not confirmatory counterbalancing.  The runner explicitly does
not attest live provider delivery or served revision, independent freeze/gate
timing, one-time grant consumption, or cross-process atomicity, and therefore
does not change this scientific evidence boundary.

## Required empirical pattern

The central conclusion is supported only if both false-promotion tests
(absolute FULL control and paired improvement over SCORE), both sealed-utility
tests (FULL over SCORE and STATIC), and every protected/cost NI and trust-closure
gate pass.  All are frozen before confirmation; low false promotion without
sealed usefulness is reject-all behavior, while utility without the reliability
family is not a credible-commit result.  Harmful promotion and accepted-edit
precision remain key secondary endpoints and precision is undefined at zero
commits.

Any `>10 pp` statement is a separate headline family whose exact members,
power, and simultaneous multiplicity-adjusted one-sided lower bounds are frozen
before search; every bound must be strictly above `0.10` on the binary-success
scale.  The atomic sealed release is terminal.  It cannot trigger more seeds,
tasks, contrasts, sample-size continuation, or method tuning in the same
lineage; a changed method requires a new preregistration and independently
committed sealed set.
