# Research position: verification science for self-evolving harnesses

Status: design freeze, 2026-08-14.  This is a claim boundary, not a result.

## Question

Given a fixed solver model, mutation space, and search budget, what evidence is
sufficient to promote an adaptively selected harness edit as a reproducible
improvement?

The working title is:

> **SpiralHarness: Mechanism-Gated Verification for Self-Evolving LLM
> Harnesses**

The paper should not sell “an LLM edits its own prompt or harness” as the new
idea.  That capability is already established.  The proposed contribution is
to measure and control *false promotion* during adaptive harness search.

## Main treatment

The main experiment compares two budget-matched optimizers:

- **SCORE:** sees only aggregate score, confidence bounds, resource totals,
  and the deterministic gate decision.
- **FULL:** has the same model, candidate grammar, candidate count, task
  coordinates, scoring summaries, and budget, but additionally receives typed
  evidence linking activation, adherence, behavior, revert, and placebo
  interventions to the proposed edit.

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
freeze, the most relevant work includes:

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
  [HarnessCompass](https://arxiv.org/abs/2608.01918), and
  [Evo-Bench](https://arxiv.org/abs/2608.09096) directly narrow the available
  novelty around full-harness search, evidence-based updates, sequential
  testing, held-out acceptance, harness archives, external auditing, and
  evolution benchmarks.

Consequently, the paper will not claim “first self-evolving harness,” “first
verification gate,” “first independent acceptor,” or “first sequentially valid
harness search.”  Most 2026 items above are recent preprints; numerical claims
about them must be phrased as author-reported until independently reproduced.

Meta-Harness's public repository does not currently contain every final
classifier implementation described by its paper.  Any local comparison must
therefore be labeled **paper-algorithm reconstruction**, not official source
reproduction or a native baseline.

## Required empirical pattern

The central conclusion is supported only if all preregistered primary tests
pass, FULL improves on SCORE on the atomic sealed release, protected behavior
and cost stay within bounds, and the upper confidence bound on false promotion
meets the safety target.  A benchmark score increase without the matched
mechanism result is an end-to-end pilot, not evidence for the paper's proposed
mechanism.
