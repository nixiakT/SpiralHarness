# Internal ICLR 2027 red-team review

Status: internal simulated manuscript/design red team, 2026-08-14.  This is not
an independent or actual ICLR review and must not be presented as evidence of
likely acceptance.  Scores use
a simulated 1--10 overall scale and 1--5 confidence scale.

## Executive verdict

**Current AC recommendation: reject / do not submit in the present state.**

The verification question is timely and narrower than “an LLM can edit a
harness.”  The typed authority boundary, explicit false-promotion endpoint, and
feedback-by-promotion decomposition could support a strong paper.  Today,
however, the manuscript is a prospective design with a sophisticated software
kernel, not a completed empirical paper.  The primary protocol is pilot-pending,
the exact replication count and final task roster are unfrozen, the sealed study
has not run, and the executable HarnessFault artifact is a deterministic
seven-family/six-surface trust-closure fixture rather than a powered benchmark.
No amount of prose review can turn those missing observations into an
acceptance guarantee.

## Scorecard

| Dimension | Current score | Reason |
|---|---:|---|
| Novelty | 5/10 | Promotion-evidence identification remains plausible, but HarnessBank v2 already gates activation and paired gain, while HarnessOpt-Bench and Evo-Bench directly benchmark held-out and long-horizon harness optimization. Only the intervention chain, campaign endpoint, and factorial identification remain distinctive. |
| Significance | 7/10 | False promotion under adaptive harness search is important if measured at realistic scale. |
| Technical soundness | 4/10 | The prospective design is thoughtful, but the main empirical effect is absent and several runtime provenance/atomicity obligations remain implementation blockers. |
| Experimental identification | 4/10 | The controlled-fault 2-by-2 can identify policy components; the real-task FULL--SCORE contrast alone changes both feedback and promotion. Final power, roster, and common-block execution are missing. |
| Reproducibility | 6/10 | Content-addressed artifacts and strict claim boundaries are strong. Reportable live execution, external authority, anonymous release, and one-command result reproduction are unfinished. |
| Clarity | 7/10 | The central question is clear after separating joint policy and factorial component estimands. The paper must remain within nine pages after real results and ablations replace status prose. |
| Ethics/security | 6/10 | Threats and limits are acknowledged; current in-process capabilities are not a hostile-code security boundary. |

## Simulated reviewer 1: novelty and positioning

**Overall: 5/10 (weak reject). Confidence: 4/5.**

### Strengths

- Frames adaptive promotion, rather than mutation generation, as the scientific
  object.
- Predeclares child/parent/revert/placebo evidence and campaign-level false
  promotion.
- Avoids “first self-evolving harness” and “official Meta-Harness reproduction”
  claims.

### Major concerns

1. Meta-Harness already performs end-to-end full-harness optimization;
   Self-Harness and Penguin demonstrate recursive harness updating; SEAL and
   Harness Updating narrow novelty around external acceptance, activation, and
   adherence.  HarnessBank v2 additionally has activation, paired-significance,
   and positive-gain gates.  The paper is novel only if it empirically
   establishes value beyond those gates through child/parent/revert/placebo,
   the adherence/behavior chain, and campaign-level false-promotion control.
2. The original FULL--SCORE wording implied a mechanism-feedback effect, but
   FULL also changes the promotion rule.  It is a joint deployable-policy
   contrast.  Only the controlled-fault `SS/MS/SM/MM` factorial can decompose
   guidance, verification, and interaction.
3. Penguin's public recursive example is one visible repeated task, not the
   advertised unreleased suite.  It is useful as a failure-mode case study, not
   a headline baseline.
4. A paper-algorithm reconstruction of Meta-Harness cannot support an official-
   system win.  A runnable official arm, if available before freeze, remains a
   secondary comparison because its mutation and feedback spaces differ.
5. HarnessOpt-Bench already evaluates 111 scored optimizer runs on OfficeQA,
   BrowseComp-Plus, Terminal-Bench 2, and GAIA with 20/40/40 split proportions,
   normalized held-out gain, and trusted execution.  Its core optimizer cells
   have only two search replicates and its resolution bands are descriptive,
   not formal significance tests, but Spiral cannot claim the first controlled
   held-out harness-optimization benchmark.
6. Evo-Bench already fixes a policy model during long-horizon harness editing
   and separates 160 visible validation from 448 evaluation tasks across five
   sources.  Spiral's remaining question is promotion-policy identification,
   not whether a model can edit a harness that transfers to held-out tasks.

### Evidence needed to change the score

- Powered, sealed `MM - SS` results plus the factorial component intervals.
- Same-budget official baselines where runnable, with reconstruction status
  visibly separated.
- An information-volume placebo showing that useful source linkage, not merely
  more feedback tokens, drives any guidance effect.
- An explicit comparison matrix showing which controls already exist in
  HarnessBank, HarnessOpt-Bench, and Evo-Bench.  At the current freeze no public
  executable repository was located for the latter two, so their numbers must
  remain paper-reported rather than presented as reproduced baselines.

### Closest-work overlap audit

| Work | Optimization/evaluation object | Existing feedback or gate | What Spiral must add empirically | Artifact status at freeze |
|---|---|---|---|---|
| Meta-Harness | Full executable harness edited by an outer coding agent | Accumulated source, traces, and scores | Identified promotion-evidence effect and campaign false promotion | Public repository is missing some paper-described final classifiers; local arm is a reconstruction. |
| HarnessBank v2 | Frozen-model harness evolution with a semantic archive | Validity, activation, paired-significance, and positive-gain screening | Revert/placebo attribution, adherence-to-behavior chain, campaign endpoint, and feedback-by-promotion factorial | Paper inspected; do not claim gating itself as novel. |
| HarnessOpt-Bench | End-to-end optimizer capability on four tasks | Graded development feedback, validation aggregates, hidden test, trusted execution | Promotion-policy treatment rather than optimizer ranking | Paper inspected; no public executable repository located. |
| Evo-Bench | Long-horizon evolver capability with a fixed policy model | Visible validation scores/trajectories and disjoint evaluation | Campaign false-promotion and component identification | Paper inspected; no public executable repository located. |
| MGV-Harness (review alias) | Promotion evidence under adaptive search | Prospective child/parent/revert/placebo and activation/adherence/behavior packets | Must demonstrate `MM - SS` and factorial margins on sealed data | Current reportable study is unexecuted. |

## Simulated reviewer 2: statistics and experimental identification

**Overall: 3/10 (reject). Confidence: 5/5.**

### Strengths

- Correctly treats a complete search as the repeated algorithmic unit.
- Uses fresh source/family gate blocks and does not count rollout seeds as fresh
  tasks.
- Plans intention-to-treat failure accounting and equal model--task weighting.

### Major concerns

1. The protocol is not preregistered: `S`, `Q`, rollout repeats, exact models,
   final tasks, ceilings, and the powered HarnessFault family roster are not
   frozen.
2. A scientifically meaningful margin must appear in the null.  The revised
   prospective protocol uses two co-primary families: absolute and relative
   family-level false-promotion control, plus sealed FULL--SCORE and
   FULL--STATIC utility margins.  Every test in both families and all NI gates
   must pass; the existing power code predates this rule.
3. Same seeds and equal manifests do not prove matched execution.  Runtime
   closure must bind schedules, preflights, retries, receipts, final ledger
   tails, failures, and one-time fresh-block consumption for every factorial
   condition.
4. Interval endpoints cannot be supplied by the projection caller.  Both
   endpoints, confidence level, estimator version, and observation closure must
   be authenticated by the independent objective producer or deterministically
   recomputed.
5. Adaptive factorial coefficients are policy effects.  After treatment changes
   history, candidates differ; they cannot be described as candidate-level
   mediation.  A separate common-candidate shadow analysis can characterize
   gate disagreement without replacing the end-to-end policy estimand.
6. The zero-event 59-family bound is only an unadjusted 95% illustration.  The
   co-primary reliability family needs a multiplicity- and power-calibrated
   count of genuinely independent null/unrepairable families at the declared
   family/source unit.

### Evidence needed to change the score

- Closed disjoint pilot and simulation through the calibrated fixed analysis
  fixing `S` before main outcomes.
- Frozen analysis code tested under each SESOI boundary null, dependence, and
  configured failure scenarios, with no outcome-selected estimator fallback.
- Complete four-cell runtime closure and one untouched sealed release.
- Holm-corrected one-sided primary margin tests, descriptive intervals, and
  prospectively multiplicity-controlled protected/cost constraints, including
  all configured failures.

## Simulated reviewer 3: artifact, benchmarks, and reproducibility

**Overall: 4/10 (reject). Confidence: 4/5.**

### Strengths

- Strong typed-artifact discipline and unusually explicit evidence ledger.
- Smoke commands clearly label GSM8K, BBH, and SkillsBench outputs as
  non-reportable.
- Public pilot reports disclose exposed data and reconstruction boundaries.

### Major concerns

1. Current benchmark/evaluation entry points do not execute the paper's
   reportable study.  The CLI exposes a GSM8K materializer, non-reportable GSM8K
   and BBH smokes, a one-task SkillsBench smoke, exposed Law/Symptom evaluations,
   and the single Penguin public recursive example.  None is a frozen
   multi-condition sealed self-evolution experiment.
2. HarnessFault v4 is a deterministic seven-family/six-surface trust-closure
   fixture.  Its 28 scenarios per partition are four roles within seven
   families, base-model output does not drive final behavior, and its
   optimizer-capability, live-model, and self-evolution flags remain false.  It
   cannot substantiate the powered prospective factor table.  The newer
   model-output-driven public-development path removes deterministic behavior
   rewriting for one finite-repair round.  Its four-context fixture now closes
   a shared paired fit block, a jointly frozen candidate set, a source/group-
   disjoint fixture GATE block, and one offline-replayable
   `4*(28+1+4*28)=116+448=564`-call ledger.  Its fixed `SS->MS->SM->MM` dispatch
   is not counterbalanced.  It is still
   not a live provider run, powered family/search replication, independently
   attested one-time block consumption, cross-process atomic campaign, or
   reportable result.
3. Process-local HMAC verification and Python capability objects are logical
   composition boundaries, not independent evaluator security.  The paper
   already acknowledges this; empirical claims must not silently rely on the
   stronger boundary.
4. The anonymous artifact has not been built from a clean identity-free checkout
   or independently reproduced.  A distinctive public system name may itself
   reveal authorship if it resolves to a public repository.
5. The official style files are hash-pinned and the source is anonymous, but a
   submission-ready build must contain no pending-result markers and must pass
   the nine-page/pdf/font/metadata gate with the final tables and figures.

### Evidence needed to change the score

- One command from frozen manifests to every paper table/figure in a clean
  anonymous environment.
- Credential-free aggregate release plus escrowed raw closures and container
  hashes.
- Out-of-process sealed authority with asymmetric verification for claims of
  evaluator independence.
- Multi-family HarnessFault generator and a real reportable benchmark runner.

## Simulated area-chair synthesis

**Verdict: reject now; encourage resubmission after the confirmatory study.**

The reviewers agree that the question is worthwhile and the design is more
careful than a typical benchmark-improvement claim.  They also agree that the
current paper has no confirmatory evidence for its central contribution.  The
software kernel and public pilots are insufficient substitutes: they neither
identify the feedback/promotion mechanisms nor establish cross-task
self-evolution.  An unfavorable but valid preregistered result would still be a
scientific contribution; a pending result is not.

No honest process can make acceptance “100%.”  The strongest feasible target is
to remove every avoidable desk-reject and validity risk, run the frozen study
once, report all outcomes, and let the claims contract to what the intervals and
closures support.

## Hard blockers before submission

1. Freeze a real, immutable preregistration; replace `pilot-pending` with exact
   hashes and complete all delegated values.
2. Freeze exact model identities, task/source groups, `S`, `Q`, repeats,
   stopping limits, budgets, retry policy, and one exact analysis
   specification.
3. Implement and independently audit source-attested score intervals and exact
   schedule/preflight/live-ledger receipt closure.
4. Execute the complete `SS/MS/SM/MM` controlled-fault factorial with isolated
   contexts and runtime-attested atomic fresh-block consumption.
5. Expand HarnessFaultBench beyond the deterministic seven-family v4 fixture to
   the frozen, powered repair/null-family sampling design with live model-driven
   behavior and independent search runs.
6. Execute independent same-model searches with no manual candidate choice or
   stronger teacher, plus distinct PURE and STATIC references, random-valid,
   and direct runnable baselines.
   Treat HarnessOpt-Bench and Evo-Bench as paper-only comparators unless an
   auditable official implementation appears before the experiment freeze.
7. Close every condition before a single untouched sealed release; rerunning
   after viewing sealed failure requires a new lineage and sealed set.
8. Freeze and calibrate one exact equal-cell analysis, audit its Holm-corrected
   margin tests, bind sealed inputs, and produce the false-promotion bound,
   protected/cost closure, and all failure sensitivities.  The current basic
   nested-bootstrap/power code is a method-development prototype, not a formal
   seed recommendation or confirmatory pipeline.
9. Replace the pending-results table with complete results, including negative
   endpoints; resolve every `EVIDENCE BLOCKED` marker.
10. Release and independently reproduce a clean anonymous artifact; audit the
    coined system name, Git history, PDFs, logs, CAS payloads, and URLs for
    identity and secrets.
11. Run the official-style submission gate on the final source and recheck the
    ICLR call, AI policy, author profiles, reciprocal-review obligation, and
    deadlines 72 and 24 hours before submission.

## Non-blocking improvements

- Include a compact main-paper comparison table separating optimization target,
  feedback, promotion authority, randomized intervention controls, and campaign
  validity for Meta-Harness, HarnessBank, HarnessOpt-Bench, Evo-Bench,
  Self-Harness/Penguin, and MGV-Harness (review alias).
- Put the principal factorial result, information-volume placebo, cost, and
  strongest failure mode in the main nine pages.
- Report gate inconclusive rate and calls/tokens per verified repair, not only
  final utility.
- Preserve a negative-result narrative: which mechanism links fail, for which
  surfaces, and at what cost can be publishable even if the headline margin does
  not clear.

## Remediation pass after this review

The same-day follow-up audit found and corrected additional design-level issues;
these corrections do not remove the missing-experiment blockers above:

- the paper now cites and distinguishes PACE, HarnessFix, and HarnessCompass;
- the cross-task primary unit is prospectively binary success, so `0.10` has a
  percentage-point interpretation rather than an arbitrary normalized-score one;
- `mu(a,m,t)` now names the frozen source-group target and algorithmic
  randomization, while models and tasks remain fixed rosters;
- every non-programmatic mechanism judge must pass a disjoint, arm-blinded
  human-label calibration protocol with frozen error thresholds;
- model-mediated candidate ranking/nomination is separated from deterministic,
  authority-owned final statistical promotion;
- infrastructure retries operate on a prospectively indivisible six-condition
  or four-cell block, retaining original failures and cost; and
- a model-call/token-budget-matched `PURE@B` reference complements the
  single-call PURE effectiveness contrast; its prospective schema now closes
  task/evaluation-unit/sample allocations and exact FULL commitment joins, but
  a live executor and receipts remain missing;
- the campaign theorem now controls only fresh-block statistical false
  authorization, explicitly excluding sealed-shift harm, out-of-scope grader or
  policy exploits, provenance failures, and judge/model misspecification;
- the controlled-fault primary unit is now one independent family assigned one
  preregistered primary optimizer seed shared across policies; extra seeds never
  increase `n`, and weighted/averaged designs require coverage-validated family-
  clustered inference rather than an exact binomial bound;
- SCORE and FULL must purchase and record the same mechanism probes and blinded
  judge calls, while SCORE receives only a feedback-free shadow projection and
  performance-only promotion rule; and
- operational owners, not favorable pilot effects, must justify and freeze the
  false-promotion, utility, repair, protection, and cost margins before the
  outcome-bearing pilot.

The paper release gate now also requires a frozen protocol and a digest-bound
release attestation.  The internal `docs/iclr-2027/` tree is explicitly excluded
from anonymous uploads; the exact allowlisted supplement still needs a separate
unpacked-archive audit and independent reproduction receipt.
