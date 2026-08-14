# ICLR 2027 submission checklist

Last verified against official pages: 2026-08-14 (Asia/Shanghai).  Recheck 72
hours and 24 hours before each deadline because the call labels dates as
planned and some reviewer-page text contains stale year/date fragments.

Official sources (all accessed 2026-08-14, Asia/Shanghai):

- [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [Dates and Deadlines](https://iclr.cc/Conferences/2027/Dates)
- [Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
- [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- [Official style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)
- [ICLR 2027 OpenReview venue](https://openreview.net/group?id=ICLR.cc/2027/Conference)

## Deadlines

- [ ] Register a real, non-placeholder abstract by 2026-09-18 AOE (about
  2026-09-19 19:59:59 in Shanghai).
- [ ] Submit the full paper and supplement by 2026-09-25 AOE (about 2026-09-26
  19:59:59 in Shanghai).
- [ ] Freeze every author by the abstract deadline; no author may be added or
  removed afterward.  Author order may change only until the full-paper
  deadline.
- [ ] Ensure every author has a complete OpenReview profile.  Non-institutional
  email moderation can take about two weeks.
- [ ] Verify reciprocal reviewing obligations and the per-author submission
  limit before registration.

## Format and anonymity

- [ ] Use the unmodified 2027 official LaTeX style, US Letter, with
  `\iclrfinalcopy` commented for initial submission.
- [ ] Keep initial main text within 9 pages; references and appendix are
  unlimited.  Rebuttal/camera-ready main text may use 10 pages.
- [ ] Put the central method, principal results, critical ablations, and core
  limitations in the main text; reviewers need not read the appendix.
- [ ] Remove author names, institutions, acknowledgments, repository ownership,
  identifying paths/metadata, and non-anonymous supplement content.
- [ ] Audit every code/data URL for anonymity and reviewer-tracking behavior.
- [ ] Search the public web for the title, system name, benchmark name, and
  distinctive exact phrases.  The manuscript currently uses `MGV-Harness` as a
  review alias, but committing that alias and title to an author-identifying
  public repository can still defeat blinding.  Until a policy-compliant
  identity-isolation strategy is verified for the final title, source, Git
  history, and artifact, treat anonymity as a submission blocker.  Restore the
  public project name only for the camera-ready version after review.
- [ ] Cite the authors' earlier work in the third person rather than hiding it.
- [ ] Do not alter margins, type sizes, line numbers, or style internals.

## Required and recommended statements

- [ ] Include the required AI use statement outside the main-text page limit.
  The official template caps this statement at one page; the recommended ethics
  statement also has a one-page cap.
  It must disclose AI assistance with hypothesis/idea
  refinement, experimental feedback/design, implementation, result
  interpretation, translation, literature work, writing/editing, code/artifact
  generation, and scientific figures where applicable.
- [ ] Repeat the required AI-use disclosure in the OpenReview submission form;
  the manuscript section does not replace the form response.
- [ ] Describe concrete human verification: literature/plagiarism checks,
  code review and tests, recomputed analyses, source validation, and author
  responsibility for every claim and artifact.  The final AI statement may list
  only checks actually completed, not planned verification as if it had occurred.
- [ ] Include an ethics statement (recommended, outside the page limit) covering
  persistent self-modification, poisoning, privilege/secret/tool boundaries,
  rollback, privacy, misuse, and benchmark/resource externalities.
- [ ] Include a reproducibility statement (strongly recommended, outside the
  page limit) pointing to exact method, appendix, data, environment, raw
  trajectory, and figure/table-generation materials.
- [ ] Confirm no concurrent submission to another archival venue.

## Scientific readiness gate

- [ ] Freeze the claim--evidence ledger and remove unsupported “first,” causal,
  generality, SOTA, official-reproduction, and `>10 pp` statements.
- [ ] Run `SUBMISSION_READY=1 paper/ci/check.sh`; a pending-result table,
  `EVIDENCE BLOCKED` marker, placeholder abstract, or unresolved primary endpoint
  is a hard submission blocker, not text to leave for reviewers.
- [ ] Complete the preregistered FULL-versus-SCORE contrast using independent
  search runs and an untouched atomic sealed release.
- [ ] Close the PURE task-payload/provider-minimal execution reference and keep
  it distinct from STATIC, the unchanged seed harness.  Match exact solver
  revision/configuration and task/group/rollout coordinates across evaluation
  arms; do not pad PURE with fictitious optimizer use.  Also run PURE@B, which
  spends FULL's frozen ex-ante model-call/token ceiling under a nonadaptive
  task-native aggregation rule; label single-call FULL-minus-PURE as an
  effectiveness rather than model-call/token-budget-matched contrast.  Do not
  define a cross-task PURE@B estimand unless every primary task binds a total
  aggregation rule.  The prospective contract now also freezes every
  task/evaluation-unit/sample allocation and joins the model, split, grader,
  query DAG, retry policy, calls, attempts, and tokens to FULL; a runtime must
  still emit receipts proving those allocations were honored.
- [ ] Before changing protocol status from `pilot-pending` to `frozen`, bind
  exact served models, task/source manifests, `S`, `Q`, repeats, budgets,
  factorial/fault-family roster, retry/failure rules, analysis/container hashes,
  graders, price table, and sealed-authority key; reject blanks and defaults.
- [ ] Report all configured conditions, failures, retries, resource usage,
  search seeds, and protected slices; do not show only the best trajectory.
- [ ] Include direct comparisons with runnable same-budget baselines; label
  paper-algorithm reconstructions explicitly.  Treat HarnessOpt-Bench and
  Evo-Bench as paper-only comparators unless an auditable official executable
  release appears before the experiment freeze.
- [ ] Treat any baseline with a stronger proposer, diagnoser, materializer,
  candidate ranker, or nominator as heterogeneous.  It cannot support a
  same-model or `>10 pp` claim; final statistical promotion must instead remain
  deterministic and authority-owned.
- [ ] For a same-model self-evolution claim, attest one exact served model
  identity/revision across solver, diagnosis, proposal, nomination, and
  every model-mediated ranking role; list any role-specific inference settings
  and exclude stronger teachers from that claim.
- [ ] Close the controlled-fault `SS/MS/SM/MM` factorial at runtime.  Planned
  topology, equal seeds, or an `atomic=true` manifest do not establish executed
  feedback, promotion, or interaction effects.
- [ ] Assign one primary optimizer seed to each independent fault family and
  pair it across policies.  Treat family as `n`; extra seeds require one frozen
  family aggregate and must not enter an exact-binomial or seed-level analysis
  as independent families.  Freeze stratum weights for a fractional roster.
- [ ] Execute and charge the same mechanism probes and blinded judge calls in
  SCORE and FULL.  Keep SCORE's mechanism results as feedback-free shadow
  measurements so disclosure/promotion, rather than purchased call topology,
  defines the treatment.
- [ ] Source both confidence-interval endpoints and their estimator from the
  independent objective artifact; no reportable decision may depend on a
  caller-supplied bound.
- [ ] Report search-seed uncertainty and task-group uncertainty; do not treat
  rollouts as independent search runs.
- [ ] Do not use the current equal-cell basic nested-bootstrap/power prototype
  to freeze `S` or label evidence confirmatory until coverage calibration,
  boundary-null auditing, closed-pilot provenance, and sealed input attestation
  are complete.  The current code's legacy event selects on three old SESOI
  rejections plus two simultaneous `>10 pp` bounds; it omits the new family-
  level false-promotion co-primary family, harmful promotion, NI, PURE/external
  headline families, cost, and provenance closure and therefore cannot freeze
  the final sample size.  It also assumes legacy repair independence and no
  failures, and leaves the Wilson-bound power target caller-declared.  Freeze
  the target/candidate sequence/no-fallback rule, cover reliability/endpoint
  dependence and ICC, require one-sided
  Monte Carlo upper-bound or theoretical type-I validation, and make every
  audit failure fail closed.
- [ ] For any `>10 pp` statement, require every applicable claim-family
  simultaneous multiplicity-adjusted one-sided lower confidence bound to be
  strictly greater than `0.10`; a point estimate is never sufficient.  Use a
  prospectively frozen binary success-probability scale so `0.10` actually means
  ten percentage points; never apply that label to an arbitrary normalized
  continuous score.
- [ ] Freeze the exact members and power of every `>10 pp` headline family
  before search.  Never add tasks, seeds, contrasts, or method iterations after
  the atomic sealed release; a changed method requires a new preregistration and
  independently committed sealed set.
- [ ] Freeze every protected/cost estimand, margin, missingness rule, and
  multiplicity family.  Verify total adaptive gate spending over every
  `(nomination, condition, dimension)` is at most `0.05`.
- [ ] Obtain operational signoff for `delta_FP`, utility/repair SESOIs, protected
  NI margins, and cost ceilings before the outcome-bearing pilot; never select a
  margin from a favorable pilot effect.
- [ ] Describe HarnessFault v4 only as a deterministic seven-family/six-surface
  trust-closure fixture.  Its 28 scenarios per partition are not 28 independent
  families and do not support live-model, optimizer, self-evolution, or powered
  benchmark claims.
- [ ] Describe the 564-call four-context HarnessFault path only as fixture-backed
  process-local development plumbing.  Its shared paired fit block, joint
  candidate freeze, source/group-disjoint fixture GATE block, and offline ledger replay do not
  attest provider revision, freeze/gate wall-clock order, one-time block
  consumption, cross-process atomicity, or a powered/reportable result.
- [ ] Prefer programmatic mechanism evidence.  For every unavoidable LLM judge,
  freeze identity/revision, prompt/rubric, decoding, blindness, sampling frame,
  disjoint human calibration set, agreement and class-conditional error bounds,
  acceptance thresholds, adjudication, abstention, and failure rules.
- [ ] Add a blinded naturally occurring fault subset and external adversarial
  audit; accept semantic-equivalent alternative repairs, calibrate placebo
  neutrality, and cluster shared generator/template sources.
- [ ] Release an anonymous artifact with lock/container hashes, frozen configs,
  manifests, receipts, raw aggregate inputs, and deterministic plot/table
  scripts.
- [ ] Build the upload from an explicit allowlist.  Exclude `.git`, the entire
  internal `docs/iclr-2027/` directory, development runs, caches, and public
  repository metadata.  Unpack and scan the exact final supplement archive;
  `paper/ci/check.sh` alone audits only the paper source/PDF and does not certify
  an anonymous supplement.
- [ ] Audit dataset/code licenses and add an explicit first-party license only
  after the rights holder selects it.
- [ ] Run secret scanning and ensure API keys, local usernames, private paths,
  endpoint credentials, and signing keys do not enter source, paper, logs, or
  supplement.

## Final submission audit

- [ ] Build from a clean anonymous checkout and reproduce all paper tables and
  figures from committed aggregate inputs.
- [ ] Check PDF page size, fonts, line overflow, figure legibility in grayscale,
  hyperlinks, bibliography consistency, accessibility text, and metadata.
- [ ] Have independent reviewers separately audit novelty, statistics/leakage,
  mechanism claims, cost/fairness, security/ethics, reproducibility, and prose.
- [ ] Resolve every critical review item or record it prominently as a
  limitation; never hide a failed preregistered endpoint.
- [ ] Recheck the official call, author guide, template, and OpenReview form at
  72 and 24 hours before submission.

The conference date page currently lists 2027-04-26 through 2027-04-30 while
also saying location and dates are to be announced.  Treat the submission
deadlines as operative and the meeting location/date details as updateable.
