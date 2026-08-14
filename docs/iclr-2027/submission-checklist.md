# ICLR 2027 submission checklist

Last verified against official pages: 2026-08-14 (Asia/Shanghai).  Recheck 72
hours and 24 hours before each deadline because the call labels dates as
planned and some reviewer-page text contains stale year/date fragments.

Official sources:

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
- [ ] Cite the authors' earlier work in the third person rather than hiding it.
- [ ] Do not alter margins, type sizes, line numbers, or style internals.

## Required and recommended statements

- [ ] Include the required AI use statement (outside the page limit, no more
  than one page).  It must disclose AI assistance with hypothesis/idea
  refinement, experimental feedback/design, implementation, result
  interpretation, translation, literature work, writing/editing, code/artifact
  generation, and scientific figures where applicable.
- [ ] Describe concrete human verification: literature/plagiarism checks,
  code review and tests, recomputed analyses, source validation, and author
  responsibility for every claim and artifact.
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
- [ ] Complete the preregistered FULL-versus-SCORE contrast using independent
  search runs and an untouched atomic sealed release.
- [ ] Report all configured conditions, failures, retries, resource usage,
  search seeds, and protected slices; do not show only the best trajectory.
- [ ] Include direct comparisons with runnable same-budget baselines; label
  paper-algorithm reconstructions explicitly.
- [ ] Report search-seed uncertainty and task-group uncertainty; do not treat
  rollouts as independent search runs.
- [ ] Calibrate LLM judges against blinded human labels or prefer programmatic
  evidence, reporting agreement and uncertainty.
- [ ] Release an anonymous artifact with lock/container hashes, frozen configs,
  manifests, receipts, raw aggregate inputs, and deterministic plot/table
  scripts.
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
