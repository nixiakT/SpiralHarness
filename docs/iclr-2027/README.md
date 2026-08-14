# ICLR 2027 research freeze

This directory separates the intended paper from exploratory engineering and
previously exposed benchmark results.  A result is not confirmatory merely
because it was produced by a frozen executable or recorded in the artifact
store.

- `research-position.md` defines the paper's narrow scientific question and
  its relationship to the closest known work.
- `preregistered-main-study.md` fixes the hypotheses, controls, statistical
  unit, leakage policy, power procedure, and stopping/invalidation rules.
- `claim-evidence-ledger.md` records which statements are currently supported,
  exploratory, or blocked on new evidence.
- `submission-checklist.md` tracks the official ICLR 2027 requirements and
  desk-reject risks.

The protocol has three evidence classes:

1. **Engineering verification** establishes that schemas, journals, receipts,
   and fail-closed checks behave as specified.
2. **Exploratory evidence** may motivate a hypothesis, estimate nuisance
   parameters, or expose a failure mode.  It may not be promoted to a
   confirmatory result after its test data have been inspected.
3. **Confirmatory evidence** requires a frozen preregistration, disjoint search
   and evaluation data, complete independent search runs, an atomic one-time
   sealed release, and the analysis fixed in advance.

No document in this directory promises acceptance.  The purpose of the freeze
is to make claims falsifiable and to make an unfavorable result publishable
rather than silently reopening the test set.
