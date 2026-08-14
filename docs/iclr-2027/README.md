# ICLR 2027 research freeze

This directory separates the intended paper from exploratory engineering and
previously exposed benchmark results.  A result is not confirmatory merely
because it was produced by a frozen executable or recorded in the artifact
store.

- `research-position.md` defines the paper's narrow scientific question and
  its relationship to the closest known work.
- `preregistered-main-study.md` specifies the proposed scientific structure and
  records the exact delegated values that must be filled before its
  `pilot-pending` status can become an immutable preregistration.
- `claim-evidence-ledger.md` records which statements are currently supported,
  exploratory, or blocked on new evidence.
- `submission-checklist.md` tracks the official ICLR 2027 requirements and
  desk-reject risks.
- `mechanism-factorial-design.md` separates feedback guidance from promotion
  verification and states the fresh-block/runtime evidence needed for either
  causal claim.
- `third-party-audit.md` records Meta-Harness code/data provenance and
  redistribution limits.
- `internal-red-team-review.md` is a simulated reviewer/AC audit and blocking
  action register; it is not acceptance evidence or a submission supplement.
- `submission-release.yml` is a fail-closed release attestation.  It remains
  `evidence-blocked` until the frozen protocol, result bundle, exact anonymous
  supplement, and independent reproduction receipt are digest-bound.

This entire directory is internal project governance and is excluded from the
anonymous supplement.  The final upload must be assembled from an explicit
allowlist and scanned after unpacking; copying the repository or Git history is
forbidden.

The real-task comparison includes three distinct references.  `PURE` is the task
payload plus only the provider-minimal wrapper: it receives no seed harness,
skills, memory, tools, middleware, or harness control flow.  `STATIC` runs the
frozen seed harness without adaptive mutation or promotion.  Neither may be
silently substituted for the other in a model-plus-harness claim.  `PURE@B`
spends FULL's frozen ex-ante model-call/token ceiling on independent PURE
samples with a nonadaptive task-native aggregation rule; it prevents extra
inference compute from being mistaken for a harness mechanism benefit.

The repository's current basic nested-bootstrap and power implementations are
method-development prototypes.  They do not freeze a replication count, prove
coverage, authenticate sealed observations, or constitute a confirmatory
analysis until the calibration, independent audit, authority attestation, and
immutable preregistration obligations in the study document are complete.

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
