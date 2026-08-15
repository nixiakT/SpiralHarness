# BFCL V4 public-development v2 trusted grader

This grader produces binary development labels for the frozen 25-task v2
roster without placing possible answers, answer-derived hashes, task IDs, or
checker diagnostics in an application- or candidate-visible artifact. It is
public-development infrastructure, not hidden-test, official BFCL, or
reportable confirmatory evidence.

## Trust boundary

`BfclV4LoadedPublicDevelopmentV2` remains a question-only bundle. Its
`possible_answers_read=False` claim is unchanged: the trusted grader does not
retroactively treat that object as an answer bundle.

The control plane opens the question bundle and exact 1,098-node campaign from
the pinned upstream commit. For each grade, a `python -I` worker independently
reads one question row, its public possible-answer row, and the five byte-pinned
upstream AST-checker sources from Git objects at that commit. The worker emits
only a boolean plus non-answer source bindings. Its v2 roster is independent of
the older 15-task grader; only the older worker's generic, exact-source JSON/Git
and AST-loader primitives are reused. Both local worker source files are pinned
by SHA-256 before execution.

The worker input contains the trusted-plane task ID because the upstream files
are keyed by it. The parent receipt does not retain or expose that ID.

## Structural resolution and admissible nodes

Task references resolve by zero-based index within each split in frozen manifest
order:

- `fit-00` through `fit-04`
- `gate-00` through `gate-03`
- `holdout-00` through `holdout-15`

The grader accepts only these split/kind combinations:

| Split | Model-call node kinds |
| --- | --- |
| FIT | `parent-fit`, `candidate-fit` |
| GATE | `gate` |
| HOLDOUT | `holdout` |

Diagnosis, proposal, nomination, and decision nodes are not gradeable. The
submitted node must equal its unique campaign node, and its plan fingerprint,
schedule hash, node hash, global call slot, provider seed, request fingerprint,
native request/response adapter hashes, tool mapping, model route, inference
configuration, and token ceilings are revalidated before any answer is read.

`pure-at-b-sample` is deliberately absent from the single-response interface.
Its 330 source calls must remain label-free: journal events carry no binary
grade, trusted-grade request, trusted-grade receipt, or consumed grader
attempt. PURE@B has a separate aggregate-only boundary described below.

## HOLDOUT authority

FIT and GATE can be graded as non-reportable development evidence. HOLDOUT is
closed by default. Opening a grader without authority material leaves every
HOLDOUT call rejected.

An authorized trusted plane must configure both a semantic-release fingerprint
and at least 32 bytes of HMAC secret. It can then authenticate a
`BfclV4PublicV2DecisionBarrierEvidence` only when the evidence binds, in frozen
order, all six `gate-decision` node hashes and six unique replayed event hashes,
asserts that all gates are terminal, and asserts that evaluation has not begun.
The resulting `BfclV4PublicV2EvaluationUnlock` is bound to the campaign,
schedule, semantic release, global decision barrier, and authority key ID.
Changing any field invalidates the HMAC. Non-HOLDOUT calls reject an unlock.

No production semantic release or authority secret is stored in this
repository, so the checked-in default remains fail-closed.

## PURE@B atomic aggregate grading

PURE@B exposes no public single-cell or single-sample grade operation. The
only entry point accepts one `BfclV4PublicV2PureAtBBatchGradeRequest` containing
exactly 48 cells in frozen outer-seed × HOLDOUT-task order. Together those
cells must reference all 330 unique `pure-at-b-sample` nodes and events.

Each cell binds its exact 6-or-7-call allocation, complete source nodes and
events, aggregation rule, recomputed modal result, selected canonical response,
semantic release, decision barrier, and authenticated evaluation unlock. A
provider-free builder can construct this request from the frozen campaign,
complete journal events, and the 48 independently replayed aggregation records.

The trusted runtime operates in two passes. First it strictly revalidates the
complete batch, exact campaign nodes and provider-request lineage, every source
event, every non-empty canonical response, and every recomputed modal result.
No worker runs during this pass. Only after all 48 cells pass does the second
pass grade each final selected modal response once. A selected no-response is
deterministically false without starting the answer-reading worker. Thus the
batch always records 48 final grader attempts, while isolated-worker execution
count equals only the number of selected-response cells; individual-sample
grade count remains zero.

The batch receipt contains 48 minimal cell receipts and binds each selected
canonical response itself, its complete source set, allocation, aggregation,
release, barrier, unlock, and trusted implementation. It contains no upstream
task ID, answer, answer-derived identity, or checker diagnostic.

## Minimal receipt

`BfclV4PublicV2TrustedGraderReceipt` contains:

- campaign, schedule, node, structural-task, request, response, implementation,
  question-bundle, checker-source, and local-worker lineage hashes;
- the split role and frozen campaign call slot;
- the authenticated unlock fingerprint for HOLDOUT only; and
- `correct: bool`.

It explicitly fixes `candidate_visible=False`, `task_id_present=False`,
`answers_present=False`, `answer_derived_identities_present=False`, and
`checker_diagnostics_present=False`. It also fixes
`score_bearing_execution_allowed=False`, `hidden_test_evidence=False`, and
`reportable_result=False`. Downstream candidate feedback must use a separately
reviewed projection rather than serialize this trusted receipt directly.

## API sketch

```python
grader = open_bfcl_v4_public_v2_trusted_grader(checkout, campaign)
receipt = grader.grade(grade_request)  # FIT or GATE
```

For a trusted HOLDOUT run, configure authority material when opening the
grader, call `issue_evaluation_unlock()` with independently replayed barrier
evidence, and pass that unlock to `grade()`. Callers never submit an answer,
split, task ID, expected label, or checker diagnostic.

For PURE@B, build the complete request with
`build_bfcl_v4_public_v2_pure_at_b_batch_grade_request(...)` and call
`grade_bfcl_v4_public_v2_pure_at_b_batch(grader, batch)`. Passing fewer than
48 cells, a graded sample event, or a forged modal result fails before answer
access.

The contract owner is
`spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts`; runtime
entry points are in
`spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader`.
PURE@B aggregate contracts and runtime are owned by the corresponding
`bfcl_v4_public_v2_pure_at_b_trusted_grader_contracts` and
`bfcl_v4_public_v2_pure_at_b_trusted_grader` leaves.
