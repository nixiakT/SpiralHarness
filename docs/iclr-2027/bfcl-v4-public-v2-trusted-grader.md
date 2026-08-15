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
| HOLDOUT | `holdout`, `pure-at-b-sample` |

Diagnosis, proposal, nomination, and decision nodes are not gradeable. The
submitted node must equal its unique campaign node, and its plan fingerprint,
schedule hash, node hash, global call slot, provider seed, request fingerprint,
native request/response adapter hashes, tool mapping, model route, inference
configuration, and token ceilings are revalidated before any answer is read.

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

The contract owner is
`spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts`; runtime
entry points are in
`spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader`.
