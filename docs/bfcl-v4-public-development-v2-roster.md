# BFCL V4 public-development v2 roster

This artifact freezes a prospective, question-only BFCL V4 development
roster. It is public development data, not hidden or sealed evaluation
evidence. It does not freeze an execution seed, model call plan, or token
budget, and it must not be used for score-bearing execution.

## Source and deterministic selection

The loader reads only the four BFCL V4 question JSONL blobs at Gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. It does not read possible-answer
files, scores, prior runs, or CAS objects and does not invoke a model or the
network.

The answer-free source layer has no dependency on the BFCL fixture bridge or
fixture contracts. It requires a real standalone checkout root at the exact
HEAD, rejects symlinked checkouts and linked worktrees, reads blobs with
`git cat-file blob <commit>:<path>`, and sets `GIT_NO_LAZY_FETCH=1` on every Git
operation. The accepted receipt binds the following ordered paths exactly:
`simple_python`, `multiple`, `parallel`, then `parallel_multiple`, including
each observed byte size and SHA-256. Reordering or forging one binding rejects
the receipt. Both the manifest validator and loader also require the source
reader's pinned commit to equal the manifest's upstream commit before Git is
opened.

All 1,000 public question rows are placed in a graph. Two rows are connected
when either condition holds:

1. they share an official function name; or
2. their complete user-question token sequences are identical after NFKC,
   case-folding, alphanumeric tokenization, and replacement of every
   digit-containing token by `<num>`.

Only graph components containing exactly one task are eligible, and the 15 v1
task IDs are excluded. This produces 202 eligible singleton components. Within
each category, eligible tasks are ranked lexicographically by:

```text
sha256(
  utf8("spiral-bfcl-v4-public-development-v2-selection/v1") || 00 ||
  utf8("spiral-bfcl-v4-public-development-v2:2026-08-15:pre-v1-result-open") || 00 ||
  utf8(category) || 00 || utf8(task_id)
)
```

The salt and formula were fixed before inspecting any v1 result. The first
category-quota entries are assigned in FIT, GATE, HOLDOUT order:

| Split | simple_python | multiple | parallel | parallel_multiple | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIT | 2 | 1 | 1 | 1 | 5 |
| GATE | 1 | 1 | 1 | 1 | 4 |
| HOLDOUT | 4 | 4 | 4 | 4 | 16 |
| Total | 7 | 6 | 6 | 6 | 25 |

The ordered v1 exclusion roster is independently bound by:

```text
canonical_sha256({
  "domain": "spiral-bfcl-v4-public-development-v2-v1-exclusion/v1",
  "task_ids": ordered_v1_task_ids
})
```

The manifest validator and loader both recompute this formula and require the
ordered IDs to equal the question-only v1 row-identity roster. Reordering even
the same 15 IDs, or drifting the underlying v1 roster, rejects before
selection.

## Frozen roster

| Split | Category | Task ID |
| --- | --- | --- |
| FIT | simple_python | `simple_python_198` |
| FIT | simple_python | `simple_python_162` |
| FIT | multiple | `multiple_30` |
| FIT | parallel | `parallel_24` |
| FIT | parallel_multiple | `parallel_multiple_2` |
| GATE | simple_python | `simple_python_179` |
| GATE | multiple | `multiple_71` |
| GATE | parallel | `parallel_26` |
| GATE | parallel_multiple | `parallel_multiple_52` |
| HOLDOUT | simple_python | `simple_python_218` |
| HOLDOUT | simple_python | `simple_python_286` |
| HOLDOUT | simple_python | `simple_python_385` |
| HOLDOUT | simple_python | `simple_python_292` |
| HOLDOUT | multiple | `multiple_16` |
| HOLDOUT | multiple | `multiple_92` |
| HOLDOUT | multiple | `multiple_19` |
| HOLDOUT | multiple | `multiple_84` |
| HOLDOUT | parallel | `parallel_13` |
| HOLDOUT | parallel | `parallel_23` |
| HOLDOUT | parallel | `parallel_41` |
| HOLDOUT | parallel | `parallel_39` |
| HOLDOUT | parallel_multiple | `parallel_multiple_13` |
| HOLDOUT | parallel_multiple | `parallel_multiple_38` |
| HOLDOUT | parallel_multiple | `parallel_multiple_50` |
| HOLDOUT | parallel_multiple | `parallel_multiple_16` |

The selected rows contain 44 official function names, all globally unique.
Selected task IDs, exact graph components, and official function names have no
overlap with the v1 roster under the question-only graph above.

## Near-duplicate audit and limitations

The frozen audit compares three-token shingles across every pair of different
v2 splits and between every v2 task and every v1 task. There are 539 fixed
comparisons: 164 cross-v2-split pairs and 375 v2-to-v1 pairs. A pair is flagged
only when shingle Jaccard similarity is at least `67/100` and containment is at
least `4/5`. The frozen roster has zero flagged pairs.

These checks establish disjointness only for shared official names, exact
normalized question templates, and the stated lexical threshold. They do not
prove broad semantic or source-family independence. No independent semantic
grouping authority has attested the roster. Consequently
`independent_semantic_family_disjointness_attested=false` and
`score_bearing_execution_allowed=false`; the runtime boundary raises rather
than releasing a score-bearing plan.

Any source-lineage, frozen-identity, deterministic-selection, family-graph,
function-name, or lexical-audit failure emits a typed rejected receipt. The
loader never changes the salt, retries a draw, or backfills a failed selection.
Lower-level exception details are erased before that typed rejection is
raised: both its Python `__context__` and `__cause__` are empty.

The loaded bundle is trusted-control-plane-only because it contains the full
manifest, split labels, and all tasks in partition order. A future executor
must release only one split-free `BfclV4PublicDevelopmentV2Task` view at a time;
it must never expose the loaded bundle itself to a candidate. The current score
guard prevents any execution release.

Frozen identities:

- candidate pool: `88cef8b2fe76409f35f735b85dc230af39c21d30af0b4457c6a2d8e1d283b064`
- eligible singletons: `3069ad528ea31fcb5cae92b3e67945262a05ad50227d22b68e135275a25ab209`
- lexical audit: `c75fdfca954773007c9475818cf408facfe958081ed946cedb5d339880d21542`
- v1 exclusion: `07a360ed6e32da5a94ee6d553aa31234f86cd37d024b411f351df4b8f0cc87c0`
- roster content: `2d629219523faf7469f17b12192e8286cb5e0ea1988b424828636c465a703d1c`
- manifest fingerprint: `3f6e410bd26466381090ced0c6144515281a9653e5d36429685ad898f168eae7`
