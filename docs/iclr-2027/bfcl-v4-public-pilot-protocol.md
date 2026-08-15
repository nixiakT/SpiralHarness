# BFCL V4 public-pilot protocol

Status: frozen engineering protocol; no model calls or scores are recorded here.

## Claim boundary

This is a 15-task, public, development-only partial BFCL V4 pilot pinned to Gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` and `bfcl-eval==2026.3.23`. BFCL publishes
its questions, possible answers, and grader. Consequently, this pilot is not hidden or sealed,
does not produce the official full-suite score, and must not be used as headline confirmatory
evidence. The information split below tests harness mechanics; it does not make public data secret.

The candidate-facing loader reads only exact-commit question Git objects. Its returned typed value
contains exact source coordinates but deliberately says `runtime_git_object_read_attested=false`:
a serializable Pydantic object is not an execution receipt. A reportable execution would still need
an independently bound runtime receipt. No possible-answer path or answer-derived identity is in
the candidate task contract.

## Frozen roster

| Split | Tasks |
|---|---|
| FIT (5) | `simple_python_0`, `simple_python_211`, `multiple_5`, `parallel_0`, `parallel_multiple_9` |
| GATE (2) | `multiple_10`, `parallel_multiple_11` |
| HOLDOUT (8) | `simple_python_87`, `simple_python_128`, `multiple_7`, `multiple_8`, `parallel_3`, `parallel_4`, `parallel_multiple_5`, `parallel_multiple_55` |

The 15 semantic-family labels are unique, as are all 26 official function names. Four exact
question-blob identities, 15 exact raw-row identities, and 15 canonical candidate-payload
identities close the roster to public source bytes.

Two older commitments are retained exactly as externally supplied values:

- external roster commitment: `3556bca5d47d625fa3ba1cc086d6ac5c71b11a5327b16abb1d142c8e81ea1fee`;
- external seed commitment: `4d210dfc8bc99e795cd8e58ed913a876180c4cf28f4310b2af68650e4e924042`;
- external outer seed: `2026081501`.

Their original canonical payload and derivation algorithm were not recoverable. The contracts
therefore set both derivation attestations to false and never assert that those values equal a new
content hash. Under the documented in-repository formulas, the independently recomputable typed
roster content hash is
`23677fd135db0f9beee3b8ea3853b48c1c4e924a1250897d97b53c64d7af4e37`.

## Pinned OpenAI-completions adapter

The adapter is bound to the exact upstream `constants/type_mappings.py` and
`model_handler/utils.py` bytes. It reproduces `GORILLA_TO_OPENAPI`, including `dict -> object`,
`any -> string`, nested item conversions, BFCL's float annotation, the root parameter conversion
to `object`, and the `ModelStyle.OPENAI_COMPLETIONS` tool wrapper.

Official dotted names become wire names by replacing each `.` with `_`. Every conversion emits an
explicit official-to-wire and wire-to-official table. A collision such as `a.b` with `a_b` fails
closed; no global underscore-to-dot guess is allowed. Source binding is not a claim that the full
upstream handler was imported or executed, so `upstream_handler_equivalence_attested=false` and
`runtime_execution_attested=false` remain fixed.

## Frozen call plan and barriers

The five arms use the same model and per-call budget, with one provider attempt and no automatic
retry:

| Arm | Calls | Allocation |
|---|---:|---|
| PURE | 8 | one bare-model call per HOLDOUT task |
| STATIC | 8 | one frozen-static-harness call per HOLDOUT task |
| SCORE | 28 | 5 parent FIT + diagnosis + proposal + 5 candidate FIT + 8 GATE + 8 HOLDOUT |
| FULL | 28 | the same task DAG and paired seeds as SCORE; only the feedback projection differs |
| PURE@B | 28 | HOLDOUT sample counts `4,4,4,4,3,3,3,3` |

Total: 100 model calls. Global slots enforce the following order:

1. ten parent FIT calls;
2. two diagnosis calls and two proposal calls;
3. a non-call joint candidate-artifact freeze barrier;
4. ten candidate FIT calls and sixteen GATE calls;
5. a non-call joint selection-artifact freeze barrier;
6. all remaining sixty HOLDOUT calls across PURE, STATIC, SCORE, FULL, and PURE@B.

Thus no HOLDOUT query is executed or scored before both SCORE and FULL selections are frozen.
Every final-60 slot requires the joint selection artifact, and HOLDOUT outcomes cannot continue
search. Invalid candidates consume all frozen call slots through parent fallback and still force
rollback. Seeds use SHA-256-derived coordinates masked to a portable non-negative signed 63-bit
provider domain, consistent with the repository's `derive_seed_v2` portability rule. The exact
typed schedule content hash is
`d92f9061e2baf224d3aea8cbb1d9ca367345ab0a453ce4b106e3e5a8e2dd783e`.

## PURE@B aggregation

Aggregation has no grader or target input. For each nonfailed sample it:

1. restores official names through the explicit reverse table;
2. parses arguments as strict finite JSON objects and emits canonical JSON;
3. sorts calls canonically while preserving duplicate multiplicity;
4. chooses the most frequent normalized output;
5. resolves a tie by the earliest sample in frozen order;
6. treats invalid samples as abstentions and, if all abstain, returns the first frozen sample.

The eight-task HOLDOUT becomes development data after its first inspected result. Reuse must be
labelled accordingly; it cannot retroactively become untouched confirmatory evidence.
