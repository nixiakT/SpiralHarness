# Native Meta-Harness Comparison

This audit answers two separate questions:

1. What harness does Meta-Harness actually discover and run?
2. With the solver fixed to `dashscope/qwen3-coder-flash`, does Spiral improve over the pure
   model and a reconstruction of Meta-Harness's published classifier?

The comparison pins upstream commit `44b9942127847f7421db70d8c7e48407f09a3c70` and uses its
native data loaders, task prompts, `inner_loop.py`, `MemorySystem` lifecycle, and graders. Local
runtime edits passed `LITELLM_API_KEY` into its existing `LLM` constructor and retained the
already-computed item predictions in external result payloads; they did not alter prompts,
examples, predicted values, or grading.

## What Meta-Harness is

Meta-Harness is an outer program-search harness, not a fixed retrieval prompt. For text
classification its proposer is Claude Code with Opus 4.6 and maximum reasoning. At each iteration
the proposer can inspect prior candidate source, scores, and raw traces, then writes an executable
Python `MemorySystem` implementing:

```python
predict(input)
learn_from_batch(batch_results)
get_state()
set_state(state)
```

The paper ran 20 iterations with two candidates per iteration and fixed the inner solver to
GPT-OSS-120B. The cleaned public skill currently asks for three candidates per iteration, so the
release defaults are not identical to the reported run.

The paper's highest-average discovered classifier is **Label-Primed Query**:

1. list all observed labels;
2. retrieve one query-relevant example per label;
3. add query-local pairs whose examples have different labels;
4. use TF-IDF with query-anchored partner selection;
5. make one large solver call.

The lower-context **Draft Verification** variant first predicts from five neighbors, then retrieves
five same-label confirmers and five different-label challengers for a second call.

## Source-availability boundary

The paper names `label_primed_query_anchored.py` and `draft_verification.py`, but neither file is in
the public tree, remote branches, tags, reachable Git history/objects, or arXiv source package. The
public repository contains only `no_memory.py`, `fewshot_all.py`, and `fewshot_memory.py` as fixed
classification agents. Therefore:

- native baselines below are official released code;
- data, runtime, and grading are official released code;
- `Label-Primed Query` is a **paper-algorithm reconstruction**, not an official-source
  reproduction;
- Spiral candidates are our own experiments.

The reconstruction in
[`meta_harness_paper.py`](../spiral_harness/benchmark/meta_harness_paper.py) discloses every paper-
unspecified choice, including Unicode tokenization, tie breaking, contrastive-pair count, and a
100K-character memory budget.

## Controlled protocol

- Solver: `dashscope/qwen3-coder-flash` for every arm.
- Temperature: 0.
- Provider output ceiling under the released API-base path: 4,096 tokens.
- Training: upstream offline mode, one epoch, ground truth visible to `learn_from_batch`.
- Selection: official validation splits only for the first frozen candidates.
- Test: complete official splits.
- Sizes: S2D 200/50/212; LawBench 200/50/100; USPTO 50/30/100.
- Credentials: injected only through the process environment; absent from Git, prompts, result
  artifacts, and logs.

## Native validation results

| Dataset | Pure | Few-shot all | Paper reconstruction | First frozen Spiral |
|---|---:|---:|---:|---:|
| Symptom2Disease | 58% | 58% | 76% | **88%** (score-band router) |
| LawBench | 32% | 14% | 42% | **54%** (taxonomy router) |
| USPTO | 3.33% | 3.33% | 3.33% | **36.67%** (reaction-aware compiler) |

Later validation-only experiments reached 56% on LawBench by adding compact local evidence. On
S2D, a grouped-memory/local/NB plurality reached 90%, but it was tested only after the public test
had already been exposed and is not a fresh held-out result. USPTO initially had no candidate to
advance; a later task-structure audit identified that its outputs are unique precursor sets rather
than repeated class labels. Reaction-family retrieval plus validation-selected high-confidence
reverse-reaction templates then reached 11/30 and was frozen before test.

## Complete test results

| Dataset | N | Pure | Few-shot all | Paper reconstruction | First frozen Spiral | Adaptive best seen |
|---|---:|---:|---:|---:|---:|---:|
| Symptom2Disease | 212 | 62.26% | 69.81% | 84.43% | 81.13% | **88.68%** local evidence |
| LawBench | 100 | 40.00% | 17.00% | 45.00% | **55.00%** | 55.00% |
| USPTO | 100 | 0.00% | — | 2.00% | **14.00%** | — |

The S2D first frozen router failed to generalize: 88% validation became 81.13% test and lost to the
paper reconstruction. Once that test result was observed, later S2D runs became adaptive public-
benchmark experiments. The 88.68% local-evidence result is useful engineering evidence, but it is
not an untouched-test estimate and must not be reported as one.

LawBench is cleaner: the first frozen candidate scored 55%, ten points above the paper
reconstruction and fifteen above the pure model. It changed ten paper-reconstruction errors to
correct answers and introduced no regressions.

USPTO is the cleanest threshold result. Candidate source and validation evidence were committed as
`e377684` before the test was opened. The frozen candidate scored 14%, fourteen points above pure
and twelve above the paper reconstruction. Relative to the reconstruction it corrected 13 errors,
regressed on one item, and shared one correct item. Its paired bootstrap 95% interval is
`[+5, +19]` points; versus pure it is `[+8, +21]` points.

For reference, the paper reports GPT-OSS-120B test scores of 14.0% USPTO, 86.8% S2D, and 45.0%
LawBench for its selected harness. Those numbers use a different solver and do not form a
same-model comparison with this study.

## Claim boundary

The native results support these claims:

- relative to the pure model, the strongest observed Spiral configuration improves S2D by
  26.42 points, frozen LawBench by 15 points, and frozen USPTO by 14 points;
- the corresponding paired bootstrap 95% intervals are `[+19.81, +33.02]` and
  `[+8.00, +23.00]` for S2D and Law, plus `[+8.00, +21.00]` for USPTO;
- relative to the paper reconstruction, adaptive S2D improves by 4.25 points and frozen LawBench
  by exactly 10 points, while frozen USPTO improves by 12 points;
- USPTO therefore provides a same-model, pre-test-frozen result exceeding the paper reconstruction
  by more than ten percentage points.

For adaptive S2D, the descriptive paired difference interval is `[+0.47, +8.49]` points, but it
does not account for candidate selection after test exposure and is not a valid confirmatory
interval. For frozen LawBench versus the reconstruction it is `[+5.00, +16.00]` points. The frozen
USPTO interval is confirmatory for the fixed split and candidate, subject to the small benchmark's
usual external-validity limits.

## Self-evolution evidence

The validation trajectory demonstrates executable mechanism search rather than model switching:

| Candidate | S2D | Law | USPTO | Decision |
|---|---:|---:|---:|---|
| Paper Label-Primed reconstruction | 76% | 42% | 3.33% | comparison arm |
| Pure-draft verification | 62% | 46% | — | retain only for sparse taxonomy |
| Compact local evidence | 84% | 42% | — | S2D candidate |
| Score-band / taxonomy routers | 88% | 54% | — | first frozen winners |
| Local-majority / local taxonomy | 88% | 56% | — | adaptive candidates |
| Grouped-memory plurality | 90% | — | — | adaptive S2D candidate |
| Reaction-family retrieval | — | — | 3.33% | diagnose near-miss outputs |
| Retrieval + reaction compiler | — | — | **36.67%** | frozen USPTO winner |

The mechanism evolved from global per-label coverage toward compact query evidence, preservation of
the model-only prior for unseen LawBench labels, output-contract cleanup, and candidate routing.
For USPTO it changed task representation entirely: unique SMILES answers stopped being treated as
class labels, same-family structural analogues reduced distraction, and deterministic templates
handled only validation-supported high-confidence transformations. A post-test decomposition found
the retrieval draft correct on 10/100; the compiler fixed five errors and introduced one regression,
for a net 14/100. This is bounded evidence-derived evolution; it is not yet unrestricted LLM
generation of arbitrary harness source like Meta-Harness's Opus proposer.

## Context and artifacts

The paper reconstruction used about 8.6K additional characters per S2D query and 94.2K per LawBench
query. Compact local S2D used about 3.5K. Multi-call Spiral token totals cannot be compared from the
saved usage counters because repeated candidates intentionally reused the upstream prompt cache.
On USPTO, reaction-aware Spiral used about 2.0K additional characters and 118,946 total tokens,
versus 18.4K characters and 1,008,826 tokens for the paper reconstruction.

| Artifact | SHA-256 |
|---|---|
| S2D pure test | `6eacffdb72d9ae1c55c67993864284b579fcfb55bc801bdaefedb114175b9b9b` |
| S2D paper reconstruction test | `98eafa811d00211c5f29b70e0af588b2b78df19a46600543daeb917229a078af` |
| S2D adaptive local test | `0b407dbab544c73e4cde02878d50708cab2e59319cfe56c76ac4765a7f28bf59` |
| Law pure test | `3206c5993233c3a43d2a2d11c7a855869fc4eb1edf382e29829354e4f7507ef7` |
| Law paper reconstruction test | `b922f039a3c5dcc74a241e3ca26b3470f84fe98e0674a6e956df5d1ceac96fc7` |
| Law first frozen test | `820d48c9383013f54affe548f33e66aabb8e252965047644a50e85449b79454e` |
| USPTO pure test | `a9b01f8c5d5ec87b09b712d0996d641623da688ed487f1c329789254136062bc` |
| USPTO paper reconstruction test | `0c3a1ec6aac80590844e86ac02e3ba1b2ad6fcd2b2295ea80f6a5a92e9f0358e` |
| USPTO first frozen test | `500512f2f853b243a4990b9fd08a0bbcd0e3800450e4c3c6714099d5b14a112a` |

Item-level artifacts contain public benchmark prompts, targets, and predictions and remain outside
Git. The committed machine-readable summary contains only aggregate evidence.

## Reproduction adapter

[`spiral_native.py`](../benchmarks/meta_harness_native/spiral_native.py) is a credential-free adapter
template for the upstream native runner. Copy it into upstream `agents/`, put both repositories on
`PYTHONPATH`, and select `paper`, `retrosynthesis`, `local`, `score-band`, or `taxonomy` through
`SPIRAL_META_HARNESS_MODE`. The upstream release must also pass `os.environ.get("LITELLM_API_KEY")`
to its existing `LLM` constructor when using an authenticated custom API base. No credential should
be written into either repository or an artifact.
