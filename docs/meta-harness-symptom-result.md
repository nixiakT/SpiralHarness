# Meta-Harness Symptom2Disease Result

> **Protocol note:** this earlier result uses Meta-Harness data and label semantics through a
> Spiral benchmark adapter; it does not run the upstream `MemorySystem` prompt wrapper or native
> `inner_loop.py`. See the [native same-runtime comparison](meta-harness-native-comparison.md).

This experiment reproduces the public Meta-Harness text-classification task at commit
`44b9942127847f7421db70d8c7e48407f09a3c70`. It uses the same complete
Symptom2Disease test split and exact-match label semantics, with a training-free retrieval
harness built only from the official training split.

## Frozen selection protocol

- Search data: the complete 50-item validation split only.
- Frozen candidate: TF-IDF retrieval with `k=16`, training-derived label profiles, temperature
  zero, and `dashscope/qwen3-coder-flash`.
- Gate data: the complete 212-item test split, evaluated once after selection.
- The test result was not used to modify or rerun this candidate.

Pinned source hashes:

- train: `b41e494b40bf971a1eadd79ed4dda527640916abe0a2dc94e047e13ef293ee4c`
- validation: `f6ad74bea9fb59f4a3abf930668042c33fbb7519bc9e2568a9758c2a0b1277e8`
- test: `6a6b224665715f40d42a46eed82498698d39fa7689241eaae2cc2276e444671d`

## Result

| System | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Public MCE reference | — | — | 83.00% |
| SpiralHarness frozen retrieval candidate | 179 | 212 | **84.43%** |

The observed difference is +1.43 percentage points. The immutable result artifact has SHA-256
`f5e24a2ab767e53ea5b5df96e33cb627d904f5571c9afe8cc2858e1b81991b49`; it records all 212
indices, predictions, targets, response hashes, and 325,652 provider-reported tokens. Independent
recomputation produced 179 exact matches, 212 unique indices spanning 0 through 211, and the same
token total.

## Scope of the claim

This is a same-data, same-split, same-label-grading comparison to the public 83.0% MCE reference,
not an official Meta-Harness leaderboard submission. The public reference reports Qwen3.5-27B,
whereas this run uses the hosted `dashscope/qwen3-coder-flash`; therefore the result establishes
that this SpiralHarness configuration exceeds the published score under the reproduced task
protocol, but does **not** isolate harness benefit under a same-model comparison.

The artifact is intentionally not committed because it contains public benchmark item-level
predictions and targets. Reproduction requires the pinned upstream repository and a configured
OpenAI-compatible endpoint; `spiral benchmark meta-harness-symptom --help` documents the runner.
