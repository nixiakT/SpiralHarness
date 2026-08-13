# Same-Model Self-Evolution Results

This study compares three arms with the same hosted model
`dashscope/qwen3-coder-flash`, temperature 0, and a 128-token output ceiling:

1. **Pure**: task prompt and label/output contract only.
2. **Baseline harness**: a fixed prefix of labeled training examples.
3. **Self-evolved harness**: query-aware TF-IDF retrieval selected using validation results.

The datasets and split hashes are pinned to Meta-Harness commit
`44b9942127847f7421db70d8c7e48407f09a3c70`. Candidate selection accepts validation artifacts
only and fails closed on test artifacts. The frozen candidates were evaluated on complete test
splits without using test errors for further evolution.

## Complete test results

| Benchmark | N | Pure | Fixed baseline | Self-evolved | Evolved − pure | Evolved − baseline |
|---|---:|---:|---:|---:|---:|---:|
| Symptom2Disease | 212 | 58.49% | 72.17% | **84.43%** | **+25.94 pp** | **+12.26 pp** |
| LawBench | 100 | 3.00% | 11.00% | **29.00%** | **+26.00 pp** | **+18.00 pp** |

Paired item-bootstrap 95% intervals (20,000 resamples, seed 42):

- Symptom2Disease: evolved − pure `[+19.34, +32.55]` pp; evolved − baseline
  `[+6.60, +17.92]` pp.
- LawBench: evolved − pure `[+18.00, +35.00]` pp; evolved − baseline
  `[+10.00, +26.00]` pp.

Thus the point-estimate improvement over both the pure model and an ordinary fixed-example harness
exceeds 10 percentage points on both benchmarks. The lower confidence bound exceeds 10 points for
the pure-model comparison on both benchmarks; the stricter evolved-versus-baseline lower bound is
6.60 points on Symptom2Disease and exactly 10.00 points on LawBench.

## Self-evolution evidence

On validation, the three arms scored:

- Symptom2Disease: pure 54%, fixed baseline 64%, evolved 82%.
- LawBench: pure 4%, fixed baseline 10%, evolved 32%.

The validation-only selector chose the evolved candidate in both cases and explicitly rejected the
other candidates. Immutable selection traces:

- Symptom2Disease: `d516ba8b9cabff55d3603ad4cc3e08d2d4b648fe05469bdf0b11ed357fcae7f4`
- LawBench: `54216d5b43cf28d4a666aa61af83f8891c7b91eca8d710b3368d047b20aab38c`

Complete-test result artifacts:

| Benchmark | Pure | Fixed baseline | Self-evolved |
|---|---|---|---|
| Symptom2Disease | `2589b115…60207b` | `a02fc1b7…040879` | `f5e24a2a…b81991` |
| LawBench | `e8c8a500…196e24` | `d4029521…efc6c5b` | `a2378624…169ea` |

The artifacts contain item-level public benchmark targets and predictions, so they remain outside
Git. Their full SHA-256 values are recoverable from the experiment output directories documented
in the run handoff. API credentials are absent from prompts, artifacts, logs, and the repository.

## Interpretation

These are controlled same-model comparisons: the gain cannot be attributed to switching to a
stronger solver model. The evolved mechanism is currently deterministic retrieval/memory evolution
over a frozen candidate family, followed by validation-only champion selection. It demonstrates
automatic evidence-based candidate acceptance and rejection; it is not yet open-ended LLM code
generation of arbitrary harness programs.
