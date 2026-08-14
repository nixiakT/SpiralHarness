# PenguinHarness public self-evolution benchmark

Date: 2026-08-14

> **ICLR evidence boundary:** this is an exploratory, repeated-public-task
> pilot. It has no hidden split or independent search-run replications and
> cannot support a confirmatory benefit, mechanism, or generalization claim.

## Outcome

With the same `dashscope/qwen3-coder-flash` model, five runs per generation,
the same 32,000-token output ceiling, and the exact public Penguin scorer,
SpiralHarness reached **9.6/10** while PenguinHarness remained at **4.0/10**.
That is **+5.6/10**, **+56 percentage points**, or **+140% relative**.  The
same-run pure-model arm scored 3.8/10, so the final evolved harness improved by
5.8/10 over its own untreated arm.

| Arm / generation | Scores | Mean |
|---|---:|---:|
| Pure model, no persistent state | 4, 4, 4, 4, 3 | 3.8/10 |
| Penguin original N | 4, 4, 4, 4, 4 | 4.0/10 |
| Penguin original N+1 | 4, 4, 4, 4, 4 | 4.0/10 |
| Penguin original N+2 | 4, 4, 4, 4, 4 | 4.0/10 |
| Spiral N+1 | 6, 6, 6, 6, 6 | 6.0/10 |
| Spiral N+2 | 9, 10, 9, 10, 10 | **9.6/10** |

The credential-free machine-readable summary is
[`benchmarks/results/penguin-public-qwen3-coder-flash-20260814.json`](../benchmarks/results/penguin-public-qwen3-coder-flash-20260814.json).

## What was actually run

The executable public benchmark is Penguin's
`examples/self-improving-agent/self-evolve-recursive.ts`, pinned to commit
`d14be6fce255cca1cecea3622805c28ad9a5b45a`.  Its source SHA-256 is
`dc46ab7735444876c5f0c20ae5aacc75def2e18051ec87f42617901d6694fa2d`.
Spiral's adapter freezes the exact task, notes, three accepted reports, call
schedule, and a byte-compatible port of its ten-point scorer under protocol
SHA-256 `edbfffe1cc2577fa13124678147e83ca509946c58d81e256ed046f125dfe13de`.

Both trajectories used 17 calls:

1. five N executions;
2. one round-one reflection;
3. five N+1 executions;
4. one recursive round-two reflection;
5. five N+2 executions.

The reflection model never saw the scorer, point labels, or missed criteria.
It saw only a rejected report, accepted reports, and its previous persistent
state.  Task execution and grading remained separate.

## Why Spiral improved and Penguin did not

Penguin asked the model to edit an absolute `AGENTS.md` path through tool calls.
For this model, round one emitted a textual pseudo-call instead of a real tool
call.  Round two made real reads but terminated before writing.  The final file
therefore remained empty (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
The two reflection trace hashes are:

- round 1: `0cdd920292a4a3ebcfbe2a6a284edbd81b7b515d52169accf8d245b24595baea`;
- round 2: `c2f8a34aea48ce4b494d4c1545bf5d4f4322b0d08943ec7d0a41715e4a817502`.

Spiral removes that fragile model-to-filesystem coupling.  Reflection returns
one bounded `<SPIRAL_STATE>` block, and trusted middleware persists the extracted
text as a content-addressed state artifact.  In round two, a generic compiler
finds non-empty lines that are identical at the same position across all
accepted reports and records those literals ahead of the model's fallible
inference.  It does not contain hard-coded ACME, classification, footer, task
answer, or grader rules.

The final run's output-contract guard changed **zero of five** N+2 outputs: the
9.6/10 score came from the learned state being followed by the model, not from
post-hoc answer editing.

## Observable self-evolution and rollback

The final independent trajectory was monotonic: **3.8 → 6.0 → 9.6**.  Both
candidates beat their parent and were promoted.  An earlier run was
**4.0 → 7.0 → 6.4**; the second candidate regressed and the validation gate
rejected it.  This supplies evidence for both required directions of the loop:
successful promotion and safe rollback.

## Reproduction

After placing the LiteLLM endpoint and key in environment variables (the key is
never written into artifacts), run:

```bash
python -m spiral_harness benchmark penguin-public \
  --model dashscope/qwen3-coder-flash \
  --penguin-source /path/to/penguin-harness/examples/self-improving-agent/self-evolve-recursive.ts \
  --runs-per-generation 5 \
  --max-output-tokens 32000 \
  --output runs/penguin-public/replication
```

The final local content-addressed result artifact is
`1c669101d4e1b5a567af6d4c8b1e59d34b7169d80d4a41f7879c940d8f90d7aa`.

## Claim boundary

Penguin's landing-page data-analysis result (10/15, 66.67%) and coding result
(57/80, 71.25%) are only aggregate numbers.  At the pinned revision, its README
still lists “Public release of the benchmark suite” as unfinished.  The 15/40
cases, gold outputs, and graders cannot be audited or run, and the landing page
uses different models for Penguin, Claude Code, and Codex.  We therefore do not
claim to have run or beaten that unreleased suite.

The public recursive example also repeats one task and has no hidden test
split.  This result is a mechanistically informative same-model pilot, not an
estimate of cross-task generalization or a confirmatory mechanism effect.
Penguin's README example of 9.8/10 used
`qwen3.6:35b`; comparing that number directly with this model would violate the
same-model requirement.
