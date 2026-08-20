# BFCL V4 fresh-development multi-agent comparison

Date: 2026-08-20

> **Result boundary:** this is a nine-task public-development comparison, not an
> official BFCL leaderboard score, hidden-test evidence, or a confirmatory
> self-evolution result. The Harness won one paired task and lost none, but the
> sample is too small and already opened for development. The result is
> explicitly `reportable_result=false`.

## Bottom line

The same provider-declared `dashscope/qwen3-30b-a3b` route was evaluated on the
five FIT and four GATE tasks from the fresh BFCL V4 public-development v2
roster. These tasks do not overlap the earlier 15-task pilot roster. The 16 v2
HOLDOUT tasks were not opened.

| Arm | Correct | Accuracy | Model calls |
|---|---:|---:|---:|
| single bare-model sample | 6 / 9 | 66.67% | 9 scored of 27 captured |
| three-sample budget-matched bare model | 6 / 9 | 66.67% | 27 |
| Analyst → Critic → Coordinator Harness | **7 / 9** | **77.78%** | 27 |

Against the call-count-matched three-sample baseline, the Harness had one win,
eight ties, and zero losses: a descriptive `+1/9 = +11.11` percentage-point
difference. The win was `parallel_multiple_2`. All three bare-model samples
were identical within every task, so modal aggregation did not change the
single-sample prediction.

This is encouraging mechanism evidence for the new multi-agent execution
path, not evidence of general improvement. There are only nine opened public
tasks, no independent replications, no confidence interval, and no untouched
evaluation split.

## Matched construction

For every task, the experiment derived three provider seeds. The bare-model arm
used all three seeds as independent direct solutions and selected their frozen
label-free modal response. The Harness used the same three seeds for its
ordered turns:

1. `Analyst` produced a candidate BFCL call array;
2. `Critic` audited function choice, arguments, values, and ordering;
3. `Coordinator` received both declared dependencies and emitted the final
   call array.

Both arms therefore used 27 model calls under the same frozen model and
inference configuration. They were call-count matched, not token matched:

| Arm | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|
| three-sample bare model | 7,797 | 1,097 | 8,894 |
| multi-agent Harness | 20,658 | 6,367 | 27,025 |

The Harness consumed more context and intermediate output by design. A future
claim-oriented study must add a token-matched or cost-matched baseline rather
than interpreting this call-matched comparison as an efficiency gain.

## Per-task outcomes

| Split | Task | Bare single | Bare ×3 modal | Harness |
|---|---|:---:|:---:|:---:|
| FIT | `simple_python_198` | ✓ | ✓ | ✓ |
| FIT | `simple_python_162` | ✓ | ✓ | ✓ |
| FIT | `multiple_30` | ✓ | ✓ | ✓ |
| FIT | `parallel_24` | ✓ | ✓ | ✓ |
| FIT | `parallel_multiple_2` | ✗ | ✗ | ✓ |
| GATE | `simple_python_179` | ✓ | ✓ | ✓ |
| GATE | `multiple_71` | ✓ | ✓ | ✓ |
| GATE | `parallel_26` | ✗ | ✗ | ✗ |
| GATE | `parallel_multiple_52` | ✗ | ✗ | ✗ |

## Execution and evidence closure

All 54 registered model calls completed. The provider declared
`dashscope/qwen3-30b-a3b` on every response. This is an internally consistent
observation, not an attestation that the same weights were served.

Every call crossed the receipt-backed `FixedModelRunner` and append-only
attempt ledger. The result root directly references all 54 model executions,
all nine immutable multi-agent transcripts, and the answer-isolated FIT/GATE
grade projections. A fresh verifier reloads these artifacts, checks paired
seeds, model specs, task/turn identities, transcript ownership, grade
projections, unique execution references, aggregate scores, and exact token
accounting.

The verified result root is:

```text
896130ed192dff957363afd3012bf9bb0b8703cf093effcc952d61d16f3a156d
```

The credential-free aggregate projection is stored at
[`benchmarks/results/bfcl-v4-multi-agent-qwen3-30b-20260820.json`](../benchmarks/results/bfcl-v4-multi-agent-qwen3-30b-20260820.json).
The full local CAS remains under
`runs/bfcl-v4-multi-agent-qwen3-30b-20260820/artifacts` and is intentionally
excluded from Git because it contains complete public-development prompts and
model outputs.

## Route selection and excluded attempts

The preferred open-source route `qwen36-35b-a3b` was advertised but became
unresponsive during execution. Its incomplete attempt was stopped and retained
locally under an explicitly aborted run directory. An initial Qwen3 fallback
attempt used provider-incompatible 63-bit seeds; every request failed closed
and that run was also excluded. The protocol now uses deterministic 31-bit
seeds and requires exactly 54 consumed calls before a result can be created.

Neither excluded attempt contributed a response, grade, task selection, or
score to the result above.

## Claim limits

- public development only;
- FIT and GATE are now opened and cannot become untouched evidence later;
- HOLDOUT remained closed;
- no official full-suite or hidden-test evidence;
- no weight-revision attestation;
- call-count matched but not token- or cost-matched;
- no statistical-significance claim;
- no autonomous mutation or promotion occurred in this comparison;
- `reportable_result=false`.

The next valid step is to use these opened tasks only for diagnosis, freeze any
instruction change, and evaluate exactly once on a new prospectively isolated
roster with both call- and token-matched controls.
