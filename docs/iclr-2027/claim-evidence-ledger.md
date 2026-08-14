# Claim--evidence ledger

Status: live ledger, initialized 2026-08-14.  Claims default to blocked.  A
green test suite or a cryptographically linked artifact proves implementation
properties only; it does not by itself prove scientific benefit or causality.

| ID | Candidate statement | Required evidence | Current status | Allowed wording now |
|---|---|---|---|---|
| C1 | The verification kernel fails closed on malformed, stale, or mismatched evidence. | Unit/integration adversarial tests and replay of exact typed artifacts. | Supported as an implementation claim; CI completion pending for the latest commit. | “The implementation enforces the tested contracts.” |
| C2 | Spiral improves same-model benchmark utility. | Frozen same-model comparison with disclosed split status and paired records. | Exploratory pilots exist for LawBench, S2D, USPTO, and one Penguin public task. Their test data are exposed and search replication is insufficient. | “Exploratory pilot,” followed by exact scope and caveats. |
| C3 | Mechanism evidence improves promotion over score-only evidence. | Budget-matched FULL versus SCORE across independent search runs; sealed outcomes; shuffled-evidence placebo. | Blocked: no reportable live FULL/SCORE study. | State as RQ/H1 only. |
| C4 | The same model self-evolves its harness. | The same frozen model family closes execute--diagnose--propose--select without manual candidate choice, with every call and cost attested. | Blocked: current automatic multi-condition path is a non-reportable replay fixture. | “The architecture targets same-model self-evolution.” |
| C5 | The gate reduces false or harmful promotion. | At least 59 independent null/unrepairable fault families for a zero-event one-sided 95% upper bound below 5%, plus repairable faults and real-task validation. | Blocked: controlled unit fixtures are not a powered fault benchmark. | State metric and target only. |
| C6 | The method exceeds the pure model or a competitor by more than 10 percentage points. | Predeclared comparison, same model/runtime/budget, independent search seeds, untouched sealed set, uncertainty interval, and multiplicity control. | Some exploratory point differences exceed 10 pp, but no confirmatory claim qualifies. | Do not use as an abstract or headline result. |
| C7 | Spiral outperforms Meta-Harness. | Official executable baseline or a clearly named reconstruction, same task/model/budget, multiple search runs, untouched sealed split. | Blocked. Existing code reconstructs paper-described algorithms and cannot establish an official-system win. | “Compared with a paper-algorithm reconstruction.” |
| C8 | Results generalize across models and harness surfaces. | At least three preregistered open model configurations and multiple task/surface families with equal-cell analysis. | Blocked: current pilots use one model and mostly prompt/retrieval changes. | Do not claim cross-model or full-harness generality. |
| C9 | The sealed evaluator is independent of candidate code. | Out-of-process evaluator, hidden targets, asymmetric signing/KMS, sandbox, and one atomic release. | Blocked: process-local HMAC and logical capabilities are not a hostile-code isolation boundary. | “Logical in-process separation” only. |

## Existing pilot inventory

All values below are development evidence and must remain outside the
confirmatory primary table.

| Benchmark | Fixed model | Spiral | Pure model | Other comparison | Boundary |
|---|---|---:|---:|---:|---|
| USPTO retrosynthesis | `dashscope/qwen3-coder-flash` | 14/100 | 0/100 | paper-algorithm reconstruction 2/100 | Frozen candidate before this evaluation, but the compiler is validation-guided and manually engineered; one search trajectory. |
| LawBench | same | 55% | 40% | reconstruction 45% | Test has been opened; one model/run. |
| Symptom2Disease | same | 88.68% | 62.26% | reconstruction 84.43% | Test was used after an earlier 81.13% result; exploratory only. |
| Penguin public task | same | 9.6/10 | not a matched pure arm | Penguin 4.0/10 | One repeated public example, no hidden split. |

The numbers are retained to prevent selective forgetting, not to license a
claim.  New tuning prompted by them starts a new development lineage and must
use a new sealed set for confirmation.

## Evidence promotion rule

To change a scientific claim to `supported`, the evidence entry must bind:

1. preregistration commit and amendment history;
2. exact model, inference, runtime, grader, task-group and price-table hashes;
3. every independent search-run closure, including failures;
4. one atomic sealed authorization and signed aggregate release;
5. prespecified analysis output, confidence interval and multiplicity result;
6. complete solver/proposer/diagnoser/grader usage;
7. a reproducible figure/table source and an explicit limitation statement.

Any opened sealed item, model revision drift, missing condition, post-hoc
endpoint change, or asymmetric rerun marks that lineage invalid rather than
silently downgrading it to a favorable complete-case analysis.
