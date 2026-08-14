# Claim--evidence ledger

Status: live ledger, initialized 2026-08-14.  Claims default to blocked.  A
green test suite or a cryptographically linked artifact proves implementation
properties only; it does not by itself prove scientific benefit or causality.

| ID | Candidate statement | Required evidence | Current status | Allowed wording now |
|---|---|---|---|---|
| C1 | The verification kernel fails closed on malformed, stale, or mismatched evidence. | Commit-scoped adversarial tests and independent replay of every exact schedule, preflight, receipt, ledger, attestor, and terminal ref used by the named claim. | Partially supported for tested contracts; a green suite does not establish a blanket or cross-process trust-closure claim. | “The implementation enforces these enumerated, tested contracts.” |
| C2 | Spiral improves same-model benchmark utility. | Frozen comparison in which the exact provider-resolved model identity/revision fills every model-mediated role, with disclosed split status, both PURE and STATIC references, independent search runs, and paired records. | Exploratory pilots exist for LawBench, S2D, USPTO, and one Penguin public task. Their test data are exposed, the reportable PURE execution closure is pending, and search replication is insufficient. | “Exploratory pilot,” followed by exact scope and caveats. |
| C3 | The joint FULL policy improves over SCORE. | Budget-matched MM versus SS across independent search runs, one untouched sealed release, Holm-corrected one-sided test of the frozen margin, and protected/cost closure. | Blocked: no reportable live FULL/SCORE study. | State as the primary deployable-policy RQ only. |
| C4 | The same model self-evolves its harness. | One exact frozen model identity/revision closes every model-mediated execute--diagnose--propose--materialize--rank--nominate role without manual choice or stronger teacher, with every role call and cost attested; a separate deterministic authority performs final statistical promotion. | Blocked: current automatic multi-condition path is a non-reportable replay fixture. | “The architecture targets same-model self-evolution.” |
| C5 | The gate reduces false or harmful promotion. | At least 59 independent, exchangeable null/unrepairable fault families from a frozen sampling target for a zero-event one-sided 95% upper bound below 5%, plus repairable faults and real-task validation. | Blocked: the deterministic seven-family v4 fixture is not a powered fault benchmark. | State metric, sampling assumptions, and target only. |
| C6 | The method exceeds the pure model or a competitor by more than 10 percentage points. | Prospectively binary success-probability scale, predeclared claim family, exact matched model/runtime/task coordinates, independent search seeds, untouched sealed set, single-call PURE plus model-call/token-budget-matched PURE@B with a total frozen aggregation rule per task, and every simultaneous multiplicity-adjusted one-sided lower confidence bound strictly above 0.10. | Some exploratory point differences exceed 10 pp, but no confirmatory claim qualifies. | Do not use as an abstract or headline result. |
| C7 | Spiral outperforms Meta-Harness. | Official executable baseline or a clearly named reconstruction, same task/model/budget, multiple search runs, untouched sealed split, and exact role-model mapping. | Blocked. Existing code reconstructs paper-described algorithms and cannot establish an official-system win; a stronger external proposer is a heterogeneous baseline. | “Compared with a paper-algorithm reconstruction”; do not use it for a same-model or `>10 pp` claim. |
| C8 | Results generalize across models and harness surfaces. | At least three preregistered open model configurations and multiple task/surface families with equal-cell analysis. | Blocked: current pilots use one model and mostly prompt/retrieval changes. | Do not claim cross-model or full-harness generality. |
| C9 | The sealed evaluator is independent of candidate code. | Out-of-process evaluator, hidden targets, asymmetric signing/KMS, sandbox, and one atomic release. | Blocked: process-local HMAC and logical capabilities are not a hostile-code isolation boundary. | “Logical in-process separation” only. |
| C10 | Mechanism feedback and mechanism-gated promotion have separately identified effects. | Powered `SS/MS/SM/MM` controlled-fault factorial, isolated contexts, counterbalanced dispatch, runtime-attested common fresh blocks, and simultaneous feedback/gate/interaction intervals. | Blocked: the factorial is a design amendment; manifest equality is not execution evidence. | Describe `FULL - SCORE` only as a joint policy contrast. |
| C11 | HarnessFaultBench is a multi-family powered benchmark. | Frozen generator/design matrix and sampling target, at least the powered repair/null-family roster, independent search runs, live model executions, and sealed external-validity results. | Blocked: v4 is a deterministic seven-family/six-surface trust-closure fixture; its 28 scenarios per partition are not 28 independent families, and base-model output does not drive behavior. | “Multi-surface deterministic implementation fixture” or “prospective powered benchmark design.” |
| C12 | Four conditions consume every fresh gate block atomically. | One runtime campaign closure binding the block, all condition states, schedules, preflights, receipts, ledger tails, failures, and one-time consumption. | Blocked: current manifests declare a planned quartet while runtime/campaign-attestation flags remain false. | “Manifest-planned atomic quartet,” never “atomically executed.” |
| C13 | SCORE feedback and its decision are fully source-attested. | Interval level/endpoints/estimator and score observations in the independent objective attestation, plus exact schedule/preflight/live-ledger receipt replay. | Blocked until caller-supplied interval endpoints and weaker receipt paths are absent from the reportable pipeline. | “Prospective trusted projection design”; do not claim an attested performance decision. |
| C14 | Spiral exceeds HarnessOpt-Bench or Evo-Bench. | Same official benchmark implementation, exact target/policy model, seed harness, disclosure policy, split, optimizer budget, independent search replication, held-out scoring, and uncertainty analysis. | Blocked: only the papers were audited, no public executable repository was located by the 2026-08-14 literature-audit cutoff, and cross-protocol numbers are not comparable. | Cite them as paper-only related-work benchmarks; do not claim a reproduced or numerical win. |
| C15 | The formal analysis and power pipeline can freeze `S` or produce confirmatory evidence. | Coverage-calibrated frozen estimator and margin tests, independent boundary-null audit, closed disjoint pilot, authority-attested sealed inputs, dependency/failure stress tests, and exact code/environment hashes. | Blocked: the equal-cell basic nested bootstrap and prospective simulator are method-development prototypes; reportability and formal-design flags remain false. | “Method-development prototype” or “conditional synthetic diagnostic” only. |

## Existing pilot inventory

All values below are development evidence and must remain outside the
confirmatory primary table.

| Benchmark | Fixed model | Spiral | Pure model | Other comparison | Boundary |
|---|---|---:|---:|---:|---|
| USPTO retrosynthesis | `dashscope/qwen3-coder-flash` | 14/100 | 0/100 | paper-algorithm reconstruction 2/100 | Frozen candidate before this evaluation, but the compiler is validation-guided and manually engineered; one search trajectory. |
| LawBench | same | 55% | 40% | reconstruction 45% | Test has been opened; one model/run. |
| Symptom2Disease | same | 88.68% | 62.26% | reconstruction 84.43% | Test was used after an earlier 81.13% result; exploratory only. |
| Penguin public task | same | 9.6/10 | contemporaneous pure-model reference 3.8/10 | Penguin 4.0/10 | One repeated public example; no preregistered matched search-run design, independent replication, or hidden split. |

The numbers are retained to prevent selective forgetting, not to license a
claim.  New tuning prompted by them starts a new development lineage and must
use a new sealed set for confirmation.

## Evidence promotion rule

To change a scientific claim to `supported`, the evidence entry must bind:

1. preregistration commit and amendment history;
2. exact model, inference, runtime, grader, task-group and price-table hashes;
3. every independent search-run closure, including failures;
4. one atomic sealed authorization and signed aggregate release;
5. prespecified margin-test output and multiplicity result, plus simultaneous
   claim-family one-sided lower bounds for any `>10 pp` statement;
6. complete solver/proposer/diagnoser/grader usage;
7. a reproducible figure/table source and an explicit limitation statement.

Any opened sealed item, model revision drift, missing condition, post-hoc
endpoint change, or asymmetric rerun marks that lineage invalid rather than
silently downgrading it to a favorable complete-case analysis.
