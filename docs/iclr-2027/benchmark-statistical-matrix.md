# ICLR 2027 benchmark and statistical execution matrix

Status: implementation audit and prospective execution plan, 2026-08-14. This
document is not a preregistration, a power result, or evidence that any blocked
cell ran. It turns the intended study into a fail-closed work matrix without
inventing a replication count.

The governing protocol remains the
[prospective main-study protocol](preregistered-main-study.md). The scientific
claim boundary is in the [claim--evidence ledger](claim-evidence-ledger.md), and
the intervention semantics are in the
[mechanism-factorial design](mechanism-factorial-design.md). If this matrix and
one of those documents disagree, study freeze is blocked until a prospective
amendment resolves the conflict; requirements from different statistical rules
must not be improvised into a new analysis.

## 1. Reading the matrix

### 1.1 Status vocabulary

Every experimental cell below records both implementation readiness and
confirmatory status:

- `I-dev / B-conf`: a development-only path can execute, but confirmatory use is
  blocked.
- `I-contract / B-conf`: typed design or replay contracts exist, but no
  reportable live runner and authority closure exist.
- `P / B-conf`: only part of the benchmark, arm, or control is implemented.
- `B`: the required implementation or external artifact does not exist locally.

No cell is currently confirmatory-ready. `B-conf` cannot be cleared by a unit
test, a replay fixture, a provider-declared model string, or an unsigned local
result.

### 1.2 Repeated-unit notation

The matrix uses symbols instead of fabricated power numbers:

- `S*` is the number of independent, complete end-to-end search-seed runs
  selected by the frozen power procedure.
- `R*` is the number of paired rollout repeats inside a search run. It improves
  measurement but is **not** statistical `n` for an algorithmic claim.
- `G[t]*` is the frozen item-side source/template/repository group roster for
  task family `t`.
- `F-R*` and `F-N*` are the powered repairable and null/unrepairable controlled-
  fault family rosters.
- `K[B]` is the number of provider-minimal samples that fit FULL's frozen
  ex-ante call/token ceiling for `PURE@B`. Its samples are not independent
  search runs.

The final design must satisfy both its structural roster and every corresponding
count selected by a coverage-audited, disjoint-pilot power analysis. Until that
analysis closes, there is no defensible numeric recommendation for `S*`, `R*`,
`G[t]*`, `F-R*`, or judge-calibration size. One complete run is enough to test
wiring only and is not an inferential minimum.

### 1.3 Required artifact register

The identifiers in later tables refer to these required content-addressed
artifacts. A file with the right name is insufficient; the final release must
verify every digest and join.

| ID | Required artifact and clearing condition |
|---|---|
| `M` | Frozen model-roster manifest with exactly three eligible open-model configurations spanning at least two families; model-card/weight licence evidence; provider route; exact provider-resolved identity and immutable served revision; tokenizer; role-specific inference settings; tool policy; timeout; and normalized price-table digest. One configuration fills every model-mediated role within a run. |
| `B` | Benchmark manifest with source revision and hashes, licence/redistribution decision, complete task and group roster, exploration/fresh-gate/sealed partition commitments, deterministic binary-success rubric, official secondary metric, weights, retry unit, and grader/container digest. |
| `A` | Seven-arm real-task design over `PURE`, `STATIC`, `Random-valid`, `Prompt-only`, `SCORE`, `FULL`, and `PURE@B`, including independent search seeds, rollout coordinates, mutation grammar, stopping limits, complete role-call topology, and ex-ante ceilings. |
| `E` | Runtime closure for every planned cell: schedules, preflights, requests, provider receipts, retries, failed reservations, ledger start/final tails, candidate lineage, feedback release, promotion/rollback, token/cost usage, and terminal state. |
| `O` | Independently authenticated SCORE objective artifact containing the scored observations, estimator identity, confidence level and endpoints, resources, and performance-only decision. The caller cannot supply interval endpoints. |
| `P` | One total, task-native, nonadaptive `PURE@B` aggregation rule per primary task; an exact task/evaluation-unit/sample allocation with globally distinct frozen seeds and per-sample call, provider-attempt, and token ceilings; exact joins to FULL's model, solver, split, grader, query-DAG, and retry commitments; plus runtime closure proving the planned calls and ceilings were honored. |
| `Q` | Ordered commitment to disjoint fresh gate blocks, one-time consumption receipts, candidate freeze before each matched batch, isolated condition feedback, and complete campaign alpha accounting. |
| `SA` | Out-of-process sealed-authority bundle: preregistration and split commitments, grader/service/container digests, sandbox policy, asymmetric public key, private-key/target isolation, accepted champion hashes, and one atomic whole-grid authorization. |
| `SR` | One signed aggregate sealed release after every cell settles, with item artifacts escrowed and no earlier score/trajectory disclosure. |
| `T` | Frozen, independently audited analysis bundle: estimands, resampling hierarchy, simultaneous tests/intervals, missingness and infrastructure retry rules, protected/cost constraints, deterministic analysis seeds, code/container hashes, and coverage evidence at all relevant boundaries. |
| `C` | Complete role-level resource ledger and frozen price table covering cached/uncached input, output/reasoning tokens, failed calls, actual billed and normalized cost, wall time, CPU/GPU, memory, and tool calls. |
| `J` | For every non-programmatic mechanism judge, a disjoint arm-blinded human-labelled calibration artifact with frozen sampling frame, adjudication, agreement/error metrics, thresholds, identity/revision, prompt, decoding, and abstention rule. |
| `X` | External-baseline manifest identifying official source versus reconstruction, upstream commit, all local patches, role-model mapping, task/split, seed harness, feedback view, optimizer budget, and execution/evaluation closure. |

## 2. Minimum model roster

Freeze models from availability-only and licence/provenance checks, never from a
benchmark score. The following is the recommended discovery slate, not a claim
that these aliases are currently served or that an API route proves open-weight
status.

| Slot | Discovery candidate | Family constraint | Current status | Required artifact | Structural minimum | Power prerequisite |
|---|---|---|---|---|---|---|
| `M1` | `dashscope/qwen36-35b-a3b`, if available and licence-eligible | Qwen | `B-conf`: no frozen exact revision receipt | `M` | one exact configuration used across solver, diagnosis, proposal, materialization, ranking, and nomination | included in every pilot covariance and failure model; no model may be chosen by outcome |
| `M2` | `dashscope/glm-5.2`, if available and licence-eligible | non-Qwen family | `B-conf`: no frozen exact revision receipt | `M` | one exact configuration across the same roles | same as `M1` |
| `M3` | `DeepSeek-V4-Flash`, or another availability-only eligible open configuration | preferably a third family; at minimum preserve two families overall | `B-conf`: route availability, open-weight eligibility, and exact revision unverified | `M` | one exact configuration across the same roles | same as `M1`; replacement is allowed only before any score-bearing pilot |

The gateway's provider-declared identity is useful observation, not a signed
served-revision attestation. If the endpoint cannot bind an immutable served
revision, the run may be reported as nominal-model-matched harness optimization,
but it cannot support the paper's exact same-model self-evolution claim. A
stronger proposer, diagnoser, ranker, or nominator creates a heterogeneous
optimizer baseline and is excluded from the same-model and same-model `>10 pp`
families.

## 3. Benchmark roster readiness

| Benchmark family | Intended confirmatory scope | Implemented now | Confirmatory blocker | Licence and release boundary | Structural minimum before power |
|---|---|---|---|---|---|
| GSM8K | mathematical generation | Pinned full-file registry, trusted adapter/scorer, non-reportable smoke, TRUE PURE runner, and a 16-task public development exercise are implemented | no externally sealed seven-arm multi-seed study or signed provider/runtime closure | MIT evidence and pinned hashes are recorded in [`THIRD_PARTY.yml`](../../THIRD_PARTY.yml); dataset bytes remain on demand | every frozen `G[GSM8K]*` group in each of `S*` matched blocks |
| BBH-23 | heterogeneous reasoning, equal macro weight over 23 subtasks | only `logical_deduction_seven_objects` has a pinned adapter and development/smoke path | the other 22 subtasks, full macro scorer, group split, licence/hash registry, and reportable seven-arm runner are absent | current audit records MIT evidence for only the one pinned file; register every file and nested notice before use | all 23 subtasks, their frozen within-subtask item rosters, and `S*` matched blocks |
| SkillsBench-87 | strict structured generation/coding and sandboxed artifact execution | one `dialogue-parser` native-compatible smoke exists and explicitly is not an official score | 86 tasks, official-suite parity, full task/group split, hidden-verifier authority, and reportable seven-arm runner are absent | SkillsBench is not yet entered in `THIRD_PARTY.yml`; verify the exact upstream licence and every nested task asset before execution or redistribution | all 87 tasks as item-side groups and `S*` matched blocks; retries/rollouts remain inside each run |
| `UntouchedClass-1` | genuinely untouched high-label classification with source/group separation | none; S2D, LawBench, and USPTO have already been inspected and are development-only | an independent authority must select or construct the source, keep sealed items and labels outside the author workspace, and publish only commitments before freeze | require a documented right to use and release aggregate results; do not select LawBench or USPTO without resolving their nested/unknown data terms | the complete authority-frozen `G[UTC]*` roster and `S*` matched blocks; exact group count comes only from the disjoint pilot |

`UntouchedClass-1` should be classification because SkillsBench already supplies
the structured/coding family. It must be chosen by a source/time/group and
licence rule fixed before labels or model scores are seen. A random resplit of
the already inspected Meta-Harness records is not “untouched.” If no qualifying
classification source can be placed under independent authority, substitute a
new, independently held strict structured-generation source and narrow the
cross-task generality claim accordingly.

Current implementation evidence is auditable in the
[GSM8K protocol](../gsm8k-protocol.md), the
[BBH adapter](../../spiral_harness/benchmark/bbh.py), the
[SkillsBench smoke](../../spiral_harness/benchmark/skillsbench_smoke.py), the
[PURE runner](../../spiral_harness/benchmark/pure_reference.py), and the
[development four-arm runner](../../spiral_harness/experiments/development_four_arm_runner.py).
The 16 public development tasks and 58 model calls in the latter are wiring and
self-evolution diagnostics, not `S*`, not a powered sample, and not a sealed
result.

## 4. Real-task benchmark-by-arm execution matrix

The minimum shown in every row is the minimum analyzable structure, not a
numerical power recommendation. `PW-R` means the real-task power gate in
Section 8 must pass. If a baseline receives an inferential claim, its named
contrast must be included in that gate and its multiplicity family before
freeze.

| Benchmark | Arm | Implementation / confirmatory status | Required artifact(s) to clear the cell | Recommended minimum analyzable sample | Power prerequisite |
|---|---|---|---|---|---|
| GSM8K | `PURE` | `I-dev / B-conf`: TRUE provider-minimal execution exists | `M,B,E,SA,SR,T,C` | `S*` matched reference closures over all `G[GSM8K]*`; no optimizer trajectory | `PW-R`; separately power FULL--PURE if claimed |
| GSM8K | `STATIC` | `I-dev / B-conf`: static arm exists in development and replay paths | `M,B,A,E,SA,SR,T,C` | `S* x G[GSM8K]*` | `PW-R` for FULL--STATIC |
| GSM8K | `Random-valid` | `I-contract / B-conf`: finite-catalog strategy/replay contracts exist, no reportable live study | `M,B,A,E,O,Q,SA,SR,T,C,J` | `S*` independent random-valid searches over `G[GSM8K]*` | `PW-R` if a superiority claim is made |
| GSM8K | `Prompt-only` | `I-contract / B-conf`: automatic prompt-search contracts exist, no reportable live study | `M,B,A,E,O,Q,SA,SR,T,C,J` | `S*` independent searches over `G[GSM8K]*` | `PW-R` if a superiority claim is made |
| GSM8K | `SCORE` | `I-dev + I-contract / B-conf`: simplified development feedback arm and prospective profile exist | `M,B,A,E,O,Q,SA,SR,T,C,J` | `S*` complete matched adaptive searches over `G[GSM8K]*` | `PW-R`; retain dependence from shared FULL runs; mechanism measurements are charged shadows, not SCORE feedback |
| GSM8K | `FULL` | `I-dev + I-contract / B-conf`: development item-feedback arm is not the full mechanism-gated policy | `M,B,A,E,O,Q,SA,SR,T,C,J` | `S*` complete searches over `G[GSM8K]*`, including every attribution quartet | `PW-R`; all primary and constraint gates must close |
| GSM8K | `PURE@B` | `I-contract / B-conf`: task-bound per-unit/sample budget and aggregation contract exists; suite roster and live executor do not | `M,B,A,E,P,SA,SR,T,C` | one `K[B]`-sample aggregate per `(model,task,search-seed)` reference, for all `S* x G[GSM8K]*` | separately power FULL--PURE@B if claimed; `K[B]` is not `n` |
| BBH-23 | `PURE` | `P / B-conf`: TRUE PURE can target the one implemented subtask, not BBH-23 | `M,B,E,SA,SR,T,C` plus full BBH-23 registry | `S* x 23` subtask groups with all frozen items | `PW-R`; estimate between-subtask and within-subtask dependence |
| BBH-23 | `STATIC` | `P / B-conf` | `M,B,A,E,SA,SR,T,C` plus full BBH-23 registry | `S* x 23` subtask groups | `PW-R` for FULL--STATIC |
| BBH-23 | `Random-valid` | `I-contract / B-conf` at generic search layer; benchmark layer incomplete | `M,B,A,E,O,Q,SA,SR,T,C,J` plus full BBH-23 registry | `S* x 23` subtask groups | `PW-R` if inferential; macro-average subtasks rather than pooling items |
| BBH-23 | `Prompt-only` | `I-contract / B-conf` at generic search layer; benchmark layer incomplete | `M,B,A,E,O,Q,SA,SR,T,C,J` plus full BBH-23 registry | `S* x 23` subtask groups | `PW-R` if inferential |
| BBH-23 | `SCORE` | `I-contract / B-conf`; no BBH-23 reportable runner | `M,B,A,E,O,Q,SA,SR,T,C,J` plus full BBH-23 registry | `S* x 23` subtask groups | `PW-R`; preserve shared-FULL dependence and charged shadow measurements |
| BBH-23 | `FULL` | `I-contract / B-conf`; no BBH-23 mechanism runner | `M,B,A,E,O,Q,SA,SR,T,C,J` plus full BBH-23 registry | `S* x 23` subtask groups and attribution quartets | `PW-R`; all constraints close |
| BBH-23 | `PURE@B` | `I-contract / B-conf`: generic task-bound contract exists; no frozen 23-rule roster or live executor | `M,B,A,E,P,SA,SR,T,C` plus full BBH-23 registry | one `K[B]` aggregate for every `S* x 23` coordinate | separately power FULL--PURE@B; `K[B]` is not `n` |
| SkillsBench-87 | `PURE` | `P / B-conf`: one-task baseline smoke, not official-suite execution | `M,B,E,SA,SR,T,C` plus SkillsBench-87 parity/licence bundle | `S* x 87` task groups | `PW-R`; pilot task-level heterogeneity and failure rate |
| SkillsBench-87 | `STATIC` | `P / B-conf`: one-task optional-guidance smoke only | `M,B,A,E,SA,SR,T,C` plus suite parity/licence bundle | `S* x 87` | `PW-R` for FULL--STATIC |
| SkillsBench-87 | `Random-valid` | `B`: no suite-level adapter or compatible mutation execution | `M,B,A,E,O,Q,SA,SR,T,C,J` plus suite parity/licence bundle | `S* x 87` | `PW-R` if inferential; sandbox failures stay in the denominator |
| SkillsBench-87 | `Prompt-only` | `B`: no suite-level adaptive runner | `M,B,A,E,O,Q,SA,SR,T,C,J` plus suite parity/licence bundle | `S* x 87` | `PW-R` if inferential |
| SkillsBench-87 | `SCORE` | `B`: no suite-level aggregate-feedback search | `M,B,A,E,O,Q,SA,SR,T,C,J` plus suite parity/licence bundle | `S* x 87` | `PW-R`; include correlated task failures and charged shadow measurements |
| SkillsBench-87 | `FULL` | `B`: no suite-level mechanism-evidence search | `M,B,A,E,O,Q,SA,SR,T,C,J` plus suite parity/licence bundle | `S* x 87` and every attribution quartet | `PW-R`; all constraints close |
| SkillsBench-87 | `PURE@B` | `I-contract / B-conf`: generic task-bound contract exists; no frozen 87-rule roster or live executor | `M,B,A,E,P,SA,SR,T,C` plus suite parity/licence bundle | one `K[B]` aggregate for every `S* x 87` coordinate | separately power FULL--PURE@B; `K[B]` is not `n` |
| `UntouchedClass-1` | `PURE` | `B`: source and adapter do not yet exist | `M,B,E,SA,SR,T,C` plus independent data-authority receipt | `S* x G[UTC]*` | `PW-R`; pilot groups must be disjoint from sealed groups |
| `UntouchedClass-1` | `STATIC` | `B` | `M,B,A,E,SA,SR,T,C` plus independent data-authority receipt | `S* x G[UTC]*` | `PW-R` for FULL--STATIC |
| `UntouchedClass-1` | `Random-valid` | `I-contract / B-conf` only at generic search layer | `M,B,A,E,O,Q,SA,SR,T,C,J` plus independent data-authority receipt | `S* x G[UTC]*` | `PW-R` if inferential |
| `UntouchedClass-1` | `Prompt-only` | `I-contract / B-conf` only at generic search layer | `M,B,A,E,O,Q,SA,SR,T,C,J` plus independent data-authority receipt | `S* x G[UTC]*` | `PW-R` if inferential |
| `UntouchedClass-1` | `SCORE` | `I-contract / B-conf`: prospective profile only | `M,B,A,E,O,Q,SA,SR,T,C,J` plus independent data-authority receipt | `S* x G[UTC]*` | `PW-R`; preserve shared-FULL dependence and charged shadow measurements |
| `UntouchedClass-1` | `FULL` | `I-contract / B-conf`: prospective profile only | `M,B,A,E,O,Q,SA,SR,T,C,J` plus independent data-authority receipt | `S* x G[UTC]*` and attribution quartets | `PW-R`; all constraints close |
| `UntouchedClass-1` | `PURE@B` | `I-contract / B-conf`: generic task-bound contract exists; source roster and live executor do not | `M,B,A,E,P,SA,SR,T,C` plus independent data-authority receipt | one `K[B]` aggregate for every `S* x G[UTC]*` coordinate | separately power FULL--PURE@B; `K[B]` is not `n` |

With the minimum fixed roster of three models, four task families, seven arms,
and `S*` independent search seeds, the design has `3 x 4 x 7 x S* = 84S*`
condition evaluations. This is not the number of provider calls: adaptive arms
and `PURE@B` contain many calls, all of which must be reserved and charged.

The current confirmatory arm types encode `PURE/STATIC/SCORE/FULL`, while the
prospective scope-baseline schema separately binds Random-valid and Prompt-only
to FULL's typed mutation-policy artifact, canonical context IDs, and resource
ceiling. The older baseline machinery additionally encodes
`STATIC/Random-valid/Prompt-only/evidence-targeted`. None is yet a seven-arm
runtime closure: before freeze, artifact `A` must join all seven conditions into
one executable plan with common-block receipts rather than treating separately
valid schemas as an executed grid. See the
[four-condition baseline protocol](../baseline-protocol.md) and the
[prospective confirmatory arm contracts](../../spiral_harness/experiments/confirmatory_arms.py)
and the task-bound
[`PURE@B` contract](../../spiral_harness/experiments/confirmatory_pure_at_b.py).

## 5. HarnessFaultBench factorial and ablation matrix

`SS/MS/SM/MM` are feedback-by-promotion treatments, not synonyms for the
real-task `PURE/STATIC/SCORE/FULL` arms. The present v4 implementation is a
deterministic seven-family/six-surface trust-closure fixture. Its 28 scenarios
per partition are four roles within seven families, not 28 independent fault
families, and deterministic middleware rather than charged model output drives
behavior. A separate fixture-backed runner closes one process-local
`4*(28+1+4*28)=116+448=564`-call four-context development path over a paired fit
block and source/group-disjoint fixture GATE block. It uses fixed
`SS->MS->SM->MM` dispatch and has no live provider, powered roster,
counterbalancing, one-time block-consumption attestation, cross-process
atomicity, or inferential authority. See
[HarnessFault v4](../harness_fault_v4.md).

Additional required fault artifacts are:

| ID | Required artifact |
|---|---|
| `F1` | Frozen live-model fault generator, compatibility-masked design matrix, family-sampling target, repairability labels, source/group partitions, semantic-equivalent oracle repairs, protected behavior, blinded naturally occurring fault subset, external adversarial audit, placebo-neutrality calibration, and all hashes. |
| `F2` | Four-context factorial runtime closure that freezes all nominations before a fresh block opens and binds counterbalanced dispatch, schedules, receipts, failures, final tails, feedback release, and one-time block consumption. |
| `F3` | Authority-owned shuffled-evidence projection with identical schema and frozen byte/token envelope, permuted only within model/surface/family/severity/round strata. |
| `F4` | Executed child--parent--exact-revert--matched-placebo attribution closure with activation, adherence, behavior, protected behavior, leakage, policy, and resource decisions. |
| `F5` | Frozen factorial/ablation analysis and multiplicity bundle, including simultaneous simple-effect and interaction intervals and family/search-run clustering. |

| Cell or control | Treatment | Implementation / confirmatory status | Required artifact(s) | Recommended minimum analyzable sample | Power prerequisite |
|---|---|---|---|---|---|
| `SS` | score-only optimizer; performance-only promotion | `I-dev / B-conf`: included in the fixture-backed shared-fit/fresh-GATE four-context closure; no live, powered, or reportable campaign | `M,F1,F2,F4,F5,SA,SR,T,C,J` | `F-R* + F-N*` independent family clusters, each assigned one preregistered primary `S-F(f)` optimizer seed and one complete four-cell campaign | `PW-F`; paired four-cell policy effect, family clustering, seed draw, and failure process included |
| `MS` | mechanism-visible optimizer; performance-only promotion | `I-dev / B-conf`: same fixture-backed four-context closure and confirmatory blockers as `SS` | same as `SS` | same complete matched blocks; no cell counted as an independent block | `PW-F` for `MS-SS` and interaction |
| `SM` | score-only optimizer; mechanism-gated promotion | `I-dev / B-conf`: same fixture-backed four-context closure and confirmatory blockers as `SS` | same as `SS` | same complete matched blocks | `PW-F` for `SM-SS` and interaction |
| `MM` | mechanism-visible optimizer; mechanism-gated promotion | `I-dev / B-conf`: same fixture-backed four-context closure and confirmatory blockers as `SS` | same as `SS` | same complete matched blocks | `PW-F` for `MM-SS`, simple effects, and interaction |
| `M~M` shuffled information-volume placebo | MM-sized optimizer packet with correctly stratified but source-misaligned evidence; genuine packet remains with independent gate | `B`: design prose only | `M,F1,F2,F3,F4,F5,SA,SR,T,C,J` | paired with the same prospectively frozen fault-family and seed blocks | dedicated `PW-A`; cannot borrow factorial power |
| exact revert control | candidate treatment removed by reusing exact parent components | `I-dev / B-conf`: the raw-output development runner executes exact revert, but not a live powered factorial | `M,F1,F2,F4,F5,SA,SR,T,C,J` | one revert observation for every eligible candidate/family/seed attribution quartet | within-block measurement, not independent `n`; power the candidate--revert contrast |
| matched placebo control | neutral, distinct sibling matched on declared surface/context envelope | `I-dev / B-conf`: the raw-output development runner executes matched placebo, but not a live powered factorial | `M,F1,F2,F4,F5,SA,SR,T,C,J` | one placebo observation for every eligible candidate/family/seed quartet | within-block measurement, not independent `n`; power candidate--placebo |
| remove-revert ablation | full policy without exact-revert requirement | `B` for reportable fault study | `M,F1,F2,F4,F5,SA,SR,T,C,J` | same independently sampled blocks as its paired control, fixed before outcomes | dedicated `PW-A` and ablation multiplicity family |
| remove-placebo ablation | full policy without matched-placebo requirement | `B` for reportable fault study | same as remove-revert | same independently sampled blocks as its paired control | dedicated `PW-A` and ablation multiplicity family |
| remaining component ablations | omit activation, adherence, protected hard negatives, campaign spending, independent verifier, or source binding one at a time | `B` for reportable fault study | `M,F1,F2,F4,F5,SA,SR,T,C,J` | frozen family/seed blocks for every declared contrast | each powered prospectively or explicitly exploratory |

The familiar count of 59 independent exchangeable null/unrepairable
**families** with zero observed false promotions is only an unadjusted one-sided
95% exact-binomial illustration. It applies literally only when each IID family
contributes one Bernoulli event, including its one primary seed or a separately
preregistered any-over-seeds campaign event; it is invalid for averaged seed
outcomes or a weighted fractional-factorial roster. It is not a justified
minimum and cannot size R1--R2: `F-N*` must jointly reflect R1's Holm-adjusted
level, R2's paired positive margin and family dependence, the conjunctive power
target, and configured failures. Nor does 59 power `MM-SS`, the simple effects,
interaction, repair, or any ablation. Multiple roles, scenarios, rollouts, or
search seeds within one fault family do not turn it into multiple families.

## 6. External harness and optimizer comparison boundary

Every external method has two orthogonal tracks. An **official-fidelity track**
preserves upstream code, roles, models, iteration counts, environment, and
budget. A **normalized track** holds task, exact model, seed harness, feedback
view, and ex-ante budget fixed, but any port or paper-filled choice is a named
reconstruction. A result must never inherit “official” from the first track and
“same-model/fair-budget” from the second.

| System or benchmark | Official-fidelity track | Normalized track | Current coverage and blocker | Minimum replication / power boundary | Allowed claim if completed |
|---|---|---|---|---|---|
| Meta-Harness | Pin the upstream runtime and its original role mapping. The paper used an Opus 4.6 coding-agent proposer, GPT-OSS-120B solver, 20 iterations, and two candidates; the public cleaned skill's three-candidate default must not silently replace the paper setting. The named selected classifier sources are absent. | Run a clearly named Meta-Harness-style or paper-algorithm reconstruction on the same tasks, exact open model in all model-mediated roles, seed harness, feedback, stopping rule, and call/token ceiling as Spiral. Freeze every paper-unspecified choice. | Official loaders/runtime/no-memory/few-shot baselines and a first-party adapter exist; current S2D/Law/USPTO results are exposed, single-trajectory development pilots. Missing selected-agent source prevents an official-system reproduction. | `S-X*` independent outer-search seeds selected by a dedicated same-task paired power analysis; inner candidates/rollouts are not `n`. Exact task/model/budget matching and a new sealed split are mandatory for inferential comparison. | official-track reproduction only within its heterogeneous role/budget scope; normalized-track comparison only against the named reconstruction, never “official Meta-Harness” |
| Penguin public recursive example | Execute the pinned upstream TypeScript example, canonical five executions per generation, its file-edit/tool behavior, and exact public scorer. A changed model is a controlled rerun, not reproduction of the README's different model. | Match task bytes, accepted reports, scorer, 17-call schedule, output ceiling, and exact model. Preserve Penguin's own persistence mechanism; endpoint/logging wrappers may be common. A byte-compatible scorer port requires equivalence tests; a harness port is a reconstruction. | The canonical public-example adapter and offline provenance verifier exist. The observed result is nominal-model-matched, repeated public-task evidence with no hidden split or independent search runs. The advertised multi-task suite and graders were unavailable at the pinned revision. | independent full 17-call trajectories can characterize the public example, but five within-generation executions are rollouts, not search-run `n`; no replication count can create an untouched multi-task claim from this one public task | “canonical public-example controlled rerun” or “normalized public-example reconstruction,” never a Penguin benchmark-suite win |
| Penguin advertised multi-task suite | Run only if the official cases, gold outputs, graders, split and licence become auditable at a frozen commit | Same-task/model/budget normalization only after official bytes and scorer semantics are available | `B`: not auditable at the current pinned revision | power from a disjoint pilot after release; landing-page aggregates are not pilot observations for our estimator | no numerical comparison now |
| HarnessOpt-Bench | Pin an official executable release, its OfficeQA/BrowseComp-Plus/Terminal-Bench 2/GAIA environment, split, optimizer interface, and scorer | Optional same-model/budget adapter, labeled normalized if any official coordinate changes | paper audited; no public executable repository located by the 2026-08-14 cutoff | official run replication is not automatically adequate for Spiral's claim; use independent optimizer seeds and a predeclared paired analysis | author-reported context until an official release is reproduced; no cross-protocol win |
| Evo-Bench | Pin official fixed-policy-model long-horizon runner and 160-validation/448-evaluation task protocol if released | Optional normalized policy-model and budget track, clearly separated | paper audited; no public executable repository located by the cutoff | independent evolver/search seeds are `n`; the 608 visible/evaluation tasks are item-side observations | author-reported context until official reproduction |
| Self-Harness / HarnessBank | Use official code and its native gates/archives if a complete auditable release is selected before freeze | Same exact model/task/seed/budget normalized comparison with every deviation recorded | related work is audited, but no local official baseline execution closure exists | `S-X*` independent complete searches and dedicated multiplicity for any named win | named official or normalized comparison only; do not claim novelty for activation or gain gating |
| DSPy/MIPRO and GEPA | Official optimizer code on a supported native task is an ecological comparator | Recommended prompt-optimizer controls on the four shared tasks under the same exact model and ex-ante optimizer/evaluation budget | no local four-task comparison closure | `S-X*` independent optimizer searches; candidate evaluations and prompt rollouts are not `n` | same-model normalized optimizer comparison, not full-harness equivalence |

The current Meta-Harness boundary and artifact hashes are in the
[native comparison](../meta-harness-native-comparison.md) and
[third-party audit](third-party-audit.md). The Penguin schedule and evidence
boundary are in the [public-example report](../penguin-public-self-evolution-results.md).
The Meta-Harness official-fidelity role mapping is heterogeneous. Any OpenAI or
Anthropic role is also high-cost; running it requires the user's explicit
approval and it cannot be silently substituted into the same-model primary
family.

## 7. Data licence and sealed-authority matrix

| Asset | Current licence/provenance state | Confirmatory use | Required authority action |
|---|---|---|---|
| SpiralHarness source | no root first-party `LICENSE` or package licence metadata | implementation can be studied, but an explicitly open artifact release is blocked | rights holder selects a licence; rebuild release allowlist and notices |
| GSM8K | pinned MIT evidence and hashes | eligible if the authority verifies the exact source/group split and non-exposure lineage | fetch on demand, commit hashes not bytes, escrow sealed items, retain notice |
| BBH logical-deduction-seven | pinned MIT evidence for one file | development only for the current one-subtask adapter | extend provenance, hashes, notices, graders, and splits to all 23 subtasks |
| SkillsBench-87 | exact smoke revision exists in code; no `THIRD_PARTY.yml` entry | blocked | verify repository licence and nested assets at the exact revision; record per-task redistribution and hidden-verifier policy |
| `UntouchedClass-1` | unselected | blocked by design | independent data steward chooses source under a prospective rule, proves use rights, keeps identities/labels out of author and candidate environments, commits source/time/group split, and signs the release |
| Meta-Harness runtime | MIT at pinned revision | official released runtime components may be used from a separate checkout | record upstream/local patch digests and preserve attribution |
| Symptom2Disease | Apache-2.0 evidence at original source; already inspected | development only unless a genuinely new independently held source/group split exists | comply with attribution/modification terms; do not relabel the exposed split as sealed |
| LawBench subset | repository Apache-2.0, unresolved nested CAIL2018 terms; already inspected | development only and no redistribution | resolve nested terms before any release; use a new untouched source for confirmation |
| USPTO subset | exact original source/licence unresolved; already inspected | development only; redistribution prohibited by project policy | obtain source, split procedure, licence/permission and notices, or exclude |
| Penguin public example | pinned Apache-2.0 source | eligible only as a public case study, not an untouched task family | preserve notice/source hash and distinguish upstream execution from ports |

The sealed authority, not the researcher process, must own targets, grader
secrets, fresh gate blocks, reserve blocks, replacement decisions, and its
private signing key. Before search it publishes only commitments, code/container
digests, the preregistration commit, and an asymmetric public key. It accepts the
complete champion grid once, executes without candidate network or secrets, and
releases one signed aggregate only after all cells settle. Process-local HMACs
and Python capability objects remain valuable engineering checks but do not
clear `SA` or `SR`.

Public benchmark availability and experimental sealing are different. An item
can be publicly downloadable yet hidden from the optimizer during a run; that
does not prove it was absent from model pretraining or author inspection. The
paper must disclose both facts. `UntouchedClass-1` adds the stronger author-side
non-exposure guarantee through independent stewardship.

## 8. Statistical, missingness, cost, and power matrix

### 8.1 Power gates

| Gate | Claims covered | Inputs that must be estimated on a disjoint pilot | Freeze criterion before main search |
|---|---|---|---|
| `PW-R` | real-task FULL--SCORE, FULL--STATIC, and every declared FULL--PURE, FULL--PURE@B, per-model, per-benchmark, or named-baseline claim | complete-search variance; paired discordance; model-by-task heterogeneity; source/task-group and rollout ICC; shared-FULL covariance; timeout/refusal/malformed/crash and provider-missingness probabilities; token/cost distributions | predeclare candidate `S`, alternatives strictly beyond the claimed margins, target power, all null/partial-null/`0.10` boundaries, estimator, Monte Carlo error rule, and no-fallback selector; calibrate on one stream and validate once on an independent stream |
| `PW-F` | `MM-SS`, `MS-SS`, `SM-SS`, interaction, repair, harmful and false promotion | family prevalence, repairability strata, joint family--primary-seed variability, paired four-cell discordance, absorbing no-nomination, judge error/abstention, infrastructure and model failures | freeze `F-R*`, `F-N*`, one primary `S-F(f)` draw per family, family/seed sampling target, stratum weights, simultaneous family-clustered inference, and failure model; verify coverage under global and partial nulls |
| `PW-A` | shuffled-evidence, revert/placebo, and component ablations | paired effect/discordance and dependence with the core factorial on a disjoint pilot | separate power calculation and multiplicity family for each inferential ablation; otherwise label it exploratory |
| `PW-X` | named official or normalized external-method superiority | complete outer-search variance, role/budget heterogeneity, paired task discordance, and failure/cost distributions under the exact comparison track | freeze `S-X*`, official/reconstruction label, estimator and family before execution; never borrow power from a different task/model/budget track |

The existing nested bootstrap and simulator are method-development prototypes.
They do not authenticate pilot inputs, prove coverage, model all failures, or
select `S*`. A formal recommendation needs independent boundary-null audits,
including partial nulls and the `0.10` claim boundary, while preserving the
dependence created because policy and end-to-end contrasts share FULL.

### 8.2 Claim and analysis matrix

| Claim family | Estimand and independent unit | Multiplicity rule | Missingness / failure rule | Cost rule | Required output |
|---|---|---|---|---|---|
| Primary commit reliability, absolute | family-level `p_FP(FULL)=E_{F,S}[Y(F,S,FULL)]` over the frozen joint null/unrepairable-family and primary-seed target; one generated family with one assigned primary seed is `n`, and `Y=1` means at least one false commit anywhere in its complete campaign | reliability R1, one-sided Holm--Bonferroni within R1--R2 at family-wise `alpha=0.05`; test `p_FP(FULL) < 0.05` using the multiplicity-adjusted exact bound only for IID Bernoulli family events, otherwise a coverage-validated weighted family-clustered bound | absorbing no-nomination contributes no false commit; malformed, failed, inconclusive, and infrastructure outcomes follow the frozen whole-block rule and remain in the family roster | retain all evidence, solver, proposer, judge, failed-call, and abandoned-candidate cost; a reject-all policy still has to pass U1--U2 | family/seed event roster, adjusted upper bound and decision, campaign traces, signed commit registry, and `F1--F5/E/T/C/SA/SR` joins |
| Primary commit reliability, paired reduction | paired `p_FP(SCORE)-p_FP(FULL)` over the same frozen family/primary-seed blocks; family remains the independent unit | reliability R2 in the same R1--R2 family; reduction must exceed the positive operationally justified `delta_FP` frozen before the outcome-bearing pilot | a missing treatment cell invalidates or triggers the prospectively defined whole four-cell block retry; original failures and costs remain | SCORE and FULL use the frozen matched policy coordinates and ceilings; report realized role-level usage and cost | paired estimate, adjusted lower bound and decision, all family/cell outcomes, and the same complete closure |
| Primary real-task policy | equal-weight fixed-roster macro mean of binary FULL--SCORE over model--task cells; one complete search seed is algorithmic `n`, with group resampling inside run | sealed-usefulness U1, one-sided Holm--Bonferroni within U1--U2 at family-wise `alpha=0.05`; margin 0.02 | candidate refusal, malformed output, candidate timeout/crash score zero; retain prior champion and all cost; infrastructure retry reruns the complete frozen block | SCORE and FULL share ex-ante topology and ceilings; report realized role-level usage rather than claiming equal realized tokens | estimate, adjusted decision, simultaneous interval, all seed/group outcomes, and `E/O/T/C/SA/SR` joins |
| Primary real-task end-to-end | equal-weight FULL--STATIC macro difference; same complete-search and group hierarchy | sealed-usefulness U2 in the same U1--U2 family; margin 0.02 | same intention-to-treat rule | STATIC uses no fictitious optimizer calls; report absolute and incremental cost | same closure, with exact STATIC seed harness |
| Secondary controlled repair | equal-weight verified-repair difference over frozen model--repair-stratum cells, paired by search seed and fault family | prospectively frozen secondary family; margin 0.15; it cannot substitute for reliability R1--R2 | no-nomination is absorbing; failures and inconclusive mechanism links remain in denominator under frozen rule | calls/tokens/cost per verified repair plus protected violations | repair, harmful promotion, false promotion, precision, inconclusive, and simultaneous inference |
| `>10 pp` same-model family | every predeclared binary-success contrast intended for the statement, such as FULL--PURE, FULL--PURE@B, FULL--named normalized comparator, overall and any named model/benchmark slice | separate simultaneous multiplicity-adjusted one-sided lower confidence bounds; **every applicable LCB must be strictly greater than `0.10`** | no favorable complete-case deletion; all cells and failures retained; provider missingness above threshold makes the family inconclusive/invalid | PURE is effectiveness, not budget matched; PURE@B is the budget-matched reference; named normalized baselines match ex-ante budgets | full vector of estimates and simultaneous LCBs, not a selected point estimate |
| Factorial components | adaptive-policy `MS-SS`, `SM-SS`, and `(MM-SM)-(MS-SS)` over complete four-cell family/primary-seed blocks; family is the cluster and cells are paired | one prospectively frozen simultaneous secondary family | a missing cell invalidates or retries the whole four-cell unit under the frozen infrastructure rule | compute full evidence in every cell; withheld feedback must not skip calls or charges | conditional simple-effect and interaction estimates/intervals plus four-cell runtime closure |
| Shuffled and component ablations | paired policy differences on frozen fault-family/search blocks | separate ablation family; each unpowered contrast is exploratory | retain original failed blocks and costs; no arm-specific favorable reruns | token/byte envelope and call ceiling matched where claimed | all declared ablations, including negative results |
| Protected behavior and safety | frozen non-inferiority estimands per protected slice | exact margins, tests/intervals and family must be fixed before search | inconclusive/missing cannot become pass; report worst-case sensitivity | cost/safety remain constraints, not terms hidden in an optimized scalar | constraint table for every cell and slice |
| Cost and efficiency | absolute cost, normalized frozen-price cost, score per million tokens, calls/tokens per verified repair, and Pareto frontier | any maximum-cost/non-inferiority claim gets a frozen family; descriptive efficiency is clearly labeled | unknown failed-call usage burns its reservation; retain retries and abandoned candidates | separate solver, proposer, diagnoser, materializer, ranker, nominator, judge, grader, cached/uncached, billed/normalized totals | complete `C` ledger and price-table sensitivity |
| Named external method | track-specific paired difference under exact task/model/budget coordinates | its own frozen family unless explicitly and prospectively included elsewhere | preserve external method failures and full-search missingness | official-fidelity inequalities are reported; normalized track matches ceilings | `X` plus complete `E/T/C/SA/SR` closure |

Report intention-to-treat as primary, with complete-case, failure-as-zero, and
worst-case sensitivity. A provider- or harness-independent infrastructure
failure may trigger only the prospectively defined whole-block retry: all six
primary real-task conditions plus the `PURE@B` reference for that coordinate,
or all four factorial cells for a nomination block. Original attempts and cost
remain. Model endpoint or revision drift invalidates the entire
model--task--seed block.

## 9. Execution order and stop/go gates

1. **Close licences and data authority.** Add the full BBH-23 and SkillsBench-87
   inventories, resolve nested terms, select `UntouchedClass-1`, and obtain an
   external sealed evaluator. If any primary dataset lacks use rights or a valid
   group split, stop before model spending.
2. **Freeze the implementation candidate.** Complete all 28 real-task cells,
   the live fault factorial, shuffled packet, attribution controls, resource
   ledger, and official/reconstruction labels. Run development studies only on
   public exploration data.
3. **Availability-only model selection.** Check candidate routes and licence
   metadata without benchmark scoring; freeze exactly three configurations from
   at least two families. Signed immutable revision closure is required for the
   exact same-model claim.
4. **Run a disjoint pilot.** Estimate only the nuisance quantities in Section 8;
   close every pilot group so none can enter the sealed study.
5. **Validate and freeze power.** Coverage-audit the exact final estimator and
   missingness model; choose `S*`, task/rollout counts, fault-family counts,
   ceilings, judge calibration, and every multiplicity family once. A validation
   failure yields no recommendation, not a larger post-hoc fallback.
6. **Immutable preregistration.** Digest-bind `M/B/A/Q/T/C/J/F1--F5/X`, code,
   containers, graders, price table, seed schedules, public key, amendment
   policy, and one-shot authorization. No `TBD`, alias-only identity, or default-
   inferred field remains.
7. **Complete search before unsealing.** All models, tasks, arms, search seeds,
   factorial cells, failures, and external baselines settle. Do not inspect a
   partial sealed aggregate.
8. **Release once and analyze once.** Verify `SR`, run the frozen analysis, and
   report every outcome. If the simultaneous `>10 pp` LCB condition fails, the
   honest result is that the claim failed. The failed lineage is never reopened;
   any successor is a new disclosed study with a new preregistration and
   genuinely new sealed set, and cannot erase or opportunistically pool the
   failure.

## 10. Current blocking summary

The nearest executable milestone is the public 58-call GSM8K/one-subtask-BBH
four-arm study. It can demonstrate candidate generation, automatic promotion or
rollback, and a PURE/STATIC/SCORE/FULL feedback-view comparison on one nominal
model. It cannot clear any row in the confirmatory matrix because it has only 16
public tasks, one search trajectory, no Random-valid/Prompt-only/PURE@B arms, no
BBH-23 or SkillsBench-87 suite, no untouched classification task, no exact
served-revision attestation, no external sealed authority, and no validated
power or multiplicity closure.

The shortest honest path to an ICLR-ready empirical package is therefore:

1. complete benchmark/licence/authority prerequisites;
2. join the seven-arm runtime and full task roster;
3. implement the live four-cell fault study and shuffled/control ablations;
4. run the disjoint pilot and freeze power without inventing `S*`; and
5. execute the atomic sealed study once.

Until those steps complete, the paper should retain the evidence-blocked result
table and describe Meta-Harness and Penguin outcomes only within the explicit
development boundaries above.
