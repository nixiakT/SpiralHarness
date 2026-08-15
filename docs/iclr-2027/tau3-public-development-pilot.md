# Tau-cubed public-development prerequisite

Status: **rejected roster / execution blocked**, 2026-08-15.

No model, user-simulator model, judge model, or external API was called while
producing this receipt. This document is not a benchmark result, a frozen
score-bearing plan, a self-evolution result, or evidence for a performance
claim. The deterministic draw below failed a pre-result source-family audit,
so the implementation prevents it from releasing candidate task contracts.

## 1. Outcome

The provider-free prerequisite now closes four useful facts:

- tau-cubed is pinned to `sierra-research/tau2-bench` tag `v1.0.1`, commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`;
- 75 task, split, policy, database, tool, agent, runner, evaluator, metric, and
  Telecom task-lineage files have exact byte identities, with ordered bundle
  SHA-256
  `25097699bd2c87a914738dd4f4e350973189cb702b16d44ab874ffe609185cd5`;
- a result-blind deterministic draw can be reproduced exactly from all base
  task IDs;
- that draw is inadmissible because source families cross its intended
  FIT/GATE/HOLDOUT partitions.

The last point is the intended fail-closed behavior. A public hash draw removes
task-level score cherry-picking, but it cannot make an invalid candidate pool
source-family-disjoint.

## 2. Deterministic rejected draw

The complete public salt is:

```text
tau3:v1.0.1:fc0055dc4e0a316c3f83133267fbd6faaa770992
```

For canonical lowercase domain `d` and canonical string task ID `t`, the
selection digest is:

```text
SHA256(UTF8(salt) || 0x00 || UTF8(d) || 0x00 || UTF8(t))
```

There is no JSON quoting, newline, group label, model output, score, or task
content in this digest. Ties are broken by canonical task ID. Airline and
Retail sort their complete `base` pools and take the first six, assigning hash
ranks 0--1 to intended FIT, 2--3 to intended GATE, and 4--5 to intended
HOLDOUT. Telecom separately sorts the 36 mobile-data, 29 service, and 49 MMS
base-prefix pools and takes rank zero from each, all as intended HOLDOUT.

| Intended partition | Domain | Upstream task ID | Pool/rank |
|---|---|---|---:|
| FIT | Airline | `5` | 50 / 0 |
| FIT | Airline | `7` | 50 / 1 |
| FIT | Retail | `32` | 114 / 0 |
| FIT | Retail | `41` | 114 / 1 |
| GATE | Airline | `38` | 50 / 2 |
| GATE | Airline | `29` | 50 / 3 |
| GATE | Retail | `86` | 114 / 2 |
| GATE | Retail | `77` | 114 / 3 |
| HOLDOUT | Airline | `23` | 50 / 4 |
| HOLDOUT | Airline | `35` | 50 / 5 |
| HOLDOUT | Retail | `95` | 114 / 4 |
| HOLDOUT | Retail | `22` | 114 / 5 |
| HOLDOUT | Telecom/mobile data | `[mobile_data_issue]bad_vpn\|data_saver_mode_on\|user_abroad_roaming_disabled_on[PERSONA:None]` | 36 / 0 |
| HOLDOUT | Telecom/service | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin\|unseat_sim_card[PERSONA:Easy]` | 29 / 0 |
| HOLDOUT | Telecom/MMS | `[mms_issue]bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_storage_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card[PERSONA:Easy]` | 49 / 0 |

Every selected row binds the canonical full-row bytes, a trusted source
projection, template text, normalized template-token set, source text, and
normalized source-token set. The source projection contains only `id`,
`description`, and `user_scenario`, plus `ticket` for Telecom. It excludes
`initial_state`, `evaluation_criteria`, gold actions, assertions, and reward
basis. Those trusted identities are not candidate-visible.

## 3. Why the draw was rejected

Airline and Retail supply no authoritative source-family or template-lineage
identifier in this release; Airline annotations are null and Retail has no
general lineage field. Treating their numeric task IDs as independent groups
would therefore be unsupported. The three Telecom representatives also share
the same John Smith fixture and are one high-level source cluster, not three
independent statistical observations.

The frozen automatic screen tokenizes:

- template text: purpose, reason for call, and user task instructions;
- source text: known user information;
- normalized emails and digit-bearing strings, with a frozen stopword list.

It records all 105 task pairs and marks cross-partition exact collisions,
template Jaccard at least 0.35, or source Jaccard at least 0.40. On this draw,
the automatic screen reports zero violations. Its maximum cross-partition
template and source similarities are 0.15 and 1/3, respectively.

That automatic pass is insufficient. A pre-model semantic audit found:

- Airline FIT task `5` and GATE task `38` both exercise delayed-flight
  compensation;
- Retail FIT task `41`, GATE task `86`, and HOLDOUT task `22` overlap on
  user/default/order address changes.

These are counterexamples to treating the token thresholds as lineage proof.
The typed receipt consequently fixes all of the following to false:

- `roster_admissible`;
- `confirmatory_eligible`;
- `score_bearing_execution_allowed`;
- `candidate_contract_release_allowed`;
- `self_evolution_evidence`;
- `trajectories_frozen` and `budget_topology_frozen`.

It contains no search seeds. The earlier 83-trajectories-per-seed arithmetic
is deliberately not frozen here, and 249 must not be reported as an executable
campaign size from this receipt.

## 4. Candidate/trusted authority boundary

A future candidate-visible contract is restricted to:

- an opaque task ID;
- domain;
- exact policy-surface hash;
- exact tool-surface hash;
- an assertion that user messages arrive dynamically.

It cannot serialize upstream task ID, partition, source group, scenario,
initial state, evaluation criteria, reward basis, gold actions or assertions,
or grader fields. The rejected roster releases no candidate contracts at all,
including HOLDOUT contracts.

The eventual runtime must use separate authorities:

1. a candidate process with no task-file, database-file, evaluator, credential,
   or unrestricted network access;
2. a trusted interaction service holding scenario, user state, domain tools,
   and database state;
3. a trusted grading/release service holding evaluation criteria and judge
   authority, with no optimizer feedback channel before one-time release.

The present code verifies information shape and source bytes. It does not yet
attest this process/network isolation, so that remains an execution blocker.

## 5. Required source-family authority

The next roster cannot be repaired by swapping the identified tasks. Before
any partition or within-group draw, the complete 278-task base roster must be
grouped into conservative source families.

The provider-free audit interface emits one trusted record for each of 50
Airline, 114 Retail, and 114 Telecom base tasks. A record contains scenario
text, structured Telecom prefix, ordered trusted action-name signature, reward
basis, and source-projection identity. It contains no partition, selection
rank, candidate artifact, model output, harness output, or score.

Because no validated leakage-free automatic grouper exists, release is blocked
pending:

1. two independent human source-family annotations over the entire shuffled
   roster, blind to model and harness results;
2. one independent adjudicator resolving all disagreements;
3. an exact coverage check proving every base task is assigned once;
4. group-level FIT/GATE/HOLDOUT assignment, with every family confined to one
   partition;
5. only then, the frozen NUL-separated digest argmin within each assigned
   group;
6. a repeated automatic and independent semantic audit, with no post-partition
   task reassignment.

The annotation guide must conservatively merge shared entity/fixture sources,
intent or workflow templates, ordered action patterns, and reward-basis
families. Annotators may use trusted gold structure for grouping, but their
group labels and rationale must never reach the candidate or optimizer.

## 6. Arm meanings

The primary bare-agent baseline is **NATIVE-MIN**: the fixed upstream
`llm_agent`-style prompt, policy, half-duplex orchestrator, and authorized
domain tools, with no adaptive Spiral search. It must not be renamed `PURE`.

`PURE` remains a secondary structurally weak provider-minimal reference with no
domain tools or native agent loop. `STATIC` is a future frozen Spiral harness
on the same native substrate. `SCORE` and `FULL` require independently rerun,
matched adaptive searches, but neither is executable under this receipt.

`PURE@B` is **blocked/undefined**. Tau has no reviewed target-free rule that
turns multiple completed interactive trajectories into one deployable policy.
Oracle best-of-K, grader-selected trajectories, and post-hoc majority over
task outcomes are not valid replacements.

## 7. Budget and model blockers after regrouping

Even an admissible roster will not close the execution budget until the runner
enforces all of the following prospectively:

- the exact tested-agent model and immutable served revision, tokenizer,
  tool-call protocol, decoding settings, context/output limits, and retry
  policy are identical across arms;
- the same fixed user-simulator model, revision, prompt, tools, decoding, and
  task/trial seed coordinates are used across tested-agent arms;
- provider retries, batch retries, auto-resume retries, review retries, and
  hallucination retries are disabled or included in a finite matched-block
  policy; exhausted failures score zero in the primary intention-to-treat
  denominator;
- the proxy enforces one tested-agent generation per turn and records every
  logical call, provider attempt, token, tool call, step, wall time, and cost;
- the fixed 200-step upstream ceiling is treated as a combined interaction
  ceiling, not 200 agent calls plus 200 user calls;
- the NL evaluator is schema-checked so an empty result cannot pass through
  `all([])`, and its exact judge model/revision and retries are frozen;
- paid OpenAI or Anthropic judge/model use receives explicit approval. No such
  approval or call is implied by this prerequisite.

The tested model must be the same for NATIVE-MIN, STATIC, SCORE, FULL, and any
other supported tested-agent arm at a model/benchmark coordinate. A LiteLLM
route name alone is not a served-revision attestation. Credentials belong only
in the trusted runtime environment and must never be committed to the
repository or embedded in artifacts.

## 8. Claim boundary

Tau-cubed tasks and evaluator artifacts are public, so even a corrected split
would be a study-internal ecological holdout, not a globally secret test. A
future result must report both the upstream metric and the full-roster
intention-to-treat metric, use source families rather than near-duplicate tasks
as the item-side unit, and disclose residual pretraining contamination.

No observed ten-percentage-point gain may trigger adaptive stopping. Search
replicates, model coordinates, multiplicity family, power analysis, failure
policy, and one-time release must be frozen before score-bearing execution.
Until the independent grouping authority closes, there is no tau-cubed
self-evolution result to compare with a pure model or another harness.
