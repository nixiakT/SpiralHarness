# SWE-bench Verified and tau-family integration audit

Status: implementation and leakage audit, 2026-08-15. No model call or
benchmark rollout was made for this audit. This document is not a benchmark
result, a preregistration, or evidence that any arm is executable. It records
the exact upstream snapshots inspected, the current host blockers, and a
fail-closed integration design.

The governing statistical protocol remains
[preregistered-main-study.md](preregistered-main-study.md). In particular, an
observed improvement above ten percentage points is not a valid adaptive
stopping rule. Any such claim needs its own prospectively frozen, powered, and
multiplicity-controlled family.

## 1. Decision summary

Neither benchmark family is ready for a reportable SpiralHarness experiment on
this host.

- SWE-bench Verified passes the published hardware recommendations, but no
  container runtime or local SWE-bench environment is present. The official
  evaluator therefore cannot run locally.
- The original tau-bench repository is useful only as a named historical
  reproduction track. Its own current README says the airline and retail tasks
  are outdated.
- The original tau-squared paper release and the current tau-cubed release must
  be different tracks. Current task fixes make direct comparison with the paper
  numbers invalid.
- The strict SpiralHarness `PURE` arm is tool-free. It is structurally
  incapable of representing the normal agent interface on both benchmark
  families. Keep it as a secondary provider-minimal reference, but add a
  separately named `NATIVE-MIN` baseline for the meaningful agent comparison.
- `PURE@B` is blocked on both families. SWE has no frozen, target-free rule for
  combining independent patches into one deployable patch. Tau has no analogous
  target-free rule for combining several completed interactive trajectories
  into one deployable action policy. Oracle best-of-K or grader-selected output
  is not an admissible replacement.
- These benchmarks are public ecological tests, not globally secret tests.
  Their task and scoring artifacts can be isolated within this study, but
  pretraining or prior-author exposure cannot be ruled out.

## 2. Pinned evidence

### 2.1 Upstream snapshots

| Track | Audited source | Exact revision | Licence finding | Intended status |
|---|---|---|---|---|
| SWE-bench evaluator | `SWE-bench/SWE-bench` | `b3863c2919862c4128e7a80ad58d56b8fac6fd7f` | Repository code declares MIT; Python `>=3.10` | Candidate primary ecological track, blocked |
| legacy tau-bench | `sierra-research/tau-bench` | `59a200c6d575d595120f1cb70fea53cef0632f6b` | MIT | Historical/paper-fidelity appendix only |
| tau-squared paper release | `sierra-research/tau2-bench`, tag `v0.1.0` | peeled commit `37199f36924c8896f5e048360691f8476cd89ba1` | MIT; Python `>=3.13`; PDM lock present | Paper-fidelity primary tau track |
| tau-cubed release | `sierra-research/tau2-bench`, tag `v1.0.1` | peeled commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` | MIT; Python `>=3.12,<3.14`; uv lock present | Current corrected secondary track |
| tau-cubed development snapshot inspected | `sierra-research/tau2-bench`, `main` | `c3398666e6559e3a063da3fc04b5acf7f941464e` | same repository licence | Audit evidence only; do not execute an untagged moving target |

The SWE-bench repository licence does **not** establish the licence of the
Hugging Face dataset, each source repository, or redistributed container
content. The Hugging Face API and Git endpoint both failed from this host with
a TLS termination error during this audit. Consequently the following are
hard freeze blockers rather than guessed values:

1. the exact `SWE-bench/SWE-bench_Verified` dataset commit;
2. its data-card licence and redistribution conditions;
3. the SHA-256 of every consumed parquet or equivalent data object;
4. the licence and pinned source revision of every included repository;
5. the OCI manifest and platform-specific image digest for every instance.

### 2.2 Task rosters observed locally

The official SWE-bench documentation in the pinned code describes Verified as
500 instances. The dataset itself is not cached locally, so the 500-row roster,
repository counts, and group hashes have not been independently verified.

The tau rosters were parsed directly from the pinned public JSON files:

| Track | Airline | Retail | Telecom | Notes |
|---|---:|---:|---:|---|
| tau-squared `v0.1.0` | 50 | 114 | 114 | Also contains a 2,285-task telecom expansion, which is not the paper roster |
| tau-cubed `v1.0.1`-generation `base` split | 50 | 114 | 114 | Current default compatibility roster; not byte-identical to `v0.1.0` |

As a leakage diagnostic, grouping the public base tasks by ordered tool-action
names plus reward basis gives 32/75/94 distinct signatures in the
`v0.1.0` airline/retail/telecom rosters and 30/76/94 in the inspected current
airline/retail/telecom rosters. Telecom also exposes three obvious ID-prefix
families: 49 `mms_issue`, 36 `mobile_data_issue`, and 29 `service_issue` tasks.
These counts show why an item-random train/test split is not acceptable:
near-duplicate templates and action plans can cross it.

### 2.3 Current host capability

The following read-only checks were made on 2026-08-15:

| Resource | Observed | Requirement/consequence |
|---|---:|---|
| Architecture | `x86_64` | Matches SWE-bench recommendation |
| Logical CPUs | 128 | Exceeds recommended 8 cores |
| RAM | 503 GiB total, about 155 GiB available at audit time | Exceeds recommended 16 GiB |
| `/tmp` storage | about 382 GiB free | Exceeds recommended 120 GB |
| Workspace-backed storage | about 95 TiB free | Capacity is not the blocker |
| System Python | 3.10.12 | Too old for both pinned tau-squared and tau-cubed tracks |
| Repository `.venv` Python | 3.12.13 | Interpreter-compatible with tau-cubed, but it is the SpiralHarness environment and contains no `tau2` package |
| Docker/Podman/nerdctl/Apptainer/Singularity | none found | Blocks official local SWE evaluation |
| `uv` / PDM on `PATH` | neither found | The user-local `uv` binary exists outside `PATH`; PDM was not found |
| local `swebench`, `tau2`, `datasets`, Docker Python package | none found in the system or repository environment | No benchmark runtime is installed |

Standard text-mode tau execution is mostly in-process and does not itself need
Docker. It still needs a separately pinned compatible interpreter and
dependency environment. A compatible tau-cubed interpreter is already cached,
so this is an environment-provisioning and isolation blocker, not a hardware
blocker or a claim that the environment cannot be created. Voice,
knowledge-retrieval, and multimodal extensions are outside this audit and must
not be silently mixed into the text track.

## 3. Upstream evaluator hazards

### 3.1 SWE-bench Verified

The pinned evaluator has four properties that a formal wrapper must address.

**Result cache identity.** `run_instance` stores a report under
`run_id / model_name_or_path / instance_id` and returns it if it already
exists. The cache key does not contain the patch. Reusing a run ID after a patch
change can therefore return the earlier result. `get_dataset_from_preds` uses
the same path to skip completed items.

The Spiral wrapper must derive a unique execution identity from at least:

```text
study + benchmark/data revision + arm + independent search seed + task id
+ rollout seed + model identity/revision + solver config + patch sha256
+ evaluator commit + grader config + OCI image digest + retry-policy digest
```

Before accepting a cached report, it must verify every joined digest. Merely
using a human-readable new `run_id` is insufficient.

**Container privilege and network.** `create_container` adds `SYS_ADMIN` for
browser-sandbox compatibility and does not disable container networking. A
candidate-controlled process must not share this grading container. Retain any
capability required for official test parity only inside a dedicated grading
host, and enforce egress denial outside the container. No credential, model
proxy, task authority, Docker socket, host workspace, or cross-item cache may
be mounted into it.

**Mutable images.** Image construction defaults to a mutable `latest` tag, and
evaluation pulls the dataset-declared image name if it is absent locally. A
confirmatory manifest must resolve every tag to a platform-specific OCI digest,
mirror it, and fail if the runtime digest differs. Local rebuilds and published
images are different environment conditions and cannot be mixed within an arm.

**Failure accounting.** The current report classifies likely infrastructure
failures, but explicitly leaves them in the unresolved/error population so the
denominator is unchanged. Spiral's primary intention-to-treat score should be

```text
number of frozen-roster instances with resolved == true / frozen roster size.
```

Missing predictions, empty patches, patch-apply failures, parse failures,
timeouts, and exhausted retries are zero. Infrastructure retries are allowed
only under a frozen classifier and as a whole paired block across arms; see
Section 7. The upstream report remains a separately labelled secondary output.

### 3.2 Tau family

The original tau-squared release defaults to one trial, 200 maximum steps,
seed 300, and temperature zero for both agent and user. The runner generates a
trial seed and assigns the same seed to the agent and user. A provider may
ignore or only partially honor that seed, so a numeric seed is not evidence of
determinism; provider receipts and complete trajectories remain necessary.

Tau's final reward is a product of the components selected in each public
task's `reward_basis`. A reward within `1 +/- 1e-6` is treated as success, and
the official family reports average reward and pass-hat-at-k. These exact
definitions and the task files must be hashed per track.

The inspected tau-cubed code excludes `INFRASTRUCTURE_ERROR` trajectories from
official metrics and retries them on resume. That is useful operationally but
can create arm-dependent denominators. Spiral must report the official metric
and, separately as primary, an intention-to-treat metric on the full frozen
task-by-trial roster. Under the latter, an exhausted infrastructure failure is
zero.

Current tau-cubed also adds task fixes and explicitly warns that some pre- and
post-`v1.0.1` scores are incomparable. Never label a tau-cubed result as a
tau-squared paper reproduction, and never compare either directly to a legacy
tau-bench leaderboard row.

## 4. Leakage and hidden-test boundary

### 4.1 What is actually public

SWE-bench's public instance type contains `patch`, `test_patch`,
`FAIL_TO_PASS`, and `PASS_TO_PASS` alongside the problem statement and base
commit. Gold prediction mode directly reads `patch`. Verified is therefore not
a hidden-test dataset in the usual sense, even if Spiral withholds those fields
during one study.

Tau stores task scenarios, initial database state, expected/gold actions,
reward basis, evaluation criteria, and simulator instructions in public files.
The normal runner hides them from the conversational agent logically, but an
in-process candidate with filesystem or import access can read them.

The paper may make an ecological claim on these public benchmarks. It must not
call either one an untouched private holdout or claim that model pretraining
contamination was excluded.

### 4.2 Required authority split

Use three processes or services with non-overlapping authority:

| Component | May receive | Must never receive |
|---|---|---|
| Candidate agent | public issue/policy, current user messages, authorized task-native tool schemas and results | gold patch/actions, test patch, test names, evaluation criteria, simulator instructions, grader result/log, host filesystem, model/API key |
| Trusted interaction service | frozen task row, DB state, user-simulator state, tool implementation, per-call proxy capability | candidate source mutation authority, sealed aggregate result before release |
| Trusted grader/release authority | candidate artifact, exact grader/test artifacts, roster commitment, signing key isolated from runner | optimizer requests, candidate-selected retries, unblinded adaptive feedback |

For SWE, generation occurs in a repository workspace that contains only the
base checkout and a candidate-visible test allowlist. Official hidden scoring
tests are applied only after the candidate patch is exported into a fresh
grader container. The candidate never receives grader stdout, per-test status,
or a second chance based on the sealed result.

For tau, expose policy text, user messages, and authorized domain tools through
an RPC boundary. Do not import the candidate harness into the process that can
read `tasks.json`, split files, database fixtures, or evaluator objects. The
trusted model proxy holds credentials and returns only the role-appropriate
response. The candidate sandbox has no network route except that proxy and the
typed tool service.

### 4.3 Split construction

Do not split either family by raw task ID.

- SWE groups must keep the same source repository together, then conservatively
  merge issues with overlapping base commits, changed-file paths, test names,
  issue/PR ancestry, or near-duplicate problem statements. A temporal split is
  preferable where coverage permits. All group-building code and thresholds
  are frozen before any score-bearing run.
- Tau groups must at least join domain, intent/template family, ordered
  action-name signature, reward basis, and shared entity/scenario source. The
  three telecom prefix groups above are a lower bound, not a sufficient final
  grouping.
- Exploration, fresh promotion-gate, and sealed evaluation groups are disjoint.
  No task, trajectory, test log, grader explanation, or derived error label
  moves backward from a later partition.

Because the source data are public, this creates study-internal sealing only.
The release must report the candidate model's training-data disclosure status
and the residual contamination threat.

## 5. Arm semantics and the `PURE` naming conflict

The following mapping is required for both benchmark families.

| Arm | SWE-bench interface | Tau interface | Search budget | Interpretation |
|---|---|---|---|---|
| `PURE` | problem payload through the provider-minimal wrapper; no repository tool loop | user payload through the provider-minimal wrapper; no domain tools or orchestrator logic | zero | Strict secondary reference only; structurally weak |
| `NATIVE-MIN` | minimal fixed repository agent loop with read/search/edit/test tools and patch export | official fixed `llm_agent`-style policy, orchestrator, and domain tools | zero | Primary bare-agent baseline |
| `STATIC` | frozen Spiral seed harness on the same native tool substrate | frozen Spiral seed harness on the same official environment/tool substrate | zero | End-to-end no-evolution baseline |
| `SCORE` | independently evolved harness with score-only feedback and performance-only promotion | same | matched to `FULL` | Primary adaptive control |
| `FULL` | independently evolved harness with typed source-linked evidence and mechanism-gated promotion | same | matched to `SCORE` | Proposed method |
| `PURE@B` | blocked: no target-free total patch aggregator | blocked: no target-free total trajectory/policy aggregator | intended to match FULL total | Undefined, not zero and not omitted silently |

`NATIVE-MIN` must not be renamed `PURE`, added as an alias of `PURE`, or
silently substituted in plots. Existing `PURE` contracts prohibit precisely
the tools and control flow needed here. This is a scientific distinction, not
an implementation detail. `NATIVE-MIN` is a prospective additional arm and
requires its own contract, label, and multiplicity treatment before freeze.

An SWE oracle `pass@K` that selects any patch passing hidden tests is an upper
bound, not a deployable `PURE@B` baseline. A tau average over K independent
dialogues estimates a success probability but does not aggregate K samples into
one action. Neither clears the `PURE@B` contract.

## 6. Same-model and same-budget contract

"Same model" means the exact tested **agent** model identity and immutable
served revision, tokenizer, tool-call protocol, inference settings, context
limit, and retry policy are identical across arms at one model-benchmark
coordinate. A provider route string alone is not an identity attestation.

The tau user simulator is stochastic benchmark machinery. Freeze its exact
model/revision, prompt, inference settings, retry policy, and trial-seed
schedule across all tested-agent arms. It may be a different model from the
agent. Indeed, keeping one fixed user simulator across tested models avoids
changing the environment when the tested model changes. Any LLM grader follows
the same trusted-machinery rule and is charged separately.

There is no honest scalar "same budget" across these arms. Freeze and report a
vector:

1. evaluation agent calls, provider attempts, input/output/reasoning tokens;
2. task turns and tool calls;
3. SWE shell commands, test invocations, wall time, CPU, RAM, and disk I/O;
4. tau orchestrator steps and wall time;
5. optimizer/proposer calls and tokens;
6. mechanism and judge calls, including SCORE shadow measurements;
7. trusted user-simulator and grader resources;
8. failed and retried attempts.

All arms receive the same per-item evaluation ceilings and the same task/trial
coordinates. `SCORE` and `FULL` additionally have an identical complete search
call graph, proposal count, nomination schedule, mechanism-query count, and
ex-ante ceiling. SCORE still computes and pays for the mechanism measurements
in shadow mode; it cannot see them or use them for promotion.

`PURE`, `NATIVE-MIN`, and `STATIC` perform no adaptive search. Their unused
search budget is reported as zero, not padded with irrelevant calls. Thus
FULL-minus-those arms is an effectiveness comparison accompanied by a complete
cost/Pareto report, not a total-compute-matched causal contrast. `PURE@B` would
be the total-call/token-matched reference, but remains undefined until a
task-native target-free aggregation rule exists.

## 7. Pairing, independent search seeds, and failures

Use the full coordinate

```text
(study, model, benchmark track, source group, independent search seed,
 task, trial seed, arm)
```

Within one independent search-seed block, pair arms on the same frozen task and
trial coordinates. For tau, this pairs the user-simulator RNG schedule, not the
realized conversation: the user must react naturally to each arm. For SWE, it
pairs the same base repository and evaluation image. Condition processes and
artifact stores remain isolated so one arm cannot read another arm's candidate,
trajectory, or result.

Each algorithmic replicate reruns the **complete** adaptive search from the
same seed harness in a fresh store. Changing only a task rollout seed inside
one search trajectory does not create another algorithmic `n`. Source groups,
not individual near-duplicate tasks, are the item-side resampling unit. The
number of search seeds, source groups, and rollout repeats must come from a
disjoint-pilot power analysis; this audit does not invent those counts.

A frozen failure classifier maps outcomes as follows:

- candidate/model errors, invalid tool calls, invalid patch, patch-apply
  failure, task timeout, max turns/steps, missing response, and malformed output
  are zero;
- an independently diagnosed evaluator/image/provider outage may trigger a
  retry only by reopening the entire matched arm block for that
  task/trial/search-seed coordinate;
- the retry schedule and cap are fixed before outcomes; no arm-specific or
  score-dependent retry is permitted;
- exhausted blocks are zero in the intention-to-treat primary analysis;
- official benchmark missingness conventions are reported separately;
- every first attempt and retry remains in the immutable resource ledger.

## 8. Minimal executable tracks

### 8.1 SWE-bench Verified

The first executable milestone is evaluator validation without a model:

1. resolve and hash the dataset revision and candidate-visible allowlist;
2. install a supported rootless or dedicated-host Docker setup;
3. resolve and mirror exact OCI digests for a small development-only roster;
4. run the official gold patch and an empty/invalid patch canary on those items;
5. verify gold resolution, zero mapping, timeout handling, egress denial,
   credential absence, patch-bound cache identity, and complete receipts;
6. expand only after the canary is reviewed.

Do not use Modal as an unexamined shortcut: it changes the execution authority
and adds credential, image, and provenance obligations. Do not run a paid model
until the offline evaluator canaries pass.

### 8.2 Tau-squared and tau-cubed

Create two isolated environments rather than one mutable install:

- `tau2-paper-v0.1.0`: Python 3.13, PDM lock, commit `37199f3...`;
- `tau3-current-v1.0.1`: compatible Python 3.12 or 3.13, uv lock, commit
  `fc0055d...`.

For each, first run upstream unit tests and deterministic mock/domain-tool
smokes with no external model. Then validate the out-of-process candidate,
interaction, and grader boundaries with a scripted fake agent and fake user.
The manifest must bind the exact base roster, every task hash, split hash,
policy/DB/tool-code hashes, reward code, max steps/errors, timeout, retry policy,
and simulator configuration.

Legacy tau-bench at `59a200c...` has unbounded provider dependencies in
`setup.py` and no modern lock. If reproduced, vendor a frozen environment and
label the result `legacy tau-bench`; do not use it as the current primary track.

## 9. Freeze blockers and next actions

The integration remains blocked until all applicable boxes close:

- [ ] SWE Hugging Face commit, data licence, source-repository obligations, and
      file hashes recorded.
- [ ] Complete SWE 500-instance roster and conservative source groups committed.
- [ ] Docker/OCI runtime installed on a dedicated grading host; image digests,
      egress policy, capabilities, and resource ceilings attested.
- [ ] Patch-bound cache wrapper and gold/empty/failure canaries pass.
- [ ] Separate pinned Python environments for tau-squared and tau-cubed pass
      upstream offline tests.
- [ ] Tau task/policy/DB/tool/reward/split hashes and conservative template
      groups committed.
- [ ] Out-of-process tau tool and hidden-task authority passes an attempted
      filesystem/import/network exfiltration test.
- [ ] `NATIVE-MIN` is added prospectively to the arm/statistical manifests; it
      is not conflated with `PURE`.
- [ ] `PURE@B` is either supplied a reviewed target-free total aggregator per
      benchmark or formally removed from that benchmark's claim family as
      undefined before study freeze.
- [ ] Exact tested-agent identities and fixed tau simulator/grader identities
      are attested; high-cost providers receive explicit approval before use.
- [ ] Full budget vector, retry blocks, independent search-seed roster, source
      groups, multiplicity family, and disjoint-pilot power result are frozen.
- [ ] One-time sealed release and no-adaptive-stopping rule are enforced.

Until then the defensible status is `integration-designed / execution-blocked`.
There is no SWE or tau performance number, no evidence of a greater-than-ten-
point gain, and no basis for a benchmark superiority claim from this audit.
