# BFCL V4 public-development pilot result

Date: 2026-08-15

> **Result boundary:** the frozen 300-call protocol completed and passed a fresh, independent
> offline replay, but this was **not a successful self-evolution result**. All six SCORE/FULL
> candidates rolled back, FULL scored below STATIC and SCORE, and no evolved candidate reached
> HOLDOUT. This is a 15-task public-development pilot, not the official BFCL V4 suite, hidden
> evidence, a reportable benchmark result, or confirmatory evidence for a `>10 pp` claim.

## Bottom line

The canonical CAS analysis contains five arms evaluated on the same eight public HOLDOUT tasks
under three independently registered search seeds. The strongest arm was STATIC, tied exactly
with SCORE at `15/24 = 62.50%`. FULL reached `14/24 = 58.33%`; PURE reached
`11/24 = 45.83%`; and budget-matched PURE@B reached `12/24 = 50.00%`.

FULL was descriptively `+12.50` percentage points above PURE, but it was `-4.17` points below
both STATIC and SCORE and only `+8.33` points above PURE@B. Moreover, every adaptive decision
selected the parent. The FULL-minus-PURE number therefore does not demonstrate a successfully
evolved harness, and its `>10 pp` flag is descriptive only.

The terminal root is
`5e3ad4750aa8bb078894cbfdf49154b8b142fde1326d7c7edd61dbf577fb7ba9`. A fresh invocation of
`verify_bfcl_v4_public_campaign_execution` reconstructed every admitted 100-call replicate,
replayed all three semantic and attempt-accounting closures, verified the three-checkpoint chain,
rederived the analysis input from verified result refs, and exactly recomputed analysis artifact
`ebda40b0547f2e2a95c0c21917c294ca3ceff9fcf4fa9ba16740e211e8e0199d`.

## What was evaluated

The pilot is pinned to Gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` and `bfcl-eval==2026.3.23`. Its frozen roster has
5 FIT, 2 GATE, and 8 HOLDOUT tasks. The full roster and construction are documented in the
[prospective protocol](iclr-2027/bfcl-v4-public-pilot-protocol.md).

All arms requested the same open-model route, `qwen36-35b-a3b`, and the same per-call inference
configuration: temperature `0.2`, top-p `0.95`, maximum output `2,048` tokens, timeout `120`
seconds, and no stop sequences. Each frozen slot allowed one provider attempt and no automatic
retry. The offline verifier established equality of the requested model specification, backend
protocol identities, inference configuration, and per-call attempt budget across the campaign.
It did not, and cannot, attest that the opaque gateway served the same actual weight revision.

Per search seed, PURE and STATIC used 8 calls each. SCORE, FULL, and PURE@B used 28 calls each, so
PURE@B is the model-call-budget-matched untreated comparator for the adaptive arms. One replicate
contained exactly 100 calls; the campaign contained exactly 300.

The denominator `24` is eight repeated public tasks times three stochastic search seeds. Those 24
task-seed cells are useful for exact paired descriptive counts, but they are **not 24 independent
inferential units**. There are only eight task clusters, and no confidence interval or
statistical-significance claim is available.

## Canonical five-arm results

| Arm | Correct / task-seed cells | Exact accuracy | Percentage |
|---|---:|---:|---:|
| PURE | 11 / 24 | `11/24` | 45.83% |
| STATIC | 15 / 24 | `5/8` | **62.50%** |
| SCORE | 15 / 24 | `5/8` | **62.50%** |
| FULL | 14 / 24 | `7/12` | 58.33% |
| PURE@B | 12 / 24 | `1/2` | 50.00% |

These values come only from the verifier-matched canonical analysis. Percentages above are display
conversions of stored reduced rational numbers; no floating-point score is stored in the analysis.

## All frozen paired contrasts

`W/T/L` is treatment wins, ties, and losses over the same 24 paired task-seed cells. Delta is
`accuracy(treatment) - accuracy(reference)`.

| Treatment | Reference | W / T / L | Exact delta | Delta | Strictly `> +10 pp` |
|---|---|---:|---:|---:|:---:|
| STATIC | PURE | 4 / 20 / 0 | `1/6` | +16.67 pp | yes |
| SCORE | PURE | 4 / 20 / 0 | `1/6` | +16.67 pp | yes |
| FULL | PURE | 4 / 19 / 1 | `1/8` | +12.50 pp | yes |
| PURE@B | PURE | 1 / 23 / 0 | `1/24` | +4.17 pp | no |
| SCORE | STATIC | 0 / 24 / 0 | `0/1` | 0.00 pp | no |
| FULL | STATIC | 0 / 23 / 1 | `-1/24` | -4.17 pp | no |
| FULL | SCORE | 0 / 23 / 1 | `-1/24` | -4.17 pp | no |
| FULL | PURE@B | 3 / 20 / 1 | `1/12` | +8.33 pp | no |

STATIC and SCORE were cell-for-cell tied. FULL's sole loss to SCORE coincided with a failed FULL
HOLDOUT provider attempt on `parallel_4` under seed `2026081501`; all other FULL/SCORE cells tied.
This observed relation does not establish a general causal provider-failure effect.

## Self-evolution decisions and rollback causes

The frozen selection rule promotes only when all five conditions hold: the candidate is
admissible; every FIT/GATE provider attempt succeeds; parent/revert/placebo shadow controls match
exactly; candidate FIT does not regress; and candidate GATE is strictly better than parent GATE.
There was no manual override.

| Seed | Arm | Candidate admissible | All FIT/GATE attempts succeeded | Shadow exact | FIT parent -> candidate | GATE parent -> candidate | Decision and exact reasons |
|---:|---|:---:|:---:|:---:|---:|---:|---|
| 2026081501 | SCORE | yes | yes | yes | 5 -> 5 | 2 -> 2 | rollback: `candidate-gate-not-strictly-better` |
| 2026081501 | FULL | yes | yes | yes | 5 -> 5 | 2 -> 2 | rollback: `candidate-gate-not-strictly-better` |
| 2026081502 | SCORE | no | yes | yes | 5 -> 5 | 2 -> 2 | forced rollback: `candidate-invalid`, `candidate-gate-not-strictly-better` |
| 2026081502 | FULL | no | yes | yes | 5 -> 5 | 2 -> 2 | forced rollback: `candidate-invalid`, `candidate-gate-not-strictly-better` |
| 2026081503 | SCORE | yes | no | yes | 5 -> 4 | 2 -> 2 | rollback: `provider-attempt-failure`, `candidate-fit-regression`, `candidate-gate-not-strictly-better` |
| 2026081503 | FULL | yes | yes | yes | 5 -> 5 | 2 -> 2 | rollback: `candidate-gate-not-strictly-better` |

Both adaptive arms therefore recorded `0/3` promotions and `3/3` rollbacks. In every seed, the
parent was already `2/2` on the two-task GATE split. A strict improvement was impossible at that
observed ceiling, so even admissible, FIT-nonregressing candidates could not promote. This is an
important pilot-design limitation: two GATE tasks were too coarse to demonstrate a positive
selection event in this campaign.

Seed `2026081502` also had failed proposal-stage calls for SCORE and FULL, plus a failed FULL
diagnosis call. Both candidates were invalid and the frozen exact-parent fallback consumed the
remaining candidate FIT/GATE slots. That is why the table can simultaneously show an invalid
candidate and successful FIT/GATE attempts. Under seed `2026081503`, one SCORE candidate-FIT call
failed, producing an imputed incorrect prediction and the observed 5-to-4 FIT regression.

The selected parent system-prompt SHA-256 in every decision was
`69ed2ef8c92024b4dc52251b2a13d3ac447592d3da1227dfa8c3536e7828c2e4`. Candidate prompt
identities were:

| Seed | SCORE candidate prompt | FULL candidate prompt |
|---:|---|---|
| 2026081501 | `15ce3a80e256e42885d0eceac2888b3152e0b81464f76561bcc73d94cf465844` | `69b718cd08e73ce0700eb5063c14048c0ac8c77b35ebc846f3e3404bc7a8ff23` |
| 2026081502 | none; invalid exact-parent fallback | none; invalid exact-parent fallback |
| 2026081503 | `bcbd2bdac51c38a6a3c6f4be57f1e7f670550ba8e8d0cc60d1bf6aed3a72ec2e` | `376994663c14069a9ae3e3a823ec2af82c472b21fe65d128ebc5f00e41d8e7ef` |

Consequently, this run demonstrates candidate generation, invalid-candidate fallback, and safe
rollback. It does not demonstrate a beneficial promoted self-evolution step.

## Provider outcomes

| Seed | Succeeded | Failed | Identity observations | Declared identity consistent within observed responses |
|---:|---:|---:|---:|:---:|
| 2026081501 | 98 | 2 | 98 | yes |
| 2026081502 | 94 | 6 | 94 | yes |
| 2026081503 | 96 | 4 | 96 | yes |
| **Total** | **288** | **12** | **288** | — |

All 300 frozen slots were consumed. The 12 provider failures remained score-bearing outcomes and
were not retried. For ordinary gradable calls, the fail-closed projection is an empty incorrect
prediction. PURE@B failed samples enter its target-free aggregation as abstentions under the frozen
rule. No provider exception text was persisted in the terminal result.

The exact failed coordinates were:

| Seed | Global slot | Arm | Kind | Task / aggregate |
|---:|---:|---|---|---|
| 2026081501 | 41 | PURE | HOLDOUT | `simple_python_128` |
| 2026081501 | 69 | FULL | HOLDOUT | `parallel_4` |
| 2026081502 | 11 | FULL | diagnosis | aggregate |
| 2026081502 | 12 | SCORE | proposal | aggregate |
| 2026081502 | 13 | FULL | proposal | aggregate |
| 2026081502 | 82 | PURE@B | sample 2 | `multiple_7` |
| 2026081502 | 86 | PURE@B | sample 2 | `multiple_8` |
| 2026081502 | 87 | PURE@B | sample 3 | `multiple_8` |
| 2026081503 | 16 | SCORE | candidate FIT | `multiple_5` |
| 2026081503 | 76 | PURE@B | sample 0 | `simple_python_128` |
| 2026081503 | 77 | PURE@B | sample 1 | `simple_python_128` |
| 2026081503 | 90 | PURE@B | sample 2 | `parallel_3` |

Across arms, the failures were PURE `1`, STATIC `0`, SCORE `2`, FULL `3`, and PURE@B `6`.
Provider-declared response identity was internally consistent within each replicate among the 288
successful responses. Full 300-call identity coverage is nevertheless false, and neither
cross-replicate provider identity nor equality of actual served weights is attested.

## Independent offline verification

The result was opened only through the terminal's typed `result_ref`. The independent verifier did
not trust the terminal's pre-rendered scores. It strictly loaded the canonical execution root,
reverified each replicate from its runner-result ref, rebuilt each binding, replayed the three
checkpoint prefixes, rederived the campaign analysis input, recomputed the descriptive analysis,
and only then allowed `build_bfcl_v4_public_terminal_summary` to expose scores.

The verification summary was:

| Check | Verified value |
|---|---|
| terminal status | `complete` |
| registered prefix | exact |
| verified replicates | 3 |
| verified model calls | 300 |
| verified checkpoints | 3 |
| result/selection/metrics lineage | verified |
| canonical analysis recomputed and matched | true |
| external performance summary trusted | false |
| public-development only | true |
| reportable result | false |

`complete` means that the registered protocol and evidence graph closed. It does **not** mean that
FULL passed a performance criterion. Here the protocol completed while the adaptive performance
result was negative: no candidate promoted, and FULL did not beat STATIC, SCORE, or PURE@B by the
descriptive ten-point threshold.

## Provenance and immutable references

### Source and verifier identities

- live run code snapshot: `b38c5e9637c092a87cc7769398025087b9069a2b`;
- independent-verification repository snapshot:
  `0b3b80ad3d239f33043ec2324baa287b10a8fb48`;
- pinned Gorilla checkout: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`;
- campaign fingerprint: `596572961f2b4045f25fcd611f3c297ef0cf88c924c74b72f0aad8567300f63b`;
- pilot manifest fingerprint: `39fcdb35e0b877e225f7349a9e5d5db6edf575c0637e6b1757e3be2442ce3096`;
- native runner fingerprint: `7243cbb5547920f0e4235bdf15c189f3ce2f76a476c5a4e57f342d75f31e06be`.

The verifier-facing source-file SHA-256 values at the verification snapshot were:

| Source | SHA-256 |
|---|---|
| `bfcl_v4_public_campaign_executor.py` | `ad6ff3236b9311980c77a9e737472697fbd23f59798528ccba8d2cffc2db7cd3` |
| `bfcl_v4_public_campaign_executor_contracts.py` | `82583ebe800a5f590b3290aab8aa509d8d197a3523fa4021f00e76a1d01c269f` |
| `bfcl_v4_public_campaign_analysis.py` | `62d929582bdc2aa48dfc9e8e4e56cd943697e3bdc6f63e9b59fa4c119fa77f1f` |
| `bfcl_v4_public_campaign_analysis_contracts.py` | `a7fa7f3ecdbc6323bd4a1cb5d05e425a0e5f9d2e7e98da8a3f1bf4a22f4829a0` |
| `cli_bfcl_v4_public_summary.py` | `99974529133851f31bc98e52d0e091533b376945092977fd5d99df568f89c7be` |

### Outer execution receipts

| Receipt | Bytes | SHA-256 |
|---|---:|---|
| `runs/bfcl-v4-public-live-qwen36-20260815-terminal.json` | 11,490 | `4ec648167532e367e509b31ffa829a8c068111aa890615a41e7959125efc802f` |
| `runs/bfcl-v4-public-live-qwen36-20260815-stderr.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/bfcl-v4-public-live-qwen36-20260815-exit.txt` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

The recorded exit code was `0`. The authenticated model-catalog observation was frozen at
`2026-08-15T08:55:14Z`. No credential is present in this document or in the rendered terminal
summary.

### Campaign roots and checkpoints

Each row below is a complete ArtifactRef: media type, byte size, and SHA-256.

| Artifact | Media type | Bytes | SHA-256 |
|---|---|---:|---|
| campaign registration | `application/vnd.spiral-harness.bfcl-v4-campaign-registration.v1+json` | 263,307 | `596572961f2b4045f25fcd611f3c297ef0cf88c924c74b72f0aad8567300f63b` |
| live execution config | `application/vnd.spiral-harness.bfcl-v4-campaign-live-config.v1+json` | 11,839 | `fd9c36f6bd9c0576f0251d864d95bb11350b5018b430b35cd27b2d1d5c073bfe` |
| execution result root | `application/vnd.spiral-harness.bfcl-v4-campaign-execution-result.v1+json` | 9,186 | `5e3ad4750aa8bb078894cbfdf49154b8b142fde1326d7c7edd61dbf577fb7ba9` |
| checkpoint after seed 1 | `application/vnd.spiral-harness.bfcl-v4-campaign-checkpoint.v1+json` | 3,328 | `1858f3739254f9bd928c83f63d9dfa00fdc2e84634fb115bd651b6740261960f` |
| checkpoint after seed 2 | `application/vnd.spiral-harness.bfcl-v4-campaign-checkpoint.v1+json` | 5,597 | `07996256d7e943ffb0925460f47050858ec399304f4b8382cb207b268fc3590e` |
| checkpoint after seed 3 / latest | `application/vnd.spiral-harness.bfcl-v4-campaign-checkpoint.v1+json` | 7,702 | `0c7f12430c0511441c70ccdff866b75622e1317576562758eb23a701af21bd22` |
| campaign analysis input | `application/vnd.spiral-harness.bfcl-v4-campaign-analysis-input.v1+json` | 314,408 | `dc9cf06c538dea7d78bd7936ff454cf91175b10ad7b8c92621b6adbd54ee74df` |
| canonical campaign analysis | `application/vnd.spiral-harness.bfcl-v4-campaign-analysis.v1+json` | 36,991 | `ebda40b0547f2e2a95c0c21917c294ca3ceff9fcf4fa9ba16740e211e8e0199d` |

For the execution result, campaign, config, analysis input, and analysis, each stored model
fingerprint equals its ArtifactRef SHA-256.

### Frozen model and runtime fingerprints

| Coordinate | Value |
|---|---|
| selected route | `qwen36-35b-a3b` |
| live-config fingerprint | `fd9c36f6bd9c0576f0251d864d95bb11350b5018b430b35cd27b2d1d5c073bfe` |
| catalog-observation fingerprint | `e65dadf52857094099235f3bd0207e0c9316c2b62f021b800fb54ef56b23e6e2` |
| advertised-model-ID-list SHA-256 | `3f06bbc91e6b58e9fd1f855d416e6069883804c1f4790927e86330e07aacdf98` |
| model-spec fingerprint | `3fd01f5fb6b4146f6d214b39adbc4d385bbb9772095edcd1842f50f39b443728` |
| backend fingerprint | `2c3c1f7aad0ba6bc69bebaafd3bdc314c179ab62e01a3596e703abe7a174919a` |
| serializer fingerprint | `f4378aa4527bb631845262e1800c5a04d44e182f82c7d7ed1f23aaf567c94bbf` |
| parser fingerprint | `6b625654f04d581f51c18c4dc5be6fec08c772d426aa9141fa07b89184d0e6ea` |
| transport fingerprint | `384131ffab9b574d5b433772e47811b574e734d18875a20aa4f064b7061edeb4` |
| inference fingerprint | `4b7eeb5d601884f62249ee68522f4e8a772b43cc390ab95433dbbdcf85bc0b09` |
| attempt-budget fingerprint | `c3b15f7306ce71b55deb8279226320f9c22525be83bcc586115eccb5bc4c0dce` |

The per-replicate attempt budget was 100 attempts, at most 32,768 accounted tokens per attempt,
and at most 3,276,800 accounted tokens over the replicate. These are accounting ceilings, not a
claim that every call consumed that amount.

### Registered seed and schedule identities

| Ordinal | Replicate / outer seed | Call-plan fingerprint | Schedule content SHA-256 |
|---:|---|---|---|
| 0 | `search-seed-2026081501` / `2026081501` | `2ad745b6d6dfda2c2a91eed0e583ae4d00712bff63993634eb1d9b76809ced4b` | `d92f9061e2baf224d3aea8cbb1d9ca367345ab0a453ce4b106e3e5a8e2dd783e` |
| 1 | `search-seed-2026081502` / `2026081502` | `386a7f78acc2e6a7a47b87be033f30eee4aaf8e9cf860744bd7b25945ffcb53c` | `2fb4edb596c1aaf44850038b6fb245e26eb93c779e72f14e417eb66baf7a0fc4` |
| 2 | `search-seed-2026081503` / `2026081503` | `172af3f0beef05ac757344358a9ea6ef5f727150ae88bb70a4d9afbb0687f8fc` | `107d2d45fe248836c3d24b9ff5ecd15b6b665213a6e6f424b38a109a762692f0` |

### Per-replicate evidence graph

The short kind names below expand to these exact media types:

- `runner result`: `application/vnd.spiral-harness.bfcl-v4-public-runner-result.v1+json`;
- `verification`: `application/vnd.spiral-harness.bfcl-v4-campaign-replicate-verification.v1+json`;
- `closure`: `application/vnd.spiral-harness.bfcl-v4-run-closure.v1+json`;
- `ledger tail`: `application/vnd.spiral-harness.attempt-outcome.v3+json`;
- `candidate freeze`: `application/vnd.spiral-harness.bfcl-v4-joint-candidate-freeze.v1+json`;
- `selection freeze`: `application/vnd.spiral-harness.bfcl-v4-joint-selection.v1+json`;
- `selection evidence`: `application/vnd.spiral-harness.bfcl-v4-selection-evidence.v1+json`;
- `selection decision`: `application/vnd.spiral-harness.bfcl-v4-selection-decision.v1+json`;
- `HOLDOUT evidence`: `application/vnd.spiral-harness.bfcl-v4-holdout-evidence.v1+json`;
- `metrics`: `application/vnd.spiral-harness.bfcl-v4-public-metrics.v1+json`.

| Seed | Kind | Bytes | SHA-256 |
|---:|---|---:|---|
| 2026081501 | runner result | 38,573 | `9ff1d920843901e432b041d3eb6ece5ebfe85b0ba203c981cd86c89a98d6f12c` |
| 2026081501 | verification | 862 | `e8f485778863d27854c7ac409e7cf8f650404528d964fbccda6caff4f22afa1c` |
| 2026081501 | closure | 17,767 | `31f06f8f191be04d4d87616360d252072a34d37a7decd29f9b3914b978bda786` |
| 2026081501 | ledger tail | 735 | `484641bf349c1e881eff4f271c89c86ab5cec8c3512bce8e3c5cca2334608827` |
| 2026081501 | candidate freeze | 2,349 | `cd87a861bc7613c62844b2d750f149671a7e13b8ea5d4a314e41ee329b9d0aa1` |
| 2026081501 | selection freeze | 3,554 | `c5b61421a46c14cfcaecb543f377c69c0b47af0f78daae3e9ee0fda9a3ed3451` |
| 2026081501 | selection evidence | 193,594 | `6f9048c5e72cfe212454987937418ad9f0a70b99fd5884d185f1edbee09c29a9` |
| 2026081501 | selection decision | 7,012 | `b7c669b88948c11001c41b73b32d77509df175d42ae0eabd782e5bbacddd2ea6` |
| 2026081501 | HOLDOUT evidence | 77,796 | `16bf20f2bcbebcd583224680d8e378ea53d27a52359baa48f6f923c73a5e21e2` |
| 2026081501 | metrics | 6,966 | `61e939f41c1818d30e79d5b15844d484bd91a7e0e79a6bb10eea9c77ede11104` |
| 2026081502 | runner result | 38,573 | `289c12a5c4bc5f882a4d5dfa6d5a3609f336dc9144c7a87d4d6b1ed9ec9854dd` |
| 2026081502 | verification | 862 | `aa267c2f98685bad9c859112a2d6678a8e470b713feee94692cd289b1b849147` |
| 2026081502 | closure | 17,767 | `3ced79c0cfd920fe06624ba0ed8f397111d564dfd41cbf080642726cbf22c6c9` |
| 2026081502 | ledger tail | 733 | `825431c734fbd120ef97a88d4a7855e2713d7b2e05a0d86f41ce1282c81e7759` |
| 2026081502 | candidate freeze | 2,023 | `b57f476132121dd2fbb27ec2459e0ab976add2d3842b055d6dd193b5a1b6e915` |
| 2026081502 | selection freeze | 3,554 | `237fb8ab8549d0fb6c862670c0daf2f4a47072afa7860d60f32d853a9530cd0c` |
| 2026081502 | selection evidence | 182,221 | `7c56d0f66f3ffdf85e14bc65074f8a5f3523c91403652a275eefea43b1ee3d84` |
| 2026081502 | selection decision | 6,928 | `d7710a398a8a827f9569f0c973c4f66623a738d903fc00ab4c9325b8a2e40b57` |
| 2026081502 | HOLDOUT evidence | 78,108 | `97880ae4d4a9c8677fc8556d24d6b7e05474d1240326a12cfb1e7177458d1d0f` |
| 2026081502 | metrics | 6,948 | `408ee504701d441e2bd56727f9dc7b005ffad6f068cc483efe2d3b3f4b490c8e` |
| 2026081503 | runner result | 38,573 | `394c713a349e0cae571a6805de50fec1ddd3cf0a5481a8f756674ea85655fb82` |
| 2026081503 | verification | 862 | `b64eec66c30ee311bb4a3431b3611059d527b8ee54a6d659f6fd45c4a9937773` |
| 2026081503 | closure | 17,767 | `6882c51fe48b82dd2c6ea8a57dcef5becd63a2731440d2f235b98dee1e7a01dd` |
| 2026081503 | ledger tail | 733 | `b53883920bffbff2f6a975f1afcdf154fefb2e1e7ead295e4353b0c1c90c7d9f` |
| 2026081503 | candidate freeze | 2,349 | `ccdac0bc415dfbe2a9d285c652ead9eeff01d2c431134fa26390a7343ddb6e31` |
| 2026081503 | selection freeze | 3,554 | `bc477893067712d18e03d943bcfebd89c318b0d36ba01bace89dc8174d4b52ad` |
| 2026081503 | selection evidence | 192,700 | `d4b749da83a68e8113d4ee353a959ab6db5455b2bae9af6620e0896aff074695` |
| 2026081503 | selection decision | 7,068 | `d2fb342fd7f41c8b48670e137511ee18a18134d3b892f1217e7606a3dfccf0e2` |
| 2026081503 | HOLDOUT evidence | 78,126 | `624d6e49bc7650c3d310d0550716994a22ca7f1ea12b3c571cc093e64e9d0281` |
| 2026081503 | metrics | 6,948 | `4890a45f1f36e9e9c0562bc7a5e794b0c6de5d3a56c716bf8c7529103cb46a72` |

All three runner results bind the same ordered set of 15 frozen public-task refs:

```text
b1eb5942b69f8f2d37e6aa9ebc224a2d9e926946aceb7e18cf4f806401457042
44a727997cd36ba3d125cb69e38f8828c47c851c945373657858a036527722bb
fbb83d7be4fbf24c2e27a94a821e574bbdead6dac4703553b2a4ff9e7a426c70
93ae7d144f6da273cc20baddbc724b267398b4f151546d403945b7ec045988b8
4d317bff6799c2f9d371ba22ce26b823166fe687bb60be603c213b6ddfe082f9
0290b4b4730e43966312a35d8ac622744fa5b8644097f93da62bf5c7ae6cfd0a
d485a3201d20a01bb8698c961fe3d87598bfe34a6894171c65636785a132b85e
694855caf40477ee2ac30445073cec838215cde4b86bc3310e4dfd4a2011fee6
85896fd8df2f575de25034154ad07f4810f076a6a11ac6cb67d16b03bb144f4e
3d0704911d025632cad86d95f7e38beede6d17eb937731b14bc7624a58a4b024
c74bd20045b5dfa6b074cdeca6bf56fb6532ca1b36f613c12bd856f33755092a
2cc28a4971d57582743520e68179eafddc1398c68d25d42333ad7dab4337fed3
a738bc785ce99f40ad9c7638828ab6e6660b3b40b9519b537c7212e935eed8c1
621a937f8bb2295b1fe3b544247384c33dc410dbfdd8cf185aca1ec59f750e37
057e5b3d435d8ae9b1d090c66e334f8285af36ee7f9c1d092a8b5c591c8e5498
```

## Claim limits and next use

The canonical analysis fixes all of the following limitations:

- `public_development_descriptive_only=true`;
- `holdout_already_development_data=true`;
- `task_seed_cells_are_independent_inferential_units=false`;
- `greater_than_ten_pp_is_descriptive_only=true`;
- `confidence_interval_available=false`;
- `statistical_significance_claimed=false`;
- `all_call_response_identity_coverage_complete=false`;
- `same_provider_weights_attested=false`;
- `official_full_suite=false`;
- `hidden_test_evidence=false`;
- `reportable_result=false`.

This evidence may support an engineering statement that the registered execution, fail-closed
fallback, rollback, and offline evidence-replay mechanisms operated as specified on this pilot. It
does not support a claim that self-evolution improved BFCL performance. The opened HOLDOUT roster
must not be reused for further tuning and then relabelled as untouched evidence. Any optimization
prompted by these outcomes must start a new development lineage on a prospectively frozen,
previously unused roster; it cannot add seeds or selectively rerun this campaign.
