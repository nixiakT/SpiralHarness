# BFCL V4 public-development v2 prospective call plan

Status: provider-free registration only. No model was called, no task payload,
possible answer, score, or prior result was used to construct this artifact.
The typed-mutation catalog is bound, but execution remains blocked because an
independent semantic-audit release is not yet bound.

This is a public-development pilot. It is not an official BFCL result, hidden-test
evidence, confirmatory evidence, or a reportable benchmark result.

## Frozen scope

- Manifest fingerprint:
  `3f6e410bd26466381090ced0c6144515281a9653e5d36429685ad898f168eae7`.
- Structural partitions only: 5 FIT references, 4 GATE references, and 16 HOLDOUT
  references. A reference such as `fit-00` means the zero-based item within that
  split in frozen-manifest order. The plan contains no question or answer payload.
- Outer seeds, in order: `2026081601`, `2026081602`, `2026081603`.
- Typed atomic-mutation catalog fingerprint:
  `a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5`.
- Model route: `qwen36-35b-a3b` for every arm and every model-mediated role.
- Inference: temperature `0.2`, top-p `0.95`, output ceiling `2,048` tokens,
  total per-call token ceiling `32,768`, timeout `120 s`, and no stop sequences.
- Exactly one provider attempt per call. There is no retry, backfill, replacement
  seed, or adaptive stopping. Provider-seed honoring and the gateway's exact
  weight/tokenizer/runtime identity remain unattested until runtime evidence exists.

## Exact budget

| Arm | Calls per outer seed | Role allocation |
|---|---:|---|
| PURE | 16 | one bare HOLDOUT call per structural task |
| STATIC | 16 | one frozen-static HOLDOUT call per structural task |
| SCORE | 110 | `10 + 6 + 30 + 48 + 16` |
| FULL | 110 | `10 + 6 + 30 + 48 + 16` |
| PURE@B | 110 | six calls per HOLDOUT task plus one extra on 14 deterministic tasks |
| Total | 362 | per outer seed |

The campaign ceiling is `3 × 362 = 1,086` calls. The DAG also has six non-call
nomination nodes and six non-call GATE-decision nodes, for 1,098 total nodes.

For each adaptive arm, the `110` calls are:

1. parent FIT: 5 tasks × 2 rollouts = 10;
2. three diagnosis→proposal pipelines: 3 × 2 = 6;
3. three typed candidates: 3 × 5 FIT tasks × 2 rollouts = 30;
4. GATE: parent/nominated-candidate/revert/negative-control × 4 tasks × 3
   paired rollouts = 48;
5. HOLDOUT: 16 selected-or-parent-fallback calls.

## Global barriers

This is one global three-seed DAG, not three independently unlockable plans.

1. Across all seeds and both adaptive arms, all parent FIT, diagnosis, proposal,
   candidate FIT, and six nomination artifacts freeze before any GATE call.
2. All 288 GATE calls across all seeds finish before the six deterministic arm
   decisions freeze.
3. All six decisions freeze before any HOLDOUT, PURE, STATIC, or PURE@B call.

Timing dependencies and allowed evidence inputs are distinct. A node may wait on a
global barrier without receiving the other seed's or other arm's evidence. PURE,
STATIC, and PURE@B wait for the final barrier but receive no decision evidence.
Before request materialization, the lineage API derives a receipt binding the complete
campaign-plan fingerprint, node-schedule hash, exact node hash, call slot, provider
seed, and mutation-catalog fingerprint. The receipt itself contains no request payload.

## Candidate and promotion semantics

Each proposal slot must select one typed atomic edit. A duplicate, no-op, malformed
edit, or proposal-provider failure makes that candidate invalid, consumes the frozen
slot, and cannot be retried or backfilled. Invalid candidates execute their already
budgeted evaluation slots with exact parent fallback and cannot be nominated. If no
valid candidate exists, nomination freezes parent fallback.

Nomination is **not label-free**. It is a target-derived FIT-score operation using
binary outcomes from the trusted grader, with possible answers, raw targets, and
checker diagnostics hidden from the nomination controller and candidates. Valid
candidates are ranked by descending FIT binary-correct count, then ascending frozen
candidate slot. GATE and HOLDOUT evidence cannot enter nomination.

Promotion is fail-closed. The nominated candidate is promoted only if it is
admissible, all relevant FIT/GATE provider attempts succeeded, revert and negative
controls exactly match parent, nominated-candidate FIT is nondecreasing, and its
paired GATE count is strictly better than parent. Every failed condition selects the
parent. HOLDOUT cannot continue search or change the decision.

## Provider-seed pairing

- Parent and all three candidate FIT calls for the same task/rollout share a seed;
  SCORE and FULL share it as well.
- Corresponding SCORE/FULL diagnosis and proposal pipeline calls share a seed.
- The eight SCORE/FULL × four-variant GATE calls for one task/rollout share a seed.
- PURE, STATIC, SCORE, and FULL calls for one HOLDOUT task share a seed.
- All 110 PURE@B samples within an outer seed have distinct seeds.
- The provider-seed sets of the three outer seeds are pairwise disjoint.

These are request-seed commitments, not evidence that the provider honored them.

## PURE@B aggregation

The allocation is prospective and structural: every HOLDOUT reference receives six
samples; a domain-separated hash of the manifest fingerprint and structural task
reference assigns the 14 remainder calls. The allocation does not inspect labels,
questions, outputs, or scores.

Aggregation is total and label-free. Each successful frozen-parser canonical response
is one vote; every failed or invalid call contributes the same `no-response` vote.
The modal vote wins. Ties use ascending domain-separated vote SHA-256 and then
canonical JSON. Empty or all-failed inputs deterministically produce no response.
Targets, grader feedback, and possible answers are not accepted as aggregation input.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| Global 1,098-node schedule | `14870d5a39c14511385e348a0fef3f4f7e3eb2d2956152da24da43d048aaea76` |
| Replicate schedule 0 | `c79e05db8a72eccf7eb7dff3e9a6bd60307d30a3dbae1954423b31ce5d4b8e11` |
| Replicate schedule 1 | `125a8a7819d3d8732f07524c49549d3eafb9a657ab56d45d844d2d60bd78d9ee` |
| Replicate schedule 2 | `fd6ef864d507994e4a14af93a0cb86ee4b104fbdeeaf4a63187839238bdcc89d` |
| Common seed-projected topology | `3842e911db5d14c148580b3437820112d7b01e6c60e4e441addfce5efccc3d4a` |
| Complete campaign plan | `29e9729c6b374b733dcd6da7b95d7c662d826415ac0878b65df1d93893a324e6` |

The strict contracts reject changes to node order, dependencies, provider seeds,
budgets, allocation, aggregation, public-development scope, or reportability claims.
