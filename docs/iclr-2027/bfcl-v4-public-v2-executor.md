# BFCL V4 public-development v2 executor and replay

Status: provider-free rehearsal infrastructure. The core imports no HTTP, LiteLLM,
OpenAI, or model SDK and has not called a real API. It does not turn the public v2
roster into official, hidden-test, confirmatory, or reportable evidence.

## Execution boundary

`execute_bfcl_v4_public_v2_rehearsal` accepts four injected interfaces:

- `BfclV4PublicV2RequestBinder` hashes the actual request bytes without storing them
  in the journal;
- `BfclV4PublicV2ProviderTransport` performs exactly one terminal attempt and returns
  only a sanitized canonical response and typed proposal disposition;
- `BfclV4PublicV2TrustedBinaryGrader` issues a decision-barrier-bound evaluation
  unlock and emits minimal trusted receipt projections for gradable calls;
- `BfclV4PublicV2PureAtBBatchGrader` grades only the 48 final modal cells, never the
  330 individual PURE@B source calls.

The core never retries the transport. Provider exceptions and malformed transport
results become terminal provider failures, consuming the frozen call slot. Duplicate,
no-op, invalid, or failed proposals remain in their original candidate slots and force
the plan's parent fallback. There is no backfill, replacement seed, or adaptive stop.

This executor is intentionally a provider-free public-development rehearsal. A
supplied semantic-release digest is bound as lineage, but this path deliberately does
not claim release authenticity: events record
`semantic_release_authenticity_attested=false` and the terminal receipt records
`score_bearing_execution=false`.

## Append-only journal

The journal has exactly one terminal event for every frozen DAG node:

- 1,086 call events, each burning one provider slot;
- 6 deterministic target-derived FIT nomination events;
- 6 deterministic GATE promotion/parent-fallback events;
- 1,098 total events in exact `node_slot` order.

Every event binds:

- complete campaign-plan and node-schedule fingerprints;
- exact node ID, node slot, and canonical node hash;
- atomic-mutation catalog fingerprint;
- runtime fingerprint;
- request or deterministic-control-input fingerprint;
- provider-response fingerprint for every successful provider attempt;
- trusted grade-request and receipt fingerprints for every successful gradable call;
- decision-barrier and evaluation-unlock fingerprints for final evaluation calls;
- semantic-release fingerprint;
- previous-event hash.

Events contain no task payload, possible answer, or checker diagnostic. A caller must
present the current tail hash when appending; stale-tail writes are rejected. The
portable `BfclV4PublicV2JournalSnapshot` contains the immutable event chain rather than
a trusted mutable counter.

## Durable reservation and crash recovery

When a checkpoint sink is supplied, every provider call follows a two-checkpoint
protocol. The executor first persists a reservation for the exact next node and
request under snapshot-fingerprint compare-and-swap, then crosses the provider
boundary, and finally replaces the reservation with its terminal event. A failed
reservation checkpoint prevents the provider call entirely.

If the process or terminal checkpoint fails after the provider boundary, the durable
snapshot retains the pending reservation. Resume independently verifies the partial
journal, converts that exact slot to `CRASH_RECOVERY_BURN`, and does not invoke the
provider again. This deliberately sacrifices an ambiguously completed slot instead of
risking an unaccounted duplicate request. Replay reports provider attempts and crash
recovery burns separately and requires their sum to equal all consumed call slots.

## Stage barriers and decisions

Replay consumes the repository-frozen stage-major plan. It cannot accept a GATE event
until all seeds' proposal, candidate-FIT, and nomination nodes are present. It cannot
accept a decision until all 288 GATE calls are present, and it cannot accept any final
evaluation until all six decisions are frozen.

Nomination is independently recomputed from valid proposal dispositions and trusted
FIT binary grades, with the frozen lowest-candidate-slot tie break. Promotion is
independently recomputed from candidate admissibility, terminal attempt success,
parent/revert/negative-control equality, nondecreasing candidate FIT count, and a
strictly better nominated-candidate GATE count. A journal-supplied control value is
never trusted.

## Independent replay and PURE@B

`replay_bfcl_v4_public_v2_journal` starts from the frozen campaign and reconstructs:

- next node and hash-chain tail;
- all 1,086 burned terminal call slots and success/failure counts;
- six nominations and six decisions;
- 48 seed-by-HOLDOUT-task PURE@B aggregates.

For PURE@B, successful calls contribute their frozen-parser canonical response and
failed calls contribute the common no-response vote. Replay invokes the already
frozen label-free modal aggregation and deterministic tie-break. Source event hashes
are retained for every 6- or 7-sample aggregate.

After all six decisions are frozen, the trusted grader issues an evaluation unlock
bound to their exact event fingerprints. The executor materializes one batch request
for 48 aggregates derived from all 330 source events. The terminal receipt retains
only each cell's aggregation/request/receipt fingerprints and Boolean correctness,
plus the batch request and receipt fingerprints; it stores neither individual-sample
grades nor the full trusted receipt.

Complete replay rejects missing, duplicate, reordered, re-chained-but-semantic-invalid,
or lineage-tampered events. Partial replay is available only when explicitly requested
for crash recovery; it cannot produce a terminal execution receipt.
