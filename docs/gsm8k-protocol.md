# GSM8K benchmark and fixed-runner protocol

## Frozen upstream identity

SpiralHarness uses GSM8K as its first real benchmark. The registry pins the
archived `openai/grade-school-math` repository at commit
`3101c7d5072418e28b9008a6636bde82a006892c` and records the upstream MIT
licence and Copyright OpenAI 2021 in `THIRD_PARTY.yml`.

Dataset bytes are downloaded on demand and are not included in the wheel. A
file is published locally only after both its exact byte length and SHA-256
match:

| Source role | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| train | 7,473 | 4,166,206 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| test | 1,319 | 749,738 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

The downloader streams into a sibling temporary file, rejects oversized,
short, non-byte, or digest-mismatched responses, flushes the verified file,
and atomically replaces the target. A valid existing target is reverified and
reused without a network request.

Materialize and verify the registered snapshot with:

```bash
uv run spiral benchmark fetch gsm8k --output data/benchmarks/gsm8k
```

The command prints only file locations, counts, and frozen fingerprints. The
download directory is ignored by Git.

## Deterministic three-way split

GSM8K publishes train and test, but no official validation split. The default
SpiralHarness protocol therefore:

1. normalizes each train question with NFKC, case folding, whitespace
   normalization, and numeric-literal replacement;
2. hashes that normalized template family;
3. deterministically orders complete families with the frozen split seed;
4. assigns whole families to exploration or gate while minimizing deviation
   from the 20% gate target;
5. maps the official test source only to the sealed partition.

No normalized template family crosses any pair of partitions in the pinned
snapshot. Source-role fields in the trusted manifest also make a train item in
sealed—or a test item in exploration/gate—invalid. For the registered bytes and
default configuration, the frozen result is:

| Partition | Tasks |
|---|---:|
| exploration | 5,978 |
| gate | 1,495 |
| sealed | 1,319 |

The split-manifest fingerprint is
`2b5678c280095a8435f50f25f9c8d4e43af9530bac9663694b23ede899c52acc`;
the complete adapter fingerprint is
`7fc22b487fde502c0730265d5844962bacd1a5c98cc6dfab4d8beb962e29a1c2`.
The adapter checks these values at startup for the pinned default. Changing
membership or trusted semantics requires a versioned protocol change rather
than silently producing a new study under the old name.

This grouping is a deterministic leakage reduction, not a semantic-duplicate
oracle. Paraphrases with different lexical templates can still cross the
train-derived boundary, and GSM8K has been public since 2021, so target-model
pretraining contamination cannot be ruled out. Results must report those
limitations and must not treat this benchmark alone as broad agent capability.

## Trusted grading boundary

The candidate-facing `GSM8KTask` can represent only an opaque task ID and the
question. It has no answer, source index, or partition field. Gold rationales
and answers remain in the trusted adapter.

Grading preserves the upstream compatibility profile exactly:

```text
#### (\-?[0-9\.\,]+)
```

The first match is selected, commas are removed, and the resulting string is
compared exactly with the gold string extracted by the same rule. It is not
converted to a numeric value, so `1` and `1.0` are different. Missing markers
score zero for a completed execution; infrastructure failures remain unscored
with a typed non-completed status.

Before scoring, the adapter rejoins task ID and question and checks the exact
task, seed, harness, and paired execution fingerprint. Runner-reported output
and resource use are treated as score-free evidence; a worker cannot add a
score field to the strict execution schema.

## Fixed model and attempt accounting

`FrozenModelSpec` binds an explicit backend identity, model and tokenizer
revisions, runtime, and complete runner inference configuration. Moving
revision aliases such as `latest` and `main` are rejected. A paired execution
fingerprint includes the frozen spec, task, seed, and backend while excluding
the treatment prompt and harness ID; a separate request hash binds those arm
differences.

`FixedModelRunner` reserves a single-use attempt in the content-addressed
ledger before invoking a backend. Successful calls persist score-free
execution evidence and settle actual token use. Timeouts, exceptions, invalid
responses, and fingerprint drift persist a failed execution and burn the full
token reservation. A backend-reported usage overrun records the larger known
usage and permanently poisons that ledger stream, so no later reservation can
compound the violation. Settlement strictly parses the referenced
`ModelExecution`, binds its task/execution/request fingerprints to the
reservation, and derives token use from that artifact rather than a separate
caller-supplied number. The budget fingerprint is embedded in every
reservation and outcome, so reopening a tail under a looser budget is rejected.

This remains a trusted in-process runner boundary, not producer attestation.
Code with arbitrary access to the repository and ledger object is inside the
trusted computing base and could construct its own structurally valid execution
claim. Only attempt and token ceilings are hard-enforced here; monetary cost is
captured when a backend reports it but is not reserved or bounded by this
ledger. A live provider integration must add a trusted producer identity,
pricing/cost reservation, and conservative failure accounting before claiming
a hard monetary budget.

The included `ReplayBackend` keys records by the exact model spec and request;
it is suitable for deterministic tests and recorded experiments. No live
provider adapter is included in this batch. Provider credentials, networking,
OS sandboxing, hard cancellation of a blocking SDK, authoritative ledger heads,
and cross-process leases remain outside this in-process logical boundary.

The four matched experiment conditions and their information/resource rules
are frozen separately in [the baseline protocol](baseline-protocol.md).
