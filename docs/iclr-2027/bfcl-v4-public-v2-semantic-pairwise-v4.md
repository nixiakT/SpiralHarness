# BFCL v2 chunked pairwise semantic review v4

This protocol replaces, rather than retries, the three terminated 40-item
semantic-review lineages v1, v2, and v3. All three failures occurred in the
`primary-one` role; the predecessor closure therefore orders them by lineage,
not by reviewer role, and binds three distinct immutable failure references.
A v4 reviewer receives a new lineage, new seed root, and no predecessor request
or output.

## Fixed coverage

The existing HMAC-permuted 40-item packet is divided by packet order into five
contiguous blocks of eight items. The schedule is immutable:

- five within-block calls, each covering all `C(8, 2) = 28` pairs;
- ten cross-block calls, one for every pair of blocks, each covering all
  `8 * 8 = 64` pairs.

Thus each reviewer consumes exactly 15 single-attempt calls and classifies
`5 * 28 + 10 * 64 = 780 = C(40, 2)` unordered pairs exactly once. Pair
orientation and response order are inherited from the opaque packet order.
The plan validator reconstructs this manifest; a missing, duplicate, reversed,
or reordered pair is inadmissible.

## Reviewer boundary

Each call exposes only:

- a protocol and opaque call identifier;
- the relevant items' opaque IDs, canonical questions, and canonical function
  schemas;
- the exact ordered list of opaque-ID pairs to classify.

It does not expose public task IDs, source paths, private boundary labels,
possible answers, scores, model/harness outputs, the trusted mapping, or old
review outputs. The provider request builder is credential-free and performs no
network operation.

Completion parsing accepts strict UTF-8 JSON only. The top-level object must
contain exactly `decisions`; every row must contain exactly the two opaque IDs,
a JSON boolean, and a bounded rationale. Coverage and order must equal the call
specification. There is no retry, repair, pair backfill, or label inference.

## Replay and composite evidence

All 15 valid call sessions are replayed in plan order. Replay independently
rebuilds the request bodies, request fingerprints, 780-pair manifest, and source
response references. The true edges are closed into connected components, then
every one of the 780 decisions is checked against component membership. This
rejects non-transitive judgments such as `A~B`, `B~C`, and `A!~C`; replay never
silently repairs them into a partition.

The admitted artifact is explicitly a 15-call composite reviewer evidence
object. It contains 15 per-call operator-recorded single-attempt evidence
objects, the complete relation, and the canonical equivalence classes. It sets
`single_call_review=false` and `single_call_attestation_present=false`.
Provider response signatures, external reviewer identity, and served weights
remain unverified limitations.

## Composite release and activation

The v4 development release consumes two independent primary composite reviews.
If any of their 780 decisions differs, a third independent 15-call adjudicator
composite is mandatory; an unsolicited adjudicator is rejected when the
primaries agree. Reviewer nonces, seed roots, declared reviewer/model/weight
coordinates, and all provider response IDs must be distinct across the admitted
composites. The adjudicator must preserve every pair on which the primaries
agree; only primary disagreements may change. Its complete relation therefore
also proves that those resolutions remain jointly transitive. The selected
final equivalence classes are remapped only in the trusted plane. Any class
spanning a private semantic boundary rejects release.

The release binds the packet, live-HMAC-verified trusted mapping, predecessor
closure, composite evidence and relation references, disagreement count, final
partition, and 30- or 45-call evidence shape. It uses a new v4 media type and
does not masquerade as the legacy single-packet release. Live configuration
accepts either the legacy v1 release or the composite v4 release and records the
chosen evidence shape in its fingerprint.

This remains public-development evidence. It is not official BFCL evidence,
hidden-test evidence, confirmatory evidence, a provider signature, or proof of
served model weights.

## Execution-readiness boundary

The modules documented here are provider-free builders, parsers, replay
verifiers, and release issuers. They do not yet provide a production transport
runner, durable per-slot reservation/checkpoint journal, crash-burn recovery,
or a CLI route for the 15/30/45-call semantic campaign. Until that separate
execution layer exists and is tested, these contracts must not be treated as
authorization to start live reviewer calls. In particular, a malformed or
transport-failed slot must be durably consumed without retry; an operator
attestation alone is not a substitute for that runtime guarantee.
