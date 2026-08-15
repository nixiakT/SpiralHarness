# BFCL V4 public-development v2 semantic reviewer projection

This protocol defines a provider-free reviewer packet for the complete
40-item semantic universe: 15 public-v1 pilot tasks plus 25 prospective-v2
public-development tasks. Its private trusted mapping, opaque-keyed response,
and integrity-only distribution receipt allow later review of both within-v2
and v1↔v2 family overlap. They are not semantic-audit evidence, reviewer
authentication, a score-run grant, or a benchmark result.

The typed implementation is split between
`bfcl_v4_public_v2_semantic_projection_contracts.py` and
`bfcl_v4_public_v2_semantic_projection.py`; each remains below the repository's
700-line architecture ceiling. The builder uses normal leaf-contract imports,
with no dynamic import. It imports neither a loader, semantic-audit verifier,
runner, grader, provider, nor execution backend.

An authorized trusted process must supply both an exact
`BfclV4LoadedPublicPilot` and an exact
`BfclV4LoadedPublicDevelopmentV2`, plus a process-local HMAC secret. Both
objects are strictly revalidated by their own frozen contracts before any
projection occurs. Missing, swapped, forged, or wrong bundle types fail
closed. The builder does not load either bundle itself.

## Outputs and trust boundary

The builder creates one trusted materialization containing two objects with
different release policies:

1. `BfclV4PublicV2SemanticReviewerPacket` is the only object that may cross the
   reviewer boundary.
2. `BfclV4PublicV2SemanticTrustedMapping` remains in the trusted control plane.

The combined `BfclV4PublicV2SemanticProjection` must never be sent to a
reviewer because it contains the private mapping. The runtime HMAC secret is
not serialized into the packet, mapping, projection, response, declaration,
or receipt.

The packet contains exactly 40 items. Each item contains exactly three fields:

- an HMAC-derived opaque item ID;
- the canonical public question JSON; and
- the canonical public function-schema JSON.

It contains no direct task ID, path, split label, source hash, candidate hash,
source-roster coordinate, HMAC order token, or secret. Packet items are ordered
by a secret-keyed HMAC permutation instead of source-roster order.

The private mapping retains the information needed by a future trusted
control-plane adapter: public task ID, source-roster ordinal, private boundary
label, path, question-blob identity, row identity, candidate-payload identity,
reviewer-content commitment, opaque ID, secret order token, and packet
position. Boundary labels are exactly `v1`, `fit`, `gate`, and `holdout`, with
populations 15, 5, 4, and 16. None appears in the reviewer packet.

The mapping separately binds the exact v1 loaded-bundle fingerprint and exact
v2 loaded-bundle fingerprint. A domain-separated source-universe fingerprint
commits to both fingerprints and the fixed 15+25 population. The mapping binds
the exact packet reference and carries an HMAC-SHA256 commitment over the
complete unsigned mapping. Duplicate public IDs, candidate-payload identities,
source coordinates, or opaque IDs fail closed.

## HMAC construction

The builder requires a `bytes` secret of at least 32 bytes. Secret quality,
generation, memory protection, rotation, and access control remain runtime
responsibilities; length validation alone does not prove entropy.

Opaque IDs, order tokens, and the mapping commitment use separate versioned
domains. Their messages are canonical JSON values and bind the combined
source-universe fingerprint. HMAC-SHA256 prevents someone who knows the finite
public task-ID vocabulary from checking a guessed ID against an opaque ID
without the runtime secret. Plain hashing or a public salt would not provide
that property.

The secret-keyed item order prevents the packet from directly copying roster
order. The private mapping records both source and packet coordinates so the
trusted plane can reverse the permutation after review.

The verifier recomputes all opaque IDs, order tokens, public-content
commitments, packet order, packet reference, and the mapping HMAC under the
live secret. A wrong secret or a change to any packet/mapping field fails
closed. A persisted distribution receipt is not self-authenticating and must
also be compared with a fresh live-secret recomputation.

HMAC is a capability inside the trusted process, not OS isolation, remote
identity authentication, or proof that an operator followed the protocol.

## Reidentification limitation

This protocol conceals direct mapping coordinates; it does not anonymize
public benchmark content. The packet intentionally includes exact public
question and function-schema content because reviewers need it for semantic
grouping. A reviewer may recognize that content, search for it in the public
BFCL corpus, or infer likely categories or partitions from prior knowledge.

Accordingly, every packet, mapping, and receipt records:

- `exact_public_content_may_be_reidentified=true`; and
- `partition_inference_cryptographically_prevented=false`.

Opaque IDs must not be described as cryptographic prevention of content-based
reidentification or partition inference. Any review report must disclose this
limitation.

## Reviewer declaration and response

A reviewer declaration records a declared principal, reviewer kind,
independent generation, non-use of a partition roster, and non-use of answers
or scores. To preserve the provenance required by the existing audit contract,
a human declaration requires a human-identity commitment. A model-assisted
declaration instead requires provider, model name, revision, served-weights
identity, prompt hash, and generation seed; partial or mixed coordinates are
rejected. These remain declarations only. The contract fixes both external
identity verification and declaration authentication to false.

Each response assignment is keyed only by the opaque packet ID and contains a
semantic-family label plus rationale. Direct task coordinates are forbidden by
the strict response schema. Trusted validation requires exact, unique,
ordered coverage of the packet's opaque IDs and exact packet/declaration
references.

Response validation does not remap assignments into the existing semantic
audit evidence type and does not make the response admissible. A future,
separately reviewed trusted adapter and authority mechanism would be required
to authenticate reviewer provenance, reverse the mapping, establish review
independence, adjudicate disagreements, and issue semantic-audit authority.

## Distribution receipt

Successful live-secret verification produces a receipt with status
`integrity-verified-external-authentication-blocked`. It can truthfully record
that packet bytes, mapping bytes, the HMAC commitment, opaque-ID derivation,
and secret permutation agree.

It deliberately fixes all stronger claims to false:

- external reviewer identity verification;
- declaration authentication;
- distribution-channel authentication;
- external packet-delivery verification;
- reviewer-response admissibility;
- semantic-audit authority; and
- score-bearing execution authority.

No caller option can set those claims to true. A future authentication system
must introduce a new authority contract rather than mutate this receipt.

## Execution boundary

The existing BFCL v2 semantic score guard is unchanged. This module does not
import it, bypass it, patch it, or produce its evidence type. Its own
`assert_bfcl_v4_public_v2_semantic_projection_score_execution_allowed`
boundary also always rejects.

None of the following permits a model call, grading, or score-bearing run:

- successful packet materialization;
- a valid HMAC mapping commitment;
- a passing packet/mapping integrity check;
- an opaque-keyed reviewer response; or
- a distribution receipt.

The implementation and tests use only synthetic 15+25 bundles; this batch does
not inspect real GATE/HOLDOUT question content. It reads no possible answers,
scores, run artifacts, or CAS results, invokes no model/provider, and supplies
no evidence of model capability, semantic-family disjointness, hidden-test
performance, or reportable results.
