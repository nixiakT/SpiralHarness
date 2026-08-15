# BFCL V4 public-development v2 semantic release

This authority closes the question-only semantic-isolation prerequisite for
the frozen BFCL v2 public-development campaign. It does not turn the public
roster into hidden, official, or confirmatory evidence.

The release consumes the 40-item opaque reviewer projection, its trusted
mapping, two independently generated model-assisted responses, one direct-call
execution attestation per response, and the live projection HMAC secret. It
revalidates the exact public BFCL row identity and reconstructs each candidate
payload from the packet question and function schemas before remapping any
opaque ID.

## Review and arbitration rule

The two primary reviewers receive the same frozen semantic-grouping prompt and
the same secret-permuted packet. They do not receive task IDs, source paths,
partition labels, possible answers, scores, benchmark-model outputs, or harness
outputs. Their declared principal, provider/model/revision route, and served
model-identity commitment must be distinct. Reusing the same declared model
under a different seed is rejected as an independent review.

Family labels are reviewer-local. The verifier compares the induced pairwise
equivalence relations over all 780 item pairs, so label spelling cannot create
or hide agreement. If the primary relations differ on any pair, a third
distinct review is mandatory. The complete third partition becomes the final
partition; mixing pairwise decisions from multiple partitions is avoided
because such a mixture need not remain transitive.

The final partition must contain no family spanning any two of `v1`, `fit`,
`gate`, or `holdout`. A consensus cross-boundary family and an adjudicator
cross-boundary family both fail closed. The private boundary check occurs only
after review generation and trusted remapping.

## Authority and integrity

Successful verification issues
`BfclV4PublicV2SemanticDevelopmentRelease`. The object binds:

- the frozen roster and 1,098-node campaign-plan fingerprints;
- exact packet, trusted mapping, mapped-review, response, distribution, and
  execution-attestation references;
- the label-invariant final semantic partition;
- the primary disagreement count and arbitration shape; and
- a domain-separated HMAC over every release field.

The live secret and complete evidence are required to recompute the release at
the execution boundary. Projection integrity or a reviewer response alone
still cannot authorize scoring.

## Explicit limitations

The gateway model catalog establishes route availability only. LiteLLM does
not provide a cryptographic signature for a completion or a verifiable hash of
the served weights. Consequently, the release truthfully fixes all of these to
false:

- external reviewer-identity verification;
- provider authentication of reviewer declarations;
- provider response signatures; and
- cryptographic served-weight verification.

The direct-session evidence is an operator attestation that binds credential-
free hashes of the request and raw response plus the provider response ID,
reported model, optional system fingerprint, exact input/output token usage,
seed, HTTP status, and single-attempt policy. Credentials and raw secrets are
never serialized.

The live adapter freezes one label-invariant grouping prompt, requires strict
JSON with exact packet-order coverage, disables redirects and environment
proxies, and permits exactly one provider attempt. It rejects truncation,
model-route mismatch, duplicate JSON keys, non-finite numbers, malformed token
usage, reordered opaque IDs, and markdown-wrapped output. The review request
contains the reviewer packet and protocol prompt only. The trusted mapping and
HMAC secret never cross the provider boundary.

This procedural authority is sufficient only for a transparent descriptive
public-development run. Every released result must retain
`official_bfcl_evidence=false`, `hidden_test_evidence=false`, and
`confirmatory_evidence=false`. It cannot support an official leaderboard or a
confirmatory ICLR claim.
