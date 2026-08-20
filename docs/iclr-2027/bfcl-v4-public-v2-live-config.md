# BFCL V4 public-development v2 live configuration

The live configuration binds the frozen 1,098-node DAG to one advertised
benchmark model route, one inference configuration, one native-function
transport implementation, the verified semantic-development release, and a
hard 1,086-attempt budget.

The exact benchmark route is `qwen36-35b-a3b`. A same-looking prefixed fallback
is not assumed weight-equivalent and cannot silently replace it. Every
`PURE`, `STATIC`, `SCORE`, `FULL`, and `PURE@B` call uses the same route and
inference settings. Paired provider seeds come from the preregistered global
DAG; semantic-review models are not part of the arm comparison.

The budget permits exactly one provider attempt per registered call slot. A
timeout, transport failure, malformed response, or rejected provider identity
burns that slot. Retry, backfill, and adaptive stopping remain disabled. The
maximum accounting envelope is 32,768 total tokens per call and 35,586,048
tokens over 1,086 slots; it is a ceiling, not expected usage.

The configuration can only be frozen from:

- the exact frozen campaign fingerprint and node schedule;
- an authenticated direct `/models` availability observation containing the
  unprefixed route;
- a process-local semantic capability minted by a complete legacy-v1 or
  composite-v4 live-HMAC verification immediately before freeze;
  and
- the serializer, parser, transport, and backend source fingerprints of the
  native function-call adapter.

LiteLLM route availability is not a served-weight, tokenizer, or runtime
attestation. Those identities remain explicitly opaque, so
`same_provider_weights_attested=false` even though exact request routing is
held fixed. This limitation must accompany every descriptive result.

The configuration authorizes only the frozen public-development execution.
It never sets official BFCL, hidden-test, or confirmatory evidence claims.
The capability itself is never serialized and does not retain the HMAC secret;
the config persists only its evidence shape, exact release ref/fingerprint, and
non-secret evidence-input fingerprint. Passing a hand-constructed release, a
copied model with a forged HMAC, or a cross-shape media type fails before freeze.
