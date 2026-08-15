# BFCL V4 public-v2 single-task request materializer

Status: provider-free control-plane implementation. It does not call a model,
read benchmark files, inspect possible answers, or authorize scoring.

## Boundary

`resolve_bfcl_v4_public_v2_task` accepts only the exact loaded public-v2
bundle and one structural `fit-NN`, `gate-NN`, or `holdout-NN` reference. The
suffix is resolved as a zero-based index within that split in frozen-manifest
order. Its receipt is trusted-control-plane only and retains the private task
ID, split, ordinal, function names, and content hashes.

`materialize_bfcl_v4_public_v2_request` additionally requires the exact
campaign plan, a deterministically derived node lineage, an exact frozen arm
treatment, and a media-typed semantic-development release `ArtifactRef`. It
revalidates the 1,098-node plan, reconstructs the lineage, and binds the node
hash, call slot, provider seed, task ref, arm, harness variant, question,
schemas, and prompt before returning.

Only `model_visible_request` may be dispatched. Its complete schema is:

- `model_route`;
- frozen `inference` settings;
- `provider_seed_u63`;
- `system_prompt`, or `null` for PURE/PURE@B;
- one canonical `question_json` value;
- that task's canonical `function_schemas_json` list.

It has no task ID, split, manifest, roster, lineage, node ID, answer, score,
result, semantic-release reference, or other task payload. The trusted wrapper
is deliberately not candidate-visible.

## Treatment closure

PURE and PURE@B are prompt-free. STATIC and adaptive `parent` nodes use the
exact static parent prompt. Candidate, nominated-candidate, and selected
variants accept either the exact candidate prompt or an explicit exact-parent
fallback. Revert and negative-control nodes accept only their corresponding
frozen runtime roles. Arm or harness-variant substitution fails closed.

## Semantic-release limitation

This batch binds only an opaque `ArtifactRef` with the exact semantic release
media type, a valid SHA-256 shape, and non-empty size. It does not load the
release, verify its HMAC or issuing authority, or invent an unlock protocol.
Consequently every materialization fixes:

- `semantic_release_authority_verified=false`;
- `semantic_release_content_loaded=false`;
- `provider_calls=0`;
- `score_bearing_execution_allowed=false`.

A later integration boundary must verify the release authority and issue a
separate execution grant before any provider or grader can be reached.
