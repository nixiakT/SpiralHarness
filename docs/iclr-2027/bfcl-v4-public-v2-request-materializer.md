# BFCL V4 public-v2 single-task request materializer

Status: provider-free control-plane implementation. It does not call a model,
read benchmark files, or inspect possible answers. It materializes a
score-bearing request only after semantic authority was fully reverified.

## Boundary

`resolve_bfcl_v4_public_v2_task` accepts only the exact loaded public-v2
bundle and one structural `fit-NN`, `gate-NN`, or `holdout-NN` reference. The
suffix is resolved as a zero-based index within that split in frozen-manifest
order. Its receipt is trusted-control-plane only and retains the private task
ID, split, ordinal, function names, and content hashes.

`materialize_bfcl_v4_public_v2_request` additionally requires the exact
campaign plan, a deterministically derived node lineage, an exact frozen arm
treatment, and a non-serializable process-local verified semantic capability.
The capability may represent either legacy single-packet v1 evidence or
chunked pairwise composite v4 evidence. It revalidates the 1,098-node plan,
reconstructs the lineage, and binds the node hash, call slot, provider seed,
task ref, arm, harness variant, question, schemas, and prompt before returning.

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

## Semantic authority

A plain release object or `ArtifactRef` is rejected. The only accepted input is
minted in the current process after the appropriate verifier recomputes the
complete release: legacy v1 rechecks packet, mapping, review sessions and HMAC;
composite v4 additionally replays every 15-call relation before checking its
release HMAC. The capability binds evidence shape, release ref, release
fingerprint, and a fingerprint of every non-secret evidence input.

The capability cannot be normally constructed, mutated, copied, pickled, or
JSON-serialized. It is registered and sealed in its issuing process. Neither
the projection/release HMAC secret nor the complete review evidence is retained
inside it. The durable materialization records only the public binding and
therefore fixes `semantic_release_authority_verified=true`,
`score_bearing_execution_allowed=true`, and `provider_calls=0`.
