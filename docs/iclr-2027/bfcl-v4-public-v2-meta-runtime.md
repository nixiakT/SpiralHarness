# BFCL V4 public-v2 diagnosis/proposal meta runtime

Status: provider-free public-development infrastructure. This runtime has not called a model,
loaded BFCL data, graded a response, or produced benchmark evidence. Its request and parse
artifacts fix `provider_calls=0` and do not authorize score-bearing execution.

## Runtime boundary

The implementation is split into three leaf modules, each below the repository's 700-line limit:

- `bfcl_v4_public_v2_meta_runtime_contracts.py` defines trusted evidence wrappers, the three
  model-visible payload schemas, strict parse results, and resolved atomic mutations;
- `bfcl_v4_public_v2_meta_runtime_native.py` owns the two fixed submit tools, prompt serialization,
  backend identity reads, executor-request reconstruction, and strict native response extraction;
- `bfcl_v4_public_v2_meta_runtime.py` joins the frozen campaign, live runtime, journal evidence,
  prompts, parsers, deterministic mutation materializer, and three-pipeline duplicate closure.

The public builders are:

- `materialize_bfcl_v4_public_v2_diagnosis_request`;
- `parse_bfcl_v4_public_v2_diagnosis_response`;
- `materialize_bfcl_v4_public_v2_proposal_request`;
- `parse_bfcl_v4_public_v2_proposal_response`; and
- `resolve_bfcl_v4_public_v2_proposal_batch`.

Builders read four secret-free backend fingerprints but never invoke a backend method. Native
requests contain exactly two messages (`system`, then `user`), exactly one controller submit tool,
the frozen model route and inference settings, and the provider seed registered on the target DAG
node. The request artifact binds the campaign, node schedule, mutation catalogue, semantic release,
live runtime, model spec, prompt/tool hashes, and the source hashes of all three runtime modules.

## Information boundary

Diagnosis nodes must receive their exact ten `allowed_evidence_from` parent-FIT events in DAG order.
Each event is strictly revalidated against its node hash, payload-free executor request, runtime,
semantic release, terminal provider disposition, and trusted binary-grade receipt. Cross-arm,
cross-seed, reordered, candidate-FIT, GATE, HOLDOUT, and unknown sources fail closed.

SCORE's complete model-visible payload contains only:

```json
{
  "controller": "diagnosis",
  "feedback_view": "fit-aggregate-binary-score-only",
  "binary_summary": {
    "observation_count": 10,
    "binary_correct_count": 0,
    "binary_incorrect_count": 10
  }
}
```

It cannot represent questions, schemas, responses, item-wise labels, task/node IDs, answers,
checker diagnostics, or a roster.

FULL additionally requires ten independently supplied FIT task projections, each bound to the
corresponding node reference and request-payload hash. Its visible observation schema contains
only `observation_index`, canonical `question_json`, canonical `function_schemas_json`, the model's
own canonical response, and one Boolean grade. Trusted projection fields such as source node IDs
and request hashes never enter the serialized payload. The question contract permits one exact
system/user conversation turn; function schemas reject private control-plane keys at their top
level. Extra payload fields are forbidden by Pydantic and an explicit model-boundary field-set
check.

A proposal node consumes exactly its own diagnosis event and no original FIT event or projection.
Its model sees the sanitized diagnosis status/text plus the complete five-ID frozen catalogue. The
submit tool accepts exactly one `catalogue_id` enum and no free-form appendix, parameter, coordinate,
answer, or additional field.

## Strict parsing and atomic closure

Both parsers require one source-bound `NativeFunctionCallResponse`, no assistant text, one call to
the exact submit tool, and one exact argument object. Request, serializer, parser, transport, and
tool-roster fingerprints must match the materialized request. Text-only output, multiple calls,
wrong tools, extra/missing fields, invalid response contracts, and response-binding changes become
typed terminal failures; there is no retry, repair, or backfill.

A known proposal ID is immediately passed through the existing deterministic atomic mutation and
prompt-runtime materializers. Catalogue mismatch, invalid materialization, or a defensive parent /
candidate prompt no-op fails closed. This establishes only local typed lineage; it does not attest
provider delivery, model adherence, behavior change, or benchmark capability.

The three proposal nodes of one arm/seed are parallel in the frozen DAG, so a single node cannot
observe its siblings. Duplicate closure therefore runs only after all three parse results freeze.
The lowest otherwise-admissible pipeline index keeps a repeated mutation ID; later repeats receive
the typed `duplicate` disposition, expose no admitted runtime batch, consume their original slot,
and are never resampled. Invalid diagnoses, provider failures, no-ops, and malformed proposals have
separate typed dispositions.

## Tests and claims

Unit tests use only synthetic questions, schemas, responses, binary receipts, and FIT projections.
They cover SCORE/FULL disclosure differences, exact evidence ordering, cross-boundary substitution,
private-field rejection, campaign/node/runtime/release/seed/prompt/tool/source tampering, strict
native parsing, deterministic materialization, no-op/invalid/duplicate closure, backend invocation
traps, network traps, and architecture limits.

These artifacts may support claims about protocol closure and deterministic request construction.
They do not support BFCL accuracy, improvement, self-evolution, promotion, hidden-test performance,
official evaluator completion, or an ICLR result.
