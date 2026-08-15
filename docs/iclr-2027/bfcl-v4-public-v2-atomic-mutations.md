# BFCL V4 public-development v2 atomic mutations

This document freezes a provider-free mutation grammar for the prospective
BFCL V4 public-development v2 campaign. It is an engineering design artifact,
not a BFCL result, a score-run grant, or evidence that a model can use any of
the mutations.

The corresponding typed implementation is
`spiral_harness/benchmark/bfcl_v4_public_v2_mutations.py`. Every catalogue,
proposal, harness, canary, and materialization contract in that module fixes
`score_bearing_execution_allowed=false`.

## Transfer boundary

The catalogue is informed by failure families observed during the earlier
public v1 development pilot. That prior public evidence may motivate a bounded
repair hypothesis, but it is not replayed as v2 evidence and it does not make
the v2 campaign independent of BFCL-style development.

The transfer direction is therefore explicit:

1. public v1 development failures motivate five generic repair families;
2. those families are frozen as a finite, task-agnostic v2 catalogue;
3. v2 questions, task coordinates, possible answers, grader state, score
   observations, and execution results are not inputs to catalogue
   construction, proposal, materialization, or canary replay;
4. later v2 evidence, if separately authorized and collected, must evaluate
   the frozen catalogue without revising it from observed v2 outcomes.

“V2 score-blind” describes this information boundary. It does not mean the
catalogue was designed without prior public BFCL knowledge. Any future result
must disclose the v1-to-v2 transfer and remain public-development evidence,
not hidden, sealed, official-full-suite, or confirmatory evidence.

## Closed atomic catalogue

The complete catalogue contains exactly five parameter-free entries, in the
following frozen order:

| Catalogue ID | Changed typed stage | Local deterministic operator | Motivating v1 failure family |
|---|---|---|---|
| `numeric-schema-lexicalizer` | `numeric-renderer` | `render-schema-number-lexemes/v1` | Numeric schema lexical form |
| `compound-clause-tool-coverage` | `clause-planner` | `require-every-clause-covered/v1` | Compound-request coverage |
| `required-argument-validator` | `argument-validator` | `block-missing-required-arguments/v1` | Required-argument completeness |
| `multiplicity-order-preserver` | `call-emitter` | `preserve-call-multiplicity-and-order/v1` | Call multiplicity and order |
| `schema-grounded-matcher` | `tool-matcher` | `match-tools-by-schema-capabilities/v1` | Schema-grounded tool selection |

Each entry owns one distinct typed stage and one distinct operator. The frozen
parent uses the same default implementation at all five stages. Materializing
an entry creates a content-addressed child that cites the exact parent and
replaces only the selected stage. The other four stages remain byte-for-byte
and value-for-value equal to the parent projection. A candidate whose parent
reference is wrong, whose stage roster or order changes, whose selected ID and
operator disagree, or whose parent and child references are equal is invalid.

This is a finite catalogue of pre-authored repair operators. Selecting one of
five bounded repairs is not open-ended mutation, autonomous invention, or a
demonstration of general self-evolution. A later performance gain could at
most support a claim about bounded repair selection under the separately
frozen campaign.

## Proposal grammar

The candidate-authored semantic payload is exactly one catalogue ID:

```json
{"catalogue_id":"required-argument-validator"}
```

The allowed values are the five IDs above. The proposal has no free-form
appendix, editable instruction text, parameters, task coordinates, answer
fields, or request for score execution. A transport layer may wrap the typed
value, but it must not enlarge this semantic grammar. Unknown IDs, multiple
IDs, extra fields, type coercions, free text, and task- or answer-conditioned
content are outside the contract and cannot be repaired into an admissible
proposal.

Repeated selections by independent proposal slots remain repeated selections;
the grammar does not silently resample or synthesize a replacement. Catalogue
closure prevents two entry IDs from naming the same typed operator or mutable
component.

## Content-addressed materialization

Catalogue, parent harness, candidate harness, and canary receipt identities
are computed from canonical JSON bytes and bind SHA-256, byte size, and an
exact versioned media type. A materialization joins all of the following:

- the canonical finite-catalogue reference;
- the strict one-ID proposal;
- the corresponding closed catalogue entry;
- the exact frozen parent value and reference;
- the deterministically reconstructed one-stage child and reference; and
- the deterministically replayed synthetic canary receipt.

Revalidation reconstructs the expected entry, child, and canary rather than
trusting caller-authored booleans. The parent and child references must differ,
so an exact structural no-op cannot become admissible. These checks establish
typed lineage and local reconstruction only; they do not establish provider
delivery or model behavior.

## Synthetic target and protected canaries

Every catalogue entry has two repository-owned synthetic cases. The target
case must change under the selected local operator, while the protected case
must remain unchanged:

| Catalogue ID | Synthetic target | Protected control |
|---|---|---|
| `numeric-schema-lexicalizer` | A schema-number rendering distinguishes a real-valued lexical form | An integer-valued field retains its integer form |
| `compound-clause-tool-coverage` | Both clauses in a synthetic compound request are covered | A single-clause request retains its one-call plan |
| `required-argument-validator` | A synthetic call missing a required field is blocked | A complete synthetic call remains emit-eligible |
| `multiplicity-order-preserver` | Repeated calls retain their original multiplicity and order | An already unique ordered sequence remains unchanged |
| `schema-grounded-matcher` | Capability matching selects the schema-compatible synthetic tool | An unambiguous first match remains unchanged |

Parent and candidate outputs are serialized as canonical JSON. A target
observation passes only when those local outputs differ. A protected
observation passes only when they are identical. Replay also checks that the
selected operator is active at exactly its owned stage. No model or provider
is called, and no benchmark item, possible answer, grader value, or score is
part of a canary.

The receipt fields named `local_operator_activation_verified`,
`deterministic_operator_adherence_verified`, and
`synthetic_behavior_change_verified` have a deliberately local scope: they
refer only to execution of the deterministic middleware operator on its
synthetic fixture. The same receipt fixes `model_invoked=false`,
`model_behavior_attested=false`, and `benchmark_capability_evidence=false`.
It does **not** show that a language model activated the operator, adhered to
an instruction, emitted a correct tool call, improved on BFCL, or generalized
to any public task. The provider-free canary proves local deterministic
behavior and structural closure, not model capability or benchmark evidence.

## Semantic-family and execution authority

Independent semantic-family review, arbitration, and split-disjointness
authority are prerequisites for any score-bearing v2 execution. Until that
separate authority has accepted the frozen roster and issued the appropriate
execution grant, score-bearing execution must remain disabled.

Acceptance by that authority would still not turn this mutation module into an
execution authority. The module does not load a roster, open an evaluation
partition, invoke a provider, grade a prediction, or issue an execution grant.
Its `assert_bfcl_v4_public_v2_score_execution_allowed` boundary always rejects.
A future score run would require a separate, frozen orchestrator to verify the
semantic-family authority, call plan, seeds, model and provider bindings,
budgets, information barriers, and all other campaign prerequisites.

Consequently, none of the following can authorize a score run:

- a valid catalogue entry;
- a well-formed proposal;
- a content-addressed candidate harness;
- a passing target/protected synthetic canary; or
- successful deterministic replay of a materialization.

## Permitted and forbidden claims

The provider-free artifacts may support only narrow engineering statements:

- the catalogue is finite and closed over five IDs;
- a proposal selects exactly one parameter-free entry;
- materialization changes exactly one typed stage of an exact parent;
- the resulting artifacts have reproducible canonical identities; and
- the local operator changes its synthetic target while preserving its
  synthetic control.

They do not support claims of BFCL accuracy, model improvement, successful
self-evolution, promotion, robustness, generalization, official evaluator
completion, hidden-test performance, or confirmatory evidence. Protocol
completeness and canary success must not be reported as performance success.
