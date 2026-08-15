# BFCL V4 provider-free fixture bridge

This bridge is a development fixture, not a BFCL result. Every published
contract fixes `partial_evaluation=true`, `hidden_test_evidence=false`,
`sealed_evidence=false`, `reportable_result=false`, and
`official_score_produced=false`.

## Bound public fixture

The fixture reads Git objects, rather than mutable worktree files, from Gorilla
commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. It uses the public
`simple_python_0` task and binds the complete question blob, exact JSONL row,
question payload, function schemas, and function-name roster by SHA-256.

Candidate-visible and grader-only data are deliberately separate:

- `BfclV4PublicAstFixture` contains the task and function schemas. It has no
  answer path, answer-row hash, answer-blob hash, or ground-truth hash.
- `BfclV4GraderOnlyAnswerBinding` is created only inside the isolated grading
  path. The score-free serializer never accepts or returns this type.

The adapter converts a provider-neutral call into BFCL's OpenAI-style result
row. For the positive fixture the row is structurally equivalent to:

```json
{"id":"simple_python_0","result":[{"calculate_triangle_area":"{\"base\":10,\"height\":5}"}]}
```

The arguments remain a JSON string because the pinned upstream OpenAI FC
decoder parses each string before calling the AST checker. The response row,
native response, serializer module source, and upstream decoder source are all
separately fingerprinted.

## Source-isolated rehearsal

`invoke_bfcl_v4_source_isolated_fixture` starts a fresh Python interpreter with
an allowlisted environment. API keys, Hugging Face tokens, proxy variables,
user Git configuration, and the parent process environment are not inherited.
The worker accepts only the system Git executable resolved from the platform's
default executable path and rechecks its exact path, bytes, size, and version.

The worker loads byte-exact copies of the pinned upstream `ast_checker.py`,
language enum, type mappings, and Java/JavaScript converter modules. It
recomputes both the candidate task binding and the grader-only answer binding
from the question blob, task row, task payload, possible-answer blob, answer
row, and ground truth it actually reads. Caller-provided hashes are therefore
not merely echoed. Positive and negative fixture calls exercise the exact
upstream Python AST checker.

The resulting `BfclV4SourceIsolatedReceipt` is grader/auditor-only. It embeds
the strict grader-only answer binding and records the runtime coordinates from
execution; revalidation does not silently substitute the verifier's current
runtime. Its validation closes the following internal substitution paths:

- the environment must equal the fixed, ordered name/value allowlist; changing
  order, duplicating an entry, or adding a credential or proxy is rejected even
  if the environment digest is recomputed;
- executed upstream files must equal the pinned ordered path/size/SHA-256
  roster, rather than merely matching a caller-generated bundle digest;
- Python, Git, worker, checkout, and argument paths are cross-bound, and the
  local worker bytes must match the currently installed pinned worker source;
- successful worker stdout is reconstructed from receipt fields and checked
  by exact size and SHA-256, while successful stderr must have been empty;
- `valid=true` requires a null normalized error, and `valid=false` requires a
  non-empty normalized error.

Candidate code receives only `BfclV4CandidateSafeFeedback`. That projection
contains the public task ID, a reference hash of the candidate's own response
row, the acceptance bit, `none | ast-mismatch`, and the permanent
public/partial/non-reportable flags. It contains no answer, ground truth,
grader binding, or receipt reference, and projection rejects an export paired
with a receipt from a different response.

This is a typed serialization boundary, not a process capability sandbox. An
orchestrator must withhold the grader receipt and worker filesystem/import
surface from untrusted candidate code and expose only the projection. The
bridge does not claim to enforce that deployment boundary by itself.

### Observation and attestation boundary

Executable path, byte size, SHA-256, and version fields are named as
observations. They bind the receipt internally but are not independent proof
of which executable the operating system launched. The same is true of the
recorded subprocess return code, stdout, stderr-emptiness, and runtime
coordinates: this bridge has no external attestor, transparency log, TPM, or
container/kernel provenance. Revalidating a saved receipt therefore checks its
internal bindings and the current pinned worker source; it does not turn those
observations into third-party execution evidence.

This is still not the complete official evaluator:

- the provider model registry is replaced with one explicit no-provider
  decoder coordinate;
- the full upstream dependency graph is not imported;
- no model is invoked and no aggregate accuracy is produced;
- there is no kernel/container egress sandbox. The receipt therefore records
  `network_isolation_attested=false`; it only states that this worker requested
  no network calls;
- public questions and public possible answers cannot provide hidden-test or
  confirmatory evidence.

## Full CLI contract

`build_bfcl_v4_full_cli_invocation_contract` separately binds the pinned BFCL
CLI entry point, evaluation runner, OpenAI-style decoder, package metadata,
argument template, and environment requirements. The four-file CLI roster is
validated by exact ordered path, size, and SHA-256 values, so replacing a path
and recomputing the aggregate digest is rejected. The contract permanently
records `official_cli_executed=false`, `invocation_ready=false`, and
`dependency_environment_attested=false`.

The blocker is upstream behavior: `bfcl evaluate` imports the full model
registry and instantiates a registered provider handler even when it is only
decoding existing result files. A reportable execution requires a separately
locked BFCL environment, a registered decoder matching the actual model result
format, an empty isolated `.env`, outbound-network isolation, and execution of
the complete official evaluator. The fixture bridge does not claim any of
those conditions.

## Verification

The unit suite covers canonical serialization, answer-data separation,
unchecked Pydantic construction, row tampering, unknown functions, task and
answer binding substitution, arbitrary Git executables, credential/proxy
environment leakage, the frozen full-CLI contract, and exact-source positive
and negative AST checks. Its second red-team layer mutates the embedded grader
binding, environment order/roster, executed source roster, execution paths,
worker source, reconstructed stdout, validity/error normalization, saved
runtime, and full-CLI roster. It also scans the candidate-safe projection
recursively for trusted-plane vocabulary, rejects cross-response projection,
rejects boolean and non-finite timeouts, and proves that changing any of the
actual answer blob, answer row, or ground truth changes the worker-computed
answer binding.

Exact-source tests use `BFCL_V4_PINNED_CHECKOUT` when set and otherwise use
`/tmp/spiral-bfcl-upstream`; they skip when no pinned checkout is available.
