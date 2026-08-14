# HarnessFaultBench v4: multi-surface trust-closure slice

HarnessFaultBench v4 is an executable integrity and mechanism slice. It is not
a live-model benchmark and is not evidence that an LLM solver or a harness
optimizer can self-evolve. The trusted runtime charges and digest-binds a base
model call, but deterministic benchmark middleware produces the final output.

## Frozen design

Each exploration, gate, and sealed partition contains seven independent fault
families spanning six mutable surfaces:

| Family | Surface |
| --- | --- |
| prompt instruction precedence | prompt |
| memory retrieval isolation | memory/retrieval |
| tool schema validation | tool |
| tool routing authorization | tool |
| middleware canonicalization | middleware |
| control-flow guard ordering | control flow |
| skill scope activation | skill |

The two tool families are independent. Consequently, a default partition has
7 families, 6 surfaces, and 28 scenarios—not 28 independent families.

Every family has exactly one case in each role:

| Role | Purpose | Routed-policy behavior rate |
| --- | --- | ---: |
| repairable target | requires the family repair | 100% |
| null control | requires preservation of correct legacy behavior | 100% |
| unrepairable control | gold is outside both runtime branches | 0% |
| distractor/shift hard-negative | reverses the tempting global route | 100% |

The unrepairable cases intentionally cap the routed-policy oracle at 75%. A
100% score would therefore indicate leakage or a verifier defect, not a better
allowed repair.

## Anti-cheat envelope

The authority recomputes the envelope from each hidden partition opening. It
does not accept rates supplied by a caller. With the default 28-scenario shape:

| Strategy | Correct | Rate | Gap to oracle |
| --- | ---: | ---: | ---: |
| routed-policy oracle | 21/28 | 75.00% | — |
| best constant branch selector | 11/28 | 39.29% | 35.71 pp |
| best single-family patch | 12/28 | 42.86% | 32.14 pp |
| best constant final output | 1/28 | 3.57% | 71.43 pp |

Gold outputs are unique and answer labels are balanced. Partition verification
fails if the best constant branch or single-family patch comes within 30
percentage points of the oracle, or if a constant output exceeds 5%.

## Compiler and runtime closure

The v4 compiler freezes seven component artifacts: system prompt, retrieval
policy, tool schema, tool router, middleware policy, control-flow policy, and
skill scope. Candidate and placebo artifacts are byte-length matched for every
component. Revert reuses the exact parent component refs. Verification rebuilds
the action, all component bytes, all five harness roles, and every parent edge.

The generic execution runner currently materializes only the system-prompt
projection. The benchmark-specific trusted runtime consumes and digest-binds
the full seven-component compilation graph. This projection is a deliberate
compatibility boundary; it must not be described as provider delivery of
memory, tools, middleware, control flow, or skills.

Final mechanism verification additionally requires three distinct live
`AttemptLedger` writers. It recovers the unique preflight from each exact
receipt set, replays trusted usage, compares the complete recomputed usage to
the persisted value, and reverifies the live ledger tails before returning.

## Claim boundary

The immutable verification artifact keeps all of these fields false:

- `base_model_output_drives_behavior`
- `supports_live_model_benchmark_claim`
- `supports_llm_solver_claim`
- `supports_optimizer_capability_claim`
- `supports_self_evolution_claim`

Multi-surface compilation shows that the trusted benchmark machinery can
attribute several deterministic surface faults without admitting constant
selectors. It does not show that a model discovered, implemented, or improved
those repairs.
