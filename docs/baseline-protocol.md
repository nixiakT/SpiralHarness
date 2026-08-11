# Four-condition baseline protocol

## Scope

This protocol freezes the structural prerequisites for a reproducible
comparison among exactly four conditions:

1. `static`;
2. `random-valid`;
3. `prompt-only`;
4. `evidence-targeted`.

The implementation in `experiments/baselines.py` is a configuration planner
and validator. It does not generate candidates, invoke an optimizer model, run
a benchmark, or choose a winner. Those execution loops belong to the next
stage and must consume a validated plan rather than reconstructing one from
command-line defaults.

## Frozen equality boundary

Every condition carries an independently serializable copy of the same:

- content-addressed benchmark manifest;
- fixed model fingerprint;
- inference-settings fingerprint;
- runtime/runner fingerprint;
- content-addressed seed harness;
- typed mutation grammar and policy;
- proposal master random seed;
- independent search-run seed schedule;
- paired repeat-seed schedule;
- evaluation, feedback-query, proposal, optimizer-model-call, token,
  wall-time, and cost ceilings.

`BaselineStudyPlan` rejects a missing or duplicate condition and rejects any
field drift among the four copies. Repeating shared values in each arm is
intentional: it makes a standalone arm auditable and lets the study validator
detect accidental per-condition overrides.

The protocol has two deliberately separate replication levels. The sorted,
unique `search_run_seeds` tuple contains at least two independent end-to-end
search runs. Within every search run, the sorted, unique `repeat_seeds` tuple
contains at least two paired task rollouts. The task pairing key is always
`(task_id, repeat_seed)` and complete pairs are required. Every final usage
report must name both exact frozen tuples. Changing or dropping a search run or
a task repeat creates an invalid comparison rather than a smaller study.

## Condition profiles

The common resource ceiling is availability, not a requirement to burn the
entire allocation. The conditions differ only in the intended treatment:
information used to choose candidates and the permitted mutation mechanism.
Evaluation, feedback-query, proposal, optimizer-call, token, and wall-time
ceilings must all be positive so a nominal four-condition plan cannot collapse
into a zero-work comparison. A zero monetary ceiling remains valid for a free
local/replay backend. The next-stage runner must additionally check that the
frozen evaluation budget can cover its trusted task roster crossed with every
search-run and repeat seed; that minimum cannot be derived from an opaque
benchmark artifact reference at this planning layer.

| Condition | Search feedback permission | Mutation capability | Optimizer model |
|---|---|---|---|
| `static` | Benchmark metadata only | None; the seed harness is unchanged | Forbidden |
| `random-valid` | Benchmark metadata, exploration inputs, aggregate gate feedback | Uniformly sampled valid atomic mutations from the full frozen grammar | Forbidden |
| `prompt-only` | Random-valid permissions plus exploration aggregates, item feedback, and trajectories | Prompt replacement only | Allowed within the common ceiling |
| `evidence-targeted` | The same ordinary exploration evidence as prompt-only, plus typed diagnostic evidence | Evidence-targeted atomic mutations over the full frozen grammar | Allowed within the common ceiling |

The table defines exact profiles, not minimum permissions. An arm cannot add a
feedback source or component kind. The mutation policy itself is identical in
all arms; a condition-specific capability is a subset or selection rule over
that shared policy. This distinction prevents a prompt-only condition from
silently receiving a different artifact-size limit or grammar.

Prompt-only receives the same ordinary exploration cases, outcomes, and traces
as evidence-targeted. It is distinguished by its prompt-only action space and
the absence of the typed causal diagnosis/mechanism-targeting contract—not by
withholding the failure evidence needed to make it a credible optimizer.

## Gate and sealed information boundary

No condition may read gate or sealed item-level inputs, outputs, scores, or
trajectories. The paired evaluation plan hard-codes gate feedback as
`aggregate-only`, sealed feedback as `final-aggregate-only`, and both
item-content exposure flags as false. The feedback enum retains explicit
`gate-item-content` and `sealed-item-content` values only so a malformed plan or
usage report fails with a precise error.

Aggregate gate feedback may be consumed only by a search condition and counts
against `max_feedback_queries`. The final sealed aggregate is a reporting
output after search closes; it is not proposer feedback. A future runner must
enforce this information flow. These Pydantic models define and audit the
contract but do not provide a worker sandbox.

## Availability versus use

`ResourceCeilings` records what each condition was allowed to consume.
`ResourceUsage` records what the caller declares was consumed. A
`BaselineUsageReport` contains both values, the study-plan fingerprint, the
reported repeat seeds, the reported independent search-run seeds, the declared
feedback types, and the component kinds declared as mutated.

The distinction is necessary for the static condition. Static receives the
same available ceilings so the allocation is not changed after condition
assignment, but it performs no search. It must report zero feedback queries,
proposals, optimizer-model calls, and mutated component kinds. It can still
report evaluation, token, wall-time, and cost use incurred while evaluating the
unchanged seed harness. It may not claim unused search resources as consumed.

`random-valid` likewise cannot claim optimizer-model calls. Prompt-only cannot
claim a skill, memory, tool, middleware, or control-flow mutation. Every arm
may consume less than its ceiling, but no field may exceed it.

## Validation before comparison

`BaselineProtocolValidator` revalidates canonical model content so unchecked
Pydantic copies cannot bypass the protocol. A structural consistency report is
emitted only after all of the following hold:

- the plan and usage reports each contain exactly the four required conditions;
- all frozen contexts, paired schedules, and available ceilings match;
- every usage report is bound to the exact plan fingerprint;
- every arm declares the exact independent search-run seeds;
- every arm declares the exact paired task-repeat seeds;
- reported use is within every resource ceiling;
- declared feedback is a subset of the condition's permission set;
- declared component kinds are a subset of the condition's capability;
- static declares zero search consumption.

Failure is typed and fail-closed at this structural boundary. The validator
does not normalize inconsistent reports, exclude an over-budget claim, impute
a missing condition, or pretend that an unused allocation was consumed.

This is intentionally not yet an attested fairness result. The aggregate usage
objects are caller-authored claims, so the report explicitly records
`execution_attested=false` and `evidence_scope=self-reported-aggregate-claims`.
The next-stage search runner must derive those claims from controller-owned
proposal, feedback, execution, and accounting events before scores may be
presented as a fair four-condition comparison.

## Reproducible execution handoff

A later baseline runner should persist the validated plan first, use its
fingerprint as the experiment identity, and bind every proposal, evaluation,
and usage event back to that identity. Random-valid sampling must derive from
the frozen grammar. For every condition, the strategy random stream for one
independent run must be deterministically derived from the frozen proposal
master seed together with that run's `search_run_seed`; a search-run seed is
never reused as a task rollout seed. Prompt-only and evidence-targeted model
calls must use the frozen model/inference/runtime context and settle against the
same model-call, token, wall-time, and cost ledger.

This batch intentionally stops before that runner and before the automatic
diagnosis/proposal/search loop. It establishes the comparison contract those
systems must obey.
