"""Provider-free contracts for the prospective BFCL V4 public v2 campaign.

The module contains only structural task references (``fit-00`` and similar),
budgets, seeds, and ordering rules.  It deliberately contains no question,
answer, grader, model output, or score payload.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import InferenceConfig

BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_ID = (
    "bfcl-v4-public-development-v2-three-search-seeds-global-dag-v1"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_OUTER_SEEDS_U64 = (
    2_026_081_601,
    2_026_081_602,
    2_026_081_603,
)
BfclV4PublicDevelopmentV2OuterSeed = Literal[
    2_026_081_601,
    2_026_081_602,
    2_026_081_603,
]
BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_IDS = (
    "search-seed-2026081601",
    "search-seed-2026081602",
    "search-seed-2026081603",
)

BFCL_V4_PUBLIC_DEVELOPMENT_V2_FIT_TASK_REFS = tuple(f"fit-{index:02d}" for index in range(5))
BFCL_V4_PUBLIC_DEVELOPMENT_V2_GATE_TASK_REFS = tuple(f"gate-{index:02d}" for index in range(4))
BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS = tuple(
    f"holdout-{index:02d}" for index in range(16)
)

BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE = "qwen36-35b-a3b"
BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE = InferenceConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=2_048,
    timeout_seconds=120.0,
    stop_sequences=(),
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING = 32_768

BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROVIDER_SEED_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-provider-seed/v1"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_REMAINDER_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-pure-at-b-remainder/v1"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_TIE_BREAK_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-pure-at-b-modal-tie/v1"
)

# These values are populated from the provider-free builder and then checked
# fail-closed.  They bind all 1,098 nodes (1,086 calls plus 12 control nodes).
BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256 = (
    "14870d5a39c14511385e348a0fef3f4f7e3eb2d2956152da24da43d048aaea76"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_SCHEDULE_SHA256 = (
    "c79e05db8a72eccf7eb7dff3e9a6bd60307d30a3dbae1954423b31ce5d4b8e11",
    "125a8a7819d3d8732f07524c49549d3eafb9a657ab56d45d844d2d60bd78d9ee",
    "fd6ef864d507994e4a14af93a0cb86ee4b104fbdeeaf4a63187839238bdcc89d",
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_REPLICATE_TOPOLOGY_SHA256 = (
    "3842e911db5d14c148580b3437820112d7b01e6c60e4e441addfce5efccc3d4a"
)
BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT = (
    "29e9729c6b374b733dcd6da7b95d7c662d826415ac0878b65df1d93893a324e6"
)


class BfclV4PublicDevelopmentV2Arm(StrEnum):
    """Five same-model comparison arms."""

    PURE = "pure"
    STATIC = "static"
    SCORE = "score"
    FULL = "full"
    PURE_AT_B = "pure-at-b"


BFCL_V4_PUBLIC_DEVELOPMENT_V2_ARM_CALL_COUNTS = MappingProxyType(
    {
        BfclV4PublicDevelopmentV2Arm.PURE: 16,
        BfclV4PublicDevelopmentV2Arm.STATIC: 16,
        BfclV4PublicDevelopmentV2Arm.SCORE: 110,
        BfclV4PublicDevelopmentV2Arm.FULL: 110,
        BfclV4PublicDevelopmentV2Arm.PURE_AT_B: 110,
    }
)


class BfclV4PublicDevelopmentV2Stage(StrEnum):
    """The global barriers in their only permitted order."""

    PARENT_FIT = "01-parent-fit"
    DIAGNOSIS = "02-diagnosis"
    PROPOSAL = "03-proposal"
    CANDIDATE_FIT = "04-candidate-fit"
    NOMINATION = "05-nomination"
    GATE = "06-gate"
    DECISION = "07-decision"
    EVALUATION = "08-evaluation"


class BfclV4PublicDevelopmentV2NodeKind(StrEnum):
    """Model-call and deterministic control-node kinds."""

    PARENT_FIT = "parent-fit"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    CANDIDATE_FIT = "candidate-fit"
    NOMINATION = "nomination"
    GATE = "gate"
    GATE_DECISION = "gate-decision"
    HOLDOUT = "holdout"
    PURE_AT_B_SAMPLE = "pure-at-b-sample"


class BfclV4PublicDevelopmentV2FeedbackView(StrEnum):
    """The sole intentional SCORE/FULL controller-input difference."""

    NONE = "none"
    SCORE_ONLY = "fit-aggregate-binary-score-only"
    CANDIDATE_SAFE_FULL = "fit-own-response-binary-and-coarse-failure"


class BfclV4PublicDevelopmentV2GateVariant(StrEnum):
    """Paired GATE variants for the deterministic promotion rule."""

    PARENT = "parent"
    NOMINATED_CANDIDATE = "nominated-candidate"
    REVERT = "revert"
    NEGATIVE_CONTROL = "negative-control"


_MODEL_CALL_KINDS = frozenset(
    {
        BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
        BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        BfclV4PublicDevelopmentV2NodeKind.GATE,
        BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
    }
)


class BfclV4PublicDevelopmentV2ExecutionProfile(ImmutableModel):
    """Same route, inference settings, and per-call ceilings for every role."""

    schema_version: Literal["1"] = "1"
    model_route: Literal["qwen36-35b-a3b"] = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
    inference: InferenceConfig = BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE
    per_call_total_token_ceiling: Literal[32_768] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING
    )
    per_call_output_token_ceiling: Literal[2_048] = 2_048
    per_call_timeout_seconds: Literal[120.0] = 120.0
    provider_attempts_per_call: Literal[1] = 1
    automatic_retries_allowed: Literal[False] = False
    retry_backfill_allowed: Literal[False] = False
    adaptive_stopping_allowed: Literal[False] = False
    same_model_all_arms_roles_and_seeds_required: Literal[True] = True
    same_inference_all_arms_roles_and_seeds_required: Literal[True] = True
    same_per_call_ceilings_all_arms_roles_and_seeds_required: Literal[True] = True
    provider_seed_requested_on_every_call: Literal[True] = True
    provider_seed_honoring_attested: Literal[False] = False
    exact_weight_revision_attested: Literal[False] = False
    tokenizer_identity_attested: Literal[False] = False
    serving_runtime_attested: Literal[False] = False

    @model_validator(mode="after")
    def _bind_inference(self) -> Self:
        if self.inference != BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE:
            raise ValueError("v2 inference settings differ from the frozen same-model profile")
        if self.per_call_output_token_ceiling != self.inference.max_output_tokens:
            raise ValueError("per-call output ceiling differs from frozen inference")
        if self.per_call_timeout_seconds != self.inference.timeout_seconds:
            raise ValueError("per-call timeout differs from frozen inference")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


BFCL_V4_PUBLIC_DEVELOPMENT_V2_EXECUTION_PROFILE = BfclV4PublicDevelopmentV2ExecutionProfile()


class BfclV4PublicDevelopmentV2NominationRule(ImmutableModel):
    """Target-derived FIT-score nomination with answers hidden from the optimizer."""

    schema_version: Literal["1"] = "1"
    rule_id: Literal[
        "valid-typed-candidate-max-fit-binary-count-then-lowest-slot-parent-fallback-v1"
    ] = "valid-typed-candidate-max-fit-binary-count-then-lowest-slot-parent-fallback-v1"
    candidate_count: Literal[3] = 3
    candidate_fit_calls_per_candidate: Literal[10] = 10
    typed_atomic_edit_required: Literal[True] = True
    duplicate_is_invalid: Literal[True] = True
    no_op_is_invalid: Literal[True] = True
    malformed_is_invalid: Literal[True] = True
    proposal_provider_failure_is_invalid: Literal[True] = True
    invalid_candidate_uses_parent_in_all_frozen_evaluation_slots: Literal[True] = True
    invalid_candidate_cannot_be_nominated: Literal[True] = True
    no_valid_candidate_nominates_parent_fallback: Literal[True] = True
    primary_rank: Literal["descending-fit-binary-correct-count"] = (
        "descending-fit-binary-correct-count"
    )
    tie_break: Literal["ascending-candidate-slot"] = "ascending-candidate-slot"
    binary_grader_outcomes_used: Literal[True] = True
    target_derived_fit_scores_used: Literal[True] = True
    trusted_grader_score_provenance_required: Literal[True] = True
    possible_answers_visible_to_nomination_controller: Literal[False] = False
    possible_answers_visible_to_candidates: Literal[False] = False
    checker_diagnostics_visible_to_nomination_controller: Literal[False] = False
    raw_targets_exposed_to_nomination_controller: Literal[False] = False
    gate_or_holdout_evidence_used: Literal[False] = False
    nomination_consumes_model_call: Literal[False] = False
    manual_override_allowed: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicDevelopmentV2PromotionRule(ImmutableModel):
    """Fail-closed GATE rule applied only after all six nominations freeze."""

    schema_version: Literal["1"] = "1"
    rule_id: Literal[
        "admissible-all-attempts-succeeded-shadow-controls-exact-fit-"
        "nondecreasing-gate-strictly-better-else-parent-v1"
    ] = (
        "admissible-all-attempts-succeeded-shadow-controls-exact-fit-"
        "nondecreasing-gate-strictly-better-else-parent-v1"
    )
    nominated_candidate_must_be_admissible: Literal[True] = True
    all_parent_candidate_fit_and_gate_attempts_must_succeed: Literal[True] = True
    revert_and_negative_control_must_exactly_match_parent: Literal[True] = True
    nominated_candidate_fit_must_be_nondecreasing: Literal[True] = True
    nominated_candidate_gate_must_be_strictly_better: Literal[True] = True
    failure_action: Literal["parent-fallback"] = "parent-fallback"
    invalid_candidate_forces_parent_fallback: Literal[True] = True
    gate_decision_consumes_model_call: Literal[False] = False
    holdout_evidence_used: Literal[False] = False
    manual_override_allowed: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


BFCL_V4_PUBLIC_DEVELOPMENT_V2_NOMINATION_RULE = BfclV4PublicDevelopmentV2NominationRule()
BFCL_V4_PUBLIC_DEVELOPMENT_V2_PROMOTION_RULE = BfclV4PublicDevelopmentV2PromotionRule()


def bfcl_v4_public_development_v2_pure_at_b_remainder_priority(task_ref: str) -> str:
    """Return the label-free priority for assigning one of fourteen extra calls."""

    if task_ref not in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS:
        raise ValueError("PURE@B remainder priority requires a structural HOLDOUT reference")
    return canonical_sha256(
        {
            "domain": BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_REMAINDER_DOMAIN,
            "manifest_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
            "task_ref": task_ref,
        }
    )


class BfclV4PublicDevelopmentV2PureAtBAllocation(ImmutableModel):
    """One structural HOLDOUT task's prospective PURE@B allocation."""

    schema_version: Literal["1"] = "1"
    task_ref: NonEmptyStr
    base_samples: Literal[6] = 6
    receives_remainder_sample: bool
    sample_count: Annotated[int, Field(ge=6, le=7, strict=True)]
    remainder_priority_sha256: Sha256

    @model_validator(mode="after")
    def _bind_task_priority(self) -> Self:
        if self.task_ref not in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS:
            raise ValueError("PURE@B allocation is outside the structural HOLDOUT roster")
        expected_priority = bfcl_v4_public_development_v2_pure_at_b_remainder_priority(
            self.task_ref
        )
        if self.remainder_priority_sha256 != expected_priority:
            raise ValueError("PURE@B remainder priority changed")
        if self.sample_count != self.base_samples + int(self.receives_remainder_sample):
            raise ValueError("PURE@B sample count differs from base-plus-remainder allocation")
        return self


def expected_bfcl_v4_public_development_v2_pure_at_b_allocation() -> tuple[
    BfclV4PublicDevelopmentV2PureAtBAllocation, ...
]:
    """Build the fixed 6-per-task base plus fourteen label-free remainder calls."""

    priorities = {
        task_ref: bfcl_v4_public_development_v2_pure_at_b_remainder_priority(task_ref)
        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS
    }
    remainder = frozenset(
        sorted(priorities, key=lambda task_ref: (priorities[task_ref], task_ref))[:14]
    )
    return tuple(
        BfclV4PublicDevelopmentV2PureAtBAllocation(
            task_ref=task_ref,
            receives_remainder_sample=task_ref in remainder,
            sample_count=7 if task_ref in remainder else 6,
            remainder_priority_sha256=priorities[task_ref],
        )
        for task_ref in BFCL_V4_PUBLIC_DEVELOPMENT_V2_HOLDOUT_TASK_REFS
    )


class BfclV4PublicDevelopmentV2PureAtBAggregationSpec(ImmutableModel):
    """Total, target-free modal aggregation over canonical response strings."""

    schema_version: Literal["1"] = "1"
    algorithm_id: Literal["modal-canonical-response-with-no-response-vote-v1"] = (
        "modal-canonical-response-with-no-response-vote-v1"
    )
    canonical_response_source: Literal["frozen-parser-canonical-response-string"] = (
        "frozen-parser-canonical-response-string"
    )
    failed_or_invalid_call_vote: Literal["one-shared-no-response-vote"] = (
        "one-shared-no-response-vote"
    )
    primary_order: Literal["descending-vote-count"] = "descending-vote-count"
    tie_break: Literal["ascending-domain-separated-vote-sha256-then-canonical-json"] = (
        "ascending-domain-separated-vote-sha256-then-canonical-json"
    )
    empty_input_result: Literal["no-response"] = "no-response"
    adaptive: Literal[False] = False
    target_labels_used: Literal[False] = False
    grader_feedback_used: Literal[False] = False
    possible_answers_used: Literal[False] = False
    total_for_all_finite_input_tuples: Literal[True] = True

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION = (
    BfclV4PublicDevelopmentV2PureAtBAggregationSpec()
)


class BfclV4PublicDevelopmentV2PureAtBAggregationResult(ImmutableModel):
    """A deterministic target-free aggregation result."""

    schema_version: Literal["1"] = "1"
    aggregation_spec_fingerprint: Sha256
    sample_count: Annotated[int, Field(ge=0, strict=True)]
    modal_count: Annotated[int, Field(ge=0, strict=True)]
    tied_for_mode: bool
    selected_canonical_response: str | None
    selected_no_response: bool

    @model_validator(mode="after")
    def _close_result_shape(self) -> Self:
        if (
            self.aggregation_spec_fingerprint
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
        ):
            raise ValueError("PURE@B aggregation spec fingerprint changed")
        if self.selected_no_response is not (self.selected_canonical_response is None):
            raise ValueError("PURE@B selected response and no-response flag disagree")
        if self.sample_count == 0:
            if self.modal_count != 0 or not self.selected_no_response or self.tied_for_mode:
                raise ValueError("empty PURE@B aggregation must return one unambiguous no-response")
        elif self.modal_count < 1 or self.modal_count > self.sample_count:
            raise ValueError("PURE@B modal count is outside the sample count")
        return self


def aggregate_bfcl_v4_public_development_v2_pure_at_b(
    canonical_responses: tuple[str | None, ...],
) -> BfclV4PublicDevelopmentV2PureAtBAggregationResult:
    """Select the label-free mode; ``None`` is the shared burned-slot vote.

    The function is total for every finite exact tuple of strings and ``None``.
    It neither accepts nor has access to targets, grades, or checker feedback.
    """

    if not isinstance(canonical_responses, tuple) or any(
        response is not None and not isinstance(response, str) for response in canonical_responses
    ):
        raise TypeError("canonical_responses must be an exact tuple of strings or None")
    if not canonical_responses:
        return BfclV4PublicDevelopmentV2PureAtBAggregationResult(
            aggregation_spec_fingerprint=(
                BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
            ),
            sample_count=0,
            modal_count=0,
            tied_for_mode=False,
            selected_canonical_response=None,
            selected_no_response=True,
        )

    counts = Counter(canonical_responses)
    modal_count = max(counts.values())
    modes = tuple(value for value, count in counts.items() if count == modal_count)

    def tie_key(value: str | None) -> tuple[str, bytes]:
        vote = {"kind": "no-response"} if value is None else {"kind": "response", "value": value}
        digest = canonical_sha256(
            {
                "domain": BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_TIE_BREAK_DOMAIN,
                "vote": vote,
            }
        )
        return digest, canonical_json_bytes(vote)

    selected = min(modes, key=tie_key)
    return BfclV4PublicDevelopmentV2PureAtBAggregationResult(
        aggregation_spec_fingerprint=(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_PURE_AT_B_AGGREGATION.fingerprint
        ),
        sample_count=len(canonical_responses),
        modal_count=modal_count,
        tied_for_mode=len(modes) > 1,
        selected_canonical_response=selected,
        selected_no_response=selected is None,
    )


def bfcl_v4_public_development_v2_node_id(
    *,
    replicate_id: str,
    arm: BfclV4PublicDevelopmentV2Arm,
    kind: BfclV4PublicDevelopmentV2NodeKind,
    task_ref: str | None,
    rollout_index: int | None,
    sample_index: int | None,
    pipeline_index: int | None,
    candidate_index: int | None,
    gate_variant: BfclV4PublicDevelopmentV2GateVariant | None,
) -> str:
    """Return a globally unique ID from structural, answer-free coordinates."""

    def part(prefix: str, value: object | None) -> str:
        if isinstance(value, StrEnum):
            value = value.value
        return f"{prefix}{'-' if value is None else value}"

    return "/".join(
        (
            replicate_id,
            arm.value,
            kind.value,
            task_ref or "aggregate",
            part("r", rollout_index),
            part("s", sample_index),
            part("p", pipeline_index),
            part("c", candidate_index),
            part("v", gate_variant),
        )
    )


class BfclV4PublicDevelopmentV2DagNode(ImmutableModel):
    """One exact model-call slot or deterministic barrier artifact."""

    schema_version: Literal["1"] = "1"
    node_slot: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    campaign_call_slot: Annotated[int, Field(ge=0, lt=1_086, strict=True)] | None
    replicate_node_slot: Annotated[int, Field(ge=0, lt=366, strict=True)]
    replicate_call_slot: Annotated[int, Field(ge=0, lt=362, strict=True)] | None
    arm_slot: Annotated[int, Field(ge=0, lt=110, strict=True)] | None
    node_id: NonEmptyStr
    replicate_id: NonEmptyStr
    outer_seed_u64: BfclV4PublicDevelopmentV2OuterSeed
    arm: BfclV4PublicDevelopmentV2Arm
    stage: BfclV4PublicDevelopmentV2Stage
    kind: BfclV4PublicDevelopmentV2NodeKind
    task_ref: NonEmptyStr | None = None
    rollout_index: Annotated[int, Field(ge=0, le=2, strict=True)] | None = None
    sample_index: Annotated[int, Field(ge=0, le=6, strict=True)] | None = None
    pipeline_index: Annotated[int, Field(ge=0, le=2, strict=True)] | None = None
    candidate_index: Annotated[int, Field(ge=0, le=2, strict=True)] | None = None
    gate_variant: BfclV4PublicDevelopmentV2GateVariant | None = None
    harness_variant: NonEmptyStr
    feedback_view: BfclV4PublicDevelopmentV2FeedbackView = (
        BfclV4PublicDevelopmentV2FeedbackView.NONE
    )
    provider_seed_u63: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)] | None
    depends_on: tuple[NonEmptyStr, ...] = ()
    allowed_evidence_from: tuple[NonEmptyStr, ...] = ()
    consumes_model_call: bool
    max_provider_attempts: Literal[1] | None
    failure_consumes_frozen_slot: bool
    retry_allowed: Literal[False] = False
    backfill_allowed: Literal[False] = False
    grader_feedback_available: bool = False
    typed_atomic_edit_required: bool = False

    @model_validator(mode="after")
    def _close_node(self) -> Self:
        expected_id = bfcl_v4_public_development_v2_node_id(
            replicate_id=self.replicate_id,
            arm=self.arm,
            kind=self.kind,
            task_ref=self.task_ref,
            rollout_index=self.rollout_index,
            sample_index=self.sample_index,
            pipeline_index=self.pipeline_index,
            candidate_index=self.candidate_index,
            gate_variant=self.gate_variant,
        )
        if self.node_id != expected_id:
            raise ValueError("v2 DAG node ID differs from its structural coordinates")
        if self.node_id in self.depends_on or len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("v2 DAG dependencies must be unique and non-reflexive")
        if len(self.allowed_evidence_from) != len(set(self.allowed_evidence_from)):
            raise ValueError("v2 allowed-evidence references must be unique")
        if not set(self.allowed_evidence_from).issubset(self.depends_on):
            raise ValueError("v2 evidence inputs must also be timing dependencies")

        is_call = self.kind in _MODEL_CALL_KINDS
        if self.consumes_model_call is not is_call:
            raise ValueError("v2 node call flag differs from its kind")
        call_shape = (
            self.campaign_call_slot is not None,
            self.replicate_call_slot is not None,
            self.arm_slot is not None,
            self.provider_seed_u63 is not None,
            self.max_provider_attempts is not None,
            self.failure_consumes_frozen_slot,
        )
        if is_call and call_shape != (True, True, True, True, True, True):
            raise ValueError(
                "v2 model-call node lacks exact slot, seed, attempt, or burn semantics"
            )
        if not is_call and call_shape != (False, False, False, False, False, False):
            raise ValueError("v2 deterministic control node cannot consume a model-call slot")

        expected_feedback = BfclV4PublicDevelopmentV2FeedbackView.NONE
        if self.kind in {
            BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
            BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        }:
            expected_feedback = {
                BfclV4PublicDevelopmentV2Arm.SCORE: (
                    BfclV4PublicDevelopmentV2FeedbackView.SCORE_ONLY
                ),
                BfclV4PublicDevelopmentV2Arm.FULL: (
                    BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL
                ),
            }.get(self.arm, BfclV4PublicDevelopmentV2FeedbackView.NONE)
        if self.feedback_view is not expected_feedback:
            raise ValueError("v2 feedback projection differs from arm and controller kind")
        expected_grader_feedback = self.kind in {
            BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS,
            BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
        }
        if self.grader_feedback_available is not expected_grader_feedback:
            raise ValueError("v2 grader-feedback availability differs from controller boundary")
        expected_typed = self.kind in {
            BfclV4PublicDevelopmentV2NodeKind.PROPOSAL,
            BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        }
        if self.typed_atomic_edit_required is not expected_typed:
            raise ValueError("v2 typed-edit requirement differs from proposal/candidate slots")
        return self


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
