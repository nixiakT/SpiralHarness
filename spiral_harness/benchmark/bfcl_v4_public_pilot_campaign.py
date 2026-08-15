"""Prospective three-search-seed campaign for the public BFCL V4 pilot."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PILOT_OUTER_SEED_U64,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BFCL_V4_PILOT_OUTER_SEEDS_U64,
    BfclV4PilotOuterSeed,
    BfclV4PublicPilotCallPlan,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr

BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID = "bfcl-v4-public-pilot-three-search-seeds-v1"
BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT = (
    "596572961f2b4045f25fcd611f3c297ef0cf88c924c74b72f0aad8567300f63b"
)
BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS = (
    "search-seed-2026081501",
    "search-seed-2026081502",
    "search-seed-2026081503",
)


def _topology_fingerprint(plan: BfclV4PublicPilotCallPlan) -> str:
    """Project away only the prospectively varied outer/provider seeds."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-pilot-replicate-topology/v1",
            "manifest_fingerprint": plan.manifest_fingerprint,
            "calls": tuple(
                call.model_dump(mode="python", exclude={"seed_u63"}) for call in plan.calls
            ),
            "total_model_call_ceiling": plan.total_model_call_ceiling,
            "max_provider_attempts_per_call": plan.max_provider_attempts_per_call,
            "adaptive_stopping": plan.adaptive_stopping,
            "holdout_can_continue_search": plan.holdout_can_continue_search,
            "invalid_candidate_slot_policy": plan.invalid_candidate_slot_policy,
            "invalid_candidate_selection_policy": plan.invalid_candidate_selection_policy,
            "both_candidates_frozen_before_candidate_fit": (
                plan.both_candidates_frozen_before_candidate_fit
            ),
            "both_arms_complete_gate_before_selection": (
                plan.both_arms_complete_gate_before_selection
            ),
            "both_selections_frozen_before_holdout": plan.both_selections_frozen_before_holdout,
            "same_model_required": plan.same_model_required,
            "same_per_call_budget_required": plan.same_per_call_budget_required,
        }
    )


class BfclV4PublicPilotCampaignReplicate(ImmutableModel):
    """One named search replicate whose journal still consumes a call plan."""

    schema_version: Literal["1"] = "1"
    ordinal: Annotated[int, Field(ge=0, lt=3, strict=True)]
    replicate_id: NonEmptyStr
    outer_seed_u64: BfclV4PilotOuterSeed
    call_plan: BfclV4PublicPilotCallPlan

    @model_validator(mode="after")
    def _close_replicate(self) -> Self:
        expected = (
            self.ordinal,
            BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS[self.ordinal],
            BFCL_V4_PILOT_OUTER_SEEDS_U64[self.ordinal],
        )
        observed = (self.ordinal, self.replicate_id, self.outer_seed_u64)
        if observed != expected:
            raise ValueError("campaign replicate ID, seed, or ordinal differs from registration")
        if self.call_plan.outer_seed_u64 != self.outer_seed_u64:
            raise ValueError("campaign replicate seed differs from its exact call plan")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicPilotCampaign(ImmutableModel):
    """Exact prospective registration; it contains no model output or score."""

    schema_version: Literal["1"] = "1"
    campaign_id: Literal[BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID] = BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID
    replicates: Annotated[
        tuple[BfclV4PublicPilotCampaignReplicate, ...],
        Field(min_length=3, max_length=3),
    ]
    replicate_count: Literal[3] = 3
    model_calls_per_replicate: Literal[100] = 100
    total_model_call_ceiling: Literal[300] = 300
    seed_order_frozen: Literal[True] = True
    same_roster_and_call_topology_across_replicates: Literal[True] = True
    provider_seeds_independently_derived_by_outer_seed: Literal[True] = True
    score_full_paired_within_each_replicate: Literal[True] = True
    same_frozen_model_across_replicates_and_arms_required: Literal[True] = True
    same_inference_config_across_replicates_and_arms_required: Literal[True] = True
    same_per_call_budget_across_replicates_and_arms_required: Literal[True] = True
    all_replicates_registered_before_first_live_result_required: Literal[True] = True
    post_result_seed_addition_allowed: Literal[False] = False
    post_result_seed_removal_allowed: Literal[False] = False
    post_result_seed_reordering_allowed: Literal[False] = False
    replicate_level_adaptive_stopping_allowed: Literal[False] = False
    all_replicates_must_consume_or_fail_closed_all_frozen_slots: Literal[True] = True
    legacy_external_seed_commitment_applies_only_to_outer_seed_u64: Literal[
        BFCL_V4_PILOT_OUTER_SEED_U64
    ] = BFCL_V4_PILOT_OUTER_SEED_U64
    external_seed_derivation_attested: Literal[False] = False
    model_outputs_present: Literal[False] = False
    scores_present: Literal[False] = False
    runtime_execution_attested: Literal[False] = False
    provider_seed_honoring_attested: Literal[False] = False
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_campaign(self) -> Self:
        expected_coordinates = tuple(
            (ordinal, replicate_id, outer_seed)
            for ordinal, (replicate_id, outer_seed) in enumerate(
                zip(
                    BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
                    BFCL_V4_PILOT_OUTER_SEEDS_U64,
                    strict=True,
                )
            )
        )
        observed_coordinates = tuple(
            (item.ordinal, item.replicate_id, item.outer_seed_u64) for item in self.replicates
        )
        if observed_coordinates != expected_coordinates:
            raise ValueError("campaign replicate IDs, seeds, or order differ from registration")

        plans = tuple(item.call_plan for item in self.replicates)
        if len({plan.fingerprint for plan in plans}) != 3:
            raise ValueError("campaign replicate call plans must have distinct fingerprints")
        if len({_topology_fingerprint(plan) for plan in plans}) != 1:
            raise ValueError("campaign replicate roster, topology, or budget semantics changed")

        provider_seed_sets = tuple({call.seed_u63 for call in plan.calls} for plan in plans)
        if any(
            not provider_seed_sets[left].isdisjoint(provider_seed_sets[right])
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("provider seed values overlap across search replicates")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def build_bfcl_v4_public_pilot_campaign() -> BfclV4PublicPilotCampaign:
    """Build all three preregistered call plans without executing any call."""

    replicates = tuple(
        BfclV4PublicPilotCampaignReplicate(
            ordinal=ordinal,
            replicate_id=replicate_id,
            outer_seed_u64=outer_seed_u64,
            call_plan=build_bfcl_v4_public_pilot_call_plan(outer_seed_u64),
        )
        for ordinal, (replicate_id, outer_seed_u64) in enumerate(
            zip(
                BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
                BFCL_V4_PILOT_OUTER_SEEDS_U64,
                strict=True,
            )
        )
    )
    campaign = BfclV4PublicPilotCampaign(replicates=replicates)
    if campaign.fingerprint != BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT:
        raise RuntimeError("campaign content differs from the prospective registration")
    return campaign


__all__ = [
    "BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT",
    "BFCL_V4_PUBLIC_PILOT_CAMPAIGN_ID",
    "BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS",
    "BfclV4PublicPilotCampaign",
    "BfclV4PublicPilotCampaignReplicate",
    "build_bfcl_v4_public_pilot_campaign",
]
