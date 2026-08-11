from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ComponentKind
from spiral_harness.evolution.models import (
    CandidateScreen,
    CandidateScreenFailure,
    CandidateScreenStatus,
    Nomination,
)
from spiral_harness.evolution.seeds import (
    derive_strategy_seed,
    uniform_without_replacement_indices,
)
from spiral_harness.evolution.strategies import (
    PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE,
    PROMPT_PROPOSAL_MEDIA_TYPE,
    PromptMutationCatalogue,
    PromptMutationEntry,
    StrategyPermissionError,
    make_search_policy,
    make_strategy_plugin_manifest,
    nominate_candidate,
    proposals_from_random_selection,
    sample_random_valid,
    validate_strategy_permissions,
)
from spiral_harness.experiments.baselines import BaselineKind, FrozenMutationPolicy


def ref(character: str, media_type: str = "application/json", *, size: int = 32) -> ArtifactRef:
    return ArtifactRef(sha256=character * 64, size=size, media_type=media_type)


def policy(kind: BaselineKind, *, proposals_per_round: int = 3):
    is_static = kind is BaselineKind.STATIC
    return make_search_policy(
        baseline_kind=kind,
        mutation_policy=FrozenMutationPolicy(
            grammar_version="atomic-prompt-replace-v1",
            allowed_component_kinds=(ComponentKind.PROMPT,),
            max_artifact_size_bytes=128,
        ),
        max_rounds=0 if is_static else 4,
        max_gate_queries=0 if is_static else 4,
        patience_rounds=0 if is_static else 2,
        max_consecutive_declines=0 if is_static else 2,
        max_diagnoses=12 if kind is BaselineKind.EVIDENCE_TARGETED else 0,
        max_proposals=0 if is_static else 12,
        max_screens=0 if is_static else 12,
        max_proposals_per_round=0 if is_static else proposals_per_round,
        max_candidates_screened_per_round=0 if is_static else proposals_per_round,
        family_alpha=0.05,
    )


def entry(
    entry_id: str,
    after_character: str,
    *,
    before_character: str = "1",
    target: str = "system",
    size: int = 32,
) -> PromptMutationEntry:
    return PromptMutationEntry(
        entry_id=entry_id,
        target_component_name=target,
        expected_before_prompt_ref=ref(before_character, "text/plain"),
        after_prompt_ref=ref(after_character, "text/plain", size=size),
        hypothesis_ref=ref("f"),
        mechanism_family=f"family-{entry_id}",
    )


def catalogue() -> PromptMutationCatalogue:
    return PromptMutationCatalogue(
        catalogue_id="finite-v1",
        grammar_version="atomic-prompt-replace-v1",
        parent_harness_ref=ref("e"),
        entries=(
            entry("valid-c", "4"),
            entry("wrong-parent-prompt", "5", before_character="2"),
            entry("valid-a", "2"),
            entry("noop", "1"),
            entry("wrong-target", "6", target="critic"),
            entry("oversize", "7", size=129),
            entry("valid-b", "3"),
        ),
    )


def sample(*, round_index: int = 0, strategy_seed: int = 123, count: int = 2):
    frozen = catalogue()
    return sample_random_valid(
        catalogue=frozen,
        policy=policy(BaselineKind.RANDOM_VALID),
        seed_harness_ref=frozen.parent_harness_ref,
        parent_harness_ref=frozen.parent_harness_ref,
        target_component_name="system",
        current_prompt_ref=ref("1", "text/plain"),
        strategy_seed=strategy_seed,
        round_index=round_index,
        requested_entry_count=count,
    )


def screen(
    character: str,
    *,
    confidence_lower: float,
    mean_delta: float,
    tokens_ratio: float,
    latency_ratio: float,
    primary_score: float = 0.8,
    regression_rate: float = 0.1,
    status: CandidateScreenStatus = CandidateScreenStatus.ELIGIBLE,
) -> CandidateScreen:
    numeric = int(character, 16)
    values: dict[str, object] = {
        "baseline_kind": BaselineKind.PROMPT_ONLY,
        "round_index": 0,
        "proposal_ref": ref(character, PROMPT_PROPOSAL_MEDIA_TYPE),
        "status": status,
    }
    if status is CandidateScreenStatus.ELIGIBLE:
        values.update(
            candidate_ref=ref(format(numeric + 1, "x")),
            candidate_harness_ref=ref(format(numeric + 2, "x")),
            evaluation_ref=ref(format(numeric + 3, "x")),
            primary_score=primary_score,
            mean_delta=mean_delta,
            confidence_lower=confidence_lower,
            regression_rate=regression_rate,
            tokens_ratio=tokens_ratio,
            latency_ratio=latency_ratio,
        )
    else:
        values["failure_codes"] = (CandidateScreenFailure.CONSTRAINT_FAILED,)
    return CandidateScreen(**values)


def test_catalogue_is_finite_canonical_and_rejects_duplicate_weighting() -> None:
    frozen = catalogue()
    reversed_catalogue = PromptMutationCatalogue(
        **{
            **frozen.model_dump(),
            "entries": tuple(reversed(frozen.entries)),
        }
    )

    assert tuple(item.entry_id for item in frozen.entries) == tuple(
        sorted(item.entry_id for item in frozen.entries)
    )
    assert frozen == reversed_catalogue
    assert frozen.fingerprint == canonical_sha256(frozen)
    assert frozen.artifact_ref.media_type == PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE

    duplicate = entry("second-id-same-mutation", "2")
    with pytest.raises(ValidationError, match="duplicates bias sampling"):
        PromptMutationCatalogue(
            **{
                **frozen.model_dump(),
                "entries": (*frozen.entries, duplicate),
            }
        )


def test_random_valid_is_reproducible_eligible_non_noop_uniform_without_replacement() -> None:
    first = sample()
    replay = sample()

    assert first == replay
    assert first.catalogue_fingerprint == catalogue().fingerprint
    assert first.catalogue_ref == catalogue().artifact_ref
    assert first.eligible_entry_count == 3
    assert len(first.selected_entries) == 2
    assert len(set(first.selected_entry_ids)) == 2
    assert set(first.selected_entry_ids).issubset({"valid-a", "valid-b", "valid-c"})
    assert all(not item.is_noop for item in first.selected_entries)
    assert first.sampling_claim == ("uniform-without-replacement-over-eligible-catalogue-entries")
    assert first.sampling_frame == "eligible-entries-in-frozen-prompt-catalogue"

    next_round = sample(round_index=1)
    another_run = sample(strategy_seed=456)
    assert next_round.sample_seed != first.sample_seed
    assert another_run.sample_seed != first.sample_seed
    assert next_round.eligible_fingerprint == first.eligible_fingerprint


def test_random_valid_records_exhaustion_without_claiming_an_unbounded_space() -> None:
    frozen = catalogue()
    selection = sample_random_valid(
        catalogue=frozen,
        policy=policy(BaselineKind.RANDOM_VALID),
        seed_harness_ref=frozen.parent_harness_ref,
        parent_harness_ref=frozen.parent_harness_ref,
        target_component_name="unknown-component",
        current_prompt_ref=ref("1", "text/plain"),
        strategy_seed=123,
        round_index=0,
        requested_entry_count=2,
    )

    assert selection.eligible_entry_count == 0
    assert selection.selected_entries == ()
    serialized = str(selection.model_dump(mode="json"))
    assert "free-text" not in serialized
    assert "free-form" not in serialized


def test_random_valid_rejects_wrong_arm_budget_grammar_parent_and_ref() -> None:
    frozen = catalogue()
    common = {
        "catalogue": frozen,
        "seed_harness_ref": frozen.parent_harness_ref,
        "parent_harness_ref": frozen.parent_harness_ref,
        "target_component_name": "system",
        "current_prompt_ref": ref("1", "text/plain"),
        "strategy_seed": 123,
        "round_index": 0,
        "requested_entry_count": 2,
    }
    with pytest.raises(StrategyPermissionError, match="only random-valid"):
        sample_random_valid(policy=policy(BaselineKind.PROMPT_ONLY), **common)
    with pytest.raises(StrategyPermissionError, match="max_proposals_per_round"):
        sample_random_valid(
            policy=policy(BaselineKind.RANDOM_VALID, proposals_per_round=1),
            **common,
        )
    with pytest.raises(StrategyPermissionError, match="different seed harness"):
        sample_random_valid(
            policy=policy(BaselineKind.RANDOM_VALID),
            **{**common, "seed_harness_ref": ref("d")},
        )
    with pytest.raises(StrategyPermissionError, match="canonical frozen catalogue"):
        sample_random_valid(
            policy=policy(BaselineKind.RANDOM_VALID),
            catalogue_ref=ref("a", PROMPT_MUTATION_CATALOGUE_MEDIA_TYPE),
            **common,
        )


def test_random_selection_materializes_shared_prompt_proposal_grammar() -> None:
    selection = sample(count=3)
    proposals = proposals_from_random_selection(selection)

    assert len(proposals) == 3
    assert {proposal.baseline_kind for proposal in proposals} == {BaselineKind.RANDOM_VALID}
    assert {proposal.target_component_kind for proposal in proposals} == {ComponentKind.PROMPT}
    assert all(proposal.catalogue_ref == selection.catalogue_ref for proposal in proposals)
    assert all(proposal.proposer_confidence is None for proposal in proposals)


def test_random_valid_excludes_entries_selected_in_prior_rounds() -> None:
    first = sample(count=2)
    second = sample_random_valid(
        catalogue=catalogue(),
        policy=policy(BaselineKind.RANDOM_VALID),
        seed_harness_ref=catalogue().parent_harness_ref,
        parent_harness_ref=catalogue().parent_harness_ref,
        target_component_name="system",
        current_prompt_ref=ref("1", "text/plain"),
        strategy_seed=123,
        round_index=1,
        requested_entry_count=2,
        excluded_entry_ids=first.selected_entry_ids,
    )

    assert second.excluded_entry_ids == tuple(sorted(first.selected_entry_ids))
    assert not set(second.selected_entry_ids).intersection(first.selected_entry_ids)
    assert second.eligible_entry_count == 1


def test_uniform_shuffle_is_a_deterministic_permutation_prefix() -> None:
    outputs = {
        uniform_without_replacement_indices(
            population_size=5,
            sample_size=5,
            sample_seed=seed,
        )
        for seed in range(20)
    }

    assert len(outputs) > 1
    assert all(tuple(sorted(output)) == tuple(range(5)) for output in outputs)
    assert uniform_without_replacement_indices(
        population_size=5,
        sample_size=3,
        sample_seed=7,
    ) == uniform_without_replacement_indices(
        population_size=5,
        sample_size=3,
        sample_seed=7,
    )
    with pytest.raises(ValueError, match="must not exceed"):
        uniform_without_replacement_indices(
            population_size=2,
            sample_size=3,
            sample_seed=7,
        )


def test_strategy_seed_domains_separate_baselines_and_search_runs() -> None:
    seeds = {
        derive_strategy_seed(
            proposal_master_seed=17,
            search_run_seed=run_seed,
            baseline_kind=kind,
        )
        for kind, run_seed in itertools.product(BaselineKind, (101, 103))
    }
    assert len(seeds) == len(BaselineKind) * 2


def test_strategy_permission_join_revalidates_unchecked_plugin_copies() -> None:
    frozen_policy = policy(BaselineKind.PROMPT_ONLY)
    plugin = make_strategy_plugin_manifest(
        plugin_id="prompt-optimizer",
        plugin_version="1",
        implementation_ref=ref("a", "application/python"),
        baseline_kind=BaselineKind.PROMPT_ONLY,
    )
    corrupted = plugin.model_copy(update={"baseline_kind": BaselineKind.STATIC})

    with pytest.raises(StrategyPermissionError, match="invalid strategy contract"):
        validate_strategy_permissions(policy=frozen_policy, plugin=corrupted)


def test_nomination_uses_lcb_delta_resources_then_stable_refs_not_primary_score() -> None:
    frozen_policy = policy(BaselineKind.PROMPT_ONLY)
    high_primary_low_lcb = screen(
        "1",
        primary_score=0.99,
        confidence_lower=0.01,
        mean_delta=0.30,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )
    lower_primary_high_lcb = screen(
        "5",
        primary_score=0.70,
        confidence_lower=0.02,
        mean_delta=0.10,
        tokens_ratio=1.5,
        latency_ratio=1.5,
    )

    nominated = nominate_candidate(
        policy=frozen_policy,
        screens=(high_primary_low_lcb, lower_primary_high_lcb),
    )
    assert nominated is not None
    assert nominated.candidate_ref == lower_primary_high_lcb.candidate_ref
    assert nominated.primary_score == 0.70
    assert nominated.confidence_lower == 0.02
    assert nominated.regression_rate == lower_primary_high_lcb.regression_rate

    # Once LCB ties, mean delta wins before either resource ratio.
    better_delta = screen(
        "8",
        confidence_lower=0.02,
        mean_delta=0.20,
        tokens_ratio=2.0,
        latency_ratio=2.0,
    )
    nominated = nominate_candidate(
        policy=frozen_policy,
        screens=(lower_primary_high_lcb, better_delta),
    )
    assert nominated is not None
    assert nominated.candidate_ref == better_delta.candidate_ref


def test_nomination_resource_and_ref_ties_are_order_independent() -> None:
    frozen_policy = policy(BaselineKind.PROMPT_ONLY)
    expensive = screen(
        "1",
        confidence_lower=0.03,
        mean_delta=0.10,
        tokens_ratio=1.2,
        latency_ratio=1.0,
    )
    efficient = screen(
        "5",
        confidence_lower=0.03,
        mean_delta=0.10,
        tokens_ratio=1.1,
        latency_ratio=1.5,
    )
    assert (
        nominate_candidate(
            policy=frozen_policy,
            screens=(expensive, efficient),
        ).candidate_ref
        == efficient.candidate_ref
    )  # type: ignore[union-attr]

    same_metrics_a = screen(
        "1",
        confidence_lower=0.03,
        mean_delta=0.10,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )
    same_metrics_b = screen(
        "5",
        confidence_lower=0.03,
        mean_delta=0.10,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )
    left = nominate_candidate(
        policy=frozen_policy,
        screens=(same_metrics_a, same_metrics_b),
    )
    right = nominate_candidate(
        policy=frozen_policy,
        screens=(same_metrics_b, same_metrics_a),
    )
    assert left == right
    assert left is not None
    assert left.candidate_ref.sha256 < same_metrics_b.candidate_ref.sha256


def test_nomination_excludes_rejected_and_has_no_confidence_input() -> None:
    frozen_policy = policy(BaselineKind.PROMPT_ONLY)
    eligible = screen(
        "1",
        confidence_lower=0.01,
        mean_delta=0.01,
        tokens_ratio=1.0,
        latency_ratio=1.0,
    )
    rejected = screen(
        "5",
        confidence_lower=0.99,
        mean_delta=0.99,
        tokens_ratio=0.1,
        latency_ratio=0.1,
        status=CandidateScreenStatus.REJECTED,
    )
    nominated = nominate_candidate(policy=frozen_policy, screens=(rejected, eligible))
    assert nominated is not None
    assert nominated.candidate_ref == eligible.candidate_ref
    assert "proposer_confidence" not in Nomination.model_fields
    with pytest.raises(ValidationError, match="Extra inputs"):
        Nomination(**nominated.model_dump(), proposer_confidence=1.0)


def test_static_never_nominates_and_screen_ceiling_is_enforced() -> None:
    assert nominate_candidate(policy=policy(BaselineKind.STATIC), screens=()) is None
    too_many = (
        screen(
            "1",
            confidence_lower=0.1,
            mean_delta=0.1,
            tokens_ratio=1.0,
            latency_ratio=1.0,
        ),
        screen(
            "5",
            confidence_lower=0.1,
            mean_delta=0.1,
            tokens_ratio=1.0,
            latency_ratio=1.0,
        ),
    )
    with pytest.raises(StrategyPermissionError, match="screen batch"):
        nominate_candidate(
            policy=policy(BaselineKind.PROMPT_ONLY, proposals_per_round=1),
            screens=too_many,
        )
