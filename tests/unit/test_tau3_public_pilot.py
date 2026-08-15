from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.tau3_public_pilot as subject
from spiral_harness.benchmark.tau3_public_pilot import (
    Tau3PublicPilotError,
    load_tau3_blind_grouping_audit_bundle,
    load_tau3_rejected_public_pilot,
    release_tau3_candidate_search_contracts,
)
from spiral_harness.benchmark.tau3_public_pilot_contracts import (
    TAU3_BASELINE_BOUNDARY,
    TAU3_GROUPING_AUTHORITY,
    TAU3_REJECTED_ROSTER_RECEIPT,
    TAU3_SELECTION_SALT,
    TAU3_SEMANTIC_CONFLICTS,
    Tau3BenchmarkDomain,
    Tau3CandidateTaskContract,
    Tau3PilotPartition,
    Tau3RejectedRosterReceipt,
    Tau3SemanticConflictCode,
    tau3_critical_file_bundle_sha256,
    tau3_selection_sha256,
    tau3_source_subset_bundle_sha256,
)
from spiral_harness.benchmark.tau3_public_pilot_identities import (
    TAU3_CRITICAL_FILE_BUNDLE_SHA256,
    TAU3_CRITICAL_FILE_IDENTITIES,
    TAU3_DOMAIN_POLICY_BUNDLE_SHA256,
    TAU3_DOMAIN_POLICY_PATHS,
    TAU3_DOMAIN_TOOL_BUNDLE_SHA256,
    TAU3_DOMAIN_TOOL_PATHS,
)
from spiral_harness.core.canonical import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKOUT = PROJECT_ROOT / "data/benchmarks/tau3-current-v1.0.1"


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("TAU3_V101_CHECKOUT")
    checkout = Path(configured) if configured else DEFAULT_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned tau3 v1.0.1 checkout unavailable")
    return checkout


@pytest.fixture(scope="module")
def loaded(pinned_checkout: Path) -> subject.Tau3LoadedRejectedPilot:
    return load_tau3_rejected_public_pilot(pinned_checkout)


@pytest.fixture(scope="module")
def grouping_bundle(pinned_checkout: Path) -> subject.Tau3BlindGroupingAuditBundle:
    return load_tau3_blind_grouping_audit_bundle(pinned_checkout)


def test_selection_formula_and_complete_salt_are_frozen_byte_for_byte() -> None:
    assert TAU3_SELECTION_SALT == ("tau3:v1.0.1:fc0055dc4e0a316c3f83133267fbd6faaa770992")
    assert tau3_selection_sha256("airline", "5") == (
        "038bdad770146dd40998398e43cec8eaca98ff300ddc2abd114c8c1c8af7fa93"
    )
    assert tau3_selection_sha256("retail", "32") == (
        "07b97be3bfd2d2a1f07813baa7c345f6b3e116c44b30069daf8e0542576d1e46"
    )
    with pytest.raises(ValueError, match="canonical"):
        tau3_selection_sha256("Airline", "5")
    with pytest.raises(ValueError, match="canonical"):
        tau3_selection_sha256("airline", " 5")


def test_rejected_roster_is_the_recomputed_global_draw_not_a_semantic_pick() -> None:
    roster = TAU3_REJECTED_ROSTER_RECEIPT.roster

    assert [(item.benchmark_domain.value, item.upstream_task_id) for item in roster] == [
        ("airline", "5"),
        ("airline", "7"),
        ("retail", "32"),
        ("retail", "41"),
        ("airline", "38"),
        ("airline", "29"),
        ("retail", "86"),
        ("retail", "77"),
        ("airline", "23"),
        ("airline", "35"),
        ("retail", "95"),
        ("retail", "22"),
        (
            "telecom",
            "[mobile_data_issue]bad_vpn|data_saver_mode_on|"
            "user_abroad_roaming_disabled_on[PERSONA:None]",
        ),
        (
            "telecom",
            "[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|"
            "lock_sim_card_pin|unseat_sim_card[PERSONA:Easy]",
        ),
        (
            "telecom",
            "[mms_issue]bad_network_preference|bad_wifi_calling|break_apn_mms_setting|"
            "break_app_storage_permission|data_mode_off|data_usage_exceeded|"
            "unseat_sim_card[PERSONA:Easy]",
        ),
    ]
    assert Counter(item.partition for item in roster) == {
        Tau3PilotPartition.FIT: 4,
        Tau3PilotPartition.GATE: 4,
        Tau3PilotPartition.HOLDOUT: 7,
    }
    assert len({item.opaque_task_id for item in roster}) == 15
    assert len({item.selection_sha256 for item in roster}) == 15
    assert len({item.source_cluster_id for item in roster}) == 3
    telecom = [item for item in roster if item.benchmark_domain == Tau3BenchmarkDomain.TELECOM]
    assert len({item.source_cluster_id for item in telecom}) == 1


def test_rejected_receipt_has_no_seed_or_score_bearing_escape_hatch() -> None:
    receipt = TAU3_REJECTED_ROSTER_RECEIPT

    assert receipt.roster_admissible is False
    assert receipt.confirmatory_eligible is False
    assert receipt.score_bearing_execution_allowed is False
    assert receipt.candidate_contract_release_allowed is False
    assert receipt.self_evolution_evidence is False
    assert receipt.search_seeds_u64 == ()
    assert receipt.trajectories_frozen is False
    assert receipt.budget_topology_frozen is False
    assert "trajectory" not in receipt.model_dump(mode="json")

    payload = receipt.model_dump(mode="python")
    payload["roster_admissible"] = True
    with pytest.raises(ValidationError):
        Tau3RejectedRosterReceipt.model_validate(payload, strict=True)
    payload = receipt.model_dump(mode="python")
    payload["search_seeds_u64"] = (2_026_081_501,)
    with pytest.raises(ValidationError):
        Tau3RejectedRosterReceipt.model_validate(payload, strict=True)


def test_semantic_conflicts_are_cross_partition_and_pre_result() -> None:
    by_opaque = {
        item.opaque_task_id: (item.benchmark_domain.value, item.upstream_task_id)
        for item in TAU3_REJECTED_ROSTER_RECEIPT.roster
    }

    assert tuple(item.code for item in TAU3_SEMANTIC_CONFLICTS) == (
        Tau3SemanticConflictCode.DELAY_COMPENSATION,
        Tau3SemanticConflictCode.ADDRESS_CHANGE,
    )
    assert tuple(by_opaque[item] for item in TAU3_SEMANTIC_CONFLICTS[0].opaque_task_ids) == (
        ("airline", "5"),
        ("airline", "38"),
    )
    assert tuple(by_opaque[item] for item in TAU3_SEMANTIC_CONFLICTS[1].opaque_task_ids) == (
        ("retail", "41"),
        ("retail", "86"),
        ("retail", "22"),
    )
    assert all(item.model_results_used is False for item in TAU3_SEMANTIC_CONFLICTS)


def test_critical_source_bundle_is_the_ordered_75_file_commitment() -> None:
    assert len(TAU3_CRITICAL_FILE_IDENTITIES) == 75
    assert tau3_critical_file_bundle_sha256() == TAU3_CRITICAL_FILE_BUNDLE_SHA256
    assert TAU3_CRITICAL_FILE_BUNDLE_SHA256 == (
        "25097699bd2c87a914738dd4f4e350973189cb702b16d44ab874ffe609185cd5"
    )
    assert len({item.git_path for item in TAU3_CRITICAL_FILE_IDENTITIES}) == 75
    assert any(
        "evaluator_nl_assertions.py" in item.git_path for item in TAU3_CRITICAL_FILE_IDENTITIES
    )
    assert any(
        "telecom/tasks/mms_issues.py" in item.git_path for item in TAU3_CRITICAL_FILE_IDENTITIES
    )
    for domain in ("airline", "retail", "telecom"):
        assert (
            tau3_source_subset_bundle_sha256(TAU3_DOMAIN_POLICY_PATHS[domain])
            == (TAU3_DOMAIN_POLICY_BUNDLE_SHA256[domain])
        )
        assert (
            tau3_source_subset_bundle_sha256(TAU3_DOMAIN_TOOL_PATHS[domain])
            == (TAU3_DOMAIN_TOOL_BUNDLE_SHA256[domain])
        )


def test_candidate_contract_has_an_exact_safe_projection_and_rejects_gold() -> None:
    contract = Tau3CandidateTaskContract(
        opaque_task_id="future-opaque-task",
        benchmark_domain=Tau3BenchmarkDomain.AIRLINE,
        policy_surface_sha256=TAU3_DOMAIN_POLICY_BUNDLE_SHA256["airline"],
        tool_surface_sha256=TAU3_DOMAIN_TOOL_BUNDLE_SHA256["airline"],
    )

    assert set(contract.model_dump(mode="json")) == {
        "schema_version",
        "opaque_task_id",
        "benchmark_domain",
        "policy_surface_sha256",
        "tool_surface_sha256",
        "user_messages_supplied_dynamically",
    }
    serialized = canonical_json(contract)
    for forbidden in (
        "upstream_task_id",
        "partition",
        "source_cluster",
        "user_scenario",
        "initial_state",
        "evaluation_criteria",
        "gold_action",
        "reward_basis",
        "grader",
        "SENTINEL_GOLD",
    ):
        assert forbidden not in serialized

    payload = contract.model_dump(mode="python")
    payload["evaluation_criteria"] = "SENTINEL_GOLD"
    with pytest.raises(ValidationError, match="Extra inputs"):
        Tau3CandidateTaskContract.model_validate(payload, strict=True)


def test_grouping_authority_is_full_roster_blind_and_still_blocked() -> None:
    authority = TAU3_GROUPING_AUTHORITY

    assert authority.full_base_roster_size == 278
    assert authority.independent_annotators_required == 2
    assert authority.independent_adjudicators_required == 1
    assert authority.source_family_before_partition is True
    assert authority.every_source_family_confined_to_one_partition is True
    assert authority.within_group_selection_after_partition is True
    assert authority.annotation_blind_to_model_outputs is True
    assert authority.annotation_blind_to_harness_outputs is True
    assert authority.post_partition_task_reassignment_allowed is False
    assert authority.independent_double_annotation_complete is False
    assert authority.adjudication_complete is False
    assert authority.execution_release_allowed is False


def test_native_min_is_primary_and_pure_at_b_is_explicitly_undefined() -> None:
    boundary = TAU3_BASELINE_BOUNDARY

    assert boundary.primary_bare_agent_baseline == "NATIVE-MIN"
    assert boundary.native_min_definition == (
        "fixed-upstream-llm-agent-policy-orchestrator-and-domain-tools"
    )
    assert boundary.pure_role == "secondary-structurally-weak-provider-minimal-reference"
    assert boundary.pure_has_domain_tools is False
    assert boundary.pure_at_b_status == "blocked-undefined"
    assert boundary.pure_at_b_blocker == "no-target-free-trajectory-to-policy-aggregator"
    assert boundary.native_min_may_be_renamed_pure is False
    assert boundary.current_score_bearing_arms_executable is False


def test_verified_checkout_recomputes_rows_draw_and_rejected_audit(
    loaded: subject.Tau3LoadedRejectedPilot,
) -> None:
    audit = loaded.near_duplicate_audit

    assert loaded.receipt == TAU3_REJECTED_ROSTER_RECEIPT
    assert loaded.model_invoked is False
    assert loaded.candidate_contracts_released is False
    assert loaded.score_bearing_execution_allowed is False
    assert len(audit.pairs) == 105
    assert audit.automatic_cross_partition_violations == 0
    assert audit.automatic_screen_passed is True
    assert audit.independent_semantic_audit_passed is False
    assert audit.overall_passed is False
    cross_partition = [item for item in audit.pairs if item.cross_partition]
    assert max(item.template_jaccard for item in cross_partition) == pytest.approx(0.15)
    assert max(item.source_jaccard for item in cross_partition) == pytest.approx(1 / 3)


def test_rejected_load_cannot_release_search_or_score_bearing_execution(
    loaded: subject.Tau3LoadedRejectedPilot,
) -> None:
    with pytest.raises(Tau3PublicPilotError, match="source-family grouping is incomplete"):
        release_tau3_candidate_search_contracts(loaded)
    with pytest.raises(Tau3PublicPilotError, match="source-family grouping is incomplete"):
        subject.assert_tau3_score_bearing_execution_allowed(loaded)


def test_grouping_audit_bundle_covers_all_278_base_tasks_once_without_partitions(
    grouping_bundle: subject.Tau3BlindGroupingAuditBundle,
) -> None:
    records = grouping_bundle.records

    assert len(records) == 278
    assert Counter(item.benchmark_domain for item in records) == {
        Tau3BenchmarkDomain.AIRLINE: 50,
        Tau3BenchmarkDomain.RETAIL: 114,
        Tau3BenchmarkDomain.TELECOM: 114,
    }
    assert len({(item.benchmark_domain, item.upstream_task_id) for item in records}) == 278
    assert len({item.audit_record_id for item in records}) == 278
    assert grouping_bundle.partition_labels_present is False
    assert grouping_bundle.selection_ranks_present is False
    assert grouping_bundle.candidate_visible is False
    assert grouping_bundle.model_or_harness_results_present is False
    assert all("partition" not in item.model_dump(mode="json") for item in records)
    assert all("selection_rank" not in item.model_dump(mode="json") for item in records)
    assert {
        item.structured_id_prefix for item in records if item.benchmark_domain == "telecom"
    } == {
        "[mobile_data_issue]",
        "[service_issue]",
        "[mms_issue]",
    }


def test_blob_identity_and_strict_json_helpers_fail_closed_on_tampering() -> None:
    identity = TAU3_CRITICAL_FILE_IDENTITIES[0]
    with pytest.raises(Tau3PublicPilotError, match="source differs"):
        subject._verify_blob(b"x" * identity.size, identity)
    with pytest.raises(Tau3PublicPilotError, match="strict UTF-8 JSON"):
        subject._strict_json(b'{"x":1,"x":2}', "duplicate probe")
    with pytest.raises(Tau3PublicPilotError, match="strict UTF-8 JSON"):
        subject._strict_json(b'{"x":NaN}', "non-finite probe")


def test_automatic_similarity_screen_is_not_allowed_to_override_semantic_rejection() -> None:
    token_sets: dict[str, tuple[set[str], set[str]]] = {}
    for index, entry in enumerate(TAU3_REJECTED_ROSTER_RECEIPT.roster):
        token_sets[entry.opaque_task_id] = ({f"template-{index}"}, {f"source-{index}"})
    audit = subject._near_duplicate_audit(token_sets)

    assert audit.automatic_screen_passed is True
    assert audit.independent_semantic_audit_passed is False
    assert audit.overall_passed is False

    first, gate = (
        TAU3_REJECTED_ROSTER_RECEIPT.roster[0],
        TAU3_REJECTED_ROSTER_RECEIPT.roster[4],
    )
    token_sets[gate.opaque_task_id] = token_sets[first.opaque_task_id]
    audit = subject._near_duplicate_audit(token_sets)
    assert audit.automatic_cross_partition_violations >= 1
    assert audit.automatic_screen_passed is False
    assert audit.overall_passed is False


def test_selection_helper_is_deterministic_without_a_checkout() -> None:
    task_ids = tuple(str(index) for index in range(20))
    selected = subject._selected_ids_for_domain(Tau3BenchmarkDomain.AIRLINE, task_ids)

    assert selected == tuple(
        sorted(
            task_ids,
            key=lambda item: (tau3_selection_sha256("airline", item), item),
        )[:6]
    )
    assert subject._selected_ids_for_domain(Tau3BenchmarkDomain.AIRLINE, task_ids) == selected
