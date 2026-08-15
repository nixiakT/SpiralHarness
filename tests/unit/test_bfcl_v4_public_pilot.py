from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_pilot as subject
from spiral_harness.benchmark.bfcl_v4 import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.benchmark.bfcl_v4_fixture_contracts import BfclV4SourceFileBinding
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    BfclV4PublicPilotError,
    adapt_bfcl_v4_openai_completions_tools,
    load_bfcl_v4_public_pilot,
    normalize_bfcl_v4_tool_calls,
    select_bfcl_v4_pure_at_b_plurality,
    verify_bfcl_v4_adapter_sources,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS,
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT,
    BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT,
    BFCL_V4_PILOT_OUTER_SEED_U64,
    BFCL_V4_PILOT_ROSTER_CONTENT_SHA256,
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4AdapterSourceVerification,
    BfclV4PilotSplit,
    BfclV4PureAtBSample,
    BfclV4WireToolCall,
    bfcl_v4_pilot_roster_content_sha256,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan import (
    build_bfcl_v4_public_pilot_call_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BFCL_V4_PILOT_OUTER_SEEDS_U64,
    BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256,
    BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256_BY_OUTER_SEED,
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotFeedbackView,
    bfcl_v4_pilot_schedule_content_sha256,
)

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for exact-source integration")
    return checkout


def test_public_pilot_roster_is_the_frozen_5_2_8_development_split() -> None:
    manifest = BFCL_V4_PUBLIC_PILOT_MANIFEST

    assert [item.task_id for item in manifest.roster] == [
        "simple_python_0",
        "simple_python_211",
        "multiple_5",
        "parallel_0",
        "parallel_multiple_9",
        "multiple_10",
        "parallel_multiple_11",
        "simple_python_87",
        "simple_python_128",
        "multiple_7",
        "multiple_8",
        "parallel_3",
        "parallel_4",
        "parallel_multiple_5",
        "parallel_multiple_55",
    ]
    assert Counter(item.split for item in manifest.roster) == {
        BfclV4PilotSplit.FIT: 5,
        BfclV4PilotSplit.GATE: 2,
        BfclV4PilotSplit.HOLDOUT: 8,
    }
    assert len({item.semantic_family for item in manifest.roster}) == 15
    assert manifest.questions_public is True
    assert manifest.possible_answers_public is True
    assert manifest.hidden_test_evidence is False
    assert manifest.reportable_result is False


def test_external_commitments_are_preserved_but_not_mislabelled_as_content_hashes() -> None:
    manifest = BFCL_V4_PUBLIC_PILOT_MANIFEST

    assert BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT == (
        "3556bca5d47d625fa3ba1cc086d6ac5c71b11a5327b16abb1d142c8e81ea1fee"
    )
    assert BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT == (
        "4d210dfc8bc99e795cd8e58ed913a876180c4cf28f4310b2af68650e4e924042"
    )
    assert BFCL_V4_PILOT_OUTER_SEED_U64 == 2_026_081_501
    assert manifest.external_roster_commitment_sha256 == BFCL_V4_PILOT_EXTERNAL_ROSTER_COMMITMENT
    assert manifest.roster_content_sha256 == bfcl_v4_pilot_roster_content_sha256(manifest.roster)
    assert manifest.roster_content_sha256 == BFCL_V4_PILOT_ROSTER_CONTENT_SHA256
    assert manifest.external_commitment_derivation_attested is False
    assert manifest.external_commitment_equals_content_fingerprint_attested is False


def test_call_plan_closes_the_frozen_100_call_five_arm_budget() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()

    assert len(plan.calls) == 100
    assert Counter(call.arm for call in plan.calls) == {
        BfclV4PilotArm.PURE: 8,
        BfclV4PilotArm.STATIC: 8,
        BfclV4PilotArm.SCORE: 28,
        BfclV4PilotArm.FULL: 28,
        BfclV4PilotArm.PURE_AT_B: 28,
    }
    assert plan.total_model_call_ceiling == 100
    assert plan.max_provider_attempts_per_call == 1
    assert plan.adaptive_stopping is False
    assert plan.holdout_can_continue_search is False
    assert plan.invalid_candidate_slot_policy == "parent-fallback-consumes-all-frozen-slots"
    assert plan.invalid_candidate_selection_policy == "forced-rollback"
    assert plan.both_candidates_frozen_before_candidate_fit is True
    assert plan.both_arms_complete_gate_before_selection is True
    assert plan.both_selections_frozen_before_holdout is True
    assert plan.external_seed_commitment_sha256 == BFCL_V4_PILOT_EXTERNAL_SEED_COMMITMENT
    assert plan.schedule_content_sha256 == bfcl_v4_pilot_schedule_content_sha256(plan.calls)
    assert plan.schedule_content_sha256 == BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256
    assert plan.fingerprint == "2ad745b6d6dfda2c2a91eed0e583ae4d00712bff63993634eb1d9b76809ced4b"
    assert plan.external_seed_derivation_attested is False
    assert all(0 <= call.seed_u63 <= (1 << 63) - 1 for call in plan.calls)
    holdout_ids = {
        item.task_id
        for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
        if item.split == BfclV4PilotSplit.HOLDOUT
    }
    assert all(call.task_id not in holdout_ids for call in plan.calls[:40])
    assert all(call.task_id in holdout_ids for call in plan.calls[40:])
    assert all(call.requires_both_selection_artifacts for call in plan.calls[40:])


def test_campaign_preregisters_three_complete_ordered_search_replicates() -> None:
    campaign = build_bfcl_v4_public_pilot_campaign()
    plans = tuple(item.call_plan for item in campaign.replicates)

    assert campaign.fingerprint == BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT
    assert tuple(item.replicate_id for item in campaign.replicates) == (
        BFCL_V4_PUBLIC_PILOT_REPLICATE_IDS
    )
    assert tuple(item.outer_seed_u64 for item in campaign.replicates) == (
        BFCL_V4_PILOT_OUTER_SEEDS_U64
    )
    assert tuple(plan.outer_seed_u64 for plan in plans) == BFCL_V4_PILOT_OUTER_SEEDS_U64
    assert tuple(plan.schedule_content_sha256 for plan in plans) == tuple(
        BFCL_V4_PILOT_SCHEDULE_CONTENT_SHA256_BY_OUTER_SEED[seed]
        for seed in BFCL_V4_PILOT_OUTER_SEEDS_U64
    )
    assert plans[0] == build_bfcl_v4_public_pilot_call_plan()
    assert sum(len(plan.calls) for plan in plans) == 300
    assert campaign.replicate_count == 3
    assert campaign.model_calls_per_replicate == 100
    assert campaign.total_model_call_ceiling == 300
    assert campaign.post_result_seed_addition_allowed is False
    assert campaign.post_result_seed_removal_allowed is False
    assert campaign.post_result_seed_reordering_allowed is False
    assert campaign.replicate_level_adaptive_stopping_allowed is False
    assert campaign.model_outputs_present is False
    assert campaign.scores_present is False
    assert campaign.runtime_execution_attested is False


def test_campaign_changes_only_independently_derived_provider_seeds_between_plans() -> None:
    plans = tuple(item.call_plan for item in build_bfcl_v4_public_pilot_campaign().replicates)

    for call_index in range(100):
        calls = tuple(plan.calls[call_index] for plan in plans)
        assert len({call.seed_u63 for call in calls}) == 3
        assert (
            len(
                {
                    tuple(
                        (key, value)
                        for key, value in call.model_dump(mode="python").items()
                        if key != "seed_u63"
                    )
                    for call in calls
                }
            )
            == 1
        )

    provider_seed_sets = tuple({call.seed_u63 for call in plan.calls} for plan in plans)
    assert all(
        provider_seed_sets[left].isdisjoint(provider_seed_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_campaign_rejects_seed_reordering_and_unregistered_outer_seeds() -> None:
    campaign = build_bfcl_v4_public_pilot_campaign()
    payload = campaign.model_dump(mode="python")
    payload["replicates"] = tuple(reversed(payload["replicates"]))

    with pytest.raises(ValidationError, match="IDs, seeds, or order"):
        type(campaign).model_validate(payload, strict=True)
    with pytest.raises(ValueError, match="absent from the frozen three-replicate campaign"):
        build_bfcl_v4_public_pilot_call_plan(2_026_081_504)
    with pytest.raises(ValueError, match="absent from the frozen three-replicate campaign"):
        build_bfcl_v4_public_pilot_call_plan(True)


def test_registered_replicate_rejects_a_self_consistently_rehashed_seed_mutation() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan(2_026_081_502)
    calls = list(plan.calls)
    used_seeds = {call.seed_u63 for call in calls}
    replacement_seed = next(seed for seed in range(2**63) if seed not in used_seeds)
    calls[-1] = type(calls[-1]).model_validate(
        {
            **calls[-1].model_dump(mode="python"),
            "seed_u63": replacement_seed,
        },
        strict=True,
    )
    payload = plan.model_dump(mode="python")
    payload["calls"] = tuple(calls)
    payload["schedule_content_sha256"] = bfcl_v4_pilot_schedule_content_sha256(
        tuple(calls),
        outer_seed_u64=plan.outer_seed_u64,
    )

    with pytest.raises(ValidationError, match="schedule content fingerprint"):
        type(plan).model_validate(payload, strict=True)


def test_score_and_full_have_identical_task_dag_and_paired_seeds() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    score = tuple(call for call in plan.calls if call.arm == BfclV4PilotArm.SCORE)
    full = tuple(call for call in plan.calls if call.arm == BfclV4PilotArm.FULL)

    def matched_signature(call: subject.BfclV4PilotCallSlot) -> tuple[object, ...]:
        return (
            call.arm_slot,
            call.kind,
            call.task_id,
            call.harness_variant,
            call.seed_u63,
            tuple(
                dependency.replace("score/", "arm/").replace("full/", "arm/")
                for dependency in call.depends_on
            ),
        )

    assert tuple(map(matched_signature, score)) == tuple(map(matched_signature, full))

    for arm in (BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL):
        calls = tuple(call for call in plan.calls if call.arm == arm)
        assert Counter(call.kind for call in calls) == {
            BfclV4PilotCallKind.PARENT_FIT: 5,
            BfclV4PilotCallKind.DIAGNOSIS: 1,
            BfclV4PilotCallKind.PROPOSAL: 1,
            BfclV4PilotCallKind.CANDIDATE_FIT: 5,
            BfclV4PilotCallKind.GATE: 8,
            BfclV4PilotCallKind.HOLDOUT: 8,
        }
        assert all(
            call.requires_both_selection_artifacts
            for call in calls
            if call.kind == BfclV4PilotCallKind.HOLDOUT
        )
        assert all(
            call.requires_both_candidate_artifacts
            for call in calls
            if call.kind == BfclV4PilotCallKind.CANDIDATE_FIT
        )


def test_pure_at_b_allocates_4_4_4_4_3_3_3_3_target_free_samples() -> None:
    calls = tuple(
        call
        for call in build_bfcl_v4_public_pilot_call_plan().calls
        if call.arm == BfclV4PilotArm.PURE_AT_B
    )

    assert Counter(call.task_id for call in calls) == {
        "simple_python_87": 4,
        "simple_python_128": 4,
        "multiple_7": 4,
        "multiple_8": 4,
        "parallel_3": 3,
        "parallel_4": 3,
        "parallel_multiple_5": 3,
        "parallel_multiple_55": 3,
    }
    assert len({call.seed_u63 for call in calls}) == 28
    assert all(call.grader_feedback_available is False for call in calls)


def test_openai_adapter_replicates_pinned_gorilla_type_mapping_and_name_recovery() -> None:
    functions = [
        {
            "name": "math.lookup",
            "description": "Nested mapping probe.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "mapping": {
                        "type": "dict",
                        "properties": {"value": {"type": "any"}},
                    },
                    "matrix": {
                        "type": "list",
                        "items": {"type": "list", "items": {"type": "float"}},
                    },
                    "ratio": {"type": "float", "description": "A ratio."},
                    "opaque": {"type": "not-in-upstream-map"},
                    "missing": {},
                },
                "required": ["mapping"],
            },
        }
    ]

    adapted = adapt_bfcl_v4_openai_completions_tools(functions)
    tool = adapted.tools[0]
    properties = tool["function"]["parameters"]["properties"]

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "math_lookup"
    assert tool["function"]["parameters"]["type"] == "object"
    assert properties["mapping"]["type"] == "object"
    assert properties["mapping"]["properties"]["value"]["type"] == "string"
    assert properties["matrix"]["type"] == "array"
    assert properties["matrix"]["items"]["type"] == "array"
    assert properties["matrix"]["items"]["items"]["type"] == "number"
    assert properties["ratio"] == {
        "description": "A ratio. This is a float type value.",
        "format": "float",
        "type": "number",
    }
    assert properties["opaque"]["type"] == "string"
    assert properties["missing"]["type"] == "string"
    assert adapted.official_to_wire("math.lookup") == "math_lookup"
    assert adapted.wire_to_official("math_lookup") == "math.lookup"
    assert json.loads(adapted.tools_json) == list(adapted.tools)
    assert functions[0]["name"] == "math.lookup"
    assert functions[0]["parameters"]["type"] == "dict"


def test_openai_adapter_contains_every_exact_upstream_gorilla_mapping() -> None:
    assert subject.BFCL_V4_GORILLA_TO_OPENAPI == {
        "integer": "integer",
        "number": "number",
        "float": "number",
        "string": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "dict": "object",
        "object": "object",
        "tuple": "array",
        "any": "string",
        "byte": "integer",
        "short": "integer",
        "long": "integer",
        "double": "number",
        "char": "string",
        "ArrayList": "array",
        "Array": "array",
        "HashMap": "object",
        "Hashtable": "object",
        "Queue": "array",
        "Stack": "array",
        "Any": "string",
        "String": "string",
        "Bigint": "integer",
    }


def test_openai_adapter_fails_closed_on_dotted_name_collision() -> None:
    functions = [
        {"name": name, "description": name, "parameters": {"type": "dict", "properties": {}}}
        for name in ("a.b", "a_b")
    ]

    with pytest.raises(BfclV4PublicPilotError, match="wire-name collision"):
        adapt_bfcl_v4_openai_completions_tools(functions)


def test_normalizer_recovers_official_names_sorts_calls_and_preserves_duplicates() -> None:
    adapted = adapt_bfcl_v4_openai_completions_tools(
        [
            {"name": name, "description": name, "parameters": {"type": "dict", "properties": {}}}
            for name in ("z.run", "a.run")
        ]
    )
    calls = (
        BfclV4WireToolCall(wire_name="z_run", arguments_json='{"b":2,"a":1}'),
        BfclV4WireToolCall(wire_name="a_run", arguments_json='{"x":3}'),
        BfclV4WireToolCall(wire_name="z_run", arguments_json='{"a":1,"b":2}'),
    )

    normalized = normalize_bfcl_v4_tool_calls(calls, adapted.name_bindings)

    assert json.loads(normalized) == [
        {"arguments": {"x": 3}, "name": "a.run"},
        {"arguments": {"a": 1, "b": 2}, "name": "z.run"},
        {"arguments": {"a": 1, "b": 2}, "name": "z.run"},
    ]


def test_pure_at_b_plurality_is_target_free_and_ties_use_frozen_order() -> None:
    adapted = adapt_bfcl_v4_openai_completions_tools(
        [{"name": "f.run", "description": "f", "parameters": {"type": "dict", "properties": {}}}]
    )
    a = (BfclV4WireToolCall(wire_name="f_run", arguments_json='{"x":1}'),)
    b = (BfclV4WireToolCall(wire_name="f_run", arguments_json='{"x":2}'),)
    samples = (
        BfclV4PureAtBSample(sample_id="s0", calls=a),
        BfclV4PureAtBSample(sample_id="s1", calls=b),
        BfclV4PureAtBSample(sample_id="s2", calls=b),
        BfclV4PureAtBSample(sample_id="s3", calls=a),
    )

    selected = select_bfcl_v4_pure_at_b_plurality(samples, adapted.name_bindings)

    assert selected.selected_sample_id == "s0"
    assert selected.selected_frozen_index == 0
    assert selected.plurality_count == 2
    assert selected.tie is True
    assert selected.all_abstained is False
    assert selected.grader_feedback_used is False
    assert selected.possible_answers_used is False


def test_pure_at_b_all_abstain_returns_first_frozen_sample() -> None:
    adapted = adapt_bfcl_v4_openai_completions_tools(
        [{"name": "f", "description": "f", "parameters": {"type": "dict", "properties": {}}}]
    )
    samples = (
        BfclV4PureAtBSample(
            sample_id="first",
            calls=(BfclV4WireToolCall(wire_name="unknown", arguments_json="{}"),),
        ),
        BfclV4PureAtBSample(sample_id="second", calls=None),
    )

    selected = select_bfcl_v4_pure_at_b_plurality(samples, adapted.name_bindings)

    assert selected.selected_sample_id == "first"
    assert selected.selected_frozen_index == 0
    assert selected.normalized_output_json is None
    assert selected.all_abstained is True


def test_loader_reads_only_pinned_question_objects_not_possible_answers(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_paths: list[str] = []
    original = subject._read_pinned_blob

    def recording_blob(git: Path, checkout: Path, git_path: str) -> bytes:
        observed_paths.append(git_path)
        return original(git, checkout, git_path)

    monkeypatch.setattr(subject, "_read_pinned_blob", recording_blob)
    loaded = load_bfcl_v4_public_pilot(pinned_checkout)

    assert tuple(task.task_id for task in loaded.tasks) == tuple(
        item.task_id for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    )
    assert set(observed_paths) == {
        item.question_git_path for item in BFCL_V4_PUBLIC_PILOT_MANIFEST.roster
    }
    assert all("possible_answer" not in path for path in observed_paths)
    assert len({name for task in loaded.tasks for name in task.official_function_names}) == 26
    serialized = loaded.model_dump_json()
    assert "ground_truth" not in serialized
    assert all(task.possible_answer_data_in_contract is False for task in loaded.tasks)
    assert all(task.runtime_git_object_read_attested is False for task in loaded.tasks)


def test_loader_adapts_each_pinned_public_task_with_reversible_names(
    pinned_checkout: Path,
) -> None:
    loaded = load_bfcl_v4_public_pilot(pinned_checkout)

    for task in loaded.tasks:
        adapted = subject.adapt_bfcl_v4_public_pilot_task(task)
        assert len(adapted.tools) == len(task.official_function_names)
        assert tuple(binding.official_name for binding in adapted.name_bindings) == (
            task.official_function_names
        )
        for binding in adapted.name_bindings:
            assert adapted.wire_to_official(binding.wire_name) == binding.official_name


def test_adapter_source_verification_binds_exact_pinned_files(pinned_checkout: Path) -> None:
    binding = verify_bfcl_v4_adapter_sources(pinned_checkout)

    assert binding.upstream_commit == BFCL_V4_UPSTREAM_COMMIT
    assert {item.git_path: (item.size, item.sha256) for item in binding.sources} == {
        "berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py": (
            1_813,
            "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
        ),
        "berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py": (
            33_694,
            "f78fd3edce603b333dc9a88ee2c041dc547d51f71aa449ffebc044c4b1e353f3",
        ),
    }
    assert binding.model_style == "openai-completions"
    assert binding.runtime_execution_attested is False
    assert binding.upstream_handler_equivalence_attested is False


def test_adapter_source_contract_rejects_self_consistent_but_unpinned_sources() -> None:
    sources = (
        BfclV4SourceFileBinding(git_path="arbitrary/a.py", size=1, sha256="a" * 64),
        BfclV4SourceFileBinding(git_path="arbitrary/b.py", size=2, sha256="b" * 64),
    )
    source_bundle_sha256 = subject.canonical_sha256(
        tuple(
            {"git_path": item.git_path, "sha256": item.sha256, "size": item.size}
            for item in sources
        )
    )

    with pytest.raises(ValidationError, match="exact pinned converter sources"):
        BfclV4AdapterSourceVerification(
            sources=sources,
            source_bundle_sha256=source_bundle_sha256,
        )


def test_task_contract_binds_candidate_payload_to_the_exact_pinned_row(
    pinned_checkout: Path,
) -> None:
    task = load_bfcl_v4_public_pilot(pinned_checkout).tasks[0]
    payload = task.model_dump(mode="python")
    payload["question_json"] = '[[{"content":"changed","role":"user"}]]'

    with pytest.raises(ValidationError, match="candidate payload fingerprint"):
        type(task).model_validate(payload, strict=True)


def test_call_plan_rejects_a_known_but_future_dependency() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    calls = list(plan.calls)
    first_payload = calls[0].model_dump(mode="python")
    first_payload["depends_on"] = (calls[1].call_id,)
    calls[0] = type(calls[0]).model_validate(first_payload, strict=True)
    plan_payload = plan.model_dump(mode="python")
    plan_payload["calls"] = tuple(calls)

    with pytest.raises(ValidationError, match="earlier global slot"):
        type(plan).model_validate(plan_payload, strict=True)


def test_call_plan_rejects_paired_but_wrong_adaptive_task_ids() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    calls = list(plan.calls)
    replacements: dict[str, str] = {}
    for index, call in enumerate(calls):
        if call.arm in {BfclV4PilotArm.SCORE, BfclV4PilotArm.FULL} and call.arm_slot == 0:
            payload = call.model_dump(mode="python")
            payload["task_id"] = "simple_python_211"
            payload["call_id"] = (
                f"{call.arm.value}/{call.arm_slot:02d}/{call.kind.value}/"
                f"simple_python_211/{call.harness_variant}"
            )
            changed = type(call).model_validate(payload, strict=True)
            replacements[call.call_id] = changed.call_id
            calls[index] = changed
    for index, call in enumerate(calls):
        dependencies = tuple(replacements.get(item, item) for item in call.depends_on)
        if dependencies != call.depends_on:
            calls[index] = type(call).model_validate(
                {**call.model_dump(mode="python"), "depends_on": dependencies},
                strict=True,
            )
    plan_payload = plan.model_dump(mode="python")
    plan_payload["calls"] = tuple(calls)

    with pytest.raises(ValidationError, match="task or variant sequence"):
        type(plan).model_validate(plan_payload, strict=True)


def test_call_plan_rejects_score_controllers_with_full_feedback_view() -> None:
    plan = build_bfcl_v4_public_pilot_call_plan()
    calls = list(plan.calls)
    for index, call in enumerate(calls):
        if call.arm == BfclV4PilotArm.SCORE and call.kind in {
            BfclV4PilotCallKind.DIAGNOSIS,
            BfclV4PilotCallKind.PROPOSAL,
        }:
            calls[index] = type(call).model_validate(
                {
                    **call.model_dump(mode="python"),
                    "feedback_view": BfclV4PilotFeedbackView.CANDIDATE_SAFE_FULL,
                },
                strict=True,
            )
    plan_payload = plan.model_dump(mode="python")
    plan_payload["calls"] = tuple(calls)

    with pytest.raises(ValidationError, match="feedback view"):
        type(plan).model_validate(plan_payload, strict=True)
