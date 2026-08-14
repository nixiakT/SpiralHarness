from __future__ import annotations

import json

from spiral_harness.benchmark._harness_fault_cases import (
    DEFAULT_HARNESS_FAULT_SPLIT_CONFIG,
    FaultFamily,
    FaultSurface,
    RepairRuleId,
    ScenarioRole,
    anti_cheat_summary,
    candidate_rule_ids,
    verify_partition_opening,
)
from spiral_harness.benchmark.harness_fault_authority import HarnessFaultScenarioAuthority
from spiral_harness.benchmark.harness_fault_compiler import (
    HarnessRole,
    compile_fault_repair,
    parse_fault_repair_action,
    verify_fault_compilation,
)
from spiral_harness.benchmark.harness_fault_runtime import TrustedRuntimeEventService
from spiral_harness.core.models import ComponentKind, HarnessManifest
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.storage.artifact_store import ArtifactStore

_EXPLORATION_SALT = b"hfb4-exploration-authority-salt-0001"
_GATE_SALT = b"hfb4-gate-authority-salt-000000000001"
_SEALED_SALT = b"hfb4-sealed-authority-salt-00000001"


def _authority(store: ArtifactStore) -> HarnessFaultScenarioAuthority:
    return HarnessFaultScenarioAuthority(
        store,
        exploration_salt=_EXPLORATION_SALT,
        gate_salt=_GATE_SALT,
        sealed_salt=_SEALED_SALT,
    )


def _spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="deterministic-replay-with-trusted-middleware",
        backend_fingerprint="harness-fault-v4-design-fixture",
        model="fixture/harness-fault-v4",
        revision="snapshot-2026-08-14",
        tokenizer="fixture/frozen-tokenizer",
        tokenizer_revision="snapshot-2026-08-14",
        runtime="python-3.12/harness-fault-v4",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
            timeout_seconds=5.0,
        ),
    )


def test_v4_opening_covers_independent_surface_families_and_control_roles(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    verified = verify_partition_opening(store, _authority(store).issue_exploration_grant())
    scenarios = verified.scenarios

    assert {item.family for item in scenarios} == set(FaultFamily)
    assert {item.surface for item in scenarios} == set(FaultSurface)
    assert len(FaultFamily) == 7
    assert len(FaultSurface) == 6
    assert len(scenarios) == (
        len(FaultFamily) * DEFAULT_HARNESS_FAULT_SPLIT_CONFIG.groups_per_family * len(ScenarioRole)
    )
    for family in FaultFamily:
        family_cases = tuple(item for item in scenarios if item.family is family)
        assert {item.role for item in family_cases} == set(ScenarioRole)
        assert len({item.expected_observable for item in family_cases}) == len(family_cases)


def test_v4_anti_cheat_envelope_keeps_constants_and_single_family_patches_below_oracle(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "cas")
    verified = verify_partition_opening(store, _authority(store).issue_gate_grant())
    summary = anti_cheat_summary(verified.scenarios)

    assert summary.scenario_count == 28
    assert summary.oracle_behavior_rate == 0.75
    assert summary.best_constant_branch_selector_rate <= 11 / 28
    assert summary.best_constant_output_rate <= 1 / 28
    assert summary.best_single_family_patch_rate <= 12 / 28
    assert summary.oracle_minus_best_constant_branch >= 10 / 28
    assert summary.oracle_minus_best_single_family_patch >= 9 / 28


def test_v4_compiler_materializes_and_matches_every_mutable_surface(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    service = TrustedRuntimeEventService()
    spec = _spec()
    routed_rule = RepairRuleId.ROUTED_POLICY
    action = parse_fault_repair_action(json.dumps({"rule_id": routed_rule.value}))
    compiled = compile_fault_repair(
        store,
        spec,
        action,
        runtime_producer_id=service.producer_id,
    )
    verified = verify_fault_compilation(
        store,
        spec,
        compiled.manifest_ref,
        expected_runtime_producer_id=service.producer_id,
    )
    candidate = verified.entry(HarnessRole.CANDIDATE)
    placebo = verified.entry(HarnessRole.PLACEBO)
    parent = verified.entry(HarnessRole.FAULTY_PARENT)
    revert = verified.entry(HarnessRole.REVERT)
    oracle = verified.entry(HarnessRole.ORACLE)
    candidate_manifest = store.get_json(candidate.harness_ref, HarnessManifest)
    revert_manifest = store.get_json(revert.harness_ref, HarnessManifest)
    oracle_manifest = store.get_json(oracle.harness_ref, HarnessManifest)

    expected_kinds = {
        ComponentKind.PROMPT,
        ComponentKind.MEMORY,
        ComponentKind.TOOL,
        ComponentKind.MIDDLEWARE,
        ComponentKind.CONTROL_FLOW,
        ComponentKind.SKILL,
    }
    assert {item.kind for item in candidate.surface_components} == expected_kinds
    assert len(candidate.surface_components) == 7  # schema and routing are separate tool parts
    assert tuple(item.artifact.size for item in candidate.surface_components) == tuple(
        item.artifact.size for item in placebo.surface_components
    )
    assert tuple(item.artifact for item in revert.surface_components) == tuple(
        item.artifact for item in parent.surface_components
    )
    # The generic runner injects the prompt projection only. The trusted v4
    # runtime consumes the exact seven-component graph above.
    assert {item.kind for item in candidate_manifest.components} == {ComponentKind.PROMPT}
    assert candidate_manifest.parent == parent.harness_ref
    assert revert_manifest.parent == candidate.harness_ref
    assert oracle_manifest.parent is None
    assert len(verified.matched_component_bytes) == 7
    assert set(candidate_rule_ids()) >= {
        RepairRuleId.CONSTANT_SAFE,
        RepairRuleId.CONSTANT_LEGACY,
        RepairRuleId.ROUTED_POLICY,
    }
