from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition, ProtocolSplit
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.evolution.feedback_media_types import (
    EXPLORATION_INPUTS_MEDIA_TYPE,
    SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
)
from spiral_harness.evolution.orchestrator import (
    DIAGNOSTIC_CLUSTER_MEDIA_TYPE,
    EXPLORATION_AGGREGATES_MEDIA_TYPE,
    EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE,
    EXPLORATION_TRAJECTORIES_MEDIA_TYPE,
    SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    SafeBenchmarkMetadata,
    SearchBenchmarkBinding,
)
from spiral_harness.evolution.seeds import (
    MANIFEST_BOUND_PAIRED_PROPOSER_SEED_DOMAIN,
    derive_manifest_bound_paired_proposer_seed,
)
from spiral_harness.experiments.baseline_profiles import make_matched_contrast_profile
from spiral_harness.experiments.baselines import BaselineKind, FrozenMutationPolicy
from spiral_harness.experiments.matched_v2 import (
    MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
    MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE,
    MatchedV2AdmissionError,
    MatchedV2ExecutionCeilings,
    MatchedV2GateQueryBlock,
    MatchedV2GateTask,
    MatchedV2PlannedTopology,
    MatchedV2PolicyBindings,
    MatchedV2RunManifest,
    MatchedV2SharedCoordinates,
    MatchedV2StudyManifest,
    admit_matched_v2_study,
    make_matched_v2_expectation,
    make_matched_v2_run_manifest,
    make_matched_v2_study_manifest,
)
from spiral_harness.storage.artifact_store import ArtifactStore


def _json(store: ArtifactStore, label: str, media_type: str = "application/json"):
    return store.put_json({"label": label}, media_type=media_type)


def _policy(*, max_size: int = 8_192) -> FrozenMutationPolicy:
    return FrozenMutationPolicy(
        grammar_version="matched-atomic-replace-v2",
        allowed_component_kinds=(ComponentKind.PROMPT, ComponentKind.SKILL),
        max_artifact_size_bytes=max_size,
    )


def _ceilings(**updates: object) -> MatchedV2ExecutionCeilings:
    values = {
        "max_rounds": 2,
        "max_proposals_per_round": 3,
        "max_total_proposals": 6,
        "max_total_nominations": 2,
        "max_optimizer_model_calls": 4,
        "max_solver_model_calls": 48,
        "max_gate_queries": 2,
        "max_evaluations": 48,
        "max_feedback_queries": 2,
        "max_attempts_per_evaluation": 2,
        "token_ceiling_per_attempt": 10,
        "max_tokens": 100_000,
        "max_wall_time_seconds": 600.0,
        "max_cost_usd": 10.0,
    }
    values.update(updates)
    return MatchedV2ExecutionCeilings(**values)


def _build_fixture(tmp_path) -> SimpleNamespace:
    store = ArtifactStore(tmp_path / "artifacts")
    exploration_ids = ("exploration-01", "exploration-02")
    safe_ref = store.put_json(
        SafeBenchmarkMetadata(
            benchmark_fingerprint="benchmark@fixed-v1",
            exploration_task_ids=exploration_ids,
        ),
        media_type=SAFE_BENCHMARK_METADATA_MEDIA_TYPE,
    )
    inputs_ref = store.put_json(
        {"partition": "exploration", "task_ids": list(exploration_ids)},
        media_type=EXPLORATION_INPUTS_MEDIA_TYPE,
    )
    aggregate_ref = _json(store, "aggregate", EXPLORATION_AGGREGATES_MEDIA_TYPE)
    item_ref = _json(store, "items", EXPLORATION_ITEM_FEEDBACK_MEDIA_TYPE)
    trajectories_ref = _json(store, "trajectories", EXPLORATION_TRAJECTORIES_MEDIA_TYPE)
    diagnostic_ref = _json(store, "diagnostic", DIAGNOSTIC_CLUSTER_MEDIA_TYPE)
    closure_refs = (
        diagnostic_ref,
        _json(store, "closure-1"),
        _json(store, "closure-2"),
        _json(store, "closure-3"),
    )
    splits = (
        ProtocolSplit(
            partition=ProtocolPartition.EXPLORATION,
            manifest_ref=_json(store, "exploration-split"),
        ),
        ProtocolSplit(
            partition=ProtocolPartition.GATE,
            manifest_ref=_json(store, "gate-split"),
        ),
    )
    benchmark = SearchBenchmarkBinding(
        benchmark_fingerprint="benchmark@fixed-v1",
        objective_aggregate_attestor_id="1" * 64,
        strategy_feedback_attestor_id="2" * 64,
        protocol_splits=splits,
        exploration_task_ids=exploration_ids,
        safe_benchmark_metadata_ref=safe_ref,
        exploration_inputs_ref=inputs_ref,
        exploration_aggregates_ref=aggregate_ref,
        exploration_item_feedback_ref=item_ref,
        exploration_trajectories_ref=trajectories_ref,
        diagnostic_evidence_ref=diagnostic_ref,
        diagnostic_closure_refs=closure_refs,
    )
    benchmark_ref = store.put_json(
        benchmark,
        media_type=SEARCH_BENCHMARK_BINDING_MEDIA_TYPE,
    )

    prompt_ref = store.put_bytes(b"Solve carefully.", media_type="text/plain")
    harness = HarnessManifest(
        model_fingerprint="model@fixed-v1",
        runtime_fingerprint="runtime@fixed-v1",
        trusted_plane_version="trusted-plane-v1",
        components=(
            HarnessComponentRef(
                name="system",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
    )
    harness_ref = store.put_json(harness, media_type=HARNESS_MANIFEST_MEDIA_TYPE)

    blocks = tuple(
        store.put_json(
            MatchedV2GateQueryBlock(
                query_index=index,
                nomination_index=index,
                tasks=(
                    MatchedV2GateTask(
                        task_id=f"gate-task-{index}",
                        source_id=f"source-{index}",
                        family_id=f"family-{index}",
                    ),
                ),
            ),
            media_type=MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
        )
        for index in range(2)
    )
    contrast = make_matched_contrast_profile(mutation_policy=_policy())
    shared = MatchedV2SharedCoordinates(
        contrast=contrast,
        contrast_fingerprint=contrast.fingerprint,
        study_id="score-v2-study",
        benchmark_binding_ref=benchmark_ref,
        model_fingerprint="model@fixed-v1",
        inference_fingerprint="inference@fixed-v1",
        runtime_fingerprint="runtime@fixed-v1",
        seed_harness_ref=harness_ref,
        proposal_master_seed=17,
        rollout_master_seed=991,
        search_run_seed=101,
        repeat_seeds=(11, 13),
        gate_query_block_refs=blocks,
        mutation_policy_fingerprint=canonical_sha256(contrast.score.mutation_policy),
        action_capability_fingerprint=canonical_sha256(contrast.score.action_capability),
        policies=MatchedV2PolicyBindings(
            proposer_policy_fingerprint="3" * 64,
            nomination_policy_fingerprint="4" * 64,
            optimizer_config_fingerprint="5" * 64,
            solver_config_fingerprint="6" * 64,
            grader_fingerprint="trusted-grader@fixed-v1",
            gate_policy_fingerprint="7" * 64,
            performance_policy_fingerprint="b" * 64,
            price_table_fingerprint="8" * 64,
        ),
        planned_topology=MatchedV2PlannedTopology(
            proposer_implementation_fingerprint="9" * 64,
            proposer_call_graph_fingerprint="a" * 64,
        ),
        ceilings=_ceilings(),
    )
    expectation = make_matched_v2_expectation(shared=shared)
    score = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    full = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
    )
    score_ref = store.put_json(score, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    full_ref = store.put_json(full, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    study = make_matched_v2_study_manifest(
        score_run_ref=score_ref,
        full_run_ref=full_ref,
        expectation=expectation,
    )
    study_ref = store.put_json(study, media_type=MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE)
    return SimpleNamespace(
        store=store,
        benchmark=benchmark,
        benchmark_ref=benchmark_ref,
        blocks=blocks,
        shared=shared,
        expectation=expectation,
        score=score,
        full=full,
        score_ref=score_ref,
        full_ref=full_ref,
        study=study,
        study_ref=study_ref,
    )


def _persist_pair(fixture: SimpleNamespace, shared: MatchedV2SharedCoordinates):
    score = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    full = make_matched_v2_run_manifest(
        shared=shared,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
    )
    score_ref = fixture.store.put_json(score, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    full_ref = fixture.store.put_json(full, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    study = MatchedV2StudyManifest(
        score_run_ref=score_ref,
        full_run_ref=full_ref,
        expectation_fingerprint=fixture.expectation.fingerprint,
        shared_coordinate_fingerprint=fixture.expectation.shared_coordinate_fingerprint,
        contrast_fingerprint=fixture.expectation.contrast_fingerprint,
    )
    return fixture.store.put_json(study, media_type=MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE)


def test_matched_v2_admission_binds_pair_but_not_runtime(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    report = admit_matched_v2_study(
        fixture.store,
        study_ref=fixture.study_ref,
        expectation=fixture.expectation,
    )
    assert fixture.score.paired_proposer_seed == fixture.full.paired_proposer_seed
    assert report.paired_proposer_seed == fixture.score.paired_proposer_seed
    assert report.manifest_pair_admitted is True
    assert report.paired_proposer_seed_manifest_bound is True
    assert report.paired_proposer_seed_runtime_attested is False
    assert report.execution_attested is False
    assert report.runtime_topology_matched is False
    assert report.fresh_gate_blocks_runtime_attested is False
    assert report.campaign_validity_attested is False
    assert fixture.score.model_dump(
        exclude={"baseline_kind", "available_feedback"}
    ) == fixture.full.model_dump(exclude={"baseline_kind", "available_feedback"})


def test_manifest_bound_seed_recomputes_unbound_inputs_and_shared_context(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    assert "manifest-bound" in MANIFEST_BOUND_PAIRED_PROPOSER_SEED_DOMAIN
    score_seed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=17,
        search_run_seed=101,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        shared_coordinate_fingerprint=fixture.shared.fingerprint,
    )
    full_seed = derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=17,
        search_run_seed=101,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
        shared_coordinate_fingerprint=fixture.shared.fingerprint,
    )
    assert score_seed == full_seed == fixture.score.paired_proposer_seed
    assert score_seed != derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=17,
        search_run_seed=103,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        shared_coordinate_fingerprint=fixture.shared.fingerprint,
    )
    assert score_seed != derive_manifest_bound_paired_proposer_seed(
        proposal_master_seed=17,
        search_run_seed=101,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
        shared_coordinate_fingerprint="f" * 64,
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        derive_manifest_bound_paired_proposer_seed(
            proposal_master_seed=17,
            search_run_seed=101,
            baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
            shared_coordinate_fingerprint="not-a-digest",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "model_fingerprint",
        "inference_fingerprint",
        "runtime_fingerprint",
        "search_run_seed",
        "policies",
        "planned_topology",
        "ceilings",
        "contrast",
    ),
)
def test_independent_expectation_rejects_joint_arm_drift(tmp_path, field_name: str) -> None:
    fixture = _build_fixture(tmp_path)
    values = fixture.shared.model_dump(mode="python", round_trip=True, warnings="none")
    if field_name in {"model_fingerprint", "inference_fingerprint", "runtime_fingerprint"}:
        values[field_name] = f"{field_name}@drifted"
    elif field_name == "search_run_seed":
        values[field_name] = 103
    elif field_name == "policies":
        values[field_name] = {
            **fixture.shared.policies.model_dump(mode="python"),
            "solver_config_fingerprint": "b" * 64,
        }
    elif field_name == "planned_topology":
        values[field_name] = {
            **fixture.shared.planned_topology.model_dump(mode="python"),
            "proposer_call_graph_fingerprint": "b" * 64,
        }
    elif field_name == "ceilings":
        values[field_name] = _ceilings(max_solver_model_calls=47).model_dump(mode="python")
    else:
        contrast = make_matched_contrast_profile(mutation_policy=_policy(max_size=4_096))
        values["contrast"] = contrast.model_dump(mode="python")
        values["contrast_fingerprint"] = contrast.fingerprint
        values["mutation_policy_fingerprint"] = canonical_sha256(contrast.score.mutation_policy)
        values["action_capability_fingerprint"] = canonical_sha256(contrast.score.action_capability)
    drifted = MatchedV2SharedCoordinates(**values)
    study_ref = _persist_pair(fixture, drifted)
    with pytest.raises(MatchedV2AdmissionError, match="shared coordinates"):
        admit_matched_v2_study(
            fixture.store,
            study_ref=study_ref,
            expectation=fixture.expectation,
        )


def test_run_rejects_paired_seed_drift_and_extra_call_ceiling(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    values = fixture.score.model_dump(mode="python", round_trip=True, warnings="none")
    with pytest.raises(ValidationError, match="manifest-bound derivation"):
        MatchedV2RunManifest(**{**values, "paired_proposer_seed": 1})

    ceilings = fixture.shared.ceilings.model_dump(mode="python")
    with pytest.raises(ValidationError, match="Extra inputs"):
        MatchedV2ExecutionCeilings(**ceilings, hidden_grader_calls=4)
    with pytest.raises(ValidationError, match="Extra inputs"):
        MatchedV2RunManifest(**values, hidden_optimizer_budget=4)


def test_gate_blocks_are_ordered_unique_and_source_family_disjoint(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    values = fixture.shared.model_dump(mode="python", round_trip=True, warnings="none")
    with pytest.raises(ValidationError, match="must not reuse"):
        MatchedV2SharedCoordinates(
            **{**values, "gate_query_block_refs": (fixture.blocks[0], fixture.blocks[0])}
        )
    with pytest.raises(ValidationError, match="count must equal"):
        MatchedV2SharedCoordinates(**{**values, "gate_query_block_refs": (fixture.blocks[0],)})

    reused = fixture.store.put_json(
        MatchedV2GateQueryBlock(
            query_index=1,
            nomination_index=1,
            tasks=(
                MatchedV2GateTask(
                    task_id="new-task",
                    source_id="source-0",
                    family_id="new-family",
                ),
            ),
        ),
        media_type=MATCHED_V2_GATE_QUERY_BLOCK_MEDIA_TYPE,
    )
    drifted = MatchedV2SharedCoordinates(
        **{**values, "gate_query_block_refs": (fixture.blocks[0], reused)}
    )
    drifted_expectation = make_matched_v2_expectation(shared=drifted)
    score = make_matched_v2_run_manifest(
        shared=drifted,
        baseline_kind=BaselineKind.SCORE_ONLY_MATCHED,
    )
    full = make_matched_v2_run_manifest(
        shared=drifted,
        baseline_kind=BaselineKind.EVIDENCE_TARGETED,
    )
    score_ref = fixture.store.put_json(score, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    full_ref = fixture.store.put_json(full, media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE)
    study = make_matched_v2_study_manifest(
        score_run_ref=score_ref,
        full_run_ref=full_ref,
        expectation=drifted_expectation,
    )
    study_ref = fixture.store.put_json(study, media_type=MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE)
    with pytest.raises(MatchedV2AdmissionError, match="reuse task/source/family"):
        admit_matched_v2_study(
            fixture.store,
            study_ref=study_ref,
            expectation=drifted_expectation,
        )


def test_admission_rejects_noncanonical_or_wrong_media_run(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    payload = fixture.score.model_dump_json(indent=2).encode()
    noncanonical_ref = fixture.store.put_bytes(
        payload,
        media_type=MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
    )
    study = MatchedV2StudyManifest(
        score_run_ref=noncanonical_ref,
        full_run_ref=fixture.full_ref,
        expectation_fingerprint=fixture.expectation.fingerprint,
        shared_coordinate_fingerprint=fixture.expectation.shared_coordinate_fingerprint,
        contrast_fingerprint=fixture.expectation.contrast_fingerprint,
    )
    study_ref = fixture.store.put_json(study, media_type=MATCHED_V2_STUDY_MANIFEST_MEDIA_TYPE)
    with pytest.raises(MatchedV2AdmissionError, match="canonical content"):
        admit_matched_v2_study(
            fixture.store,
            study_ref=study_ref,
            expectation=fixture.expectation,
        )

    wrong_media = fixture.score_ref.model_copy(update={"media_type": "application/json"})
    with pytest.raises(ValidationError, match="wrong media type"):
        MatchedV2StudyManifest(
            score_run_ref=wrong_media,
            full_run_ref=fixture.full_ref,
            expectation_fingerprint=fixture.expectation.fingerprint,
            shared_coordinate_fingerprint=fixture.expectation.shared_coordinate_fingerprint,
            contrast_fingerprint=fixture.expectation.contrast_fingerprint,
        )


def test_gate_query_block_rejects_quartet_and_rollout_freshness_spoof() -> None:
    block = MatchedV2GateQueryBlock(
        query_index=0,
        nomination_index=0,
        tasks=(MatchedV2GateTask(task_id="t", source_id="s", family_id="f"),),
    )
    values = block.model_dump(mode="python")
    assert block.planned_cross_condition_batch == ("score-full-cross-condition-atomic-batch")
    assert block.feedback_release_boundary == ("release-after-complete-cross-condition-batch")
    assert block.feedback_isolation_boundary == ("condition-local-no-cross-condition-disclosure")
    assert block.nomination_index == block.query_index
    with pytest.raises(ValidationError, match="rollout_seed_counts_as_fresh_task"):
        MatchedV2GateQueryBlock(**{**values, "rollout_seed_counts_as_fresh_task": True})
    with pytest.raises(ValidationError, match="exact matched quartet"):
        MatchedV2GateQueryBlock(
            **{
                **values,
                "attribution_arms": ("candidate", "parent", "revert", "placebo"),
            }
        )
    with pytest.raises(ValidationError, match="indexes must be identical"):
        MatchedV2GateQueryBlock(**{**values, "nomination_index": 1})
    with pytest.raises(ValidationError, match="planned_cross_condition_batch"):
        MatchedV2GateQueryBlock(**{**values, "planned_cross_condition_batch": "not-atomic"})
