from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import TypeAdapter, ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    BudgetPolicy,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
)
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.baselines import FrozenMutationPolicy, MutationOperation
from spiral_harness.experiments.confirmatory_arms import (
    CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE,
    PURE_AT_B_PLAN_MEDIA_TYPE,
    ConfirmatoryFourArmDesign,
    RealTaskArm,
    make_confirmatory_four_arm_design,
)
from spiral_harness.experiments.confirmatory_execution_artifacts import (
    ConfirmatoryRehearsalArtifactError,
    load_confirmatory_rehearsal_plan,
    publish_confirmatory_rehearsal_plan,
    verify_confirmatory_rehearsal_plan,
)
from spiral_harness.experiments.confirmatory_execution_contracts import (
    CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE,
    CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE,
    ConfirmatoryEvaluationUnitSeed,
    ConfirmatoryExecutionCondition,
    ConfirmatoryRehearsalPlan,
    ConfirmatorySearchSeed,
    ConfirmatoryTaskRuntimeBinding,
    confirmatory_search_seed_schedule_fingerprint,
    make_confirmatory_rehearsal_plan,
    task_implementation_bundle_fingerprint,
)
from spiral_harness.experiments.confirmatory_pure_at_b import (
    AggregationMethod,
    PureAtBAggregationRule,
    PureAtBEvaluationUnitAllocation,
    PureAtBPlan,
    PureAtBSampleAllocation,
    make_pure_at_b_plan,
)
from spiral_harness.experiments.confirmatory_resources import (
    CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
    CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE,
    AdaptiveConditionContext,
    AdaptiveExecutionCeilings,
    AdaptiveProtocolCommitments,
    ConfirmatoryTaskSplitManifest,
    ExAnteAdaptiveTopology,
    FrozenMutationPolicyArtifact,
    ModelMediatedRole,
    ModelRoleCeiling,
    MutableSeedComponent,
    MutationSurfaceGrammarBinding,
    TaskSplitEvaluationUnit,
    TaskSplitTask,
)
from spiral_harness.storage.artifact_store import ArtifactStore


@dataclass(frozen=True)
class RehearsalFixture:
    store: ArtifactStore
    plan: ConfirmatoryRehearsalPlan
    plan_ref: ArtifactRef


def _digest(label: str) -> str:
    return canonical_sha256({"test": "confirmatory-rehearsal", "label": label})


def _put_json(store: ArtifactStore, label: str) -> ArtifactRef:
    return store.put_json({"fixture": label}, media_type="application/json")


def _put_source(store: ArtifactStore, label: str) -> ArtifactRef:
    return store.put_bytes(
        f"# immutable fixture: {label}\n".encode(),
        media_type="text/x-python; charset=utf-8",
    )


def _model_spec() -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="fixture-openai-compatible",
        backend_fingerprint="fixture-backend/v1",
        model="fixture/same-model",
        revision="snapshot-2026-08-15",
        tokenizer="fixture/tokenizer",
        tokenizer_revision="snapshot-2026-08-15",
        runtime="fixture-runtime/1",
        inference=InferenceConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=100,
            timeout_seconds=30.0,
        ),
    )


def _ceilings() -> AdaptiveExecutionCeilings:
    calls = {
        ModelMediatedRole.SOLVER: 4,
        ModelMediatedRole.DIAGNOSER: 0,
        ModelMediatedRole.PROPOSER: 1,
        ModelMediatedRole.MATERIALIZER: 0,
        ModelMediatedRole.RANKER: 0,
        ModelMediatedRole.NOMINATOR: 1,
        ModelMediatedRole.JUDGE: 0,
        ModelMediatedRole.GRADER: 0,
    }
    roles = tuple(
        ModelRoleCeiling(
            role=role,
            max_calls=calls[role],
            max_tokens=calls[role] * 100,
        )
        for role in ModelMediatedRole
    )
    total_calls = sum(item.max_calls for item in roles)
    total_tokens = sum(item.max_tokens for item in roles)
    return AdaptiveExecutionCeilings(
        max_rounds=1,
        max_proposals_per_round=1,
        max_total_proposals=1,
        max_total_nominations=1,
        max_gate_queries=1,
        max_feedback_releases=1,
        max_evaluations=4,
        max_attempts_per_model_call=1,
        token_ceiling_per_model_call=100,
        role_model_calls=roles,
        max_total_model_calls=total_calls,
        max_total_tokens=total_tokens,
        max_total_model_attempts=total_calls,
        max_total_attempt_tokens=total_tokens,
        max_wall_time_seconds=600.0,
        max_cost_usd=2.0,
    )


def _build_fixture(tmp_path: object) -> RehearsalFixture:
    store = ArtifactStore(tmp_path)  # type: ignore[arg-type]
    spec = _model_spec()
    prompt_ref = store.put_bytes(b"Solve exactly.\n", media_type="text/plain; charset=utf-8")
    seed_harness = HarnessManifest(
        model_fingerprint=spec.model_fingerprint,
        runtime_fingerprint=spec.runtime_fingerprint,
        trusted_plane_version="confirmatory-rehearsal-only/v1",
        components=(
            HarnessComponentRef(
                name="system_prompt",
                kind=ComponentKind.PROMPT,
                artifact=prompt_ref,
            ),
        ),
        budget=BudgetPolicy(max_tokens=100),
    )
    seed_harness_ref = store.put_json(
        seed_harness,
        media_type=HARNESS_MANIFEST_MEDIA_TYPE,
    )

    surface_seed_ref = prompt_ref
    mutation_policy = FrozenMutationPolicyArtifact(
        seed_harness_ref=seed_harness_ref,
        policy=FrozenMutationPolicy(
            grammar_version="confirmatory-prompt-v1",
            allowed_component_kinds=(ComponentKind.PROMPT,),
            allowed_operations=(MutationOperation.REPLACE,),
            max_artifact_size_bytes=4_096,
        ),
        surface_grammars=(
            MutationSurfaceGrammarBinding(
                component_kind=ComponentKind.PROMPT,
                grammar_ref=_put_json(store, "prompt-grammar"),
                candidate_schema_ref=_put_json(store, "prompt-schema"),
                parser_implementation_ref=_put_source(store, "prompt-parser"),
                materializer_implementation_ref=_put_source(store, "prompt-materializer"),
                seed_components=(
                    MutableSeedComponent(
                        component_name="system_prompt",
                        artifact_ref=surface_seed_ref,
                    ),
                ),
                allowed_media_types=("text/plain; charset=utf-8",),
            ),
        ),
        construction_provenance_ref=_put_json(store, "mutation-policy-provenance"),
    )
    mutation_policy_ref = store.put_json(
        mutation_policy,
        media_type=CONFIRMATORY_MUTATION_POLICY_MEDIA_TYPE,
    )
    assert mutation_policy_ref == mutation_policy.artifact_ref

    task_bindings: list[ConfirmatoryTaskRuntimeBinding] = []
    split_tasks: list[TaskSplitTask] = []
    for index, task_id in enumerate(("task-alpha",)):
        task_ref = _put_json(store, f"{task_id}-manifest")
        unit_ref = _put_json(store, f"{task_id}-unit")
        unit = TaskSplitEvaluationUnit(
            evaluation_unit_id=f"{task_id}:unit-0",
            evaluation_unit_ref=unit_ref,
        )
        split_tasks.append(
            TaskSplitTask(
                task_id=task_id,
                task_manifest_ref=task_ref,
                evaluation_units=(unit,),
            )
        )
        task_bindings.append(
            ConfirmatoryTaskRuntimeBinding(
                task_id=task_id,
                task_manifest_ref=task_ref,
                evaluation_units=(unit,),
                evaluation_seeds=(
                    ConfirmatoryEvaluationUnitSeed(
                        evaluation_unit_id=unit.evaluation_unit_id,
                        evaluation_unit_ref=unit_ref,
                        rollout_seed=10_000 + index,
                    ),
                ),
                adapter_implementation_ref=_put_source(store, f"{task_id}-adapter"),
                grader_implementation_ref=_put_source(store, f"{task_id}-grader"),
                query_dag_implementation_ref=_put_json(store, f"{task_id}-query-dag"),
                retry_policy_implementation_ref=_put_source(store, f"{task_id}-retry"),
                price_table_ref=_put_json(store, f"{task_id}-price-table"),
                aggregation_implementation_ref=_put_source(store, f"{task_id}-aggregation"),
                aggregation_normalizer_ref=_put_source(store, f"{task_id}-normalizer"),
                aggregation_output_domain_ref=_put_json(store, f"{task_id}-output-domain"),
            )
        )
    bindings = tuple(task_bindings)
    split = ConfirmatoryTaskSplitManifest(
        split_id="confirmatory-rehearsal-two-task-v1",
        tasks=tuple(split_tasks),
    )
    assert (
        store.put_json(split, media_type=CONFIRMATORY_TASK_SPLIT_MEDIA_TYPE) == split.artifact_ref
    )

    solver_config_ref = _put_json(store, "solver-config")
    optimizer_config_ref = _put_json(store, "optimizer-config")
    candidate_parser_ref = _put_source(store, "candidate-parser")
    search_seeds = (ConfirmatorySearchSeed(seed_id="primary-000", seed=700_001),)
    commitments = AdaptiveProtocolCommitments(
        model_spec_fingerprint=spec.fingerprint,
        solver_config_fingerprint=solver_config_ref.sha256,
        optimizer_config_fingerprint=optimizer_config_ref.sha256,
        seed_harness_ref=seed_harness_ref,
        mutation_policy_ref=mutation_policy_ref,
        task_split_fingerprint=split.fingerprint,
        task_split_manifest=split,
        task_split_manifest_ref=split.artifact_ref,
        seed_schedule_fingerprint=confirmatory_search_seed_schedule_fingerprint(
            bindings,
            search_seeds,
        ),
        candidate_parser_fingerprint=candidate_parser_ref.sha256,
        grader_fingerprint=task_implementation_bundle_fingerprint(
            bindings,
            "grader_implementation_ref",
        ),
        query_dag_fingerprint=task_implementation_bundle_fingerprint(
            bindings,
            "query_dag_implementation_ref",
        ),
        retry_policy_fingerprint=task_implementation_bundle_fingerprint(
            bindings,
            "retry_policy_implementation_ref",
        ),
    )
    design = make_confirmatory_four_arm_design(
        adaptive_topology=ExAnteAdaptiveTopology(
            condition_context_count=2,
            condition_context_ids=(
                AdaptiveConditionContext.SCORE,
                AdaptiveConditionContext.FULL,
            ),
            protocol_commitments=commitments,
        ),
        adaptive_ceilings=_ceilings(),
    )
    store.put_json(design, media_type=CONFIRMATORY_FOUR_ARM_DESIGN_MEDIA_TYPE)

    aggregations = tuple(
        PureAtBAggregationRule(
            task_id=task.task_id,
            task_manifest_ref=task.task_manifest_ref,
            rule_id=f"{task.task_id}-fixed-aggregation-v1",
            method=AggregationMethod.TASK_NATIVE_FIXED,
            implementation_fingerprint=task.aggregation_implementation_ref.sha256,
            normalizer_fingerprint=task.aggregation_normalizer_ref.sha256,
            output_domain_fingerprint=task.aggregation_output_domain_ref.sha256,
        )
        for task in bindings
    )
    pure_units = tuple(
        PureAtBEvaluationUnitAllocation(
            evaluation_unit_id=task.evaluation_units[0].evaluation_unit_id,
            evaluation_unit_ref=task.evaluation_units[0].evaluation_unit_ref,
            task_id=task.task_id,
            task_manifest_ref=task.task_manifest_ref,
            samples=tuple(
                PureAtBSampleAllocation(
                    sample_id=f"{task.task_id}:sample-{index}",
                    seed_fingerprint=_digest(f"{task.task_id}:sample-seed-{index}"),
                    token_ceiling=100,
                    max_provider_attempts=1,
                    provider_attempt_token_ceiling=100,
                )
                for index in range(6)
            ),
            sample_count_ceiling=6,
            model_call_ceiling=6,
            token_ceiling=600,
            provider_attempt_ceiling=6,
            provider_attempt_token_ceiling=600,
        )
        for task in bindings
    )
    pure_at_b = make_pure_at_b_plan(
        four_arm_design=design,
        aggregations=aggregations,
        evaluation_units=pure_units,
    )
    store.put_json(pure_at_b, media_type=PURE_AT_B_PLAN_MEDIA_TYPE)
    store.put_json(spec, media_type=CONFIRMATORY_FROZEN_MODEL_SPEC_MEDIA_TYPE)
    preregistration_ref = _put_json(store, "sealed-preregistration-root")
    plan = make_confirmatory_rehearsal_plan(
        study_id="confirmatory-rehearsal-2026-08-15-a",
        preregistration_ref=preregistration_ref,
        four_arm_design=design,
        pure_at_b_plan=pure_at_b,
        model_spec=spec,
        seed_harness=seed_harness,
        solver_config_ref=solver_config_ref,
        optimizer_config_ref=optimizer_config_ref,
        candidate_parser_implementation_ref=candidate_parser_ref,
        task_bindings=bindings,
        search_seeds=search_seeds,
    )
    plan_ref = publish_confirmatory_rehearsal_plan(store, plan)
    return RehearsalFixture(store=store, plan=plan, plan_ref=plan_ref)


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory: pytest.TempPathFactory) -> RehearsalFixture:
    return _build_fixture(tmp_path_factory.mktemp("confirmatory-rehearsal"))


def test_plan_derives_complete_matched_blocks_unique_cells_and_one_model(
    rehearsal: RehearsalFixture,
) -> None:
    plan = rehearsal.plan
    keys = plan.execution_keys

    assert len(keys) == 5 * 1 * 1
    assert len({key.key_fingerprint for key in keys}) == len(keys)
    assert {key.condition for key in keys} == set(ConfirmatoryExecutionCondition)
    for task_id in ("task-alpha",):
        task_keys = [key for key in keys if key.task_id == task_id]
        assert len({key.matched_block_fingerprint for key in task_keys}) == 1
        assert len({key.key_fingerprint for key in task_keys}) == 5
    assert plan.verify_execution_key_membership(keys[0]) == keys[0]

    role_bindings = plan.model_role_bindings
    assert tuple(item.role for item in role_bindings) == tuple(ModelMediatedRole)
    assert all(item.model_spec_ref == plan.model_spec_ref for item in role_bindings if item.enabled)
    assert all(item.model_spec_ref is None for item in role_bindings if not item.enabled)
    assert plan.development_only
    assert not plan.live_model_calls_permitted
    assert not plan.runtime_capability_attested
    assert not plan.execution_attested
    assert not plan.provider_identity_attested
    assert not plan.same_served_revision_claim
    assert not plan.sealed_evidence
    assert not plan.reportable_result


def test_plan_rejects_missing_duplicate_or_caller_reported_matrix_entries(
    rehearsal: RehearsalFixture,
) -> None:
    plan = rehearsal.plan
    values = plan.model_dump(mode="python")

    with pytest.raises(ValidationError, match="at least 1 item"):
        plan.model_copy(update={"task_bindings": ()})
    with pytest.raises(ValidationError, match="at most 1 item"):
        plan.model_copy(update={"task_bindings": (plan.task_bindings[0],) * 2})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ConfirmatoryRehearsalPlan.model_validate(
            {**values, "execution_keys": ()},
            strict=True,
        )
    with pytest.raises(ValidationError, match="at most 1 item"):
        plan.model_copy(
            update={
                "search_seeds": (
                    ConfirmatorySearchSeed(seed_id="primary-000", seed=1),
                    ConfirmatorySearchSeed(seed_id="primary-001", seed=2),
                )
            }
        )


def test_raw_search_seed_reuse_is_rejected_even_under_a_different_id(
    rehearsal: RehearsalFixture,
) -> None:
    with pytest.raises(ValueError, match="raw search seed values must not repeat"):
        confirmatory_search_seed_schedule_fingerprint(
            rehearsal.plan.task_bindings,
            (
                ConfirmatorySearchSeed(seed_id="seed-a", seed=42),
                ConfirmatorySearchSeed(seed_id="seed-b", seed=42),
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "development_only",
        "live_model_calls_permitted",
        "runtime_capability_attested",
        "execution_attested",
        "provider_identity_attested",
        "same_served_revision_claim",
        "sealed_evidence",
        "reportable_result",
    ),
)
def test_no_construction_or_publication_path_can_upgrade_rehearsal_claims(
    rehearsal: RehearsalFixture,
    field_name: str,
) -> None:
    plan = rehearsal.plan
    values = plan.model_dump(mode="python")
    forbidden_value = field_name != "development_only"
    changed = {**values, field_name: forbidden_value}

    with pytest.raises(ValidationError):
        ConfirmatoryRehearsalPlan.model_validate(changed, strict=True)
    with pytest.raises(ValidationError):
        plan.model_copy(update={field_name: forbidden_value})
    with pytest.raises(ValidationError):
        ConfirmatoryRehearsalPlan.model_construct(**changed)
    with pytest.raises(ValidationError):
        TypeAdapter(ConfirmatoryRehearsalPlan).validate_python(changed, strict=True)


def test_lineage_and_runtime_content_changes_change_derived_keys(
    rehearsal: RehearsalFixture,
) -> None:
    plan = rehearsal.plan
    original = plan.execution_keys[0]
    foreign_prereg = _put_json(rehearsal.store, "different-preregistration")
    foreign_adapter = _put_source(rehearsal.store, "different-adapter")
    foreign_price = _put_json(rehearsal.store, "different-price-table")
    first = plan.task_bindings[0]

    alternatives = (
        plan.model_copy(update={"study_id": "different-study"}),
        plan.model_copy(update={"preregistration_ref": foreign_prereg}),
        plan.model_copy(
            update={
                "task_bindings": (
                    first.model_copy(update={"adapter_implementation_ref": foreign_adapter}),
                )
            }
        ),
        plan.model_copy(
            update={"task_bindings": (first.model_copy(update={"price_table_ref": foreign_price}),)}
        ),
    )
    for alternative in alternatives:
        corresponding = next(
            key
            for key in alternative.execution_keys
            if key.condition is original.condition and key.task_id == original.task_id
        )
        assert corresponding.matched_block_fingerprint != original.matched_block_fingerprint
        assert corresponding.key_fingerprint != original.key_fingerprint
        with pytest.raises(ValueError, match="not a member"):
            plan.verify_execution_key_membership(corresponding)


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("aggregation_implementation_ref", "implementation ref"),
        ("aggregation_normalizer_ref", "normalizer ref"),
        ("aggregation_output_domain_ref", "output-domain ref"),
    ),
)
def test_every_aggregation_runtime_identity_is_bound_to_pure_at_b(
    rehearsal: RehearsalFixture,
    field_name: str,
    message: str,
) -> None:
    task = rehearsal.plan.task_bindings[0]
    changed = task.model_copy(
        update={field_name: _put_source(rehearsal.store, f"foreign-{field_name}")}
    )
    with pytest.raises(ValidationError, match=message):
        rehearsal.plan.model_copy(update={"task_bindings": (changed,)})


@pytest.mark.parametrize(
    ("binding_field", "rule_field", "json_content"),
    (
        ("aggregation_normalizer_ref", "normalizer_fingerprint", False),
        ("aggregation_output_domain_ref", "output_domain_fingerprint", True),
    ),
)
def test_valid_aggregation_content_change_changes_the_cell_key(
    rehearsal: RehearsalFixture,
    binding_field: str,
    rule_field: str,
    json_content: bool,
) -> None:
    replacement = (
        _put_json(rehearsal.store, f"replacement-{binding_field}")
        if json_content
        else _put_source(rehearsal.store, f"replacement-{binding_field}")
    )
    task = rehearsal.plan.task_bindings[0].model_copy(update={binding_field: replacement})
    rule = rehearsal.plan.pure_at_b_plan.aggregations[0].model_copy(
        update={rule_field: replacement.sha256}
    )
    pure_at_b = rehearsal.plan.pure_at_b_plan.model_copy(update={"aggregations": (rule,)})
    pure_ref = rehearsal.store.put_json(pure_at_b, media_type=PURE_AT_B_PLAN_MEDIA_TYPE)
    changed = rehearsal.plan.model_copy(
        update={
            "task_bindings": (task,),
            "pure_at_b_plan": pure_at_b,
            "pure_at_b_plan_ref": pure_ref,
        }
    )

    assert (
        changed.execution_keys[0].matched_block_fingerprint
        != rehearsal.plan.execution_keys[0].matched_block_fingerprint
    )


def test_artifact_boundary_publishes_reloads_and_verifies_every_dependency(
    rehearsal: RehearsalFixture,
) -> None:
    assert load_confirmatory_rehearsal_plan(rehearsal.store, rehearsal.plan_ref) == rehearsal.plan
    assert (
        verify_confirmatory_rehearsal_plan(
            rehearsal.store,
            rehearsal.plan_ref,
            rehearsal.plan,
        )
        == rehearsal.plan
    )


def test_artifact_boundary_rejects_wrong_media_missing_content_and_expected_drift(
    rehearsal: RehearsalFixture,
) -> None:
    wrong_media = rehearsal.plan_ref.model_copy(update={"media_type": "application/json"})
    with pytest.raises(ConfirmatoryRehearsalArtifactError, match="wrong media type"):
        load_confirmatory_rehearsal_plan(rehearsal.store, wrong_media)

    missing_adapter = ArtifactRef(
        sha256=_digest("missing-adapter"),
        size=123,
        media_type="text/x-python; charset=utf-8",
    )
    changed_task = rehearsal.plan.task_bindings[0].model_copy(
        update={"adapter_implementation_ref": missing_adapter}
    )
    changed_plan = rehearsal.plan.model_copy(update={"task_bindings": (changed_task,)})
    with pytest.raises(ConfirmatoryRehearsalArtifactError, match="cannot be loaded"):
        publish_confirmatory_rehearsal_plan(rehearsal.store, changed_plan)

    drifted = rehearsal.plan.model_copy(update={"study_id": "expected-drift"})
    with pytest.raises(ConfirmatoryRehearsalArtifactError, match="expected canonical plan"):
        verify_confirmatory_rehearsal_plan(rehearsal.store, rehearsal.plan_ref, drifted)


def test_artifact_boundary_rejects_noncanonical_json_and_conflicting_alias_metadata(
    rehearsal: RehearsalFixture,
) -> None:
    payload = b" " + canonical_json_bytes(rehearsal.plan)
    noncanonical_ref = ArtifactRef(
        sha256=sha256_bytes(payload),
        size=len(payload),
        media_type=CONFIRMATORY_REHEARSAL_PLAN_MEDIA_TYPE,
    )
    path = rehearsal.store.path_for(noncanonical_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with pytest.raises(ConfirmatoryRehearsalArtifactError, match="canonical typed"):
        load_confirmatory_rehearsal_plan(rehearsal.store, noncanonical_ref)

    task = rehearsal.plan.task_bindings[0]
    alias = task.price_table_ref.model_copy(update={"media_type": "text/x-python; charset=utf-8"})
    changed_task = task.model_copy(update={"adapter_implementation_ref": alias})
    changed = rehearsal.plan.model_copy(update={"task_bindings": (changed_task,)})
    with pytest.raises(ConfirmatoryRehearsalArtifactError, match="conflicting refs"):
        publish_confirmatory_rehearsal_plan(rehearsal.store, changed)


def test_artifact_boundary_detects_seed_component_tampering(
    rehearsal: RehearsalFixture,
) -> None:
    component_ref = rehearsal.plan.seed_harness.components[0].artifact
    path = rehearsal.store.path_for(component_ref)
    original = path.read_bytes()
    try:
        path.write_bytes(b"tampered")
        with pytest.raises(ConfirmatoryRehearsalArtifactError, match="cannot be loaded"):
            load_confirmatory_rehearsal_plan(rehearsal.store, rehearsal.plan_ref)
    finally:
        path.write_bytes(original)


def test_typed_refs_and_evaluation_seed_roster_fail_closed(
    rehearsal: RehearsalFixture,
) -> None:
    plan = rehearsal.plan
    fake_model_ref = plan.model_spec_ref.model_copy(update={"size": plan.model_spec_ref.size + 1})
    with pytest.raises(ValidationError, match="model spec ref differs"):
        plan.model_copy(update={"model_spec_ref": fake_model_ref})

    first = plan.task_bindings[0]
    with pytest.raises(ValidationError, match="at least 1 item"):
        first.model_copy(update={"evaluation_seeds": ()})

    duplicate = first.evaluation_seeds[0].model_copy(update={"evaluation_unit_id": "foreign-unit"})
    with pytest.raises(ValidationError, match="not bijective"):
        first.model_copy(update={"evaluation_seeds": (duplicate,)})

    second_ref = _put_json(rehearsal.store, "second-evaluation-unit")
    second_unit = TaskSplitEvaluationUnit(
        evaluation_unit_id="task-alpha:unit-1",
        evaluation_unit_ref=second_ref,
    )
    second_seed = ConfirmatoryEvaluationUnitSeed(
        evaluation_unit_id=second_unit.evaluation_unit_id,
        evaluation_unit_ref=second_ref,
        rollout_seed=first.evaluation_seeds[0].rollout_seed,
    )
    with pytest.raises(ValidationError, match="rollout seeds must not repeat"):
        first.model_copy(
            update={
                "evaluation_units": (*first.evaluation_units, second_unit),
                "evaluation_seeds": (*first.evaluation_seeds, second_seed),
            }
        )


def test_design_and_pure_at_b_remain_prospective_nonreportable_values(
    rehearsal: RehearsalFixture,
) -> None:
    design: ConfirmatoryFourArmDesign = rehearsal.plan.four_arm_design
    pure_at_b: PureAtBPlan = rehearsal.plan.pure_at_b_plan

    assert not design.execution_attested
    assert not design.provider_identity_attested
    assert not design.sealed_evidence
    assert not design.reportable_result
    assert not pure_at_b.execution_attested
    assert not pure_at_b.provider_identity_attested
    assert not pure_at_b.sealed_evidence
    assert not pure_at_b.reportable_result
    assert design.arm(RealTaskArm.FULL).adaptive_topology is not None
