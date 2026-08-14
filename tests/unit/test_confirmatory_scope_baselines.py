from __future__ import annotations

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.experiments.baselines import FrozenMutationPolicy, MutationOperation
from spiral_harness.experiments.confirmatory_arms import (
    ConfirmatoryFourArmDesign,
    OptimizerFeedbackMode,
    PromotionRule,
    make_confirmatory_four_arm_design,
)
from spiral_harness.experiments.confirmatory_resources import (
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
from spiral_harness.experiments.confirmatory_scope_baselines import (
    CandidateSource,
    ConfirmatoryScopeBaselineDesign,
    RandomValidCatalog,
    RandomValidCatalogEntry,
    ScopeBaselineKind,
    ScopeBaselinePlan,
    make_confirmatory_scope_baseline_design,
)
from spiral_harness.experiments.confirmatory_scope_contracts import (
    PromptOnlyEnforcementReceipt,
    RandomValidParentSnapshot,
    RandomValidSamplerConfig,
    RandomValidSelectionReceipt,
    make_random_valid_selection_receipt,
    verify_random_valid_selection_chain,
)


def _digest(label: str) -> str:
    return canonical_sha256({"test": "confirmatory-scope-baselines", "label": label})


def _mutation_policy(
    component_kinds: tuple[ComponentKind, ...] = (
        ComponentKind.CONTROL_FLOW,
        ComponentKind.PROMPT,
        ComponentKind.SKILL,
    ),
) -> FrozenMutationPolicy:
    return FrozenMutationPolicy(
        grammar_version="confirmatory-full-v1",
        allowed_component_kinds=component_kinds,
        allowed_operations=(MutationOperation.REPLACE,),
        max_artifact_size_bytes=65_536,
    )


def _ref(label: str, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=_digest(label), size=100 + len(label), media_type=media_type)


def _mutation_policy_artifact(
    policy: FrozenMutationPolicy | None = None,
) -> FrozenMutationPolicyArtifact:
    checked = policy or _mutation_policy()
    return FrozenMutationPolicyArtifact(
        seed_harness_ref=_ref("seed-harness"),
        policy=checked,
        surface_grammars=tuple(
            MutationSurfaceGrammarBinding(
                component_kind=kind,
                grammar_ref=_ref(f"{kind.value}-grammar"),
                candidate_schema_ref=_ref(f"{kind.value}-schema"),
                parser_implementation_ref=_ref(f"{kind.value}-parser", "text/x-python"),
                materializer_implementation_ref=_ref(
                    f"{kind.value}-materializer",
                    "text/x-python",
                ),
                seed_components=(
                    MutableSeedComponent(
                        component_name=f"{kind.value}-component",
                        artifact_ref=_ref(f"{kind.value}-seed", "text/plain"),
                    ),
                ),
                allowed_media_types=("text/plain",),
            )
            for kind in checked.allowed_component_kinds
        ),
        construction_provenance_ref=_ref("policy-provenance"),
    )


def _catalog(policy_artifact: FrozenMutationPolicyArtifact) -> RandomValidCatalog:
    entries: list[RandomValidCatalogEntry] = []
    surfaces = policy_artifact.surface_grammars
    for index in range(max(4, len(surfaces))):
        surface = surfaces[index % len(surfaces)]
        seed = surface.seed_components[0]
        entries.append(
            RandomValidCatalogEntry(
                entry_id=f"entry-{index:02d}-{surface.component_kind.value}",
                component_kind=surface.component_kind,
                target_component_name=seed.component_name,
                replacement_ref=_ref(f"replacement-{index}", "text/plain"),
                transformation_implementation_ref=_ref(
                    f"{surface.component_kind.value}-replacement-template",
                    "text/x-python",
                ),
                hypothesis_ref=_ref(f"hypothesis-{index}"),
            )
        )
    return RandomValidCatalog(
        catalog_id="confirmatory-random-valid-v1",
        seed_harness_ref=policy_artifact.seed_harness_ref,
        mutation_policy_artifact=policy_artifact,
        mutation_policy_ref=policy_artifact.artifact_ref,
        entries=tuple(entries),
        sampling_implementation_ref=_ref("uniform-parent-sampler", "text/x-python"),
        construction_provenance_ref=_ref("catalog-provenance"),
        construction_usage_ref=_ref("catalog-construction-usage"),
    )


def _commitments(
    mutation_policy_artifact: FrozenMutationPolicyArtifact | None = None,
) -> AdaptiveProtocolCommitments:
    artifact = mutation_policy_artifact or _mutation_policy_artifact()
    split = ConfirmatoryTaskSplitManifest(
        split_id="scope-baseline-real-task-v1",
        tasks=(
            TaskSplitTask(
                task_id="scope-task",
                task_manifest_ref=_ref("scope-task-manifest"),
                evaluation_units=(
                    TaskSplitEvaluationUnit(
                        evaluation_unit_id="scope-task:unit-0",
                        evaluation_unit_ref=_ref("scope-task-unit-0"),
                    ),
                ),
            ),
        ),
    )
    return AdaptiveProtocolCommitments(
        model_spec_fingerprint=_digest("model"),
        solver_config_fingerprint=_digest("solver"),
        optimizer_config_fingerprint=_digest("full-optimizer"),
        seed_harness_ref=artifact.seed_harness_ref,
        mutation_policy_ref=artifact.artifact_ref,
        task_split_fingerprint=split.fingerprint,
        task_split_manifest=split,
        task_split_manifest_ref=split.artifact_ref,
        seed_schedule_fingerprint=_digest("seed-schedule"),
        candidate_parser_fingerprint=_digest("full-parser"),
        grader_fingerprint=_digest("grader"),
        query_dag_fingerprint=_digest("query-dag"),
        retry_policy_fingerprint=_digest("retry"),
    )


def _ceilings() -> AdaptiveExecutionCeilings:
    calls = {
        ModelMediatedRole.SOLVER: 40,
        ModelMediatedRole.DIAGNOSER: 2,
        ModelMediatedRole.PROPOSER: 4,
        ModelMediatedRole.MATERIALIZER: 0,
        ModelMediatedRole.RANKER: 2,
        ModelMediatedRole.NOMINATOR: 2,
        ModelMediatedRole.JUDGE: 0,
        ModelMediatedRole.GRADER: 0,
    }
    roles = tuple(
        ModelRoleCeiling(
            role=role,
            max_calls=calls[role],
            max_tokens=calls[role] * 1_000,
        )
        for role in ModelMediatedRole
    )
    total_calls = sum(item.max_calls for item in roles)
    total_tokens = sum(item.max_tokens for item in roles)
    return AdaptiveExecutionCeilings(
        max_rounds=2,
        max_proposals_per_round=2,
        max_total_proposals=4,
        max_total_nominations=2,
        max_gate_queries=2,
        max_feedback_releases=2,
        max_evaluations=40,
        max_attempts_per_model_call=2,
        token_ceiling_per_model_call=1_000,
        role_model_calls=roles,
        max_total_model_calls=total_calls,
        max_total_tokens=total_tokens,
        max_total_model_attempts=total_calls * 2,
        max_total_attempt_tokens=total_tokens * 2,
        max_wall_time_seconds=1_800.0,
        max_cost_usd=10.0,
    )


def _four_arm_design(
    mutation_policy_artifact: FrozenMutationPolicyArtifact | None = None,
) -> ConfirmatoryFourArmDesign:
    return make_confirmatory_four_arm_design(
        adaptive_topology=ExAnteAdaptiveTopology(
            condition_context_count=2,
            condition_context_ids=(
                AdaptiveConditionContext.SCORE,
                AdaptiveConditionContext.FULL,
            ),
            protocol_commitments=_commitments(mutation_policy_artifact),
        ),
        adaptive_ceilings=_ceilings(),
    )


def _scope_design() -> ConfirmatoryScopeBaselineDesign:
    mutation_policy_artifact = _mutation_policy_artifact()
    catalog = _catalog(mutation_policy_artifact)
    return make_confirmatory_scope_baseline_design(
        four_arm_design=_four_arm_design(mutation_policy_artifact),
        full_mutation_policy_artifact=mutation_policy_artifact,
        random_candidate_parser_fingerprint=_digest("catalog-parser"),
        random_catalog=catalog,
        random_sampler_config=RandomValidSamplerConfig(
            seed_fingerprint=_digest("random-valid-selection-seed"),
            max_proposals_per_round=2,
        ),
        prompt_projection_implementation_ref=_ref("prompt-policy-projector", "text/x-python"),
        prompt_enforcement_implementation_ref=_ref("prompt-policy-enforcer", "text/x-python"),
        prompt_projection_provenance_ref=_ref("prompt-policy-provenance"),
    )


def _parent_snapshot(
    catalog: RandomValidCatalog,
    *,
    label: str,
    overrides: dict[tuple[ComponentKind, str], ArtifactRef] | None = None,
) -> RandomValidParentSnapshot:
    current = {
        (surface.component_kind, component.component_name): component.artifact_ref
        for surface in catalog.mutation_policy_artifact.surface_grammars
        for component in surface.seed_components
    }
    current.update(overrides or {})
    targets = {(entry.component_kind, entry.target_component_name) for entry in catalog.entries}
    manifest = HarnessManifest(
        model_fingerprint=f"model:{label}",
        runtime_fingerprint="runtime:v1",
        trusted_plane_version="trusted:v1",
        components=tuple(
            HarnessComponentRef(
                kind=kind,
                name=name,
                artifact=current[(kind, name)],
            )
            for kind, name in targets
        ),
    )
    payload = canonical_json_bytes(manifest)
    return RandomValidParentSnapshot(
        parent_manifest=manifest,
        parent_harness_ref=ArtifactRef(
            sha256=canonical_sha256(manifest),
            size=len(payload),
            media_type=HARNESS_MANIFEST_MEDIA_TYPE,
        ),
    )


def _candidate_mutation(kind: ComponentKind, label: str) -> CandidateMutation:
    component_name = f"{kind.value}-component"
    return CandidateMutation(
        target_component=component_name,
        before=HarnessComponentRef(
            name=component_name,
            kind=kind,
            artifact=_ref(f"{label}:before", "text/plain"),
        ),
        after=HarnessComponentRef(
            name=component_name,
            kind=kind,
            artifact=_ref(f"{label}:after", "text/plain"),
        ),
        hypothesis=MutationHypothesis(
            evidence_refs=(_ref(f"{label}:evidence"),),
            where="scope enforcement test",
            why="exercise projected policy",
            expected_activation="candidate parsed",
            expected_adherence="scope filter invoked",
            expected_behavior="prompt allowed and other surfaces rejected",
            expected_benefit="prevent treatment leakage",
            protected_slices=("all",),
            falsifier="non-prompt accepted",
            negative_control="prompt accepted",
            risks=("forged kind",),
        ),
    )


def _candidate_mutation_ref(mutation: CandidateMutation) -> ArtifactRef:
    payload = canonical_json_bytes(mutation)
    return ArtifactRef(
        sha256=canonical_sha256(mutation),
        size=len(payload),
        media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
    )


def _content(value: object) -> dict[str, object]:
    return value.model_dump(mode="python", round_trip=True)  # type: ignore[attr-defined,no-any-return]


def test_scope_design_closes_the_two_secondary_baselines_without_claim_upgrade() -> None:
    design = _scope_design()
    assert tuple(item.kind for item in design.baselines) == (
        ScopeBaselineKind.RANDOM_VALID,
        ScopeBaselineKind.PROMPT_ONLY,
    )
    random_valid = design.baseline(ScopeBaselineKind.RANDOM_VALID)
    prompt_only = design.baseline(ScopeBaselineKind.PROMPT_ONLY)

    assert random_valid.candidate_source is CandidateSource.UNIFORM_PARENT_CONDITIONAL_CATALOG
    assert random_valid.optimizer_feedback is OptimizerFeedbackMode.NONE
    assert random_valid.promotion_rule is PromotionRule.MECHANISM_GATED
    assert random_valid.optimizer_model_calls_permitted is False
    assert random_valid.permitted_model_roles == (
        ModelMediatedRole.SOLVER,
        ModelMediatedRole.JUDGE,
        ModelMediatedRole.GRADER,
    )
    assert random_valid.champion_state_transition_visible is True
    assert random_valid.realized_usage_equality_claimed is False
    assert random_valid.role_permissions_runtime_attested is False
    assert random_valid.finite_catalog is not None
    assert len(random_valid.finite_catalog.entries) == 4
    assert random_valid.finite_catalog_ref == random_valid.finite_catalog.artifact_ref
    assert random_valid.random_sampler_binding is not None
    assert random_valid.random_sampler_binding.catalog_ref == random_valid.finite_catalog_ref
    assert random_valid.random_sampler_binding.sampling_algorithm == (
        "sha256-counter-rejection-parent-conditional-uniform-v1"
    )
    assert (
        random_valid.random_sampler_binding.sampler_config.max_proposals_per_round
        == random_valid.ceilings.max_proposals_per_round
    )
    assert (
        random_valid.random_sampler_binding.sampler_config.without_replacement_across_complete_chain
        is True
    )
    assert random_valid.random_parent_eligible_selection_receipts_required is True
    assert random_valid.prompt_surface_enforcement_receipts_required is False
    assert random_valid.mutation_policy_artifact == design.full_mutation_policy_artifact
    assert random_valid.mutation_policy_ref == design.full_mutation_policy_ref
    assert random_valid.action_capability.mutable_component_kinds == (
        design.full_mutation_policy_artifact.policy.allowed_component_kinds
    )

    assert prompt_only.candidate_source is CandidateSource.SAME_MODEL_PROMPT_OPTIMIZER
    assert prompt_only.optimizer_feedback is OptimizerFeedbackMode.MECHANISM_VISIBLE
    assert prompt_only.promotion_rule is PromotionRule.MECHANISM_GATED
    assert prompt_only.optimizer_model_calls_permitted is True
    assert prompt_only.permitted_model_roles == tuple(ModelMediatedRole)
    assert prompt_only.mutation_policy_artifact == design.full_mutation_policy_artifact
    assert prompt_only.action_capability.mutable_component_kinds == (ComponentKind.PROMPT,)
    assert prompt_only.finite_catalog is None
    assert prompt_only.finite_catalog_ref is None
    assert prompt_only.prompt_policy_projection is not None
    assert (
        prompt_only.prompt_policy_projection_ref
        == prompt_only.prompt_policy_projection.artifact_ref
    )
    assert prompt_only.prompt_policy_projection.projected_policy.allowed_component_kinds == (
        ComponentKind.PROMPT,
    )
    assert prompt_only.random_parent_eligible_selection_receipts_required is False
    assert prompt_only.prompt_surface_enforcement_receipts_required is True
    assert prompt_only.scope_enforcement_runtime_attested is False

    assert random_valid.ceilings == prompt_only.ceilings
    assert design.secondary_scope_baselines is True
    assert design.primary_policy_contrast_unchanged is True
    assert design.structural_scope_profile_matched is True
    assert design.scope_effect_identified is False
    assert design.catalog_coverage_attested is False
    assert design.execution_attested is False
    assert design.provider_identity_attested is False
    assert design.sealed_evidence is False
    assert design.reportable_result is False


def test_scope_baselines_share_all_non_optimizer_primary_commitments() -> None:
    design = _scope_design()
    anchor = _commitments()
    for baseline in design.baselines:
        commitments = baseline.topology.protocol_commitments
        assert commitments.model_spec_fingerprint == anchor.model_spec_fingerprint
        assert commitments.solver_config_fingerprint == anchor.solver_config_fingerprint
        assert commitments.mutation_policy_ref == anchor.mutation_policy_ref
        assert commitments.seed_harness_ref == anchor.seed_harness_ref
        assert commitments.task_split_fingerprint == anchor.task_split_fingerprint
        assert commitments.seed_schedule_fingerprint == anchor.seed_schedule_fingerprint
        assert commitments.grader_fingerprint == anchor.grader_fingerprint
        assert commitments.query_dag_fingerprint == anchor.query_dag_fingerprint
        assert commitments.retry_policy_fingerprint == anchor.retry_policy_fingerprint
    random_commitments = design.baseline(
        ScopeBaselineKind.RANDOM_VALID
    ).topology.protocol_commitments
    prompt_commitments = design.baseline(
        ScopeBaselineKind.PROMPT_ONLY
    ).topology.protocol_commitments
    assert random_commitments.optimizer_config_fingerprint != anchor.optimizer_config_fingerprint
    assert prompt_commitments.optimizer_config_fingerprint == anchor.optimizer_config_fingerprint
    assert prompt_commitments.candidate_parser_fingerprint == anchor.candidate_parser_fingerprint


@pytest.mark.parametrize(
    ("kind", "update", "message"),
    [
        (
            ScopeBaselineKind.RANDOM_VALID,
            {"optimizer_model_calls_permitted": True},
            "frozen scope profile",
        ),
        (
            ScopeBaselineKind.RANDOM_VALID,
            {"finite_catalog_ref": None},
            "frozen finite catalog",
        ),
        (
            ScopeBaselineKind.RANDOM_VALID,
            {"permitted_model_roles": tuple(ModelMediatedRole)},
            "forbid every optimizer-model role",
        ),
        (
            ScopeBaselineKind.PROMPT_ONLY,
            {"optimizer_feedback": OptimizerFeedbackMode.SCORE_ONLY},
            "frozen scope profile",
        ),
        (
            ScopeBaselineKind.PROMPT_ONLY,
            {
                "action_capability": _scope_design()
                .baseline(ScopeBaselineKind.RANDOM_VALID)
                .action_capability
            },
            "action capability differs",
        ),
    ],
)
def test_scope_profiles_fail_closed(
    kind: ScopeBaselineKind,
    update: dict[str, object],
    message: str,
) -> None:
    values = _content(_scope_design().baseline(kind))
    values.update(update)
    with pytest.raises(ValidationError, match=message):
        ScopeBaselinePlan.model_validate(values, strict=True)


def test_design_rejects_budget_or_shared_coordinate_drift() -> None:
    design = _scope_design()
    baselines = list(design.baselines)
    prompt = baselines[1]
    changed_commitments = prompt.topology.protocol_commitments.model_copy(
        update={"grader_fingerprint": _digest("another-grader")}
    )
    changed_topology = prompt.topology.model_copy(
        update={"protocol_commitments": changed_commitments}
    )
    baselines[1] = prompt.model_copy(update={"topology": changed_topology})
    values = _content(design)
    values["baselines"] = tuple(baselines)

    with pytest.raises(ValidationError, match="shared grader_fingerprint"):
        ConfirmatoryScopeBaselineDesign.model_validate(values, strict=True)

    baselines = list(design.baselines)
    ceilings = _content(baselines[0].ceilings)
    ceilings["max_cost_usd"] = 9.0
    baselines[0] = baselines[0].model_copy(
        update={"ceilings": AdaptiveExecutionCeilings.model_validate(ceilings, strict=True)}
    )
    values = _content(design)
    values["baselines"] = tuple(baselines)
    with pytest.raises(ValidationError, match="ceilings differ from FULL"):
        ConfirmatoryScopeBaselineDesign.model_validate(values, strict=True)


def test_scope_baseline_cannot_swap_the_full_mutation_policy() -> None:
    design = _scope_design()
    random_valid = design.baseline(ScopeBaselineKind.RANDOM_VALID)
    prompt_policy = _mutation_policy((ComponentKind.PROMPT,))
    prompt_artifact = _mutation_policy_artifact(prompt_policy)
    changed_commitments = random_valid.topology.protocol_commitments.model_copy(
        update={
            "seed_harness_ref": prompt_artifact.seed_harness_ref,
            "mutation_policy_ref": prompt_artifact.artifact_ref,
        }
    )
    changed_topology = random_valid.topology.model_copy(
        update={"protocol_commitments": changed_commitments}
    )
    changed_catalog = _catalog(prompt_artifact)
    with pytest.raises(ValidationError, match="sampler binds a different catalog"):
        random_valid.model_copy(
            update={
                "topology": changed_topology,
                "mutation_policy_artifact": prompt_artifact,
                "mutation_policy_ref": prompt_artifact.artifact_ref,
                "finite_catalog": changed_catalog,
                "finite_catalog_ref": changed_catalog.artifact_ref,
                "action_capability": random_valid.action_capability.model_copy(
                    update={"mutable_component_kinds": (ComponentKind.PROMPT,)}
                ),
            }
        )


def test_prompt_only_requires_a_strictly_narrower_surface() -> None:
    prompt_policy = _mutation_policy((ComponentKind.PROMPT,))
    prompt_artifact = _mutation_policy_artifact(prompt_policy)
    with pytest.raises(ValidationError, match="undefined"):
        make_confirmatory_scope_baseline_design(
            four_arm_design=_four_arm_design(prompt_artifact),
            full_mutation_policy_artifact=prompt_artifact,
            random_candidate_parser_fingerprint=_digest("catalog-parser"),
            random_catalog=_catalog(prompt_artifact),
            random_sampler_config=RandomValidSamplerConfig(
                seed_fingerprint=_digest("random-valid-selection-seed"),
                max_proposals_per_round=2,
            ),
            prompt_projection_implementation_ref=_ref("prompt-policy-projector", "text/x-python"),
            prompt_enforcement_implementation_ref=_ref("prompt-policy-enforcer", "text/x-python"),
            prompt_projection_provenance_ref=_ref("prompt-policy-provenance"),
        )


@pytest.mark.parametrize(
    "field_name",
    ("optimizer_config_fingerprint", "candidate_parser_fingerprint"),
)
def test_prompt_only_reuses_full_optimizer_and_parser(field_name: str) -> None:
    design = _scope_design()
    baselines = list(design.baselines)
    prompt = baselines[1]
    changed_commitments = prompt.topology.protocol_commitments.model_copy(
        update={field_name: _digest(f"changed-{field_name}")}
    )
    changed_topology = prompt.topology.model_copy(
        update={"protocol_commitments": changed_commitments}
    )
    baselines[1] = prompt.model_copy(update={"topology": changed_topology})
    values = _content(design)
    values["baselines"] = tuple(baselines)
    with pytest.raises(ValidationError, match=f"shared {field_name}"):
        ConfirmatoryScopeBaselineDesign.model_validate(values, strict=True)


def test_mutation_policy_artifact_ref_cannot_be_forged_at_the_design_boundary() -> None:
    design = _scope_design()
    values = _content(design)
    values["full_mutation_policy_ref"] = _ref(
        "forged-policy-ref",
        design.full_mutation_policy_ref.media_type,
    )
    with pytest.raises(ValidationError, match="canonical artifact"):
        ConfirmatoryScopeBaselineDesign.model_validate(values, strict=True)


def test_random_catalog_rejects_duplicate_or_unknown_templates() -> None:
    catalog = _catalog(_mutation_policy_artifact())
    content = _content(catalog)
    entries = list(content["entries"])  # type: ignore[arg-type]
    entries[1] = {**entries[1], "entry_id": entries[0]["entry_id"]}
    content["entries"] = tuple(entries)
    with pytest.raises(ValidationError, match="entry IDs must be unique"):
        RandomValidCatalog.model_validate(content, strict=True)

    entries = list(catalog.entries)
    entries[1] = entries[0].model_copy(
        update={
            "entry_id": "different-id-same-final-action",
            "transformation_implementation_ref": _ref(
                "different-transformer-same-final-action", "text/x-python"
            ),
            "hypothesis_ref": _ref("different-hypothesis-same-final-action"),
        }
    )
    with pytest.raises(ValidationError, match="duplicate catalog templates"):
        RandomValidCatalog.model_validate(
            {**_content(catalog), "entries": tuple(entries)},
            strict=True,
        )

    entries = list(catalog.entries)
    entries[0] = entries[0].model_copy(update={"target_component_name": "unknown-component"})
    with pytest.raises(ValidationError, match="unknown component coordinate"):
        RandomValidCatalog.model_validate(
            {**_content(catalog), "entries": tuple(entries)},
            strict=True,
        )


def test_random_catalog_need_not_fake_continuous_eligibility_from_initial_length() -> None:
    design = _scope_design()
    random_valid = design.baseline(ScopeBaselineKind.RANDOM_VALID)
    assert random_valid.finite_catalog is not None
    reduced_catalog = RandomValidCatalog.model_validate(
        {
            **_content(random_valid.finite_catalog),
            "entries": random_valid.finite_catalog.entries[:3],
        },
        strict=True,
    )
    assert len(reduced_catalog.entries) < random_valid.ceilings.max_total_proposals


def test_random_selection_receipts_replay_parent_eligibility_selection_and_chain() -> None:
    random_valid = _scope_design().baseline(ScopeBaselineKind.RANDOM_VALID)
    catalog = random_valid.finite_catalog
    binding = random_valid.random_sampler_binding
    assert catalog is not None and binding is not None
    parent_0 = _parent_snapshot(catalog, label="round-0")
    receipt_0 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent_0,
        previous_receipt=None,
    )
    assert receipt_0.selected_entry_id in receipt_0.eligible_entry_ids
    selected = next(
        entry for entry in catalog.entries if entry.entry_id == receipt_0.selected_entry_id
    )
    receipt_1 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent_0,
        previous_receipt=receipt_0,
    )
    assert selected.entry_id in receipt_1.eligible_entry_ids
    assert selected.entry_id not in receipt_1.selection_pool_entry_ids
    assert receipt_1.selected_entry_id != selected.entry_id

    parent_1 = _parent_snapshot(
        catalog,
        label="round-1",
        overrides={
            (selected.component_kind, selected.target_component_name): selected.replacement_ref
        },
    )
    receipt_2 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent_1,
        previous_receipt=receipt_1,
    )

    assert selected.entry_id not in receipt_2.eligible_entry_ids
    assert receipt_2.round_index == 1
    assert receipt_2.proposal_index == 0
    assert set(receipt_2.prior_selected_entry_ids_in_chain) == {
        receipt_0.selected_entry_id,
        receipt_1.selected_entry_id,
    }
    assert verify_random_valid_selection_chain((receipt_0, receipt_1, receipt_2)) == (
        receipt_0,
        receipt_1,
        receipt_2,
    )

    assert receipt_1.selected_entry_id is not None
    omitted_earlier_round = receipt_2.model_copy(
        update={"prior_selected_entry_ids_in_chain": (receipt_1.selected_entry_id,)}
    )
    with pytest.raises(ValueError, match="global without-replacement chain is broken"):
        verify_random_valid_selection_chain((receipt_0, receipt_1, omitted_earlier_round))


def test_random_selection_receipt_rejects_forged_eligible_or_noneligible_selection() -> None:
    random_valid = _scope_design().baseline(ScopeBaselineKind.RANDOM_VALID)
    catalog = random_valid.finite_catalog
    binding = random_valid.random_sampler_binding
    assert catalog is not None and binding is not None
    parent_snapshot = _parent_snapshot(catalog, label="forgery")
    with pytest.raises(ValidationError, match="ref differs from its typed harness manifest"):
        RandomValidParentSnapshot.model_validate(
            {
                **_content(parent_snapshot),
                "parent_harness_ref": _ref("forged-parent-ref"),
            },
            strict=True,
        )
    receipt = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent_snapshot,
        previous_receipt=None,
    )
    content = _content(receipt)
    content["eligible_entry_ids"] = receipt.eligible_entry_ids[1:]
    with pytest.raises(ValidationError, match="eligible set differs from replay"):
        RandomValidSelectionReceipt.model_validate(content, strict=True)

    selected = next(
        entry for entry in catalog.entries if entry.entry_id == receipt.selected_entry_id
    )
    parent = _parent_snapshot(
        catalog,
        label="noneligible",
        overrides={
            (selected.component_kind, selected.target_component_name): selected.replacement_ref
        },
    )
    valid = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent,
        previous_receipt=None,
    )
    content = _content(valid)
    content["selected_entry_id"] = selected.entry_id
    with pytest.raises(ValidationError, match="not the replayed eligible selection"):
        RandomValidSelectionReceipt.model_validate(content, strict=True)


def test_random_selection_receipt_chain_rejects_a_valid_but_wrong_predecessor() -> None:
    random_valid = _scope_design().baseline(ScopeBaselineKind.RANDOM_VALID)
    catalog = random_valid.finite_catalog
    binding = random_valid.random_sampler_binding
    assert catalog is not None and binding is not None
    parent = _parent_snapshot(catalog, label="chain-main")
    receipt_0 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=parent,
        previous_receipt=None,
    )
    alternate_0 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=_parent_snapshot(catalog, label="chain-alternate"),
        previous_receipt=None,
    )
    alternate_1 = make_random_valid_selection_receipt(
        catalog=catalog,
        sampler_binding=binding,
        parent_snapshot=_parent_snapshot(catalog, label="chain-alternate"),
        previous_receipt=alternate_0,
    )
    with pytest.raises(ValueError, match="predecessor chain is broken"):
        verify_random_valid_selection_chain((receipt_0, alternate_1))


def test_prompt_projection_enforcement_rejects_nonprompt_and_forged_receipts() -> None:
    design = _scope_design()
    prompt = design.baseline(ScopeBaselineKind.PROMPT_ONLY)
    projection = prompt.prompt_policy_projection
    catalog = design.baseline(ScopeBaselineKind.RANDOM_VALID).finite_catalog
    assert projection is not None and catalog is not None
    action = _candidate_mutation(ComponentKind.CONTROL_FLOW, "candidate-control-flow-action")
    parent = _parent_snapshot(
        catalog,
        label="prompt-enforcement-nonprompt",
        overrides={(action.before.kind, action.before.name): action.before.artifact},
    )
    receipt = PromptOnlyEnforcementReceipt(
        projection=projection,
        projection_ref=projection.artifact_ref,
        enforcement_implementation_ref=projection.enforcement_implementation_ref,
        action_id="candidate-control-flow-action",
        action=action,
        action_ref=_candidate_mutation_ref(action),
        parent_snapshot=parent,
        parent_snapshot_ref=parent.artifact_ref,
        accepted=False,
        decision="rejected-non-prompt-surface",
    )
    assert receipt.accepted is False
    assert receipt.runtime_execution_attested is False

    with pytest.raises(ValidationError, match="decision differs from projected policy"):
        PromptOnlyEnforcementReceipt.model_validate(
            {**_content(receipt), "accepted": True},
            strict=True,
        )
    with pytest.raises(ValidationError, match="implementation differs from projection"):
        PromptOnlyEnforcementReceipt.model_validate(
            {
                **_content(receipt),
                "enforcement_implementation_ref": _ref("swapped-enforcer", "text/x-python"),
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="action ref differs from typed mutation"):
        PromptOnlyEnforcementReceipt.model_validate(
            {**_content(receipt), "action_ref": _ref("forged-action-ref")},
            strict=True,
        )
    with pytest.raises(ValidationError, match="action_component_kind"):
        PromptOnlyEnforcementReceipt.model_validate(
            {**_content(receipt), "action_component_kind": ComponentKind.PROMPT},
            strict=True,
        )


def test_prompt_enforcement_checks_parent_coordinate_media_and_size() -> None:
    design = _scope_design()
    prompt = design.baseline(ScopeBaselineKind.PROMPT_ONLY)
    catalog = design.baseline(ScopeBaselineKind.RANDOM_VALID).finite_catalog
    projection = prompt.prompt_policy_projection
    assert projection is not None and catalog is not None
    action = _candidate_mutation(ComponentKind.PROMPT, "valid-prompt-action")
    parent = _parent_snapshot(
        catalog,
        label="prompt-enforcement-valid",
        overrides={(action.before.kind, action.before.name): action.before.artifact},
    )
    accepted = PromptOnlyEnforcementReceipt(
        projection=projection,
        projection_ref=projection.artifact_ref,
        enforcement_implementation_ref=projection.enforcement_implementation_ref,
        action_id="valid-prompt-action",
        action=action,
        action_ref=_candidate_mutation_ref(action),
        parent_snapshot=parent,
        parent_snapshot_ref=parent.artifact_ref,
        accepted=True,
        decision="accepted-projected-prompt-action",
    )
    assert accepted.accepted is True

    oversized_after = action.after.model_copy(
        update={
            "artifact": action.after.artifact.model_copy(
                update={"size": projection.projected_policy.max_artifact_size_bytes + 1}
            )
        }
    )
    oversized = action.model_copy(update={"after": oversized_after})
    rejected = PromptOnlyEnforcementReceipt(
        projection=projection,
        projection_ref=projection.artifact_ref,
        enforcement_implementation_ref=projection.enforcement_implementation_ref,
        action_id="oversized-prompt-action",
        action=oversized,
        action_ref=_candidate_mutation_ref(oversized),
        parent_snapshot=parent,
        parent_snapshot_ref=parent.artifact_ref,
        accepted=False,
        decision="rejected-invalid-prompt-action",
    )
    assert rejected.accepted is False
    with pytest.raises(ValidationError, match="decision differs from projected policy"):
        PromptOnlyEnforcementReceipt.model_validate(
            {
                **_content(rejected),
                "accepted": True,
                "decision": "accepted-projected-prompt-action",
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "flag",
    ("execution_attested", "provider_identity_attested", "sealed_evidence", "reportable_result"),
)
def test_scope_design_cannot_publish_an_upgraded_claim(flag: str) -> None:
    content = _content(_scope_design())
    content[flag] = True
    forged = ConfirmatoryScopeBaselineDesign.model_construct(**content)
    with pytest.raises((ValidationError, ValueError)):
        forged.model_dump(mode="json")
