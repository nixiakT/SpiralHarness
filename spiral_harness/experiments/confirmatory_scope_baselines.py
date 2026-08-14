"""Prospective Random-valid and Prompt-only scope baselines.

The primary real-task design deliberately keeps PURE, STATIC, SCORE, and FULL
separate from these secondary scope baselines.  This module closes the two
additional conditions promised by the paper without pretending that a design
artifact proves runtime execution or provider identity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef, ComponentKind, Sha256
from spiral_harness.experiments.baseline_profiles import (
    V2ActionCapability,
    action_capability_profile,
)
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.confirmatory_arms import (
    ConfirmatoryFourArmDesign,
    OptimizerFeedbackMode,
    PromotionRule,
    RealTaskArm,
)
from spiral_harness.experiments.confirmatory_resources import (
    AdaptiveConditionContext,
    AdaptiveExecutionCeilings,
    AdaptiveProtocolCommitments,
    ExAnteAdaptiveTopology,
    FrozenMutationPolicyArtifact,
    ModelMediatedRole,
    ProspectiveConfirmatoryModel,
)
from spiral_harness.experiments.confirmatory_scope_contracts import (
    PromptOnlyPolicyProjection,
    RandomValidCatalog,
    RandomValidSamplerBinding,
    RandomValidSamplerConfig,
    make_prompt_only_policy_projection,
)
from spiral_harness.experiments.confirmatory_scope_contracts import (
    RandomValidCatalogEntry as RandomValidCatalogEntry,
)

CONFIRMATORY_SCOPE_BASELINE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-scope-baselines.v1+json"
)


class ScopeBaselineKind(StrEnum):
    """The two secondary scope baselines absent from the primary four arms."""

    RANDOM_VALID = "random-valid"
    PROMPT_ONLY = "prompt-only"


class CandidateSource(StrEnum):
    """How a scope baseline obtains its next syntactically valid candidate."""

    UNIFORM_PARENT_CONDITIONAL_CATALOG = "uniform-parent-conditional-catalog"
    SAME_MODEL_PROMPT_OPTIMIZER = "same-model-prompt-only-optimizer"


_BASELINE_ORDER = (ScopeBaselineKind.RANDOM_VALID, ScopeBaselineKind.PROMPT_ONLY)
_SCOPE_CONTEXTS = (
    AdaptiveConditionContext.RANDOM_VALID,
    AdaptiveConditionContext.PROMPT_ONLY,
)
_PRIMARY_ADAPTIVE_CONTEXTS = (
    AdaptiveConditionContext.SCORE,
    AdaptiveConditionContext.FULL,
)
_COMBINED_ADAPTIVE_CONTEXTS = (*_SCOPE_CONTEXTS, *_PRIMARY_ADAPTIVE_CONTEXTS)
_RANDOM_PERMITTED_MODEL_ROLES = (
    ModelMediatedRole.SOLVER,
    ModelMediatedRole.JUDGE,
    ModelMediatedRole.GRADER,
)
_ALL_MODEL_ROLES = tuple(ModelMediatedRole)
_RANDOM_COMMITMENT_DIFFERENCES = frozenset(
    {"optimizer_config_fingerprint", "candidate_parser_fingerprint"}
)


class ScopeBaselinePlan(ProspectiveConfirmatoryModel):
    """One secondary condition tied to the primary design's execution budget."""

    schema_version: Literal["2"] = "2"
    kind: ScopeBaselineKind
    topology: ExAnteAdaptiveTopology
    ceilings: AdaptiveExecutionCeilings
    candidate_source: CandidateSource
    optimizer_feedback: OptimizerFeedbackMode
    promotion_rule: PromotionRule
    mutation_policy_artifact: FrozenMutationPolicyArtifact
    mutation_policy_ref: ArtifactRef
    action_capability: V2ActionCapability
    finite_catalog: RandomValidCatalog | None = None
    finite_catalog_ref: ArtifactRef | None = None
    random_sampler_binding: RandomValidSamplerBinding | None = None
    prompt_policy_projection: PromptOnlyPolicyProjection | None = None
    prompt_policy_projection_ref: ArtifactRef | None = None
    candidate_schedule_fingerprint: Sha256
    optimizer_model_calls_permitted: bool
    permitted_model_roles: Annotated[tuple[ModelMediatedRole, ...], Field(min_length=1)]
    champion_state_transition_visible: Literal[True] = True
    assigned_resource_ceiling_equal_full: Literal[True] = True
    realized_usage_equality_claimed: Literal[False] = False
    role_permissions_runtime_attested: Literal[False] = False
    same_nominal_model_all_model_roles_required: Literal[True] = True
    same_exact_served_revision_attested: Literal[False] = False
    random_parent_eligible_selection_receipts_required: bool
    prompt_surface_enforcement_receipts_required: bool
    scope_enforcement_runtime_attested: Literal[False] = False
    execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _exact_condition_profile(self) -> Self:
        if self.topology.condition_context_ids != _SCOPE_CONTEXTS:
            raise ValueError("scope baselines require exact Random-valid and Prompt-only contexts")
        if self.mutation_policy_ref != self.mutation_policy_artifact.artifact_ref:
            raise ValueError("scope baseline mutation-policy ref differs from canonical artifact")
        if self.topology.protocol_commitments.mutation_policy_ref != self.mutation_policy_ref:
            raise ValueError("scope baseline topology binds a different mutation policy")
        if self.topology.protocol_commitments.seed_harness_ref != (
            self.mutation_policy_artifact.seed_harness_ref
        ):
            raise ValueError("scope baseline policy belongs to a different seed harness")
        baseline_kind = (
            BaselineKind.RANDOM_VALID
            if self.kind is ScopeBaselineKind.RANDOM_VALID
            else BaselineKind.PROMPT_ONLY
        )
        expected_capability = action_capability_profile(
            baseline_kind,
            self.mutation_policy_artifact.policy,
        )
        if self.action_capability != expected_capability:
            raise ValueError(f"{self.kind.value} action capability differs from shared policy")
        if self.kind is ScopeBaselineKind.RANDOM_VALID:
            expected = (
                CandidateSource.UNIFORM_PARENT_CONDITIONAL_CATALOG,
                OptimizerFeedbackMode.NONE,
                PromotionRule.MECHANISM_GATED,
                False,
            )
            if self.finite_catalog is None or self.finite_catalog_ref is None:
                raise ValueError("random-valid requires one frozen finite catalog")
            if self.finite_catalog_ref != self.finite_catalog.artifact_ref:
                raise ValueError("random-valid catalog ref differs from canonical catalog")
            if self.finite_catalog.mutation_policy_ref != self.mutation_policy_ref:
                raise ValueError("random-valid catalog binds a different mutation policy")
            if self.finite_catalog.seed_harness_ref != (
                self.topology.protocol_commitments.seed_harness_ref
            ):
                raise ValueError("random-valid catalog binds a different seed harness")
            if self.random_sampler_binding is None:
                raise ValueError("random-valid requires a canonical sampler binding")
            if self.random_sampler_binding.catalog_ref != self.finite_catalog_ref:
                raise ValueError("random-valid sampler binds a different catalog")
            if self.topology.protocol_commitments.optimizer_config_fingerprint != (
                self.random_sampler_binding.fingerprint
            ):
                raise ValueError("random-valid topology does not bind its exact sampler")
            if self.random_sampler_binding.sampling_algorithm != (
                self.finite_catalog.sampling_algorithm
            ):
                raise ValueError("random-valid sampler algorithm differs from catalog")
            if self.random_sampler_binding.sampling_implementation_ref != (
                self.finite_catalog.sampling_implementation_ref
            ):
                raise ValueError("random-valid sampler implementation differs from catalog")
            if (
                self.random_sampler_binding.sampler_config.max_proposals_per_round
                != self.ceilings.max_proposals_per_round
            ):
                raise ValueError("random-valid sampler round width differs from its ceilings")
            if (
                self.prompt_policy_projection is not None
                or self.prompt_policy_projection_ref is not None
            ):
                raise ValueError("random-valid must not claim a prompt-only policy projection")
            if self.permitted_model_roles != _RANDOM_PERMITTED_MODEL_ROLES:
                raise ValueError("random-valid must forbid every optimizer-model role")
            receipt_requirements = (True, False)
        else:
            expected = (
                CandidateSource.SAME_MODEL_PROMPT_OPTIMIZER,
                OptimizerFeedbackMode.MECHANISM_VISIBLE,
                PromotionRule.MECHANISM_GATED,
                True,
            )
            if self.finite_catalog is not None or self.finite_catalog_ref is not None:
                raise ValueError("prompt-only must not claim a random mutation catalog")
            if self.random_sampler_binding is not None:
                raise ValueError("prompt-only must not claim a random sampler binding")
            projection = self.prompt_policy_projection
            if projection is None or self.prompt_policy_projection_ref is None:
                raise ValueError("prompt-only requires an executable canonical policy projection")
            if self.prompt_policy_projection_ref != projection.artifact_ref:
                raise ValueError("prompt-only projection ref differs from canonical projection")
            if projection.full_policy_artifact != self.mutation_policy_artifact:
                raise ValueError("prompt-only projection binds a different FULL policy")
            if self.permitted_model_roles != _ALL_MODEL_ROLES:
                raise ValueError("prompt-only must retain FULL's complete model-role permissions")
            receipt_requirements = (False, True)
        expected_schedule = _derived_candidate_schedule_fingerprint(
            self.topology.protocol_commitments,
            self.kind,
            random_sampler_binding=self.random_sampler_binding,
            prompt_policy_projection=self.prompt_policy_projection,
        )
        if self.candidate_schedule_fingerprint != expected_schedule:
            raise ValueError("candidate schedule is not derived from exact scope artifacts")
        if (
            self.random_parent_eligible_selection_receipts_required,
            self.prompt_surface_enforcement_receipts_required,
        ) != receipt_requirements:
            raise ValueError(f"{self.kind.value} receipt requirements differ from its profile")
        actual = (
            self.candidate_source,
            self.optimizer_feedback,
            self.promotion_rule,
            self.optimizer_model_calls_permitted,
        )
        if actual != expected:
            raise ValueError(f"{self.kind.value} differs from its frozen scope profile")
        return self


class ConfirmatoryScopeBaselineDesign(ProspectiveConfirmatoryModel):
    """Random-valid and Prompt-only joined to one primary four-arm design."""

    schema_version: Literal["1"] = "1"
    protocol_version: Literal["confirmatory-scope-baselines-v1"] = "confirmatory-scope-baselines-v1"
    four_arm_design: ConfirmatoryFourArmDesign
    four_arm_design_fingerprint: Sha256
    baselines: Annotated[tuple[ScopeBaselinePlan, ...], Field(min_length=2, max_length=2)]
    full_mutation_policy_artifact: FrozenMutationPolicyArtifact
    full_mutation_policy_ref: ArtifactRef
    combined_adaptive_context_ids: tuple[
        Literal[AdaptiveConditionContext.RANDOM_VALID],
        Literal[AdaptiveConditionContext.PROMPT_ONLY],
        Literal[AdaptiveConditionContext.SCORE],
        Literal[AdaptiveConditionContext.FULL],
    ] = _COMBINED_ADAPTIVE_CONTEXTS
    contrast_roles: tuple[
        Literal["guided-search-vs-uniform-valid-search"],
        Literal["full-surface-vs-prompt-only-search"],
    ] = (
        "guided-search-vs-uniform-valid-search",
        "full-surface-vs-prompt-only-search",
    )
    secondary_scope_baselines: Literal[True] = True
    primary_policy_contrast_unchanged: Literal[True] = True
    structural_scope_profile_matched: Literal[True] = True
    scope_effect_identified: Literal[False] = False
    combined_context_isolation_attested: Literal[False] = False
    catalog_coverage_attested: Literal[False] = False
    execution_attested: Literal[False] = False
    provider_identity_attested: Literal[False] = False
    sealed_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @field_validator("baselines")
    @classmethod
    def _canonicalize_baselines(
        cls,
        values: tuple[ScopeBaselinePlan, ...],
    ) -> tuple[ScopeBaselinePlan, ...]:
        by_kind = {value.kind: value for value in values}
        if len(by_kind) != len(values) or frozenset(by_kind) != frozenset(_BASELINE_ORDER):
            raise ValueError("scope design requires exactly Random-valid and Prompt-only")
        return tuple(by_kind[kind] for kind in _BASELINE_ORDER)

    @model_validator(mode="after")
    def _bind_primary_design_and_non_treatment_coordinates(self) -> Self:
        if self.four_arm_design_fingerprint != self.four_arm_design.fingerprint:
            raise ValueError("scope baselines bind a different four-arm design")
        full = self.four_arm_design.arm(RealTaskArm.FULL)
        anchor_topology = full.adaptive_topology
        anchor_ceilings = full.adaptive_ceilings
        if anchor_topology is None or anchor_ceilings is None:  # pragma: no cover
            raise ValueError("FULL adaptive resources are unavailable")
        anchor_commitments = anchor_topology.protocol_commitments
        if anchor_topology.condition_context_ids != _PRIMARY_ADAPTIVE_CONTEXTS:
            raise ValueError("four-arm design does not bind exact SCORE and FULL contexts")
        if self.combined_adaptive_context_ids != _COMBINED_ADAPTIVE_CONTEXTS:
            raise ValueError("combined adaptive context roster drifted")
        policy_ref = self.full_mutation_policy_artifact.artifact_ref
        if self.full_mutation_policy_ref != policy_ref:
            raise ValueError("FULL mutation-policy ref differs from canonical artifact")
        if anchor_commitments.mutation_policy_ref != policy_ref:
            raise ValueError("four-arm FULL commitments bind a different mutation policy")
        if (
            anchor_commitments.seed_harness_ref
            != self.full_mutation_policy_artifact.seed_harness_ref
        ):
            raise ValueError("FULL mutation policy belongs to a different seed harness")
        if len(self.full_mutation_policy_artifact.policy.allowed_component_kinds) < 2:
            raise ValueError("prompt-only is undefined when FULL can mutate only prompts")
        by_kind = {baseline.kind: baseline for baseline in self.baselines}
        for baseline in self.baselines:
            if baseline.ceilings != anchor_ceilings:
                raise ValueError("scope baseline ceilings differ from FULL")
            _require_common_commitments(
                anchor_commitments,
                baseline.topology.protocol_commitments,
                allowed_differences=(
                    _RANDOM_COMMITMENT_DIFFERENCES
                    if baseline.kind is ScopeBaselineKind.RANDOM_VALID
                    else frozenset()
                ),
            )
            if baseline.mutation_policy_artifact != self.full_mutation_policy_artifact:
                raise ValueError("scope baseline does not share FULL's exact mutation policy")
            if baseline.mutation_policy_ref != policy_ref:
                raise ValueError("scope baseline binds a different mutation-policy ref")
        random_valid = by_kind[ScopeBaselineKind.RANDOM_VALID]
        if random_valid.action_capability.mutable_component_kinds != (
            self.full_mutation_policy_artifact.policy.allowed_component_kinds
        ):
            raise ValueError("random-valid catalog must cover FULL's mutation surface")
        prompt_only = by_kind[ScopeBaselineKind.PROMPT_ONLY]
        if prompt_only.action_capability.mutable_component_kinds != (ComponentKind.PROMPT,):
            raise ValueError("prompt-only surface drifted")
        prompt_commitments = prompt_only.topology.protocol_commitments
        if prompt_commitments.optimizer_config_fingerprint != (
            anchor_commitments.optimizer_config_fingerprint
        ):
            raise ValueError("prompt-only optimizer differs from FULL")
        if prompt_commitments.candidate_parser_fingerprint != (
            anchor_commitments.candidate_parser_fingerprint
        ):
            raise ValueError("prompt-only parser differs from FULL")
        return self

    def baseline(self, kind: ScopeBaselineKind) -> ScopeBaselinePlan:
        if type(kind) is not ScopeBaselineKind:
            raise TypeError("kind must be an exact ScopeBaselineKind")
        return next(item for item in self.baselines if item.kind is kind)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def make_confirmatory_scope_baseline_design(
    *,
    four_arm_design: ConfirmatoryFourArmDesign,
    full_mutation_policy_artifact: FrozenMutationPolicyArtifact,
    random_candidate_parser_fingerprint: str,
    random_catalog: RandomValidCatalog,
    random_sampler_config: RandomValidSamplerConfig,
    prompt_projection_implementation_ref: ArtifactRef,
    prompt_enforcement_implementation_ref: ArtifactRef,
    prompt_projection_provenance_ref: ArtifactRef,
) -> ConfirmatoryScopeBaselineDesign:
    """Derive both honest scope baselines from FULL's frozen coordinates."""

    design = ConfirmatoryFourArmDesign.model_validate(four_arm_design, strict=True)
    full = design.arm(RealTaskArm.FULL)
    topology = full.adaptive_topology
    ceilings = full.adaptive_ceilings
    if topology is None or ceilings is None:  # pragma: no cover - design invariant
        raise ValueError("FULL adaptive resources are unavailable")
    anchor = topology.protocol_commitments
    policy_artifact = FrozenMutationPolicyArtifact.model_validate(
        full_mutation_policy_artifact,
        strict=True,
    )
    policy = policy_artifact.policy
    policy_ref = policy_artifact.artifact_ref
    if anchor.mutation_policy_ref != policy_ref:
        raise ValueError("four-arm FULL commitments bind a different mutation policy")
    if anchor.seed_harness_ref != policy_artifact.seed_harness_ref:
        raise ValueError("four-arm FULL commitments bind a different seed harness")
    catalog = RandomValidCatalog.model_validate(random_catalog, strict=True)
    if catalog.mutation_policy_artifact != policy_artifact:
        raise ValueError("random-valid catalog does not embed FULL's mutation policy")
    sampler_config = RandomValidSamplerConfig.model_validate(random_sampler_config, strict=True)
    random_sampler_binding = RandomValidSamplerBinding(
        catalog_ref=catalog.artifact_ref,
        sampling_algorithm=catalog.sampling_algorithm,
        sampling_implementation_ref=catalog.sampling_implementation_ref,
        sampler_config=sampler_config,
        sampler_config_ref=sampler_config.artifact_ref,
        seed_derivation_domain=catalog.seed_derivation_domain,
    )
    random_commitments = _replace_optimizer_commitments(
        anchor,
        optimizer_config_fingerprint=random_sampler_binding.fingerprint,
        candidate_parser_fingerprint=random_candidate_parser_fingerprint,
    )
    prompt_projection = make_prompt_only_policy_projection(
        full_policy_artifact=policy_artifact,
        projection_implementation_ref=prompt_projection_implementation_ref,
        enforcement_implementation_ref=prompt_enforcement_implementation_ref,
        construction_provenance_ref=prompt_projection_provenance_ref,
    )
    scope_topology = topology.model_copy(update={"condition_context_ids": _SCOPE_CONTEXTS})
    baselines = (
        ScopeBaselinePlan(
            kind=ScopeBaselineKind.RANDOM_VALID,
            topology=scope_topology.model_copy(update={"protocol_commitments": random_commitments}),
            ceilings=ceilings,
            candidate_source=CandidateSource.UNIFORM_PARENT_CONDITIONAL_CATALOG,
            optimizer_feedback=OptimizerFeedbackMode.NONE,
            promotion_rule=PromotionRule.MECHANISM_GATED,
            mutation_policy_artifact=policy_artifact,
            mutation_policy_ref=policy_ref,
            action_capability=action_capability_profile(BaselineKind.RANDOM_VALID, policy),
            finite_catalog=catalog,
            finite_catalog_ref=catalog.artifact_ref,
            random_sampler_binding=random_sampler_binding,
            candidate_schedule_fingerprint=_derived_candidate_schedule_fingerprint(
                random_commitments,
                ScopeBaselineKind.RANDOM_VALID,
                random_sampler_binding=random_sampler_binding,
            ),
            optimizer_model_calls_permitted=False,
            permitted_model_roles=_RANDOM_PERMITTED_MODEL_ROLES,
            random_parent_eligible_selection_receipts_required=True,
            prompt_surface_enforcement_receipts_required=False,
        ),
        ScopeBaselinePlan(
            kind=ScopeBaselineKind.PROMPT_ONLY,
            topology=scope_topology,
            ceilings=ceilings,
            candidate_source=CandidateSource.SAME_MODEL_PROMPT_OPTIMIZER,
            optimizer_feedback=OptimizerFeedbackMode.MECHANISM_VISIBLE,
            promotion_rule=PromotionRule.MECHANISM_GATED,
            mutation_policy_artifact=policy_artifact,
            mutation_policy_ref=policy_ref,
            action_capability=action_capability_profile(BaselineKind.PROMPT_ONLY, policy),
            prompt_policy_projection=prompt_projection,
            prompt_policy_projection_ref=prompt_projection.artifact_ref,
            candidate_schedule_fingerprint=_derived_candidate_schedule_fingerprint(
                anchor,
                ScopeBaselineKind.PROMPT_ONLY,
                prompt_policy_projection=prompt_projection,
            ),
            optimizer_model_calls_permitted=True,
            permitted_model_roles=_ALL_MODEL_ROLES,
            random_parent_eligible_selection_receipts_required=False,
            prompt_surface_enforcement_receipts_required=True,
        ),
    )
    return ConfirmatoryScopeBaselineDesign(
        four_arm_design=design,
        four_arm_design_fingerprint=design.fingerprint,
        baselines=baselines,
        full_mutation_policy_artifact=policy_artifact,
        full_mutation_policy_ref=policy_ref,
    )


def _derived_candidate_schedule_fingerprint(
    commitments: AdaptiveProtocolCommitments,
    kind: ScopeBaselineKind,
    *,
    random_sampler_binding: RandomValidSamplerBinding | None = None,
    prompt_policy_projection: PromptOnlyPolicyProjection | None = None,
) -> str:
    """Bind shared coordinates and the complete condition-specific execution identity."""

    return canonical_sha256(
        {
            "domain": "spiral-harness/confirmatory-scope-candidate-schedule/v1",
            "kind": kind.value,
            "seed_schedule_fingerprint": commitments.seed_schedule_fingerprint,
            "query_dag_fingerprint": commitments.query_dag_fingerprint,
            "random_sampler_binding_fingerprint": (
                random_sampler_binding.fingerprint if random_sampler_binding is not None else None
            ),
            "prompt_policy_projection_ref": (
                prompt_policy_projection.artifact_ref
                if prompt_policy_projection is not None
                else None
            ),
        }
    )


def _replace_optimizer_commitments(
    anchor: AdaptiveProtocolCommitments,
    *,
    optimizer_config_fingerprint: str,
    candidate_parser_fingerprint: str,
) -> AdaptiveProtocolCommitments:
    content = anchor.model_dump(mode="python", round_trip=True)
    content["optimizer_config_fingerprint"] = optimizer_config_fingerprint
    content["candidate_parser_fingerprint"] = candidate_parser_fingerprint
    return AdaptiveProtocolCommitments.model_validate(content, strict=True)


def _require_common_commitments(
    anchor: AdaptiveProtocolCommitments,
    candidate: AdaptiveProtocolCommitments,
    *,
    allowed_differences: frozenset[str],
) -> None:
    field_names = frozenset(type(anchor).model_fields)
    if not allowed_differences.issubset(field_names):  # pragma: no cover - frozen constants
        raise RuntimeError("scope baseline difference allowlist names an unknown commitment")
    for field_name in sorted(field_names.difference(allowed_differences)):
        if getattr(candidate, field_name) != getattr(anchor, field_name):
            raise ValueError(f"scope baseline drifted on shared {field_name}")


__all__ = [
    "CONFIRMATORY_SCOPE_BASELINE_MEDIA_TYPE",
    "CandidateSource",
    "ConfirmatoryScopeBaselineDesign",
    "ScopeBaselineKind",
    "ScopeBaselinePlan",
    "make_confirmatory_scope_baseline_design",
]
