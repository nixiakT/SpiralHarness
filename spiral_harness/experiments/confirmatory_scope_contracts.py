"""Replayable contracts for Random-valid sampling and Prompt-only enforcement."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessManifest,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.experiments.baselines import FrozenMutationPolicy, MutationOperation
from spiral_harness.experiments.confirmatory_resources import (
    FrozenMutationPolicyArtifact,
    MutationSurfaceGrammarBinding,
    ProspectiveConfirmatoryModel,
)

RANDOM_VALID_CATALOG_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-random-valid-catalog.v2+json"
)
RANDOM_VALID_SAMPLER_CONFIG_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-random-valid-sampler-config.v1+json"
)
RANDOM_VALID_PARENT_SNAPSHOT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-random-valid-parent.v1+json"
)
RANDOM_VALID_SELECTION_RECEIPT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-random-valid-selection-receipt.v1+json"
)
PROMPT_ONLY_POLICY_PROJECTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-prompt-only-policy.v1+json"
)
PROMPT_ONLY_ENFORCEMENT_RECEIPT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.confirmatory-prompt-enforcement-receipt.v1+json"
)

_RANDOM_ALGORITHM = "sha256-counter-rejection-parent-conditional-uniform-v1"
_RANDOM_SEED_DOMAIN = "spiral-harness/confirmatory-random-valid-sample/v2"


def _canonical_ref(value: object, media_type: str) -> ArtifactRef:
    payload = canonical_json_bytes(value)
    return ArtifactRef(
        sha256=canonical_sha256(value),
        size=len(payload),
        media_type=media_type,
    )


class RandomValidCatalogEntry(ProspectiveConfirmatoryModel):
    schema_version: Literal["2"] = "2"
    entry_id: NonEmptyStr
    component_kind: ComponentKind
    target_component_name: NonEmptyStr
    operation: Literal[MutationOperation.REPLACE] = MutationOperation.REPLACE
    replacement_ref: ArtifactRef
    transformation_implementation_ref: ArtifactRef
    hypothesis_ref: ArtifactRef
    parent_binding: Literal["runtime-current-component-ref"] = "runtime-current-component-ref"


class RandomValidCatalog(ProspectiveConfirmatoryModel):
    schema_version: Literal["2"] = "2"
    catalog_id: NonEmptyStr
    seed_harness_ref: ArtifactRef
    mutation_policy_artifact: FrozenMutationPolicyArtifact
    mutation_policy_ref: ArtifactRef
    entries: Annotated[tuple[RandomValidCatalogEntry, ...], Field(min_length=1)]
    sampling_algorithm: Literal["sha256-counter-rejection-parent-conditional-uniform-v1"] = (
        _RANDOM_ALGORITHM
    )
    sampling_implementation_ref: ArtifactRef
    seed_derivation_domain: Literal["spiral-harness/confirmatory-random-valid-sample/v2"] = (
        _RANDOM_SEED_DOMAIN
    )
    eligibility_rule: Literal[
        "policy-valid-template-whose-replacement-differs-from-current-parent-v1"
    ] = "policy-valid-template-whose-replacement-differs-from-current-parent-v1"
    construction_provenance_ref: ArtifactRef
    construction_usage_ref: ArtifactRef
    construction_cost_rule: Literal["reported-separately-and-amortized"] = (
        "reported-separately-and-amortized"
    )
    outcome_bearing_data_used_for_construction: Literal[False] = False
    selection_execution_attested: Literal[False] = False

    @field_validator("entries")
    @classmethod
    def _canonicalize_entries(
        cls,
        values: tuple[RandomValidCatalogEntry, ...],
    ) -> tuple[RandomValidCatalogEntry, ...]:
        ordered = tuple(sorted(values, key=lambda item: item.entry_id))
        ids = tuple(item.entry_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("random-valid catalog entry IDs must be unique")
        coordinates = tuple(
            (
                item.component_kind,
                item.target_component_name,
                item.operation,
                item.replacement_ref.sha256,
                item.replacement_ref.size,
                item.replacement_ref.media_type,
            )
            for item in ordered
        )
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("duplicate catalog templates would bias uniform sampling")
        return ordered

    @model_validator(mode="after")
    def _bind_policy_and_validate_templates(self) -> Self:
        if self.mutation_policy_ref != self.mutation_policy_artifact.artifact_ref:
            raise ValueError("catalog mutation-policy ref differs from canonical artifact")
        if self.seed_harness_ref != self.mutation_policy_artifact.seed_harness_ref:
            raise ValueError("catalog belongs to a different seed harness")
        policy = self.mutation_policy_artifact.policy
        surfaces = {
            item.component_kind: item for item in self.mutation_policy_artifact.surface_grammars
        }
        covered: set[ComponentKind] = set()
        for entry in self.entries:
            try:
                surface = surfaces[entry.component_kind]
            except KeyError as exc:
                raise ValueError("catalog template targets a surface outside FULL policy") from exc
            component_names = {item.component_name for item in surface.seed_components}
            if entry.target_component_name not in component_names:
                raise ValueError("catalog template targets an unknown component coordinate")
            replacement_media = entry.replacement_ref.media_type.partition(";")[0].lower()
            if replacement_media not in surface.allowed_media_types:
                raise ValueError("catalog replacement media type violates FULL policy")
            if entry.replacement_ref.size > policy.max_artifact_size_bytes:
                raise ValueError("catalog replacement exceeds FULL artifact-size limit")
            covered.add(entry.component_kind)
        if covered != set(policy.allowed_component_kinds):
            raise ValueError("random-valid catalog does not cover every FULL policy surface")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=RANDOM_VALID_CATALOG_MEDIA_TYPE,
        )


class RandomValidSamplerConfig(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    seed_fingerprint: Sha256
    counter_start: Literal[0] = 0
    max_proposals_per_round: Annotated[int, Field(gt=0, strict=True)]
    without_replacement_across_complete_chain: Literal[True] = True
    selection_algorithm: Literal["sha256-counter-rejection-parent-conditional-uniform-v1"] = (
        _RANDOM_ALGORITHM
    )
    seed_derivation_domain: Literal["spiral-harness/confirmatory-random-valid-sample/v2"] = (
        _RANDOM_SEED_DOMAIN
    )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=RANDOM_VALID_SAMPLER_CONFIG_MEDIA_TYPE,
        )


class RandomValidSamplerBinding(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    catalog_ref: ArtifactRef
    sampling_algorithm: Literal["sha256-counter-rejection-parent-conditional-uniform-v1"] = (
        _RANDOM_ALGORITHM
    )
    sampling_implementation_ref: ArtifactRef
    sampler_config: RandomValidSamplerConfig
    sampler_config_ref: ArtifactRef
    seed_derivation_domain: Literal["spiral-harness/confirmatory-random-valid-sample/v2"] = (
        _RANDOM_SEED_DOMAIN
    )

    @model_validator(mode="after")
    def _bind_config(self) -> Self:
        if self.sampler_config_ref != self.sampler_config.artifact_ref:
            raise ValueError("random sampler config ref differs from canonical config")
        if self.sampling_algorithm != self.sampler_config.selection_algorithm:
            raise ValueError("random sampler algorithm differs from its config")
        if self.seed_derivation_domain != self.sampler_config.seed_derivation_domain:
            raise ValueError("random sampler seed domain differs from its config")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class RandomValidParentSnapshot(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    parent_manifest: HarnessManifest
    parent_harness_ref: ArtifactRef

    @model_validator(mode="after")
    def _bind_typed_manifest(self) -> Self:
        expected = _canonical_ref(self.parent_manifest, HARNESS_MANIFEST_MEDIA_TYPE)
        if self.parent_harness_ref != expected:
            raise ValueError("random-valid parent ref differs from its typed harness manifest")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=RANDOM_VALID_PARENT_SNAPSHOT_MEDIA_TYPE,
        )


class RandomValidSelectionReceipt(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    selection_index: Annotated[int, Field(ge=0, strict=True)]
    round_index: Annotated[int, Field(ge=0, strict=True)]
    proposal_index: Annotated[int, Field(ge=0, strict=True)]
    catalog: RandomValidCatalog
    catalog_ref: ArtifactRef
    sampler_binding: RandomValidSamplerBinding
    parent_snapshot: RandomValidParentSnapshot
    parent_snapshot_ref: ArtifactRef
    eligible_entry_ids: tuple[NonEmptyStr, ...]
    prior_selected_entry_ids_in_chain: tuple[NonEmptyStr, ...]
    selection_pool_entry_ids: tuple[NonEmptyStr, ...]
    selected_entry_id: NonEmptyStr | None
    rejection_counter: Annotated[int, Field(ge=0, strict=True)] | None
    previous_receipt_ref: ArtifactRef | None
    replay_verified: Literal[True] = True
    runtime_execution_attested: Literal[False] = False

    @field_validator(
        "eligible_entry_ids",
        "prior_selected_entry_ids_in_chain",
        "selection_pool_entry_ids",
    )
    @classmethod
    def _canonicalize_eligible_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        ordered = tuple(sorted(values))
        if len(ordered) != len(set(ordered)):
            raise ValueError("eligible entry IDs must not repeat")
        return ordered

    @model_validator(mode="after")
    def _replay_selection(self) -> Self:
        if self.catalog_ref != self.catalog.artifact_ref:
            raise ValueError("selection receipt catalog ref differs from canonical catalog")
        if self.sampler_binding.catalog_ref != self.catalog_ref:
            raise ValueError("selection receipt sampler binds a different catalog")
        if self.sampler_binding.sampling_algorithm != self.catalog.sampling_algorithm:
            raise ValueError("selection receipt sampler algorithm differs from catalog")
        if (
            self.sampler_binding.sampling_implementation_ref
            != self.catalog.sampling_implementation_ref
        ):
            raise ValueError("selection receipt sampler implementation differs from catalog")
        if self.parent_snapshot_ref != self.parent_snapshot.artifact_ref:
            raise ValueError("selection receipt parent ref differs from canonical snapshot")
        proposals_per_round = self.sampler_binding.sampler_config.max_proposals_per_round
        expected_round, expected_proposal = divmod(self.selection_index, proposals_per_round)
        if (self.round_index, self.proposal_index) != (expected_round, expected_proposal):
            raise ValueError("selection receipt round/proposal coordinates are not canonical")
        if (self.selection_index == 0) != (self.previous_receipt_ref is None):
            raise ValueError("only the genesis selection receipt may omit its predecessor")

        parent = {
            (item.kind, item.name): item.artifact
            for item in self.parent_snapshot.parent_manifest.components
        }
        targets = {
            (entry.component_kind, entry.target_component_name) for entry in self.catalog.entries
        }
        if not targets.issubset(parent):
            raise ValueError("typed parent manifest is missing a catalog target coordinate")
        expected_eligible = tuple(
            sorted(
                entry.entry_id
                for entry in self.catalog.entries
                if parent[(entry.component_kind, entry.target_component_name)].sha256
                != entry.replacement_ref.sha256
            )
        )
        if self.eligible_entry_ids != expected_eligible:
            raise ValueError("eligible set differs from replay against the current parent")
        expected_pool = tuple(
            item for item in expected_eligible if item not in self.prior_selected_entry_ids_in_chain
        )
        if self.selection_pool_entry_ids != expected_pool:
            raise ValueError("selection pool does not remove prior global-chain selections")
        expected_selection, expected_counter = _selected_entry_id(
            binding=self.sampler_binding,
            parent_ref=self.parent_snapshot_ref,
            selection_pool_entry_ids=expected_pool,
            selection_index=self.selection_index,
            round_index=self.round_index,
            proposal_index=self.proposal_index,
            previous_receipt_ref=self.previous_receipt_ref,
        )
        if (
            self.selected_entry_id != expected_selection
            or self.rejection_counter != expected_counter
        ):
            raise ValueError("selected entry is not the replayed eligible selection")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=RANDOM_VALID_SELECTION_RECEIPT_MEDIA_TYPE,
        )


def _selected_entry_id(
    *,
    binding: RandomValidSamplerBinding,
    parent_ref: ArtifactRef,
    selection_pool_entry_ids: tuple[str, ...],
    selection_index: int,
    round_index: int,
    proposal_index: int,
    previous_receipt_ref: ArtifactRef | None,
) -> tuple[str | None, int | None]:
    if not selection_pool_entry_ids:
        return None, None
    population_size = len(selection_pool_entry_ids)
    hash_domain_size = 1 << 256
    acceptance_limit = hash_domain_size - (hash_domain_size % population_size)
    counter = binding.sampler_config.counter_start
    while True:
        digest = canonical_sha256(
            {
                "domain": binding.seed_derivation_domain,
                "seed_fingerprint": binding.sampler_config.seed_fingerprint,
                "binding_fingerprint": binding.fingerprint,
                "parent_snapshot_ref": parent_ref,
                "selection_pool_entry_ids": selection_pool_entry_ids,
                "selection_index": selection_index,
                "round_index": round_index,
                "proposal_index": proposal_index,
                "previous_receipt_sha256": (
                    previous_receipt_ref.sha256 if previous_receipt_ref is not None else "genesis"
                ),
                "rejection_counter": counter,
            }
        )
        draw = int(digest, 16)
        if draw < acceptance_limit:
            return selection_pool_entry_ids[draw % population_size], counter
        counter += 1


def make_random_valid_selection_receipt(
    *,
    catalog: RandomValidCatalog,
    sampler_binding: RandomValidSamplerBinding,
    parent_snapshot: RandomValidParentSnapshot,
    previous_receipt: RandomValidSelectionReceipt | None,
) -> RandomValidSelectionReceipt:
    checked_catalog = RandomValidCatalog.model_validate(catalog, strict=True)
    checked_binding = RandomValidSamplerBinding.model_validate(sampler_binding, strict=True)
    checked_parent = RandomValidParentSnapshot.model_validate(parent_snapshot, strict=True)
    checked_previous = (
        RandomValidSelectionReceipt.model_validate(previous_receipt, strict=True)
        if previous_receipt is not None
        else None
    )
    previous_ref = checked_previous.artifact_ref if checked_previous is not None else None
    selection_index = checked_previous.selection_index + 1 if checked_previous is not None else 0
    round_index, proposal_index = divmod(
        selection_index,
        checked_binding.sampler_config.max_proposals_per_round,
    )
    if (
        checked_previous is not None
        and checked_previous.round_index == round_index
        and checked_previous.parent_snapshot_ref != checked_parent.artifact_ref
    ):
        raise ValueError("random-valid parent must remain frozen within one proposal round")
    parent = {
        (item.kind, item.name): item.artifact for item in checked_parent.parent_manifest.components
    }
    targets = {
        (entry.component_kind, entry.target_component_name) for entry in checked_catalog.entries
    }
    if not targets.issubset(parent):
        raise ValueError("typed parent manifest is missing a catalog target coordinate")
    eligible = tuple(
        sorted(
            entry.entry_id
            for entry in checked_catalog.entries
            if parent[(entry.component_kind, entry.target_component_name)].sha256
            != entry.replacement_ref.sha256
        )
    )
    prior_selected = (
        tuple(
            sorted(
                (
                    *checked_previous.prior_selected_entry_ids_in_chain,
                    *(
                        (checked_previous.selected_entry_id,)
                        if checked_previous.selected_entry_id is not None
                        else ()
                    ),
                )
            )
        )
        if checked_previous is not None
        else ()
    )
    selection_pool = tuple(item for item in eligible if item not in prior_selected)
    selected, rejection_counter = _selected_entry_id(
        binding=checked_binding,
        parent_ref=checked_parent.artifact_ref,
        selection_pool_entry_ids=selection_pool,
        selection_index=selection_index,
        round_index=round_index,
        proposal_index=proposal_index,
        previous_receipt_ref=previous_ref,
    )
    return RandomValidSelectionReceipt(
        selection_index=selection_index,
        round_index=round_index,
        proposal_index=proposal_index,
        catalog=checked_catalog,
        catalog_ref=checked_catalog.artifact_ref,
        sampler_binding=checked_binding,
        parent_snapshot=checked_parent,
        parent_snapshot_ref=checked_parent.artifact_ref,
        eligible_entry_ids=eligible,
        prior_selected_entry_ids_in_chain=prior_selected,
        selection_pool_entry_ids=selection_pool,
        selected_entry_id=selected,
        rejection_counter=rejection_counter,
        previous_receipt_ref=previous_ref,
    )


def verify_random_valid_selection_chain(
    receipts: tuple[RandomValidSelectionReceipt, ...],
) -> tuple[RandomValidSelectionReceipt, ...]:
    if not receipts:
        raise ValueError("random-valid selection receipt chain must contain its genesis")
    checked = tuple(
        RandomValidSelectionReceipt.model_validate(receipt, strict=True) for receipt in receipts
    )
    for index, receipt in enumerate(checked):
        if receipt.selection_index != index:
            raise ValueError("random-valid selection receipt index sequence is broken")
        expected_previous = checked[index - 1].artifact_ref if index else None
        if receipt.previous_receipt_ref != expected_previous:
            raise ValueError("random-valid selection receipt predecessor chain is broken")
        if index and receipt.sampler_binding != checked[0].sampler_binding:
            raise ValueError("random-valid sampler binding changed inside the receipt chain")
        if index and receipt.catalog_ref != checked[0].catalog_ref:
            raise ValueError("random-valid catalog changed inside the receipt chain")
        previous = checked[index - 1] if index else None
        if (
            previous is not None
            and previous.round_index == receipt.round_index
            and previous.parent_snapshot_ref != receipt.parent_snapshot_ref
        ):
            raise ValueError("random-valid parent changed within one proposal round")
        expected_prior = (
            tuple(
                sorted(
                    (
                        *previous.prior_selected_entry_ids_in_chain,
                        *(
                            (previous.selected_entry_id,)
                            if previous.selected_entry_id is not None
                            else ()
                        ),
                    )
                )
            )
            if previous is not None
            else ()
        )
        if receipt.prior_selected_entry_ids_in_chain != expected_prior:
            raise ValueError("random-valid global without-replacement chain is broken")
    return checked


class PromptOnlyPolicyProjection(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    full_policy_artifact: FrozenMutationPolicyArtifact
    full_policy_ref: ArtifactRef
    projected_policy: FrozenMutationPolicy
    projected_prompt_surface: MutationSurfaceGrammarBinding
    projection_implementation_ref: ArtifactRef
    enforcement_implementation_ref: ArtifactRef
    construction_provenance_ref: ArtifactRef
    outcome_bearing_data_used_for_construction: Literal[False] = False
    execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _validate_exact_projection(self) -> Self:
        if self.full_policy_ref != self.full_policy_artifact.artifact_ref:
            raise ValueError("prompt projection FULL-policy ref differs from canonical artifact")
        try:
            prompt_surface = next(
                item
                for item in self.full_policy_artifact.surface_grammars
                if item.component_kind is ComponentKind.PROMPT
            )
        except StopIteration as exc:
            raise ValueError("FULL mutation policy has no prompt surface to project") from exc
        expected_policy = FrozenMutationPolicy(
            grammar_version=self.full_policy_artifact.policy.grammar_version,
            allowed_component_kinds=(ComponentKind.PROMPT,),
            allowed_operations=self.full_policy_artifact.policy.allowed_operations,
            max_artifact_size_bytes=self.full_policy_artifact.policy.max_artifact_size_bytes,
        )
        if self.projected_policy != expected_policy:
            raise ValueError("prompt-only policy is not the canonical FULL-policy projection")
        if self.projected_prompt_surface != prompt_surface:
            raise ValueError("prompt-only projection changed the FULL prompt execution surface")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=PROMPT_ONLY_POLICY_PROJECTION_MEDIA_TYPE,
        )


def make_prompt_only_policy_projection(
    *,
    full_policy_artifact: FrozenMutationPolicyArtifact,
    projection_implementation_ref: ArtifactRef,
    enforcement_implementation_ref: ArtifactRef,
    construction_provenance_ref: ArtifactRef,
) -> PromptOnlyPolicyProjection:
    full = FrozenMutationPolicyArtifact.model_validate(full_policy_artifact, strict=True)
    prompt_surface = next(
        (item for item in full.surface_grammars if item.component_kind is ComponentKind.PROMPT),
        None,
    )
    if prompt_surface is None:
        raise ValueError("FULL mutation policy has no prompt surface to project")
    return PromptOnlyPolicyProjection(
        full_policy_artifact=full,
        full_policy_ref=full.artifact_ref,
        projected_policy=FrozenMutationPolicy(
            grammar_version=full.policy.grammar_version,
            allowed_component_kinds=(ComponentKind.PROMPT,),
            allowed_operations=full.policy.allowed_operations,
            max_artifact_size_bytes=full.policy.max_artifact_size_bytes,
        ),
        projected_prompt_surface=prompt_surface,
        projection_implementation_ref=projection_implementation_ref,
        enforcement_implementation_ref=enforcement_implementation_ref,
        construction_provenance_ref=construction_provenance_ref,
    )


class PromptOnlyEnforcementReceipt(ProspectiveConfirmatoryModel):
    schema_version: Literal["1"] = "1"
    projection: PromptOnlyPolicyProjection
    projection_ref: ArtifactRef
    enforcement_implementation_ref: ArtifactRef
    action_id: NonEmptyStr
    action: CandidateMutation
    action_ref: ArtifactRef
    parent_snapshot: RandomValidParentSnapshot
    parent_snapshot_ref: ArtifactRef
    accepted: bool
    decision: Literal[
        "accepted-projected-prompt-action",
        "rejected-non-prompt-surface",
        "rejected-invalid-prompt-action",
    ]
    replay_verified: Literal[True] = True
    runtime_execution_attested: Literal[False] = False

    @model_validator(mode="after")
    def _replay_enforcement(self) -> Self:
        if self.projection_ref != self.projection.artifact_ref:
            raise ValueError("prompt enforcement receipt binds a forged projection ref")
        if self.enforcement_implementation_ref != self.projection.enforcement_implementation_ref:
            raise ValueError("prompt enforcement implementation differs from projection")
        expected_action_ref = _canonical_ref(self.action, CANDIDATE_MUTATION_MEDIA_TYPE)
        if self.action_ref != expected_action_ref:
            raise ValueError("prompt enforcement action ref differs from typed mutation")
        if self.parent_snapshot_ref != self.parent_snapshot.artifact_ref:
            raise ValueError("prompt enforcement parent ref differs from typed manifest")
        parent_components = {
            (item.kind, item.name): item for item in self.parent_snapshot.parent_manifest.components
        }
        parent_component = parent_components.get((self.action.before.kind, self.action.before.name))
        before_matches_parent = parent_component == self.action.before
        prompt_action = self.action.before.kind is ComponentKind.PROMPT
        prompt_names = {
            item.component_name for item in self.projection.projected_prompt_surface.seed_components
        }
        after_media_type = self.action.after.artifact.media_type.partition(";")[0].lower()
        valid_prompt_action = (
            prompt_action
            and self.action.target_component in prompt_names
            and before_matches_parent
            and after_media_type in self.projection.projected_prompt_surface.allowed_media_types
            and self.action.after.artifact.size
            <= self.projection.projected_policy.max_artifact_size_bytes
            and MutationOperation.REPLACE in self.projection.projected_policy.allowed_operations
        )
        expected_decision = (
            "accepted-projected-prompt-action"
            if valid_prompt_action
            else (
                "rejected-non-prompt-surface"
                if not prompt_action
                else "rejected-invalid-prompt-action"
            )
        )
        if self.accepted is not valid_prompt_action or self.decision != expected_decision:
            raise ValueError("prompt-only enforcement decision differs from projected policy")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def artifact_ref(self) -> ArtifactRef:
        payload = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(payload),
            media_type=PROMPT_ONLY_ENFORCEMENT_RECEIPT_MEDIA_TYPE,
        )


__all__ = [
    "PROMPT_ONLY_ENFORCEMENT_RECEIPT_MEDIA_TYPE",
    "PROMPT_ONLY_POLICY_PROJECTION_MEDIA_TYPE",
    "RANDOM_VALID_CATALOG_MEDIA_TYPE",
    "RANDOM_VALID_PARENT_SNAPSHOT_MEDIA_TYPE",
    "RANDOM_VALID_SAMPLER_CONFIG_MEDIA_TYPE",
    "RANDOM_VALID_SELECTION_RECEIPT_MEDIA_TYPE",
    "PromptOnlyEnforcementReceipt",
    "PromptOnlyPolicyProjection",
    "RandomValidCatalog",
    "RandomValidCatalogEntry",
    "RandomValidParentSnapshot",
    "RandomValidSamplerBinding",
    "RandomValidSamplerConfig",
    "RandomValidSelectionReceipt",
    "make_prompt_only_policy_projection",
    "make_random_valid_selection_receipt",
    "verify_random_valid_selection_chain",
]
