"""Provider-free atomic mutation catalogue for the prospective BFCL V4 v2 pilot.

The finite catalogue is informed by failures observed on the earlier public v1
development pilot.  It does not import a BFCL roster, question, grader, runner,
or provider.  Its synthetic canaries authorize neither benchmark scoring nor a
future execution campaign.
"""

from __future__ import annotations

import json
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.core.canonical import canonical_json, canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr

BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-mutation-catalogue.v1+json"
)
BFCL_V4_PUBLIC_V2_HARNESS_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-harness.v1+json"
)
BFCL_V4_PUBLIC_V2_CANARY_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-canary-receipt.v1+json"
)
BFCL_V4_PUBLIC_V2_MUTATION_TRANSFER_SOURCE = "bfcl-v1-public-development-failure-analysis"
BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL = "spiral-bfcl-v4-public-v2-atomic-mutations/v1"
_PARENT_IMPLEMENTATION = "parent-model-authored/default-v1"


class BfclV4PublicV2MutationId(StrEnum):
    """The complete finite set of v2 behavioral edits."""

    NUMERIC_SCHEMA_LEXICALIZER = "numeric-schema-lexicalizer"
    COMPOUND_CLAUSE_TOOL_COVERAGE = "compound-clause-tool-coverage"
    REQUIRED_ARGUMENT_VALIDATOR = "required-argument-validator"
    MULTIPLICITY_ORDER_PRESERVER = "multiplicity-order-preserver"
    SCHEMA_GROUNDED_MATCHER = "schema-grounded-matcher"


class BfclV4PublicV2MutableComponent(StrEnum):
    """Exactly one typed harness stage changed by a catalogue entry."""

    NUMERIC_RENDERER = "numeric-renderer"
    CLAUSE_PLANNER = "clause-planner"
    ARGUMENT_VALIDATOR = "argument-validator"
    CALL_EMITTER = "call-emitter"
    TOOL_MATCHER = "tool-matcher"


class BfclV4PublicV2Operator(StrEnum):
    """Executable provider-free operator identities used by the materializer."""

    RENDER_SCHEMA_NUMBER_LEXEMES = "render-schema-number-lexemes/v1"
    REQUIRE_EVERY_CLAUSE_COVERED = "require-every-clause-covered/v1"
    BLOCK_MISSING_REQUIRED_ARGUMENTS = "block-missing-required-arguments/v1"
    PRESERVE_CALL_MULTIPLICITY_AND_ORDER = "preserve-call-multiplicity-and-order/v1"
    MATCH_TOOLS_BY_SCHEMA_CAPABILITIES = "match-tools-by-schema-capabilities/v1"


class BfclV4PublicV2FailureFamily(StrEnum):
    """Public-v1 failure families that motivated the finite transfer catalogue."""

    NUMERIC_LEXEME = "numeric-schema-lexeme"
    COMPOUND_COVERAGE = "compound-request-coverage"
    REQUIRED_ARGUMENTS = "required-argument-completeness"
    MULTIPLICITY_ORDER = "call-multiplicity-order"
    SCHEMA_MATCHING = "schema-grounded-tool-selection"


class BfclV4PublicV2HarnessLineage(StrEnum):
    """Parent or one atomic candidate child."""

    PARENT = "parent"
    CANDIDATE = "candidate"


_ENTRY_SPECS = MappingProxyType(
    {
        BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER: (
            BfclV4PublicV2MutableComponent.NUMERIC_RENDERER,
            BfclV4PublicV2Operator.RENDER_SCHEMA_NUMBER_LEXEMES,
            BfclV4PublicV2FailureFamily.NUMERIC_LEXEME,
            "synthetic-number-v1",
        ),
        BfclV4PublicV2MutationId.COMPOUND_CLAUSE_TOOL_COVERAGE: (
            BfclV4PublicV2MutableComponent.CLAUSE_PLANNER,
            BfclV4PublicV2Operator.REQUIRE_EVERY_CLAUSE_COVERED,
            BfclV4PublicV2FailureFamily.COMPOUND_COVERAGE,
            "synthetic-compound-v1",
        ),
        BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR: (
            BfclV4PublicV2MutableComponent.ARGUMENT_VALIDATOR,
            BfclV4PublicV2Operator.BLOCK_MISSING_REQUIRED_ARGUMENTS,
            BfclV4PublicV2FailureFamily.REQUIRED_ARGUMENTS,
            "synthetic-required-v1",
        ),
        BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER: (
            BfclV4PublicV2MutableComponent.CALL_EMITTER,
            BfclV4PublicV2Operator.PRESERVE_CALL_MULTIPLICITY_AND_ORDER,
            BfclV4PublicV2FailureFamily.MULTIPLICITY_ORDER,
            "synthetic-multiplicity-v1",
        ),
        BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER: (
            BfclV4PublicV2MutableComponent.TOOL_MATCHER,
            BfclV4PublicV2Operator.MATCH_TOOLS_BY_SCHEMA_CAPABILITIES,
            BfclV4PublicV2FailureFamily.SCHEMA_MATCHING,
            "synthetic-matcher-v1",
        ),
    }
)


class BfclV4PublicV2MutationCatalogueEntry(ImmutableModel):
    """One closed, parameter-free behavioral mutation."""

    schema_version: Literal["1"] = "1"
    mutation_id: BfclV4PublicV2MutationId
    component: BfclV4PublicV2MutableComponent
    operator: BfclV4PublicV2Operator
    motivating_failure_family: BfclV4PublicV2FailureFamily
    synthetic_canary_id: NonEmptyStr
    atomic_edit_count: Literal[1] = 1
    executable_free_form_text_present: Literal[False] = False
    task_coordinates_present: Literal[False] = False
    answer_data_present: Literal[False] = False
    provider_required: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_closed_entry(self) -> Self:
        expected = _ENTRY_SPECS[self.mutation_id]
        observed = (
            self.component,
            self.operator,
            self.motivating_failure_family,
            self.synthetic_canary_id,
        )
        if observed != expected:
            raise ValueError("mutation entry differs from its closed catalogue specification")
        return self


class BfclV4PublicV2MutationCatalogue(ImmutableModel):
    """The complete v1-informed, v2-score-blind finite mutation grammar."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL] = BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL
    transfer_source: Literal[BFCL_V4_PUBLIC_V2_MUTATION_TRANSFER_SOURCE] = (
        BFCL_V4_PUBLIC_V2_MUTATION_TRANSFER_SOURCE
    )
    entries: Annotated[
        tuple[BfclV4PublicV2MutationCatalogueEntry, ...],
        Field(min_length=5, max_length=5),
    ]
    public_v1_failure_informed: Literal[True] = True
    v2_question_or_score_observation_used: Literal[False] = False
    candidate_selects_exactly_one_entry: Literal[True] = True
    free_form_appendix_allowed: Literal[False] = False
    task_or_answer_conditioning_allowed: Literal[False] = False
    provider_required: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_catalogue(self) -> Self:
        expected_ids = tuple(BfclV4PublicV2MutationId)
        if tuple(item.mutation_id for item in self.entries) != expected_ids:
            raise ValueError("mutation catalogue must retain its complete frozen order")
        if len({item.component for item in self.entries}) != len(self.entries):
            raise ValueError("each mutation must own one distinct typed component")
        if len({item.operator for item in self.entries}) != len(self.entries):
            raise ValueError("mutation operators must be globally unique")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        content = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=canonical_sha256(self),
            size=len(content),
            media_type=BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE_MEDIA_TYPE,
        )


def _catalogue_entry(mutation_id: BfclV4PublicV2MutationId) -> BfclV4PublicV2MutationCatalogueEntry:
    component, operator, failure_family, canary_id = _ENTRY_SPECS[mutation_id]
    return BfclV4PublicV2MutationCatalogueEntry(
        mutation_id=mutation_id,
        component=component,
        operator=operator,
        motivating_failure_family=failure_family,
        synthetic_canary_id=canary_id,
    )


BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE = BfclV4PublicV2MutationCatalogue(
    entries=tuple(_catalogue_entry(mutation_id) for mutation_id in BfclV4PublicV2MutationId)
)


class BfclV4PublicV2MutationProposal(ImmutableModel):
    """Candidate-authored value: exactly one finite catalogue ID and no text."""

    schema_version: Literal["1"] = "1"
    catalogue_id: BfclV4PublicV2MutationId
    selected_entry_count: Literal[1] = 1
    free_form_appendix_present: Literal[False] = False
    task_coordinates_present: Literal[False] = False
    answer_data_present: Literal[False] = False
    score_execution_requested: Literal[False] = False


class BfclV4PublicV2HarnessStage(ImmutableModel):
    """One stage of the typed harness pipeline."""

    component: BfclV4PublicV2MutableComponent
    implementation: NonEmptyStr


def _parent_stages() -> tuple[BfclV4PublicV2HarnessStage, ...]:
    return tuple(
        BfclV4PublicV2HarnessStage(
            component=component,
            implementation=_PARENT_IMPLEMENTATION,
        )
        for component in BfclV4PublicV2MutableComponent
    )


def _candidate_stages(
    entry: BfclV4PublicV2MutationCatalogueEntry,
) -> tuple[BfclV4PublicV2HarnessStage, ...]:
    return tuple(
        BfclV4PublicV2HarnessStage(
            component=component,
            implementation=(
                entry.operator.value if component is entry.component else _PARENT_IMPLEMENTATION
            ),
        )
        for component in BfclV4PublicV2MutableComponent
    )


class BfclV4PublicV2HarnessArtifact(ImmutableModel):
    """Content-addressed parent or single-operator candidate harness."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL] = BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL
    lineage: BfclV4PublicV2HarnessLineage
    parent_ref: ArtifactRef | None = None
    selected_catalogue_id: BfclV4PublicV2MutationId | None = None
    stages: Annotated[
        tuple[BfclV4PublicV2HarnessStage, ...],
        Field(min_length=5, max_length=5),
    ]
    public_v1_failure_informed: Literal[True] = True
    v2_question_or_score_observation_used: Literal[False] = False
    candidate_visible_task_coordinates_present: Literal[False] = False
    candidate_visible_answer_data_present: Literal[False] = False
    provider_required: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_atomic_pipeline(self) -> Self:
        if tuple(item.component for item in self.stages) != tuple(BfclV4PublicV2MutableComponent):
            raise ValueError("harness stages must retain the complete typed pipeline order")
        parent_stages = _parent_stages()
        if self.lineage is BfclV4PublicV2HarnessLineage.PARENT:
            if self.parent_ref is not None or self.selected_catalogue_id is not None:
                raise ValueError("parent harness cannot cite a parent or mutation")
            if self.stages != parent_stages:
                raise ValueError("parent harness stages differ from the frozen parent")
            return self

        if self.parent_ref != BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref:
            raise ValueError("candidate does not cite the exact frozen parent")
        if self.selected_catalogue_id is None:
            raise ValueError("candidate must select exactly one catalogue entry")
        entry = _catalogue_entry(self.selected_catalogue_id)
        if self.stages != _candidate_stages(entry):
            raise ValueError("candidate pipeline is not the selected atomic operator")
        differences = sum(
            left != right for left, right in zip(self.stages, parent_stages, strict=True)
        )
        if differences != 1:
            raise ValueError("candidate must change exactly one behavioral stage")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        content = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=canonical_sha256(self),
            size=len(content),
            media_type=BFCL_V4_PUBLIC_V2_HARNESS_MEDIA_TYPE,
        )


BFCL_V4_PUBLIC_V2_PARENT_HARNESS = BfclV4PublicV2HarnessArtifact(
    lineage=BfclV4PublicV2HarnessLineage.PARENT,
    stages=_parent_stages(),
)


class BfclV4PublicV2CanaryObservation(ImmutableModel):
    """One target or protected synthetic observation."""

    case_id: NonEmptyStr
    protected_case: bool
    parent_output_json: NonEmptyStr
    candidate_output_json: NonEmptyStr
    behavior_changed: bool
    passed: Literal[True] = True

    @model_validator(mode="after")
    def _require_canonical_outputs(self) -> Self:
        for value in (self.parent_output_json, self.candidate_output_json):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("canary output is not JSON") from error
            if canonical_json(parsed) != value:
                raise ValueError("canary output is not canonical JSON")
        if self.behavior_changed != (self.parent_output_json != self.candidate_output_json):
            raise ValueError("canary behavior-change flag differs from its outputs")
        if self.protected_case and self.behavior_changed:
            raise ValueError("protected canary behavior changed")
        if not self.protected_case and not self.behavior_changed:
            raise ValueError("target canary did not change behavior")
        return self


def _operator_active(
    artifact: BfclV4PublicV2HarnessArtifact,
    mutation_id: BfclV4PublicV2MutationId,
) -> bool:
    entry = _catalogue_entry(mutation_id)
    stage = next(item for item in artifact.stages if item.component is entry.component)
    return stage.implementation == entry.operator.value


def _synthetic_outputs(
    artifact: BfclV4PublicV2HarnessArtifact,
    mutation_id: BfclV4PublicV2MutationId,
) -> tuple[tuple[str, bool, object], ...]:
    active = _operator_active(artifact, mutation_id)
    if mutation_id is BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER:
        arguments_json = '{"ratio":7.0}' if active else '{"ratio":7}'
        return (
            ("synthetic-number-target", False, {"arguments_json": arguments_json}),
            ("synthetic-integer-protected", True, {"arguments_json": '{"count":7}'}),
        )
    if mutation_id is BfclV4PublicV2MutationId.COMPOUND_CLAUSE_TOOL_COVERAGE:
        calls = ("lookup_alpha", "lookup_beta") if active else ("lookup_alpha",)
        return (
            ("synthetic-compound-target", False, {"calls": calls}),
            ("synthetic-single-clause-protected", True, {"calls": ("lookup_alpha",)}),
        )
    if mutation_id is BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR:
        return (
            (
                "synthetic-missing-required-target",
                False,
                {"emit": not active, "missing": ("unit",)},
            ),
            (
                "synthetic-complete-arguments-protected",
                True,
                {"emit": True, "missing": ()},
            ),
        )
    if mutation_id is BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER:
        calls = ("beta", "alpha", "beta") if active else tuple(sorted({"beta", "alpha"}))
        return (
            ("synthetic-multiplicity-target", False, {"calls": calls}),
            ("synthetic-unique-order-protected", True, {"calls": ("alpha", "beta")}),
        )
    if mutation_id is BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER:
        selected = "forecast" if active else "notify"
        return (
            ("synthetic-capability-target", False, {"selected": selected}),
            ("synthetic-first-match-protected", True, {"selected": "forecast"}),
        )
    raise AssertionError("unreachable mutation ID")


def _synthetic_canary_observations(
    candidate: BfclV4PublicV2HarnessArtifact,
) -> tuple[BfclV4PublicV2CanaryObservation, ...]:
    if candidate.selected_catalogue_id is None:
        raise ValueError("synthetic canary requires a candidate mutation")
    mutation_id = candidate.selected_catalogue_id
    parent_outputs = _synthetic_outputs(BFCL_V4_PUBLIC_V2_PARENT_HARNESS, mutation_id)
    candidate_outputs = _synthetic_outputs(candidate, mutation_id)
    observations = tuple(
        BfclV4PublicV2CanaryObservation(
            case_id=parent_case_id,
            protected_case=parent_protected,
            parent_output_json=canonical_json(parent_output),
            candidate_output_json=canonical_json(candidate_output),
            behavior_changed=parent_output != candidate_output,
        )
        for (parent_case_id, parent_protected, parent_output), (
            candidate_case_id,
            candidate_protected,
            candidate_output,
        ) in zip(parent_outputs, candidate_outputs, strict=True)
        if parent_case_id == candidate_case_id and parent_protected == candidate_protected
    )
    if len(observations) != 2 or not _operator_active(candidate, mutation_id):
        raise ValueError("synthetic canary did not close its target and protected cases")
    return observations


class BfclV4PublicV2CanaryReceipt(ImmutableModel):
    """Provider-free proof about one local deterministic synthetic operator."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL] = BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL
    catalogue_id: BfclV4PublicV2MutationId
    parent_ref: ArtifactRef
    candidate_ref: ArtifactRef
    observations: Annotated[
        tuple[BfclV4PublicV2CanaryObservation, ...],
        Field(min_length=2, max_length=2),
    ]
    local_operator_activation_verified: Literal[True] = True
    deterministic_operator_adherence_verified: Literal[True] = True
    synthetic_behavior_change_verified: Literal[True] = True
    protected_behavior_verified: Literal[True] = True
    provider_calls: Literal[0] = 0
    model_invoked: Literal[False] = False
    model_behavior_attested: Literal[False] = False
    benchmark_capability_evidence: Literal[False] = False
    benchmark_items_present: Literal[False] = False
    answer_data_present: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_deterministic_canary(self) -> Self:
        expected_candidate = _candidate_artifact(_catalogue_entry(self.catalogue_id))
        if self.parent_ref != BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref:
            raise ValueError("canary does not cite the exact frozen parent")
        if self.candidate_ref != expected_candidate.ref:
            raise ValueError("canary does not cite the selected atomic candidate")
        if self.observations != _synthetic_canary_observations(expected_candidate):
            raise ValueError("canary observations differ from deterministic replay")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        content = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=canonical_sha256(self),
            size=len(content),
            media_type=BFCL_V4_PUBLIC_V2_CANARY_MEDIA_TYPE,
        )


def _candidate_artifact(
    entry: BfclV4PublicV2MutationCatalogueEntry,
) -> BfclV4PublicV2HarnessArtifact:
    return BfclV4PublicV2HarnessArtifact(
        lineage=BfclV4PublicV2HarnessLineage.CANDIDATE,
        parent_ref=BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref,
        selected_catalogue_id=entry.mutation_id,
        stages=_candidate_stages(entry),
    )


def _canary_receipt(
    candidate: BfclV4PublicV2HarnessArtifact,
) -> BfclV4PublicV2CanaryReceipt:
    if candidate.selected_catalogue_id is None:
        raise ValueError("synthetic canary requires a candidate mutation")
    mutation_id = candidate.selected_catalogue_id
    return BfclV4PublicV2CanaryReceipt(
        catalogue_id=mutation_id,
        parent_ref=BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref,
        candidate_ref=candidate.ref,
        observations=_synthetic_canary_observations(candidate),
    )


class BfclV4PublicV2MutationMaterialization(ImmutableModel):
    """Exact parent, atomic child, and provider-free canary closure."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL] = BFCL_V4_PUBLIC_V2_MUTATION_PROTOCOL
    catalogue_ref: ArtifactRef
    proposal: BfclV4PublicV2MutationProposal
    entry: BfclV4PublicV2MutationCatalogueEntry
    parent: BfclV4PublicV2HarnessArtifact
    candidate: BfclV4PublicV2HarnessArtifact
    parent_ref: ArtifactRef
    candidate_ref: ArtifactRef
    canary: BfclV4PublicV2CanaryReceipt
    behavioral_stage_diff_count: Literal[1] = 1
    deterministic_materialization: Literal[True] = True
    provider_calls: Literal[0] = 0
    v2_question_or_score_observation_used: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_materialization(self) -> Self:
        expected_entry = _catalogue_entry(self.proposal.catalogue_id)
        expected_candidate = _candidate_artifact(expected_entry)
        if self.catalogue_ref != BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.ref:
            raise ValueError("materialization cites the wrong mutation catalogue")
        if self.entry != expected_entry:
            raise ValueError("materialization entry differs from the selected catalogue ID")
        if self.parent != BFCL_V4_PUBLIC_V2_PARENT_HARNESS:
            raise ValueError("materialization does not contain the exact frozen parent")
        if self.parent_ref != self.parent.ref:
            raise ValueError("materialization parent ref differs from parent content")
        if self.candidate != expected_candidate or self.candidate_ref != self.candidate.ref:
            raise ValueError("materialization candidate differs from the atomic operator")
        if self.parent_ref == self.candidate_ref:
            raise ValueError("materialization cannot admit a no-op candidate")
        if self.canary != _canary_receipt(self.candidate):
            raise ValueError("materialization canary differs from deterministic replay")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2MutationExecutionError(RuntimeError):
    """The provider-free mutation grammar cannot authorize benchmark scoring."""


def materialize_bfcl_v4_public_v2_mutation(
    proposal: BfclV4PublicV2MutationProposal,
) -> BfclV4PublicV2MutationMaterialization:
    """Materialize and replay one finite atomic mutation without a provider."""

    checked = BfclV4PublicV2MutationProposal.model_validate(proposal, strict=True)
    entry = _catalogue_entry(checked.catalogue_id)
    candidate = _candidate_artifact(entry)
    canary = _canary_receipt(candidate)
    return BfclV4PublicV2MutationMaterialization(
        catalogue_ref=BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.ref,
        proposal=checked,
        entry=entry,
        parent=BFCL_V4_PUBLIC_V2_PARENT_HARNESS,
        candidate=candidate,
        parent_ref=BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref,
        candidate_ref=candidate.ref,
        canary=canary,
    )


def verify_bfcl_v4_public_v2_mutation_canary(
    materialization: BfclV4PublicV2MutationMaterialization,
) -> BfclV4PublicV2CanaryReceipt:
    """Strictly revalidate and deterministically replay the synthetic canary."""

    checked = BfclV4PublicV2MutationMaterialization.model_validate(
        materialization,
        strict=True,
    )
    return _canary_receipt(checked.candidate)


def assert_bfcl_v4_public_v2_score_execution_allowed(
    materialization: BfclV4PublicV2MutationMaterialization,
) -> None:
    """Always reject: this module is not a score-bearing execution authority."""

    BfclV4PublicV2MutationMaterialization.model_validate(materialization, strict=True)
    raise BfclV4PublicV2MutationExecutionError(
        "BFCL v2 atomic mutation canaries are provider-free and cannot authorize scoring"
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("assert_bfcl")
    or name.startswith("materialize_bfcl")
    or name.startswith("verify_bfcl")
]
