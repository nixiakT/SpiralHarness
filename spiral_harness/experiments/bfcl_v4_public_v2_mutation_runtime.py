"""Provider-free prompt runtime for frozen BFCL V4 public-v2 mutations.

The catalogue models logical operators and a local synthetic reference canary.
The native solver treatment is deliberately narrower: one predeclared prompt
instruction overlay on the exact STATIC parent prompt.  Materializing an
overlay proves prompt bytes and lineage, not provider inclusion, model
adherence, behavior change, capability, or permission to score.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE,
    BfclV4PublicV2MutableComponent,
    BfclV4PublicV2MutationCatalogueEntry,
    BfclV4PublicV2MutationId,
    BfclV4PublicV2MutationMaterialization,
    BfclV4PublicV2Operator,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_SEED_SYSTEM_PROMPT,
)

BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_PROTOCOL = "spiral-bfcl-v4-public-v2-prompt-runtime/v1"
BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_BATCH_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-prompt-runtime-batch.v1+json"
)
BFCL_V4_PUBLIC_V2_MUTATION_MATERIALIZATION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-mutation-materialization.v1+json"
)
BFCL_V4_PUBLIC_V2_SOLVER_PROMPT_MEDIA_TYPE = "text/plain; charset=utf-8"
BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT = BFCL_V4_SEED_SYSTEM_PROMPT
BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT_SHA256 = (
    "69ed2ef8c92024b4dc52251b2a13d3ac447592d3da1227dfa8c3536e7828c2e4"
)
BFCL_V4_PUBLIC_V2_PROMPT_OVERLAY_SEPARATOR = (
    "\n\nBFCL v2 predeclared runtime instruction overlay:\n"
)
BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_INSTRUCTION = (
    "Before responding, silently reread the immutable solver core and existing strategy "
    "appendix once. Add no operator-specific rule, and do not alter the supplied tools or "
    "native response protocol."
)

BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS = MappingProxyType(
    {
        BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER: (
            "At the numeric-renderer stage only, preserve numeric lexical intent while obeying "
            "the schema: keep explicitly integral values integral, retain an explicitly supplied "
            "real-valued form when the schema accepts a number, and never quote a JSON number."
        ),
        BfclV4PublicV2MutationId.COMPOUND_CLAUSE_TOOL_COVERAGE: (
            "At the clause-planner stage only, enumerate every independently requested actionable "
            "clause before calling tools, and ensure that each clause is covered by its required "
            "call or calls without omitting later clauses."
        ),
        BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR: (
            "At the argument-validator stage only, check the selected function's required fields "
            "before emission. Emit a call only when every required value is grounded in the "
            "request; never invent a missing required value or emit a malformed partial call."
        ),
        BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER: (
            "At the call-emitter stage only, preserve one call for every requested occurrence, "
            "including repeats, in request order. Never deduplicate or reorder calls; parallelize "
            "only when doing so preserves the requested call sequence."
        ),
        BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER: (
            "At the tool-matcher stage only, compare documented names, descriptions, parameters, "
            "and required capabilities against the requested operation. Select by full schema "
            "compatibility, never by surface-name overlap or tool position alone."
        ),
    }
)


def _utf8(value: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:  # pragma: no cover - frozen constants are scalar text
        raise ValueError("runtime prompt must contain Unicode scalar text") from error


def _prompt_ref(value: str) -> ArtifactRef:
    content = _utf8(value)
    return ArtifactRef(
        sha256=sha256_bytes(content),
        size=len(content),
        media_type=BFCL_V4_PUBLIC_V2_SOLVER_PROMPT_MEDIA_TYPE,
    )


def _compose_overlay(instruction: str) -> str:
    return (
        BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT
        + BFCL_V4_PUBLIC_V2_PROMPT_OVERLAY_SEPARATOR
        + instruction
    )


if (
    _prompt_ref(BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT).sha256
    != BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT_SHA256
):  # pragma: no cover - import-time fail-closed drift check
    raise RuntimeError("BFCL v2 STATIC parent prompt changed")


BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTION_SHA256 = MappingProxyType(
    {
        mutation_id: sha256_bytes(_utf8(instruction))
        for mutation_id, instruction in BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS.items()
    }
)
BFCL_V4_PUBLIC_V2_CANDIDATE_PROMPT_SHA256 = MappingProxyType(
    {
        mutation_id: _prompt_ref(_compose_overlay(instruction)).sha256
        for mutation_id, instruction in BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS.items()
    }
)
BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_PROMPT = _compose_overlay(
    BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_INSTRUCTION
)
BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_PROMPT_SHA256 = _prompt_ref(
    BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_PROMPT
).sha256


class BfclV4PublicV2MutationTreatmentRole(StrEnum):
    """The exact four prompt sides materialized for a selected mutation."""

    STATIC_PARENT = "static-parent"
    CANDIDATE = "candidate"
    REVERT = "revert"
    NEGATIVE_CONTROL = "negative-control"


class BfclV4PublicV2StageInstructionOverlay(ImmutableModel):
    """One logical stage's parent-to-candidate prompt overlay projection."""

    component: BfclV4PublicV2MutableComponent
    parent_instruction: None = None
    candidate_instruction: NonEmptyStr | None = None
    instruction_changed: bool

    @model_validator(mode="after")
    def _bind_diff(self) -> Self:
        if self.instruction_changed is not (self.candidate_instruction is not None):
            raise ValueError("stage instruction change flag differs from its prompt overlay")
        return self


def _entry(mutation_id: BfclV4PublicV2MutationId) -> BfclV4PublicV2MutationCatalogueEntry:
    return next(
        item
        for item in BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.entries
        if item.mutation_id is mutation_id
    )


def _expected_prompt_values(
    role: BfclV4PublicV2MutationTreatmentRole,
    mutation_id: BfclV4PublicV2MutationId,
) -> tuple[
    BfclV4PublicV2MutableComponent | None,
    BfclV4PublicV2Operator | None,
    str | None,
    str,
    bool,
    bool,
]:
    entry = _entry(mutation_id)
    if role is BfclV4PublicV2MutationTreatmentRole.CANDIDATE:
        instruction = BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS[mutation_id]
        return (
            entry.component,
            entry.operator,
            instruction,
            _compose_overlay(instruction),
            True,
            True,
        )
    if role is BfclV4PublicV2MutationTreatmentRole.NEGATIVE_CONTROL:
        instruction = BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_INSTRUCTION
        return None, None, instruction, BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_PROMPT, True, False
    return None, None, None, BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT, False, False


class BfclV4PublicV2SolverTreatmentPrompt(ImmutableModel):
    """One exact solver prompt, without a task, request, provider call, or score."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_PROTOCOL
    )
    role: BfclV4PublicV2MutationTreatmentRole
    mutation_id: BfclV4PublicV2MutationId
    logical_component: BfclV4PublicV2MutableComponent | None = None
    operator: BfclV4PublicV2Operator | None = None
    instruction_block: NonEmptyStr | None = None
    instruction_block_sha256: Sha256 | None = None
    system_prompt: NonEmptyStr
    system_prompt_ref: ArtifactRef
    instruction_overlay_materialized: bool
    selected_operator_prompt_activation_verified: bool
    prompt_realized_treatment: Literal[True] = True
    deterministic_middleware_transform_claimed: Literal[False] = False
    native_request_inclusion_verified: Literal[False] = False
    model_instruction_adherence_verified: Literal[False] = False
    model_behavior_change_verified: Literal[False] = False
    benchmark_capability_evidence: Literal[False] = False
    provider_calls: Literal[0] = 0
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_prompt(self) -> Self:
        component, operator, instruction, prompt, overlay, operator_active = (
            _expected_prompt_values(self.role, self.mutation_id)
        )
        expected_instruction_sha = None if instruction is None else sha256_bytes(_utf8(instruction))
        observed = (
            self.logical_component,
            self.operator,
            self.instruction_block,
            self.instruction_block_sha256,
            self.system_prompt,
            self.system_prompt_ref,
            self.instruction_overlay_materialized,
            self.selected_operator_prompt_activation_verified,
        )
        expected = (
            component,
            operator,
            instruction,
            expected_instruction_sha,
            prompt,
            _prompt_ref(prompt),
            overlay,
            operator_active,
        )
        if observed != expected:
            raise ValueError("solver treatment prompt differs from its frozen role and mutation")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _treatment_prompt(
    role: BfclV4PublicV2MutationTreatmentRole,
    mutation_id: BfclV4PublicV2MutationId,
) -> BfclV4PublicV2SolverTreatmentPrompt:
    component, operator, instruction, prompt, overlay, operator_active = _expected_prompt_values(
        role,
        mutation_id,
    )
    return BfclV4PublicV2SolverTreatmentPrompt(
        role=role,
        mutation_id=mutation_id,
        logical_component=component,
        operator=operator,
        instruction_block=instruction,
        instruction_block_sha256=(
            None if instruction is None else sha256_bytes(_utf8(instruction))
        ),
        system_prompt=prompt,
        system_prompt_ref=_prompt_ref(prompt),
        instruction_overlay_materialized=overlay,
        selected_operator_prompt_activation_verified=operator_active,
    )


def bfcl_v4_public_v2_mutation_materialization_ref(
    materialization: BfclV4PublicV2MutationMaterialization,
) -> ArtifactRef:
    """Content-address one strict frozen materialization under the runtime boundary."""

    checked = BfclV4PublicV2MutationMaterialization.model_validate(materialization, strict=True)
    content = canonical_json_bytes(checked)
    return ArtifactRef(
        sha256=canonical_sha256(checked),
        size=len(content),
        media_type=BFCL_V4_PUBLIC_V2_MUTATION_MATERIALIZATION_MEDIA_TYPE,
    )


class BfclV4PublicV2MutationRuntimeBatch(ImmutableModel):
    """Exact parent/candidate/revert/control prompt batch for one materialization."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_PROTOCOL
    )
    catalogue_ref: ArtifactRef
    materialization: BfclV4PublicV2MutationMaterialization
    materialization_ref: ArtifactRef
    mutation_id: BfclV4PublicV2MutationId
    selected_component: BfclV4PublicV2MutableComponent
    selected_operator: BfclV4PublicV2Operator
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    canary_ref: ArtifactRef
    stage_instruction_overlays: Annotated[
        tuple[BfclV4PublicV2StageInstructionOverlay, ...],
        Field(min_length=5, max_length=5),
    ]
    prompts: Annotated[
        tuple[BfclV4PublicV2SolverTreatmentPrompt, ...],
        Field(min_length=4, max_length=4),
    ]
    static_parent_prompt_sha256: Sha256
    candidate_prompt_sha256: Sha256
    revert_prompt_sha256: Sha256
    negative_control_prompt_sha256: Sha256
    candidate_instruction_sha256: Sha256
    selected_stage_instruction_diff_count: Literal[1] = 1
    revert_is_exact_parent_prompt: Literal[True] = True
    negative_control_is_predeclared_generic_prompt: Literal[True] = True
    negative_control_behavioral_neutrality_attested: Literal[False] = False
    non_prompt_request_semantics_intended_identical: Literal[True] = True
    native_request_semantics_equality_verified: Literal[False] = False
    future_native_orchestrator_must_verify_request_equality: Literal[True] = True
    provider_calls: Literal[0] = 0
    v2_question_answer_score_or_result_used: Literal[False] = False
    model_instruction_adherence_verified: Literal[False] = False
    model_behavior_change_verified: Literal[False] = False
    benchmark_capability_evidence: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    v2_orchestrator_execution_grant_bound: Literal[False] = False

    @model_validator(mode="after")
    def _bind_batch(self) -> Self:
        materialization = BfclV4PublicV2MutationMaterialization.model_validate(
            self.materialization,
            strict=True,
        )
        entry = materialization.entry
        expected_overlays = tuple(
            BfclV4PublicV2StageInstructionOverlay(
                component=component,
                candidate_instruction=(
                    BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS[entry.mutation_id]
                    if component is entry.component
                    else None
                ),
                instruction_changed=component is entry.component,
            )
            for component in BfclV4PublicV2MutableComponent
        )
        expected_prompts = tuple(
            _treatment_prompt(role, entry.mutation_id)
            for role in BfclV4PublicV2MutationTreatmentRole
        )
        by_role = {prompt.role: prompt for prompt in expected_prompts}
        expected = (
            BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE.ref,
            bfcl_v4_public_v2_mutation_materialization_ref(materialization),
            entry.mutation_id,
            entry.component,
            entry.operator,
            materialization.parent_ref,
            materialization.candidate_ref,
            materialization.canary.ref,
            expected_overlays,
            expected_prompts,
            by_role[BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT].system_prompt_ref.sha256,
            by_role[BfclV4PublicV2MutationTreatmentRole.CANDIDATE].system_prompt_ref.sha256,
            by_role[BfclV4PublicV2MutationTreatmentRole.REVERT].system_prompt_ref.sha256,
            by_role[BfclV4PublicV2MutationTreatmentRole.NEGATIVE_CONTROL].system_prompt_ref.sha256,
            BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTION_SHA256[entry.mutation_id],
        )
        observed = (
            self.catalogue_ref,
            self.materialization_ref,
            self.mutation_id,
            self.selected_component,
            self.selected_operator,
            self.parent_harness_ref,
            self.candidate_harness_ref,
            self.canary_ref,
            self.stage_instruction_overlays,
            self.prompts,
            self.static_parent_prompt_sha256,
            self.candidate_prompt_sha256,
            self.revert_prompt_sha256,
            self.negative_control_prompt_sha256,
            self.candidate_instruction_sha256,
        )
        if observed != expected:
            raise ValueError("mutation runtime batch differs from its frozen materialization")
        if sum(item.instruction_changed for item in self.stage_instruction_overlays) != 1:
            raise ValueError("runtime candidate must change exactly one logical instruction stage")
        if self.candidate_prompt_sha256 == self.static_parent_prompt_sha256:
            raise ValueError("runtime candidate prompt cannot be a no-op")
        if self.revert_prompt_sha256 != self.static_parent_prompt_sha256:
            raise ValueError("runtime revert must be the exact STATIC parent prompt")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @property
    def ref(self) -> ArtifactRef:
        content = canonical_json_bytes(self)
        return ArtifactRef(
            sha256=self.fingerprint,
            size=len(content),
            media_type=BFCL_V4_PUBLIC_V2_MUTATION_RUNTIME_BATCH_MEDIA_TYPE,
        )


def materialize_bfcl_v4_public_v2_mutation_runtime_batch(
    materialization: BfclV4PublicV2MutationMaterialization,
) -> BfclV4PublicV2MutationRuntimeBatch:
    """Build the four score-free prompts for one strict frozen mutation."""

    checked = BfclV4PublicV2MutationMaterialization.model_validate(materialization, strict=True)
    entry = checked.entry
    overlays = tuple(
        BfclV4PublicV2StageInstructionOverlay(
            component=component,
            candidate_instruction=(
                BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS[entry.mutation_id]
                if component is entry.component
                else None
            ),
            instruction_changed=component is entry.component,
        )
        for component in BfclV4PublicV2MutableComponent
    )
    prompts = tuple(
        _treatment_prompt(role, entry.mutation_id) for role in BfclV4PublicV2MutationTreatmentRole
    )
    by_role = {prompt.role: prompt for prompt in prompts}
    return BfclV4PublicV2MutationRuntimeBatch(
        catalogue_ref=checked.catalogue_ref,
        materialization=checked,
        materialization_ref=bfcl_v4_public_v2_mutation_materialization_ref(checked),
        mutation_id=entry.mutation_id,
        selected_component=entry.component,
        selected_operator=entry.operator,
        parent_harness_ref=checked.parent_ref,
        candidate_harness_ref=checked.candidate_ref,
        canary_ref=checked.canary.ref,
        stage_instruction_overlays=overlays,
        prompts=prompts,
        static_parent_prompt_sha256=by_role[
            BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
        ].system_prompt_ref.sha256,
        candidate_prompt_sha256=by_role[
            BfclV4PublicV2MutationTreatmentRole.CANDIDATE
        ].system_prompt_ref.sha256,
        revert_prompt_sha256=by_role[
            BfclV4PublicV2MutationTreatmentRole.REVERT
        ].system_prompt_ref.sha256,
        negative_control_prompt_sha256=by_role[
            BfclV4PublicV2MutationTreatmentRole.NEGATIVE_CONTROL
        ].system_prompt_ref.sha256,
        candidate_instruction_sha256=BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTION_SHA256[
            entry.mutation_id
        ],
    )


def verify_bfcl_v4_public_v2_mutation_runtime_batch(
    batch: BfclV4PublicV2MutationRuntimeBatch,
) -> BfclV4PublicV2MutationRuntimeBatch:
    """Strictly reconstruct the complete prompt batch and canonical lineage."""

    checked = BfclV4PublicV2MutationRuntimeBatch.model_validate(batch, strict=True)
    expected = materialize_bfcl_v4_public_v2_mutation_runtime_batch(checked.materialization)
    if checked != expected:
        raise ValueError("mutation runtime batch differs from deterministic reconstruction")
    return checked


class BfclV4PublicV2MutationRuntimeExecutionError(RuntimeError):
    """Prompt materialization alone cannot authorize score-bearing execution."""


def assert_bfcl_v4_public_v2_mutation_runtime_score_execution_allowed(
    batch: BfclV4PublicV2MutationRuntimeBatch,
) -> None:
    """Always reject until a separate v2 orchestrator binds all execution grants."""

    verify_bfcl_v4_public_v2_mutation_runtime_batch(batch)
    raise BfclV4PublicV2MutationRuntimeExecutionError(
        "BFCL v2 prompt runtime cannot authorize scoring without the frozen v2 orchestrator"
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("assert_bfcl")
    or name.startswith("bfcl_v4")
    or name.startswith("materialize_bfcl")
    or name.startswith("verify_bfcl")
]
