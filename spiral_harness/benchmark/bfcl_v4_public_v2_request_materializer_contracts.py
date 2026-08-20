"""Fail-closed contracts for one BFCL V4 public-v2 solver request.

Only :class:`BfclV4PublicV2ModelVisibleRequest` may cross the model boundary.
The surrounding materialization retains private task, DAG, and verified
semantic-authority bindings for trusted replay.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2NodeRequestLineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Category,
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority_contracts import (
    BfclV4PublicV2SemanticReleaseEvidenceShape,
    bfcl_v4_public_v2_semantic_release_media_type,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import InferenceConfig
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationTreatmentRole,
    BfclV4PublicV2SolverTreatmentPrompt,
)

_TASK_REF = re.compile(r"^(fit|gate|holdout)-([0-9]{2})$")
_CANDIDATE_VARIANT = re.compile(r"^typed-candidate-[0-2]-or-parent-fallback$")
_BARE_SAMPLE_VARIANT = re.compile(r"^bare-sample-[0-6]$")
_SPLIT_OFFSETS = {
    BfclV4PublicDevelopmentV2Split.FIT: 0,
    BfclV4PublicDevelopmentV2Split.GATE: 5,
    BfclV4PublicDevelopmentV2Split.HOLDOUT: 9,
}
_SPLIT_COUNTS = {
    BfclV4PublicDevelopmentV2Split.FIT: 5,
    BfclV4PublicDevelopmentV2Split.GATE: 4,
    BfclV4PublicDevelopmentV2Split.HOLDOUT: 16,
}
_MODEL_VISIBLE_FIELDS = frozenset(
    {
        "model_route",
        "inference",
        "provider_seed_u63",
        "system_prompt",
        "question_json",
        "function_schemas_json",
    }
)
_PRIVATE_TOP_LEVEL_FIELDS = frozenset(
    {
        "task_id",
        "task_ref",
        "split",
        "manifest",
        "roster",
        "lineage",
        "node_id",
        "semantic_release_ref",
        "answer",
        "possible_answer",
        "score",
        "result",
    }
)


def _canonical_json_text(value: str, *, label: str) -> object:
    try:
        parsed = json.loads(value)
        if canonical_json(parsed) != value:
            raise ValueError("non-canonical JSON")
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be canonical strict JSON") from error
    return parsed


def _task_ref_coordinates(
    task_ref: str,
) -> tuple[BfclV4PublicDevelopmentV2Split, int, int]:
    match = _TASK_REF.fullmatch(task_ref)
    if match is None:
        raise ValueError("BFCL v2 task ref must be fit-NN, gate-NN, or holdout-NN")
    split = BfclV4PublicDevelopmentV2Split(match.group(1))
    split_index = int(match.group(2))
    if split_index >= _SPLIT_COUNTS[split]:
        raise ValueError("BFCL v2 task ref index is outside its frozen split")
    return split, split_index, _SPLIT_OFFSETS[split] + split_index


class BfclV4PublicV2ResolvedTaskReceipt(ImmutableModel):
    """Trusted resolution of one structural ref; never send this to a model."""

    schema_version: Literal["1"] = "1"
    manifest_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    )
    task_ref: NonEmptyStr
    split: BfclV4PublicDevelopmentV2Split
    split_index: Annotated[int, Field(ge=0, le=15, strict=True)]
    manifest_ordinal: Annotated[int, Field(ge=0, lt=25, strict=True)]
    task_id: NonEmptyStr
    category: BfclV4PublicDevelopmentV2Category
    official_function_names: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    candidate_payload_sha256: Sha256
    question_json_sha256: Sha256
    function_schemas_json_sha256: Sha256
    trusted_control_plane_only: Literal[True] = True
    candidate_visible: Literal[False] = False
    possible_answer_present: Literal[False] = False
    score_or_result_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_frozen_task_coordinate(self) -> Self:
        split, split_index, ordinal = _task_ref_coordinates(self.task_ref)
        expected_task_id = BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS[ordinal]
        frozen = BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES[expected_task_id]
        expected_category = next(
            category
            for category in ("parallel_multiple", "simple_python", "multiple", "parallel")
            if expected_task_id.startswith(f"{category}_")
        )
        observed = (
            self.split,
            self.split_index,
            self.manifest_ordinal,
            self.task_id,
            self.category,
            self.official_function_names,
            self.candidate_payload_sha256,
        )
        expected = (
            split,
            split_index,
            ordinal,
            expected_task_id,
            expected_category,
            frozen[3],
            frozen[2],
        )
        if observed != expected or frozen[7] != split.value:
            raise ValueError("resolved task receipt differs from the frozen manifest coordinate")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2FrozenArmTreatment(ImmutableModel):
    """Exact prompt-side resolution for one task-bound DAG node."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4PublicDevelopmentV2Arm
    harness_variant: NonEmptyStr
    prompt: BfclV4PublicV2SolverTreatmentPrompt | None = None
    parent_fallback_used: bool = False
    prompt_bytes_frozen: Literal[True] = True
    provider_calls: Literal[0] = 0
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_arm_variant_and_prompt(self) -> Self:
        if self.prompt is not None and type(self.prompt) is not BfclV4PublicV2SolverTreatmentPrompt:
            raise ValueError("treatment prompt must use the exact BFCL v2 runtime contract")
        role = None if self.prompt is None else self.prompt.role
        variant = self.harness_variant

        if self.arm is BfclV4PublicDevelopmentV2Arm.PURE:
            valid = variant == "bare" and role is None and not self.parent_fallback_used
        elif self.arm is BfclV4PublicDevelopmentV2Arm.PURE_AT_B:
            valid = (
                _BARE_SAMPLE_VARIANT.fullmatch(variant) is not None
                and role is None
                and not self.parent_fallback_used
            )
        elif self.arm is BfclV4PublicDevelopmentV2Arm.STATIC:
            valid = (
                variant == "static-frozen"
                and role is BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
                and not self.parent_fallback_used
            )
        elif self.arm in {
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2Arm.FULL,
        }:
            candidate_or_fallback = _CANDIDATE_VARIANT.fullmatch(
                variant
            ) is not None or variant in {"nominated-candidate", "selected-or-parent-fallback"}
            valid = any(
                (
                    variant == "parent"
                    and role is BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
                    and not self.parent_fallback_used,
                    candidate_or_fallback
                    and role is BfclV4PublicV2MutationTreatmentRole.CANDIDATE
                    and not self.parent_fallback_used,
                    candidate_or_fallback
                    and role is BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
                    and self.parent_fallback_used,
                    variant == "revert"
                    and role is BfclV4PublicV2MutationTreatmentRole.REVERT
                    and not self.parent_fallback_used,
                    variant == "negative-control"
                    and role is BfclV4PublicV2MutationTreatmentRole.NEGATIVE_CONTROL
                    and not self.parent_fallback_used,
                )
            )
        else:  # pragma: no cover - enum is closed
            valid = False
        if not valid:
            raise ValueError("frozen treatment differs from its arm and harness variant")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ModelVisibleRequest(ImmutableModel):
    """The complete and deliberately minimal payload visible at model dispatch."""

    model_route: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE]
    inference: InferenceConfig
    provider_seed_u63: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
    system_prompt: NonEmptyStr | None = None
    question_json: NonEmptyStr
    function_schemas_json: NonEmptyStr

    @model_validator(mode="after")
    def _close_visible_schema(self) -> Self:
        if self.inference != BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE:
            raise ValueError("model-visible inference differs from the frozen campaign")
        _canonical_json_text(self.question_json, label="BFCL v2 question")
        functions = _canonical_json_text(
            self.function_schemas_json,
            label="BFCL v2 function schemas",
        )
        if not isinstance(functions, list) or not functions:
            raise ValueError("BFCL v2 function schemas must be one non-empty list")
        visible_fields = frozenset(self.model_dump(mode="python"))
        if visible_fields != _MODEL_VISIBLE_FIELDS or visible_fields & _PRIVATE_TOP_LEVEL_FIELDS:
            raise ValueError("model-visible request schema crossed the private control boundary")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2RequestMaterialization(ImmutableModel):
    """Trusted wrapper binding one minimal request to its frozen private lineage."""

    schema_version: Literal["2"] = "2"
    semantic_release_ref: ArtifactRef
    semantic_release_fingerprint: Sha256
    semantic_release_evidence_shape: BfclV4PublicV2SemanticReleaseEvidenceShape
    semantic_authority_verification_input_fingerprint: Sha256
    lineage: BfclV4PublicDevelopmentV2NodeRequestLineage
    node: BfclV4PublicDevelopmentV2DagNode
    resolved_task: BfclV4PublicV2ResolvedTaskReceipt
    treatment: BfclV4PublicV2FrozenArmTreatment
    model_visible_request: BfclV4PublicV2ModelVisibleRequest
    request_payload_sha256: Sha256
    per_call_total_token_ceiling: Literal[
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING
    ] = BFCL_V4_PUBLIC_DEVELOPMENT_V2_PER_CALL_TOTAL_TOKEN_CEILING
    trusted_control_plane_only: Literal[True] = True
    candidate_visible_wrapper: Literal[False] = False
    request_payload_present: Literal[True] = True
    semantic_release_media_type_verified: Literal[True] = True
    semantic_release_authority_verified: Literal[True] = True
    semantic_release_content_loaded: Literal[False] = False
    provider_seed_honoring_attested: Literal[False] = False
    provider_calls: Literal[0] = 0
    score_bearing_execution_allowed: Literal[True] = True

    @model_validator(mode="after")
    def _bind_request_to_verified_authority(self) -> Self:
        if type(self.semantic_release_ref) is not ArtifactRef:
            raise ValueError("semantic release must be an exact ArtifactRef")
        if (
            self.semantic_release_ref.media_type
            != bfcl_v4_public_v2_semantic_release_media_type(self.semantic_release_evidence_shape)
            or self.semantic_release_ref.sha256 != self.semantic_release_fingerprint
            or self.semantic_release_ref.size <= 0
        ):
            raise ValueError("semantic release shape, ref, or fingerprint differs")
        if type(self.lineage) is not BfclV4PublicDevelopmentV2NodeRequestLineage:
            raise ValueError("request lineage must use the exact BFCL v2 contract")
        if type(self.node) is not BfclV4PublicDevelopmentV2DagNode:
            raise ValueError("request node must use the exact BFCL v2 DAG contract")
        if type(self.resolved_task) is not BfclV4PublicV2ResolvedTaskReceipt:
            raise ValueError("resolved task must use the exact BFCL v2 receipt contract")
        if type(self.treatment) is not BfclV4PublicV2FrozenArmTreatment:
            raise ValueError("treatment must use the exact BFCL v2 arm contract")
        if type(self.model_visible_request) is not BfclV4PublicV2ModelVisibleRequest:
            raise ValueError("visible request must use the exact BFCL v2 request contract")

        node = self.node
        lineage = self.lineage
        if (
            not node.consumes_model_call
            or node.task_ref is None
            or node.node_id != lineage.node_id
            or canonical_sha256(node) != lineage.node_reference_sha256
            or node.campaign_call_slot != lineage.campaign_call_slot
            or node.provider_seed_u63 != lineage.provider_seed_u63
            or node.task_ref != self.resolved_task.task_ref
            or node.arm is not self.treatment.arm
            or node.harness_variant != self.treatment.harness_variant
        ):
            raise ValueError("trusted request wrapper differs from node, task, or lineage")

        request = self.model_visible_request
        expected_prompt = (
            None if self.treatment.prompt is None else self.treatment.prompt.system_prompt
        )
        if (
            request.model_route != BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE
            or request.inference != BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE
            or request.provider_seed_u63 != lineage.provider_seed_u63
            or request.system_prompt != expected_prompt
            or self.request_payload_sha256 != request.fingerprint
        ):
            raise ValueError("model-visible request differs from frozen inference or treatment")

        question = _canonical_json_text(request.question_json, label="BFCL v2 question")
        functions = _canonical_json_text(
            request.function_schemas_json,
            label="BFCL v2 function schemas",
        )
        assert isinstance(functions, list)
        names = tuple(item.get("name") for item in functions if isinstance(item, dict))
        task = self.resolved_task
        candidate_payload = {"id": task.task_id, "question": question, "function": functions}
        if (
            sha256_bytes(request.question_json.encode("utf-8")) != task.question_json_sha256
            or sha256_bytes(request.function_schemas_json.encode("utf-8"))
            != task.function_schemas_json_sha256
            or canonical_sha256(candidate_payload) != task.candidate_payload_sha256
            or names != task.official_function_names
            or len(names) != len(functions)
        ):
            raise ValueError("model-visible question or schemas differ from the resolved task")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [name for name in globals() if name.startswith("Bfcl")]
