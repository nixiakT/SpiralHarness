"""Resolve and materialize one provider-free BFCL V4 public-v2 request.

The resolver consumes an already loaded, trusted question-only bundle.  It
does not open benchmark files, inspect answers or results, invoke a provider,
or turn an opaque semantic-release reference into execution authority.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    BfclV4PublicDevelopmentV2NodeRequestLineage,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2FrozenArmTreatment,
    BfclV4PublicV2ModelVisibleRequest,
    BfclV4PublicV2RequestMaterialization,
    BfclV4PublicV2ResolvedTaskReceipt,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef

_TASK_REF = re.compile(r"^(fit|gate|holdout)-([0-9]{2})$")
_SPLIT_COUNTS = {
    BfclV4PublicDevelopmentV2Split.FIT: 5,
    BfclV4PublicDevelopmentV2Split.GATE: 4,
    BfclV4PublicDevelopmentV2Split.HOLDOUT: 16,
}


class BfclV4PublicV2RequestMaterializationError(ValueError):
    """A private coordinate could not be resolved into one safe request."""


def _reject(message: str) -> None:
    raise BfclV4PublicV2RequestMaterializationError(message) from None


def _checked_loaded(
    loaded: BfclV4LoadedPublicDevelopmentV2,
) -> BfclV4LoadedPublicDevelopmentV2:
    if type(loaded) is not BfclV4LoadedPublicDevelopmentV2:
        _reject("loaded bundle must use the exact BFCL v2 contract")
    try:
        return BfclV4LoadedPublicDevelopmentV2.model_validate(loaded, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("loaded bundle differs from the frozen BFCL v2 contract")


def _checked_campaign(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> BfclV4PublicDevelopmentV2CampaignPlan:
    if type(campaign) is not BfclV4PublicDevelopmentV2CampaignPlan:
        _reject("campaign must use the exact BFCL v2 plan contract")
    try:
        return BfclV4PublicDevelopmentV2CampaignPlan.model_validate(campaign, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("campaign differs from the exact frozen BFCL v2 plan")


def _checked_lineage(
    lineage: BfclV4PublicDevelopmentV2NodeRequestLineage,
) -> BfclV4PublicDevelopmentV2NodeRequestLineage:
    if type(lineage) is not BfclV4PublicDevelopmentV2NodeRequestLineage:
        _reject("lineage must use the exact BFCL v2 request contract")
    try:
        return BfclV4PublicDevelopmentV2NodeRequestLineage.model_validate(
            lineage,
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        _reject("lineage differs from the frozen BFCL v2 request contract")


def _checked_treatment(
    treatment: BfclV4PublicV2FrozenArmTreatment,
) -> BfclV4PublicV2FrozenArmTreatment:
    if type(treatment) is not BfclV4PublicV2FrozenArmTreatment:
        _reject("treatment must use the exact BFCL v2 arm contract")
    try:
        return BfclV4PublicV2FrozenArmTreatment.model_validate(treatment, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("treatment differs from the frozen BFCL v2 prompt runtime")


def _checked_release_ref(semantic_release_ref: ArtifactRef) -> ArtifactRef:
    if type(semantic_release_ref) is not ArtifactRef:
        _reject("semantic release must be an exact ArtifactRef")
    try:
        checked = ArtifactRef.model_validate(semantic_release_ref, strict=True)
    except (TypeError, ValidationError, ValueError):
        _reject("semantic release ref is malformed")
    if (
        checked.media_type != BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE
        or checked.size <= 0
    ):
        _reject("semantic release ref has the wrong media type or empty content")
    return checked


def _parse_task_ref(
    task_ref: str,
) -> tuple[BfclV4PublicDevelopmentV2Split, int]:
    if type(task_ref) is not str:
        _reject("task ref must be exact text")
    match = _TASK_REF.fullmatch(task_ref)
    if match is None:
        _reject("task ref must be fit-NN, gate-NN, or holdout-NN")
    split = BfclV4PublicDevelopmentV2Split(match.group(1))
    split_index = int(match.group(2))
    if split_index >= _SPLIT_COUNTS[split]:
        _reject("task ref index is outside its frozen split")
    return split, split_index


def _resolve_checked_task(
    loaded: BfclV4LoadedPublicDevelopmentV2,
    task_ref: str,
) -> BfclV4PublicV2ResolvedTaskReceipt:
    split, split_index = _parse_task_ref(task_ref)
    split_pairs = tuple(
        (entry, task)
        for entry, task in zip(loaded.manifest.roster, loaded.tasks, strict=True)
        if entry.split is split
    )
    if len(split_pairs) != _SPLIT_COUNTS[split]:
        _reject("loaded bundle differs from the frozen 5/4/16 split")
    entry, task = split_pairs[split_index]
    if (
        entry.ordinal < 0
        or loaded.manifest.roster[entry.ordinal] != entry
        or loaded.tasks[entry.ordinal] != task
        or task.task_id != entry.task_id
        or task.category != entry.category
        or task.official_function_names != entry.official_function_names
        or task.candidate_payload_sha256 != entry.candidate_payload_sha256
    ):
        _reject("task ref did not resolve to one exact manifest task")
    try:
        return BfclV4PublicV2ResolvedTaskReceipt(
            task_ref=task_ref,
            split=split,
            split_index=split_index,
            manifest_ordinal=entry.ordinal,
            task_id=task.task_id,
            category=task.category,
            official_function_names=task.official_function_names,
            candidate_payload_sha256=task.candidate_payload_sha256,
            question_json_sha256=sha256_bytes(task.question_json.encode("utf-8")),
            function_schemas_json_sha256=sha256_bytes(task.function_schemas_json.encode("utf-8")),
        )
    except (TypeError, ValidationError, ValueError):
        _reject("resolved task differs from its frozen identity")


def resolve_bfcl_v4_public_v2_task(
    *,
    loaded: BfclV4LoadedPublicDevelopmentV2,
    task_ref: str,
) -> BfclV4PublicV2ResolvedTaskReceipt:
    """Resolve one structural ref in split-local frozen-manifest order."""

    return _resolve_checked_task(_checked_loaded(loaded), task_ref)


def materialize_bfcl_v4_public_v2_request(
    *,
    loaded: BfclV4LoadedPublicDevelopmentV2,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    lineage: BfclV4PublicDevelopmentV2NodeRequestLineage,
    semantic_release_ref: ArtifactRef,
    treatment: BfclV4PublicV2FrozenArmTreatment,
) -> BfclV4PublicV2RequestMaterialization:
    """Build one minimal task-bound request without calling a provider."""

    checked_loaded = _checked_loaded(loaded)
    checked_campaign = _checked_campaign(campaign)
    checked_lineage = _checked_lineage(lineage)
    checked_release_ref = _checked_release_ref(semantic_release_ref)
    checked_treatment = _checked_treatment(treatment)

    if checked_loaded.manifest.fingerprint != checked_campaign.manifest_fingerprint:
        _reject("loaded manifest and campaign plan fingerprints differ")
    try:
        expected_lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
            campaign=checked_campaign,
            node_id=checked_lineage.node_id,
        )
    except (TypeError, ValidationError, ValueError):
        _reject("lineage does not select one registered model-call node")
    if checked_lineage != expected_lineage:
        _reject("lineage differs from deterministic campaign reconstruction")

    matches = tuple(
        node for node in checked_campaign.nodes if node.node_id == checked_lineage.node_id
    )
    if len(matches) != 1:
        _reject("lineage does not resolve to one campaign node")
    node = matches[0]
    if not node.consumes_model_call or node.task_ref is None:
        _reject("single-task request requires a task-bound model-call node")
    if (
        checked_treatment.arm is not node.arm
        or checked_treatment.harness_variant != node.harness_variant
    ):
        _reject("treatment arm or harness variant differs from the selected node")

    resolved_task = _resolve_checked_task(checked_loaded, node.task_ref)
    task = checked_loaded.tasks[resolved_task.manifest_ordinal]
    assert node.provider_seed_u63 is not None
    try:
        request = BfclV4PublicV2ModelVisibleRequest(
            model_route=BFCL_V4_PUBLIC_DEVELOPMENT_V2_MODEL_ROUTE,
            inference=BFCL_V4_PUBLIC_DEVELOPMENT_V2_INFERENCE,
            provider_seed_u63=node.provider_seed_u63,
            system_prompt=(
                None if checked_treatment.prompt is None else checked_treatment.prompt.system_prompt
            ),
            question_json=task.question_json,
            function_schemas_json=task.function_schemas_json,
        )
        return BfclV4PublicV2RequestMaterialization(
            semantic_release_ref=checked_release_ref,
            lineage=checked_lineage,
            node=node,
            resolved_task=resolved_task,
            treatment=checked_treatment,
            model_visible_request=request,
            request_payload_sha256=canonical_sha256(request),
        )
    except (TypeError, ValidationError, ValueError):
        _reject("task, treatment, or request failed the private-to-model boundary")


__all__ = [
    "BfclV4PublicV2RequestMaterializationError",
    "materialize_bfcl_v4_public_v2_request",
    "resolve_bfcl_v4_public_v2_task",
]
