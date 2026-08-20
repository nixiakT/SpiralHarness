from __future__ import annotations

import ast
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer as subject
import spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts as contracts
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationId,
    BfclV4PublicV2MutationProposal,
    materialize_bfcl_v4_public_v2_mutation,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer import (
    BfclV4PublicV2RequestMaterializationError,
    materialize_bfcl_v4_public_v2_request,
    resolve_bfcl_v4_public_v2_task,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2FrozenArmTreatment,
    BfclV4PublicV2RequestMaterialization,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_authority_contracts import (
    BfclV4PublicV2SemanticReleaseEvidenceShape,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationTreatmentRole,
    materialize_bfcl_v4_public_v2_mutation_runtime_batch,
)


@dataclass(frozen=True)
class _SyntheticRosterEntry:
    ordinal: int
    task_id: str
    category: str
    split: BfclV4PublicDevelopmentV2Split
    official_function_names: tuple[str, ...]
    candidate_payload_sha256: str


@dataclass(frozen=True)
class _SyntheticTask:
    task_id: str
    category: str
    question_json: str
    function_schemas_json: str
    official_function_names: tuple[str, ...]
    candidate_payload_sha256: str


@dataclass(frozen=True)
class _SyntheticManifest:
    roster: tuple[_SyntheticRosterEntry, ...]
    fingerprint: str = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT


def _category(task_id: str) -> str:
    return next(
        category
        for category in ("parallel_multiple", "simple_python", "multiple", "parallel")
        if task_id.startswith(f"{category}_")
    )


@pytest.fixture(scope="module")
def campaign() -> BfclV4PublicDevelopmentV2CampaignPlan:
    return build_bfcl_v4_public_development_v2_campaign_plan()


@pytest.fixture()
def synthetic_loaded(
    monkeypatch: pytest.MonkeyPatch,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
) -> BfclV4LoadedPublicDevelopmentV2:
    identities = dict(BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES)
    entries: list[_SyntheticRosterEntry] = []
    tasks: list[_SyntheticTask] = []
    for ordinal, task_id in enumerate(BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS):
        frozen = list(identities[task_id])
        names = frozen[3]
        question = [[{"role": "user", "content": f"synthetic question {ordinal}"}]]
        functions = [
            {
                "name": name,
                "description": "synthetic schema",
                "parameters": {"type": "object", "properties": {}},
            }
            for name in names
        ]
        candidate_sha256 = canonical_sha256(
            {"id": task_id, "question": question, "function": functions}
        )
        frozen[2] = candidate_sha256
        identities[task_id] = tuple(frozen)
        category = _category(task_id)
        split = BfclV4PublicDevelopmentV2Split(frozen[7])
        entries.append(
            _SyntheticRosterEntry(
                ordinal=ordinal,
                task_id=task_id,
                category=category,
                split=split,
                official_function_names=names,
                candidate_payload_sha256=candidate_sha256,
            )
        )
        tasks.append(
            _SyntheticTask(
                task_id=task_id,
                category=category,
                question_json=canonical_json(question),
                function_schemas_json=canonical_json(functions),
                official_function_names=names,
                candidate_payload_sha256=candidate_sha256,
            )
        )

    monkeypatch.setattr(contracts, "BFCL_V4_PUBLIC_DEVELOPMENT_V2_ROW_IDENTITIES", identities)

    def accept_synthetic(cls, value, *, strict, **_kwargs):
        assert cls is BfclV4LoadedPublicDevelopmentV2
        assert strict is True
        return value

    monkeypatch.setattr(
        BfclV4LoadedPublicDevelopmentV2,
        "model_validate",
        classmethod(accept_synthetic),
    )

    # The module-scoped campaign fixture is built and closed by its production
    # constructor. Revalidating all 1,098 nodes twice in every synthetic case
    # obscures the materializer assertions, so these cases reuse that closure.
    def accept_closed_campaign(cls, _value, *, strict, **_kwargs):
        assert cls is BfclV4PublicDevelopmentV2CampaignPlan
        assert strict is True
        return campaign

    monkeypatch.setattr(
        BfclV4PublicDevelopmentV2CampaignPlan,
        "model_validate",
        classmethod(accept_closed_campaign),
    )
    monkeypatch.setattr(
        BfclV4PublicDevelopmentV2CampaignPlan,
        "fingerprint",
        property(lambda _self: BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT),
    )
    return BfclV4LoadedPublicDevelopmentV2.model_construct(
        manifest=_SyntheticManifest(tuple(entries)),
        tasks=tuple(tasks),
        audit_receipt=None,
    )


@pytest.fixture(scope="module")
def prompts():
    materialization = materialize_bfcl_v4_public_v2_mutation(
        BfclV4PublicV2MutationProposal(
            catalogue_id=BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER
        )
    )
    batch = materialize_bfcl_v4_public_v2_mutation_runtime_batch(materialization)
    return {prompt.role: prompt for prompt in batch.prompts}


@pytest.fixture(scope="module")
def semantic_authority(bfcl_v2_verified_legacy_authority):
    return bfcl_v2_verified_legacy_authority.capability


def _node(campaign, *, arm, kind, variant=None):
    return next(
        node
        for node in campaign.nodes
        if node.arm is arm
        and node.kind is kind
        and node.task_ref is not None
        and (variant is None or node.harness_variant == variant)
    )


def _treatment(node, prompts, *, fallback=False):
    if node.arm in {
        BfclV4PublicDevelopmentV2Arm.PURE,
        BfclV4PublicDevelopmentV2Arm.PURE_AT_B,
    }:
        prompt = None
    elif node.harness_variant == "revert":
        prompt = prompts[BfclV4PublicV2MutationTreatmentRole.REVERT]
    elif node.harness_variant == "negative-control":
        prompt = prompts[BfclV4PublicV2MutationTreatmentRole.NEGATIVE_CONTROL]
    elif node.harness_variant in {
        "nominated-candidate",
        "selected-or-parent-fallback",
    } or node.harness_variant.startswith("typed-candidate-"):
        role = (
            BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT
            if fallback
            else BfclV4PublicV2MutationTreatmentRole.CANDIDATE
        )
        prompt = prompts[role]
    else:
        prompt = prompts[BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT]
    return BfclV4PublicV2FrozenArmTreatment(
        arm=node.arm,
        harness_variant=node.harness_variant,
        prompt=prompt,
        parent_fallback_used=fallback,
    )


def _materialize(
    *,
    loaded,
    campaign,
    node,
    prompts,
    semantic_authority,
    fallback=False,
):
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    return materialize_bfcl_v4_public_v2_request(
        loaded=loaded,
        campaign=campaign,
        lineage=lineage,
        semantic_authority=semantic_authority,
        treatment=_treatment(node, prompts, fallback=fallback),
    )


def test_resolver_maps_all_25_structural_refs_to_one_frozen_task_without_data_io(
    synthetic_loaded,
) -> None:
    refs = (
        *(f"fit-{index:02d}" for index in range(5)),
        *(f"gate-{index:02d}" for index in range(4)),
        *(f"holdout-{index:02d}" for index in range(16)),
    )
    receipts = tuple(
        resolve_bfcl_v4_public_v2_task(loaded=synthetic_loaded, task_ref=task_ref)
        for task_ref in refs
    )

    assert tuple(receipt.task_ref for receipt in receipts) == refs
    assert tuple(receipt.manifest_ordinal for receipt in receipts) == tuple(range(25))
    assert tuple(receipt.task_id for receipt in receipts) == (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
    )
    assert len({receipt.fingerprint for receipt in receipts}) == 25
    assert all(receipt.trusted_control_plane_only for receipt in receipts)
    assert not any(receipt.candidate_visible for receipt in receipts)


@pytest.mark.parametrize(
    ("arm", "kind", "variant", "fallback"),
    (
        (
            BfclV4PublicDevelopmentV2Arm.PURE,
            BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
            "bare",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.STATIC,
            BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
            "static-frozen",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
            "parent",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.FULL,
            BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
            "typed-candidate-0-or-parent-fallback",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
            "typed-candidate-0-or-parent-fallback",
            True,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.FULL,
            BfclV4PublicDevelopmentV2NodeKind.GATE,
            "nominated-candidate",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2NodeKind.GATE,
            "nominated-candidate",
            True,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.SCORE,
            BfclV4PublicDevelopmentV2NodeKind.GATE,
            "revert",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.FULL,
            BfclV4PublicDevelopmentV2NodeKind.GATE,
            "negative-control",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.FULL,
            BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
            "selected-or-parent-fallback",
            False,
        ),
        (
            BfclV4PublicDevelopmentV2Arm.PURE_AT_B,
            BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
            "bare-sample-0",
            False,
        ),
    ),
)
def test_materializer_covers_every_task_bound_treatment_shape(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
    arm,
    kind,
    variant,
    fallback,
) -> None:
    node = _node(campaign, arm=arm, kind=kind, variant=variant)
    materialized = _materialize(
        loaded=synthetic_loaded,
        campaign=campaign,
        node=node,
        prompts=prompts,
        semantic_authority=semantic_authority,
        fallback=fallback,
    )

    request = materialized.model_visible_request
    assert request.provider_seed_u63 == node.provider_seed_u63
    assert request.fingerprint == materialized.request_payload_sha256
    assert materialized.node == node
    assert materialized.treatment.parent_fallback_used is fallback
    assert materialized.semantic_release_authority_verified is True
    assert materialized.score_bearing_execution_allowed is True
    assert materialized.provider_calls == 0


def test_all_1098_nodes_have_exact_lineage_or_deterministic_control_rejection(
    campaign,
    monkeypatch,
) -> None:
    # Exercise one real strict closure, then reuse it while checking every
    # individual node binding. This keeps the test O(nodes), not O(nodes^2).
    assert (
        BfclV4PublicDevelopmentV2CampaignPlan.model_validate(
            campaign,
            strict=True,
        )
        == campaign
    )

    def accept_closed_campaign(cls, _value, *, strict, **_kwargs):
        assert cls is BfclV4PublicDevelopmentV2CampaignPlan
        assert strict is True
        return campaign

    monkeypatch.setattr(
        BfclV4PublicDevelopmentV2CampaignPlan,
        "model_validate",
        classmethod(accept_closed_campaign),
    )
    monkeypatch.setattr(
        BfclV4PublicDevelopmentV2CampaignPlan,
        "fingerprint",
        property(lambda _self: BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT),
    )
    call_count = 0
    task_call_count = 0
    control_count = 0
    for node in campaign.nodes:
        if node.consumes_model_call:
            call_count += 1
            task_call_count += node.task_ref is not None
            lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
                campaign=campaign,
                node_id=node.node_id,
            )
            assert lineage.node_id == node.node_id
            assert lineage.node_reference_sha256 == canonical_sha256(node)
            assert lineage.campaign_call_slot == node.campaign_call_slot
            assert lineage.provider_seed_u63 == node.provider_seed_u63
        else:
            control_count += 1
            with pytest.raises(ValueError):
                derive_bfcl_v4_public_development_v2_node_request_lineage(
                    campaign=campaign,
                    node_id=node.node_id,
                )
    assert (call_count, task_call_count, control_count) == (1_086, 1_050, 12)


@pytest.mark.parametrize(
    "task_ref",
    ("fit-5", "fit-05", "gate-04", "holdout-16", "holdout--1", "HOLDOUT-00", "x-00"),
)
def test_resolver_rejects_malformed_or_out_of_range_task_refs(
    synthetic_loaded,
    task_ref,
) -> None:
    with pytest.raises(BfclV4PublicV2RequestMaterializationError):
        resolve_bfcl_v4_public_v2_task(loaded=synthetic_loaded, task_ref=task_ref)


def test_materializer_rejects_non_task_calls_and_control_node_lineage(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
) -> None:
    diagnosis = next(
        node for node in campaign.nodes if node.kind is BfclV4PublicDevelopmentV2NodeKind.DIAGNOSIS
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=diagnosis.node_id,
    )
    parent = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        variant="parent",
    )
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="task-bound"):
        materialize_bfcl_v4_public_v2_request(
            loaded=synthetic_loaded,
            campaign=campaign,
            lineage=lineage,
            semantic_authority=semantic_authority,
            treatment=_treatment(parent, prompts),
        )

    control = next(node for node in campaign.nodes if not node.consumes_model_call)
    forged = lineage.model_copy(update={"node_id": control.node_id})
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="registered model-call"):
        materialize_bfcl_v4_public_v2_request(
            loaded=synthetic_loaded,
            campaign=campaign,
            lineage=forged,
            semantic_authority=semantic_authority,
            treatment=_treatment(parent, prompts),
        )


def test_exact_contract_types_and_tampered_lineage_are_rejected(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.PURE,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        variant="bare",
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    treatment = _treatment(node, prompts)
    base = dict(
        loaded=synthetic_loaded,
        campaign=campaign,
        lineage=lineage,
        semantic_authority=semantic_authority,
        treatment=treatment,
    )
    for field_name, value in (
        ("campaign_plan_fingerprint", "0" * 64),
        ("node_schedule_content_sha256", "1" * 64),
        ("node_reference_sha256", "2" * 64),
        ("campaign_call_slot", (lineage.campaign_call_slot + 1) % 1_086),
        ("provider_seed_u63", lineage.provider_seed_u63 ^ 1),
    ):
        with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="reconstruction"):
            materialize_bfcl_v4_public_v2_request(
                **{**base, "lineage": lineage.model_copy(update={field_name: value})}
            )

    class _CampaignSubclass(BfclV4PublicDevelopmentV2CampaignPlan):
        pass

    wrong_campaign = _CampaignSubclass.model_construct(**campaign.model_dump(mode="python"))
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="exact"):
        materialize_bfcl_v4_public_v2_request(**{**base, "campaign": wrong_campaign})
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="loaded bundle"):
        materialize_bfcl_v4_public_v2_request(**{**base, "loaded": campaign})
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="lineage"):
        materialize_bfcl_v4_public_v2_request(**{**base, "lineage": campaign})


def test_request_dump_is_minimal_and_contains_no_private_roster_or_other_task(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        variant="typed-candidate-0-or-parent-fallback",
    )
    materialized = _materialize(
        loaded=synthetic_loaded,
        campaign=campaign,
        node=node,
        prompts=prompts,
        semantic_authority=semantic_authority,
    )
    visible = materialized.model_visible_request.model_dump(mode="json")
    assert set(visible) == {
        "model_route",
        "inference",
        "provider_seed_u63",
        "system_prompt",
        "question_json",
        "function_schemas_json",
    }
    private_keys = {
        "task_id",
        "task_ref",
        "split",
        "manifest",
        "roster",
        "lineage",
        "node_id",
        "answer",
        "score",
        "result",
        "semantic_release_ref",
    }
    assert private_keys.isdisjoint(visible)
    serialized = canonical_json(visible)
    assert materialized.resolved_task.task_id not in serialized
    other_task = next(
        task
        for task in synthetic_loaded.tasks
        if task.task_id != materialized.resolved_task.task_id
    )
    assert other_task.question_json not in serialized
    assert "synthetic question" in materialized.model_visible_request.question_json


def test_question_schema_and_treatment_substitution_fail_closed(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
        variant="typed-candidate-0-or-parent-fallback",
    )
    materialized = _materialize(
        loaded=synthetic_loaded,
        campaign=campaign,
        node=node,
        prompts=prompts,
        semantic_authority=semantic_authority,
    )
    other_task = synthetic_loaded.tasks[(materialized.resolved_task.manifest_ordinal + 1) % 25]
    wrong_question = materialized.model_visible_request.model_copy(
        update={"question_json": other_task.question_json}
    )
    wrong_schemas = materialized.model_visible_request.model_copy(
        update={"function_schemas_json": other_task.function_schemas_json}
    )
    wrong_treatment = BfclV4PublicV2FrozenArmTreatment(
        arm=node.arm,
        harness_variant=node.harness_variant,
        prompt=prompts[BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT],
        parent_fallback_used=True,
    )
    for forged in (
        materialized.model_copy(update={"model_visible_request": wrong_question}),
        materialized.model_copy(update={"model_visible_request": wrong_schemas}),
        materialized.model_copy(update={"treatment": wrong_treatment}),
    ):
        with pytest.raises(ValidationError):
            BfclV4PublicV2RequestMaterialization.model_validate(forged, strict=True)


def test_materializer_requires_live_authority_and_binds_its_exact_release(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
    bfcl_v2_verified_legacy_authority,
) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.PURE,
        kind=BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
        variant="bare",
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    base = dict(
        loaded=synthetic_loaded,
        campaign=campaign,
        lineage=lineage,
        treatment=_treatment(node, prompts),
    )
    for ordinary_value in (
        bfcl_v2_verified_legacy_authority.release,
        bfcl_v2_verified_legacy_authority.release.ref,
        object(),
    ):
        with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="authority"):
            materialize_bfcl_v4_public_v2_request(
                **base,
                semantic_authority=ordinary_value,
            )

    materialized = materialize_bfcl_v4_public_v2_request(
        **base,
        semantic_authority=semantic_authority,
    )
    assert materialized.semantic_release_ref == semantic_authority.release_ref
    assert materialized.semantic_release_fingerprint == semantic_authority.release_fingerprint
    assert materialized.semantic_authority_verification_input_fingerprint == (
        semantic_authority.verification_input_fingerprint
    )
    assert materialized.semantic_release_media_type_verified
    assert materialized.semantic_release_authority_verified
    assert not materialized.semantic_release_content_loaded
    assert materialized.score_bearing_execution_allowed

    cross_shape = materialized.model_copy(
        update={
            "semantic_release_evidence_shape": (
                BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4
            )
        }
    )
    with pytest.raises(ValidationError, match="shape, ref, or fingerprint"):
        BfclV4PublicV2RequestMaterialization.model_validate(cross_shape, strict=True)


def test_treatment_arm_variant_and_fallback_must_match_exact_node(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
) -> None:
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.SCORE,
        kind=BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
        variant="parent",
    )
    lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
        campaign=campaign,
        node_id=node.node_id,
    )
    static_treatment = BfclV4PublicV2FrozenArmTreatment(
        arm=BfclV4PublicDevelopmentV2Arm.STATIC,
        harness_variant="static-frozen",
        prompt=prompts[BfclV4PublicV2MutationTreatmentRole.STATIC_PARENT],
    )
    with pytest.raises(BfclV4PublicV2RequestMaterializationError, match="arm"):
        materialize_bfcl_v4_public_v2_request(
            loaded=synthetic_loaded,
            campaign=campaign,
            lineage=lineage,
            semantic_authority=semantic_authority,
            treatment=static_treatment,
        )
    with pytest.raises(ValidationError):
        BfclV4PublicV2FrozenArmTreatment(
            arm=BfclV4PublicDevelopmentV2Arm.SCORE,
            harness_variant="parent",
            prompt=prompts[BfclV4PublicV2MutationTreatmentRole.CANDIDATE],
        )


def test_materialization_is_deterministic_and_never_invokes_network_or_provider(
    synthetic_loaded,
    campaign,
    prompts,
    semantic_authority,
    monkeypatch,
) -> None:
    def invocation_trap(*_args, **_kwargs):
        raise AssertionError("provider-free request materializer attempted network access")

    monkeypatch.setattr(socket, "socket", invocation_trap)
    monkeypatch.setattr(socket, "create_connection", invocation_trap)
    node = _node(
        campaign,
        arm=BfclV4PublicDevelopmentV2Arm.FULL,
        kind=BfclV4PublicDevelopmentV2NodeKind.GATE,
        variant="negative-control",
    )
    first = _materialize(
        loaded=synthetic_loaded,
        campaign=campaign,
        node=node,
        prompts=prompts,
        semantic_authority=semantic_authority,
    )
    second = _materialize(
        loaded=synthetic_loaded,
        campaign=campaign,
        node=node,
        prompts=prompts,
        semantic_authority=semantic_authority,
    )
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.provider_calls == 0


def test_new_production_modules_are_small_static_and_use_no_dynamic_imports() -> None:
    paths = (
        Path(subject.__file__),
        Path(contracts.__file__),
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert len(source.splitlines()) <= 700
        assert "importlib" not in source
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            for node in ast.walk(tree)
        )
