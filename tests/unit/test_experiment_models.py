from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import (
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    MutationPolicy,
    ProtocolManifest,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.models import ArtifactRef, BudgetPolicy, ComponentKind


def artifact(digit: str, *, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(sha256=digit * 64, size=1, media_type=media_type)


def split(partition: ProtocolPartition, digit: str) -> ProtocolSplit:
    return ProtocolSplit(partition=partition, manifest_ref=artifact(digit))


def protocol(*splits: ProtocolSplit) -> ProtocolManifest:
    return ProtocolManifest(
        benchmark_fingerprint="benchmark@v1",
        splits=splits,
        model_fingerprint="model@sha256:fixed",
        inference_fingerprint="temperature=0;seeded=true",
        runtime_fingerprint="runner@sha256:fixed",
        sandbox_fingerprint="sandbox@sha256:fixed",
        capability_policy_ref=artifact("e"),
        grader_fingerprint="grader@sha256:fixed",
        gate_batch_attestor_id="f" * 64,
        mechanism_evidence_attestor_id="e" * 64,
        gate_config_ref=artifact("d"),
        trusted_plane_version="trusted-plane-v1",
        budget=BudgetPolicy(max_evaluations=40, max_cost_usd=10.0),
    )


def test_protocol_requires_exploration_and_gate_and_canonicalizes_splits() -> None:
    manifest = protocol(
        split(ProtocolPartition.SEALED, "c"),
        split(ProtocolPartition.GATE, "b"),
        split(ProtocolPartition.EXPLORATION, "a"),
    )

    assert manifest.schema_version == "2"
    assert [item.partition for item in manifest.splits] == [
        ProtocolPartition.EXPLORATION,
        ProtocolPartition.GATE,
        ProtocolPartition.SEALED,
    ]
    assert (
        protocol(
            split(ProtocolPartition.GATE, "b"),
            split(ProtocolPartition.EXPLORATION, "a"),
            split(ProtocolPartition.SEALED, "c"),
        )
        == manifest
    )

    with pytest.raises(ValidationError, match="missing required split partitions: gate"):
        protocol(
            split(ProtocolPartition.EXPLORATION, "a"),
            split(ProtocolPartition.SEALED, "c"),
        )


def test_protocol_rejects_duplicate_partitions_and_split_manifests() -> None:
    with pytest.raises(ValidationError, match="partitions must be unique"):
        protocol(
            split(ProtocolPartition.EXPLORATION, "a"),
            split(ProtocolPartition.GATE, "b"),
            split(ProtocolPartition.GATE, "c"),
        )

    with pytest.raises(ValidationError, match="distinct manifests"):
        protocol(
            split(ProtocolPartition.EXPLORATION, "a"),
            split(ProtocolPartition.GATE, "a"),
        )


def test_protocol_is_strict_frozen_and_forbids_unregistered_fields() -> None:
    values = protocol(
        split(ProtocolPartition.EXPLORATION, "a"),
        split(ProtocolPartition.GATE, "b"),
    ).model_dump()
    values["unfrozen_override"] = "grader-v2"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProtocolManifest.model_validate(values)

    manifest = ProtocolManifest.model_validate(
        {key: value for key, value in values.items() if key != "unfrozen_override"}
    )
    with pytest.raises((ValidationError, FrozenInstanceError)):
        manifest.gate_config_ref = artifact("e")

    missing_attestor = manifest.model_dump()
    missing_attestor.pop("mechanism_evidence_attestor_id")
    with pytest.raises(ValidationError, match="mechanism_evidence_attestor_id"):
        ProtocolManifest.model_validate(missing_attestor)


def test_mutation_policy_defaults_to_prompt_only_and_canonicalizes_allowlists() -> None:
    default = MutationPolicy()
    assert default.allowed_kinds == (ComponentKind.PROMPT,)
    assert default.allowed_media_types == ("text/plain",)
    assert default.max_artifact_size_bytes == 65_536
    policy = MutationPolicy(
        allowed_kinds=(ComponentKind.SKILL, ComponentKind.PROMPT),
        allowed_component_names=("system", "planner"),
        allowed_media_types=("Text/Plain; charset=utf-8", "application/json"),
        max_artifact_size_bytes=1_024,
    )

    assert policy.allowed_kinds == (ComponentKind.PROMPT, ComponentKind.SKILL)
    assert policy.allowed_component_names == ("planner", "system")
    assert policy.allowed_media_types == ("application/json", "text/plain")

    with pytest.raises(ValidationError, match="invalid media type"):
        MutationPolicy(allowed_media_types=("not-a-media-type",))


@pytest.mark.parametrize(
    "media_type",
    [
        "/",
        "/plain",
        "text/",
        "text/plain/extra",
        "text/pla in",
        "text/pl@in",
        "text/\x00plain",
    ],
)
def test_mutation_policy_requires_a_mime_token_pair(media_type: str) -> None:
    with pytest.raises(ValidationError, match="invalid media type"):
        MutationPolicy(allowed_media_types=(media_type,))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allowed_kinds", ()),
        ("allowed_component_names", ()),
        ("allowed_media_types", ()),
        ("allowed_kinds", (ComponentKind.PROMPT, ComponentKind.PROMPT)),
        ("allowed_component_names", ("system", "system")),
        ("allowed_media_types", ("text/plain", "text/plain")),
    ],
)
def test_mutation_policy_rejects_empty_and_duplicate_allowlists(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        MutationPolicy(**{field_name: value})


def test_experiment_binds_search_contract_and_canonicalizes_set_like_fields() -> None:
    values = {
        "protocol_ref": artifact("a", media_type=PROTOCOL_MANIFEST_MEDIA_TYPE),
        "seed_harness_ref": artifact("b"),
        "objective": "maximize paired benchmark score subject to frozen constraints",
        "search_budget": BudgetPolicy(max_evaluations=24, max_cost_usd=5.0),
    }
    left = ExperimentManifest(
        **values,
        baselines=("static", "random-valid", "prompt-only"),
        stopping=("budget-exhausted", "no-promotable-candidates"),
    )
    right = ExperimentManifest(
        **values,
        baselines=("prompt-only", "static", "random-valid"),
        stopping=("no-promotable-candidates", "budget-exhausted"),
    )

    assert left.mutation_policy == MutationPolicy()
    assert left.baselines == ("prompt-only", "random-valid", "static")
    assert left.stopping == ("budget-exhausted", "no-promotable-candidates")
    assert left == right
    assert canonical_sha256(left) == canonical_sha256(right)


@pytest.mark.parametrize("field_name", ["baselines", "stopping"])
def test_experiment_rejects_empty_or_duplicate_set_like_fields(field_name: str) -> None:
    values = {
        "protocol_ref": artifact("a", media_type=PROTOCOL_MANIFEST_MEDIA_TYPE),
        "seed_harness_ref": artifact("b"),
        "objective": "score",
        "baselines": ("static",),
        "stopping": ("budget",),
        "search_budget": BudgetPolicy(max_evaluations=1),
    }
    for invalid in ((), ("duplicate", "duplicate")):
        with pytest.raises(ValidationError):
            ExperimentManifest(**{**values, field_name: invalid})


def test_protocol_and_search_require_an_explicit_evaluation_ceiling() -> None:
    protocol_values = protocol(
        split(ProtocolPartition.EXPLORATION, "a"),
        split(ProtocolPartition.GATE, "b"),
    ).model_dump()
    protocol_values["budget"] = BudgetPolicy(max_tokens=1_000)
    with pytest.raises(ValidationError, match="protocol budget"):
        ProtocolManifest.model_validate(protocol_values)

    with pytest.raises(ValidationError, match="search_budget"):
        ExperimentManifest(
            protocol_ref=artifact("a", media_type=PROTOCOL_MANIFEST_MEDIA_TYPE),
            seed_harness_ref=artifact("b"),
            objective="score",
            baselines=("static",),
            stopping=("budget",),
            search_budget=BudgetPolicy(max_cost_usd=1.0),
        )


def test_candidate_manifest_binds_atomic_child_evidence_and_evaluation_plan() -> None:
    candidate = CandidateManifest(
        experiment_ref=artifact("a"),
        parent_harness_ref=artifact("b"),
        child_harness_ref=artifact("c"),
        mutation_ref=artifact("d"),
        evidence_refs=(artifact("f"), artifact("e")),
        evaluation_plan_ref=artifact("9"),
    )

    assert tuple(ref.sha256 for ref in candidate.evidence_refs) == ("e" * 64, "f" * 64)

    values = candidate.model_dump()
    values["evidence_refs"] = (artifact("e"), artifact("e"))
    with pytest.raises(ValidationError, match="duplicate"):
        CandidateManifest.model_validate(values)

    values = candidate.model_dump()
    values["child_harness_ref"] = values["parent_harness_ref"]
    with pytest.raises(ValidationError, match="must differ"):
        CandidateManifest.model_validate(values)


def test_manifest_references_fail_closed_on_non_json_types() -> None:
    with pytest.raises(ValidationError, match="manifest_ref"):
        ProtocolSplit(
            partition=ProtocolPartition.EXPLORATION,
            manifest_ref=artifact("a", media_type="text/plain"),
        )

    with pytest.raises(ValidationError, match="protocol_ref"):
        ExperimentManifest(
            protocol_ref=artifact("a", media_type="text/plain"),
            seed_harness_ref=artifact("b"),
            objective="score",
            baselines=("static",),
            stopping=("budget",),
            search_budget=BudgetPolicy(max_evaluations=1),
        )

    with pytest.raises(ValidationError, match="mutation_ref"):
        CandidateManifest(
            experiment_ref=artifact("a"),
            parent_harness_ref=artifact("b"),
            child_harness_ref=artifact("c"),
            mutation_ref=artifact("d", media_type="text/plain"),
            evidence_refs=(artifact("e", media_type="text/plain"),),
            evaluation_plan_ref=artifact("f"),
        )
