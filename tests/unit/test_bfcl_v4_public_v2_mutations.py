from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_v2_mutations as subject
from spiral_harness.core.canonical import canonical_json, canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ArtifactRef

CATALOGUE_SHA256 = "a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5"
PARENT_SHA256 = "318c9238725c182851ac45fac85101f482a7e4744013ab7f9cfeb0a7fa954d11"

EXPECTED_IDS = (
    subject.BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER,
    subject.BfclV4PublicV2MutationId.COMPOUND_CLAUSE_TOOL_COVERAGE,
    subject.BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR,
    subject.BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER,
    subject.BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER,
)

EXPECTED_TARGET_OUTPUTS = {
    subject.BfclV4PublicV2MutationId.NUMERIC_SCHEMA_LEXICALIZER: (
        '{"arguments_json":"{\\"ratio\\":7.0}"}'
    ),
    subject.BfclV4PublicV2MutationId.COMPOUND_CLAUSE_TOOL_COVERAGE: (
        '{"calls":["lookup_alpha","lookup_beta"]}'
    ),
    subject.BfclV4PublicV2MutationId.REQUIRED_ARGUMENT_VALIDATOR: (
        '{"emit":false,"missing":["unit"]}'
    ),
    subject.BfclV4PublicV2MutationId.MULTIPLICITY_ORDER_PRESERVER: (
        '{"calls":["beta","alpha","beta"]}'
    ),
    subject.BfclV4PublicV2MutationId.SCHEMA_GROUNDED_MATCHER: ('{"selected":"forecast"}'),
}


def _proposal(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> subject.BfclV4PublicV2MutationProposal:
    return subject.BfclV4PublicV2MutationProposal(catalogue_id=mutation_id)


def _materialization(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> subject.BfclV4PublicV2MutationMaterialization:
    return subject.materialize_bfcl_v4_public_v2_mutation(_proposal(mutation_id))


def _different_ref(ref: ArtifactRef) -> ArtifactRef:
    replacement_sha = "0" * 64 if ref.sha256 != "0" * 64 else "1" * 64
    return ref.model_copy(update={"sha256": replacement_sha})


def test_catalogue_is_the_frozen_five_entry_atomic_grammar() -> None:
    catalogue = subject.BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE

    assert tuple(entry.mutation_id for entry in catalogue.entries) == EXPECTED_IDS
    assert len({entry.component for entry in catalogue.entries}) == 5
    assert len({entry.operator for entry in catalogue.entries}) == 5
    assert all(entry.atomic_edit_count == 1 for entry in catalogue.entries)
    assert all(not entry.executable_free_form_text_present for entry in catalogue.entries)
    assert all(not entry.task_coordinates_present for entry in catalogue.entries)
    assert all(not entry.answer_data_present for entry in catalogue.entries)
    assert catalogue.transfer_source == "bfcl-v1-public-development-failure-analysis"
    assert catalogue.fingerprint == CATALOGUE_SHA256
    assert catalogue.ref.sha256 == CATALOGUE_SHA256


def test_parent_harness_is_content_addressed_and_frozen() -> None:
    parent = subject.BFCL_V4_PUBLIC_V2_PARENT_HARNESS

    assert parent.fingerprint == PARENT_SHA256
    assert parent.ref.sha256 == canonical_sha256(parent)
    assert parent.ref.size == len(canonical_json_bytes(parent))
    assert parent.ref.media_type == subject.BFCL_V4_PUBLIC_V2_HARNESS_MEDIA_TYPE
    assert parent.parent_ref is None
    assert parent.selected_catalogue_id is None
    assert tuple(stage.component for stage in parent.stages) == tuple(
        subject.BfclV4PublicV2MutableComponent
    )


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_materialization_is_deterministic_and_changes_exactly_one_stage(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> None:
    left = _materialization(mutation_id)
    right = _materialization(mutation_id)

    assert left == right
    assert canonical_json(left) == canonical_json(right)
    assert left.fingerprint == right.fingerprint
    assert left.parent_ref == subject.BFCL_V4_PUBLIC_V2_PARENT_HARNESS.ref
    assert left.candidate.parent_ref == left.parent_ref
    assert left.candidate_ref == left.candidate.ref
    assert left.candidate_ref != left.parent_ref
    differences = [
        (parent_stage, candidate_stage)
        for parent_stage, candidate_stage in zip(
            left.parent.stages,
            left.candidate.stages,
            strict=True,
        )
        if parent_stage != candidate_stage
    ]
    assert len(differences) == 1
    assert differences[0][1].component == left.entry.component
    assert differences[0][1].implementation == left.entry.operator.value


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_each_synthetic_canary_changes_target_and_preserves_protected_behavior(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> None:
    materialization = _materialization(mutation_id)
    target, protected = materialization.canary.observations

    assert not target.protected_case
    assert target.behavior_changed
    assert target.parent_output_json != target.candidate_output_json
    assert target.candidate_output_json == EXPECTED_TARGET_OUTPUTS[mutation_id]
    assert protected.protected_case
    assert not protected.behavior_changed
    assert protected.parent_output_json == protected.candidate_output_json
    assert materialization.canary.local_operator_activation_verified
    assert materialization.canary.deterministic_operator_adherence_verified
    assert materialization.canary.synthetic_behavior_change_verified
    assert materialization.canary.protected_behavior_verified


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_materializer_is_provider_free_and_never_attests_model_capability(
    mutation_id: subject.BfclV4PublicV2MutationId,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invocation_trap(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider-free materialization attempted network access")

    monkeypatch.setattr(socket, "socket", invocation_trap)
    monkeypatch.setattr(socket, "create_connection", invocation_trap)

    materialization = _materialization(mutation_id)

    assert materialization.provider_calls == 0
    assert materialization.canary.provider_calls == 0
    assert not materialization.canary.model_invoked
    assert not materialization.canary.model_behavior_attested
    assert not materialization.canary.benchmark_capability_evidence
    assert not materialization.v2_question_or_score_observation_used
    assert not materialization.score_bearing_execution_allowed


@pytest.mark.parametrize(
    "extra",
    [
        {"appendix": "free-form mutation"},
        {"task_id": "synthetic-task-coordinate"},
        {"answer": {"tool": "candidate-visible-answer"}},
        {"free_form_appendix_present": True},
        {"task_coordinates_present": True},
        {"answer_data_present": True},
        {"score_execution_requested": True},
        {"selected_entry_count": 2},
    ],
)
def test_proposal_rejects_text_coordinates_answers_scores_and_cardinality(
    extra: dict[str, object],
) -> None:
    values: dict[str, object] = {"catalogue_id": EXPECTED_IDS[0], **extra}

    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2MutationProposal.model_validate(values, strict=True)


@pytest.mark.parametrize(
    "catalogue_id",
    [
        "unknown-mutation",
        [EXPECTED_IDS[0]],
        [EXPECTED_IDS[0], EXPECTED_IDS[1]],
        None,
    ],
)
def test_proposal_rejects_unknown_or_multiple_catalogue_ids(catalogue_id: object) -> None:
    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2MutationProposal.model_validate(
            {"catalogue_id": catalogue_id},
            strict=True,
        )


def test_catalogue_rejects_reordering_duplicates_and_wrong_operator() -> None:
    catalogue = subject.BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE
    reversed_catalogue = catalogue.model_copy(
        update={"entries": tuple(reversed(catalogue.entries))}
    )
    duplicate_catalogue = catalogue.model_copy(
        update={"entries": (catalogue.entries[0],) * len(catalogue.entries)}
    )
    wrong_entry = catalogue.entries[0].model_copy(
        update={"operator": catalogue.entries[1].operator}
    )

    with pytest.raises(ValidationError, match="frozen order"):
        subject.BfclV4PublicV2MutationCatalogue.model_validate(
            reversed_catalogue,
            strict=True,
        )
    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2MutationCatalogue.model_validate(
            duplicate_catalogue,
            strict=True,
        )
    with pytest.raises(ValidationError, match="closed catalogue specification"):
        subject.BfclV4PublicV2MutationCatalogueEntry.model_validate(
            wrong_entry,
            strict=True,
        )


def test_forged_noop_wrong_stage_operator_and_parent_are_rejected() -> None:
    materialization = _materialization(EXPECTED_IDS[0])
    candidate = materialization.candidate
    wrong_stage = candidate.stages[1].model_copy(update={"implementation": "foreign-edit/v1"})
    wrong_stage_candidate = candidate.model_copy(
        update={"stages": (candidate.stages[0], wrong_stage, *candidate.stages[2:])}
    )
    wrong_operator_stage = candidate.stages[0].model_copy(
        update={"implementation": subject.BfclV4PublicV2Operator.REQUIRE_EVERY_CLAUSE_COVERED}
    )
    wrong_operator_candidate = candidate.model_copy(
        update={"stages": (wrong_operator_stage, *candidate.stages[1:])}
    )
    forged_candidates = (
        candidate.model_copy(update={"stages": materialization.parent.stages}),
        wrong_stage_candidate,
        wrong_operator_candidate,
        candidate.model_copy(update={"parent_ref": _different_ref(materialization.parent_ref)}),
    )

    for forged in forged_candidates:
        with pytest.raises(ValidationError):
            subject.BfclV4PublicV2HarnessArtifact.model_validate(forged, strict=True)


def test_model_construct_and_model_copy_cannot_bypass_publication_validation() -> None:
    materialization = _materialization(EXPECTED_IDS[0])
    values = materialization.candidate.model_dump(mode="python")
    values["stages"] = materialization.parent.stages
    constructed_noop = subject.BfclV4PublicV2HarnessArtifact.model_construct(**values)
    copied_bad_proposal = materialization.proposal.model_copy(
        update={"catalogue_id": "not-an-enum"}
    )

    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2HarnessArtifact.model_validate(
            constructed_noop,
            strict=True,
        )
    with pytest.raises(ValidationError):
        subject.materialize_bfcl_v4_public_v2_mutation(copied_bad_proposal)


def test_wrong_materialization_refs_and_forged_canary_are_rejected() -> None:
    materialization = _materialization(EXPECTED_IDS[0])
    wrong_candidate_ref = materialization.model_copy(
        update={"candidate_ref": materialization.parent_ref}
    )
    wrong_parent_ref = materialization.model_copy(
        update={"parent_ref": _different_ref(materialization.parent_ref)}
    )
    reversed_observations = tuple(reversed(materialization.canary.observations))
    wrong_canary_observations = materialization.canary.model_copy(
        update={"observations": reversed_observations}
    )
    wrong_canary_ref = materialization.canary.model_copy(
        update={"candidate_ref": materialization.parent_ref}
    )

    for forged in (wrong_candidate_ref, wrong_parent_ref):
        with pytest.raises(ValidationError):
            subject.BfclV4PublicV2MutationMaterialization.model_validate(
                forged,
                strict=True,
            )
    for forged_canary in (wrong_canary_observations, wrong_canary_ref):
        with pytest.raises(ValidationError):
            subject.BfclV4PublicV2CanaryReceipt.model_validate(
                forged_canary,
                strict=True,
            )


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_canary_verifier_replays_exact_receipt(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> None:
    materialization = _materialization(mutation_id)

    assert (
        subject.verify_bfcl_v4_public_v2_mutation_canary(materialization) == materialization.canary
    )


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_score_execution_guard_always_rejects(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> None:
    with pytest.raises(
        subject.BfclV4PublicV2MutationExecutionError,
        match="cannot authorize scoring",
    ):
        subject.assert_bfcl_v4_public_v2_score_execution_allowed(_materialization(mutation_id))


def test_isolated_import_graph_contains_no_bfcl_data_runner_or_provider_module() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import sys
import spiral_harness.benchmark.bfcl_v4_public_v2_mutations
print("\\n".join(sorted(name for name in sys.modules if name.startswith("spiral_harness"))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = set(completed.stdout.splitlines())

    assert imported == {
        "spiral_harness",
        "spiral_harness.benchmark",
        "spiral_harness.benchmark.bfcl_v4_public_v2_mutations",
        "spiral_harness.core",
        "spiral_harness.core.canonical",
        "spiral_harness.core.models",
    }


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_canary_payload_contains_only_generic_synthetic_cases(
    mutation_id: subject.BfclV4PublicV2MutationId,
) -> None:
    canary = _materialization(mutation_id).canary
    payload = canary.model_dump(mode="json")

    assert all(item.case_id.startswith("synthetic-") for item in canary.observations)
    assert all("task_id" not in json.loads(item.parent_output_json) for item in canary.observations)
    assert all(
        "task_id" not in json.loads(item.candidate_output_json) for item in canary.observations
    )
    assert "candidate-visible" not in json.dumps(payload, sort_keys=True)
    assert "question_json" not in payload
    assert payload["benchmark_items_present"] is False
    assert payload["answer_data_present"] is False
