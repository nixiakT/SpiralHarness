from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection as subject
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ImmutableModel

SECRET_A = b"A" * 32
SECRET_B = b"B" * 32


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _FakeSplit(StrEnum):
    FIT = "fit"
    GATE = "gate"
    HOLDOUT = "holdout"


class _FakeRosterEntry(ImmutableModel):
    ordinal: int
    task_id: str
    split: _FakeSplit
    question_git_path: str
    question_blob_size: int
    question_blob_sha256: str
    row_size: int
    row_sha256: str
    candidate_payload_sha256: str


class _FakeManifest(ImmutableModel):
    roster: tuple[_FakeRosterEntry, ...]


class _FakeTask(ImmutableModel):
    question_json: str
    function_schemas_json: str


class _FakeV1LoadedBundle(ImmutableModel):
    manifest: _FakeManifest
    tasks: tuple[_FakeTask, ...]


class _FakeV2LoadedBundle(ImmutableModel):
    manifest: _FakeManifest
    tasks: tuple[_FakeTask, ...]


def _v1_loaded_bundle() -> _FakeV1LoadedBundle:
    roster = tuple(
        _FakeRosterEntry(
            ordinal=index,
            task_id=f"v1-public-task-SENTINEL-{index}",
            split=_FakeSplit.GATE if index % 2 else _FakeSplit.HOLDOUT,
            question_git_path=f"private/v1-path-SENTINEL-{index}.json",
            question_blob_size=1_000 + index,
            question_blob_sha256=_digest(f"v1-question-blob-SENTINEL-{index}"),
            row_size=200 + index,
            row_sha256=_digest(f"v1-source-row-SENTINEL-{index}"),
            candidate_payload_sha256=_digest(f"v1-candidate-payload-SENTINEL-{index}"),
        )
        for index in range(15)
    )
    tasks = tuple(
        _FakeTask(
            question_json=f'{{"question":"v1 public semantic content {index}"}}',
            function_schemas_json=f'[{{"name":"v1_semantic_tool_{index}"}}]',
        )
        for index in range(15)
    )
    return _FakeV1LoadedBundle(manifest=_FakeManifest(roster=roster), tasks=tasks)


def _v2_split(index: int) -> _FakeSplit:
    if index < 5:
        return _FakeSplit.FIT
    if index < 9:
        return _FakeSplit.GATE
    return _FakeSplit.HOLDOUT


def _v2_loaded_bundle() -> _FakeV2LoadedBundle:
    roster = tuple(
        _FakeRosterEntry(
            ordinal=index,
            task_id=f"v2-public-task-SENTINEL-{index}",
            split=_v2_split(index),
            question_git_path=f"private/v2-path-SENTINEL-{index}.json",
            question_blob_size=2_000 + index,
            question_blob_sha256=_digest(f"v2-question-blob-SENTINEL-{index}"),
            row_size=400 + index,
            row_sha256=_digest(f"v2-source-row-SENTINEL-{index}"),
            candidate_payload_sha256=_digest(f"v2-candidate-payload-SENTINEL-{index}"),
        )
        for index in range(25)
    )
    tasks = tuple(
        _FakeTask(
            question_json=f'{{"question":"v2 public semantic content {index}"}}',
            function_schemas_json=f'[{{"name":"v2_semantic_tool_{index}"}}]',
        )
        for index in range(25)
    )
    return _FakeV2LoadedBundle(manifest=_FakeManifest(roster=roster), tasks=tasks)


def _install_fake_loaded_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "BfclV4LoadedPublicPilot", _FakeV1LoadedBundle)
    monkeypatch.setattr(subject, "BfclV4LoadedPublicDevelopmentV2", _FakeV2LoadedBundle)


def _projection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    secret: bytes = SECRET_A,
    v1_loaded_bundle: object | None = None,
    v2_loaded_bundle: object | None = None,
) -> subject.BfclV4PublicV2SemanticProjection:
    _install_fake_loaded_contract(monkeypatch)
    return subject.build_bfcl_v4_public_v2_semantic_reviewer_projection(
        _v1_loaded_bundle() if v1_loaded_bundle is None else v1_loaded_bundle,
        _v2_loaded_bundle() if v2_loaded_bundle is None else v2_loaded_bundle,
        runtime_hmac_secret=secret,
    )


def _declaration() -> subject.BfclV4PublicV2SemanticReviewerDeclaration:
    return subject.BfclV4PublicV2SemanticReviewerDeclaration(
        declared_reviewer_kind=subject.BfclV4PublicV2SemanticReviewerKind.HUMAN,
        declared_principal_id="declared-reviewer-SENTINEL",
        human_identity_commitment_sha256=_digest("human-reviewer-SENTINEL"),
    )


def _valid_response(
    projection: subject.BfclV4PublicV2SemanticProjection,
) -> subject.BfclV4PublicV2SemanticReviewerResponse:
    declaration = _declaration()
    return subject.BfclV4PublicV2SemanticReviewerResponse(
        reviewer_packet_ref=projection.reviewer_packet.ref,
        reviewer_declaration_ref=declaration.ref,
        assignments=tuple(
            subject.BfclV4PublicV2SemanticReviewerAssignment(
                opaque_item_id=item.opaque_item_id,
                semantic_family_label=f"family-{index}",
                rationale="Grouped from only the visible canonical question and schemas.",
            )
            for index, item in enumerate(projection.reviewer_packet.items)
        ),
    )


def test_builder_revalidates_exact_loaded_bundle_type_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_loaded = _v1_loaded_bundle()
    v2_loaded = _v2_loaded_bundle()
    left = _projection(
        monkeypatch,
        v1_loaded_bundle=v1_loaded,
        v2_loaded_bundle=v2_loaded,
    )
    right = _projection(
        monkeypatch,
        v1_loaded_bundle=v1_loaded,
        v2_loaded_bundle=v2_loaded,
    )

    assert left == right
    assert left.fingerprint == right.fingerprint
    assert left.reviewer_packet.ref.sha256 == canonical_sha256(left.reviewer_packet)
    assert left.reviewer_packet.ref.size == len(canonical_json_bytes(left.reviewer_packet))
    assert left.trusted_mapping.reviewer_packet_ref == left.reviewer_packet.ref
    assert left.trusted_mapping.hmac_secret_present is False
    assert len(left.reviewer_packet.items) == 40
    assert len(left.trusted_mapping.entries) == 40
    assert left.both_loaded_bundle_contracts_strictly_revalidated
    assert left.trusted_mapping.source_universe_fingerprint == (
        subject._source_universe_fingerprint(
            left.trusted_mapping.v1_loaded_bundle_fingerprint,
            left.trusted_mapping.v2_loaded_bundle_fingerprint,
        )
    )


def test_builder_rejects_wrong_or_forged_loaded_bundle_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_loaded_contract(monkeypatch)

    v1_loaded = _v1_loaded_bundle()
    v2_loaded = _v2_loaded_bundle()
    invalid_pairs = (
        (object(), v2_loaded),
        (v1_loaded, object()),
        (None, v2_loaded),
        (v1_loaded, None),
        (v2_loaded, v1_loaded),
    )
    for v1_value, v2_value in invalid_pairs:
        with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
            subject.build_bfcl_v4_public_v2_semantic_reviewer_projection(
                v1_value,
                v2_value,
                runtime_hmac_secret=SECRET_A,
            )

    malformed_v1 = v1_loaded.model_copy(update={"tasks": "unchecked-forgery"})
    malformed_v2 = v2_loaded.model_copy(update={"tasks": "unchecked-forgery"})
    for v1_value, v2_value in (
        (malformed_v1, v2_loaded),
        (v1_loaded, malformed_v2),
    ):
        with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
            subject.build_bfcl_v4_public_v2_semantic_reviewer_projection(
                v1_value,
                v2_value,
                runtime_hmac_secret=SECRET_A,
            )


@pytest.mark.parametrize("secret", [b"short", "A" * 32, bytearray(b"A" * 32), None])
def test_runtime_hmac_secret_is_strict_bytes_with_minimum_length(
    monkeypatch: pytest.MonkeyPatch,
    secret: object,
) -> None:
    _install_fake_loaded_contract(monkeypatch)

    with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError, match="bytes secret"):
        subject.build_bfcl_v4_public_v2_semantic_reviewer_projection(
            _v1_loaded_bundle(),
            _v2_loaded_bundle(),
            runtime_hmac_secret=secret,  # type: ignore[arg-type]
        )


def test_different_secrets_change_opaque_ids_order_and_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _projection(monkeypatch, secret=SECRET_A)
    right = _projection(monkeypatch, secret=SECRET_B)
    left_mapping = left.trusted_mapping
    right_mapping = right.trusted_mapping

    assert tuple(item.opaque_item_id for item in left.reviewer_packet.items) != tuple(
        item.opaque_item_id for item in right.reviewer_packet.items
    )
    assert tuple(item.public_task_id for item in left_mapping.entries) != tuple(
        item.public_task_id for item in right_mapping.entries
    )
    assert left_mapping.mapping_commitment_hmac_sha256 != (
        right_mapping.mapping_commitment_hmac_sha256
    )
    source_order = (
        *((subject.BfclV4PublicV2SemanticPrivateBoundary.V1, index) for index in range(15)),
        *(
            (subject.BfclV4PublicV2SemanticPrivateBoundary(_v2_split(index).value), index)
            for index in range(25)
        ),
    )
    assert (
        tuple(
            (item.private_boundary_label, item.source_roster_ordinal)
            for item in left_mapping.entries
        )
        != source_order
    )


def test_reviewer_packet_has_only_opaque_ids_and_required_public_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    packet = projection.reviewer_packet
    encoded = packet.model_dump_json()

    assert all(
        set(item.model_dump())
        == {
            "opaque_item_id",
            "canonical_question_json",
            "canonical_function_schemas_json",
        }
        for item in packet.items
    )
    assert all(item.opaque_item_id.startswith("bfclv2-review-") for item in packet.items)
    for generation, count in (("v1", 15), ("v2", 25)):
        for index in range(count):
            forbidden_values = (
                f"{generation}-public-task-SENTINEL-{index}",
                f"private/{generation}-path-SENTINEL-{index}.json",
                _digest(f"{generation}-question-blob-SENTINEL-{index}"),
                _digest(f"{generation}-source-row-SENTINEL-{index}"),
                _digest(f"{generation}-candidate-payload-SENTINEL-{index}"),
            )
            assert all(value not in encoded for value in forbidden_values)
    assert SECRET_A.decode() not in encoded
    assert packet.direct_public_task_ids_present is False
    assert packet.source_paths_present is False
    assert packet.partition_labels_present is False
    assert packet.source_or_candidate_hashes_present is False
    assert packet.source_roster_order_present is False
    assert packet.hmac_secret_present is False
    assert packet.exact_public_content_may_be_reidentified
    assert packet.partition_inference_cryptographically_prevented is False


def test_private_mapping_retains_control_plane_coordinates_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _projection(monkeypatch).trusted_mapping
    encoded = mapping.model_dump_json()

    assert mapping.trusted_control_plane_only
    assert mapping.reviewer_facing is False
    assert mapping.public_task_ids_and_partitions_present
    assert mapping.source_coordinates_present
    assert "v1-public-task-SENTINEL-0" in encoded
    assert "v2-public-task-SENTINEL-0" in encoded
    assert "private/v1-path-SENTINEL-0.json" in encoded
    assert "private/v2-path-SENTINEL-0.json" in encoded
    boundary_counts = {
        boundary: sum(item.private_boundary_label is boundary for item in mapping.entries)
        for boundary in subject.BfclV4PublicV2SemanticPrivateBoundary
    }
    assert boundary_counts == {
        subject.BfclV4PublicV2SemanticPrivateBoundary.V1: 15,
        subject.BfclV4PublicV2SemanticPrivateBoundary.FIT: 5,
        subject.BfclV4PublicV2SemanticPrivateBoundary.GATE: 4,
        subject.BfclV4PublicV2SemanticPrivateBoundary.HOLDOUT: 16,
    }
    assert SECRET_A.decode() not in encoded
    assert mapping.hmac_secret_present is False


def test_noncanonical_question_or_nonlist_schemas_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_loaded_contract(monkeypatch)
    v1_loaded = _v1_loaded_bundle()
    v2_loaded = _v2_loaded_bundle()
    bad_question = v2_loaded.tasks[0].model_copy(
        update={"question_json": '{ "question": "not canonical" }'}
    )
    malformed_question = v2_loaded.model_copy(
        update={"tasks": (bad_question, *v2_loaded.tasks[1:])}
    )
    bad_schema = v2_loaded.tasks[0].model_copy(update={"function_schemas_json": "{}"})
    malformed_schema = v2_loaded.model_copy(update={"tasks": (bad_schema, *v2_loaded.tasks[1:])})

    for malformed in (malformed_question, malformed_schema):
        with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
            subject.build_bfcl_v4_public_v2_semantic_reviewer_projection(
                v1_loaded,
                malformed,
                runtime_hmac_secret=SECRET_A,
            )


@pytest.mark.parametrize("collision", ["public-task-id", "candidate-payload", "opaque-id"])
def test_cross_bundle_and_opaque_id_collisions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    v1_loaded = _v1_loaded_bundle()
    v2_loaded = _v2_loaded_bundle()
    if collision == "opaque-id":
        monkeypatch.setattr(
            subject,
            "_opaque_item_id",
            lambda _secret, _universe, _task_id: "bfclv2-review-" + "0" * 64,
        )
    else:
        first_v2 = v2_loaded.manifest.roster[0]
        update = (
            {"task_id": v1_loaded.manifest.roster[0].task_id}
            if collision == "public-task-id"
            else {
                "candidate_payload_sha256": (v1_loaded.manifest.roster[0].candidate_payload_sha256)
            }
        )
        changed = first_v2.model_copy(update=update)
        manifest = v2_loaded.manifest.model_copy(
            update={"roster": (changed, *v2_loaded.manifest.roster[1:])}
        )
        v2_loaded = v2_loaded.model_copy(update={"manifest": manifest})

    with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
        _projection(
            monkeypatch,
            v1_loaded_bundle=v1_loaded,
            v2_loaded_bundle=v2_loaded,
        )


def test_reviewer_declaration_requires_audit_compatible_provenance() -> None:
    with pytest.raises(ValidationError, match="identity commitment"):
        subject.BfclV4PublicV2SemanticReviewerDeclaration(
            declared_reviewer_kind=subject.BfclV4PublicV2SemanticReviewerKind.HUMAN,
            declared_principal_id="human-without-commitment",
        )
    with pytest.raises(ValidationError, match="complete model provenance"):
        subject.BfclV4PublicV2SemanticReviewerDeclaration(
            declared_reviewer_kind=(subject.BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED),
            declared_principal_id="incomplete-model-reviewer",
            model_provider="provider",
        )

    declaration = subject.BfclV4PublicV2SemanticReviewerDeclaration(
        declared_reviewer_kind=subject.BfclV4PublicV2SemanticReviewerKind.MODEL_ASSISTED,
        declared_principal_id="model-reviewer",
        model_provider="declared-provider",
        model_name="declared-model",
        model_revision="declared-revision",
        served_model_weights_identity_sha256=_digest("served-weights"),
        model_prompt_sha256=_digest("review-prompt"),
        model_generation_seed_u64=17,
    )

    assert declaration.human_identity_commitment_sha256 is None
    assert declaration.served_model_weights_identity_sha256 == _digest("served-weights")
    assert declaration.external_reviewer_identity_verified is False
    assert declaration.declaration_authenticated is False


def test_distribution_integrity_verification_remains_external_auth_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    receipt = subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        _declaration(),
        runtime_hmac_secret=SECRET_A,
    )

    assert receipt.status == "integrity-verified-external-authentication-blocked"
    assert receipt.reviewer_packet_integrity_verified
    assert receipt.trusted_mapping_integrity_verified
    assert receipt.mapping_commitment_hmac_verified
    assert receipt.opaque_id_hmac_derivation_verified
    assert receipt.packet_order_hmac_derivation_verified
    assert receipt.receipt_self_authenticating is False
    assert receipt.live_secret_reverification_required
    assert receipt.external_reviewer_identity_verified is False
    assert receipt.review_declaration_authenticated is False
    assert receipt.distribution_channel_authenticated is False
    assert receipt.packet_delivery_externally_verified is False
    assert receipt.reviewer_response_admissible is False
    assert receipt.semantic_audit_authority_issued is False
    assert receipt.score_bearing_execution_allowed is False

    assert (
        subject.verify_bfcl_v4_public_v2_semantic_distribution_receipt(
            receipt,
            projection.reviewer_packet,
            projection.trusted_mapping,
            _declaration(),
            runtime_hmac_secret=SECRET_A,
        )
        == receipt
    )


def test_wrong_secret_rejects_distribution_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch, secret=SECRET_A)

    with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
        subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
            projection.reviewer_packet,
            projection.trusted_mapping,
            _declaration(),
            runtime_hmac_secret=SECRET_B,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "packet-content",
        "packet-opaque-id",
        "mapping-task-id",
        "mapping-order-hmac",
        "mapping-content-hash",
        "mapping-commitment",
    ],
)
def test_packet_and_mapping_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    projection = _projection(monkeypatch)
    packet = projection.reviewer_packet
    mapping = projection.trusted_mapping
    if tamper == "packet-content":
        item = packet.items[0].model_copy(
            update={"canonical_question_json": '{"question":"tampered"}'}
        )
        packet = packet.model_copy(update={"items": (item, *packet.items[1:])})
    elif tamper == "packet-opaque-id":
        item = packet.items[0].model_copy(update={"opaque_item_id": "bfclv2-review-" + "0" * 64})
        packet = packet.model_copy(update={"items": (item, *packet.items[1:])})
    elif tamper == "mapping-commitment":
        mapping = mapping.model_copy(update={"mapping_commitment_hmac_sha256": "0" * 64})
    else:
        entry = mapping.entries[0]
        update: dict[str, object]
        if tamper == "mapping-task-id":
            update = {"public_task_id": "forged-public-task"}
        elif tamper == "mapping-order-hmac":
            update = {"order_hmac_sha256": "0" * 64}
        else:
            update = {"reviewer_content_sha256": "0" * 64}
        changed = entry.model_copy(update=update)
        mapping = mapping.model_copy(update={"entries": (changed, *mapping.entries[1:])})

    with pytest.raises((ValidationError, subject.BfclV4PublicV2SemanticProjectionError)):
        subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
            packet,
            mapping,
            _declaration(),
            runtime_hmac_secret=SECRET_A,
        )


def test_mapping_order_rewrite_with_valid_shape_still_fails_hmac_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    entries = projection.trusted_mapping.entries
    reordered = tuple(
        entry.model_copy(update={"packet_position": index})
        for index, entry in enumerate(reversed(entries))
    )
    mapping = projection.trusted_mapping.model_copy(update={"entries": reordered})
    packet = projection.reviewer_packet.model_copy(
        update={"items": tuple(reversed(projection.reviewer_packet.items))}
    )

    with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
        subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
            packet,
            mapping,
            _declaration(),
            runtime_hmac_secret=SECRET_A,
        )


def test_reviewer_response_has_exact_ordered_opaque_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    declaration = _declaration()
    response = _valid_response(projection)

    assert (
        subject.validate_bfcl_v4_public_v2_semantic_reviewer_response(
            projection.reviewer_packet,
            declaration,
            response,
        )
        == response
    )
    assert all(
        set(assignment.model_dump())
        == {"opaque_item_id", "semantic_family_label", "rationale", "abstained"}
        for assignment in response.assignments
    )

    for assignments in (
        response.assignments[:-1],
        tuple(reversed(response.assignments)),
    ):
        forged = response.model_copy(update={"assignments": assignments})
        with pytest.raises((ValidationError, subject.BfclV4PublicV2SemanticProjectionError)):
            subject.validate_bfcl_v4_public_v2_semantic_reviewer_response(
                projection.reviewer_packet,
                declaration,
                forged,
            )


def test_response_rejects_task_coordinates_and_authenticated_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    assignment_values = _valid_response(projection).assignments[0].model_dump()
    assignment_values["task_id"] = "forbidden-direct-task-id"

    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2SemanticReviewerAssignment.model_validate(
            assignment_values,
            strict=True,
        )
    forged = _valid_response(projection).model_copy(
        update={"reviewer_identity_externally_verified": True}
    )
    with pytest.raises(ValidationError):
        subject.BfclV4PublicV2SemanticReviewerResponse.model_validate(forged, strict=True)


def test_distribution_receipt_cannot_forge_external_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    receipt = subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        _declaration(),
        runtime_hmac_secret=SECRET_A,
    )

    for field_name in (
        "external_reviewer_identity_verified",
        "review_declaration_authenticated",
        "distribution_channel_authenticated",
        "packet_delivery_externally_verified",
        "reviewer_response_admissible",
        "semantic_audit_authority_issued",
        "score_bearing_execution_allowed",
    ):
        forged = receipt.model_copy(update={field_name: True})
        with pytest.raises(ValidationError):
            subject.BfclV4PublicV2SemanticDistributionReceipt.model_validate(
                forged,
                strict=True,
            )

    forged_ref = receipt.model_copy(update={"mapping_commitment_hmac_sha256": "0" * 64})
    with pytest.raises(subject.BfclV4PublicV2SemanticProjectionError):
        subject.verify_bfcl_v4_public_v2_semantic_distribution_receipt(
            forged_ref,
            projection.reviewer_packet,
            projection.trusted_mapping,
            _declaration(),
            runtime_hmac_secret=SECRET_A,
        )


def test_projection_score_guard_always_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    receipt = subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        _declaration(),
        runtime_hmac_secret=SECRET_A,
    )

    with pytest.raises(
        subject.BfclV4PublicV2SemanticScoreExecutionError,
        match="cannot authorize",
    ):
        subject.assert_bfcl_v4_public_v2_semantic_projection_score_execution_allowed(receipt)


def test_projection_is_provider_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invocation_trap(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("semantic packet materialization attempted network access")

    monkeypatch.setattr(socket, "socket", invocation_trap)
    monkeypatch.setattr(socket, "create_connection", invocation_trap)
    projection = _projection(monkeypatch)

    assert projection.network_used is False
    assert projection.model_invoked is False
    assert projection.possible_answers_read is False
    assert projection.scores_read is False
    assert projection.runs_or_cas_read is False


def test_import_graph_is_isolated_from_roster_audit_runner_and_provider() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import sys
import spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection
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
    assert {
        "spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection",
        "spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection_contracts",
        "spiral_harness.benchmark.bfcl_v4_public_pilot_contracts",
        "spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts",
        "spiral_harness.core.canonical",
        "spiral_harness.core.models",
    } <= imported
    assert {
        "spiral_harness.benchmark.bfcl_v4_public_pilot",
        "spiral_harness.benchmark.bfcl_v4_public_development_v2",
        "spiral_harness.benchmark.bfcl_v4_public_development_v2_semantic_audit",
        "spiral_harness.benchmark.bfcl_v4_public_runner",
        "spiral_harness.benchmark.bfcl_v4_public_grader",
    }.isdisjoint(imported)
    assert not any(
        name.startswith(("spiral_harness.execution", "spiral_harness.providers"))
        for name in imported
    )


def test_projection_modules_obey_architecture_size_and_import_rules() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    module_paths = (
        repository_root / "spiral_harness/benchmark/bfcl_v4_public_v2_semantic_projection.py",
        repository_root
        / "spiral_harness/benchmark/bfcl_v4_public_v2_semantic_projection_contracts.py",
    )

    for module_path in module_paths:
        source = module_path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 700
        assert "import importlib" not in source
        assert "importlib.import_module" not in source


def test_projection_payloads_do_not_contain_the_runtime_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(monkeypatch)
    declaration = _declaration()
    receipt = subject.verify_bfcl_v4_public_v2_semantic_distribution_integrity(
        projection.reviewer_packet,
        projection.trusted_mapping,
        declaration,
        runtime_hmac_secret=SECRET_A,
    )
    serialized = json.dumps(
        {
            "projection": projection.model_dump(mode="json"),
            "declaration": declaration.model_dump(mode="json"),
            "receipt": receipt.model_dump(mode="json"),
        },
        sort_keys=True,
    )

    assert SECRET_A.decode() not in serialized
