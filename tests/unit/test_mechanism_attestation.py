from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_terminal_decision import build_graph

from spiral_harness.core.models import ArtifactRef
from spiral_harness.verification.mechanism import (
    RESERVED_SKILL_MECHANISM_IDS,
    SKILL_ADHERENCE_MECHANISM_ID,
    SKILL_BEHAVIOR_MECHANISM_ID,
    SKILL_REQUEST_ACTIVATION_MECHANISM_ID,
    AttestedMechanismEvidence,
    MechanismEvidenceAttestationError,
    MechanismEvidenceContent,
    MechanismEvidenceVerificationCapability,
    TrustedMechanismEvidenceService,
)
from spiral_harness.verification.models import MechanismCheck, MechanismEvidence

_RESERVED_NAME_VARIANTS = (
    *sorted(RESERVED_SKILL_MECHANISM_IDS),
    SKILL_REQUEST_ACTIVATION_MECHANISM_ID.upper(),
    f" {SKILL_ADHERENCE_MECHANISM_ID} ",
    f"\t{SKILL_BEHAVIOR_MECHANISM_ID.title()}\n",
)


def content_with_check(graph, *, check_name: str, passed: bool | None):
    envelope = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    source_ref = envelope.source_refs[0]
    evidence = MechanismEvidence(
        candidate_harness_id=graph.candidate.child_harness_ref.sha256,
        checks=(
            MechanismCheck(
                name=check_name,
                passed=passed,
                evidence_refs=(source_ref.sha256,) if passed is True else (),
            ),
        ),
    )
    return MechanismEvidenceContent.model_validate(
        {
            **envelope.content.model_dump(mode="python"),
            "evidence": evidence,
        }
    )


def create_from_content(graph, content: MechanismEvidenceContent):
    return graph.mechanism_evidence_service.create(
        protocol_ref=graph.protocol_ref,
        protocol=graph.protocol,
        candidate_ref=graph.candidate_ref,
        candidate_harness_ref=graph.candidate.child_harness_ref,
        source_refs=content.source_refs,
        evidence=content.evidence,
    )


def test_mechanism_content_canonicalizes_sources_and_rejects_duplicates(tmp_path) -> None:
    graph = build_graph(tmp_path)
    envelope = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    values = envelope.content.model_dump(mode="python")
    reversed_content = MechanismEvidenceContent.model_validate(
        {**values, "source_refs": tuple(reversed(envelope.source_refs))}
    )

    assert reversed_content.source_refs == envelope.source_refs
    with pytest.raises(ValidationError, match="duplicate artifacts"):
        MechanismEvidenceContent.model_validate(
            {**values, "source_refs": (envelope.source_refs[0], envelope.source_refs[0])}
        )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"task_set_fingerprint": "0" * 64}, "exploration split"),
        (
            {
                "candidate_harness_ref": ArtifactRef(
                    sha256="1" * 64,
                    size=1,
                    media_type="application/json",
                )
            },
            "candidate harness",
        ),
        (
            {
                "candidate_ref": ArtifactRef(
                    sha256="2" * 64,
                    size=1,
                    media_type="text/plain",
                )
            },
            "JSON media type",
        ),
    ],
)
def test_mechanism_content_rejects_context_inconsistency(tmp_path, updates, error) -> None:
    graph = build_graph(tmp_path)
    envelope = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )
    values = envelope.content.model_dump(mode="python")

    with pytest.raises(ValidationError, match=error):
        MechanismEvidenceContent.model_validate({**values, **updates})


def test_mechanism_verifier_rejects_short_secret_and_malformed_envelope(tmp_path) -> None:
    graph = build_graph(tmp_path)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        MechanismEvidenceVerificationCapability(b"short")
    with pytest.raises(MechanismEvidenceAttestationError, match="malformed"):
        graph.mechanism_evidence_service.verification_capability.verify({"schema_version": "1"})


def test_mechanism_service_must_match_protocol_frozen_producer(tmp_path) -> None:
    graph = build_graph(tmp_path)
    envelope = graph.store.get_json(
        graph.mechanism_evidence_ref,
        AttestedMechanismEvidence,
    )

    with pytest.raises(MechanismEvidenceAttestationError, match="protocol-frozen attestor"):
        TrustedMechanismEvidenceService().create(
            protocol_ref=graph.protocol_ref,
            protocol=graph.protocol,
            candidate_ref=graph.candidate_ref,
            candidate_harness_ref=graph.candidate.child_harness_ref,
            source_refs=envelope.source_refs,
            evidence=graph.mechanism_evidence,
        )


@pytest.mark.parametrize("passed", [True, False, None])
@pytest.mark.parametrize("check_name", _RESERVED_NAME_VARIANTS)
def test_public_issue_rejects_all_reserved_skill_mechanism_variants(
    tmp_path,
    check_name,
    passed,
) -> None:
    graph = build_graph(tmp_path)
    content = content_with_check(graph, check_name=check_name, passed=passed)

    with pytest.raises(MechanismEvidenceAttestationError, match="reserved skill mechanism"):
        graph.mechanism_evidence_service.issue(content)


@pytest.mark.parametrize("passed", [True, False, None])
@pytest.mark.parametrize("check_name", _RESERVED_NAME_VARIANTS)
def test_public_create_rejects_all_reserved_skill_mechanism_variants(
    tmp_path,
    check_name,
    passed,
) -> None:
    graph = build_graph(tmp_path)
    content = content_with_check(graph, check_name=check_name, passed=passed)

    with pytest.raises(MechanismEvidenceAttestationError, match="reserved skill mechanism"):
        create_from_content(graph, content)


@pytest.mark.parametrize("check_name", ["activation", "adherence", "behavior"])
def test_generic_mechanism_names_remain_signable(tmp_path, check_name) -> None:
    graph = build_graph(tmp_path)
    content = content_with_check(graph, check_name=check_name, passed=True)
    service = graph.mechanism_evidence_service

    issued = service.issue(content)
    created = create_from_content(graph, content)

    assert service.verification_capability.verify(issued) == issued
    assert service.verification_capability.verify(created) == created
