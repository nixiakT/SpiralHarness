from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_terminal_decision import build_graph

from spiral_harness.core.models import ArtifactRef
from spiral_harness.verification.mechanism import (
    AttestedMechanismEvidence,
    MechanismEvidenceAttestationError,
    MechanismEvidenceContent,
    MechanismEvidenceVerificationCapability,
    TrustedMechanismEvidenceService,
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
