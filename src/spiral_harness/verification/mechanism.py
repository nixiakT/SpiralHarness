"""Producer-attested mechanism evidence for preregistered candidate probes.

An optimizer may nominate evidence, but it may not author the boolean that lets
its own candidate enter the gate.  A trusted probe producer signs the complete
mechanism-evidence envelope with a process-local capability.  Controllers hold
only the paired verification capability and rejoin the signed envelope with the
frozen protocol, candidate, exploration split, execution context, and source
artifacts before advancing the lifecycle.

This is an in-process API capability, not an OS security boundary or a durable
signature.  The signing service must remain outside candidate code.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import ProtocolManifest, ProtocolPartition
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.verification.artifacts import GateBatchExecutionContext
from spiral_harness.verification.models import MechanismEvidence

ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.attested-mechanism-evidence.v1+json"
)

_ATTESTATION_DOMAIN = b"spiral-harness:mechanism-evidence-attestation:v1\x00"
_ATTESTOR_ID_DOMAIN = b"spiral-harness:mechanism-evidence-attestor-id:v1\x00"


class MechanismEvidenceAttestationError(ValueError):
    """Raised when mechanism evidence is not authentic under the frozen producer."""


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


class MechanismEvidenceContent(ImmutableModel):
    """Complete unsigned probe result asserted by one trusted producer."""

    schema_version: Literal["1"] = "1"
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    exploration_split_ref: ArtifactRef
    task_set_fingerprint: Sha256
    execution_context: GateBatchExecutionContext
    source_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    evidence: MechanismEvidence

    @field_validator("source_refs")
    @classmethod
    def canonicalize_source_refs(
        cls,
        refs: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(refs, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if len({ref.sha256 for ref in ordered}) != len(ordered):
            raise ValueError("source_refs must not contain duplicate artifacts")
        return ordered

    @model_validator(mode="after")
    def bind_evidence_to_signed_context(self) -> MechanismEvidenceContent:
        for field_name in (
            "protocol_ref",
            "candidate_ref",
            "candidate_harness_ref",
            "exploration_split_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)
        if self.task_set_fingerprint != self.exploration_split_ref.sha256:
            raise ValueError("task_set_fingerprint must identify the exploration split")
        if self.evidence.candidate_harness_id != self.candidate_harness_ref.sha256:
            raise ValueError("mechanism evidence must identify the candidate harness")

        source_digests = {ref.sha256 for ref in self.source_refs}
        cited_digests = {
            evidence_ref for check in self.evidence.checks for evidence_ref in check.evidence_refs
        }
        unknown = tuple(sorted(cited_digests.difference(source_digests)))
        if unknown:
            raise ValueError(
                "mechanism checks cite artifacts outside source_refs: " + ", ".join(unknown)
            )
        return self


class AttestedMechanismEvidence(MechanismEvidenceContent):
    """One candidate-bound mechanism result with trusted-producer attestation."""

    attestor_id: Sha256
    attestation_sha256: Sha256

    @property
    def content(self) -> MechanismEvidenceContent:
        """Return the exact unsigned value covered by the HMAC."""

        return MechanismEvidenceContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"attestor_id", "attestation_sha256"},
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )


def _attestor_id(secret: bytes) -> str:
    return hashlib.sha256(_ATTESTOR_ID_DOMAIN + secret).hexdigest()


def _attestation(secret: bytes, content: MechanismEvidenceContent, attestor_id: str) -> str:
    return hmac.new(
        secret,
        _ATTESTATION_DOMAIN + attestor_id.encode("ascii") + b"\x00" + canonical_json_bytes(content),
        hashlib.sha256,
    ).hexdigest()


class MechanismEvidenceVerificationCapability:
    """Verify one producer's envelopes without exposing a public signing API."""

    __slots__ = ("__secret", "_attestor_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("mechanism verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._attestor_id = _attestor_id(secret)

    @property
    def attestor_id(self) -> str:
        """Stable identifier for this ephemeral verification capability."""

        return self._attestor_id

    def verify(
        self,
        value: AttestedMechanismEvidence | object,
    ) -> AttestedMechanismEvidence:
        """Validate schema, producer identity, and the complete signed content."""

        try:
            validated = AttestedMechanismEvidence.model_validate(value)
        except (TypeError, ValueError) as exc:
            raise MechanismEvidenceAttestationError(
                f"malformed attested mechanism evidence: {exc}"
            ) from exc
        if validated.attestor_id != self._attestor_id:
            raise MechanismEvidenceAttestationError(
                "mechanism evidence was signed by an untrusted attestor"
            )
        expected = _attestation(self.__secret, validated.content, self._attestor_id)
        if not hmac.compare_digest(validated.attestation_sha256, expected):
            raise MechanismEvidenceAttestationError(
                "mechanism evidence HMAC attestation is invalid"
            )
        return validated


class TrustedMechanismEvidenceService:
    """Trusted probe-producer capability that signs candidate mechanism evidence."""

    __slots__ = ("__secret", "_verification_capability")

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)
        self._verification_capability = MechanismEvidenceVerificationCapability(self.__secret)

    @property
    def verification_capability(self) -> MechanismEvidenceVerificationCapability:
        """Return the paired verification capability for controllers and graders."""

        return self._verification_capability

    @property
    def attestor_id(self) -> str:
        """Return the ID that must be frozen into ``ProtocolManifest``."""

        return self._verification_capability.attestor_id

    def issue(self, content: MechanismEvidenceContent) -> AttestedMechanismEvidence:
        """Sign one exact, already validated trusted-producer assertion."""

        validated = MechanismEvidenceContent.model_validate(content)
        fields = validated.model_dump(mode="python", round_trip=True, warnings="none")
        return AttestedMechanismEvidence(
            **fields,
            attestor_id=self.attestor_id,
            attestation_sha256=_attestation(self.__secret, validated, self.attestor_id),
        )

    def create(
        self,
        *,
        protocol_ref: ArtifactRef,
        protocol: ProtocolManifest,
        candidate_ref: ArtifactRef,
        candidate_harness_ref: ArtifactRef,
        source_refs: Sequence[ArtifactRef],
        evidence: MechanismEvidence,
    ) -> AttestedMechanismEvidence:
        """Build and sign evidence under the exact frozen exploration context."""

        validated_protocol = ProtocolManifest.model_validate(protocol)
        if validated_protocol.mechanism_evidence_attestor_id != self.attestor_id:
            raise MechanismEvidenceAttestationError(
                "mechanism signer does not match the protocol-frozen attestor"
            )
        exploration_split_ref = next(
            split.manifest_ref
            for split in validated_protocol.splits
            if split.partition is ProtocolPartition.EXPLORATION
        )
        return self.issue(
            MechanismEvidenceContent(
                protocol_ref=protocol_ref,
                candidate_ref=candidate_ref,
                candidate_harness_ref=candidate_harness_ref,
                exploration_split_ref=exploration_split_ref,
                task_set_fingerprint=exploration_split_ref.sha256,
                execution_context=GateBatchExecutionContext.from_protocol(validated_protocol),
                source_refs=tuple(source_refs),
                evidence=evidence,
            )
        )


__all__ = [
    "ATTESTED_MECHANISM_EVIDENCE_MEDIA_TYPE",
    "AttestedMechanismEvidence",
    "MechanismEvidenceAttestationError",
    "MechanismEvidenceContent",
    "MechanismEvidenceVerificationCapability",
    "TrustedMechanismEvidenceService",
]
