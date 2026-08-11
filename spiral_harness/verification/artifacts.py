"""Producer-attested evidence envelopes consumed by the trusted gate.

Raw observation arrays are deliberately not terminal-decision evidence.  A
trusted producer signs the exact batch content with an ephemeral HMAC key.  A
controller or terminal verifier must possess the paired verify-only capability
and must rejoin the signed execution context with the frozen protocol before
the promotion gate is allowed to consume any score.

This is a single-process capability boundary, not OS isolation.  The signer
must never be passed to candidate code, and durable replay after process loss
requires a future trusted key service or asymmetric attestation design.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.core.experiment import ProtocolManifest
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.verification.models import TrialObservation

GATE_TRIAL_BATCH_MEDIA_TYPE = "application/vnd.spiral-harness.gate-trial-batch.v2+json"

_GATE_BATCH_ATTESTATION_DOMAIN = b"spiral-harness:gate-trial-batch-attestation:v2\x00"
_GATE_BATCH_ATTESTOR_ID_DOMAIN = b"spiral-harness:gate-trial-batch-attestor-id:v2\x00"


def _require_json_ref(ref: ArtifactRef, *, field_name: str) -> None:
    media_type = ref.media_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise ValueError(f"{field_name} must declare a JSON media type")


class GateTrialArm(StrEnum):
    """The frozen harness side represented by a gate trial batch."""

    PARENT = "parent"
    CANDIDATE = "candidate"


class GateBatchAttestationError(ValueError):
    """Raised when a gate batch is not authentic under the frozen producer."""


class GateBatchExecutionContext(ImmutableModel):
    """Protocol-owned execution context repeated inside the signed payload."""

    benchmark_fingerprint: NonEmptyStr
    model_fingerprint: NonEmptyStr
    inference_fingerprint: NonEmptyStr
    runtime_fingerprint: NonEmptyStr
    sandbox_fingerprint: NonEmptyStr
    grader_fingerprint: NonEmptyStr
    capability_policy_ref: ArtifactRef
    trusted_plane_version: NonEmptyStr

    @model_validator(mode="after")
    def capability_policy_is_json(self) -> GateBatchExecutionContext:
        _require_json_ref(self.capability_policy_ref, field_name="capability_policy_ref")
        return self

    @classmethod
    def from_protocol(cls, protocol: ProtocolManifest) -> GateBatchExecutionContext:
        """Copy the exact score-affecting execution context from a frozen protocol."""

        validated = ProtocolManifest.model_validate(protocol)
        return cls(
            benchmark_fingerprint=validated.benchmark_fingerprint,
            model_fingerprint=validated.model_fingerprint,
            inference_fingerprint=validated.inference_fingerprint,
            runtime_fingerprint=validated.runtime_fingerprint,
            sandbox_fingerprint=validated.sandbox_fingerprint,
            grader_fingerprint=validated.grader_fingerprint,
            capability_policy_ref=validated.capability_policy_ref,
            trusted_plane_version=validated.trusted_plane_version,
        )


class GateTrialBatchContent(ImmutableModel):
    """The complete unsigned content that one trusted producer attests."""

    schema_version: Literal["2"] = "2"
    protocol_ref: ArtifactRef
    candidate_ref: ArtifactRef
    arm: GateTrialArm
    harness_ref: ArtifactRef
    gate_split_ref: ArtifactRef
    task_set_fingerprint: Sha256
    mechanism_evidence_ref: ArtifactRef
    execution_context: GateBatchExecutionContext
    source_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    observations: Annotated[tuple[TrialObservation, ...], Field(min_length=1)]

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
    def bind_observations_to_batch(self) -> GateTrialBatchContent:
        for field_name in (
            "protocol_ref",
            "candidate_ref",
            "harness_ref",
            "gate_split_ref",
            "mechanism_evidence_ref",
        ):
            _require_json_ref(getattr(self, field_name), field_name=field_name)

        mismatched = tuple(
            f"{observation.task_id}::seed={observation.seed}"
            for observation in self.observations
            if observation.harness_id != self.harness_ref.sha256
        )
        if mismatched:
            raise ValueError(
                "observations must belong to harness_ref; mismatched pairs: "
                + ", ".join(mismatched)
            )

        pairs = tuple((observation.task_id, observation.seed) for observation in self.observations)
        if len(pairs) != len(set(pairs)):
            raise ValueError("observations must contain unique task_id/seed pairs")
        return self


class GateTrialBatch(GateTrialBatchContent):
    """One producer-attested, candidate-bound arm of gate observations."""

    attestor_id: Sha256
    attestation_sha256: Sha256

    @property
    def content(self) -> GateTrialBatchContent:
        """Return the exact unsigned content covered by the HMAC."""

        return GateTrialBatchContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"attestor_id", "attestation_sha256"},
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )


def _attestor_id(secret: bytes) -> str:
    return hashlib.sha256(_GATE_BATCH_ATTESTOR_ID_DOMAIN + secret).hexdigest()


def _attestation(secret: bytes, content: GateTrialBatchContent, attestor_id: str) -> str:
    payload = canonical_json_bytes(content)
    return hmac.new(
        secret,
        _GATE_BATCH_ATTESTATION_DOMAIN + attestor_id.encode("ascii") + b"\x00" + payload,
        hashlib.sha256,
    ).hexdigest()


class GateBatchVerificationCapability:
    """Verify batches from one signer without exposing a public signing API.

    The object still contains symmetric key material in the trusted Python
    interpreter.  It is an API capability boundary, not protection against
    arbitrary code execution or memory inspection in that interpreter.
    """

    __slots__ = ("__secret", "_attestor_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("gate batch verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._attestor_id = _attestor_id(secret)

    @property
    def attestor_id(self) -> str:
        """Return the stable identity of this process-scoped capability."""

        return self._attestor_id

    def verify(self, batch: GateTrialBatch | object) -> GateTrialBatch:
        """Validate schema, signer identity, and the exact content HMAC."""

        validated = GateTrialBatch.model_validate(batch)
        if validated.attestor_id != self._attestor_id:
            raise GateBatchAttestationError("gate trial batch was signed by an untrusted attestor")
        expected = _attestation(self.__secret, validated.content, self._attestor_id)
        if not hmac.compare_digest(validated.attestation_sha256, expected):
            raise GateBatchAttestationError("gate trial batch HMAC attestation is invalid")
        return validated


class TrustedGateBatchService:
    """Process-local capability that issues authentic gate trial batches.

    Trusted orchestration may hand arbitrary candidate output to its isolated
    runner or grader, but it must pass only independently verified observations
    into this signer.  Candidate code must never receive this service.
    """

    __slots__ = ("__secret", "_verification_capability")

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)
        self._verification_capability = GateBatchVerificationCapability(self.__secret)

    @property
    def verification_capability(self) -> GateBatchVerificationCapability:
        """Return the paired verify-only capability for trusted consumers."""

        return self._verification_capability

    @property
    def attestor_id(self) -> str:
        """Return the ID that must be frozen into ``ProtocolManifest``."""

        return self._verification_capability.attestor_id

    def issue(self, content: GateTrialBatchContent) -> GateTrialBatch:
        """Sign one exact, validated batch content value."""

        validated = GateTrialBatchContent.model_validate(content)
        fields = validated.model_dump(
            mode="python",
            round_trip=True,
            warnings="none",
        )
        return GateTrialBatch(
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
        arm: GateTrialArm,
        harness_ref: ArtifactRef,
        gate_split_ref: ArtifactRef,
        mechanism_evidence_ref: ArtifactRef,
        source_refs: Sequence[ArtifactRef],
        observations: Sequence[TrialObservation],
    ) -> GateTrialBatch:
        """Build and sign content tied to one exact frozen protocol."""

        validated_protocol = ProtocolManifest.model_validate(protocol)
        if validated_protocol.gate_batch_attestor_id != self.attestor_id:
            raise GateBatchAttestationError(
                "gate batch signer does not match the protocol-frozen attestor"
            )
        return self.issue(
            GateTrialBatchContent(
                protocol_ref=protocol_ref,
                candidate_ref=candidate_ref,
                arm=arm,
                harness_ref=harness_ref,
                gate_split_ref=gate_split_ref,
                task_set_fingerprint=gate_split_ref.sha256,
                mechanism_evidence_ref=mechanism_evidence_ref,
                execution_context=GateBatchExecutionContext.from_protocol(validated_protocol),
                source_refs=tuple(source_refs),
                observations=tuple(observations),
            )
        )


__all__ = [
    "GATE_TRIAL_BATCH_MEDIA_TYPE",
    "GateBatchAttestationError",
    "GateBatchExecutionContext",
    "GateBatchVerificationCapability",
    "GateTrialArm",
    "GateTrialBatch",
    "GateTrialBatchContent",
    "TrustedGateBatchService",
]
