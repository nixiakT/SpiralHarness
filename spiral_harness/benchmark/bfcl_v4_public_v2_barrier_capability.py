"""Process-local authority produced only by an independent decision-barrier replay."""

from __future__ import annotations

import hmac
import secrets
import weakref
from hashlib import sha256
from typing import Annotated, Literal, NoReturn, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2DecisionBarrierEvidence,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.models import ImmutableModel, Sha256

_PROCESS_KEY = secrets.token_bytes(32)


class BfclV4PublicV2VerifiedBarrierError(ValueError):
    """A purported barrier capability lacks complete replay provenance."""


class BfclV4PublicV2VerifiedDecisionBarrierReceipt(ImmutableModel):
    """Serializable audit projection of the independently replayed prefix."""

    schema_version: Literal["1"] = "1"
    protocol: Literal["spiral-bfcl-v4-public-v2-verified-decision-barrier/v1"] = (
        "spiral-bfcl-v4-public-v2-verified-decision-barrier/v1"
    )
    evidence: BfclV4PublicV2DecisionBarrierEvidence
    evidence_fingerprint: Sha256
    journal_snapshot_fingerprint: Sha256
    journal_prefix_event_count: Annotated[int, Field(gt=0, lt=1_098, strict=True)]
    journal_tail_event_fingerprint: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    replay_state_fingerprint: Sha256
    decision_count: Literal[6] = 6
    pending_reservation_present: Literal[False] = False
    evaluation_started: Literal[False] = False
    independently_replayed: Literal[True] = True

    @model_validator(mode="after")
    def _bind_replayed_roots(self) -> Self:
        if (
            self.evidence_fingerprint != self.evidence.fingerprint
            or self.semantic_release_fingerprint != self.evidence.semantic_release_fingerprint
            or self.journal_tail_event_fingerprint != self.evidence.final_decision_event_fingerprint
        ):
            raise ValueError("verified barrier receipt has inconsistent replay roots")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2VerifiedDecisionBarrierCapability:
    """Non-copyable proof registered only inside the current trusted process."""

    __slots__ = ("__weakref__", "_process_tag", "_receipt")

    def __new__(cls, *_args: object, **_kwargs: object) -> NoReturn:
        del cls, _args, _kwargs
        raise BfclV4PublicV2VerifiedBarrierError(
            "verified decision barrier can only be minted by independent replay"
        )

    def __setattr__(self, _name: str, _value: object) -> NoReturn:
        raise BfclV4PublicV2VerifiedBarrierError("verified decision barrier is immutable")

    def __delattr__(self, _name: str) -> NoReturn:
        raise BfclV4PublicV2VerifiedBarrierError("verified decision barrier is immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("verified decision barrier cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("verified decision barrier cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified decision barrier cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("verified decision barrier cannot be serialized")

    @property
    def receipt(self) -> BfclV4PublicV2VerifiedDecisionBarrierReceipt:
        return self._receipt

    def __repr__(self) -> str:
        return (
            f"BfclV4PublicV2VerifiedDecisionBarrierCapability(receipt={self.receipt.fingerprint!r})"
        )


_ISSUED: weakref.WeakSet[BfclV4PublicV2VerifiedDecisionBarrierCapability] = weakref.WeakSet()


def _tag(receipt: BfclV4PublicV2VerifiedDecisionBarrierReceipt) -> str:
    return hmac.new(
        _PROCESS_KEY,
        canonical_json_bytes(
            {
                "domain": "spiral-bfcl-v4-public-v2-process-local-barrier/v1",
                "receipt_fingerprint": receipt.fingerprint,
            }
        ),
        sha256,
    ).hexdigest()


def _mint_bfcl_v4_public_v2_verified_decision_barrier(
    receipt: BfclV4PublicV2VerifiedDecisionBarrierReceipt,
) -> BfclV4PublicV2VerifiedDecisionBarrierCapability:
    """Internal handoff used by the independent replay verifier."""

    checked = BfclV4PublicV2VerifiedDecisionBarrierReceipt.model_validate(receipt, strict=True)
    capability = object.__new__(BfclV4PublicV2VerifiedDecisionBarrierCapability)
    object.__setattr__(capability, "_receipt", checked)
    object.__setattr__(capability, "_process_tag", _tag(checked))
    _ISSUED.add(capability)
    return capability


def validate_bfcl_v4_public_v2_verified_decision_barrier(
    value: object,
) -> BfclV4PublicV2VerifiedDecisionBarrierCapability:
    """Reject reconstructed, copied, cross-process, or internally altered values."""

    if type(value) is not BfclV4PublicV2VerifiedDecisionBarrierCapability:
        raise BfclV4PublicV2VerifiedBarrierError(
            "evaluation authority requires an exact verified replay capability"
        )
    capability = value
    try:
        receipt = BfclV4PublicV2VerifiedDecisionBarrierReceipt.model_validate(
            capability.receipt,
            strict=True,
        )
        valid = capability in _ISSUED and hmac.compare_digest(
            capability._process_tag,
            _tag(receipt),
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise BfclV4PublicV2VerifiedBarrierError(
            "evaluation authority was not minted by this process replay path"
        )
    return capability


__all__ = [
    "BfclV4PublicV2VerifiedBarrierError",
    "BfclV4PublicV2VerifiedDecisionBarrierCapability",
    "BfclV4PublicV2VerifiedDecisionBarrierReceipt",
    "validate_bfcl_v4_public_v2_verified_decision_barrier",
]
