"""Authenticated objective-interval provenance shared by trusted graders."""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

import spiral_harness.execution.receipts as _receipts
from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.evolution.matched_media_types import MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE
from spiral_harness.evolution.models import (
    PROMPT_PROPOSAL_MEDIA_TYPE,
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
)
from spiral_harness.storage.protocol import ArtifactRepository

SEARCH_BENCHMARK_BINDING_MEDIA_TYPE = (
    "application/vnd.spiral-harness.search-benchmark-binding.v1+json"
)
TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.trusted-objective-aggregate.v4+json"
)


class TrustedObjectiveIntervalEvidence(ImmutableModel):
    """Estimator configuration, task counts, and both confidence bounds."""

    schema_version: Literal["1"] = "1"
    estimator_version: Literal["paired-task-percentile-bootstrap-v1"] = (
        "paired-task-percentile-bootstrap-v1"
    )
    statistical_unit: Literal["paired-task-mean-delta"] = "paired-task-mean-delta"
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    lower: Annotated[float, Field(allow_inf_nan=False)]
    upper: Annotated[float, Field(allow_inf_nan=False)]
    bootstrap_samples: Annotated[int, Field(ge=1_000, strict=True)]
    bootstrap_seed: Annotated[int, Field(ge=0, strict=True)]
    n_tasks: Annotated[int, Field(gt=0, strict=True)]
    n_valid_pairs: Annotated[int, Field(gt=0, strict=True)]

    @model_validator(mode="after")
    def _interval_shape_is_consistent(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("objective confidence interval lower exceeds upper")
        if self.n_valid_pairs < self.n_tasks:
            raise ValueError("objective valid-pair count is smaller than task count")
        return self


class TrustedObjectiveAggregateContent(ImmutableModel):
    """Trusted grader authorization for scores and interval over one batch."""

    schema_version: Literal["4"] = "4"
    search_run_ref: ArtifactRef
    proposal_ref: ArtifactRef
    candidate_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    candidate_harness_ref: ArtifactRef
    benchmark_binding_ref: ArtifactRef
    grader_fingerprint: NonEmptyStr
    schedule_fingerprint: Sha256
    receipt_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    primary_score: Annotated[float, Field(allow_inf_nan=False)]
    mean_delta: Annotated[float, Field(allow_inf_nan=False)]
    confidence_interval: TrustedObjectiveIntervalEvidence
    regression_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    tokens_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    latency_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @field_validator("receipt_refs")
    @classmethod
    def _canonicalize_objective_receipts(
        cls,
        value: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        ordered = tuple(sorted(value, key=lambda ref: (ref.sha256, ref.size, ref.media_type)))
        if {ref.media_type for ref in ordered} != {_receipts.EXECUTION_RECEIPT_MEDIA_TYPE}:
            raise ValueError("objective receipt refs must be execution receipts")
        if len(ordered) != len({ref.sha256 for ref in ordered}):
            raise ValueError("objective receipt refs must not contain duplicates")
        return ordered

    @model_validator(mode="after")
    def _objective_refs_have_exact_media_types(self) -> Self:
        if self.search_run_ref.media_type not in {
            SEARCH_RUN_MANIFEST_MEDIA_TYPE,
            MATCHED_V2_RUN_MANIFEST_MEDIA_TYPE,
        }:
            raise ValueError("search_run_ref declares the wrong media type")
        if self.proposal_ref.media_type != PROMPT_PROPOSAL_MEDIA_TYPE:
            raise ValueError("proposal_ref declares the wrong media type")
        if self.benchmark_binding_ref.media_type != SEARCH_BENCHMARK_BINDING_MEDIA_TYPE:
            raise ValueError("benchmark_binding_ref declares the wrong media type")
        if not self.confidence_interval.lower <= self.mean_delta <= self.confidence_interval.upper:
            raise ValueError("objective confidence interval must contain mean_delta")
        return self

    @property
    def confidence_level(self) -> float:
        return self.confidence_interval.confidence_level

    @property
    def confidence_lower(self) -> float:
        return self.confidence_interval.lower

    @property
    def confidence_upper(self) -> float:
        return self.confidence_interval.upper


class TrustedObjectiveAggregate(ImmutableModel):
    """HMAC-attested score aggregate issued by the independent grader plane."""

    schema_version: Literal["4"] = "4"
    content: TrustedObjectiveAggregateContent
    attestor_id: Sha256
    authentication_tag: Sha256


class ObjectiveAggregateVerificationError(ValueError):
    """Raised when an objective aggregate is not authentic and canonical."""


class ObjectiveAggregateVerificationCapability:
    """Exact verify-only capability for independent score attestations."""

    __slots__ = ("__attestor_id", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("objective aggregate verification capability cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        if type(secret) is not bytes or len(secret) < 32:
            raise ValueError("objective aggregate attestor secret must contain at least 32 bytes")
        self.__store = store
        self.__secret = secret
        domain = b"spiral-harness/objective-aggregate-attestor/v4\x00"
        self.__attestor_id = sha256_bytes(domain + secret)

    @property
    def attestor_id(self) -> str:
        return self.__attestor_id

    def verify(self, aggregate_ref: ArtifactRef) -> TrustedObjectiveAggregateContent:
        if aggregate_ref.media_type != TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE:
            raise ObjectiveAggregateVerificationError(
                "objective aggregate declares the wrong media type"
            )
        try:
            payload = self.__store.get_bytes(aggregate_ref)
            aggregate = self.__store.get_json(aggregate_ref, TrustedObjectiveAggregate)
        except Exception as exc:
            raise ObjectiveAggregateVerificationError(
                "objective aggregate cannot be loaded"
            ) from exc
        if payload != canonical_json_bytes(aggregate):
            raise ObjectiveAggregateVerificationError("objective aggregate is not canonical")
        if aggregate.attestor_id != self.__attestor_id:
            raise ObjectiveAggregateVerificationError("objective aggregate uses another attestor")
        expected = hmac.new(
            self.__secret,
            b"spiral-harness/objective-aggregate/v4\x00",
            sha256,
        )
        expected.update(self.__attestor_id.encode("ascii") + b"\x00")
        expected.update(canonical_json_bytes(aggregate.content))
        if not hmac.compare_digest(aggregate.authentication_tag, expected.hexdigest()):
            raise ObjectiveAggregateVerificationError("objective aggregate authentication failed")
        return aggregate.content


class TrustedObjectiveAggregateService:
    """Trusted setup authority kept outside the general search runtime."""

    __slots__ = ("__capability", "__secret", "__store")

    def __init_subclass__(cls, **kwargs: object) -> None:  # pragma: no cover
        raise TypeError("objective aggregate service cannot be subclassed")

    def __init__(self, store: ArtifactRepository, *, secret: bytes) -> None:
        self.__store = store
        self.__secret = secret
        self.__capability = ObjectiveAggregateVerificationCapability(store, secret=secret)

    @property
    def verification_capability(self) -> ObjectiveAggregateVerificationCapability:
        return self.__capability

    def attest(self, content: TrustedObjectiveAggregateContent) -> ArtifactRef:
        checked = TrustedObjectiveAggregateContent.model_validate(
            content.model_dump(mode="python", round_trip=True, warnings="none"),
            strict=True,
        )
        authentication = hmac.new(
            self.__secret,
            b"spiral-harness/objective-aggregate/v4\x00",
            sha256,
        )
        authentication.update(self.__capability.attestor_id.encode("ascii") + b"\x00")
        authentication.update(canonical_json_bytes(checked))
        aggregate = TrustedObjectiveAggregate(
            content=checked,
            attestor_id=self.__capability.attestor_id,
            authentication_tag=authentication.hexdigest(),
        )
        ref = self.__store.put_json(
            aggregate,
            media_type=TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE,
        )
        if self.__capability.verify(ref) != checked:
            raise ObjectiveAggregateVerificationError(
                "published objective aggregate changed content"
            )
        return ref


__all__ = [
    "SEARCH_BENCHMARK_BINDING_MEDIA_TYPE",
    "TRUSTED_OBJECTIVE_AGGREGATE_MEDIA_TYPE",
    "ObjectiveAggregateVerificationCapability",
    "ObjectiveAggregateVerificationError",
    "TrustedObjectiveAggregate",
    "TrustedObjectiveAggregateContent",
    "TrustedObjectiveAggregateService",
    "TrustedObjectiveIntervalEvidence",
]
