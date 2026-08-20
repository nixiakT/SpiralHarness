"""Compact, replayable dispatch lineage for the BFCL public-v2 executor."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2DagNode,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256

BFCL_V4_PUBLIC_V2_JOURNAL_PREFIX_PROTOCOL = "spiral-bfcl-v4-public-v2-journal-prefix/v1"
BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT = canonical_sha256(
    {"domain": "spiral-bfcl-v4-public-v2-proposal-batch-set/v1", "batches": ()}
)


def bfcl_v4_public_v2_journal_prefix_fingerprint(
    *,
    campaign_plan_fingerprint: str,
    node_schedule_content_sha256: str,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
    event_count: int,
    tail_event_sha256: str | None,
) -> str:
    """Content-address one prefix in O(1) from its hash-chain commitment."""

    return canonical_sha256(
        {
            "domain": BFCL_V4_PUBLIC_V2_JOURNAL_PREFIX_PROTOCOL,
            "campaign_plan_fingerprint": campaign_plan_fingerprint,
            "node_schedule_content_sha256": node_schedule_content_sha256,
            "runtime_fingerprint": runtime_fingerprint,
            "semantic_release_fingerprint": semantic_release_fingerprint,
            "event_count": event_count,
            "tail_event_sha256": tail_event_sha256,
        }
    )


class BfclV4PublicV2DispatchControl(ImmutableModel):
    """Minimal nomination/decision projection safe for a request binder."""

    node_id: NonEmptyStr
    replicate_id: NonEmptyStr
    arm: BfclV4PublicDevelopmentV2Arm
    kind: Literal["nomination", "gate-decision"]
    value: Literal[
        "parent-fallback",
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "promote",
    ]
    event_fingerprint: Sha256


class BfclV4PublicV2DispatchContext(ImmutableModel):
    """Compact trusted prefix/control view; it carries no responses or grades."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    journal_prefix_event_count: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    journal_prefix_tail_event_sha256: Sha256 | None
    journal_prefix_fingerprint: Sha256
    controls: Annotated[tuple[BfclV4PublicV2DispatchControl, ...], Field(max_length=12)] = ()
    proposal_batch_set_fingerprint: Sha256 = BFCL_V4_PUBLIC_V2_EMPTY_PROPOSAL_BATCH_SET_FINGERPRINT
    journal_prefix_replayed: Literal[True] = True
    responses_present: Literal[False] = False
    grades_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_context(self) -> Self:
        expected = bfcl_v4_public_v2_journal_prefix_fingerprint(
            campaign_plan_fingerprint=self.campaign_plan_fingerprint,
            node_schedule_content_sha256=self.node_schedule_content_sha256,
            runtime_fingerprint=self.runtime_fingerprint,
            semantic_release_fingerprint=self.semantic_release_fingerprint,
            event_count=self.journal_prefix_event_count,
            tail_event_sha256=self.journal_prefix_tail_event_sha256,
        )
        coordinates = tuple((item.replicate_id, item.arm, item.kind) for item in self.controls)
        if (
            self.journal_prefix_fingerprint != expected
            or (self.journal_prefix_tail_event_sha256 is None)
            is not (self.journal_prefix_event_count == 0)
            or len(set(coordinates)) != len(coordinates)
        ):
            raise ValueError("dispatch context differs from its prefix or control projection")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2DispatchReceipt(ImmutableModel):
    """Exact pre-reservation handoff from a materializer to the executor core."""

    schema_version: Literal["1"] = "1"
    node: BfclV4PublicDevelopmentV2DagNode
    node_reference_sha256: Sha256
    journal_prefix_fingerprint: Sha256
    journal_prefix_event_count: Annotated[int, Field(ge=0, lt=1_098, strict=True)]
    journal_prefix_tail_event_sha256: Sha256 | None
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    request_materialization_fingerprint: Sha256
    native_request_fingerprint: Sha256
    request_payload_sha256: Sha256
    proposal_batch_set_fingerprint: Sha256
    actual_request_materialized: Literal[True] = True
    journal_prefix_replayed_before_dispatch: Literal[True] = True
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _bind_dispatch_lineage(self) -> Self:
        expected_prefix = bfcl_v4_public_v2_journal_prefix_fingerprint(
            campaign_plan_fingerprint=self.campaign_plan_fingerprint,
            node_schedule_content_sha256=self.node_schedule_content_sha256,
            runtime_fingerprint=self.runtime_fingerprint,
            semantic_release_fingerprint=self.semantic_release_fingerprint,
            event_count=self.journal_prefix_event_count,
            tail_event_sha256=self.journal_prefix_tail_event_sha256,
        )
        if (
            self.node_reference_sha256 != canonical_sha256(self.node)
            or self.journal_prefix_event_count != self.node.node_slot
            or (self.journal_prefix_tail_event_sha256 is None)
            is not (self.journal_prefix_event_count == 0)
            or self.journal_prefix_fingerprint != expected_prefix
            or self.native_request_fingerprint != self.request_payload_sha256
        ):
            raise ValueError(
                "dispatch receipt differs from its node, replayed prefix, or native request"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("bfcl_v4_public_v2_journal_prefix")
]
