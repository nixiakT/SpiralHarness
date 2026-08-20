"""Trusted evaluation-authority boundary for the BFCL public-v2 executor."""

from __future__ import annotations

from typing import Protocol

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2DagNode,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_barrier_capability import (
    BfclV4PublicV2VerifiedDecisionBarrierCapability,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2EvaluationUnlock,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2JournalSnapshot,
    BfclV4PublicV2TrustedGradeProjection,
)
from spiral_harness.experiments.bfcl_v4_public_v2_executor_replay import (
    BfclV4PublicV2ExecutorError,
)
from spiral_harness.experiments.bfcl_v4_public_v2_verified_barrier import (
    BfclV4PublicV2VerifiedBarrierReplayError,
    verify_bfcl_v4_public_v2_decision_barrier_snapshot,
)


class BfclV4PublicV2TrustedBinaryGrader(Protocol):
    """Answer-side adapter exposing only opaque authority and grade projections."""

    def issue_evaluation_unlock(
        self,
        capability: BfclV4PublicV2VerifiedDecisionBarrierCapability,
    ) -> BfclV4PublicV2EvaluationUnlock: ...

    def grade(
        self,
        node: BfclV4PublicDevelopmentV2DagNode,
        canonical_response: str,
        *,
        request_payload_sha256: str,
        provider_response_fingerprint: str,
        evaluation_unlock_fingerprint: str | None,
    ) -> BfclV4PublicV2TrustedGradeProjection: ...


def authorize_bfcl_v4_public_v2_evaluation(
    *,
    grader: BfclV4PublicV2TrustedBinaryGrader,
    snapshot: BfclV4PublicV2JournalSnapshot,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    runtime_fingerprint: str,
    semantic_release_fingerprint: str,
) -> BfclV4PublicV2EvaluationUnlock:
    """Independently replay the barrier before asking the grader for an unlock."""

    try:
        capability = verify_bfcl_v4_public_v2_decision_barrier_snapshot(
            snapshot,
            campaign=campaign,
            runtime_fingerprint=runtime_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
        )
    except BfclV4PublicV2VerifiedBarrierReplayError as exc:
        raise BfclV4PublicV2ExecutorError("trusted evaluation barrier replay failed") from exc
    evidence = capability.receipt.evidence
    try:
        unlock = BfclV4PublicV2EvaluationUnlock.model_validate(
            grader.issue_evaluation_unlock(capability),
            strict=True,
        )
    except Exception as exc:
        raise BfclV4PublicV2ExecutorError("trusted evaluation authorization failed") from exc
    if unlock.barrier_evidence != evidence:
        raise BfclV4PublicV2ExecutorError("evaluation unlock binds another decision barrier")
    return unlock


__all__ = [
    "BfclV4PublicV2TrustedBinaryGrader",
    "authorize_bfcl_v4_public_v2_evaluation",
]
