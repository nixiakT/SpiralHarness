"""Exploration-only trusted grading for BFCL v2 FIT and GATE responses."""

from __future__ import annotations

import json
from typing import Literal

from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader import (
    BfclV4PublicV2TrustedGrader,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    parse_bfcl_v4_public_v2_canonical_response,
)

BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-v2-development-grade.v1+json"
)


class BfclV4PublicV2DevelopmentGradeReceipt(ImmutableModel):
    """Minimal answer-free receipt for one opened public-development response."""

    schema_version: Literal["1"] = "1"
    task_ref: NonEmptyStr
    task_id: NonEmptyStr
    split: Literal["fit", "gate"]
    candidate_payload_sha256: Sha256
    canonical_response_sha256: Sha256
    correct: bool
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def grade_bfcl_v4_public_v2_development_response(
    *,
    grader: BfclV4PublicV2TrustedGrader,
    task_ref: str,
    canonical_response: str,
) -> BfclV4PublicV2DevelopmentGradeReceipt:
    """Grade one FIT/GATE response after strict canonical parsing.

    HOLDOUT is deliberately unrepresentable at this exploration boundary. The
    isolated trusted worker reads possible answers only after validating the
    frozen task identity and emits one Boolean.
    """

    if not isinstance(grader, BfclV4PublicV2TrustedGrader):
        raise TypeError("grader must be a BFCL v2 trusted grader")
    parsed = parse_bfcl_v4_public_v2_canonical_response(canonical_response)
    split, task = grader._resolve_task(task_ref)
    if split not in {
        BfclV4PublicDevelopmentV2Split.FIT,
        BfclV4PublicDevelopmentV2Split.GATE,
    }:
        raise ValueError("development comparison cannot grade HOLDOUT")
    calls = tuple(
        {
            "arguments": json.loads(call.arguments_json),
            "function_name": call.function_name,
        }
        for call in parsed.calls
    )
    correct = grader._run_worker_calls(task=task, calls=calls)
    return BfclV4PublicV2DevelopmentGradeReceipt(
        task_ref=task_ref,
        task_id=task.task_id,
        split=split.value,
        candidate_payload_sha256=task.candidate_payload_sha256,
        canonical_response_sha256=canonical_sha256(parsed),
        correct=correct,
    )


__all__ = [
    "BFCL_V4_PUBLIC_V2_DEVELOPMENT_GRADE_MEDIA_TYPE",
    "BfclV4PublicV2DevelopmentGradeReceipt",
    "grade_bfcl_v4_public_v2_development_response",
]
