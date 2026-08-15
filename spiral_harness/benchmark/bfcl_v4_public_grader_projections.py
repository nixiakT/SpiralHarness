"""Candidate-safe projections from BFCL public grader receipts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4StrictGraderContract,
    PilotTaskId,
)
from spiral_harness.core.models import Sha256

FIT_TASK_IDS = (
    "simple_python_0",
    "simple_python_211",
    "multiple_5",
    "parallel_0",
    "parallel_multiple_9",
)


class BfclV4FullFitFeedback(BfclV4StrictGraderContract):
    """The only per-task feedback that a FULL-arm optimizer may receive."""

    schema_version: Literal["1"] = "1"
    information_scope: Literal["own-public-fit-binary-and-coarse-failure"] = (
        "own-public-fit-binary-and-coarse-failure"
    )
    task_id: PilotTaskId
    own_prediction_reference_sha256: Sha256
    accepted: bool
    failure_class: Literal["none", "call-count", "function-or-arguments"]
    candidate_visible: Literal[True] = True
    partial_evaluation: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_feedback(self) -> Self:
        if self.task_id not in FIT_TASK_IDS:
            raise ValueError("FULL feedback is restricted to the five FIT tasks")
        if self.accepted != (self.failure_class == "none"):
            raise ValueError("FULL feedback validity and failure class disagree")
        return self


class BfclV4ScoreFitAggregate(BfclV4StrictGraderContract):
    """Exactly one five-task aggregate; no task-wise labels or receipt refs."""

    schema_version: Literal["1"] = "1"
    information_scope: Literal["five-public-fit-aggregate-score-only"] = (
        "five-public-fit-aggregate-score-only"
    )
    plan_fingerprint: Sha256
    batch_reference_sha256: Sha256
    fit_task_count: Literal[5] = 5
    aggregate_accuracy_basis_points: Annotated[
        int,
        Field(ge=0, le=10_000, multiple_of=2_000, strict=True),
    ]
    candidate_visible: Literal[True] = True
    partial_evaluation: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False


__all__ = ["FIT_TASK_IDS", "BfclV4FullFitFeedback", "BfclV4ScoreFitAggregate"]
