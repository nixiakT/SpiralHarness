"""Portable result contracts for one BFCL V4 public-pilot replicate."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotOuterSeed,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BFCL_V4_RUN_CLOSURE_MEDIA_TYPE,
    BfclV4RunClosureVerification,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    AttemptBudget,
    FrozenModelSpec,
)
from spiral_harness.execution.native_function_contracts import (
    NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE,
)

BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-runner-harness.v1+json"
BFCL_V4_RUNNER_RESULT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-runner-result.v1+json"
)
BFCL_V4_RUNNER_DIAGNOSIS_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-diagnosis-parse.v1+json"
)
BFCL_V4_RUNNER_CANDIDATE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-candidate-parse.v1+json"
)
BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-selection-evidence.v1+json"
)
BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-selection-decision.v1+json"
)
BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-holdout-evidence.v1+json"
)
BFCL_V4_RUNNER_METRICS_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-public-metrics.v1+json"
BFCL_V4_RUNNER_PURE_AT_B_SELECTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-pure-at-b-selection.v1+json"
)
BFCL_V4_RUNNER_PREDICTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-prediction.v1+json"
)
BFCL_V4_RUNNER_GRADER_RECEIPT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-public-grader-receipt.v1+json"
)
BFCL_V4_RUNNER_META_PROMPT_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-meta-prompt.v1+json"
BFCL_V4_RUNNER_ARM_DECISION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-arm-selection-decision.v1+json"
)
BFCL_V4_RUNNER_PURE_AT_B_SAMPLE_MEDIA_TYPE = (
    "application/vnd.spiral-harness.bfcl-v4-pure-at-b-sample.v1+json"
)
BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE = "application/vnd.spiral-harness.bfcl-v4-public-task.v1+json"


class BfclV4RunnerHarnessKind(StrEnum):
    """Exact solver prompt roles persisted before any provider call."""

    BARE = "bare"
    STATIC = "static"
    PARENT = "parent"
    CANDIDATE = "candidate"


class BfclV4RunnerHarnessArtifact(ImmutableModel):
    """Auditable prompt text selected by a materialized solver call."""

    schema_version: Literal["1"] = "1"
    kind: BfclV4RunnerHarnessKind
    arm: Literal["score", "full"] | None = None
    system_prompt: str | None = None
    system_prompt_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _bind_prompt(self) -> Self:
        if self.kind is BfclV4RunnerHarnessKind.BARE:
            if self.arm is not None or self.system_prompt is not None:
                raise ValueError("bare harness cannot contain a system prompt or arm")
            if self.system_prompt_sha256 is not None:
                raise ValueError("bare harness cannot contain a prompt digest")
            return self
        if not self.system_prompt:
            raise ValueError("non-bare harness requires an exact system prompt")
        expected = sha256_bytes(self.system_prompt.encode("utf-8"))
        if self.system_prompt_sha256 != expected:
            raise ValueError("harness prompt digest differs from exact text")
        if self.kind is BfclV4RunnerHarnessKind.CANDIDATE:
            if self.arm not in {"score", "full"}:
                raise ValueError("candidate harness requires its adaptive arm")
        elif self.arm is not None:
            raise ValueError("only candidate harnesses are arm-specific")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


def _exact_ref(value: ImmutableModel, ref: ArtifactRef, media_type: str) -> bool:
    payload = canonical_json_bytes(value)
    return (
        ref.media_type == media_type
        and ref.size == len(payload)
        and ref.sha256 == sha256_bytes(payload)
    )


class BfclV4PublicPilotRunResult(ImmutableModel):
    """Content-addressed terminal evidence for exactly one 100-call replicate."""

    schema_version: Literal["1"] = "1"
    plan_fingerprint: Sha256
    schedule_content_sha256: Sha256
    outer_seed_u64: BfclV4PilotOuterSeed
    manifest_fingerprint: Sha256
    public_task_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=15, max_length=15)]
    model_spec: FrozenModelSpec
    native_runner_fingerprint: Sha256
    attempt_ledger_id: NonEmptyStr
    attempt_budget: AttemptBudget
    attempt_ledger_tail_ref: ArtifactRef
    attempt_outcome_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=100, max_length=100)]
    native_execution_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=100, max_length=100)]
    journal_closure_ref: ArtifactRef
    closure_verification: BfclV4RunClosureVerification
    candidate_freeze_ref: ArtifactRef
    joint_selection_freeze_ref: ArtifactRef
    joint_selection_evidence_ref: ArtifactRef
    joint_selection_decision_ref: ArtifactRef
    holdout_evidence_ref: ArtifactRef
    descriptive_metrics_ref: ArtifactRef
    provider_attempts_succeeded: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_attempts_failed: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_identity_observation_count: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_declared_identity_consistent: bool
    total_model_calls: Literal[100] = 100
    same_frozen_request_model_for_all_calls: Literal[True] = True
    same_per_call_budget_for_all_calls: Literal[True] = True
    provider_served_same_weights_attested: Literal[False] = False
    automatic_retries_used: Literal[False] = False
    holdout_used_for_search: Literal[False] = False
    target_free_pure_at_b_selection: Literal[True] = True
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_result(self) -> Self:
        if self.attempt_budget.max_attempts != 100:
            raise ValueError("runner result requires one exact 100-attempt ledger")
        if self.provider_attempts_succeeded + self.provider_attempts_failed != 100:
            raise ValueError("provider outcome counts must cover all 100 calls")
        if len({item.sha256 for item in self.public_task_refs}) != 15 or any(
            item.media_type != BFCL_V4_RUNNER_PUBLIC_TASK_MEDIA_TYPE
            for item in self.public_task_refs
        ):
            raise ValueError("runner result requires fifteen distinct frozen public task refs")
        if self.provider_identity_observation_count > self.provider_attempts_succeeded:
            raise ValueError("provider identity coverage exceeds successful responses")
        if self.provider_declared_identity_consistent and not (
            self.provider_identity_observation_count
        ):
            raise ValueError("identity consistency requires at least one provider observation")
        if self.attempt_ledger_tail_ref != self.attempt_outcome_refs[-1]:
            raise ValueError("attempt ledger tail must be the final frozen outcome")
        if len({item.sha256 for item in self.attempt_outcome_refs}) != 100:
            raise ValueError("attempt outcome references must not repeat")
        if len({item.sha256 for item in self.native_execution_refs}) != 100:
            raise ValueError("native execution references must not repeat")
        for ref in self.attempt_outcome_refs:
            if ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
                raise ValueError("attempt outcome reference has the wrong media type")
        for ref in self.native_execution_refs:
            if ref.media_type != NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE:
                raise ValueError("native execution reference has the wrong media type")
        if self.journal_closure_ref.media_type != BFCL_V4_RUN_CLOSURE_MEDIA_TYPE:
            raise ValueError("journal closure reference has the wrong media type")
        if self.closure_verification.plan_fingerprint != self.plan_fingerprint:
            raise ValueError("closure verification belongs to another call plan")
        expected_media = (
            (self.joint_selection_evidence_ref, BFCL_V4_RUNNER_SELECTION_EVIDENCE_MEDIA_TYPE),
            (self.joint_selection_decision_ref, BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE),
            (self.holdout_evidence_ref, BFCL_V4_RUNNER_HOLDOUT_EVIDENCE_MEDIA_TYPE),
            (self.descriptive_metrics_ref, BFCL_V4_RUNNER_METRICS_MEDIA_TYPE),
        )
        if any(ref.media_type != media_type for ref, media_type in expected_media):
            raise ValueError("runner evidence reference has the wrong media type")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicPilotRunRecord(ImmutableModel):
    """Return value that proves the published result ref contains this result."""

    schema_version: Literal["1"] = "1"
    result: BfclV4PublicPilotRunResult
    result_ref: ArtifactRef

    @model_validator(mode="after")
    def _bind_result_ref(self) -> Self:
        if not _exact_ref(self.result, self.result_ref, BFCL_V4_RUNNER_RESULT_MEDIA_TYPE):
            raise ValueError("runner result ref differs from the exact result")
        return self


class BfclV4PublicPilotRunVerification(ImmutableModel):
    """Offline audit result for both semantic and attempt-accounting chains."""

    schema_version: Literal["1"] = "1"
    result_fingerprint: Sha256
    plan_fingerprint: Sha256
    completed_model_calls: Literal[100] = 100
    verified_attempt_outcomes: Literal[100] = 100
    verified_native_executions: Literal[100] = 100
    verified_journal_transitions: Literal[204] = 204
    one_attempt_ledger_verified: Literal[True] = True
    same_requested_model_and_inference_verified: Literal[True] = True
    same_backend_protocol_identities_verified: Literal[True] = True
    task_and_meta_prompt_fingerprints_verified: Literal[True] = True
    provider_identity_observation_count: Annotated[int, Field(ge=0, le=100, strict=True)]
    provider_declared_identity_consistent: bool
    provider_served_same_weights_attested: Literal[False] = False
    one_attempt_per_frozen_slot_verified: Literal[True] = True
    request_execution_outcome_completion_join_verified: Literal[True] = True
    semantic_closure_verified: Literal[True] = True
    public_development_only: Literal[True] = True
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
