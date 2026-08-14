"""Immutable contracts for a genuinely bare-model reference condition.

PURE is deliberately not a member of the harness baseline taxonomy.  Its
provider-visible request contains one user message copied byte-for-byte from
the candidate-facing task and no representable harness surface.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import (
    ATTEMPT_OUTCOME_MEDIA_TYPE,
    PURE_REFERENCE_EXECUTION_MEDIA_TYPE,
    CandidateTask,
    ExecutionError,
    ExecutionStatus,
    FrozenModelSpec,
    ModelUsage,
)

_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_ExactNonEmptyText = Annotated[str, Field(min_length=1)]


class _FrozenPureModel(BaseModel):
    """Strict immutable base that never strips provider-visible text."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class PureReferenceCapabilities(_FrozenPureModel):
    """Machine-checkable declaration of every intentionally absent surface."""

    schema_version: Literal["1"] = "1"
    harness: Literal[False] = False
    seed_prompt: Literal[False] = False
    skill: Literal[False] = False
    memory: Literal[False] = False
    tools: Literal[False] = False
    middleware: Literal[False] = False
    control_flow: Literal[False] = False
    search: Literal[False] = False
    evolution: Literal[False] = False


class PureUserMessage(_FrozenPureModel):
    """The only provider-visible message representable by the PURE contract."""

    role: Literal["user"] = "user"
    content: _ExactNonEmptyText


class PureReferenceRequest(_FrozenPureModel):
    """Exact bare-model request: task payload plus the minimum chat wrapper."""

    schema_version: Literal["1"] = "1"
    condition: Literal["pure-model-reference"] = "pure-model-reference"
    reference_id: _Sha256
    task_id: _ExactNonEmptyText
    task_payload_sha256: _Sha256
    task_payload_size_bytes: Annotated[int, Field(ge=1, strict=True)]
    messages: tuple[PureUserMessage]
    rollout_seed: Annotated[int, Field(ge=0, strict=True)]
    capabilities: PureReferenceCapabilities = PureReferenceCapabilities()

    @model_validator(mode="after")
    def payload_identity_matches_the_only_message(self) -> Self:
        payload = self.messages[0].content.encode("utf-8")
        if self.task_payload_sha256 != sha256_bytes(payload):
            raise ValueError("task_payload_sha256 does not match the exact UTF-8 task bytes")
        if self.task_payload_size_bytes != len(payload):
            raise ValueError("task_payload_size_bytes does not match the exact UTF-8 task bytes")
        return self

    @property
    def task_payload(self) -> str:
        """Return the exact provider-visible task text without re-rendering it."""

        return self.messages[0].content

    @property
    def seed(self) -> int:
        return self.rollout_seed

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class PureReferenceExecution(_FrozenPureModel):
    """Score-free execution artifact for a PURE model call."""

    schema_version: Literal["1"] = "1"
    condition: Literal["pure-model-reference"] = "pure-model-reference"
    capabilities: PureReferenceCapabilities = PureReferenceCapabilities()
    task: CandidateTask
    request: PureReferenceRequest
    output: str | None
    status: ExecutionStatus
    usage: ModelUsage
    spec: FrozenModelSpec
    execution_fingerprint: _Sha256
    request_sha256: _Sha256
    error: ExecutionError | None

    @model_validator(mode="after")
    def provenance_and_status_are_consistent(self) -> Self:
        if self.request.task_id != self.task.task_id:
            raise ValueError("request task_id does not match task")
        if self.request.task_payload != self.task.question:
            raise ValueError("PURE request payload does not match candidate-facing task bytes")
        if self.request_sha256 != self.request.fingerprint:
            raise ValueError("request_sha256 does not match the exact PURE request")
        if self.capabilities != self.request.capabilities:
            raise ValueError("execution capabilities differ from the PURE request")
        if self.status is ExecutionStatus.COMPLETED:
            if self.output is None:
                raise ValueError("completed execution requires an output")
            if self.error is not None:
                raise ValueError("completed execution must not contain an error")
        else:
            if self.output is not None:
                raise ValueError("failed execution must not expose an output")
            if self.error is None:
                raise ValueError("failed execution requires an error classification")
        return self

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def seed(self) -> int:
        return self.request.rollout_seed

    @property
    def reference_id(self) -> str:
        return self.request.reference_id

    @property
    def harness_id(self) -> str:
        """Opaque adapter coordinate; this is not a harness-manifest identity."""

        return self.reference_id

    @property
    def output_text(self) -> str | None:
        return self.output

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def latency_ms(self) -> float:
        return self.usage.latency_ms

    @property
    def tool_calls(self) -> int:
        return 0

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd


class PureReferenceExecutionRecord(_FrozenPureModel):
    """One persisted PURE execution and its terminal accounting outcome."""

    schema_version: Literal["1"] = "1"
    execution: PureReferenceExecution
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef

    @model_validator(mode="after")
    def references_have_exact_media_types(self) -> Self:
        if self.execution_ref.media_type != PURE_REFERENCE_EXECUTION_MEDIA_TYPE:
            raise ValueError("execution_ref declares the wrong PURE execution media type")
        if self.outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("outcome_ref declares the wrong attempt outcome media type")
        return self


def materialize_pure_request(
    task: object,
    *,
    reference_id: str,
    rollout_seed: int,
) -> PureReferenceRequest:
    """Copy exact task text into one user message without harness rendering."""

    checked_task = CandidateTask.from_task_view(task)
    if type(rollout_seed) is not int:
        raise TypeError("rollout_seed must be an integer")
    if rollout_seed < 0:
        raise ValueError("rollout_seed must not be negative")
    payload = checked_task.question.encode("utf-8")
    return PureReferenceRequest(
        reference_id=reference_id,
        task_id=checked_task.task_id,
        task_payload_sha256=sha256_bytes(payload),
        task_payload_size_bytes=len(payload),
        messages=(PureUserMessage(content=checked_task.question),),
        rollout_seed=rollout_seed,
    )


__all__ = [
    "PureReferenceCapabilities",
    "PureReferenceExecution",
    "PureReferenceExecutionRecord",
    "PureReferenceRequest",
    "PureUserMessage",
    "materialize_pure_request",
]
