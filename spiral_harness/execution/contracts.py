"""Immutable data contracts shared by execution and attempt accounting."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import (
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    ImmutableModel,
    NonEmptyStr,
    Sha256,
)
from spiral_harness.skills.loading import SkillDisclosure
from spiral_harness.skills.package import (
    SKILL_CONTEXT_END_DELIMITER,
    SKILL_CONTEXT_START_DELIMITER,
)

ATTEMPT_RESERVATION_MEDIA_TYPE = "application/vnd.spiral-harness.attempt-reservation.v3+json"
ATTEMPT_OUTCOME_MEDIA_TYPE = "application/vnd.spiral-harness.attempt-outcome.v3+json"
MODEL_EXECUTION_MEDIA_TYPE = "application/vnd.spiral-harness.model-execution.v2+json"
PURE_REFERENCE_EXECUTION_MEDIA_TYPE = (
    "application/vnd.spiral-harness.pure-reference-execution.v1+json"
)
ATTEMPT_EXECUTION_MEDIA_TYPES = frozenset(
    {MODEL_EXECUTION_MEDIA_TYPE, PURE_REFERENCE_EXECUTION_MEDIA_TYPE}
)

_UNPINNED_REVISIONS = frozenset(
    {
        "auto",
        "current",
        "default",
        "head",
        "latest",
        "main",
        "master",
    }
)


class AttemptDisposition(StrEnum):
    """Terminal accounting treatment for one reservation."""

    SETTLED = "settled"
    BURNED = "burned"
    POISONED = "poisoned"


class AttemptBudget(ImmutableModel):
    """Hard ceilings enforced before a backend is invoked."""

    schema_version: Literal["1"] = "1"
    max_attempts: Annotated[int, Field(ge=1, strict=True)]
    max_total_tokens: Annotated[int, Field(ge=1, strict=True)]
    max_tokens_per_attempt: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def per_attempt_ceiling_fits_total(self) -> Self:
        if self.max_tokens_per_attempt > self.max_total_tokens:
            raise ValueError("max_tokens_per_attempt must not exceed max_total_tokens")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class AttemptReservation(ImmutableModel):
    """A persisted, single-use authorization created before model execution."""

    schema_version: Literal["3"] = "3"
    ledger_id: NonEmptyStr
    writer_epoch_id: Sha256
    budget_fingerprint: Sha256
    sequence: Annotated[int, Field(ge=0, strict=True)]
    task_fingerprint: Sha256
    execution_fingerprint: Sha256
    request_sha256: Sha256
    reserved_tokens: Annotated[int, Field(ge=1, strict=True)]
    previous_outcome_ref: ArtifactRef | None

    @model_validator(mode="after")
    def previous_ref_has_exact_media_type(self) -> Self:
        if (
            self.previous_outcome_ref is not None
            and self.previous_outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE
        ):
            raise ValueError("previous_outcome_ref declares the wrong media type")
        if self.sequence == 0 and self.previous_outcome_ref is not None:
            raise ValueError("reservation sequence 0 must not have a previous outcome")
        if self.sequence > 0 and self.previous_outcome_ref is None:
            raise ValueError("reservation sequence greater than 0 requires a previous outcome")
        return self


class AttemptOutcome(ImmutableModel):
    """The immutable terminal record consuming exactly one reservation."""

    schema_version: Literal["3"] = "3"
    ledger_id: NonEmptyStr
    writer_epoch_id: Sha256
    budget_fingerprint: Sha256
    sequence: Annotated[int, Field(ge=0, strict=True)]
    reservation_ref: ArtifactRef
    disposition: AttemptDisposition
    reported_tokens: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]
    execution_ref: ArtifactRef | None
    error_class: NonEmptyStr | None

    @model_validator(mode="after")
    def disposition_shape_is_valid(self) -> Self:
        if self.reservation_ref.media_type != ATTEMPT_RESERVATION_MEDIA_TYPE:
            raise ValueError("reservation_ref declares the wrong media type")
        if (
            self.execution_ref is not None
            and self.execution_ref.media_type not in ATTEMPT_EXECUTION_MEDIA_TYPES
        ):
            raise ValueError("execution_ref declares the wrong media type")
        if self.disposition is AttemptDisposition.SETTLED:
            if self.execution_ref is None:
                raise ValueError("a settled outcome requires an execution_ref")
            if self.error_class is not None:
                raise ValueError("a settled outcome must not declare an error_class")
            if self.charged_tokens != self.reported_tokens:
                raise ValueError("a settled outcome charges exactly its reported tokens")
        elif self.error_class is None:
            raise ValueError("a burned or poisoned outcome requires an error_class")
        if (
            self.disposition is AttemptDisposition.POISONED
            and self.charged_tokens != self.reported_tokens
        ):
            raise ValueError("a poisoned outcome charges its reported token overrun")
        return self


class AttemptLedgerState(ImmutableModel):
    """Replay-derived state owned by one process-local ledger writer."""

    ledger_id: NonEmptyStr
    writer_epoch_id: Sha256
    budget: AttemptBudget
    tail_ref: ArtifactRef | None
    pending_reservation_ref: ArtifactRef | None
    poisoned: bool
    attempts_used: Annotated[int, Field(ge=0, strict=True)]
    completed_attempts: Annotated[int, Field(ge=0, strict=True)]
    charged_tokens: Annotated[int, Field(ge=0, strict=True)]
    encumbered_tokens: Annotated[int, Field(ge=0, strict=True)]
    remaining_attempts: Annotated[int, Field(ge=0, strict=True)]
    remaining_tokens: Annotated[int, Field(ge=0, strict=True)]


class _FrozenExecutionModel(BaseModel):
    """Strict immutable model that preserves prompt whitespace byte-for-byte."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


ExactNonEmptyText = Annotated[str, Field(min_length=1)]
ExecutionSha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class InferenceConfig(_FrozenExecutionModel):
    """Complete provider-neutral inference settings for a fixed run."""

    schema_version: Literal["1"] = "1"
    temperature: Annotated[float, Field(ge=0, strict=True)]
    top_p: Annotated[float, Field(gt=0, le=1, strict=True)]
    max_output_tokens: Annotated[int, Field(ge=1, strict=True)]
    timeout_seconds: Annotated[float, Field(gt=0, strict=True)]
    stop_sequences: tuple[ExactNonEmptyText, ...] = ()

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_are_exact_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("stop_sequences must not contain duplicates")
        return values

    @property
    def fingerprint(self) -> str:
        """Canonical identity of every frozen inference setting."""

        return canonical_sha256(self)


class FrozenModelSpec(_FrozenExecutionModel):
    """Pinned provider/model/tokenizer/runtime contract."""

    schema_version: Literal["1"] = "1"
    backend: ExactNonEmptyText
    backend_fingerprint: ExactNonEmptyText
    model: ExactNonEmptyText
    revision: ExactNonEmptyText
    tokenizer: ExactNonEmptyText
    tokenizer_revision: ExactNonEmptyText
    runtime: ExactNonEmptyText
    inference: InferenceConfig

    @field_validator(
        "backend",
        "backend_fingerprint",
        "model",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "runtime",
    )
    @classmethod
    def identities_are_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("model identity fields must not have surrounding whitespace")
        return value

    @field_validator("revision", "tokenizer_revision")
    @classmethod
    def revisions_are_pinned(cls, value: str) -> str:
        normalized = value.casefold()
        if normalized in _UNPINNED_REVISIONS or normalized.rsplit("/", maxsplit=1)[-1] in {
            "main",
            "master",
            "latest",
        }:
            raise ValueError("revision must be an explicit immutable snapshot, not a moving alias")
        return value

    @property
    def model_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/model-identity/v1",
                "model": self.model,
                "revision": self.revision,
                "tokenizer": self.tokenizer,
                "tokenizer_revision": self.tokenizer_revision,
            }
        )

    @property
    def inference_fingerprint(self) -> str:
        return self.inference.fingerprint

    @property
    def runtime_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "spiral-harness/runtime-identity/v1",
                "runtime": self.runtime,
            }
        )

    @property
    def fingerprint(self) -> str:
        """Identity covering backend, model, tokenizer, runtime, and inference."""

        return canonical_sha256(self)


class CandidateTask(_FrozenExecutionModel):
    """The complete candidate-facing task view; no answer is representable."""

    schema_version: Literal["1"] = "1"
    task_id: ExactNonEmptyText
    question: ExactNonEmptyText

    @field_validator("task_id")
    @classmethod
    def task_id_is_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("task_id must not have surrounding whitespace")
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)

    @classmethod
    def from_task_view(cls, task: object) -> CandidateTask:
        """Copy a structural candidate view without importing its benchmark type."""

        if isinstance(task, cls):
            return cls.model_validate(task, strict=True)
        if isinstance(task, Mapping):
            return cls.model_validate(task, strict=True)
        forbidden = ("answer", "gold", "grade", "reference", "reference_answer", "score")
        exposed = tuple(name for name in forbidden if hasattr(task, name))
        if exposed:
            joined = ", ".join(exposed)
            raise ValueError(f"candidate task view exposes trusted fields: {joined}")
        try:
            task_id = task.task_id  # type: ignore[attr-defined]
            question = task.question  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError("task must expose task_id and question attributes") from exc
        return cls(task_id=task_id, question=question)


def resolve_system_prompt(
    base_system_prompt: str,
    skill_disclosure: SkillDisclosure | None,
) -> str:
    """Render the one canonical prompt shape accepted by the runner."""

    if not isinstance(base_system_prompt, str):
        raise TypeError("base_system_prompt must be a string")
    if not base_system_prompt:
        raise ValueError("base_system_prompt must not be empty")
    if (
        SKILL_CONTEXT_START_DELIMITER in base_system_prompt
        or SKILL_CONTEXT_END_DELIMITER in base_system_prompt
    ):
        raise ValueError("base_system_prompt contains a reserved skill context delimiter")
    if skill_disclosure is None:
        return base_system_prompt
    checked = SkillDisclosure.model_validate(skill_disclosure, strict=True)
    expected_start = SKILL_CONTEXT_START_DELIMITER + "\n"
    expected_end = "\n" + SKILL_CONTEXT_END_DELIMITER
    if not checked.context.startswith(expected_start) or not checked.context.endswith(expected_end):
        raise ValueError("skill disclosure context does not use the frozen framing delimiters")
    return base_system_prompt + "\n\n" + checked.context


class ResolvedHarness(_FrozenExecutionModel):
    """Fully materialized prompt/skill state presented to the model backend."""

    schema_version: Literal["2"] = "2"
    harness_ref: ArtifactRef
    base_system_prompt: ExactNonEmptyText
    base_system_prompt_sha256: ExecutionSha256
    skill_disclosure: SkillDisclosure | None = None
    system_prompt: ExactNonEmptyText
    resolved_prompt_sha256: ExecutionSha256

    @model_validator(mode="after")
    def declared_prompt_hashes_match_exact_bytes(self) -> Self:
        if self.harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError("harness_ref must declare the exact harness manifest v2 media type")
        actual_base_sha256 = sha256_bytes(self.base_system_prompt.encode("utf-8"))
        if self.base_system_prompt_sha256 != actual_base_sha256:
            raise ValueError("base_system_prompt_sha256 does not match the exact base prompt bytes")
        expected_prompt = resolve_system_prompt(
            self.base_system_prompt,
            self.skill_disclosure,
        )
        if self.system_prompt != expected_prompt:
            raise ValueError(
                "system_prompt is not the canonical base prompt and skill disclosure rendering"
            )
        actual_resolved_sha256 = sha256_bytes(self.system_prompt.encode("utf-8"))
        if self.resolved_prompt_sha256 != actual_resolved_sha256:
            raise ValueError("resolved_prompt_sha256 does not match the exact system_prompt bytes")
        return self

    @property
    def harness_id(self) -> str:
        """Return the manifest digest used by schedule and grading coordinates."""

        return self.harness_ref.sha256

    @classmethod
    def from_prompt(cls, *, harness_ref: ArtifactRef, system_prompt: str) -> ResolvedHarness:
        """Construct the prompt-only form without inventing a compatibility facade."""

        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        prompt_sha256 = sha256_bytes(system_prompt.encode("utf-8"))
        return cls(
            harness_ref=harness_ref,
            base_system_prompt=system_prompt,
            base_system_prompt_sha256=prompt_sha256,
            skill_disclosure=None,
            system_prompt=system_prompt,
            resolved_prompt_sha256=prompt_sha256,
        )

    @classmethod
    def from_skill(
        cls,
        *,
        harness_ref: ArtifactRef,
        base_system_prompt: str,
        skill_disclosure: SkillDisclosure,
    ) -> ResolvedHarness:
        """Construct the single-skill form using the frozen delimiter contract."""

        checked_disclosure = SkillDisclosure.model_validate(skill_disclosure, strict=True)
        system_prompt = resolve_system_prompt(base_system_prompt, checked_disclosure)
        return cls(
            harness_ref=harness_ref,
            base_system_prompt=base_system_prompt,
            base_system_prompt_sha256=sha256_bytes(base_system_prompt.encode("utf-8")),
            skill_disclosure=checked_disclosure,
            system_prompt=system_prompt,
            resolved_prompt_sha256=sha256_bytes(system_prompt.encode("utf-8")),
        )


class ModelRequest(_FrozenExecutionModel):
    """Exact structured request sent to a backend."""

    schema_version: Literal["2"] = "2"
    task_id: ExactNonEmptyText
    harness_ref: ArtifactRef
    base_system_prompt: ExactNonEmptyText
    base_system_prompt_sha256: ExecutionSha256
    skill_disclosure: SkillDisclosure | None = None
    system_prompt: ExactNonEmptyText
    resolved_prompt_sha256: ExecutionSha256
    user_prompt: ExactNonEmptyText
    seed: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def resolved_prompt_is_self_authenticating(self) -> Self:
        ResolvedHarness(
            harness_ref=self.harness_ref,
            base_system_prompt=self.base_system_prompt,
            base_system_prompt_sha256=self.base_system_prompt_sha256,
            skill_disclosure=self.skill_disclosure,
            system_prompt=self.system_prompt,
            resolved_prompt_sha256=self.resolved_prompt_sha256,
        )
        return self

    @property
    def harness_id(self) -> str:
        """Return the exact manifest digest bound into this request."""

        return self.harness_ref.sha256

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BackendTokenUsage(_FrozenExecutionModel):
    """Provider-reported token counts before trusted latency capture."""

    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelUsage(_FrozenExecutionModel):
    """Score-independent resources captured for one execution."""

    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    latency_ms: Annotated[float, Field(ge=0, strict=True)]
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens(self) -> int:
        return self.total_tokens

    @property
    def tool_calls(self) -> int:
        return 0


class BackendResponse(_FrozenExecutionModel):
    """Minimal score-free value returned by a provider adapter."""

    output: str
    usage: BackendTokenUsage
    cost_usd: Annotated[float, Field(ge=0, strict=True)] | None = None


class ExecutionStatus(StrEnum):
    """Whether an output is admissible for trusted grading."""

    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionErrorClass(StrEnum):
    """Stable infrastructure failure classes; none imply a task score."""

    BACKEND_TIMEOUT = "backend_timeout"
    BACKEND_EXCEPTION = "backend_exception"
    BACKEND_FINGERPRINT_MISMATCH = "backend_fingerprint_mismatch"
    INVALID_BACKEND_RESPONSE = "invalid_backend_response"
    USAGE_EXCEEDED = "usage_exceeded"


class ExecutionError(_FrozenExecutionModel):
    """Captured infrastructure classification without grading authority."""

    error_class: ExecutionErrorClass
    detail: ExactNonEmptyText


class ModelExecution(_FrozenExecutionModel):
    """Strict score-free execution artifact."""

    schema_version: Literal["2"] = "2"
    task: CandidateTask
    request: ModelRequest
    output: str | None
    status: ExecutionStatus
    usage: ModelUsage
    spec: FrozenModelSpec
    execution_fingerprint: ExecutionSha256
    request_sha256: ExecutionSha256
    error: ExecutionError | None

    @model_validator(mode="after")
    def provenance_and_status_are_consistent(self) -> Self:
        if self.request.task_id != self.task.task_id:
            raise ValueError("request task_id does not match task")
        if self.request.user_prompt != self.task.question:
            raise ValueError("request user_prompt does not match candidate-facing question")
        if self.request_sha256 != self.request.fingerprint:
            raise ValueError("request_sha256 does not match the exact request")
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
        return self.request.seed

    @property
    def harness_id(self) -> str:
        return self.request.harness_id

    @property
    def backend_fingerprint(self) -> str:
        return self.spec.backend_fingerprint

    @property
    def model_fingerprint(self) -> str:
        return self.spec.model_fingerprint

    @property
    def inference_fingerprint(self) -> str:
        return self.spec.inference_fingerprint

    @property
    def runtime_fingerprint(self) -> str:
        return self.spec.runtime_fingerprint

    @property
    def spec_fingerprint(self) -> str:
        return self.spec.fingerprint

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


class ModelExecutionRecord(_FrozenExecutionModel):
    """Atomic result of one fully persisted and terminally accounted execution."""

    schema_version: Literal["3"] = "3"
    execution: ModelExecution
    execution_ref: ArtifactRef
    outcome_ref: ArtifactRef

    @model_validator(mode="after")
    def references_have_exact_media_types(self) -> Self:
        if self.execution_ref.media_type != MODEL_EXECUTION_MEDIA_TYPE:
            raise ValueError("execution_ref declares the wrong media type")
        if self.outcome_ref.media_type != ATTEMPT_OUTCOME_MEDIA_TYPE:
            raise ValueError("outcome_ref declares the wrong media type")
        return self


__all__ = [
    "ATTEMPT_EXECUTION_MEDIA_TYPES",
    "ATTEMPT_OUTCOME_MEDIA_TYPE",
    "ATTEMPT_RESERVATION_MEDIA_TYPE",
    "MODEL_EXECUTION_MEDIA_TYPE",
    "PURE_REFERENCE_EXECUTION_MEDIA_TYPE",
    "AttemptBudget",
    "AttemptDisposition",
    "AttemptLedgerState",
    "AttemptOutcome",
    "AttemptReservation",
    "BackendResponse",
    "BackendTokenUsage",
    "CandidateTask",
    "ExecutionError",
    "ExecutionErrorClass",
    "ExecutionStatus",
    "FrozenModelSpec",
    "InferenceConfig",
    "ModelExecution",
    "ModelExecutionRecord",
    "ModelRequest",
    "ModelUsage",
    "ResolvedHarness",
    "resolve_system_prompt",
]
