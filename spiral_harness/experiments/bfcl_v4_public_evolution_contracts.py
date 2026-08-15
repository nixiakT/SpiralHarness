"""Typed information boundary for the BFCL V4 public-pilot evolution step.

The contracts in this module are intentionally public-development-only.  They
join candidate-visible BFCL questions, a model's own prediction, and the
coarse FULL feedback projection without making grader answers representable.
They also make malformed-candidate fallback an auditable value rather than a
runner convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicPrediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_projections import (
    FIT_TASK_IDS,
    BfclV4FullFitFeedback,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4PilotSplit,
    BfclV4PublicPilotTask,
)
from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.core.models import Sha256
from spiral_harness.providers.openai_native_contracts import FrozenNativeFunctionTool

BFCL_V4_PUBLIC_EVOLUTION_SCOPE = "public-development-partial-bfcl-pilot"
BFCL_V4_INVALID_SLOT_POLICY = "parent-fallback-consumes-all-frozen-slots"
BFCL_V4_INVALID_SELECTION_POLICY = "forced-rollback"


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("Unicode surrogate code points are forbidden")
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _reject_surrogates(getattr(value, name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _reject_surrogates(item)


class BfclV4EvolutionModel(BaseModel):
    """Strict immutable model that preserves prompt whitespace byte-for-byte."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        str_strip_whitespace=False,
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _text_is_unicode_scalar(cls, value: Any) -> Any:
        _reject_surrogates(value)
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4AdaptiveArm(StrEnum):
    """The two matched adaptive arms in the public pilot."""

    SCORE = "score"
    FULL = "full"


class BfclV4DiagnosisFailure(StrEnum):
    """Sanitized, exhaustive diagnosis parser outcomes."""

    NONE = "none"
    NO_VERIFIED_RESPONSE = "no-verified-response"
    INVALID_RESPONSE_CONTRACT = "invalid-response-contract"
    TEXT_ONLY = "text-only"
    WRONG_CALL_COUNT = "wrong-call-count"
    ASSISTANT_TEXT_PRESENT = "assistant-text-present"
    WRONG_TOOL = "wrong-tool"
    INVALID_ARGUMENTS = "invalid-arguments"
    EXTRA_ARGUMENT_FIELDS = "extra-argument-fields"
    MISSING_ARGUMENT = "missing-argument"
    ARGUMENT_NOT_TEXT = "argument-not-text"
    OUTPUT_TOO_LARGE = "output-too-large"
    INVALID_UNICODE = "invalid-unicode"
    INVALID_CONTROL_CHARACTER = "invalid-control-character"
    INVALID_ENVELOPE = "invalid-envelope"
    EMPTY_DIAGNOSIS = "empty-diagnosis"
    DIAGNOSIS_TOO_LARGE = "diagnosis-too-large"
    FORBIDDEN_DELIMITER = "forbidden-delimiter"


class BfclV4CandidateParseFailure(StrEnum):
    """Sanitized, exhaustive proposal parser outcomes."""

    NONE = "none"
    NO_VERIFIED_RESPONSE = "no-verified-response"
    INVALID_RESPONSE_CONTRACT = "invalid-response-contract"
    TEXT_ONLY = "text-only"
    WRONG_CALL_COUNT = "wrong-call-count"
    ASSISTANT_TEXT_PRESENT = "assistant-text-present"
    WRONG_TOOL = "wrong-tool"
    INVALID_ARGUMENTS = "invalid-arguments"
    EXTRA_ARGUMENT_FIELDS = "extra-argument-fields"
    MISSING_ARGUMENT = "missing-argument"
    ARGUMENT_NOT_TEXT = "argument-not-text"
    OUTPUT_TOO_LARGE = "output-too-large"
    INVALID_UNICODE = "invalid-unicode"
    INVALID_CONTROL_CHARACTER = "invalid-control-character"
    INVALID_ENVELOPE = "invalid-envelope"
    EMPTY_STRATEGY = "empty-strategy"
    STRATEGY_TOO_LARGE = "strategy-too-large"
    FORBIDDEN_DELIMITER = "forbidden-delimiter"
    NO_OP = "no-op"


class BfclV4CandidateResolutionFailure(StrEnum):
    """Why a proposal is or is not eligible for gate selection."""

    NONE = "none"
    DIAGNOSIS_INVALID = "diagnosis-invalid"
    CANDIDATE_PARSE_INVALID = "candidate-parse-invalid"


class BfclV4FullFitDiagnosisObservation(BfclV4EvolutionModel):
    """One source-bound FULL view: task, own prediction, binary/coarse feedback."""

    schema_version: Literal["1"] = "1"
    task: BfclV4PublicPilotTask
    own_prediction: BfclV4PublicPrediction
    feedback: BfclV4FullFitFeedback

    @model_validator(mode="after")
    def _join_candidate_safe_planes(self) -> Self:
        if self.task.split is not BfclV4PilotSplit.FIT:
            raise ValueError("FULL diagnosis observations are restricted to FIT")
        if self.task.task_id not in FIT_TASK_IDS:
            raise ValueError("FULL diagnosis task is outside the frozen FIT roster")
        if not (self.task.task_id == self.own_prediction.task_id == self.feedback.task_id):
            raise ValueError("FULL diagnosis task, prediction, and feedback IDs differ")
        if self.own_prediction.fingerprint != self.feedback.own_prediction_reference_sha256:
            raise ValueError("FULL feedback belongs to another prediction")
        return self


class BfclV4FullFitDiagnosisBatch(BfclV4EvolutionModel):
    """Exactly the five ordered, candidate-safe parent FIT observations."""

    schema_version: Literal["1"] = "1"
    observations: Annotated[
        tuple[BfclV4FullFitDiagnosisObservation, ...],
        Field(min_length=5, max_length=5),
    ]
    information_scope: Literal[
        "five-public-fit-questions-schemas-own-predictions-binary-coarse-feedback"
    ] = "five-public-fit-questions-schemas-own-predictions-binary-coarse-feedback"
    candidate_visible: Literal[True] = True
    partial_evaluation: Literal[True] = True
    possible_answer_data_present: Literal[False] = False
    grader_diagnostics_present: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _exact_fit_roster_order(self) -> Self:
        task_ids = tuple(item.task.task_id for item in self.observations)
        if task_ids != FIT_TASK_IDS:
            raise ValueError("FULL diagnosis batch must use the exact frozen FIT order")
        return self


class BfclV4DiagnosisPrompt(BfclV4EvolutionModel):
    """Exact same-grammar diagnosis request with arm-specific authorized input."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4AdaptiveArm
    feedback_view: Literal["score-only", "candidate-safe-full"]
    system_prompt: Annotated[str, Field(min_length=1)]
    user_prompt: Annotated[str, Field(min_length=1)]
    parent_system_prompt: Annotated[str, Field(min_length=1)]
    parent_system_prompt_sha256: Sha256
    authorized_input_sha256: Sha256
    submit_tool: FrozenNativeFunctionTool
    submit_tool_fingerprint: Sha256
    output_grammar: Literal["one-submit-bfcl-diagnosis-native-call-v1"] = (
        "one-submit-bfcl-diagnosis-native-call-v1"
    )
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = (
        BFCL_V4_PUBLIC_EVOLUTION_SCOPE
    )
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_parent_and_view(self) -> Self:
        if sha256_bytes(self.parent_system_prompt.encode("utf-8")) != (
            self.parent_system_prompt_sha256
        ):
            raise ValueError("diagnosis parent prompt hash differs from exact text")
        expected = {
            BfclV4AdaptiveArm.SCORE: "score-only",
            BfclV4AdaptiveArm.FULL: "candidate-safe-full",
        }[self.arm]
        if self.feedback_view != expected:
            raise ValueError("diagnosis feedback view differs from its arm")
        if canonical_sha256(self.submit_tool) != self.submit_tool_fingerprint:
            raise ValueError("diagnosis submit-tool fingerprint differs from its schema")
        return self


class BfclV4DiagnosisParseResult(BfclV4EvolutionModel):
    """Total parser result; invalid model output is represented, not repaired."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4AdaptiveArm
    diagnosis_prompt_fingerprint: Sha256
    native_response_fingerprint: Sha256 | None
    valid: bool
    failure: BfclV4DiagnosisFailure
    diagnosis_text: str | None = None
    diagnosis_text_sha256: Sha256 | None = None
    automatic_retry_used: Literal[False] = False
    output_repair_used: Literal[False] = False

    @model_validator(mode="after")
    def _validity_shape(self) -> Self:
        text_fields = (self.diagnosis_text, self.diagnosis_text_sha256)
        if self.valid:
            if self.failure is not BfclV4DiagnosisFailure.NONE or any(
                item is None for item in text_fields
            ):
                raise ValueError("valid diagnosis requires exact parsed text and no failure")
            assert self.diagnosis_text is not None
            if sha256_bytes(self.diagnosis_text.encode("utf-8")) != (self.diagnosis_text_sha256):
                raise ValueError("diagnosis text hash differs from exact text")
        elif self.failure is BfclV4DiagnosisFailure.NONE or any(
            item is not None for item in text_fields
        ):
            raise ValueError("invalid diagnosis requires one failure and no parsed text")
        no_response_failures = {
            BfclV4DiagnosisFailure.NO_VERIFIED_RESPONSE,
            BfclV4DiagnosisFailure.INVALID_RESPONSE_CONTRACT,
        }
        if (self.native_response_fingerprint is None) != (self.failure in no_response_failures):
            raise ValueError("diagnosis response fingerprint differs from response validity")
        return self


class BfclV4ProposalPrompt(BfclV4EvolutionModel):
    """Exact proposal request bound to the diagnosis parse result, valid or not."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4AdaptiveArm
    feedback_view: Literal["score-only", "candidate-safe-full"]
    system_prompt: Annotated[str, Field(min_length=1)]
    user_prompt: Annotated[str, Field(min_length=1)]
    parent_system_prompt: Annotated[str, Field(min_length=1)]
    parent_system_prompt_sha256: Sha256
    diagnosis_result_fingerprint: Sha256
    diagnosis_valid: bool
    submit_tool: FrozenNativeFunctionTool
    submit_tool_fingerprint: Sha256
    output_grammar: Literal["one-submit-bfcl-candidate-native-call-v1"] = (
        "one-submit-bfcl-candidate-native-call-v1"
    )
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = (
        BFCL_V4_PUBLIC_EVOLUTION_SCOPE
    )
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _bind_parent_and_view(self) -> Self:
        if sha256_bytes(self.parent_system_prompt.encode("utf-8")) != (
            self.parent_system_prompt_sha256
        ):
            raise ValueError("proposal parent prompt hash differs from exact text")
        expected = {
            BfclV4AdaptiveArm.SCORE: "score-only",
            BfclV4AdaptiveArm.FULL: "candidate-safe-full",
        }[self.arm]
        if self.feedback_view != expected:
            raise ValueError("proposal feedback view differs from its arm")
        if canonical_sha256(self.submit_tool) != self.submit_tool_fingerprint:
            raise ValueError("candidate submit-tool fingerprint differs from its schema")
        return self


class BfclV4CandidateParseResult(BfclV4EvolutionModel):
    """Total strict-parser result for one model-authored strategy appendix."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4AdaptiveArm
    proposal_prompt_fingerprint: Sha256
    native_response_fingerprint: Sha256 | None
    valid: bool
    failure: BfclV4CandidateParseFailure
    strategy_text: str | None = None
    strategy_text_sha256: Sha256 | None = None
    candidate_system_prompt: str | None = None
    candidate_system_prompt_sha256: Sha256 | None = None
    automatic_retry_used: Literal[False] = False
    output_repair_used: Literal[False] = False

    @model_validator(mode="after")
    def _validity_shape(self) -> Self:
        parsed = (
            self.strategy_text,
            self.strategy_text_sha256,
            self.candidate_system_prompt,
            self.candidate_system_prompt_sha256,
        )
        if self.valid:
            if self.failure is not BfclV4CandidateParseFailure.NONE or any(
                item is None for item in parsed
            ):
                raise ValueError("valid candidate requires exact parsed content and no failure")
            assert self.strategy_text is not None
            assert self.candidate_system_prompt is not None
            if sha256_bytes(self.strategy_text.encode("utf-8")) != self.strategy_text_sha256:
                raise ValueError("strategy hash differs from exact text")
            if sha256_bytes(self.candidate_system_prompt.encode("utf-8")) != (
                self.candidate_system_prompt_sha256
            ):
                raise ValueError("candidate prompt hash differs from exact text")
        elif self.failure is BfclV4CandidateParseFailure.NONE or any(
            item is not None for item in parsed
        ):
            raise ValueError("invalid candidate requires one failure and no parsed content")
        no_response_failures = {
            BfclV4CandidateParseFailure.NO_VERIFIED_RESPONSE,
            BfclV4CandidateParseFailure.INVALID_RESPONSE_CONTRACT,
        }
        if (self.native_response_fingerprint is None) != (self.failure in no_response_failures):
            raise ValueError("candidate response fingerprint differs from response validity")
        return self


class BfclV4CandidateResolution(BfclV4EvolutionModel):
    """Candidate admission plus exact-parent fallback and forced-rollback state."""

    schema_version: Literal["1"] = "1"
    arm: BfclV4AdaptiveArm
    diagnosis_result: BfclV4DiagnosisParseResult
    proposal_prompt: BfclV4ProposalPrompt
    candidate_parse_result: BfclV4CandidateParseResult
    resolution_failure: BfclV4CandidateResolutionFailure
    candidate_admissible: bool
    parent_system_prompt: Annotated[str, Field(min_length=1)]
    parent_system_prompt_sha256: Sha256
    evaluation_system_prompt: Annotated[str, Field(min_length=1)]
    evaluation_system_prompt_sha256: Sha256
    executed_harness_variant: Literal["parent", "candidate"]
    exact_parent_fallback_used: bool
    forced_rollback: bool
    selection_eligibility: Literal["gate-pending", "forced-rollback"]
    invalid_candidate_slot_policy: Literal["parent-fallback-consumes-all-frozen-slots"] = (
        BFCL_V4_INVALID_SLOT_POLICY
    )
    invalid_candidate_selection_policy: Literal["forced-rollback"] = (
        BFCL_V4_INVALID_SELECTION_POLICY
    )
    all_frozen_candidate_and_gate_slots_must_execute: Literal[True] = True
    evidence_scope: Literal["public-development-partial-bfcl-pilot"] = (
        BFCL_V4_PUBLIC_EVOLUTION_SCOPE
    )
    hidden_test_evidence: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_resolution(self) -> Self:
        if self.arm is not self.diagnosis_result.arm or self.arm is not self.proposal_prompt.arm:
            raise ValueError("diagnosis, proposal, and resolution arms differ")
        if self.arm is not self.candidate_parse_result.arm:
            raise ValueError("candidate parser result belongs to another arm")
        if self.proposal_prompt.diagnosis_result_fingerprint != (self.diagnosis_result.fingerprint):
            raise ValueError("proposal is bound to another diagnosis result")
        if self.proposal_prompt.diagnosis_valid is not self.diagnosis_result.valid:
            raise ValueError("proposal diagnosis-valid flag differs from parser result")
        if self.candidate_parse_result.proposal_prompt_fingerprint != (
            self.proposal_prompt.fingerprint
        ):
            raise ValueError("candidate parser result belongs to another proposal prompt")
        if self.parent_system_prompt != self.proposal_prompt.parent_system_prompt:
            raise ValueError("resolution parent differs from the proposal parent")
        if sha256_bytes(self.parent_system_prompt.encode("utf-8")) != (
            self.parent_system_prompt_sha256
        ):
            raise ValueError("resolution parent prompt hash differs from exact text")
        if sha256_bytes(self.evaluation_system_prompt.encode("utf-8")) != (
            self.evaluation_system_prompt_sha256
        ):
            raise ValueError("evaluation prompt hash differs from exact text")

        admissible = self.diagnosis_result.valid and self.candidate_parse_result.valid
        if self.candidate_admissible is not admissible:
            raise ValueError("candidate admissibility differs from both parser outcomes")
        expected_failure = (
            BfclV4CandidateResolutionFailure.NONE
            if admissible
            else (
                BfclV4CandidateResolutionFailure.DIAGNOSIS_INVALID
                if not self.diagnosis_result.valid
                else BfclV4CandidateResolutionFailure.CANDIDATE_PARSE_INVALID
            )
        )
        if self.resolution_failure is not expected_failure:
            raise ValueError("candidate resolution failure has the wrong precedence")

        if admissible:
            candidate = self.candidate_parse_result.candidate_system_prompt
            if candidate is None or self.evaluation_system_prompt != candidate:
                raise ValueError("admissible candidate must be the exact evaluation prompt")
            expected_state = ("candidate", False, False, "gate-pending")
        else:
            if self.evaluation_system_prompt != self.parent_system_prompt:
                raise ValueError("invalid candidate must evaluate the exact parent prompt")
            expected_state = ("parent", True, True, "forced-rollback")
        actual_state = (
            self.executed_harness_variant,
            self.exact_parent_fallback_used,
            self.forced_rollback,
            self.selection_eligibility,
        )
        if actual_state != expected_state:
            raise ValueError("fallback or selection state differs from candidate validity")
        return self


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
