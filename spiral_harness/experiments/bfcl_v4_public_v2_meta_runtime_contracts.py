"""Closed contracts for provider-free BFCL V4 public-v2 meta calls.

Trusted wrappers retain DAG identities and source hashes.  The three model-visible
payload contracts deliberately cannot represent task IDs, answers, checker diagnostics,
the full roster, or evidence from another arm, seed, or pipeline.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BfclV4PublicDevelopmentV2Arm,
    BfclV4PublicDevelopmentV2FeedbackView,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationId,
    BfclV4PublicV2MutationMaterialization,
    BfclV4PublicV2MutationProposal,
)
from spiral_harness.core.canonical import canonical_json, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, NonEmptyStr, Sha256
from spiral_harness.execution.contracts import FrozenModelSpec
from spiral_harness.experiments.bfcl_v4_public_v2_executor_contracts import (
    BfclV4PublicV2ProposalDisposition,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationRuntimeBatch,
)
from spiral_harness.providers.openai_native_contracts import (
    FrozenNativeFunctionTool,
    NativeFunctionCallRequest,
)

_PRIVATE_SOURCE_KEYS = frozenset(
    {"task_id", "task_ref", "possible_answer", "answer", "checker_diagnostics", "roster"}
)


class BfclV4PublicV2MetaModel(ImmutableModel):
    """Immutable exact-text base; provider output is never whitespace-normalized."""

    model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=False)


class BfclV4PublicV2MetaControllerKind(StrEnum):
    """The two paid aggregate controller roles in the frozen DAG."""

    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"


class BfclV4PublicV2DiagnosisFailure(StrEnum):
    """Total, sanitized outcomes of one diagnosis native response."""

    NONE = "none"
    NO_VERIFIED_RESPONSE = "no-verified-response"
    INVALID_RESPONSE_CONTRACT = "invalid-response-contract"
    RESPONSE_BINDING_MISMATCH = "response-binding-mismatch"
    TEXT_ONLY = "text-only"
    WRONG_CALL_COUNT = "wrong-call-count"
    ASSISTANT_TEXT_PRESENT = "assistant-text-present"
    WRONG_TOOL = "wrong-tool"
    INVALID_ARGUMENTS = "invalid-arguments"
    MISSING_ARGUMENT = "missing-argument"
    EXTRA_ARGUMENT_FIELDS = "extra-argument-fields"
    ARGUMENT_NOT_TEXT = "argument-not-text"
    EMPTY_DIAGNOSIS = "empty-diagnosis"
    DIAGNOSIS_TOO_LARGE = "diagnosis-too-large"
    INVALID_CONTROL_CHARACTER = "invalid-control-character"
    FORBIDDEN_DELIMITER = "forbidden-delimiter"


class BfclV4PublicV2ProposalFailure(StrEnum):
    """Total, sanitized outcomes of one closed-catalogue proposal response."""

    NONE = "none"
    NO_VERIFIED_RESPONSE = "no-verified-response"
    INVALID_RESPONSE_CONTRACT = "invalid-response-contract"
    RESPONSE_BINDING_MISMATCH = "response-binding-mismatch"
    TEXT_ONLY = "text-only"
    WRONG_CALL_COUNT = "wrong-call-count"
    ASSISTANT_TEXT_PRESENT = "assistant-text-present"
    WRONG_TOOL = "wrong-tool"
    INVALID_ARGUMENTS = "invalid-arguments"
    MISSING_ARGUMENT = "missing-argument"
    EXTRA_ARGUMENT_FIELDS = "extra-argument-fields"
    ARGUMENT_NOT_TEXT = "argument-not-text"
    UNKNOWN_CATALOGUE_ID = "unknown-catalogue-id"
    MATERIALIZATION_INVALID = "materialization-invalid"
    NO_OP = "no-op"


def _canonical_json(value: str, *, label: str) -> object:
    try:
        parsed = json.loads(value)
        if canonical_json(parsed) != value:
            raise ValueError("non-canonical JSON")
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be canonical strict JSON") from error
    return parsed


def _candidate_safe_sources(question_json: str, schemas_json: str) -> None:
    question = _canonical_json(question_json, label="FIT question")
    schemas = _canonical_json(schemas_json, label="FIT function schemas")
    valid_question = (
        isinstance(question, list)
        and len(question) == 1
        and isinstance(question[0], list)
        and bool(question[0])
        and all(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message.get("role") in {"system", "user"}
            and isinstance(message.get("content"), str)
            for message in question[0]
        )
    )
    if not valid_question:
        raise ValueError("FIT question must contain one exact system/user conversation turn")
    if (
        not isinstance(schemas, list)
        or not schemas
        or any(not isinstance(item, dict) or set(item) & _PRIVATE_SOURCE_KEYS for item in schemas)
    ):
        raise ValueError("FIT function schemas are empty or cross a private top-level field")


class BfclV4PublicV2MetaFitTaskProjection(BfclV4PublicV2MetaModel):
    """Trusted question/schema projection for one parent-FIT event.

    ``source_node_id`` stays in the trusted wrapper and is never copied into a
    model-visible observation.
    """

    schema_version: Literal["1"] = "1"
    source_node_id: NonEmptyStr
    source_node_reference_sha256: Sha256
    source_request_payload_sha256: Sha256
    question_json: NonEmptyStr
    function_schemas_json: NonEmptyStr
    question_sha256: Sha256
    function_schemas_sha256: Sha256
    private_task_id_present: Literal[False] = False
    possible_answer_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False
    roster_present: Literal[False] = False
    trusted_projection_only: Literal[True] = True
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _close_projection(self) -> Self:
        _candidate_safe_sources(self.question_json, self.function_schemas_json)
        if self.question_sha256 != sha256_bytes(
            self.question_json.encode("utf-8")
        ) or self.function_schemas_sha256 != sha256_bytes(
            self.function_schemas_json.encode("utf-8")
        ):
            raise ValueError("FIT projection text differs from its source hash")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ScoreBinarySummary(BfclV4PublicV2MetaModel):
    """SCORE's complete label-free, item-free parent-FIT feedback."""

    observation_count: Literal[10] = 10
    binary_correct_count: Annotated[int, Field(ge=0, le=10, strict=True)]
    binary_incorrect_count: Annotated[int, Field(ge=0, le=10, strict=True)]

    @model_validator(mode="after")
    def _close_counts(self) -> Self:
        if self.binary_correct_count + self.binary_incorrect_count != self.observation_count:
            raise ValueError("SCORE binary counts must cover all ten frozen observations")
        return self


class BfclV4PublicV2ScoreDiagnosisPayload(BfclV4PublicV2MetaModel):
    """Only payload shape visible to a SCORE diagnosis model."""

    controller: Literal["diagnosis"] = "diagnosis"
    feedback_view: Literal["fit-aggregate-binary-score-only"] = "fit-aggregate-binary-score-only"
    binary_summary: BfclV4PublicV2ScoreBinarySummary


class BfclV4PublicV2FullVisibleObservation(BfclV4PublicV2MetaModel):
    """One candidate-safe FULL observation with no control-plane identity."""

    observation_index: Annotated[int, Field(ge=0, lt=10, strict=True)]
    question_json: NonEmptyStr
    function_schemas_json: NonEmptyStr
    own_canonical_response: str | None
    binary_grade: bool

    @model_validator(mode="after")
    def _close_visible_item(self) -> Self:
        _candidate_safe_sources(self.question_json, self.function_schemas_json)
        if self.binary_grade and self.own_canonical_response is None:
            raise ValueError("a correct FULL observation requires an own response")
        return self


class BfclV4PublicV2FullDiagnosisPayload(BfclV4PublicV2MetaModel):
    """Only payload shape visible to a FULL diagnosis model."""

    controller: Literal["diagnosis"] = "diagnosis"
    feedback_view: Literal["fit-own-response-binary-and-coarse-failure"] = (
        "fit-own-response-binary-and-coarse-failure"
    )
    fit_observations: Annotated[
        tuple[BfclV4PublicV2FullVisibleObservation, ...],
        Field(min_length=10, max_length=10),
    ]

    @model_validator(mode="after")
    def _close_order(self) -> Self:
        if tuple(item.observation_index for item in self.fit_observations) != tuple(range(10)):
            raise ValueError("FULL visible observations must retain exact evidence order")
        return self


class BfclV4PublicV2VisibleDiagnosisResult(BfclV4PublicV2MetaModel):
    """The sole source-derived value that may enter a proposal prompt."""

    valid: bool
    failure: BfclV4PublicV2DiagnosisFailure
    diagnosis: str | None = None

    @model_validator(mode="after")
    def _close_status(self) -> Self:
        if self.valid is not (self.failure is BfclV4PublicV2DiagnosisFailure.NONE):
            raise ValueError("visible diagnosis validity differs from its failure")
        if self.valid is not (self.diagnosis is not None):
            raise ValueError("only a valid diagnosis may expose diagnosis text")
        return self


class BfclV4PublicV2ProposalPayload(BfclV4PublicV2MetaModel):
    """Only payload shape visible to a proposal model."""

    controller: Literal["proposal"] = "proposal"
    feedback_view: Literal[
        "fit-aggregate-binary-score-only",
        "fit-own-response-binary-and-coarse-failure",
    ]
    diagnosis: BfclV4PublicV2VisibleDiagnosisResult
    mutation_catalogue_ids: Annotated[
        tuple[BfclV4PublicV2MutationId, ...],
        Field(min_length=5, max_length=5),
    ] = tuple(BfclV4PublicV2MutationId)

    @model_validator(mode="after")
    def _close_catalogue(self) -> Self:
        if self.mutation_catalogue_ids != tuple(BfclV4PublicV2MutationId):
            raise ValueError("proposal payload must retain the complete frozen catalogue order")
        return self


BfclV4PublicV2MetaVisiblePayload = (
    BfclV4PublicV2ScoreDiagnosisPayload
    | BfclV4PublicV2FullDiagnosisPayload
    | BfclV4PublicV2ProposalPayload
)


class BfclV4PublicV2MetaEvidenceBinding(BfclV4PublicV2MetaModel):
    """Trusted exact-evidence receipt; this object is never model input."""

    schema_version: Literal["1"] = "1"
    controller_kind: BfclV4PublicV2MetaControllerKind
    arm: Literal[BfclV4PublicDevelopmentV2Arm.SCORE, BfclV4PublicDevelopmentV2Arm.FULL]
    feedback_view: Literal[
        BfclV4PublicDevelopmentV2FeedbackView.SCORE_ONLY,
        BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL,
    ]
    target_node_id: NonEmptyStr
    target_node_reference_sha256: Sha256
    allowed_evidence_node_ids: tuple[NonEmptyStr, ...]
    source_event_fingerprints: tuple[Sha256, ...]
    source_projection_fingerprints: tuple[Sha256, ...]
    campaign_plan_fingerprint: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    exact_allowed_evidence_order_verified: Literal[True] = True
    cross_arm_seed_pipeline_evidence_present: Literal[False] = False
    gate_or_holdout_evidence_present: Literal[False] = False
    private_task_id_model_visible: Literal[False] = False
    possible_answer_model_visible: Literal[False] = False
    checker_diagnostics_model_visible: Literal[False] = False
    roster_model_visible: Literal[False] = False
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _close_cardinality(self) -> Self:
        event_count = len(self.source_event_fingerprints)
        if len(self.allowed_evidence_node_ids) != event_count:
            raise ValueError("evidence IDs and event fingerprints differ in cardinality")
        expected_events = 10 if self.controller_kind is self.controller_kind.DIAGNOSIS else 1
        if event_count != expected_events:
            raise ValueError("meta evidence count differs from the frozen controller DAG")
        expected_projections = (
            10
            if self.controller_kind is self.controller_kind.DIAGNOSIS
            and self.feedback_view is BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL
            else 0
        )
        if len(self.source_projection_fingerprints) != expected_projections:
            raise ValueError("meta task projections differ from the authorized feedback view")
        if (
            len(set(self.allowed_evidence_node_ids)) != event_count
            or len(set(self.source_event_fingerprints)) != event_count
        ):
            raise ValueError("meta evidence sources must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2MetaPrompt(BfclV4PublicV2MetaModel):
    """Deterministic two-message prompt with one native submit tool."""

    schema_version: Literal["1"] = "1"
    controller_kind: BfclV4PublicV2MetaControllerKind
    arm: Literal[BfclV4PublicDevelopmentV2Arm.SCORE, BfclV4PublicDevelopmentV2Arm.FULL]
    feedback_view: Literal[
        BfclV4PublicDevelopmentV2FeedbackView.SCORE_ONLY,
        BfclV4PublicDevelopmentV2FeedbackView.CANDIDATE_SAFE_FULL,
    ]
    pipeline_index: Annotated[int, Field(ge=0, le=2, strict=True)]
    evidence_binding_fingerprint: Sha256
    model_visible_payload: BfclV4PublicV2MetaVisiblePayload
    model_visible_payload_fingerprint: Sha256
    system_prompt: NonEmptyStr
    user_prompt: NonEmptyStr
    user_prompt_sha256: Sha256
    submit_tool: FrozenNativeFunctionTool
    submit_tool_fingerprint: Sha256
    output_grammar: Literal[
        "one-native-diagnosis-call-v2",
        "one-closed-catalogue-mutation-call-v2",
    ]
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _close_prompt_hashes(self) -> Self:
        if (
            self.model_visible_payload_fingerprint != canonical_sha256(self.model_visible_payload)
            or self.user_prompt_sha256 != sha256_bytes(self.user_prompt.encode("utf-8"))
            or self.submit_tool_fingerprint != canonical_sha256(self.submit_tool)
        ):
            raise ValueError("meta prompt content differs from its fingerprints")
        if self.controller_kind is self.controller_kind.DIAGNOSIS:
            expected_type = {
                BfclV4PublicDevelopmentV2Arm.SCORE: BfclV4PublicV2ScoreDiagnosisPayload,
                BfclV4PublicDevelopmentV2Arm.FULL: BfclV4PublicV2FullDiagnosisPayload,
            }[self.arm]
        else:
            expected_type = BfclV4PublicV2ProposalPayload
        if type(self.model_visible_payload) is not expected_type:
            raise ValueError("model-visible payload differs from controller kind and arm")
        if self.model_visible_payload.feedback_view != self.feedback_view.value:
            raise ValueError("model-visible payload differs from the authorized feedback view")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2MetaRequestMaterialization(BfclV4PublicV2MetaModel):
    """Complete provider input plus private lineage; no provider is invoked."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    node_schedule_content_sha256: Sha256
    mutation_catalog_fingerprint: Sha256
    target_node_id: NonEmptyStr
    target_node_reference_sha256: Sha256
    runtime_fingerprint: Sha256
    semantic_release_ref: ArtifactRef
    semantic_release_fingerprint: Sha256
    model_spec: FrozenModelSpec
    model_spec_fingerprint: Sha256
    meta_runtime_source_fingerprint: Sha256
    source_diagnosis_result_fingerprint: Sha256 | None = None
    evidence_binding: BfclV4PublicV2MetaEvidenceBinding
    prompt: BfclV4PublicV2MetaPrompt
    native_request: NativeFunctionCallRequest
    native_request_fingerprint: Sha256
    provider_request_payload_sha256: Sha256
    provider_calls: Literal[0] = 0
    request_materialized_only: Literal[True] = True
    provider_seed_honoring_attested: Literal[False] = False
    served_weights_attested: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    benchmark_result_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_request(self) -> Self:
        if (
            self.semantic_release_ref.sha256 != self.semantic_release_fingerprint
            or self.model_spec_fingerprint != self.model_spec.fingerprint
            or self.evidence_binding.target_node_id != self.target_node_id
            or self.evidence_binding.target_node_reference_sha256
            != self.target_node_reference_sha256
            or self.prompt.evidence_binding_fingerprint != self.evidence_binding.fingerprint
            or self.prompt.controller_kind is not self.evidence_binding.controller_kind
            or self.prompt.arm is not self.evidence_binding.arm
            or self.prompt.feedback_view is not self.evidence_binding.feedback_view
            or self.native_request_fingerprint != self.native_request.fingerprint
            or self.provider_request_payload_sha256 != self.native_request.fingerprint
        ):
            raise ValueError("meta request differs from its private lineage or content hashes")
        has_diagnosis_source = self.source_diagnosis_result_fingerprint is not None
        if has_diagnosis_source is not (
            self.prompt.controller_kind is BfclV4PublicV2MetaControllerKind.PROPOSAL
        ):
            raise ValueError("only a proposal request may bind one diagnosis result")
        messages = self.native_request.messages
        if (
            len(messages) != 2
            or tuple(message.role for message in messages) != ("system", "user")
            or tuple(message.content for message in messages)
            != (self.prompt.system_prompt, self.prompt.user_prompt)
            or self.native_request.task_required_tools != (self.prompt.submit_tool,)
            or self.native_request.harness_added_tools
            or self.native_request.requested_model != self.model_spec.model
            or self.native_request.inference != self.model_spec.inference
        ):
            raise ValueError("native meta request differs from its prompt, tool, or model spec")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2DiagnosisParseResult(BfclV4PublicV2MetaModel):
    """Strict total parse result for one diagnosis call."""

    schema_version: Literal["1"] = "1"
    target_node_id: NonEmptyStr
    request_materialization_fingerprint: Sha256
    native_response_fingerprint: Sha256 | None
    journal_canonical_response: str | None
    valid: bool
    failure: BfclV4PublicV2DiagnosisFailure
    diagnosis_text: str | None = None
    diagnosis_text_sha256: Sha256 | None = None
    automatic_retry_used: Literal[False] = False
    output_repair_used: Literal[False] = False
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _close_result(self) -> Self:
        parsed = (self.diagnosis_text, self.diagnosis_text_sha256)
        if self.valid:
            if self.failure is not self.failure.NONE or any(value is None for value in parsed):
                raise ValueError("valid diagnosis requires exact text and no failure")
            assert self.diagnosis_text is not None
            if self.diagnosis_text_sha256 != sha256_bytes(self.diagnosis_text.encode("utf-8")):
                raise ValueError("diagnosis text differs from its fingerprint")
        elif self.failure is self.failure.NONE or any(value is not None for value in parsed):
            raise ValueError("invalid diagnosis requires one failure and no parsed text")
        no_response = {
            self.failure.NO_VERIFIED_RESPONSE,
            self.failure.INVALID_RESPONSE_CONTRACT,
        }
        if (self.native_response_fingerprint is None) is not (self.failure in no_response):
            raise ValueError("diagnosis response fingerprint differs from failure class")
        if (self.journal_canonical_response is None) is not (
            self.native_response_fingerprint is None
        ):
            raise ValueError("diagnosis journal response differs from verified response presence")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ProposalParseResult(BfclV4PublicV2MetaModel):
    """Strict parse and deterministic materialization before duplicate closure."""

    schema_version: Literal["1"] = "1"
    target_node_id: NonEmptyStr
    pipeline_index: Annotated[int, Field(ge=0, le=2, strict=True)]
    request_materialization_fingerprint: Sha256
    diagnosis_result_fingerprint: Sha256
    diagnosis_valid: bool
    native_response_fingerprint: Sha256 | None
    journal_canonical_response: str | None
    valid: bool
    failure: BfclV4PublicV2ProposalFailure
    mutation_id: BfclV4PublicV2MutationId | None = None
    proposal: BfclV4PublicV2MutationProposal | None = None
    materialization: BfclV4PublicV2MutationMaterialization | None = None
    runtime_batch: BfclV4PublicV2MutationRuntimeBatch | None = None
    automatic_retry_used: Literal[False] = False
    output_repair_used: Literal[False] = False
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _close_result(self) -> Self:
        parsed = (self.mutation_id, self.proposal, self.materialization, self.runtime_batch)
        if self.valid:
            if self.failure is not self.failure.NONE or any(value is None for value in parsed):
                raise ValueError("valid proposal requires complete deterministic materialization")
            assert self.mutation_id is not None
            assert self.proposal is not None
            assert self.materialization is not None
            assert self.runtime_batch is not None
            if (
                self.proposal.catalogue_id is not self.mutation_id
                or self.materialization.proposal != self.proposal
                or self.runtime_batch.materialization != self.materialization
            ):
                raise ValueError("proposal ID, materialization, and runtime batch differ")
        elif self.failure is self.failure.NONE or any(value is not None for value in parsed):
            raise ValueError("invalid proposal requires one failure and no candidate artifacts")
        no_response = {
            self.failure.NO_VERIFIED_RESPONSE,
            self.failure.INVALID_RESPONSE_CONTRACT,
        }
        if (self.native_response_fingerprint is None) is not (self.failure in no_response):
            raise ValueError("proposal response fingerprint differs from failure class")
        if (self.journal_canonical_response is None) is not (
            self.native_response_fingerprint is None
        ):
            raise ValueError("proposal journal response differs from verified response presence")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2ResolvedAtomicMutation(BfclV4PublicV2MetaModel):
    """One proposal after deterministic three-pipeline duplicate closure."""

    schema_version: Literal["1"] = "1"
    pipeline_index: Annotated[int, Field(ge=0, le=2, strict=True)]
    proposal_node_id: NonEmptyStr
    parse_result: BfclV4PublicV2ProposalParseResult
    parse_result_fingerprint: Sha256
    disposition: BfclV4PublicV2ProposalDisposition
    mutation_id: BfclV4PublicV2MutationId | None = None
    duplicate_of_pipeline_index: Annotated[int, Field(ge=0, le=1, strict=True)] | None = None
    admitted_materialization: BfclV4PublicV2MutationMaterialization | None = None
    admitted_runtime_batch: BfclV4PublicV2MutationRuntimeBatch | None = None
    candidate_artifact_sha256: Sha256 | None = None
    provider_calls: Literal[0] = 0
    duplicate_replacement_or_backfill_used: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_disposition(self) -> Self:
        if (
            self.pipeline_index != self.parse_result.pipeline_index
            or self.proposal_node_id != self.parse_result.target_node_id
            or self.parse_result_fingerprint != self.parse_result.fingerprint
        ):
            raise ValueError("resolved proposal differs from its parse result")
        admitted = (
            self.admitted_materialization,
            self.admitted_runtime_batch,
            self.candidate_artifact_sha256,
        )
        if self.disposition is BfclV4PublicV2ProposalDisposition.VALID:
            if (
                any(value is None for value in admitted)
                or self.duplicate_of_pipeline_index is not None
            ):
                raise ValueError("valid proposal requires one admitted atomic runtime")
            if not self.parse_result.valid or not self.parse_result.diagnosis_valid:
                raise ValueError("invalid diagnosis or parse cannot become a valid proposal")
        elif any(value is not None for value in admitted):
            raise ValueError("non-valid proposal cannot expose an admitted runtime")
        if self.disposition is BfclV4PublicV2ProposalDisposition.DUPLICATE:
            if (
                not self.parse_result.valid
                or not self.parse_result.diagnosis_valid
                or self.duplicate_of_pipeline_index is None
                or self.duplicate_of_pipeline_index >= self.pipeline_index
            ):
                raise ValueError("duplicate must cite an earlier otherwise-admissible proposal")
        elif self.duplicate_of_pipeline_index is not None:
            raise ValueError("only a duplicate proposal may cite an earlier pipeline")
        if self.disposition is BfclV4PublicV2ProposalDisposition.PROVIDER_FAILURE and (
            self.parse_result.failure is not BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE
        ):
            raise ValueError("provider-failure disposition requires no verified response")
        if self.disposition is BfclV4PublicV2ProposalDisposition.NO_OP and (
            self.parse_result.failure is not BfclV4PublicV2ProposalFailure.NO_OP
        ):
            raise ValueError("no-op disposition requires a typed no-op parse failure")
        if self.disposition is BfclV4PublicV2ProposalDisposition.INVALID and (
            self.parse_result.failure
            in {
                BfclV4PublicV2ProposalFailure.NO_VERIFIED_RESPONSE,
                BfclV4PublicV2ProposalFailure.NO_OP,
            }
        ):
            raise ValueError("invalid disposition cannot hide provider failure or no-op")
        if self.mutation_id is not self.parse_result.mutation_id:
            raise ValueError("resolved mutation ID differs from parsed mutation ID")
        return self


class BfclV4PublicV2ResolvedProposalBatch(BfclV4PublicV2MetaModel):
    """Trusted atomic closure for exactly three parallel proposal pipelines."""

    schema_version: Literal["1"] = "1"
    campaign_plan_fingerprint: Sha256
    runtime_fingerprint: Sha256
    semantic_release_fingerprint: Sha256
    replicate_id: NonEmptyStr
    arm: Literal[BfclV4PublicDevelopmentV2Arm.SCORE, BfclV4PublicDevelopmentV2Arm.FULL]
    proposals: Annotated[
        tuple[BfclV4PublicV2ResolvedAtomicMutation, ...],
        Field(min_length=3, max_length=3),
    ]
    duplicate_resolution_rule: Literal["lowest-pipeline-index-wins-v1"] = (
        "lowest-pipeline-index-wins-v1"
    )
    all_three_proposals_frozen_before_resolution: Literal[True] = True
    proposal_models_received_duplicate_feedback: Literal[False] = False
    provider_calls: Literal[0] = 0
    score_bearing_execution_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _close_batch(self) -> Self:
        if tuple(item.pipeline_index for item in self.proposals) != (0, 1, 2):
            raise ValueError("resolved proposal batch must retain pipeline order 0,1,2")
        first: dict[BfclV4PublicV2MutationId, int] = {}
        for item in self.proposals:
            parsed = item.parse_result
            otherwise_admissible = parsed.valid and parsed.diagnosis_valid
            expected_duplicate = first.get(parsed.mutation_id) if otherwise_admissible else None
            if expected_duplicate is not None:
                if (
                    item.disposition is not BfclV4PublicV2ProposalDisposition.DUPLICATE
                    or item.duplicate_of_pipeline_index != expected_duplicate
                ):
                    raise ValueError("duplicate resolution differs from lowest pipeline index")
            elif otherwise_admissible:
                if item.disposition is not BfclV4PublicV2ProposalDisposition.VALID:
                    raise ValueError("first admissible mutation ID must remain valid")
                assert parsed.mutation_id is not None
                first[parsed.mutation_id] = item.pipeline_index
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [name for name in globals() if name.startswith("Bfcl")]
