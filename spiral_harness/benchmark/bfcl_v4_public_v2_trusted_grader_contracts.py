"""Fail-closed contracts for the BFCL V4 public-development v2 grader.

The candidate plane supplies a native request/response pair and structural
campaign lineage only.  Possible answers and checker diagnostics are never
representable by these contracts.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spiral_harness.benchmark.bfcl_v4_identity import BFCL_V4_UPSTREAM_COMMIT
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BfclV4PublicDevelopmentV2DagNode,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2NodeRequestLineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4PublicDevelopmentV2Split,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, Sha256
from spiral_harness.providers.openai_native_contracts import (
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)

BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL = "spiral-bfcl-v4-public-development-v2-trusted-grader/v1"
BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_PROTOCOL = (
    "spiral-bfcl-v4-public-development-v2-trusted-grader-worker/v1"
)
BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_FAILURE_PROTOCOL = (
    "spiral-bfcl-v4-public-development-v2-trusted-grader-worker-failure/v1"
)
BFCL_V4_PUBLIC_V2_EVALUATION_BARRIER_PROTOCOL = (
    "spiral-bfcl-v4-public-development-v2-evaluation-barrier/v1"
)
BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_PROTOCOL = (
    "spiral-bfcl-v4-public-development-v2-evaluation-unlock/v1"
)
BFCL_V4_PUBLIC_V2_EVALUATION_AUTHORITY_KEY_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-evaluation-authority-key/v1"
)
BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_HMAC_DOMAIN = (
    "spiral-bfcl-v4-public-development-v2-evaluation-unlock-hmac/v1"
)
BFCL_V4_PUBLIC_V2_MINIMUM_EVALUATION_AUTHORITY_SECRET_BYTES = 32
BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256 = (
    "cf538a0dc09f515bd0cefee3f7b81f8dcc2c904a386de6d1da5013e2f5e6300d"
)
BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256 = (
    "939eec4aced36ae1c96aa35c5fd67e1db19c17be7cfd29ed637b39d807a0b03d"
)
BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256 = (
    "8b192d2e3f30a4508079996059d2452d6eef61d318ec4afd525cff43710a8993"
)
BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT = canonical_sha256(
    {
        "domain": "spiral-bfcl-v4-public-development-v2-trusted-authority/v1",
        "upstream_commit": BFCL_V4_UPSTREAM_COMMIT,
        "manifest_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
        "checker_source_bundle_sha256": BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256,
        "answer_authority": "pinned-git-object-only-not-released",
    }
)
BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT = canonical_sha256(
    {
        "domain": "spiral-bfcl-v4-public-development-v2-trusted-grader-implementation/v1",
        "protocol": BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL,
        "worker_protocol": BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_PROTOCOL,
        "upstream_commit": BFCL_V4_UPSTREAM_COMMIT,
        "checker_source_bundle_sha256": BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256,
        "worker_source_sha256": BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256,
        "worker_primitives_source_sha256": (
            BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256
        ),
        "receipt_projection": "binary-only-no-task-id-answer-identity-or-diagnostics",
    }
)


class BfclV4PublicV2TrustedGradeRequest(ImmutableModel):
    """One exact native response bound to a frozen model-call DAG node."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL
    )
    node: BfclV4PublicDevelopmentV2DagNode
    node_reference_sha256: Sha256
    request_lineage: BfclV4PublicDevelopmentV2NodeRequestLineage
    task_payload_sha256: Sha256
    request: NativeFunctionCallRequest
    request_fingerprint: Sha256
    request_payload_sha256: Sha256
    raw_response: NativeFunctionCallResponse
    response_fingerprint: Sha256
    caller_supplied_task_id_present: Literal[False] = False
    caller_supplied_split_present: Literal[False] = False
    caller_supplied_answer_present: Literal[False] = False
    possible_answer_identity_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_native_lineage(self) -> Self:
        node = self.node
        lineage = self.request_lineage
        if not node.consumes_model_call or node.task_ref is None:
            raise ValueError("trusted grading requires a task-bearing model-call node")
        if self.node_reference_sha256 != canonical_sha256(node):
            raise ValueError("trusted grade node reference changed")
        if (
            lineage.campaign_plan_fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
            or lineage.node_schedule_content_sha256
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
            or lineage.node_id != node.node_id
            or lineage.node_reference_sha256 != self.node_reference_sha256
            or lineage.campaign_call_slot != node.campaign_call_slot
            or lineage.provider_seed_u63 != node.provider_seed_u63
        ):
            raise ValueError("trusted grade request lineage differs from its node")
        if node.provider_seed_u63 is None or self.request.seed != node.provider_seed_u63:
            raise ValueError("native request seed differs from the frozen node")
        if (
            self.request_fingerprint != self.request.fingerprint
            or self.request_payload_sha256 != self.request.fingerprint
        ):
            raise ValueError("native request fingerprint changed")
        response = self.raw_response
        if (
            self.response_fingerprint != response.fingerprint
            or response.request_fingerprint != self.request_fingerprint
            or response.serializer_fingerprint != self.request.serializer_fingerprint
            or response.parser_fingerprint != self.request.parser_fingerprint
            or response.transport_fingerprint != self.request.transport_fingerprint
            or response.tools_fingerprint != self.request.tools_fingerprint
        ):
            raise ValueError("native response lineage differs from its exact request")
        if response.usage.output_tokens > self.request.inference.max_output_tokens:
            raise ValueError("native response exceeds the request output-token ceiling")
        tools_by_wire = {
            tool.wire_name: tool
            for tool in self.request.task_required_tools + self.request.harness_added_tools
        }
        response_call_ids: set[str] = set()
        historical_call_ids = self.request.historical_call_ids
        for call in response.tool_calls:
            tool = tools_by_wire.get(call.wire_name)
            if tool is None or tool.official_name != call.official_name:
                raise ValueError("native response call differs from the request tool mapping")
            if call.call_id in response_call_ids or call.call_id in historical_call_ids:
                raise ValueError("native response repeats a tool-call identifier")
            response_call_ids.add(call.call_id)
        observation = response.provider_identity_observation
        if observation is not None and (
            observation.requested_model != self.request.requested_model
            or observation.backend_fingerprint != self.request.backend_fingerprint
        ):
            raise ValueError("provider identity observation differs from the request")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2DecisionBarrierEvidence(ImmutableModel):
    """Trusted replay evidence that all six decisions precede evaluation."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_EVALUATION_BARRIER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_EVALUATION_BARRIER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    semantic_release_fingerprint: Sha256
    decision_node_references: Annotated[tuple[Sha256, ...], Field(min_length=6, max_length=6)]
    decision_event_fingerprints: Annotated[
        tuple[Sha256, ...],
        Field(min_length=6, max_length=6),
    ]
    final_decision_event_fingerprint: Sha256
    all_seed_candidates_and_nominations_frozen: Literal[True] = True
    all_gate_calls_terminal: Literal[True] = True
    all_six_decisions_frozen: Literal[True] = True
    independently_replayed: Literal[True] = True
    evaluation_started: Literal[False] = False
    candidate_visible: Literal[False] = False

    @model_validator(mode="after")
    def _close_global_barrier(self) -> Self:
        if len(set(self.decision_node_references)) != 6:
            raise ValueError("evaluation barrier repeats a decision node")
        if len(set(self.decision_event_fingerprints)) != 6:
            raise ValueError("evaluation barrier repeats a decision event")
        if self.final_decision_event_fingerprint != self.decision_event_fingerprints[-1]:
            raise ValueError("evaluation barrier tail differs from the final decision event")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2EvaluationUnlock(ImmutableModel):
    """HMAC-authenticated trusted-plane authority for HOLDOUT evaluation."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_PROTOCOL
    )
    barrier_evidence: BfclV4PublicV2DecisionBarrierEvidence
    barrier_evidence_fingerprint: Sha256
    authority_key_id: Sha256
    authentication_tag_hmac_sha256: Sha256
    candidate_visible: Literal[False] = False
    authority_secret_present: Literal[False] = False
    answers_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False

    @model_validator(mode="after")
    def _bind_barrier(self) -> Self:
        if self.barrier_evidence_fingerprint != self.barrier_evidence.fingerprint:
            raise ValueError("evaluation unlock points to another decision barrier")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


class BfclV4PublicV2TrustedGraderReceipt(ImmutableModel):
    """Minimal trusted receipt with no task ID, answer identity, or diagnostic."""

    schema_version: Literal["1"] = "1"
    protocol: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    node_reference_sha256: Sha256
    campaign_call_slot: Annotated[int, Field(ge=0, lt=1_086, strict=True)]
    task_ref: Annotated[str, Field(pattern=r"^(fit|gate|holdout)-[0-9]{2}$")]
    task_reference_sha256: Sha256
    split_role: BfclV4PublicDevelopmentV2Split
    request_fingerprint: Sha256
    response_fingerprint: Sha256
    evaluation_unlock_fingerprint: Sha256 | None = None
    loaded_question_bundle_fingerprint: Sha256
    trusted_authority_fingerprint: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT
    )
    trusted_grader_implementation_fingerprint: Literal[
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    ] = BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
    grader_source_sha256: Sha256
    worker_source_sha256: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256
    )
    worker_primitives_source_sha256: Literal[
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256
    ] = BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256
    checker_source_bundle_sha256: Literal[BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256] = (
        BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256
    )
    correct: bool
    isolated_worker_executed: Literal[True] = True
    exact_upstream_ast_checker_executed: Literal[True] = True
    possible_answers_read_in_isolated_trusted_worker: Literal[True] = True
    trusted_control_plane_only: Literal[True] = True
    candidate_visible: Literal[False] = False
    task_id_present: Literal[False] = False
    answers_present: Literal[False] = False
    answer_derived_identities_present: Literal[False] = False
    checker_diagnostics_present: Literal[False] = False
    public_development_only: Literal[True] = True
    official_bfcl_evidence: Literal[False] = False
    hidden_test_evidence: Literal[False] = False
    score_bearing_execution_allowed: Literal[False] = False
    reportable_result: Literal[False] = False

    @model_validator(mode="after")
    def _close_minimal_projection(self) -> Self:
        if not self.task_ref.startswith(f"{self.split_role.value}-"):
            raise ValueError("trusted receipt task reference differs from its split")
        has_unlock = self.evaluation_unlock_fingerprint is not None
        if has_unlock is not (self.split_role is BfclV4PublicDevelopmentV2Split.HOLDOUT):
            raise ValueError("only HOLDOUT receipts bind an evaluation unlock")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


__all__ = [name for name in globals() if name.startswith("BFCL_") or name.startswith("Bfcl")]
