"""Verified native-call producer contracts for BFCL public-v2 grading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, model_validator

from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_request_materializer_contracts import (
    BfclV4PublicV2RequestMaterialization,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BfclV4PublicV2TrustedGradeRequest,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ImmutableModel, Sha256
from spiral_harness.execution.contracts import ExecutionStatus
from spiral_harness.execution.native_function_contracts import NativeFunctionExecution
from spiral_harness.experiments.bfcl_v4_public_v2_live_config import (
    BfclV4PublicV2LiveExecutionConfig,
)
from spiral_harness.experiments.bfcl_v4_public_v2_native import (
    materialize_bfcl_v4_public_v2_native_request,
)
from spiral_harness.experiments.bfcl_v4_public_v2_response_canonical import (
    canonicalize_bfcl_v4_public_v2_native_response,
)

BFCL_V4_PUBLIC_V2_TRUSTED_CALL_PRODUCER_PROTOCOL = (
    "spiral-bfcl-v4-public-v2-trusted-call-producer/v1"
)


class BfclV4PublicV2TrustedCallProducerError(ValueError):
    """A provider result lacks exact materialization or execution provenance."""


def _raw_model_content(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {name: _raw_model_content(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _raw_model_content(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_raw_model_content(item) for item in value)
    if isinstance(value, list):
        return [_raw_model_content(item) for item in value]
    return value


def _checked[ModelT: BaseModel](
    value: ModelT,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return model.model_validate(_raw_model_content(value), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise BfclV4PublicV2TrustedCallProducerError(
            f"{label} failed strict producer validation"
        ) from error


class _LiveBackendIdentity:
    """Credential-free native implementation identity from the frozen live config."""

    def __init__(self, config: BfclV4PublicV2LiveExecutionConfig) -> None:
        self._config = config

    @property
    def fingerprint(self) -> str:
        return self._config.backend_fingerprint

    @property
    def serializer_fingerprint(self) -> str:
        return self._config.serializer_fingerprint

    @property
    def parser_fingerprint(self) -> str:
        return self._config.parser_fingerprint

    @property
    def transport_fingerprint(self) -> str:
        return self._config.transport_fingerprint


def bfcl_v4_public_v2_native_slot_fingerprint(
    *,
    live_config: BfclV4PublicV2LiveExecutionConfig,
    materialization: BfclV4PublicV2RequestMaterialization,
) -> str:
    """Bind the accounted native slot to one frozen BFCL request wrapper."""

    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-v2-native-slot/v1",
            "campaign_plan_fingerprint": live_config.campaign_plan_fingerprint,
            "node_schedule_content_sha256": live_config.node_schedule_content_sha256,
            "semantic_release_fingerprint": live_config.semantic_release_fingerprint,
            "live_execution_config_fingerprint": live_config.fingerprint,
            "node_reference_sha256": materialization.lineage.node_reference_sha256,
            "campaign_call_slot": materialization.lineage.campaign_call_slot,
            "request_materialization_fingerprint": materialization.fingerprint,
        }
    )


def _producer_fingerprint(
    campaign: BfclV4PublicDevelopmentV2CampaignPlan,
    live_config: BfclV4PublicV2LiveExecutionConfig,
) -> str:
    return canonical_sha256(
        {
            "domain": BFCL_V4_PUBLIC_V2_TRUSTED_CALL_PRODUCER_PROTOCOL,
            "campaign_plan_fingerprint": campaign.fingerprint,
            "node_schedule_content_sha256": campaign.node_schedule_content_sha256,
            "live_execution_config_fingerprint": live_config.fingerprint,
        }
    )


class BfclV4PublicV2TrustedCallRecord(ImmutableModel):
    """Full call retained in the trusted plane until its one grading attempt."""

    schema_version: Literal["2"] = "2"
    protocol: Literal[BFCL_V4_PUBLIC_V2_TRUSTED_CALL_PRODUCER_PROTOCOL] = (
        BFCL_V4_PUBLIC_V2_TRUSTED_CALL_PRODUCER_PROTOCOL
    )
    campaign_plan_fingerprint: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
    )
    node_schedule_content_sha256: Literal[BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256] = (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
    )
    producer_fingerprint: Sha256
    live_execution_config: BfclV4PublicV2LiveExecutionConfig
    live_execution_config_fingerprint: Sha256
    request_materialization: BfclV4PublicV2RequestMaterialization
    request_materialization_fingerprint: Sha256
    native_execution: NativeFunctionExecution
    native_execution_receipt_fingerprint: Sha256
    grade_request: BfclV4PublicV2TrustedGradeRequest
    canonical_response: str
    request_payload_sha256: Sha256
    provider_response_fingerprint: Sha256
    trusted_plane_only: Literal[True] = True
    verified_producer_required: Literal[True] = True

    @model_validator(mode="after")
    def _bind_materialization_execution_and_grade(self) -> BfclV4PublicV2TrustedCallRecord:
        live = self.live_execution_config
        materialization = self.request_materialization
        execution = self.native_execution
        grade = self.grade_request
        response = execution.response
        expected_native = materialize_bfcl_v4_public_v2_native_request(
            visible_request=materialization.model_visible_request,
            expected_visible_request_sha256=materialization.request_payload_sha256,
            spec=live.model_spec,
            backend=_LiveBackendIdentity(live),
        )
        expected_slot = bfcl_v4_public_v2_native_slot_fingerprint(
            live_config=live,
            materialization=materialization,
        )
        if (
            self.live_execution_config_fingerprint != live.fingerprint
            or self.request_materialization_fingerprint != materialization.fingerprint
            or self.native_execution_receipt_fingerprint != canonical_sha256(execution)
            or live.semantic_release_ref != materialization.semantic_release_ref
            or live.semantic_release_fingerprint != materialization.semantic_release_ref.sha256
            or live.model_spec != execution.spec
            or expected_native != execution.request
            or execution.status is not ExecutionStatus.COMPLETED
            or response is None
            or execution.task.task_fingerprint != materialization.resolved_task.fingerprint
            or execution.task.slot_fingerprint != expected_slot
        ):
            raise ValueError(
                "trusted call differs from its live config, materialization, or native execution"
            )
        if (
            grade.node != materialization.node
            or grade.node_reference_sha256 != materialization.lineage.node_reference_sha256
            or grade.request_lineage != materialization.lineage
            or grade.task_payload_sha256 != materialization.resolved_task.candidate_payload_sha256
            or grade.request != execution.request
            or grade.request_fingerprint != execution.request.fingerprint
            or grade.request_payload_sha256 != execution.request.fingerprint
            or grade.raw_response != response
            or grade.response_fingerprint != response.fingerprint
            or self.request_payload_sha256 != execution.request.fingerprint
            or self.provider_response_fingerprint != response.fingerprint
            or self.canonical_response != canonicalize_bfcl_v4_public_v2_native_response(response)
        ):
            raise ValueError("trusted grade input differs from the verified native producer result")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True, slots=True)
class BfclV4PublicV2IssuedTrustedCall:
    """Process-local capability proving the configured producer issued a record."""

    record: BfclV4PublicV2TrustedCallRecord
    _producer_token: object


class BfclV4PublicV2TrustedCallProducer:
    """Sole constructor accepted by the one-shot trusted call registry."""

    def __init__(
        self,
        *,
        campaign: BfclV4PublicDevelopmentV2CampaignPlan,
        live_config: BfclV4PublicV2LiveExecutionConfig,
    ) -> None:
        checked_campaign = _checked(
            campaign,
            BfclV4PublicDevelopmentV2CampaignPlan,
            "producer campaign",
        )
        checked_live = _checked(
            live_config,
            BfclV4PublicV2LiveExecutionConfig,
            "producer live config",
        )
        if (
            checked_campaign.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
            or checked_campaign.node_schedule_content_sha256
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
            or checked_live.campaign_plan_fingerprint != checked_campaign.fingerprint
            or checked_live.node_schedule_content_sha256
            != checked_campaign.node_schedule_content_sha256
        ):
            raise BfclV4PublicV2TrustedCallProducerError(
                "producer campaign and live config differ from the frozen BFCL v2 plan"
            )
        self._campaign = checked_campaign
        self._live_config = checked_live
        self._fingerprint = _producer_fingerprint(checked_campaign, checked_live)
        self._token = object()

    @property
    def campaign(self) -> BfclV4PublicDevelopmentV2CampaignPlan:
        return self._campaign

    @property
    def live_config(self) -> BfclV4PublicV2LiveExecutionConfig:
        return self._live_config

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def produce(
        self,
        *,
        request_materialization: BfclV4PublicV2RequestMaterialization,
        native_execution: NativeFunctionExecution,
    ) -> BfclV4PublicV2IssuedTrustedCall:
        materialization = _checked(
            request_materialization,
            BfclV4PublicV2RequestMaterialization,
            "request materialization",
        )
        execution = _checked(
            native_execution,
            NativeFunctionExecution,
            "native execution receipt",
        )
        node = materialization.node
        if (
            node.node_slot >= len(self._campaign.nodes)
            or self._campaign.nodes[node.node_slot] != node
        ):
            raise BfclV4PublicV2TrustedCallProducerError(
                "request materialization selects another campaign node"
            )
        response = execution.response
        if response is None:
            raise BfclV4PublicV2TrustedCallProducerError(
                "successful trusted grading requires a native response"
            )
        grade_request = BfclV4PublicV2TrustedGradeRequest(
            node=node,
            node_reference_sha256=materialization.lineage.node_reference_sha256,
            request_lineage=materialization.lineage,
            task_payload_sha256=materialization.resolved_task.candidate_payload_sha256,
            request=execution.request,
            request_fingerprint=execution.request.fingerprint,
            request_payload_sha256=execution.request.fingerprint,
            raw_response=response,
            response_fingerprint=response.fingerprint,
        )
        try:
            record = BfclV4PublicV2TrustedCallRecord(
                producer_fingerprint=self._fingerprint,
                live_execution_config=self._live_config,
                live_execution_config_fingerprint=self._live_config.fingerprint,
                request_materialization=materialization,
                request_materialization_fingerprint=materialization.fingerprint,
                native_execution=execution,
                native_execution_receipt_fingerprint=canonical_sha256(execution),
                grade_request=grade_request,
                canonical_response=canonicalize_bfcl_v4_public_v2_native_response(response),
                request_payload_sha256=execution.request.fingerprint,
                provider_response_fingerprint=response.fingerprint,
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise BfclV4PublicV2TrustedCallProducerError(
                "native result failed verified trusted-call production"
            ) from error
        return BfclV4PublicV2IssuedTrustedCall(record=record, _producer_token=self._token)

    def verify_issued(
        self,
        issued: BfclV4PublicV2IssuedTrustedCall,
    ) -> BfclV4PublicV2TrustedCallRecord:
        if (
            type(issued) is not BfclV4PublicV2IssuedTrustedCall
            or issued._producer_token is not self._token
        ):
            raise BfclV4PublicV2TrustedCallProducerError(
                "trusted registry input was not issued by its configured producer"
            )
        record = _checked(
            issued.record,
            BfclV4PublicV2TrustedCallRecord,
            "issued trusted call",
        )
        if (
            record.producer_fingerprint != self._fingerprint
            or record.live_execution_config != self._live_config
            or record.grade_request.node
            != self._campaign.nodes[record.grade_request.node.node_slot]
        ):
            raise BfclV4PublicV2TrustedCallProducerError(
                "issued trusted call differs from its producer context"
            )
        return record


__all__ = [
    name
    for name in globals()
    if name.startswith("BFCL_")
    or name.startswith("Bfcl")
    or name.startswith("bfcl_v4_public_v2_native_slot")
]
