"""Trusted deterministic middleware and signed branch events for v3."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from threading import RLock
from typing import Literal

from spiral_harness.benchmark._harness_fault_cases import (
    RepairRuleId,
    RuntimeBranch,
    branch_for_rule,
    evaluate_branch,
    parse_public_task_input,
)
from spiral_harness.benchmark.harness_fault_compiler import (
    verify_fault_compilation,
)
from spiral_harness.core.canonical import canonical_json_bytes, sha256_bytes
from spiral_harness.core.models import ArtifactRef, ImmutableModel, Sha256
from spiral_harness.execution.contracts import (
    BackendResponse,
    FrozenModelSpec,
    ModelRequest,
)
from spiral_harness.execution.model import ModelBackend
from spiral_harness.storage.artifact_store import ArtifactStore

RUNTIME_EVENT_MEDIA_TYPE = (
    "application/vnd.spiral-harness.harness-fault-runtime-branch-event.v3+json"
)
RUNTIME_PRODUCER_VERSION = "spiral-harness.harness-fault-middleware-runtime:v3"
_ATTESTATION_DOMAIN = b"spiral-harness:harness-fault-runtime-event:v3\x00"
_PRODUCER_DOMAIN = b"spiral-harness:harness-fault-runtime-producer:v3\x00"


class HarnessFaultRuntimeError(ValueError):
    """Runtime branch evidence or middleware execution failed closed."""


class RuntimeBranchEventContent(ImmutableModel):
    """Raw deterministic branch trace; contains no pass/fail assertion."""

    schema_version: Literal["3"] = "3"
    producer_version: Literal["spiral-harness.harness-fault-middleware-runtime:v3"] = (
        RUNTIME_PRODUCER_VERSION
    )
    compilation_ref: ArtifactRef
    request_sha256: Sha256
    task_id: str
    harness_ref: ArtifactRef
    rule_id: RepairRuleId
    branch: RuntimeBranch
    raw_left_sha256: Sha256
    raw_right_sha256: Sha256
    base_output_sha256: Sha256
    final_output_sha256: Sha256


class AttestedRuntimeBranchEvent(RuntimeBranchEventContent):
    producer_id: Sha256
    attestation_sha256: Sha256

    @property
    def content(self) -> RuntimeBranchEventContent:
        return RuntimeBranchEventContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"producer_id", "attestation_sha256"},
                round_trip=True,
                warnings="none",
            ),
            strict=True,
        )


class RuntimeEventVerificationCapability:
    """Process-local verification capability; it exposes no signing operation."""

    __slots__ = ("__secret", "_producer_id")

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("runtime verification secret must contain at least 32 bytes")
        self.__secret = secret
        self._producer_id = hashlib.sha256(_PRODUCER_DOMAIN + secret).hexdigest()

    @property
    def producer_id(self) -> str:
        return self._producer_id

    def verify(self, value: AttestedRuntimeBranchEvent | object) -> AttestedRuntimeBranchEvent:
        try:
            event = AttestedRuntimeBranchEvent.model_validate(value, strict=True)
        except Exception as exc:
            raise HarnessFaultRuntimeError("runtime event is malformed") from exc
        if event.producer_id != self._producer_id:
            raise HarnessFaultRuntimeError("runtime event comes from another producer")
        expected = hmac.new(
            self.__secret,
            _ATTESTATION_DOMAIN + canonical_json_bytes(event.content),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(event.attestation_sha256, expected):
            raise HarnessFaultRuntimeError("runtime event attestation is invalid")
        return event


class TrustedRuntimeEventService:
    """Trusted producer used by the actual middleware backend invocation path."""

    __slots__ = ("__secret", "_verification")

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)
        self._verification = RuntimeEventVerificationCapability(self.__secret)

    @property
    def verification_capability(self) -> RuntimeEventVerificationCapability:
        return self._verification

    @property
    def producer_id(self) -> str:
        return self._verification.producer_id

    def issue(self, content: RuntimeBranchEventContent) -> AttestedRuntimeBranchEvent:
        checked = RuntimeBranchEventContent.model_validate(content, strict=True)
        attestation = hmac.new(
            self.__secret,
            _ATTESTATION_DOMAIN + canonical_json_bytes(checked),
            hashlib.sha256,
        ).hexdigest()
        return AttestedRuntimeBranchEvent(
            **checked.model_dump(mode="python", round_trip=True, warnings="none"),
            producer_id=self.producer_id,
            attestation_sha256=attestation,
        )


class HarnessFaultMiddlewareBackend:
    """Charge a base call, then execute a deterministic middleware branch.

    The base output is digest-bound for closure but does not drive the final
    behavior.  Consequently this runtime validates middleware attribution only;
    it is not live-model benchmark evidence.
    """

    __slots__ = (
        "_compilation",
        "_compilation_ref",
        "_events",
        "_lock",
        "_service",
        "_spec",
        "_store",
        "_underlying",
    )

    def __init__(
        self,
        *,
        store: ArtifactStore,
        spec: FrozenModelSpec,
        underlying: ModelBackend,
        compilation_ref: ArtifactRef,
        event_service: TrustedRuntimeEventService,
    ) -> None:
        if type(store) is not ArtifactStore:
            raise TypeError("store must be an exact ArtifactStore")
        if not isinstance(underlying, ModelBackend):
            raise TypeError("underlying must implement ModelBackend")
        if type(event_service) is not TrustedRuntimeEventService:
            raise TypeError("event_service must be an exact TrustedRuntimeEventService")
        self._store = store
        self._spec = FrozenModelSpec.model_validate(spec, strict=True)
        self._service = event_service
        self._compilation_ref = ArtifactRef.model_validate(compilation_ref, strict=True)
        self._compilation = verify_fault_compilation(
            store,
            self._spec,
            self._compilation_ref,
            expected_runtime_producer_id=event_service.producer_id,
        )
        if underlying.fingerprint != self._spec.backend_fingerprint:
            raise HarnessFaultRuntimeError("underlying backend differs from frozen model spec")
        self._underlying = underlying
        self._events: dict[str, ArtifactRef] = {}
        self._lock = RLock()

    @property
    def fingerprint(self) -> str:
        return self._underlying.fingerprint

    @property
    def compilation_ref(self) -> ArtifactRef:
        return self._compilation_ref

    @property
    def producer_id(self) -> str:
        return self._service.producer_id

    @property
    def repository(self) -> ArtifactStore:
        return self._store

    def event_ref_for(self, request_sha256: str) -> ArtifactRef:
        with self._lock:
            try:
                return self._events[request_sha256]
            except KeyError as exc:
                raise HarnessFaultRuntimeError(
                    "no trusted runtime event exists for this exact request"
                ) from exc

    def invoke(
        self,
        *,
        spec: FrozenModelSpec,
        request: ModelRequest,
    ) -> BackendResponse:
        checked_spec = FrozenModelSpec.model_validate(spec, strict=True)
        checked_request = ModelRequest.model_validate(request, strict=True)
        if checked_spec != self._spec:
            raise HarnessFaultRuntimeError("runtime received another frozen model spec")
        request_sha256 = checked_request.fingerprint
        with self._lock:
            if request_sha256 in self._events:
                raise HarnessFaultRuntimeError("one frozen request cannot execute twice")
        try:
            role = self._compilation.role_for_harness(checked_request.harness_ref)
        except KeyError as exc:
            raise HarnessFaultRuntimeError("request harness is outside compiler graph") from exc
        rule_id = self._compilation.entry(role).rule_id
        task_input = parse_public_task_input(checked_request.user_prompt)
        branch = branch_for_rule(rule_id, task_input.policy)

        raw_response = self._underlying.invoke(spec=checked_spec, request=checked_request)
        response = BackendResponse.model_validate(raw_response, strict=True)
        answer, observable = evaluate_branch(
            branch,
            left=task_input.left,
            right=task_input.right,
        )
        final_output = json.dumps(
            {"answer": answer, "observable": observable},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content = RuntimeBranchEventContent(
            compilation_ref=self._compilation_ref,
            request_sha256=request_sha256,
            task_id=checked_request.task_id,
            harness_ref=checked_request.harness_ref,
            rule_id=rule_id,
            branch=branch,
            raw_left_sha256=sha256_bytes(task_input.left.encode("utf-8")),
            raw_right_sha256=sha256_bytes(task_input.right.encode("utf-8")),
            base_output_sha256=sha256_bytes(response.output.encode("utf-8")),
            final_output_sha256=sha256_bytes(final_output.encode("utf-8")),
        )
        event = self._service.issue(content)
        event_ref = self._store.put_json(event, media_type=RUNTIME_EVENT_MEDIA_TYPE)
        with self._lock:
            if request_sha256 in self._events:  # pragma: no cover - serialized duplicate guard
                raise HarnessFaultRuntimeError("runtime request raced with a duplicate")
            self._events[request_sha256] = event_ref
        return BackendResponse(
            output=final_output,
            usage=response.usage,
            cost_usd=response.cost_usd,
        )


__all__ = [
    "RUNTIME_EVENT_MEDIA_TYPE",
    "RUNTIME_PRODUCER_VERSION",
    "AttestedRuntimeBranchEvent",
    "HarnessFaultMiddlewareBackend",
    "HarnessFaultRuntimeError",
    "RuntimeBranchEventContent",
    "RuntimeEventVerificationCapability",
    "TrustedRuntimeEventService",
]
