"""Low-level, score-free helpers for the BFCL V4 public-pilot runner."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel

from spiral_harness.benchmark.bfcl_v4_public_grader import (
    make_bfcl_v4_public_prediction,
)
from spiral_harness.benchmark.bfcl_v4_public_grader_contracts import (
    BfclV4PublicGraderReceipt,
    BfclV4PublicPrediction,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    adapt_bfcl_v4_public_pilot_task,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BfclV4OpenAiToolAdapterResult,
    BfclV4PublicPilotTask,
    BfclV4PureAtBSample,
    BfclV4WireToolCall,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_plan_contracts import (
    BfclV4PilotArm,
    BfclV4PilotCallKind,
    BfclV4PilotCallSlot,
)
from spiral_harness.benchmark.bfcl_v4_public_run_contracts import (
    BfclV4CallCompletion,
    BfclV4CallMaterialization,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.contracts import ExecutionStatus, FrozenModelSpec
from spiral_harness.execution.native_function_contracts import (
    NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
    NativeFunctionExecution,
    NativeFunctionExecutionRecord,
)
from spiral_harness.experiments.bfcl_v4_public_native import (
    BfclV4PublicNativeBackendIdentity,
    BfclV4PublicNativePromptKind,
    materialize_bfcl_v4_public_native_request,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE,
    BfclV4RunnerHarnessArtifact,
    BfclV4RunnerHarnessKind,
)
from spiral_harness.providers.openai_native_contracts import (
    NativeFunctionCallRequest,
    NativeFunctionCallResponse,
)
from spiral_harness.storage.protocol import ArtifactRepository


class BfclV4PublicRunnerError(RuntimeError):
    """The live/replay runner could not preserve a frozen protocol invariant."""


@dataclass(frozen=True, slots=True)
class BfclV4RunnerCallRecord:
    """Process-local join retained until portable evidence is published."""

    slot: BfclV4PilotCallSlot
    materialization: BfclV4CallMaterialization
    materialization_ref: ArtifactRef
    completion: BfclV4CallCompletion
    completion_ref: ArtifactRef
    execution_record: NativeFunctionExecutionRecord
    prediction: BfclV4PublicPrediction | None
    grader_receipt: BfclV4PublicGraderReceipt | None
    pure_at_b_sample: BfclV4PureAtBSample | None

    @property
    def provider_succeeded(self) -> bool:
        return self.execution_record.execution.status is ExecutionStatus.COMPLETED

    @property
    def accepted(self) -> bool:
        return bool(
            self.grader_receipt is not None and self.grader_receipt.upstream_ast_checker_valid
        )


def publish_model[ModelT: BaseModel](
    repository: ArtifactRepository,
    value: ModelT,
    *,
    media_type: str,
) -> ArtifactRef:
    """Publish and byte-verify one canonical model artifact."""

    payload = canonical_json_bytes(value)
    raw_ref = repository.put_json(value, media_type=media_type)
    try:
        ref = ArtifactRef.model_validate(raw_ref, strict=True)
        stored = repository.get_bytes(ref)
    except Exception as exc:
        raise BfclV4PublicRunnerError("published runner artifact cannot be verified") from exc
    if ref.media_type != media_type or ref.size == 0 or stored != payload:
        raise BfclV4PublicRunnerError("published runner artifact changed bytes or media type")
    return ref


def load_canonical_model[ModelT: BaseModel](
    repository: ArtifactRepository,
    ref: ArtifactRef,
    model_type: type[ModelT],
    *,
    media_type: str | None = None,
) -> ModelT:
    """Load JSON representation and require an exact canonical-byte round trip."""

    try:
        checked_ref = ArtifactRef.model_validate(ref, strict=True)
        if media_type is not None and checked_ref.media_type != media_type:
            raise ValueError("wrong media type")
        payload = repository.get_bytes(checked_ref)
        raw = repository.get_json(checked_ref)
        checked = model_type.model_validate(raw, strict=False)
    except Exception as exc:
        raise BfclV4PublicRunnerError("runner artifact failed canonical loading") from exc
    if canonical_json_bytes(checked) != payload:
        raise BfclV4PublicRunnerError("runner artifact is not canonical for its model")
    return checked


def publish_native_request(
    repository: ArtifactRepository,
    request: NativeFunctionCallRequest,
) -> ArtifactRef:
    return publish_model(
        repository,
        request,
        media_type=NATIVE_FUNCTION_REQUEST_MEDIA_TYPE,
    )


def make_harness_artifact(
    *,
    kind: BfclV4RunnerHarnessKind,
    system_prompt: str | None,
    arm: BfclV4PilotArm | None = None,
) -> BfclV4RunnerHarnessArtifact:
    return BfclV4RunnerHarnessArtifact(
        kind=kind,
        arm=None if arm is None else arm.value,
        system_prompt=system_prompt,
        system_prompt_sha256=(
            None if system_prompt is None else sha256_bytes(system_prompt.encode("utf-8"))
        ),
    )


def publish_harness(
    repository: ArtifactRepository,
    artifact: BfclV4RunnerHarnessArtifact,
) -> ArtifactRef:
    return publish_model(repository, artifact, media_type=BFCL_V4_RUNNER_HARNESS_MEDIA_TYPE)


def solver_prompt_kind(kind: BfclV4RunnerHarnessKind) -> BfclV4PublicNativePromptKind:
    return {
        BfclV4RunnerHarnessKind.BARE: BfclV4PublicNativePromptKind.PURE,
        BfclV4RunnerHarnessKind.STATIC: BfclV4PublicNativePromptKind.STATIC,
        BfclV4RunnerHarnessKind.PARENT: BfclV4PublicNativePromptKind.PARENT,
        BfclV4RunnerHarnessKind.CANDIDATE: BfclV4PublicNativePromptKind.CANDIDATE,
    }[kind]


def materialize_solver_request(
    *,
    task: BfclV4PublicPilotTask,
    adapter: BfclV4OpenAiToolAdapterResult,
    harness: BfclV4RunnerHarnessArtifact,
    spec: FrozenModelSpec,
    backend: BfclV4PublicNativeBackendIdentity,
    seed: int,
) -> NativeFunctionCallRequest:
    return materialize_bfcl_v4_public_native_request(
        task=task,
        adapter=adapter,
        spec=spec,
        backend=backend,
        seed=seed,
        prompt_kind=solver_prompt_kind(harness.kind),
        system_prompt=harness.system_prompt,
    )


def task_and_adapter_maps(
    tasks: tuple[BfclV4PublicPilotTask, ...],
) -> tuple[
    dict[str, BfclV4PublicPilotTask],
    dict[str, BfclV4OpenAiToolAdapterResult],
]:
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise BfclV4PublicRunnerError("loaded BFCL task IDs repeat")
    adapters = {task_id: adapt_bfcl_v4_public_pilot_task(task) for task_id, task in by_id.items()}
    return by_id, adapters


def prediction_from_response(
    task_id: str,
    response: NativeFunctionCallResponse | None,
) -> BfclV4PublicPrediction:
    """Normalize a verified native response; provider failure is explicit empty output."""

    calls: tuple[tuple[str, dict[str, object]], ...] = ()
    if response is not None:
        parsed: list[tuple[str, dict[str, object]]] = []
        for call in response.tool_calls:
            try:
                arguments = json.loads(call.arguments_json)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:  # pragma: no cover
                raise BfclV4PublicRunnerError("verified tool arguments became invalid") from exc
            if not isinstance(arguments, dict):
                raise BfclV4PublicRunnerError("verified tool arguments are not an object")
            parsed.append((call.official_name, arguments))
        calls = tuple(parsed)
    return make_bfcl_v4_public_prediction(task_id, calls)


def completed_native_response(
    execution: NativeFunctionExecution,
) -> NativeFunctionCallResponse | None:
    """Expose a retained response only when its provider attempt completed."""

    return execution.response if execution.status is ExecutionStatus.COMPLETED else None


def pure_at_b_sample_from_response(
    slot: BfclV4PilotCallSlot,
    response: NativeFunctionCallResponse | None,
) -> BfclV4PureAtBSample:
    calls = None
    if response is not None:
        calls = tuple(
            BfclV4WireToolCall(
                wire_name=call.wire_name,
                arguments_json=call.arguments_json,
            )
            for call in response.tool_calls
        )
    return BfclV4PureAtBSample(sample_id=slot.call_id, calls=calls)


def prediction_from_wire_calls(
    *,
    task_id: str,
    calls: tuple[BfclV4WireToolCall, ...] | None,
    adapter: BfclV4OpenAiToolAdapterResult,
) -> BfclV4PublicPrediction:
    normalized: list[tuple[str, dict[str, object]]] = []
    for call in calls or ():
        official = adapter.wire_to_official(call.wire_name)
        arguments = json.loads(call.arguments_json)
        if not isinstance(arguments, dict):
            raise BfclV4PublicRunnerError("PURE@B selected arguments are not an object")
        normalized.append((official, arguments))
    return make_bfcl_v4_public_prediction(task_id, tuple(normalized))


def slot_task_fingerprint(
    slot: BfclV4PilotCallSlot,
    *,
    task: BfclV4PublicPilotTask | None,
    meta_prompt: BaseModel | None,
) -> str:
    if (task is None) == (meta_prompt is None):
        raise BfclV4PublicRunnerError("slot must bind exactly one task or meta prompt")
    return canonical_sha256(task if task is not None else meta_prompt)


def is_meta_slot(slot: BfclV4PilotCallSlot) -> bool:
    return slot.kind in {BfclV4PilotCallKind.DIAGNOSIS, BfclV4PilotCallKind.PROPOSAL}


__all__ = [
    "BfclV4PublicRunnerError",
    "BfclV4RunnerCallRecord",
    "completed_native_response",
    "is_meta_slot",
    "load_canonical_model",
    "make_harness_artifact",
    "materialize_solver_request",
    "prediction_from_response",
    "prediction_from_wire_calls",
    "publish_harness",
    "publish_model",
    "publish_native_request",
    "pure_at_b_sample_from_response",
    "slot_task_fingerprint",
    "task_and_adapter_maps",
]
