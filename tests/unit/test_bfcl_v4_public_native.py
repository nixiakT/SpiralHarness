from __future__ import annotations

from dataclasses import dataclass

import pytest

import spiral_harness.experiments.bfcl_v4_public_native as subject
from spiral_harness.benchmark.bfcl_v4_public_pilot import (
    adapt_bfcl_v4_public_pilot_task,
)
from spiral_harness.benchmark.bfcl_v4_public_pilot_contracts import (
    BFCL_V4_PUBLIC_PILOT_MANIFEST,
    BfclV4PublicPilotTask,
)
from spiral_harness.core.canonical import canonical_json
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.bfcl_v4_public_native import (
    BfclV4PublicNativeMaterializationError,
    BfclV4PublicNativePromptKind,
    materialize_bfcl_v4_public_native_request,
    materialize_bfcl_v4_public_native_tools,
)

BACKEND = "a" * 64
SERIALIZER = "b" * 64
PARSER = "c" * 64
TRANSPORT = "d" * 64

_SIMPLE_QUESTION = [
    [
        {
            "role": "user",
            "content": "Find the area of a triangle with a base of 10 units and height of 5 units.",
        }
    ]
]
_SIMPLE_FUNCTIONS = [
    {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "The base of the triangle."},
                "height": {"type": "integer", "description": "The height of the triangle."},
                "unit": {
                    "type": "string",
                    "description": "The unit of measure (defaults to 'units' if not specified)",
                },
            },
            "required": ["base", "height"],
        },
    }
]

_DOTTED_QUESTION = [
    [
        {
            "role": "user",
            "content": (
                "Get the protein sequence of human HbA1c, normal hemoglobin, and rat "
                "hemoglobin and their 3D models"
            ),
        }
    ]
]
_DOTTED_FUNCTIONS = [
    {
        "name": "protein_info.get_sequence_and_3D",
        "description": "Retrive the sequence and 3D models of proteins.",
        "parameters": {
            "type": "dict",
            "properties": {
                "protein_name": {
                    "type": "string",
                    "description": "The name of the protein.",
                },
                "model_3d": {
                    "type": "boolean",
                    "description": "Set true to get 3D model of the protein.",
                    "default": True,
                },
            },
            "required": ["protein_name"],
        },
    }
]


@dataclass(frozen=True)
class _Backend:
    fingerprint: str = BACKEND
    serializer_fingerprint: str = SERIALIZER
    parser_fingerprint: str = PARSER
    transport_fingerprint: str = TRANSPORT


class _InvocationTrapBackend:
    fingerprint = BACKEND
    serializer_fingerprint = SERIALIZER
    parser_fingerprint = PARSER
    transport_fingerprint = TRANSPORT

    def __init__(self) -> None:
        self.invocations = 0

    def invoke(self, **_kwargs: object) -> None:
        self.invocations += 1
        raise AssertionError("request materialization must not invoke a model")


def _task(ordinal: int = 0) -> BfclV4PublicPilotTask:
    entry = BFCL_V4_PUBLIC_PILOT_MANIFEST.roster[ordinal]
    if ordinal == 0:
        question = _SIMPLE_QUESTION
        functions = _SIMPLE_FUNCTIONS
    elif ordinal == 11:
        question = _DOTTED_QUESTION
        functions = _DOTTED_FUNCTIONS
    else:  # pragma: no cover - test helper is deliberately closed
        raise AssertionError("unsupported fixture ordinal")
    return BfclV4PublicPilotTask(
        task_id=entry.task_id,
        category=entry.category,
        split=entry.split,
        semantic_family=entry.semantic_family,
        question_git_path=entry.question_git_path,
        question_blob_size=entry.question_blob_size,
        question_blob_sha256=entry.question_blob_sha256,
        row_size=entry.row_size,
        row_sha256=entry.row_sha256,
        candidate_payload_sha256=entry.candidate_payload_sha256,
        question_json=canonical_json(question),
        function_schemas_json=canonical_json(functions),
        official_function_names=tuple(item["name"] for item in functions),
    )


def _inference() -> InferenceConfig:
    return InferenceConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=512,
        timeout_seconds=30.0,
    )


def _spec(*, backend_fingerprint: str = BACKEND) -> FrozenModelSpec:
    return FrozenModelSpec(
        backend="openai-compatible-native",
        backend_fingerprint=backend_fingerprint,
        model="dashscope/qwen36-35b-a3b",
        revision="lab-routing-snapshot-2026-08-15",
        tokenizer="provider-reported-qwen-tokenizer",
        tokenizer_revision="lab-routing-snapshot-2026-08-15",
        runtime="spiral-native-openai-compatible@v2",
        inference=_inference(),
    )


def _request(
    *,
    task: BfclV4PublicPilotTask | None = None,
    backend: object | None = None,
    prompt_kind: BfclV4PublicNativePromptKind = BfclV4PublicNativePromptKind.PURE,
    system_prompt: str | None = None,
    seed: int = 17,
):
    checked_task = task or _task()
    return materialize_bfcl_v4_public_native_request(
        task=checked_task,
        adapter=adapt_bfcl_v4_public_pilot_task(checked_task),
        spec=_spec(),
        backend=_Backend() if backend is None else backend,  # type: ignore[arg-type]
        seed=seed,
        prompt_kind=prompt_kind,
        system_prompt=system_prompt,
    )


def test_native_tools_restore_explicit_official_and_wire_names() -> None:
    task = _task(11)
    adapter = adapt_bfcl_v4_public_pilot_task(task)

    tools = materialize_bfcl_v4_public_native_tools(task, adapter)

    assert len(tools) == 1
    assert tools[0].official_name == "protein_info.get_sequence_and_3D"
    assert tools[0].wire_name == "protein_info_get_sequence_and_3D"
    assert tools[0].wire_schema == adapter.tools[0]["function"]
    assert tools[0].wire_schema["parameters"]["type"] == "object"
    assert tools[0].function_schema_json != task.function_schemas_json


def test_pure_request_flattens_the_single_bfcl_turn_without_a_system_message() -> None:
    request = _request(seed=91)

    assert tuple(message.role for message in request.messages) == ("user",)
    assert request.messages[0].content == _SIMPLE_QUESTION[0][0]["content"]
    assert request.requested_model == _spec().model
    assert request.inference == _spec().inference
    assert request.seed == 91
    assert request.backend_fingerprint == BACKEND
    assert request.serializer_fingerprint == SERIALIZER
    assert request.parser_fingerprint == PARSER
    assert request.transport_fingerprint == TRANSPORT
    assert request.harness_added_tools == ()


@pytest.mark.parametrize(
    "prompt_kind",
    [
        BfclV4PublicNativePromptKind.STATIC,
        BfclV4PublicNativePromptKind.PARENT,
        BfclV4PublicNativePromptKind.CANDIDATE,
    ],
)
def test_harness_variants_preserve_the_exact_system_prompt(
    prompt_kind: BfclV4PublicNativePromptKind,
) -> None:
    prompt = "  Preserve leading whitespace.\nPreserve the final newline.\n"

    request = _request(prompt_kind=prompt_kind, system_prompt=prompt, seed=83)

    assert tuple(message.role for message in request.messages) == ("system", "user")
    assert request.messages[0].content == prompt
    assert request.messages[1].content == _SIMPLE_QUESTION[0][0]["content"]
    assert request.seed == 83


def test_all_prompt_variants_share_the_exact_backend_spec_inference_seed_and_tools() -> None:
    prompt = "Call only tools justified by the user request."
    requests = (
        _request(seed=101),
        _request(
            prompt_kind=BfclV4PublicNativePromptKind.STATIC,
            system_prompt=prompt,
            seed=101,
        ),
        _request(
            prompt_kind=BfclV4PublicNativePromptKind.PARENT,
            system_prompt=prompt,
            seed=101,
        ),
        _request(
            prompt_kind=BfclV4PublicNativePromptKind.CANDIDATE,
            system_prompt=prompt,
            seed=101,
        ),
    )

    assert (
        len(
            {
                (
                    request.backend_fingerprint,
                    request.serializer_fingerprint,
                    request.parser_fingerprint,
                    request.transport_fingerprint,
                    request.requested_model,
                    request.inference,
                    request.seed,
                    request.task_required_tools,
                )
                for request in requests
            }
        )
        == 1
    )


def test_materializer_never_invokes_the_backend() -> None:
    backend = _InvocationTrapBackend()

    request = _request(backend=backend)

    assert request.requested_model == _spec().model
    assert backend.invocations == 0


def test_pure_and_harness_prompt_shapes_fail_closed() -> None:
    with pytest.raises(BfclV4PublicNativeMaterializationError, match="PURE request"):
        _request(
            prompt_kind=BfclV4PublicNativePromptKind.PURE,
            system_prompt="not permitted",
        )
    for prompt_kind in (
        BfclV4PublicNativePromptKind.STATIC,
        BfclV4PublicNativePromptKind.PARENT,
        BfclV4PublicNativePromptKind.CANDIDATE,
    ):
        with pytest.raises(BfclV4PublicNativeMaterializationError, match="exact system prompt"):
            _request(prompt_kind=prompt_kind)

    with pytest.raises(BfclV4PublicNativeMaterializationError, match="exact BFCL native"):
        _request(prompt_kind="pure")  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2**63, True])
def test_provider_seed_is_an_exact_unsigned_63_bit_integer(seed: int) -> None:
    with pytest.raises(BfclV4PublicNativeMaterializationError, match="unsigned 63-bit"):
        _request(seed=seed)


def test_backend_and_model_spec_must_bind_the_same_native_implementation() -> None:
    task = _task()
    adapter = adapt_bfcl_v4_public_pilot_task(task)
    with pytest.raises(BfclV4PublicNativeMaterializationError, match="model spec"):
        materialize_bfcl_v4_public_native_request(
            task=task,
            adapter=adapter,
            spec=_spec(backend_fingerprint="e" * 64),
            backend=_Backend(),
            seed=17,
            prompt_kind=BfclV4PublicNativePromptKind.PURE,
        )
    with pytest.raises(BfclV4PublicNativeMaterializationError, match="lowercase SHA-256"):
        _request(backend=_Backend(serializer_fingerprint="not-a-fingerprint"))


def test_adapter_from_another_frozen_task_is_rejected() -> None:
    task = _task()
    wrong_adapter = adapt_bfcl_v4_public_pilot_task(_task(11))

    with pytest.raises(BfclV4PublicNativeMaterializationError, match="pinned task adapter"):
        materialize_bfcl_v4_public_native_tools(task, wrong_adapter)


@pytest.mark.parametrize(
    "question_json, message",
    [
        ('[[{"content":"first","role":"user"}],[{"content":"second","role":"user"}]]', "one"),
        ('[[{"content":"x","content":"y","role":"user"}]]', "canonical strict JSON"),
        ('[[{"content":"x","extra":1,"role":"user"}]]', "unsupported shape"),
        ('[[{"content":"x","role":"assistant"}]]', "system/user role"),
    ],
)
def test_question_dialogue_parser_rejects_ambiguous_or_unsupported_shapes(
    question_json: str,
    message: str,
) -> None:
    forged = BfclV4PublicPilotTask.model_construct(
        **{
            **_task().model_dump(mode="python"),
            "question_json": question_json,
        }
    )

    with pytest.raises(BfclV4PublicNativeMaterializationError, match=message):
        subject._question_messages(forged)
