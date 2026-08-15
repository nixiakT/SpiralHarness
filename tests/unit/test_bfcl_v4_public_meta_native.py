from __future__ import annotations

from dataclasses import dataclass

import pytest

from spiral_harness.core.canonical import canonical_sha256, sha256_bytes
from spiral_harness.execution.contracts import FrozenModelSpec, InferenceConfig
from spiral_harness.experiments.bfcl_v4_public_evolution_contracts import (
    BfclV4AdaptiveArm,
    BfclV4DiagnosisPrompt,
    BfclV4ProposalPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_CANDIDATE_SUBMIT_TOOL,
    BFCL_V4_DIAGNOSER_SYSTEM_PROMPT,
    BFCL_V4_DIAGNOSIS_SUBMIT_TOOL,
    BFCL_V4_PROPOSER_SYSTEM_PROMPT,
)
from spiral_harness.experiments.bfcl_v4_public_meta_native import (
    BfclV4PublicMetaNativeMaterializationError,
    materialize_bfcl_v4_public_meta_native_request,
)
from spiral_harness.providers.openai_native_contracts import FrozenNativeFunctionTool

BACKEND = "a" * 64
SERIALIZER = "b" * 64
PARSER = "c" * 64
TRANSPORT = "d" * 64
PARENT = "Pinned parent solver system prompt.\n"


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


def _diagnosis_prompt() -> BfclV4DiagnosisPrompt:
    return BfclV4DiagnosisPrompt(
        arm=BfclV4AdaptiveArm.SCORE,
        feedback_view="score-only",
        system_prompt=BFCL_V4_DIAGNOSER_SYSTEM_PROMPT,
        user_prompt=(
            '<AUTHORIZED_PUBLIC_FIT_INPUT_JSON>\n{"aggregate":4000}\n'
            "</AUTHORIZED_PUBLIC_FIT_INPUT_JSON>\n\n"
            "Call `submit_bfcl_diagnosis` exactly once."
        ),
        parent_system_prompt=PARENT,
        parent_system_prompt_sha256=sha256_bytes(PARENT.encode("utf-8")),
        authorized_input_sha256="e" * 64,
        submit_tool=BFCL_V4_DIAGNOSIS_SUBMIT_TOOL,
        submit_tool_fingerprint=canonical_sha256(BFCL_V4_DIAGNOSIS_SUBMIT_TOOL),
    )


def _proposal_prompt() -> BfclV4ProposalPrompt:
    return BfclV4ProposalPrompt(
        arm=BfclV4AdaptiveArm.FULL,
        feedback_view="candidate-safe-full",
        system_prompt=BFCL_V4_PROPOSER_SYSTEM_PROMPT,
        user_prompt=(
            '<PARENT_AND_DIAGNOSIS_JSON>\n{"diagnosis":"general"}\n'
            "</PARENT_AND_DIAGNOSIS_JSON>\n\n"
            "Call `submit_bfcl_candidate` exactly once."
        ),
        parent_system_prompt=PARENT,
        parent_system_prompt_sha256=sha256_bytes(PARENT.encode("utf-8")),
        diagnosis_result_fingerprint="f" * 64,
        diagnosis_valid=True,
        submit_tool=BFCL_V4_CANDIDATE_SUBMIT_TOOL,
        submit_tool_fingerprint=canonical_sha256(BFCL_V4_CANDIDATE_SUBMIT_TOOL),
    )


@pytest.mark.parametrize("prompt", [_diagnosis_prompt(), _proposal_prompt()])
def test_meta_request_is_exactly_two_messages_and_one_submit_tool(prompt) -> None:
    request = materialize_bfcl_v4_public_meta_native_request(
        prompt=prompt,
        spec=_spec(),
        backend=_Backend(),
        seed=91,
    )

    assert tuple(message.role for message in request.messages) == ("system", "user")
    assert tuple(message.content for message in request.messages) == (
        prompt.system_prompt,
        prompt.user_prompt,
    )
    assert all(message.name is None for message in request.messages)
    assert all(message.tool_call_id is None for message in request.messages)
    assert all(not message.tool_calls for message in request.messages)
    assert request.task_required_tools == (prompt.submit_tool,)
    assert request.harness_added_tools == ()
    assert request.tools_fingerprint == canonical_sha256(
        {
            "schema": "spiral-harness/native-function-tools/v2",
            "task_required_tools": (prompt.submit_tool,),
            "harness_added_tools": (),
        }
    )


@pytest.mark.parametrize("prompt", [_diagnosis_prompt(), _proposal_prompt()])
def test_meta_request_binds_all_backend_model_inference_and_seed_fields(prompt) -> None:
    spec = _spec()
    request = materialize_bfcl_v4_public_meta_native_request(
        prompt=prompt,
        spec=spec,
        backend=_Backend(),
        seed=2**63 - 1,
    )

    assert request.backend_fingerprint == BACKEND
    assert request.serializer_fingerprint == SERIALIZER
    assert request.parser_fingerprint == PARSER
    assert request.transport_fingerprint == TRANSPORT
    assert request.requested_model == spec.model
    assert request.inference == spec.inference
    assert request.seed == 2**63 - 1


def test_materializer_reads_identities_but_never_invokes_the_backend() -> None:
    backend = _InvocationTrapBackend()

    request = materialize_bfcl_v4_public_meta_native_request(
        prompt=_diagnosis_prompt(),
        spec=_spec(),
        backend=backend,
        seed=17,
    )

    assert request.requested_model == _spec().model
    assert backend.invocations == 0


@pytest.mark.parametrize("seed", [-1, 2**63, True])
def test_meta_seed_must_be_an_exact_unsigned_63_bit_integer(seed: int) -> None:
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="unsigned 63-bit"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=_diagnosis_prompt(),
            spec=_spec(),
            backend=_Backend(),
            seed=seed,
        )


def test_backend_must_match_the_spec_and_expose_four_sha256_identities() -> None:
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="model spec"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=_diagnosis_prompt(),
            spec=_spec(backend_fingerprint="0" * 64),
            backend=_Backend(),
            seed=17,
        )
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="lowercase SHA-256"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=_diagnosis_prompt(),
            spec=_spec(),
            backend=_Backend(parser_fingerprint="not-a-fingerprint"),
            seed=17,
        )
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="four frozen"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=_diagnosis_prompt(),
            spec=_spec(),
            backend=object(),  # type: ignore[arg-type]
            seed=17,
        )


def _solver_tool() -> FrozenNativeFunctionTool:
    return FrozenNativeFunctionTool.from_schema(
        {
            "name": "benchmark_solver",
            "description": "A benchmark task tool that must not enter a meta request.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        }
    )


@pytest.mark.parametrize("prompt", [_diagnosis_prompt(), _proposal_prompt()])
def test_solver_or_benchmark_tool_is_rejected_even_with_a_matching_forged_fingerprint(
    prompt,
) -> None:
    solver = _solver_tool()
    forged = type(prompt).model_construct(
        **{
            **prompt.model_dump(mode="python"),
            "submit_tool": solver,
            "submit_tool_fingerprint": canonical_sha256(solver),
        }
    )

    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="controller submit tool"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=forged,
            spec=_spec(),
            backend=_Backend(),
            seed=17,
        )


def test_cross_controller_tool_and_stale_tool_fingerprint_are_rejected() -> None:
    prompt = _diagnosis_prompt()
    cross_controller = BfclV4DiagnosisPrompt.model_construct(
        **{
            **prompt.model_dump(mode="python"),
            "submit_tool": BFCL_V4_CANDIDATE_SUBMIT_TOOL,
            "submit_tool_fingerprint": canonical_sha256(BFCL_V4_CANDIDATE_SUBMIT_TOOL),
        }
    )
    stale_fingerprint = BfclV4DiagnosisPrompt.model_construct(
        **{
            **prompt.model_dump(mode="python"),
            "submit_tool_fingerprint": "0" * 64,
        }
    )

    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="controller submit tool"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=cross_controller,
            spec=_spec(),
            backend=_Backend(),
            seed=17,
        )
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="strict frozen"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=stale_fingerprint,
            spec=_spec(),
            backend=_Backend(),
            seed=17,
        )


def test_controller_system_prompt_is_pinned_after_strict_prompt_revalidation() -> None:
    prompt = _proposal_prompt()
    forged = BfclV4ProposalPrompt.model_construct(
        **{
            **prompt.model_dump(mode="python"),
            "system_prompt": "Use the benchmark solver tool instead.",
        }
    )

    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="controller system"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=forged,
            spec=_spec(),
            backend=_Backend(),
            seed=17,
        )


def test_non_prompt_values_fail_before_request_materialization() -> None:
    with pytest.raises(BfclV4PublicMetaNativeMaterializationError, match="exact BFCL"):
        materialize_bfcl_v4_public_meta_native_request(
            prompt=object(),  # type: ignore[arg-type]
            spec=_spec(),
            backend=_Backend(),
            seed=17,
        )
