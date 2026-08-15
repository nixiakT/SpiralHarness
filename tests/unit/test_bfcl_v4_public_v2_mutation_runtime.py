from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BFCL_V4_PUBLIC_V2_MUTATION_CATALOGUE as MUTATION_CATALOGUE,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutableComponent as MutableComponent,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationId as MutationId,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationMaterialization as MutationMaterialization,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2MutationProposal as MutationProposal,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    BfclV4PublicV2Operator as Operator,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_mutations import (
    materialize_bfcl_v4_public_v2_mutation as materialize_mutation,
)
from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256, sha256_bytes
from spiral_harness.core.models import ArtifactRef
from spiral_harness.experiments.bfcl_v4_public_evolution_prompts import (
    BFCL_V4_SEED_SYSTEM_PROMPT,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BFCL_V4_PUBLIC_V2_NEGATIVE_CONTROL_PROMPT as NEGATIVE_CONTROL_PROMPT,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BFCL_V4_PUBLIC_V2_OPERATOR_INSTRUCTIONS as OPERATOR_INSTRUCTIONS,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BFCL_V4_PUBLIC_V2_PROMPT_OVERLAY_SEPARATOR as PROMPT_OVERLAY_SEPARATOR,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BFCL_V4_PUBLIC_V2_STATIC_PARENT_PROMPT_SHA256 as STATIC_PARENT_PROMPT_SHA256,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationRuntimeBatch as MutationRuntimeBatch,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationRuntimeExecutionError as MutationRuntimeExecutionError,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2MutationTreatmentRole as MutationTreatmentRole,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    BfclV4PublicV2SolverTreatmentPrompt as SolverTreatmentPrompt,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    assert_bfcl_v4_public_v2_mutation_runtime_score_execution_allowed as assert_score_allowed,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    materialize_bfcl_v4_public_v2_mutation_runtime_batch as materialize_runtime_batch,
)
from spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime import (
    verify_bfcl_v4_public_v2_mutation_runtime_batch as verify_runtime_batch,
)

EXPECTED_IDS = tuple(MutationId)
EXPECTED_COMPONENTS = (
    MutableComponent.NUMERIC_RENDERER,
    MutableComponent.CLAUSE_PLANNER,
    MutableComponent.ARGUMENT_VALIDATOR,
    MutableComponent.CALL_EMITTER,
    MutableComponent.TOOL_MATCHER,
)
EXPECTED_OPERATORS = tuple(Operator)
EXPECTED_INSTRUCTION_HASHES = (
    "53a64f6fc6f2126a304dbb61a542f1533031e96a004c2154242d4757a87fa845",
    "b9fbfff530204763d9b9a139003f63da9c037fe32bfd868b269674c82070d7b4",
    "05d9d3f972d2fb786a2b2b74526f20c6d74d2a24ec7221cb8f2afd9729f0cfe8",
    "00d4324ec2fa87e575cc25d6029c2b60eb7466f49badeab4fcd43ba4a83c5420",
    "c96114d293e5729b089955cc0157cae8e11253a7cb00f2ed994a0178165b3f2b",
)
EXPECTED_CANDIDATE_PROMPT_HASHES = (
    "b5da8df8c716b6f0a12873325142b11995ecbed1d0d87029886da683a5f54fd8",
    "7703c12fc5a37b66a9556f7b2d30f372879bec9585bcbc8c65363d30ccdeb15c",
    "fddd999bcd768ce2abf79bf641279eeb8f9f180de99488307e4b0ac5ce6e0e35",
    "5ddda7d3d1617144da2f52ffbc1f1b9357808c437989d87f39af4b5547ea58b4",
    "3850d482643e615b3a58f73cf86629abc3d37a1b57df9f42c41346c6d6bc82ba",
)
EXPECTED_BATCH_FINGERPRINTS = (
    "6312e8e96ae45cf35c5acbe31d580937e2389025e32c654a1269019a44efd195",
    "28408cd4bb69cd95be493f071f80b2a54cc6f1dfdb9db44c8ab44829c3361fd5",
    "d4b6629e7b33efa1a7970fb64a22d55b49b7dcbb272e28d15aac9c9d6578c42a",
    "8cdbcb88422a4cb67f6942e5c10639ca73424052e5bf1e9e5fdd623778679031",
    "e7592df712b8c9ccdabd3054d432da2eedadc83c29427dd2cc1cca664ed803a1",
)


def _materialization(
    mutation_id: MutationId,
) -> MutationMaterialization:
    return materialize_mutation(MutationProposal(catalogue_id=mutation_id))


def _batch(
    mutation_id: MutationId,
) -> MutationRuntimeBatch:
    return materialize_runtime_batch(_materialization(mutation_id))


def _different_ref(ref: ArtifactRef) -> ArtifactRef:
    digest = ("0" if ref.sha256[0] != "0" else "1") + ref.sha256[1:]
    return ref.model_copy(update={"sha256": digest})


@pytest.mark.parametrize(
    ("mutation_id", "component", "operator", "instruction_hash", "prompt_hash", "batch_hash"),
    tuple(
        zip(
            EXPECTED_IDS,
            EXPECTED_COMPONENTS,
            EXPECTED_OPERATORS,
            EXPECTED_INSTRUCTION_HASHES,
            EXPECTED_CANDIDATE_PROMPT_HASHES,
            EXPECTED_BATCH_FINGERPRINTS,
            strict=True,
        )
    ),
)
def test_all_five_runtime_mappings_are_exact_and_canonical(
    mutation_id: MutationId,
    component: MutableComponent,
    operator: Operator,
    instruction_hash: str,
    prompt_hash: str,
    batch_hash: str,
) -> None:
    batch = _batch(mutation_id)
    parent, candidate, revert, negative = batch.prompts
    changed = tuple(item for item in batch.stage_instruction_overlays if item.instruction_changed)

    assert batch.catalogue_ref == MUTATION_CATALOGUE.ref
    assert batch.catalogue_ref.sha256 == (
        "a9ea9ce1533703994a8d01b78439fd41d4242cca223116c1082e7bb974fac5d5"
    )
    assert (batch.selected_component, batch.selected_operator) == (component, operator)
    assert batch.parent_harness_ref == batch.materialization.parent_ref
    assert batch.candidate_harness_ref == batch.materialization.candidate_ref
    assert batch.canary_ref == batch.materialization.canary.ref
    assert len(changed) == 1
    assert changed[0].component is component
    assert changed[0].candidate_instruction == OPERATOR_INSTRUCTIONS[mutation_id]
    assert tuple(item.component for item in batch.stage_instruction_overlays) == tuple(
        MutableComponent
    )

    assert tuple(item.role for item in batch.prompts) == tuple(MutationTreatmentRole)
    assert parent.system_prompt == BFCL_V4_SEED_SYSTEM_PROMPT
    assert parent.system_prompt_ref.sha256 == STATIC_PARENT_PROMPT_SHA256
    assert candidate.logical_component is component
    assert candidate.operator is operator
    assert candidate.instruction_block_sha256 == instruction_hash
    assert candidate.system_prompt.startswith(BFCL_V4_SEED_SYSTEM_PROMPT + PROMPT_OVERLAY_SEPARATOR)
    assert candidate.system_prompt_ref.sha256 == prompt_hash
    assert candidate.selected_operator_prompt_activation_verified
    assert revert.system_prompt == parent.system_prompt
    assert revert.system_prompt_ref == parent.system_prompt_ref
    assert negative.system_prompt == NEGATIVE_CONTROL_PROMPT
    assert negative.system_prompt_ref.sha256 == (
        "5aca21d09f0befb2aafee9f66ee6751f13465157d7aad9dd6943362d3048b64e"
    )
    assert negative.system_prompt_ref not in {parent.system_prompt_ref, candidate.system_prompt_ref}
    assert not negative.selected_operator_prompt_activation_verified
    assert not batch.negative_control_behavioral_neutrality_attested

    assert batch.candidate_instruction_sha256 == instruction_hash
    assert batch.candidate_prompt_sha256 == prompt_hash
    assert batch.fingerprint == batch_hash
    assert batch.ref.sha256 == canonical_sha256(batch)
    assert batch.ref.size == len(canonical_json_bytes(batch))
    assert verify_runtime_batch(batch) == batch


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_prompt_materialization_does_not_attest_request_model_behavior_or_capability(
    mutation_id: MutationId,
) -> None:
    batch = _batch(mutation_id)

    assert batch.provider_calls == 0
    assert not batch.v2_question_answer_score_or_result_used
    assert batch.non_prompt_request_semantics_intended_identical
    assert not batch.native_request_semantics_equality_verified
    assert batch.future_native_orchestrator_must_verify_request_equality
    assert not batch.model_instruction_adherence_verified
    assert not batch.model_behavior_change_verified
    assert not batch.benchmark_capability_evidence
    assert not batch.score_bearing_execution_allowed
    assert not batch.v2_orchestrator_execution_grant_bound
    for prompt in batch.prompts:
        assert prompt.prompt_realized_treatment
        assert not prompt.deterministic_middleware_transform_claimed
        assert not prompt.native_request_inclusion_verified
        assert not prompt.model_instruction_adherence_verified
        assert not prompt.model_behavior_change_verified
        assert not prompt.benchmark_capability_evidence


def test_runtime_rejects_tampered_and_wrong_ref_materializations() -> None:
    materialization = _materialization(EXPECTED_IDS[0])
    forged = (
        materialization.model_copy(
            update={"catalogue_ref": _different_ref(materialization.catalogue_ref)}
        ),
        materialization.model_copy(update={"candidate_ref": materialization.parent_ref}),
        materialization.model_copy(update={"candidate": materialization.parent}),
        materialization.model_copy(
            update={"parent_ref": _different_ref(materialization.parent_ref)}
        ),
    )

    for value in forged:
        with pytest.raises(ValidationError):
            materialize_runtime_batch(value)


def test_runtime_verifier_rejects_prompt_tamper_noop_stage_tamper_and_wrong_refs() -> None:
    batch = _batch(EXPECTED_IDS[0])
    parent, candidate, revert, negative = batch.prompts
    no_op_candidate = candidate.model_copy(
        update={
            "system_prompt": parent.system_prompt,
            "system_prompt_ref": parent.system_prompt_ref,
        }
    )
    second_changed = batch.stage_instruction_overlays[1].model_copy(
        update={
            "candidate_instruction": "foreign instruction",
            "instruction_changed": True,
        }
    )
    forged = (
        batch.model_copy(update={"prompts": (parent, no_op_candidate, revert, negative)}),
        batch.model_copy(
            update={
                "stage_instruction_overlays": (
                    batch.stage_instruction_overlays[0],
                    second_changed,
                    *batch.stage_instruction_overlays[2:],
                )
            }
        ),
        batch.model_copy(update={"candidate_prompt_sha256": batch.static_parent_prompt_sha256}),
        batch.model_copy(update={"catalogue_ref": _different_ref(batch.catalogue_ref)}),
        batch.model_copy(update={"materialization_ref": _different_ref(batch.materialization_ref)}),
        batch.model_copy(
            update={"candidate_harness_ref": _different_ref(batch.candidate_harness_ref)}
        ),
    )

    for value in forged:
        with pytest.raises(ValidationError):
            verify_runtime_batch(value)


def test_runtime_contract_rejects_forged_evidence_claims() -> None:
    batch = _batch(EXPECTED_IDS[0])
    candidate = batch.prompts[1]

    for field_name in (
        "deterministic_middleware_transform_claimed",
        "native_request_inclusion_verified",
        "model_instruction_adherence_verified",
        "model_behavior_change_verified",
        "benchmark_capability_evidence",
        "score_bearing_execution_allowed",
    ):
        forged = candidate.model_copy(update={field_name: True})
        with pytest.raises(ValidationError):
            SolverTreatmentPrompt.model_validate(forged, strict=True)


@pytest.mark.parametrize("mutation_id", EXPECTED_IDS)
def test_runtime_score_guard_remains_blocked_without_orchestrator(
    mutation_id: MutationId,
) -> None:
    with pytest.raises(
        MutationRuntimeExecutionError,
        match="without the frozen v2 orchestrator",
    ):
        assert_score_allowed(_batch(mutation_id))


def test_runtime_materialization_is_provider_free(monkeypatch: pytest.MonkeyPatch) -> None:
    def invocation_trap(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider-free prompt runtime attempted network access")

    monkeypatch.setattr(socket, "socket", invocation_trap)
    monkeypatch.setattr(socket, "create_connection", invocation_trap)

    for mutation_id in EXPECTED_IDS:
        _batch(mutation_id)


def test_runtime_import_does_not_open_v2_data_runner_grader_or_transport_boundaries() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import sys
import spiral_harness.experiments.bfcl_v4_public_v2_mutation_runtime
print("\\n".join(sorted(name for name in sys.modules if name.startswith("spiral_harness"))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = set(completed.stdout.splitlines())
    forbidden_prefixes = (
        "spiral_harness.benchmark.bfcl_v4_public_development_v2",
        "spiral_harness.benchmark.bfcl_v4_public_v2_semantic_projection",
        "spiral_harness.benchmark._bfcl_v4_public_grader_worker",
        "spiral_harness.experiments.bfcl_v4_public_runner",
        "spiral_harness.experiments.bfcl_v4_public_campaign_executor",
        "spiral_harness.providers.openai_compatible",
        "spiral_harness.providers.openai_native_function",
    )

    assert not any(name.startswith(prefix) for name in imported for prefix in forbidden_prefixes)


def test_prompt_and_materialization_refs_bind_exact_bytes() -> None:
    batch = _batch(EXPECTED_IDS[0])
    materialization_content = canonical_json_bytes(batch.materialization)

    assert batch.materialization_ref.sha256 == canonical_sha256(batch.materialization)
    assert batch.materialization_ref.size == len(materialization_content)
    for prompt in batch.prompts:
        content = prompt.system_prompt.encode("utf-8")
        assert prompt.system_prompt_ref.sha256 == sha256_bytes(content)
        assert prompt.system_prompt_ref.size == len(content)
