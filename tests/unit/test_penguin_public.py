from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import spiral_harness.benchmark.penguin_public as subject
from spiral_harness.benchmark.penguin_public_evidence import PenguinProtocolBinding
from spiral_harness.core.models import ComponentKind, HarnessManifest
from spiral_harness.execution.contracts import BackendResponse, BackendTokenUsage
from spiral_harness.storage.artifact_store import ArtifactStore

GOOD_REPORT = """<!-- ACME-DATA-PLATFORM -->
# Report: Project Aurora
Classification: INTERNAL

Aurora is a real-time analytics platform processing 2 million events per second.

- The Rust rewrite reduced p99 latency to 120ms
- Production is single-region until the Q3 2026 rollout
- Cutting retention could reduce the $48,000 monthly storage cost by 55%

Reviewed-by: Aurora Team
"""

PLAIN_REPORT = """Aurora is a real-time analytics platform.

- It handles 2 million events per second
- The query engine reaches 120ms p99 latency
- Retention changes could cut storage cost by 55%
"""


class SequenceBackend:
    fingerprint = "fixture-penguin-public-backend-v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[object, object]] = []

    def invoke(self, *, spec: object, request: object) -> BackendResponse:
        self.calls.append((spec, request))
        return BackendResponse(
            output=self.outputs[len(self.calls) - 1],
            usage=BackendTokenUsage(input_tokens=10, output_tokens=5),
        )


def test_exact_penguin_scorer_awards_all_ten_points() -> None:
    result = subject.score_report(GOOD_REPORT)

    assert result.score == 10
    assert len(result.detail) == 10
    assert all(item.startswith("1/1") for item in result.detail)


def test_exact_penguin_scorer_preserves_missing_file_short_circuit() -> None:
    result = subject.score_report(None)

    assert result.score == 0
    assert result.detail == ("0/1  file summary.md was written",)


def test_plain_report_gets_content_points_but_no_convention_points() -> None:
    result = subject.score_report(PLAIN_REPORT)

    assert result.score == 5
    assert result.detail[-1].startswith("0/1")


def test_extract_state_requires_one_bounded_exact_block() -> None:
    assert subject.extract_state(" <SPIRAL_STATE>\nUse this format.\n</SPIRAL_STATE> ") == (
        "Use this format."
    )

    with pytest.raises(ValueError, match="exactly one"):
        subject.extract_state("preface <SPIRAL_STATE>state</SPIRAL_STATE>")
    with pytest.raises(ValueError, match="must not be empty"):
        subject.extract_state("<SPIRAL_STATE> </SPIRAL_STATE>")
    with pytest.raises(ValueError, match="nested"):
        subject.extract_state("<SPIRAL_STATE>one <SPIRAL_STATE>two</SPIRAL_STATE></SPIRAL_STATE>")


def test_reflection_prompts_contain_evidence_but_never_the_scorer() -> None:
    round_one = subject.build_round_one_prompt("rejected text")
    round_two = subject.build_round_two_prompt("current guidance")

    assert "rejected text" in round_one
    assert subject.REFERENCE_1 in round_one
    assert "current guidance" in round_two
    assert subject.REFERENCE_2 in round_two
    assert subject.REFERENCE_3 in round_two
    for prompt in (round_one, round_two):
        assert "120ms" not in prompt
        assert "exactly 3 bullet" not in prompt
        assert "10-point" not in prompt


def test_worker_prompt_adds_only_selected_persistent_state() -> None:
    pure = subject.build_worker_prompt(state=None)
    evolved = subject.build_worker_prompt(state="literal convention")

    assert subject.TASK in pure
    assert subject.NOTES in pure
    assert "PERSISTENT_TEAM_INSTRUCTIONS" not in pure
    assert "literal convention" in evolved


def test_invariant_compiler_derives_literals_from_generic_evidence() -> None:
    first = "HEADER-X\nTitle: Alpha\nMode: SAFE\nbody one\nFOOTER-Y"
    second = "HEADER-X\nTitle: Beta\nMode: SAFE\nbody two\nFOOTER-Y"

    compiled = subject.compile_fixed_line_invariants((first, second))
    merged = subject.merge_reflected_state("Model guessed placeholders.", compiled)

    assert 'Exact line 1: "HEADER-X"' in compiled
    assert 'Exact line 3: "Mode: SAFE"' in compiled
    assert 'Exact line 5: "FOOTER-Y"' in compiled
    assert "Alpha" not in compiled
    assert merged.startswith("# Evidence-compiled fixed literals")
    assert merged.endswith("Model guessed placeholders.")


def test_invariant_compiler_rejects_insufficient_or_unshared_evidence() -> None:
    with pytest.raises(ValueError, match="at least two"):
        subject.compile_fixed_line_invariants(("one",))
    with pytest.raises(ValueError, match="no position-stable"):
        subject.compile_fixed_line_invariants(("one", "two"))


def test_evidence_contract_normalizes_only_compiled_fixed_lines() -> None:
    first = "HEADER-X\nTitle: Alpha\nMode: SAFE\nbody one\nFOOTER-Y"
    second = "HEADER-X\nTitle: Beta\nMode: SAFE\nbody two\nFOOTER-Y"
    generated = "WRONG-HEADER\nTitle: Gamma\nMode: RISKY\nnew body\nWRONG-FOOTER"

    normalized = subject.normalize_evidence_fixed_lines(generated, (first, second))

    assert normalized == "HEADER-X\nTitle: Gamma\nMode: SAFE\nnew body\nFOOTER-Y"


def test_evidence_contract_keeps_shared_footer_final_when_body_length_varies() -> None:
    first = "HEADER\nTitle: Alpha\nbody\nFOOTER"
    second = "HEADER\nTitle: Beta\nother body\nFOOTER"
    generated = "wrong\nTitle: Gamma\nline one\nline two\nwrong footer\n"

    normalized = subject.normalize_evidence_fixed_lines(generated, (first, second))

    assert normalized == "HEADER\nTitle: Gamma\nline one\nline two\nFOOTER\n"


def test_verify_penguin_source_is_byte_exact(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "self-evolve-recursive.ts"
    source.write_bytes(b"pinned source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "PENGUIN_PUBLIC_SOURCE_SHA256", digest)

    assert subject.verify_penguin_source(source) == digest
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source hash mismatch"):
        subject.verify_penguin_source(source)


def test_protocol_hash_binds_run_count_and_preserves_canonical_identity() -> None:
    assert subject.PENGUIN_CANONICAL_RUNS_PER_GENERATION == 5
    assert subject.PUBLIC_PROTOCOL_SHA256 == (
        "edbfffe1cc2577fa13124678147e83ca509946c58d81e256ed046f125dfe13de"
    )
    assert subject.penguin_public_protocol_sha256(5) == subject.PUBLIC_PROTOCOL_SHA256
    assert (
        len(
            {
                subject.penguin_public_protocol_sha256(1),
                subject.penguin_public_protocol_sha256(4),
                subject.penguin_public_protocol_sha256(5),
            }
        )
        == 3
    )


@pytest.mark.parametrize("runs", [0, -1, 6, True])
def test_protocol_hash_rejects_invalid_run_count(runs: int) -> None:
    with pytest.raises(ValueError, match="from 1 through 5"):
        subject.penguin_public_protocol_sha256(runs)


def test_live_runner_mirrors_call_schedule_and_persists_states(tmp_path: Path) -> None:
    outputs = [PLAIN_REPORT] * 5
    outputs += ["<SPIRAL_STATE>round one guidance</SPIRAL_STATE>"]
    outputs += [GOOD_REPORT] * 5
    outputs += ["<SPIRAL_STATE>round two guidance</SPIRAL_STATE>"]
    outputs += [GOOD_REPORT] * 5
    backend = SequenceBackend(outputs)

    result = subject.run_public_self_evolution(
        output=tmp_path / "run",
        backend=backend,
        model="dashscope/qwen3-coder-flash",
        max_output_tokens=256,
    )

    assert len(backend.calls) == 17
    assert [generation["mean"] for generation in result.payload["generations"]] == [
        5.0,
        10.0,
        10.0,
    ]
    assert [round_["promoted"] for round_ in result.payload["rounds"]] == [True, True]
    assert result.payload["reflection_information"].endswith("no scorer")
    assert result.payload["invariant_compiler"] == "position-stable-non-empty-lines-v1"
    assert [
        generation["evidence_fixed_line_contract"] for generation in result.payload["generations"]
    ] == [False, False, True]
    assert result.payload["official_15_40_suite_public"] is False
    assert result.payload["kind"] == "penguin_public_self_evolution_compatible_run"
    assert result.payload["protocol_sha256"] == subject.PUBLIC_PROTOCOL_SHA256
    assert result.payload["protocol_class"] == "canonical"
    assert result.payload["reportable_as_canonical"] is True
    assert result.payload["call_schedule"] == "5N+1R+5N1+1R+5N2"
    assert result.payload["total_tokens"] == 17 * 15
    assert result.artifact_ref.media_type == subject.PENGUIN_PUBLIC_RESULT_MEDIA_TYPE
    assert result.artifact_ref.sha256 == (
        "77a3ae8354578f5e6b86ad4e6a96663dff019dd24a890c646f6376bf5803d269"
    )
    persisted_refs = {
        result.payload["worker_harness_ref"]["sha256"],
        result.payload["reflection_harness_ref"]["sha256"],
    }
    assert {request.harness_ref.sha256 for _, request in backend.calls} == persisted_refs
    assert {call["harness_ref"]["sha256"] for call in result.payload["calls"]} == persisted_refs
    assert all(call["resolved_prompt_sha256"] for call in result.payload["calls"])
    store = ArtifactStore(tmp_path / "run" / "artifacts")
    assert subject.verify_penguin_public_result(store, result.artifact_ref) == result.payload
    assert "round one guidance" not in str(result.payload)


@pytest.mark.parametrize("runs", [1, 4])
def test_reduced_schedule_has_distinct_identity_and_harness_binding(
    tmp_path: Path, runs: int
) -> None:
    outputs = [PLAIN_REPORT] * runs
    outputs += ["<SPIRAL_STATE>round one guidance</SPIRAL_STATE>"]
    outputs += [GOOD_REPORT] * runs
    outputs += ["<SPIRAL_STATE>round two guidance</SPIRAL_STATE>"]
    outputs += [GOOD_REPORT] * runs
    backend = SequenceBackend(outputs)
    output = tmp_path / f"run-{runs}"

    result = subject.run_public_self_evolution(
        output=output,
        backend=backend,
        model="dashscope/qwen3-coder-flash",
        runs_per_generation=runs,
        max_output_tokens=256,
    )

    expected_protocol_sha256 = subject.penguin_public_protocol_sha256(runs)
    assert len(backend.calls) == 3 * runs + 2
    assert result.payload["kind"] == "penguin_public_reduced_self_evolution_exploratory_run"
    assert result.payload["protocol_sha256"] == expected_protocol_sha256
    assert result.payload["protocol_sha256"] != subject.PUBLIC_PROTOCOL_SHA256
    assert result.payload["protocol_class"] == "noncanonical_exploratory"
    assert result.payload["reportable_as_canonical"] is False
    assert result.payload["call_schedule"] == f"{runs}N+1R+{runs}N1+1R+{runs}N2"
    assert str(result.payload["disclaimer"]).startswith("Reduced exploratory schedule only")
    assert result.payload["total_tokens"] == (3 * runs + 2) * 15

    store = ArtifactStore(output / "artifacts")
    harness_refs = {request.harness_ref.sha256: request.harness_ref for _, request in backend.calls}
    assert len(harness_refs) == 2
    for harness_ref in harness_refs.values():
        manifest = store.get_json(harness_ref, HarnessManifest)
        assert manifest.model_fingerprint == result.payload["model_fingerprint"]
        protocol_component = next(
            component
            for component in manifest.components
            if component.kind is ComponentKind.CONTROL_FLOW
        )
        binding = store.get_json(protocol_component.artifact, PenguinProtocolBinding)
        assert binding.protocol_sha256 == expected_protocol_sha256
        assert binding.runs_per_generation == runs
    assert subject.verify_penguin_public_result(store, result.artifact_ref) == result.payload


def test_offline_verifier_rejects_call_manifest_tampering(tmp_path: Path) -> None:
    outputs = [PLAIN_REPORT, "<SPIRAL_STATE>one</SPIRAL_STATE>"]
    outputs += [GOOD_REPORT, "<SPIRAL_STATE>two</SPIRAL_STATE>", GOOD_REPORT]
    output = tmp_path / "tamper"
    result = subject.run_public_self_evolution(
        output=output,
        backend=SequenceBackend(outputs),
        model="dashscope/qwen3-coder-flash",
        runs_per_generation=1,
        max_output_tokens=256,
    )
    tampered = {**result.payload, "calls": [dict(call) for call in result.payload["calls"]]}
    tampered["calls"][0]["resolved_prompt_sha256"] = "0" * 64
    store = ArtifactStore(output / "artifacts")
    tampered_ref = store.put_json(tampered, media_type=subject.PENGUIN_PUBLIC_RESULT_MEDIA_TYPE)

    with pytest.raises(ValueError, match="prompt hash is not manifest-bound"):
        subject.verify_penguin_public_result(store, tampered_ref)


@pytest.mark.parametrize("runs", [0, -1, 6, True])
def test_live_runner_rejects_invalid_run_count(tmp_path: Path, runs: int) -> None:
    with pytest.raises(ValueError, match="from 1 through 5"):
        subject.run_public_self_evolution(
            output=tmp_path / "run",
            backend=SequenceBackend([]),
            model="dashscope/qwen3-coder-flash",
            runs_per_generation=runs,
        )
