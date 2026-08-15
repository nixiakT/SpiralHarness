from __future__ import annotations

import ast
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

import spiral_harness.benchmark.bfcl_v4_public_development_v2 as subject
import spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts as contracts_subject
import spiral_harness.benchmark.bfcl_v4_question_source as question_source
from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    BfclV4PublicDevelopmentV2Error,
    BfclV4PublicDevelopmentV2ExecutionError,
    assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed,
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256,
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2AuditReceipt,
    BfclV4PublicDevelopmentV2AuditStatus,
    BfclV4PublicDevelopmentV2FailureStage,
    BfclV4PublicDevelopmentV2RosterEntry,
    BfclV4PublicDevelopmentV2Split,
    bfcl_v4_public_development_v2_v1_exclusion_sha256,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_identities import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS,
)
from spiral_harness.core.canonical import sha256_bytes

_DEFAULT_PINNED_CHECKOUT = Path("/tmp/spiral-bfcl-upstream")


@pytest.fixture(scope="module")
def pinned_checkout() -> Path:
    configured = os.environ.get("BFCL_V4_PINNED_CHECKOUT")
    checkout = Path(configured) if configured else _DEFAULT_PINNED_CHECKOUT
    if not checkout.is_dir():
        pytest.skip("pinned BFCL checkout unavailable for v2 question-only audit")
    return checkout


@pytest.fixture(scope="module")
def loaded(pinned_checkout: Path):
    return load_bfcl_v4_public_development_v2(pinned_checkout)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, (list, tuple)):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_manifest_freezes_exact_25_task_5_4_16_category_stratified_roster() -> None:
    manifest = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST

    assert manifest.fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    assert tuple(item.task_id for item in manifest.roster) == (
        BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS
    )
    assert Counter(item.split for item in manifest.roster) == {
        BfclV4PublicDevelopmentV2Split.FIT: 5,
        BfclV4PublicDevelopmentV2Split.GATE: 4,
        BfclV4PublicDevelopmentV2Split.HOLDOUT: 16,
    }
    assert Counter(item.category for item in manifest.roster) == {
        "simple_python": 7,
        "multiple": 6,
        "parallel": 6,
        "parallel_multiple": 6,
    }
    assert not set(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS) & {
        item.task_id for item in manifest.roster
    }
    assert manifest.execution_seeds_frozen is False
    assert manifest.call_plan_frozen is False
    assert manifest.independent_semantic_family_disjointness_attested is False
    assert manifest.score_bearing_execution_allowed is False
    assert manifest.hidden_test_evidence is False
    assert manifest.reportable_result is False


def test_loader_recomputes_exact_question_only_lineage_and_family_audits(loaded) -> None:
    receipt = loaded.audit_receipt

    assert receipt.status is BfclV4PublicDevelopmentV2AuditStatus.ACCEPTED
    assert receipt.source_bindings == BFCL_V4_PUBLIC_DEVELOPMENT_V2_SOURCE_BINDINGS
    assert receipt.manifest_fingerprint == BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST_FINGERPRINT
    assert receipt.candidate_pool_sha256 == BFCL_V4_PUBLIC_DEVELOPMENT_V2_CANDIDATE_POOL_SHA256
    assert (
        receipt.eligible_singletons_sha256
        == BFCL_V4_PUBLIC_DEVELOPMENT_V2_ELIGIBLE_SINGLETONS_SHA256
    )
    assert receipt.lexical_audit_sha256 == BFCL_V4_PUBLIC_DEVELOPMENT_V2_LEXICAL_AUDIT_SHA256
    assert receipt.source_row_count == 1000
    assert receipt.eligible_singleton_count == 202
    assert receipt.selected_task_count == 25
    assert receipt.lexical_comparison_count == 539
    assert receipt.lexical_near_duplicate_count == 0
    assert receipt.official_function_name_overlap_count == 0
    assert receipt.v1_task_overlap_count == 0
    assert receipt.v1_family_overlap_count == 0
    assert receipt.possible_answers_read is False
    assert receipt.scores_read is False
    assert receipt.runs_or_cas_read is False
    assert receipt.model_invoked is False
    assert receipt.network_used is False
    assert receipt.independent_semantic_family_disjointness_attested is False
    assert receipt.score_bearing_execution_allowed is False


def test_public_development_load_cannot_cross_score_bearing_boundary(loaded) -> None:
    assert loaded.trusted_control_plane_only is True
    assert loaded.candidate_visible is False
    assert loaded.independent_semantic_family_disjointness_attested is False
    assert loaded.score_bearing_execution_allowed is False

    with pytest.raises(
        BfclV4PublicDevelopmentV2ExecutionError,
        match="independent semantic family disjointness is not proven",
    ):
        assert_bfcl_v4_public_development_v2_score_bearing_execution_allowed(loaded)


def test_official_names_and_exact_families_are_disjoint_across_every_split(loaded) -> None:
    by_split = {
        split: tuple(item for item in loaded.manifest.roster if item.split is split)
        for split in BfclV4PublicDevelopmentV2Split
    }
    for left_index, left in enumerate(BfclV4PublicDevelopmentV2Split):
        left_names = {name for item in by_split[left] for name in item.official_function_names}
        left_source = {item.source_family_sha256 for item in by_split[left]}
        left_semantic = {item.semantic_template_sha256 for item in by_split[left]}
        for right in tuple(BfclV4PublicDevelopmentV2Split)[left_index + 1 :]:
            assert left_names.isdisjoint(
                name for item in by_split[right] for name in item.official_function_names
            )
            assert left_source.isdisjoint(item.source_family_sha256 for item in by_split[right])
            assert left_semantic.isdisjoint(
                item.semantic_template_sha256 for item in by_split[right]
            )


def test_candidate_visible_tasks_exclude_partition_selection_and_audit_coordinates(loaded) -> None:
    forbidden = {
        "split",
        "source_family_sha256",
        "semantic_template_sha256",
        "selection_token_sha256",
        "question_git_path",
        "row_sha256",
        "audit_receipt",
        "manifest",
    }

    for task in loaded.tasks:
        assert forbidden.isdisjoint(_all_keys(task.model_dump(mode="json")))
        assert task.possible_answer_data_in_contract is False
        assert task.answer_derived_identity_present is False
        assert task.score_derived_identity_present is False


def test_loader_reads_only_the_four_pinned_question_paths(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = question_source.read_bfcl_v4_question_blob

    def recording(checkout, git_path: str):
        observed.append(git_path)
        return original(checkout, git_path)

    monkeypatch.setattr(question_source, "read_bfcl_v4_question_blob", recording)
    loaded = load_bfcl_v4_public_development_v2(pinned_checkout)

    assert len(loaded.tasks) == 25
    assert len(observed) == 4
    assert all("possible_answer" not in path for path in observed)
    assert set(observed) == {item.question_git_path for item in loaded.manifest.roster}


def test_question_blob_tamper_emits_rejected_receipt_without_fallback(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = question_source.read_bfcl_v4_question_blob
    calls = 0

    def tampered(checkout, git_path: str):
        nonlocal calls
        calls += 1
        content, binding = original(checkout, git_path)
        if calls != 1:
            return content, binding
        changed = content + b"x"
        return changed, question_source.BfclV4QuestionSourceBinding(
            git_path=git_path,
            size=len(changed),
            sha256=sha256_bytes(changed),
        )

    monkeypatch.setattr(question_source, "read_bfcl_v4_question_blob", tampered)
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)

    receipt = captured.value.receipt
    assert receipt.status is BfclV4PublicDevelopmentV2AuditStatus.REJECTED
    assert receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.QUESTION_SOURCE
    assert receipt.manifest_fingerprint is None
    assert receipt.no_reselection_after_audit_failure is True
    assert calls == 1


def test_accepted_receipt_rejects_forged_ordered_bindings_or_manifest(loaded) -> None:
    payload = loaded.audit_receipt.model_dump(mode="python")
    payload["source_bindings"] = tuple(reversed(payload["source_bindings"]))
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2AuditReceipt.model_validate(payload, strict=True)

    payload = loaded.audit_receipt.model_dump(mode="python")
    payload["manifest_fingerprint"] = "a" * 64
    with pytest.raises(ValidationError):
        BfclV4PublicDevelopmentV2AuditReceipt.model_validate(payload, strict=True)

    manifest_payload = loaded.manifest.model_dump(mode="python")
    manifest_payload["v1_exclusion_sha256"] = "b" * 64
    forged = type(loaded.manifest).model_construct(**manifest_payload)
    with pytest.raises(ValidationError):
        BfclV4LoadedPublicDevelopmentV2.model_validate(
            loaded.model_copy(update={"manifest": forged}),
            strict=True,
        )


def test_bottom_exception_is_erased_before_typed_rejection(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    def fail(*_args, **_kwargs):
        raise SentinelError("must not escape")

    monkeypatch.setattr(question_source, "read_bfcl_v4_question_blob", fail)
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)

    assert (
        captured.value.receipt.failure_stage
        is BfclV4PublicDevelopmentV2FailureStage.QUESTION_SOURCE
    )
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None


def test_v1_exclusion_domain_formula_is_closed_and_loader_checks_it(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        bfcl_v4_public_development_v2_v1_exclusion_sha256(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS)
        == BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256
    )
    assert (
        bfcl_v4_public_development_v2_v1_exclusion_sha256(
            tuple(reversed(BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_TASK_IDS))
        )
        != BFCL_V4_PUBLIC_DEVELOPMENT_V2_V1_EXCLUSION_SHA256
    )
    monkeypatch.setattr(
        subject, "bfcl_v4_public_development_v2_v1_exclusion_sha256", lambda _: "0" * 64
    )
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)
    assert captured.value.receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.V1_LINEAGE
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None


def test_v1_exclusion_is_bound_to_question_only_v1_identity_roster(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "BFCL_V4_PILOT_ROW_IDENTITIES", {"drifted_v1": (1, "x", "y")})
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)
    assert captured.value.receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.V1_LINEAGE
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None

    monkeypatch.setattr(
        contracts_subject,
        "BFCL_V4_PILOT_ROW_IDENTITIES",
        {"drifted_v1": (1, "x", "y")},
    )
    with pytest.raises(ValidationError):
        type(BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST).model_validate(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.model_dump(mode="python"),
            strict=True,
        )


def test_question_source_commit_drift_rejects_before_checkout(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_commit = contracts_subject.BFCL_V4_UPSTREAM_COMMIT
    monkeypatch.setattr(
        contracts_subject,
        "BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT",
        "e" * 40,
    )
    with pytest.raises(ValidationError):
        type(BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST).model_validate(
            BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.model_dump(mode="python"),
            strict=True,
        )
    monkeypatch.setattr(
        contracts_subject,
        "BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT",
        expected_commit,
    )

    opened = False

    def observe_open(_checkout):
        nonlocal opened
        opened = True
        raise AssertionError("commit drift must reject before opening Git")

    monkeypatch.setattr(question_source, "BFCL_V4_QUESTION_SOURCE_UPSTREAM_COMMIT", "f" * 40)
    monkeypatch.setattr(question_source, "open_bfcl_v4_question_checkout", observe_open)
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)
    assert opened is False
    assert (
        captured.value.receipt.failure_stage
        is BfclV4PublicDevelopmentV2FailureStage.CHECKOUT_LINEAGE
    )
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None


def test_question_source_forces_no_lazy_fetch(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_environments: list[dict[str, str]] = []
    original = question_source.subprocess.run

    def recording(*args, **kwargs):
        observed_environments.append(kwargs["env"])
        return original(*args, **kwargs)

    monkeypatch.setattr(question_source.subprocess, "run", recording)
    question_source.open_bfcl_v4_question_checkout(pinned_checkout)
    assert observed_environments
    assert all(environment["GIT_NO_LAZY_FETCH"] == "1" for environment in observed_environments)


def test_question_source_is_strict_and_has_no_fixture_or_answer_symbols() -> None:
    source_path = Path(question_source.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not any("fixture_bridge" in name or "fixture_contracts" in name for name in imported)
    assert not any(name.startswith(("ANSWER", "GROUND_TRUTH")) for name in names)

    for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'"\xed\xa0\x80"', b"\xff"):
        with pytest.raises(question_source.BfclV4QuestionSourceError):
            question_source.strict_bfcl_v4_question_json(payload, "strict probe")


def test_question_source_rejects_wrong_head_and_symlink(
    tmp_path: Path,
    pinned_checkout: Path,
) -> None:
    wrong = tmp_path / "wrong-head"
    subprocess.run(("git", "init", str(wrong)), check=True, capture_output=True)
    (wrong / "probe").write_text("wrong\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(wrong), "add", "probe"), check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(wrong),
            "-c",
            "user.name=Spiral Test",
            "-c",
            "user.email=spiral@example.invalid",
            "commit",
            "-m",
            "wrong",
        ),
        check=True,
        capture_output=True,
    )
    for checkout in (wrong, tmp_path / "linked"):
        if checkout.name == "linked":
            checkout.symlink_to(pinned_checkout, target_is_directory=True)
        with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
            load_bfcl_v4_public_development_v2(checkout)
        assert (
            captured.value.receipt.failure_stage
            is BfclV4PublicDevelopmentV2FailureStage.CHECKOUT_LINEAGE
        )
        assert captured.value.__context__ is None
        assert captured.value.__cause__ is None


def test_deterministic_selection_drift_rejects_without_reselection(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = subject._select
    calls = 0

    def reversed_selection(eligible):
        nonlocal calls
        calls += 1
        return tuple(reversed(original(eligible)))

    monkeypatch.setattr(subject, "_select", reversed_selection)
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)

    receipt = captured.value.receipt
    assert receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.DETERMINISTIC_SELECTION
    assert receipt.selected_task_count == 25
    assert receipt.attempted_selected_task_ids != (BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS)
    assert receipt.manifest_fingerprint is None
    assert calls == 1


def test_near_duplicate_audit_rejects_the_fixed_roster_without_trying_replacements(
    pinned_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_calls = 0
    original_select = subject._select

    def select_once(eligible):
        nonlocal selected_calls
        selected_calls += 1
        return original_select(eligible)

    monkeypatch.setattr(subject, "_select", select_once)
    monkeypatch.setattr(subject, "_lexical_audit", lambda selected, by_id: ("f" * 64, 539, 1))
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(pinned_checkout)

    receipt = captured.value.receipt
    assert receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.NEAR_DUPLICATE_AUDIT
    assert receipt.attempted_selected_task_ids == (BFCL_V4_PUBLIC_DEVELOPMENT_V2_SELECTED_TASK_IDS)
    assert receipt.lexical_near_duplicate_count == 1
    assert receipt.manifest_fingerprint is None
    assert selected_calls == 1


def test_entry_rejects_self_consistent_selection_or_family_tampering() -> None:
    entry = BFCL_V4_PUBLIC_DEVELOPMENT_V2_MANIFEST.roster[0]

    for update in (
        {"selection_token_sha256": "a" * 64},
        {"source_family_sha256": "b" * 64},
        {"semantic_template_sha256": "c" * 64},
    ):
        with pytest.raises(ValidationError):
            BfclV4PublicDevelopmentV2RosterEntry.model_validate(
                entry.model_copy(update=update),
                strict=True,
            )


def test_wrong_checkout_produces_safe_lineage_rejection(tmp_path: Path) -> None:
    with pytest.raises(BfclV4PublicDevelopmentV2Error) as captured:
        load_bfcl_v4_public_development_v2(tmp_path / "missing")

    receipt = captured.value.receipt
    assert receipt.failure_stage is BfclV4PublicDevelopmentV2FailureStage.CHECKOUT_LINEAGE
    assert receipt.source_bindings == ()
    assert receipt.exception_text_persisted is False
    assert receipt.manifest_fingerprint is None
