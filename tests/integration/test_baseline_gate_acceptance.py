from __future__ import annotations

from pathlib import Path

import pytest

from spiral_harness.evolution.replay_study import run_non_reportable_replay_study
from spiral_harness.experiments.baseline_gate_acceptance import (
    BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE,
    BaselineGateAcceptanceError,
    BaselineGateStudyAcceptance,
    publish_baseline_gate_study_acceptance,
    verify_baseline_gate_study_acceptance,
)
from spiral_harness.experiments.baseline_gate_closure import BaselineGateStudyClosure
from tests.integration.replay_gate_fixture import foreign_gate_boundary, replay_gate_closure


def test_acceptance_binds_gate_closure_to_study_sealed_barrier(tmp_path: Path) -> None:
    execution = run_non_reportable_replay_study(tmp_path / "artifacts")
    gate_closure_ref = replay_gate_closure(execution)

    acceptance_ref = publish_baseline_gate_study_acceptance(
        execution.fixture.store,
        study_controller_manifest_ref=execution.result.study_controller_manifest_ref,
        sealed_authorized_tail_ref=execution.result.sealed_authorized_tail_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
        baseline_gate_closure_ref=gate_closure_ref,
    )

    assert acceptance_ref.media_type == BASELINE_GATE_STUDY_ACCEPTANCE_MEDIA_TYPE
    acceptance = verify_baseline_gate_study_acceptance(
        execution.fixture.store,
        acceptance_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
    )
    assert isinstance(acceptance, BaselineGateStudyAcceptance)
    assert acceptance.baseline_gate_closure_ref == gate_closure_ref
    assert acceptance.sealed_authorized_tail_ref == execution.result.sealed_authorized_tail_ref
    assert acceptance.expected_run_closure_refs == tuple(
        binding.run_closure_ref for binding in execution.study_controller.snapshot.bindings
    )
    assert acceptance.reportable_benchmark_result is False


def test_acceptance_rejects_foreign_or_tampered_gate_closure(tmp_path: Path) -> None:
    execution = run_non_reportable_replay_study(tmp_path / "artifacts")
    gate_closure_ref = replay_gate_closure(execution)
    gate_closure = execution.fixture.store.get_json(gate_closure_ref, BaselineGateStudyClosure)
    foreign_protocol_ref = execution.fixture.store.put_json({"protocol": "foreign"})
    forged_ref = execution.fixture.store.put_json(
        gate_closure.model_copy(update={"protocol_ref": foreign_protocol_ref}),
        media_type=gate_closure_ref.media_type,
    )

    with pytest.raises(BaselineGateAcceptanceError, match="gate batch differs"):
        publish_baseline_gate_study_acceptance(
            execution.fixture.store,
            study_controller_manifest_ref=execution.result.study_controller_manifest_ref,
            sealed_authorized_tail_ref=execution.result.sealed_authorized_tail_ref,
            event_verifier=execution.study_controller.event_service.verification_capability,
            baseline_gate_closure_ref=forged_ref,
        )


def test_acceptance_rejects_self_consistent_foreign_gate_boundary(tmp_path: Path) -> None:
    execution = run_non_reportable_replay_study(tmp_path / "artifacts")
    foreign_gate_closure_ref = replay_gate_closure(
        execution,
        boundary=foreign_gate_boundary(execution),
    )

    with pytest.raises(BaselineGateAcceptanceError, match="study execution boundary"):
        publish_baseline_gate_study_acceptance(
            execution.fixture.store,
            study_controller_manifest_ref=execution.result.study_controller_manifest_ref,
            sealed_authorized_tail_ref=execution.result.sealed_authorized_tail_ref,
            event_verifier=execution.study_controller.event_service.verification_capability,
            baseline_gate_closure_ref=foreign_gate_closure_ref,
        )
