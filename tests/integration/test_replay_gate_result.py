from __future__ import annotations

from pathlib import Path

import pytest

from spiral_harness.core.models import ArtifactRef
from spiral_harness.evolution.replay_gate_result import (
    REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE,
    ReplayStudyGateResult,
    ReplayStudyGateResultError,
    publish_replay_study_gate_result,
    verify_replay_study_gate_result,
)
from spiral_harness.evolution.replay_study import run_non_reportable_replay_study
from spiral_harness.experiments.baseline_gate_acceptance import (
    BaselineGateStudyAcceptance,
    publish_baseline_gate_study_acceptance,
)
from tests.integration.replay_gate_fixture import replay_gate_closure


def test_replay_gate_result_packages_accepted_post_barrier_evidence(tmp_path: Path) -> None:
    execution = run_non_reportable_replay_study(tmp_path / "artifacts")
    acceptance_ref = _publish_acceptance(execution)

    result_ref = publish_replay_study_gate_result(
        execution.fixture.store,
        study_result_ref=execution.result_ref,
        baseline_gate_acceptance_ref=acceptance_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
    )

    assert result_ref.media_type == REPLAY_STUDY_GATE_RESULT_MEDIA_TYPE
    result = verify_replay_study_gate_result(
        execution.fixture.store,
        result_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
    )
    assert isinstance(result, ReplayStudyGateResult)
    assert result.study_result_ref == execution.result_ref
    assert result.baseline_gate_acceptance_ref == acceptance_ref
    assert (
        result.baseline_gate_closure_ref
        == execution.fixture.store.get_json(
            acceptance_ref,
            BaselineGateStudyAcceptance,
        ).baseline_gate_closure_ref
    )
    assert result.sealed_authorized_tail_ref == execution.result.sealed_authorized_tail_ref
    assert result.reportable_benchmark_result is False
    assert result.non_reportable_fixture is True
    assert len(result.runs) == 8
    assert tuple(run.run_closure_ref for run in result.runs) == tuple(
        binding.run_closure_ref for binding in execution.study_controller.snapshot.bindings
    )


def test_replay_gate_result_rejects_tampered_published_join(tmp_path: Path) -> None:
    execution = run_non_reportable_replay_study(tmp_path / "artifacts")
    acceptance_ref = _publish_acceptance(execution)
    result_ref = publish_replay_study_gate_result(
        execution.fixture.store,
        study_result_ref=execution.result_ref,
        baseline_gate_acceptance_ref=acceptance_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
    )
    result = execution.fixture.store.get_json(result_ref, ReplayStudyGateResult)
    forged = result.model_dump(mode="json", round_trip=True, warnings="none")
    forged["runs"] = forged["runs"][:-1]
    forged_ref = execution.fixture.store.put_json(
        forged,
        media_type=result_ref.media_type,
    )

    with pytest.raises(ReplayStudyGateResultError, match="cannot be loaded"):
        verify_replay_study_gate_result(
            execution.fixture.store,
            forged_ref,
            event_verifier=execution.study_controller.event_service.verification_capability,
        )


def test_replay_gate_result_rejects_acceptance_from_another_study(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    first = run_non_reportable_replay_study(root)
    second = run_non_reportable_replay_study(root)
    foreign_acceptance_ref = _publish_acceptance(first)

    with pytest.raises(ReplayStudyGateResultError, match=r"foreign attestor|does not match"):
        publish_replay_study_gate_result(
            second.fixture.store,
            study_result_ref=second.result_ref,
            baseline_gate_acceptance_ref=foreign_acceptance_ref,
            event_verifier=second.study_controller.event_service.verification_capability,
        )


def _publish_acceptance(execution) -> ArtifactRef:
    gate_closure_ref = replay_gate_closure(execution)
    return publish_baseline_gate_study_acceptance(
        execution.fixture.store,
        study_controller_manifest_ref=execution.result.study_controller_manifest_ref,
        sealed_authorized_tail_ref=execution.result.sealed_authorized_tail_ref,
        event_verifier=execution.study_controller.event_service.verification_capability,
        baseline_gate_closure_ref=gate_closure_ref,
    )
