from __future__ import annotations

from pathlib import Path

from spiral_harness.evolution.models import CANDIDATE_SCREEN_MEDIA_TYPE
from spiral_harness.evolution.orchestrator import (
    CANDIDATE_MATERIALIZATION_MEDIA_TYPE,
    SearchRunAdmissionReport,
)
from spiral_harness.evolution.replay_setup import SEARCH_RUN_SEEDS
from spiral_harness.evolution.replay_study import (
    ReplayStudyResult,
    run_non_reportable_replay_study,
)
from spiral_harness.experiments.baselines import LEGACY_BASELINE_KINDS, BaselineKind
from spiral_harness.experiments.study import StudyState
from spiral_harness.storage.artifact_store import ArtifactStore


def test_four_by_two_automatic_loops_cross_the_complete_study_barrier(
    tmp_path: Path,
) -> None:
    execution = run_non_reportable_replay_study(tmp_path)
    result = execution.result
    store = ArtifactStore(tmp_path)

    assert result.non_reportable_fixture is True
    assert result.exact_run_count == 8
    assert len(execution.runs) == 8
    assert store.get_json(execution.result_ref, ReplayStudyResult) == result

    coordinates = {
        (run.result.key.baseline_kind, run.result.key.search_run_seed) for run in execution.runs
    }
    assert coordinates == {
        (kind, seed) for kind in LEGACY_BASELINE_KINDS for seed in SEARCH_RUN_SEEDS
    }
    assert {run.result.key.search_run_seed for run in execution.runs} == set(SEARCH_RUN_SEEDS)

    # The study keeps eight distinct process-local verification capabilities.
    assert len({id(run.search_controller) for run in execution.runs}) == 8
    assert len({id(run.experiment_controller) for run in execution.runs}) == 8
    assert (
        len({run.search_controller.controller_manifest_ref.sha256 for run in execution.runs}) == 8
    )

    for run in execution.runs:
        loop_result = run.automatic_search.result
        assert loop_result.non_reportable_fixture is True
        admission = store.get_json(
            loop_result.admission_report_ref,
            SearchRunAdmissionReport,
        )
        assert admission.admitted is True
        assert admission.search_run_ref == run.result.search_run_ref
        assert loop_result.final_search_tail_ref == run.result.final_search_tail_ref
        assert loop_result.final_experiment_tail_ref == run.result.final_experiment_tail_ref
        assert loop_result.gate_nominations == 0

        if run.result.key.baseline_kind is BaselineKind.STATIC:
            assert loop_result.completed_rounds == 0
            assert loop_result.proposal_count == 0
            assert loop_result.screen_count == 0
            assert run.runtime.feedback_call_count == 0
            assert run.runtime.materialization_call_count == 0
            assert run.runtime.screen_call_count == 0
            continue

        # These refs are emitted only after the real loop calls both runtime stages.
        archived_media_types = {ref.media_type for ref in loop_result.archived_artifact_refs}
        assert CANDIDATE_MATERIALIZATION_MEDIA_TYPE in archived_media_types
        assert CANDIDATE_SCREEN_MEDIA_TYPE in archived_media_types
        assert loop_result.completed_rounds == 1
        assert loop_result.proposal_count == 1
        assert loop_result.screen_count == 1
        assert run.runtime.feedback_call_count == 1
        assert run.runtime.materialization_call_count == 1
        assert run.runtime.screen_call_count == 1
        assert run.runtime.gate_call_count == 0

        if run.result.key.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            assert loop_result.diagnosis_count == 1
            assert loop_result.diagnosis_invocation_count == 1
            assert loop_result.proposal_invocation_count == 1
        elif run.result.key.baseline_kind is BaselineKind.PROMPT_ONLY:
            assert loop_result.diagnosis_count == 0
            assert loop_result.proposal_invocation_count == 1
        else:
            assert run.result.key.baseline_kind is BaselineKind.RANDOM_VALID
            assert loop_result.diagnosis_count == 0
            assert loop_result.proposal_invocation_count == 0

    assert execution.study_controller.state is StudyState.SEALED_AUTHORIZED
    snapshot = execution.study_controller.snapshot
    assert snapshot is not None
    assert snapshot.missing_keys == ()
    assert len(snapshot.bindings) == 8
    assert snapshot.run_registration_open is False
    assert snapshot.proposal_api_open is False
    authorization = execution.study_controller.verify_sealed_authorization(
        result.sealed_authorized_tail_ref
    )
    assert authorization.analysis_plan_ref == result.analysis_plan_ref
    assert len(authorization.bindings) == 8
