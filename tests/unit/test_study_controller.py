from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_terminal_decision import put_json

from spiral_harness.core.models import ArtifactRef
from spiral_harness.evolution.models import (
    SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    SearchRunManifest,
    StrategyPluginManifest,
)
from spiral_harness.evolution.orchestrator import AutomaticSearchLoop
from spiral_harness.evolution.replay_study import run_non_reportable_replay_study
from spiral_harness.experiments.baselines import BaselineKind
from spiral_harness.experiments.controller import ExperimentController
from spiral_harness.experiments.search import (
    SearchController,
    SearchEventVerificationCapability,
    SearchJournal,
    TrustedSearchEventService,
)
from spiral_harness.experiments.study import (
    STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    STUDY_JOURNAL_EVENT_MEDIA_TYPE,
    StaleStudyTailError,
    StudyBarrierError,
    StudyController,
    StudyEventAttestationError,
    StudyEventKind,
    StudyEventVerificationCapability,
    StudyJournal,
    StudyJournalEvent,
    StudyJournalEventContent,
    StudyJournalIntegrityError,
    StudyRunKey,
    StudySealedAuthorization,
    StudyState,
    TrustedStudyEventService,
)
from spiral_harness.storage.artifact_store import ArtifactStore


@dataclass(frozen=True)
class ClosedRun:
    key: StudyRunKey
    search_run: SearchRunManifest
    search_run_ref: ArtifactRef
    automatic_search_loop: AutomaticSearchLoop
    automatic_search_result_ref: ArtifactRef
    search_controller: SearchController
    search_tail: ArtifactRef
    experiment_controller: ExperimentController
    experiment_tail: ArtifactRef


def _close_search(controller: SearchController, store: ArtifactStore) -> ArtifactRef:
    tail = controller.freeze()
    tail = controller.start_search(previous_tail_ref=tail)
    if controller.manifest.baseline_kind is not BaselineKind.STATIC:
        tail = controller.open_round(previous_tail_ref=tail)
        tail = controller.record_evidence(
            previous_tail_ref=tail,
            evidence_packet_ref=put_json(store, {"kind": "exploration-evidence"}),
        )
        tail = controller.record_candidate_pool(
            previous_tail_ref=tail,
            candidate_pool_ref=put_json(store, {"kind": "empty-pool"}),
            candidate_refs=(),
            declined=True,
        )
        tail = controller.record_screen(
            previous_tail_ref=tail,
            screen_report_ref=put_json(store, {"kind": "empty-screen"}),
        )
        tail = controller.record_no_candidate(
            previous_tail_ref=tail,
            reason="no valid candidate",
        )
        tail = controller.complete_no_candidate_round(previous_tail_ref=tail)
    return controller.close_selection(previous_tail_ref=tail)


def build_study(tmp_path: Path) -> SimpleNamespace:
    replay = run_non_reportable_replay_study(tmp_path)
    store = ArtifactStore(tmp_path)
    runs: dict[tuple[str, int], ClosedRun] = {}
    for execution in replay.runs:
        key = execution.result.key
        search_run_ref = execution.result.search_run_ref
        search_run = store.get_json(search_run_ref, SearchRunManifest)
        key_tuple = (key.baseline_kind.value, key.search_run_seed)
        runs[key_tuple] = ClosedRun(
            key=key,
            search_run=search_run,
            search_run_ref=search_run_ref,
            automatic_search_loop=execution.automatic_search_loop,
            automatic_search_result_ref=execution.automatic_search.result_ref,
            search_controller=execution.search_controller,
            search_tail=execution.result.final_search_tail_ref,
            experiment_controller=execution.experiment_controller,
            experiment_tail=execution.result.final_experiment_tail_ref,
        )

    study_service = replay.study_controller.event_service
    manifest_ref = replay.result.study_controller_manifest_ref
    controller = StudyController(
        store,
        study_controller_manifest_ref=manifest_ref,
        event_service=study_service,
    )
    manifest = controller.manifest
    first_run = next(iter(runs.values())).search_run
    return SimpleNamespace(
        store=store,
        replay=replay,
        plan=controller.plan,
        plan_ref=manifest.baseline_study_plan_ref,
        analysis_plan_ref=manifest.analysis_plan_ref,
        experiment_ref=first_run.experiment_ref,
        manifest=manifest,
        manifest_ref=manifest_ref,
        service=study_service,
        controller=controller,
        runs=runs,
    )


def _start_collection(env: SimpleNamespace) -> ArtifactRef:
    tail = env.controller.freeze()
    return env.controller.start_collection(previous_tail_ref=tail)


def _register(
    env: SimpleNamespace,
    tail: ArtifactRef,
    run: ClosedRun,
) -> ArtifactRef:
    return env.controller.register_run_closure(
        previous_tail_ref=tail,
        automatic_search_loop=run.automatic_search_loop,
        automatic_search_result_ref=run.automatic_search_result_ref,
        search_controller=run.search_controller,
        search_selection_tail_ref=run.search_tail,
        experiment_controller=run.experiment_controller,
        experiment_selection_tail_ref=run.experiment_tail,
    )


def _register_all(env: SimpleNamespace, tail: ArtifactRef) -> ArtifactRef:
    for run in env.runs.values():
        tail = _register(env, tail, run)
    return tail


def test_complete_four_arm_barrier_authorizes_only_after_every_run(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    assert len(env.controller.expected_keys) == 8
    assert tuple(run.key for run in env.manifest.expected_run_manifests) == (
        env.controller.expected_keys
    )
    tail = _start_collection(env)
    with pytest.raises(StudyBarrierError, match="missing runs"):
        env.controller.close_selection_barrier(previous_tail_ref=tail)

    tail = _register_all(env, tail)
    assert env.controller.snapshot is not None
    assert env.controller.snapshot.missing_keys == ()
    barrier_tail = env.controller.close_selection_barrier(previous_tail_ref=tail)
    assert env.controller.state is StudyState.BARRIER_CLOSED
    assert env.controller.snapshot is not None
    assert env.controller.snapshot.run_registration_open is False
    assert env.controller.snapshot.proposal_api_open is False
    with pytest.raises(StudyBarrierError, match="permanently closed"):
        env.controller.assert_proposal_api_open(previous_tail_ref=barrier_tail)

    sealed_tail = env.controller.authorize_sealed(previous_tail_ref=barrier_tail)
    authorization = env.controller.verify_sealed_authorization(sealed_tail)
    assert isinstance(authorization, StudySealedAuthorization)
    assert authorization.analysis_plan_ref == env.analysis_plan_ref
    assert len(authorization.bindings) == 8
    assert env.controller.state is StudyState.SEALED_AUTHORIZED


def test_duplicate_foreign_and_stale_run_registration_fail_closed(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    tail = _start_collection(env)
    first, second = tuple(env.runs.values())[:2]
    registered = _register(env, tail, first)
    with pytest.raises(StaleStudyTailError, match="stale"):
        _register(env, tail, second)
    with pytest.raises(StudyBarrierError, match="already registered"):
        _register(env, registered, first)

    second.search_controller.invalidate(
        previous_tail_ref=second.search_tail,
        invalidation_ref=put_json(env.store, {"reason": "post-selection invalidation"}),
        reason="invalidate run",
    )
    with pytest.raises(StudyBarrierError, match="search selection verification failed"):
        _register(env, registered, second)


class DuckSelectionVerifier:
    def __init__(self, value: object) -> None:
        self.value = value

    def verify_selection_closure(self, tail_ref: ArtifactRef) -> object:
        del tail_ref
        return self.value

    def verify_experiment_selection_closure(self, tail_ref: ArtifactRef) -> object:
        del tail_ref
        return self.value


def test_self_consistent_duck_verifiers_are_not_capabilities(tmp_path: Path) -> None:
    env = build_study(tmp_path)
    tail = _start_collection(env)
    run = next(iter(env.runs.values()))
    fake_search = DuckSelectionVerifier(run.search_controller.snapshot)
    with pytest.raises(TypeError, match="exact SearchController"):
        env.controller.register_run_closure(
            previous_tail_ref=tail,
            automatic_search_loop=run.automatic_search_loop,
            automatic_search_result_ref=run.automatic_search_result_ref,
            search_controller=fake_search,  # type: ignore[arg-type]
            search_selection_tail_ref=run.search_tail,
            experiment_controller=run.experiment_controller,
            experiment_selection_tail_ref=run.experiment_tail,
        )
    fake_experiment = DuckSelectionVerifier(
        run.experiment_controller.verify_experiment_selection_closure(run.experiment_tail)
    )
    with pytest.raises(TypeError, match="exact ExperimentController"):
        env.controller.register_run_closure(
            previous_tail_ref=tail,
            automatic_search_loop=run.automatic_search_loop,
            automatic_search_result_ref=run.automatic_search_result_ref,
            search_controller=run.search_controller,
            search_selection_tail_ref=run.search_tail,
            experiment_controller=fake_experiment,  # type: ignore[arg-type]
            experiment_selection_tail_ref=run.experiment_tail,
        )


def test_manual_generic_search_closure_cannot_reuse_an_automatic_result(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    tail = _start_collection(env)
    run = next(
        value for value in env.runs.values() if value.key.baseline_kind is not BaselineKind.STATIC
    )
    manual_controller = SearchController(
        env.store,
        controller_manifest_ref=run.search_controller.controller_manifest_ref,
        event_service=run.search_controller.event_service,
        terminal_authorization_verifier=run.experiment_controller,
    )
    manual_tail = _close_search(manual_controller, env.store)

    with pytest.raises(
        StudyBarrierError,
        match="automatic search execution verification failed",
    ):
        env.controller.register_run_closure(
            previous_tail_ref=tail,
            automatic_search_loop=run.automatic_search_loop,
            automatic_search_result_ref=run.automatic_search_result_ref,
            search_controller=manual_controller,
            search_selection_tail_ref=manual_tail,
            experiment_controller=run.experiment_controller,
            experiment_selection_tail_ref=run.experiment_tail,
        )


def test_sealed_authorization_rechecks_live_tails_and_fails_after_early_sealed(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    tail = _register_all(env, _start_collection(env))
    barrier_tail = env.controller.close_selection_barrier(previous_tail_ref=tail)
    run = next(iter(env.runs.values()))
    run.experiment_controller.start_sealed(previous_tail_ref=run.experiment_tail)
    with pytest.raises(
        StudyBarrierError,
        match="live experiment selection re-verification failed",
    ):
        env.controller.authorize_sealed(previous_tail_ref=barrier_tail)
    assert env.controller.state is StudyState.BARRIER_CLOSED


def test_rehydrated_controller_without_live_capabilities_cannot_authorize_sealed(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    tail = _register_all(env, _start_collection(env))
    barrier_tail = env.controller.close_selection_barrier(previous_tail_ref=tail)
    rehydrated = StudyController(
        env.store,
        study_controller_manifest_ref=env.manifest_ref,
        event_service=env.service,
        tail_ref=barrier_tail,
    )
    with pytest.raises(StudyBarrierError, match="process-local live run capability"):
        rehydrated.authorize_sealed(previous_tail_ref=barrier_tail)


def test_manifest_rejects_missing_cells_analysis_drift_and_policy_drift(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    missing = env.manifest.model_copy(
        update={"expected_run_manifests": env.manifest.expected_run_manifests[:-1]}
    )
    missing_ref = put_json(
        env.store,
        missing,
        media_type=STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(StudyJournalIntegrityError, match="cover exactly"):
        StudyController(
            env.store,
            study_controller_manifest_ref=missing_ref,
            event_service=env.service,
        )

    target = env.manifest.expected_run_manifests[0]
    original_run = env.store.get_json(target.search_run_ref, SearchRunManifest)
    drift_analysis_ref = put_json(env.store, {"analysis": "post-hoc drift"})
    drift_run = original_run.model_copy(update={"analysis_plan_ref": drift_analysis_ref})
    drift_run_ref = put_json(
        env.store,
        drift_run,
        media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    )
    drift_expected = tuple(
        entry.model_copy(update={"search_run_ref": drift_run_ref})
        if entry.key == target.key
        else entry
        for entry in env.manifest.expected_run_manifests
    )
    drift_manifest = env.manifest.model_copy(update={"expected_run_manifests": drift_expected})
    drift_manifest_ref = put_json(
        env.store,
        drift_manifest,
        media_type=STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(StudyJournalIntegrityError, match="exact study cell"):
        StudyController(
            env.store,
            study_controller_manifest_ref=drift_manifest_ref,
            event_service=env.service,
        )

    same_kind = tuple(
        entry
        for entry in env.manifest.expected_run_manifests
        if entry.key.baseline_kind is BaselineKind.EVIDENCE_TARGETED
    )
    target = same_kind[-1]
    run = env.store.get_json(target.search_run_ref, SearchRunManifest)
    plugin = env.store.get_json(run.strategy_plugin_ref, StrategyPluginManifest)
    plugin_ref = put_json(
        env.store,
        plugin.model_copy(update={"plugin_version": "1.0.1-drift"}),
        media_type=STRATEGY_PLUGIN_MANIFEST_MEDIA_TYPE,
    )
    run = run.model_copy(
        update={
            "strategy_plugin_ref": plugin_ref,
            "strategy_plugin_fingerprint": plugin_ref.sha256,
        }
    )
    run_ref = put_json(
        env.store,
        run,
        media_type=SEARCH_RUN_MANIFEST_MEDIA_TYPE,
    )
    policy_drift_expected = tuple(
        entry.model_copy(update={"search_run_ref": run_ref}) if entry.key == target.key else entry
        for entry in env.manifest.expected_run_manifests
    )
    policy_drift_manifest = env.manifest.model_copy(
        update={"expected_run_manifests": policy_drift_expected}
    )
    policy_drift_ref = put_json(
        env.store,
        policy_drift_manifest,
        media_type=STUDY_CONTROLLER_MANIFEST_MEDIA_TYPE,
    )
    with pytest.raises(StudyJournalIntegrityError, match="share one frozen"):
        StudyController(
            env.store,
            study_controller_manifest_ref=policy_drift_ref,
            event_service=env.service,
        )


def test_hmac_tamper_gap_and_foreign_study_are_detected(tmp_path: Path) -> None:
    env = build_study(tmp_path)
    root = env.controller.freeze()
    root_event = env.store.get_json(root, StudyJournalEvent)
    tampered = root_event.model_copy(update={"reason": "tampered after signing"})
    tampered_ref = put_json(
        env.store,
        tampered,
        media_type=STUDY_JOURNAL_EVENT_MEDIA_TYPE,
    )
    with pytest.raises(StudyEventAttestationError, match="HMAC"):
        env.controller.journal.replay(tampered_ref)

    def signed_started(**updates: object) -> ArtifactRef:
        values = root_event.content.model_dump(mode="python", round_trip=True, warnings="none")
        values.update(
            sequence=updates.pop("sequence", 1),
            previous_tail_ref=root,
            kind=StudyEventKind.COLLECTION_STARTED,
            from_state=StudyState.FROZEN,
            to_state=StudyState.COLLECTING,
            reason="signed test event",
        )
        values.update(updates)
        event = env.service._issue(StudyJournalEventContent.model_validate(values))
        return put_json(
            env.store,
            event,
            media_type=STUDY_JOURNAL_EVENT_MEDIA_TYPE,
        )

    with pytest.raises(StudyJournalIntegrityError, match="expected 1, got 2"):
        env.controller.journal.replay(signed_started(sequence=2))
    with pytest.raises(StudyJournalIntegrityError, match="foreign study"):
        env.controller.journal.replay(signed_started(study_id="foreign-study"))


class SpoofedStudyVerifier(StudyEventVerificationCapability):
    @property
    def attestor_id(self) -> str:
        return "0" * 64

    def verify(self, event: object) -> StudyJournalEvent:
        return StudyJournalEvent.model_validate(event)


class SpoofedSearchVerifier(SearchEventVerificationCapability):
    @property
    def attestor_id(self) -> str:
        return "0" * 64


class SpoofedStudyService(TrustedStudyEventService):
    @property
    def verification_capability(self) -> StudyEventVerificationCapability:
        raise AssertionError("subclass override must not be reached")


class SpoofedSearchService(TrustedSearchEventService):
    @property
    def verification_capability(self) -> SearchEventVerificationCapability:
        raise AssertionError("subclass override must not be reached")


def test_event_capability_subclasses_are_rejected_before_override_use(
    tmp_path: Path,
) -> None:
    env = build_study(tmp_path)
    with pytest.raises(TypeError, match="exact StudyEventVerificationCapability"):
        StudyJournal(
            env.store,
            study_controller_manifest_ref=env.manifest_ref,
            event_verifier=SpoofedStudyVerifier(b"x" * 32),
        )
    run = next(iter(env.runs.values()))
    with pytest.raises(TypeError, match="exact SearchEventVerificationCapability"):
        SearchJournal(
            env.store,
            controller_manifest_ref=run.search_controller.controller_manifest_ref,
            event_verifier=SpoofedSearchVerifier(b"x" * 32),
            terminal_authorization_verifier=run.experiment_controller,
        )
    with pytest.raises(TypeError, match="exact TrustedStudyEventService"):
        StudyController(
            env.store,
            study_controller_manifest_ref=env.manifest_ref,
            event_service=SpoofedStudyService(b"x" * 32),
        )
    with pytest.raises(TypeError, match="exact TrustedSearchEventService"):
        SearchController(
            env.store,
            controller_manifest_ref=run.search_controller.controller_manifest_ref,
            event_service=SpoofedSearchService(b"x" * 32),
            terminal_authorization_verifier=run.experiment_controller,
        )


def test_study_controller_has_no_usage_report_admission_api(tmp_path: Path) -> None:
    env = build_study(tmp_path)
    assert not hasattr(env.controller, "register_usage_report")
    assert not hasattr(env.controller, "authorize_from_usage_reports")
